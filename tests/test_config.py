"""One data folder (config.py): where it is for each profile, that LFLABS_MEMORY_HOME wins, and that nothing the
engine writes lands beside its code.

  .venv/bin/python -m unittest -q test_config
"""
import _env  # noqa: F401  (first: the test data folder and user)
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from lflabs_memory import config, mem

PACKAGE_DIR = Path(config.__file__).resolve().parent
REPO = PACKAGE_DIR.parent.parent
ENVS = (config.HOME_ENV, config.PROFILE_ENV, config.HOST_ENV, config.MODELS_ENV)


def files_under(folder):
    return {p for p in folder.rglob("*") if p.is_file() and "__pycache__" not in p.parts}


def fake_host(returns="replace(PROFILE, name='fakehost', hooks=ProfileHooks(voice_session=Path(host) / 'voice/session.json'))"):
    """A folder holding an installed host that registers a fakehost profile (a dist-info with the entry point and its
    module), for sys.path or PYTHONPATH."""
    d = Path(tempfile.mkdtemp(prefix="fakehost")).resolve()
    (d / "fakehost-0.dist-info").mkdir()
    (d / "fakehost-0.dist-info/METADATA").write_text("Metadata-Version: 2.1\nName: fakehost\nVersion: 0\n")
    (d / "fakehost-0.dist-info/entry_points.txt").write_text("[lflabs_memory.profiles]\nfakehost = fakehost_hooks:hooks\n")
    (d / "fakehost_hooks.py").write_text("from dataclasses import replace\nfrom pathlib import Path\n"
                                         "from lflabs_memory.profiles.generic import PROFILE\n"
                                         "from lflabs_memory.profiles.hooks import ProfileHooks\n\n\n"
                                         f"def hooks(host):\n    return {returns}\n")
    return d


def on_path(test, d):
    """d on sys.path for the rest of the test."""
    sys.path.insert(0, str(d))
    test.addCleanup(lambda: str(d) in sys.path and sys.path.remove(str(d)))
    test.addCleanup(sys.modules.pop, "fakehost_hooks", None)


class Load(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp(prefix="cfghome")).resolve()
        env = {k: v for k, v in os.environ.items() if k not in ENVS}
        p = mock.patch.dict(os.environ, {**env, "HOME": str(self.home)}, clear=True)
        p.start()
        self.addCleanup(p.stop)
        self.host = fake_host()  # a host's hooks, as its own package registers them
        on_path(self, self.host)
        os.environ[config.HOST_ENV] = str(self.home / "host")

    def test_env_wins(self):
        os.environ[config.HOME_ENV] = str(self.home / "elsewhere")
        self.assertEqual(config.load().paths.data, Path(os.environ[config.HOME_ENV]))
        self.assertEqual(config.load("fakehost").paths.data, self.home / "elsewhere")  # a host's profile too

    def test_default_data_dir(self):
        cfg = config.load()
        self.assertEqual(cfg.paths.data, self.home / "Library/Application Support/LF Labs/Memory")
        self.assertEqual(cfg.paths.db, cfg.paths.data / "memory.db")
        self.assertEqual(cfg.paths.sock, cfg.paths.data / "engine.sock")
        self.assertEqual(cfg.paths.models, cfg.paths.data / "models")
        self.assertEqual(cfg.paths.logs, self.home / "Library/Logs/LF Labs")
        self.assertEqual(cfg.profile.name, "generic")

    def test_data_dir_mode(self):
        old = os.umask(0o022)
        try:
            data = config.load().paths.data
        finally:
            os.umask(old)
        self.assertEqual(stat.S_IMODE(os.stat(data).st_mode), 0o700)

    def test_a_host_profile_gets_its_folder_and_the_usual_data_folder(self):
        cfg = config.load("fakehost")
        self.assertEqual(cfg.profile.hooks.voice_session, self.home / "host/voice/session.json")  # the host's own
        self.assertEqual(cfg.paths.data, self.home / "Library/Application Support/LF Labs/Memory")
        self.assertEqual(cfg.paths.sock, cfg.paths.data / "engine.sock")
        self.assertEqual(cfg.paths.models, cfg.paths.data / "models")
        self.assertEqual(cfg.launcher[1], str(Path(sys.executable).parent / "lflabs-memory"))

    def test_a_host_profile_without_its_hooks_is_refused(self):
        """Fail closed: a host profile's trust rules need its hooks (its session files, above all)."""
        sys.path.remove(str(self.host))
        with self.assertRaisesRegex(RuntimeError, "needs its host's profile"):
            config.load("fakehost")
        sys.modules.pop("fakehost_hooks", None)
        on_path(self, fake_host(returns="None"))
        with self.assertRaisesRegex(RuntimeError, "returned NoneType"):
            config.load("fakehost")
        self.assertIsNone(config.load().profile.hooks.vouch)  # a plain install needs none

    def test_config_toml(self):
        data = self.home / "data"
        data.mkdir()
        (data / "config.toml").write_text('version = 1\nprofile = "generic"\nowner_name = "Sam"\nmodels = "~/m"\n'
                                          '[llm]\nbackend = "none"\ndaily_calls = 50\n[privacy]\nchat_index = true\n')
        os.environ[config.HOME_ENV] = str(data)
        cfg = config.load()
        self.assertEqual((cfg.owner_name, cfg.llm.backend, cfg.llm.daily_calls, cfg.llm.main_model),
                         ("Sam", "none", 50, "haiku"))
        self.assertTrue(cfg.privacy.chat_index)
        self.assertFalse(cfg.privacy.codex_sessions)
        self.assertEqual(cfg.paths.models, self.home / "m")
        os.environ[config.MODELS_ENV] = str(self.home / "shared")
        self.assertEqual(config.load().paths.models, self.home / "shared")  # the environment over the file
        (data / "config.toml").write_text('profile = "no body"\n')
        with self.assertRaises(ValueError):
            config.load()
        (data / "config.toml").write_text('profile = "nobody"\n')
        with self.assertRaisesRegex(RuntimeError, "needs its host's profile"):  # no host registered it
            config.load()

    def test_with_data_moves_every_data_file(self):
        cfg = config.load().with_data(self.home / "bench")
        for p in (cfg.paths.db, cfg.paths.sock, cfg.paths.clients, cfg.paths.outline, cfg.paths.backup,
                  cfg.paths.remote("remote-on"), cfg.paths.models):
            self.assertEqual(p.parent, self.home / "bench")


class Cli(unittest.TestCase):
    def test_the_profile_is_set_before_the_engine_loads(self):
        """cli.main(profile="fakehost"), as a host's launcher calls it: the engine then runs with the host's profile,
        and a process it starts inherits it."""
        home = Path(tempfile.mkdtemp(prefix="clihome")).resolve()
        env = {**{k: v for k, v in os.environ.items() if k not in ENVS + ("LFLABS_MEMORY_UNDER_TEST",)}, "HOME": str(home),
               "PYTHONPATH": str(fake_host()), config.HOST_ENV: str(home / "host")}
        code = ("import os; from lflabs_memory import cli, config\n"
                "try:\n    cli.main(['version'], profile='fakehost')\nexcept SystemExit:\n    pass\n"
                "print(config.CFG.profile.name, os.environ[config.PROFILE_ENV])")
        out = subprocess.run([sys.executable, "-c", code], env=env, cwd=home, capture_output=True, text=True,
                             timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.split()[-2:], ["fakehost", "fakehost"])


class NoHost(unittest.TestCase):
    def test_engine_imports_without_a_host(self):
        """A plain install: no host anywhere (HOME is empty), the engine loads, no host hook is set, and nothing
        from a voice/ folder is on the path."""
        home = Path(tempfile.mkdtemp(prefix="nohost")).resolve()
        env = {**{k: v for k, v in os.environ.items() if k not in ENVS + ("LFLABS_MEMORY_UNDER_TEST",)}, "HOME": str(home)}
        code = ("import dataclasses, json, sys\nimport lflabs_memory.mem as m\n"
                "print(json.dumps([m.CFG.profile.name, [v is None for v in dataclasses.astuple(m.HOST)],"
                " [p for p in sys.path if 'voice' in p], m.activity is None]))")
        out = subprocess.run([sys.executable, "-c", code], env=env, cwd=home, capture_output=True, text=True,
                             timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr)
        name, unset, voice, no_log = json.loads(out.stdout.splitlines()[-1])
        self.assertEqual((name, set(unset), voice, no_log), ("generic", {True}, [], True))

    def test_no_code_is_loaded_from_another_folder(self):
        found = sorted((f.name, line.strip()) for f in PACKAGE_DIR.rglob("*.py") for line in f.read_text().splitlines()
                       if re.search(r"spec_from_file_location|sys\.path\.(append|insert)|\bexec\(", line))
        # the one exception: the sense model's runtime, which comes with the model files (models/sense/), run from the
        # bytes models.read checked against models.json (never a .pyc beside it)
        self.assertEqual(found, [("mem.py", 'exec(compile(src, fs.__file__, "exec"), fs.__dict__)')])


class NothingBesideCode(unittest.TestCase):
    def test_no_root_left(self):
        for name in ("mem.py", "access.py", "connect_apps.py", "whatsup.py"):
            src = (PACKAGE_DIR / name).read_text()
            self.assertNotIn("ROOT /", src, name)
            self.assertNotIn("ROOT.parent", src, name)
        beside = sorted(m for f in PACKAGE_DIR.glob("*.py")
                        for m in re.findall(r'Path\(__file__\)\.resolve\(\)\.parent / "([^"]+)"', f.read_text()))
        # the three files shipped with the code, and CODE_DB: the guard that refuses a store beside it
        self.assertEqual(beside, ["adapt-prior.json", "gate.json", "memory.db", "models.json"])

    def test_the_process_data_folder_is_not_the_code(self):
        self.assertNotEqual(mem.CFG.paths.data.resolve(), PACKAGE_DIR)
        self.assertFalse(mem.CFG.paths.data.resolve().is_relative_to(REPO))

    def test_nothing_writes_beside_code(self):
        """add, classify, consolidate, forget and outline-file, each a fresh `lflabs-memory` process with
        LFLABS_MEMORY_HOME set: every file they write is in that folder, none beside the code. The model is a fake
        `claude` that always errors, so nothing leaves the machine."""
        models = Path(os.environ.get(config.MODELS_ENV) or mem.CFG.paths.models)
        if not (models / "embed").is_dir():
            self.skipTest(f"no embed model in {models} (set {config.MODELS_ENV})")
        tmp = Path(tempfile.mkdtemp(prefix="besidecode")).resolve()
        data, fake = tmp / "data", tmp / "bin"
        fake.mkdir()
        (fake / "claude").write_text('#!/bin/sh\ncat >/dev/null\necho \'{"is_error": true, "result": "fake"}\'\n')
        (fake / "claude").chmod(0o755)
        env = {**{k: v for k, v in os.environ.items() if k not in ENVS}, config.HOME_ENV: str(data),
               config.MODELS_ENV: str(models), "PATH": f"{fake}:{os.environ.get('PATH', '')}",
               "LFLABS_MEMORY_NOTIFY": "off"}
        before = files_under(REPO / "src") | files_under(REPO / "tests")
        for cmd in (["add", "Sam prefers tea over coffee."], ["classify"], ["consolidate"], ["forget", "1"],
                    ["outline-file"]):
            subprocess.run([sys.executable, "-m", "lflabs_memory.cli", *cmd], env=env, cwd=tmp, capture_output=True,
                           text=True, timeout=300)
        new = (files_under(REPO / "src") | files_under(REPO / "tests")) - before
        self.assertEqual(sorted(str(p) for p in new), [])
        self.assertTrue((data / "memory.db").exists())
        self.assertTrue((data / "OUTLINE.md").exists())
        self.assertEqual(stat.S_IMODE(os.stat(data).st_mode), 0o700)


if __name__ == "__main__":
    unittest.main()


class ImportSafe(unittest.TestCase):
    def test_importing_connect_apps_creates_no_data_folder(self):
        """The installer's CLI imports connect_apps (uninstall too): that must never create (or recreate) the data
        folder. Only loading the config for real (current()) does."""
        import subprocess
        import sys
        import tempfile
        home = Path(tempfile.mkdtemp(prefix="home"))
        self.addCleanup(__import__("shutil").rmtree, home, True)
        env = {k: v for k, v in os.environ.items() if k not in ("LFLABS_MEMORY_HOME", "LFLABS_MEMORY_UNDER_TEST",
                                                                "LFLABS_MEMORY_PROFILE")}
        env["HOME"] = str(home)
        code = "from lflabs_memory import connect_apps; print(connect_apps.SERVER)"
        out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(out.stdout.strip(), __import__("lflabs_memory.brand").brand.COMMAND, out.stderr)
        self.assertFalse((home / "Library").exists())
        out = subprocess.run([sys.executable, "-c", code], env={**env, "LFLABS_MEMORY_HOME": str(home / "data")},
                             capture_output=True, text=True, timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertFalse((home / "data").exists())
