"""The lflabs-memory command line (cli.py): help, an unknown command and a missing argument are answered before the
engine loads, so none of them opens, creates or writes a store; a missing model is one line, not a traceback.

  .venv/bin/python -m unittest -q test_cli
"""
import _env  # noqa: F401  (first: the test data folder and user)
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from lflabs_memory import brand, cli, config

MEM = Path(config.__file__).resolve().parent / "mem.py"


def run(*args, models=None):
    """`lflabs-memory ARGS` in a fresh process whose data folder does not exist yet: (result, data folder)."""
    data = Path(tempfile.mkdtemp(prefix="cli")).resolve() / "data"
    env = {**os.environ, config.HOME_ENV: str(data)}
    if models is not None:
        env[config.MODELS_ENV] = str(models)
    out = subprocess.run([sys.executable, "-m", "lflabs_memory.cli", *args], env=env, capture_output=True, text=True,
                         timeout=120)
    return out, data


class Commands(unittest.TestCase):
    def test_every_command_of_mem_main_is_known(self):
        src = MEM.read_text()
        body = src[src.index("\ndef main():"):]
        found = set(re.findall(r'cmd (?:==|in) \(?((?:"[a-z-]+",? ?)+)', body))
        names = {n for group in found for n in re.findall(r'"([a-z-]+)"', group)}
        names |= set(re.findall(r'"(hook-[a-z-]+)":', src))
        self.assertEqual(sorted(names - set(cli.COMMANDS)), [])


class Help(unittest.TestCase):
    def test_help_is_for_the_user_and_touches_nothing(self):
        for args in (["--help"], ["-h"], ["help"]):
            out, data = run(*args)
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertIn("usage: lflabs-memory COMMAND", out.stdout)
            self.assertNotRegex(out.stdout, r"mem\.py")
            self.assertFalse(data.exists(), args)

    def test_a_command_help_is_never_its_argument(self):
        """`add --help` once saved a memory reading "--help"; `forget --help` was a traceback."""
        for cmd in ("add", "search", "forget", "restore", "outline", "clients", "connect", "serve", "paths"):
            out, data = run(cmd, "--help")
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertTrue(out.stdout.startswith(f"usage: {cli._prog(cmd)} {cmd}"), out.stdout)
            self.assertFalse(data.exists(), cmd)

    def test_unknown_command(self):
        out, data = run("remmber", "x")
        self.assertEqual(out.returncode, 2)
        self.assertIn("unknown command 'remmber'", out.stderr)
        self.assertFalse(data.exists())


class Arguments(unittest.TestCase):
    def test_a_missing_argument_is_a_usage_line(self):
        for args in (["add"], ["search"], ["forget"], ["restore"], ["extract"], ["search-chats"], ["forget", "#3"],
                     ["add", "--kind", "feedback", "text after the option"], ["search", "-k", "3", "query"]):
            out, data = run(*args)
            self.assertEqual(out.returncode, 2, args)
            self.assertRegex(out.stderr, r"usage: (lflabs-memory|" + brand.COMMAND + ") ")
            self.assertNotIn("Traceback", out.stderr)
            self.assertFalse(data.exists(), args)

    def test_a_missing_model_is_one_line(self):
        out, _ = run("search", "anything", models=tempfile.mkdtemp(prefix="nomodels"))
        self.assertEqual(out.returncode, 1)
        self.assertIn("embed model refused", out.stderr)
        self.assertNotIn("Traceback", out.stderr)


    def test_a_bad_config_is_one_line(self):
        for text, said in (("owner_name = Sam\n", "config.toml: Invalid value"), ("llm = 3\n", "[llm] must be a table")):
            data = Path(tempfile.mkdtemp(prefix="cli")).resolve()
            (data / "config.toml").write_text(text)
            out = subprocess.run([sys.executable, "-m", "lflabs_memory.cli", "stats"], capture_output=True, text=True,
                                 env={**os.environ, config.HOME_ENV: str(data)}, timeout=120)
            self.assertEqual(out.returncode, 1, out.stderr)
            self.assertIn(said, out.stderr)
            self.assertNotIn("Traceback", out.stderr)


class AddSays(unittest.TestCase):
    def test_add_prints_a_plain_line(self):
        """Not a Python tuple, "(1, 'added')" (VM gate 0.1.10 rc1)."""
        from lflabs_memory import mem
        self.assertEqual(mem.add_said((1, "added")), "Saved as #1.")
        self.assertEqual(mem.add_said((4, "unchanged")), "Already in memory as #4.")
        self.assertEqual(mem.add_said((4, "updated")), "Updated #4.")
        self.assertEqual(mem.add_said((None, "refused: it looks like a key, token or password")),
                         "Not saved: it looks like a key, token or password.")

    def test_add_says_when_the_note_is_held(self):
        """A note added from a Terminal without Allow in the dialog (or from a script) is held like an app's save:
        "Saved as #1." alone read as done, and the note then never reached a prompt (public 0.1.10 install check)."""
        data = Path(tempfile.mkdtemp(prefix="cli")).resolve() / "data"
        data.mkdir()
        (data / "config.toml").write_text('version = 1\nprofile = "generic"\nowner_name = "Robin"\n')
        out = subprocess.run([sys.executable, "-m", "lflabs_memory.cli", "add", "the heron build ships on Thursdays"],
                             capture_output=True, text=True, env={**os.environ, config.HOME_ENV: str(data)}, timeout=180)
        self.assertEqual(out.returncode, 0, out.stderr)
        said = out.stdout.strip()  # one line: the id first
        self.assertTrue(said.startswith("Saved as #1. "), said)
        self.assertIn("kept out of prompts until you confirm it", said)
        self.assertIn(f"`{brand.COMMAND} held`", said)


class CommandName(unittest.TestCase):
    def test_messages_name_the_command_this_install_has(self):
        """A plain install's messages say `lflabs-memory connect`; a host's mem.py launcher keeps `mem.py connect`."""
        cfg = config.current()
        self.assertEqual(replace(cfg, launcher=(sys.executable, "/x/bin/lflabs-memory")).command, "lflabs-memory")
        self.assertEqual(replace(cfg, launcher=("/j/python", "/j/host/mem.py")).command, "mem.py")
        self.assertEqual(replace(cfg, launcher=("", "")).command, "lflabs-memory")

    def test_the_cli_front_doors_are_named_as_the_user_types_them(self):
        """held, trust, forget, restore, search and add also run as the product CLI's own commands: messages and
        usage lines about them name that command on a plain install; a host's launcher keeps its own name."""
        cfg = config.current()
        self.assertEqual(replace(cfg, launcher=(sys.executable, "/x/bin/lflabs-memory")).front, brand.COMMAND)
        self.assertEqual(replace(cfg, launcher=("/j/python", "/j/host/mem.py")).front, "mem.py")
        for cmd in cli.FRONT:
            self.assertIn(cmd, cli.COMMANDS)
            self.assertEqual(cli._prog(cmd), brand.COMMAND)
        self.assertEqual(cli._prog("connect"), "lflabs-memory")
        out, _ = run("held", "--help")
        self.assertTrue(out.stdout.startswith(f"usage: {brand.COMMAND} held"), out.stdout)


LEGACY = "zh_CN.GB2312"  # a locale that is not UTF-8, as some Terminals in China, Japan and Europe are still set


def _has_locale(name):
    try:
        return name in subprocess.run(["locale", "-a"], capture_output=True, text=True, timeout=10).stdout.split()
    except OSError:
        return False


@unittest.skipUnless(_has_locale(LEGACY), f"this machine has no {LEGACY} locale")
class LegacyLocale(unittest.TestCase):
    """Under a locale that is not UTF-8, Python reads every file and stdin in that locale: a prompt in Japanese was
    a decode error in the hook (no recall, and a typed "remember that" never saved), and `held` or `search` showing a
    memory the locale cannot write was a traceback. The files and the apps' JSON are UTF-8 whatever the Terminal is."""

    def run_in(self, data, *args, stdin=None):
        env = {**os.environ, config.HOME_ENV: str(data), "LC_ALL": LEGACY, "LANG": LEGACY}
        env.pop("PYTHONUTF8", None)
        env.pop("PYTHONIOENCODING", None)
        return subprocess.run([sys.executable, "-m", "lflabs_memory.cli", *args], env=env, capture_output=True,
                              input=stdin, timeout=180)

    def setUp(self):
        self.data = Path(tempfile.mkdtemp(prefix="cli")).resolve() / "data"
        self.data.mkdir()
        (self.data / "config.toml").write_text('version = 1\nprofile = "generic"\nowner_name = "Robin"\n')

    def test_the_prompt_hook_reads_utf8(self):
        payload = json.dumps({"session_id": "s1", "cwd": str(self.data), "prompt": "ヘロンのビルドはいつ出荷しますか"},
                             ensure_ascii=False).encode("utf-8")
        out = self.run_in(self.data, "hook-prompt", stdin=payload)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertNotIn(b"codec", out.stderr)
        self.assertNotIn(b"error", out.stderr)

    def test_a_memory_the_locale_cannot_write_is_still_shown(self):
        text = "ヘロンのビルドは毎週木曜日に出荷する"
        out = self.run_in(self.data, "add", text)
        self.assertEqual(out.returncode, 0, out.stderr)
        for cmd in (["held"], ["search", "ヘロン"]):
            out = self.run_in(self.data, *cmd)
            self.assertEqual(out.returncode, 0, out.stderr.decode(errors="replace"))
            self.assertNotIn(b"Traceback", out.stderr)
            self.assertIn(text, out.stdout.decode("utf-8"))  # a pipe, not a Terminal: UTF-8, as the files are


class Words(unittest.TestCase):
    """`add buy milk tomorrow` saved "buy", and `search heron build` searched "heron": only the
    first word was read, without a word said. Every word before the first option is the text."""

    def run_in(self, data, *args):
        return subprocess.run([sys.executable, "-m", "lflabs_memory.cli", *args], capture_output=True, text=True,
                              env={**os.environ, config.HOME_ENV: str(data)}, timeout=180)

    def test_add_and_search_take_every_word(self):
        data = Path(tempfile.mkdtemp(prefix="cli")).resolve() / "data"
        data.mkdir()
        (data / "config.toml").write_text('version = 1\nprofile = "generic"\nowner_name = "Robin"\n')
        out = self.run_in(data, "add", "the", "heron", "build", "ships", "every", "Thursday", "--kind", "project")
        self.assertEqual(out.returncode, 0, out.stderr)
        out = self.run_in(data, "search", "heron", "build", "-k", "3")
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("[project/", out.stdout)
        self.assertIn("the heron build ships every Thursday", out.stdout)

    def test_a_bad_option_is_refused_before_anything_is_saved(self):
        for args, said in ((["search", "heron", "-k"], "-k needs a number"),
                           (["search", "heron", "-k", "many"], "-k needs a number"),
                           (["add", "a note", "--kind"], "--kind needs a value"),
                           (["add", "never", "use", "--force", "here"], "unknown option --force"),
                           (["search", "heron", "--sectoin", "work"], "unknown option --sectoin")):
            out, data = run(*args)
            self.assertEqual(out.returncode, 2, (args, out.stderr))
            self.assertIn(said, out.stderr)
            self.assertIn(f"usage: {brand.COMMAND} {args[0]}", out.stderr)
            self.assertNotIn("Traceback", out.stderr)
            self.assertFalse(data.exists(), args)


class ExitCodes(unittest.TestCase):
    """A forget or restore that did nothing exits 1 (a script, or `&&`, can tell), and restoring a memory that was
    never forgotten no longer answers "restored"."""

    def run_in(self, data, *args):
        return subprocess.run([sys.executable, "-m", "lflabs_memory.cli", *args], capture_output=True, text=True,
                              env={**os.environ, config.HOME_ENV: str(data)}, timeout=180)

    def test_nothing_done_is_exit_1(self):
        data = Path(tempfile.mkdtemp(prefix="cli")).resolve() / "data"
        data.mkdir()
        (data / "config.toml").write_text('version = 1\nprofile = "generic"\nowner_name = "Robin"\n')
        out = self.run_in(data, "add", "the heron build ships every Thursday")
        self.assertEqual((out.returncode, out.stdout.split(". ")[0]), (0, "Saved as #1"), out.stderr)  # and that it is held
        for args, said in ((["forget", "999"], "no active memory #999"),
                           (["forget", "c999"], "no active claim c999"),
                           (["restore", "999"], "#999 cannot be restored"),
                           (["restore", "c999"], "c999 is not waiting to be deleted"),
                           (["restore", "1"], "#1 was not forgotten: there is nothing to restore")):
            out = self.run_in(data, *args)
            self.assertIn(said, out.stdout, args)
            self.assertEqual(out.returncode, 1, (args, out.stdout, out.stderr))
        out = self.run_in(data, "search", "heron build")  # and the memory is as it was
        self.assertIn("the heron build ships every Thursday", out.stdout)
        out = self.run_in(data, "forget", "1")  # the user's forget, or an app's (when the suite runs in an app's shell)
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)


if __name__ == "__main__":
    unittest.main()
