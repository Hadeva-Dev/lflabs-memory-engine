"""The lflabs-memory command line (cli.py): help, an unknown command and a missing argument are answered before the
engine loads, so none of them opens, creates or writes a store; a missing model is one line, not a traceback.

  .venv/bin/python -m unittest -q test_cli
"""
import _env  # noqa: F401  (first: the test data folder and user)
import os
import re
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from lflabs_memory import cli, config

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
            self.assertTrue(out.stdout.startswith(f"usage: lflabs-memory {cmd}"), out.stdout)
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
            self.assertIn("usage: lflabs-memory", out.stderr)
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


class CommandName(unittest.TestCase):
    def test_messages_name_the_command_this_install_has(self):
        """A plain install's messages say `lflabs-memory connect`; a host's mem.py launcher keeps `mem.py connect`."""
        cfg = config.current()
        self.assertEqual(replace(cfg, launcher=(sys.executable, "/x/bin/lflabs-memory")).command, "lflabs-memory")
        self.assertEqual(replace(cfg, launcher=("/j/python", "/j/host/mem.py")).command, "mem.py")
        self.assertEqual(replace(cfg, launcher=("", "")).command, "lflabs-memory")


if __name__ == "__main__":
    unittest.main()
