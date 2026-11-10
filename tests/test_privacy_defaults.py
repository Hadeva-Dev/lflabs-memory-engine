"""A plain install's defaults (spec 5.6): no experiments (holdout, shadows), no reading of Codex's session files or
indexing of past chats unless config.toml's [privacy] says so, new apps at work only, banners on, and no network from
the engine's own work.

  .venv/bin/python -m unittest -q test_privacy_defaults
"""
import _env  # noqa: F401  (first: the test data folder and user)
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from lflabs_memory import access, mem
from test_redact import DBCase

_OPENED, _CONNECTED = [], []  # what the audit hook saw while a test watched (an audit hook can't be removed)
_WATCH = [False]


def _audit(event, args):
    if not _WATCH[0]:
        return
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        _OPENED.append(os.fsdecode(args[0]))
    elif event == "socket.connect" and len(args) > 1 and isinstance(args[1], tuple):
        _CONNECTED.append(args[1])  # an internet address; a unix socket's is a path


sys.addaudithook(_audit)


class Watch:
    def __enter__(self):
        _OPENED.clear(), _CONNECTED.clear()
        _WATCH[0] = True
        return self

    def __exit__(self, *a):
        _WATCH[0] = False


class Switches(unittest.TestCase):
    def test_no_experiments(self):
        self.assertEqual((mem.HOLDOUT_MODE, mem.ADAPT_MODE, mem.WIDE_SHADOW, mem.PRECISE_RECALL_MODE),
                         ("off", "off", "off", "off"))
        self.assertEqual(mem.SENSE_MODE, "on")  # the word list decides until the sensitivity model is installed
        self.assertEqual((mem.CHAT_INDEX_MODE, mem.CODEX_SESSIONS_MODE), ("off", "off"))
        self.assertEqual(mem.HANDOFF_MODE, "on")
        self.assertEqual(access.BANNERS, "on")

    def test_generic_holdout_off(self):
        self.assertEqual(mem.HOLDOUT_MODE, "off")
        self.assertFalse(mem.CFG.privacy.holdout)

    def test_new_client_work_only(self):
        self.assertEqual(mem.KEYLESS_MODE, "on")
        self.assertEqual(mem.new_client_entry("cursor"), {"allow": list(mem.WORK_ONLY), "deny": []})
        self.assertEqual(mem.CFG.privacy.new_clients, "work")

    def test_privacy_choices_turn_them_on(self):
        home = Path(tempfile.mkdtemp(prefix="privacy"))
        (home / "config.toml").write_text('version = 1\n[privacy]\nchat_index = true\ncodex_sessions = true\n'
                                          'holdout = true\n')
        code = "import json\nfrom lflabs_memory import mem\nprint(json.dumps([mem.CHAT_INDEX_MODE, " \
               "mem.CODEX_SESSIONS_MODE, mem.HOLDOUT_MODE, mem.ADAPT_MODE]))"
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
                           env={**os.environ, "LFLABS_MEMORY_HOME": str(home)}, cwd=home)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(json.loads(r.stdout), ["on", "on", "on", "off"])  # an experiment stays off whatever


class Reading(DBCase):
    def setUp(self):
        super().setUp()
        self.codex = self.tmp / "home/.codex/sessions"
        day = self.codex / "2026/10/26"
        day.mkdir(parents=True)
        self.rollout = day / "rollout-2026-10-26T10-00-00-abc.jsonl"
        self.rollout.write_text(json.dumps({"type": "session_meta", "payload": {"id": "abc", "source": "cli",
                                                                                "cwd": "/tmp/x"}}) + "\n")
        os.utime(self.rollout, (time.time(), time.time()))
        p = mock.patch.object(mem, "CODEX_SESSIONS", self.codex)
        p.start()
        self.addCleanup(p.stop)

    def test_codex_sessions_off_by_default(self):
        with mock.patch.object(mem, "HANDOFF_MODE", "on"), Watch():
            mem._codex_tick(self.db)
        self.assertEqual([f for f in _OPENED if str(self.codex) in f], [])
        with mock.patch.object(mem, "HANDOFF_MODE", "on"), mock.patch.object(mem, "CODEX_SESSIONS_MODE", "on"), Watch():
            mem._codex_tick(self.db)  # chosen in setup: read
        self.assertIn(str(self.rollout), _OPENED)

    def test_chat_index_off(self):
        def refuse(*a, **k):
            raise AssertionError("the chat index ran")
        with mock.patch.object(mem, "chat_index_allowed", lambda now=None: True), \
                mock.patch.object(mem, "chats_due", lambda db: True), mock.patch.object(mem.subprocess, "run", refuse), \
                mock.patch.object(mem, "_chat_wait", [0.0]):
            mem._chat_tick(self.db, lambda *a: None)  # everything else says go: only the switch stops it


class NoNetwork(DBCase):
    def test_the_engines_own_work_opens_no_connection(self):
        """Saving, searching, recall, the prompt hook and filing: no internet connection. The model backend (here a
        fake) is the only thing that may reach out."""
        calls = []

        def llm(prompt, schema, system, model=None, **k):
            calls.append(model)
            return {"items": []}
        with mock.patch.object(mem, "llm_json", llm), Watch():
            a, _ = mem.remember(self.db, "Robin deploys Orchard from main on Fridays.", "project", "orchard")
            mem.remember(self.db, "Robin's sister Bea moved to Lisbon.", "note", None)
            mem.search(self.db, "orchard deploys", k=4)
            mem.recall_text(self.db, "orchard deploys")
            mem.prompt_context(self.db, "when does orchard deploy", "s1", str(mem.HOME / "orchard"))
            mem.classify(self.db, log=lambda *x: None)
            mem.write_outline(self.db, self.tmp / "OUTLINE.md")
        self.assertEqual(_CONNECTED, [])
        self.assertTrue(calls)  # the classifier asked the (fake) backend: default filings are filed again
        self.assertTrue(a)


if __name__ == "__main__":
    unittest.main()
