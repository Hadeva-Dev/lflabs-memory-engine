"""lflabs-memory/1 on the connector socket (connector.py): one JSON line in, one out; who asks decides what they see
(a declared name without its key gets work only, Claude Code by its reported name sees everything); limits on
size, time, results, saves an hour and which ids a client may forget; a second engine is refused; a slow connector
call never holds up the prompt hook. Synthetic only: temp stores, a fake embedder.

  .venv/bin/python -m unittest -q test_connector
"""
import _env  # noqa: F401  (first: the test data folder and user)
import io
import json
import os
import shutil
import socket
import socketserver
import stat
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from lflabs_memory import brand, connector, mem
from _fakes import fake_rerank
from test_redact import DBCase


def ask(path, req, raw=None, timeout=20):
    with socket.socket(socket.AF_UNIX) as s:
        s.settimeout(timeout)
        s.connect(str(path))
        s.sendall(raw if raw is not None else (json.dumps(req) + "\n").encode())
        return json.loads(s.makefile().readline())


def call(path, tool, args=None, name=None, key=None, hint=None):
    return ask(path, {"protocol": "lflabs-memory/1", "tool": tool, "args": args or {},
                      "client": {"name": name, "key": key, "hint": hint, "connector": "0.1.0"}})


class Case(DBCase):
    def setUp(self):
        super().setUp()
        self.sockdir = Path(tempfile.mkdtemp(prefix="lfc", dir="/tmp"))  # a short path: sun_path is 104 bytes
        self.addCleanup(shutil.rmtree, self.sockdir, True)
        self.path = self.sockdir / "memory-connector.sock"
        for p in (mock.patch.object(mem, "CLIENT_SCOPES_MODE", "on"), mock.patch.object(mem, "KEYLESS_MODE", "on"),
                  mock.patch.object(mem, "TALK_PERMS_MODE", "on"),
                  mock.patch.object(mem, "rerank", fake_rerank({"staging": 5.0, "sister": 5.0, "tabs": 5.0})),
                  mock.patch.dict(mem._clients, {"key": None, "data": {}})):
            p.start()
            self.addCleanup(p.stop)
        self.work, _ = mem.remember(self.db, "The staging database for orchard listens on port 6543.", "project",
                                    "orchard", origin="owner_typed")
        self.private, _ = mem.remember(self.db, "Robin's sister Bea moved to Lisbon in May.", "user", None, origin="owner_typed")
        self.assertEqual(self.row(self.private)["section"], "personal")

    def start(self):
        srv = connector.serve(mem, self.path)
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        return srv


class Protocol(Case):
    def test_ping(self):
        self.start()
        got = call(self.path, "ping")
        self.assertTrue(got["ok"])
        self.assertTrue(got["text"].startswith(brand.PRODUCT + " "))

    def test_unknown_protocol_refused(self):
        self.start()
        got = ask(self.path, {"protocol": "lflabs-memory/9", "tool": "ping", "args": {}})
        self.assertEqual(got, {"ok": False, "error": "this engine speaks lflabs-memory/1"})
        self.assertFalse(ask(self.path, None, raw=b"not json\n")["ok"])
        self.assertFalse(call(self.path, "drop_tables")["ok"])

    def test_request_over_1_mib_refused(self):
        self.start()
        got = call(self.path, "recall", {"query": "x" * (1 << 20)})
        self.assertFalse(got["ok"])
        self.assertIn("too large", got["error"])

    def test_socket_mode_0600(self):
        self.start()
        self.assertTrue(stat.S_ISSOCK(os.stat(self.path).st_mode))
        self.assertEqual(stat.S_IMODE(os.stat(self.path).st_mode), 0o600)

    def test_stale_socket_replaced(self):
        s = socket.socket(socket.AF_UNIX)
        s.bind(str(self.path))
        s.close()  # a socket file nobody answers on: a crashed engine's
        self.start()
        self.assertTrue(call(self.path, "ping")["ok"])

    def test_second_engine_refused(self):
        self.start()
        with self.assertRaises(connector.AnotherEngine) as e:
            connector.claim(self.path)
        self.assertIn("another memory engine is running", str(e.exception))

    def test_daemon_exits_when_another_engine_answers(self):
        self.start()
        err = io.StringIO()
        with mock.patch.object(mem, "CONNECTOR_MODE", "on"), mock.patch.object(mem, "CONNECTOR_SOCK", self.path), \
                mock.patch.object(mem, "embed", side_effect=AssertionError("models loaded")), \
                mock.patch("sys.stderr", err), self.assertRaises(SystemExit) as e:
            mem.daemon()
        self.assertEqual(e.exception.code, 1)
        self.assertIn("another memory engine is running", err.getvalue())

    def test_slow_call_times_out(self):
        self.start()
        with mock.patch.object(connector, "CALL_TIMEOUT", 0.3), \
                mock.patch.object(mem, "recall_text", lambda *a, **k: time.sleep(2) or "late"):
            got = call(self.path, "recall", {"query": "staging"}, hint="claude-code")
        self.assertFalse(got["ok"])
        self.assertIn("too long", got["error"])


class Clients(Case):
    def test_recall_work_only_for_declared_cursor(self):
        self.start()
        got = call(self.path, "recall", {"query": "staging database port and Robin's sister"}, name="cursor")
        self.assertTrue(got["ok"])
        self.assertIn(f"#{self.work}", got["text"].replace('"id": ', "#"))
        self.assertNotIn("Lisbon", got["text"])

    def test_keyed_cursor_given_more_sees_it(self):
        """Its own key proves the name: what the user gave Cursor (everything, here) applies."""
        mem.clients_add("cursor", with_key=True, key="k-cursor-123")
        mem.clients_put("cursor", lambda e: {**{k: v for k, v in e.items() if k not in ("allow", "level")},
                                             "deny": []})
        self.start()
        self.assertIn("Lisbon", call(self.path, "recall", {"query": "Robin's sister"}, name="cursor",
                                     key="k-cursor-123")["text"])
        got = call(self.path, "recall", {"query": "Robin's sister"}, name="cursor", key="wrong")
        self.assertNotIn("Lisbon", got["text"])

    def test_claude_code_by_hint_gets_work_only(self):
        """0.1.4: any process can report "claude-code"; Claude Code's own prompt hook still sees everything."""
        self.start()
        got = call(self.path, "recall", {"query": "Robin's sister"}, hint="claude-code")
        self.assertNotIn("Lisbon", got["text"])

    def test_desk_never_by_declared_name(self):
        self.start()
        got = call(self.path, "recall", {"query": "Robin's sister"}, name="desk")
        self.assertNotIn("Lisbon", got["text"])

    def test_recall_limits(self):
        self.start()
        seen = {}
        real = mem.recall_text

        def spy(db, query, k=8, section=None, scope=None, max_chars=mem.RECALL_CHARS, client=None, shown=None):
            seen.update(k=k, max_chars=max_chars)
            return real(db, query, k, section, scope, max_chars, client, shown)
        with mock.patch.object(mem, "recall_text", spy):
            call(self.path, "recall", {"query": "staging", "k": 50}, hint="claude-code")
        self.assertEqual(seen, {"k": 20, "max_chars": 12000})

    def test_remember_files_by_default_and_cursor_sees_its_project_save(self):
        self.start()
        got = call(self.path, "remember", {"content": "Orchard deploys from main on Fridays.", "kind": "project",
                                           "project": "orchard"}, name="cursor")
        self.assertTrue(got["ok"], got)
        mid = int(got["text"].split("#")[-1].split()[0])
        row = self.row(mid)
        self.assertEqual((row["section"], row["origin"]), ("projects", "unknown"))
        self.assertIn("Fridays", call(self.path, "recall", {"query": "orchard deploys"}, name="cursor")["text"])

    def test_cursor_recalls_its_project_save_with_no_project(self):
        self.start()
        got = call(self.path, "remember", {"content": "Our CI runs on GitHub Actions with uv", "kind": "project"},
                   name="cursor")
        self.assertTrue(got["ok"], got)
        self.assertEqual(self.row(int(got["text"].split("#")[-1].split()[0]))["section"], "projects")
        self.assertIn("GitHub Actions", call(self.path, "recall", {"query": "what CI do we use"}, name="cursor")["text"])

    def test_remember_of_a_forgotten_memory_is_an_error(self):
        self.start()
        mem.forget(self.db, self.work)
        got = call(self.path, "remember", {"content": "The staging database for orchard listens on port 6543.",
                                           "kind": "project", "project": "orchard"}, hint="claude-code")
        self.assertFalse(got["ok"], got)
        self.assertIn(f"restates memory #{self.work}", got["error"])
        self.assertNotIn("restore", got["error"])

    def test_remember_too_long_refused(self):
        self.start()
        got = call(self.path, "remember", {"content": "x" * 4001}, hint="claude-code")
        self.assertFalse(got["ok"])

    def test_remember_rate_limit(self):
        self.start()
        for i in range(30):
            got = call(self.path, "remember", {"content": f"Robin's note number {i} about the tide tables {i * 7919}"},
                       name="cursor")
            self.assertTrue(got["ok"], (i, got))
        got = call(self.path, "remember", {"content": "one more about the tide tables"}, name="cursor")
        self.assertFalse(got["ok"])
        self.assertIn("30", got["error"])
        self.assertTrue(call(self.path, "remember", {"content": "Codex keeps its own count of the tide"},
                             name="codex")["ok"])  # per client

    def test_forget_only_recalled_ids(self):
        self.start()
        got = call(self.path, "forget", {"id": str(self.work)}, hint="claude-code")
        self.assertFalse(got["ok"])
        self.assertIn("recall", got["error"])
        self.assertEqual(self.row(self.work)["status"], "active")
        call(self.path, "recall", {"query": "staging database"}, hint="claude-code")
        got = call(self.path, "forget", {"id": str(self.work)}, hint="claude-code")
        self.assertTrue(got["ok"], got)
        self.assertEqual(self.row(self.work)["status"], "held")  # an app's forget: held a day (test_app_trust)

    def test_forget_never_by_ids_the_caller_wrote(self):
        """A section name (or a query) that echoes back as '"id": N' never makes N forgettable."""
        self.start()
        got = call(self.path, "recall", {"query": "x", "section": f'"id": {self.work}'}, hint="claude-code")
        self.assertIn(f'"id": {self.work}', got["text"])  # echoed in the error text
        self.assertFalse(call(self.path, "forget", {"id": str(self.work)}, hint="claude-code")["ok"])
        self.assertEqual(self.row(self.work)["status"], "active")

    def test_forget_needs_the_same_clients_recall(self):
        self.start()
        call(self.path, "recall", {"query": "staging database"}, hint="claude-code")
        self.assertFalse(call(self.path, "forget", {"id": str(self.work)}, name="cursor")["ok"])

    def test_access_status(self):
        self.start()
        got = call(self.path, "access_status", name="cursor")
        self.assertTrue(got["ok"])
        self.assertIn("cursor", got["text"].lower())


class NotBlocked(Case):
    def test_hook_not_blocked(self):
        """A slow recall on the connector while the prompt hook asks engine.sock: the hook answers under 1.5 s."""
        self.start()
        hook_sock = self.sockdir / "engine.sock"
        hook = socketserver.UnixStreamServer(str(hook_sock), mem._handler(self.db))
        threading.Thread(target=hook.serve_forever, daemon=True).start()
        self.addCleanup(hook.server_close)
        self.addCleanup(hook.shutdown)
        started = threading.Event()

        def slow(*a, **k):
            started.set()
            time.sleep(3)
            return "slow"
        with mock.patch.object(mem, "recall_text", slow):
            t = threading.Thread(target=lambda: call(self.path, "recall", {"query": "staging"}, hint="claude-code"))
            t.start()
            started.wait(5)
            out = io.StringIO()
            t0 = time.monotonic()
            with mock.patch.object(mem, "SOCK", hook_sock), mock.patch.object(mem, "gate", lambda db, p, hits, scope=None, **k: hits), \
                    redirect_stdout(out):
                mem.hook_prompt({"prompt": "which port does the staging database use", "session_id": "s1",
                                 "cwd": str(self.tmp)})
            took = time.monotonic() - t0
            t.join()
        self.assertLess(took, 1.5)
        self.assertIn("6543", out.getvalue())


if __name__ == "__main__":
    unittest.main()
