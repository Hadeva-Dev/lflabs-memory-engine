"""A typed command waits in typed_pending (its whole prompt) until the transcript shows it. Its words must not outlive
that wait, nor a forget of the memory they made (review 2026-11-10: a Codex session's last "remember that X" left X in
the file for good, and forgetting X erased the memory but not the pending row, nor its copy in a snapshot).

  .venv/bin/python -m unittest -q test_typed_pending
"""
import _env  # noqa: F401  (first: the test data folder and user)
import json
import sqlite3
import time
from unittest import mock

from lflabs_memory import mem
from test_forget_erase import SECRET, _with_backend
from test_redact import DBCase

WORDS = f"the orchard vendor contact is Dr. {SECRET}"


class Pending(DBCase):
    def holders(self):
        """The files of the data folder that hold the secret word: memory.db, its log, every backup copy."""
        files = [mem.DB_PATH, mem.DB_PATH.with_name("memory.db-wal"), *mem.paths().backup.glob("*")]
        return [f.name for f in files if f.exists() and SECRET.encode() in f.read_bytes()]

    def typed(self, mid=None, kind="remember", at=None, fact=WORDS, sid="codex-1", arg=None):
        data = {"prompt": f"remember that {fact}", "session_id": sid, "transcript_path": "/tmp/x.jsonl"}
        a = arg if arg is not None else json.dumps({"id": mid, "hash": mem._hash(mem.clean_text(fact)), "fact": fact})
        mem.defer_typed(self.db, data, "codex", kind, a)
        if at is not None:
            with self.db:
                self.db.execute("update typed_pending set at = ? where session_id = ?", (at, sid))

    def pending(self):
        return self.db.execute("select count(*) from typed_pending").fetchone()[0]

    def test_an_expired_wait_is_gone_when_the_store_opens(self):
        mid, _ = mem.remember(self.db, WORDS, origin="unknown")
        self.typed(mid, at=time.time() - mem.TYPED_WAIT - 60)
        self.typed(mid, sid="codex-2")  # fresh: stays
        self.db.close()
        self.db = mem.connect()
        self.assertEqual([r[0] for r in self.db.execute("select session_id from typed_pending")], ["codex-2"])

    def test_and_on_the_workers_pass(self):
        self.typed(1, at=time.time() - mem.TYPED_WAIT - 60)
        self.assertEqual(mem.prune_typed(self.db), 1)
        self.assertEqual(self.pending(), 0)
        import inspect
        self.assertIn("prune_typed(db)", inspect.getsource(mem._worker))

    def test_a_fresh_wait_still_promotes(self):
        mid, _ = mem.remember(self.db, WORDS, origin="unknown")
        self.typed(mid)
        self.assertEqual(mem.prune_typed(self.db), 0)
        with mock.patch.object(mem, "typed_in_transcript", lambda *a, **k: True):
            mem.run_typed(self.db, "codex-1")
        self.assertEqual(self.row(mid)["origin"], "owner_typed")
        self.assertEqual(self.pending(), 0)

    def test_forgetting_the_memory_takes_its_typed_words_everywhere(self):
        mid, _ = mem.remember(self.db, WORDS, origin="unknown")
        self.typed(mid)  # the Codex session's last prompt: no later hook will make it
        old = mem.paths().backup / "memory-2026-01-01-000000-older.db"  # a snapshot made before snapshot() left them out
        old.parent.mkdir(parents=True, exist_ok=True)
        dst = sqlite3.connect(old)
        self.db.backup(dst)
        dst.close()
        self.db.execute("pragma wal_checkpoint(TRUNCATE)")
        self.assertIn("memory.db", self.holders())
        self.assertIn(old.name, self.holders())
        with mock.patch.object(mem, "CFG", _with_backend(mem.CFG, "none")):
            mem.forget_reply(self.db, mid)
        self.assertEqual(self.pending(), 0)
        self.assertEqual(self.holders(), [])  # memory.db, its log and every backup copy

    def test_a_pending_correction_of_it_goes_too(self):
        mid, _ = mem.remember(self.db, WORDS, origin="owner_typed")
        newer = WORDS + ", from Monday"
        self.typed(kind="remember-text", fact=newer,
                   arg=json.dumps({"fact": newer, "kind": "note", "project": None, "replace": mid}))
        with mock.patch.object(mem, "CFG", _with_backend(mem.CFG, "none")):
            mem.forget_reply(self.db, mid)
        self.assertEqual(self.pending(), 0)

    def test_a_snapshot_keeps_no_waiting_commands(self):
        mid, _ = mem.remember(self.db, WORDS, origin="unknown")
        self.typed(mid)
        snap = mem.snapshot(self.db, "test")
        dst = sqlite3.connect(snap)
        try:
            self.assertEqual(dst.execute("select count(*) from typed_pending").fetchone()[0], 0)
        finally:
            dst.close()
        self.assertEqual(self.pending(), 1)  # the live one waits on
