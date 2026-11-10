"""Forget deletes for good (0.1.4): after a forget, the words are in no file the engine keeps: not memory.db, not its
write-ahead log, not the keyword indexes, not the dated snapshots in backup/, not backup/memories.json. What stays is
a tombstone row with no text and a fingerprint (mem.forget_sketch) that is not the embedding.

  .venv/bin/python -m unittest -q test_forget_erase
"""
import _env  # noqa: F401  (first: the test data folder and user)
import json
import sqlite3
import time
import unittest
from unittest import mock

import numpy as np

from lflabs_memory import claimstore as cs
from lflabs_memory import mem
from test_redact import DBCase, fake_embed

SECRET = "Quillfeather"  # one word that appears nowhere else
TEXT = f"Robin's therapist is Dr. {SECRET}, Wednesdays at 5, phone 415 555 0199."


class Erase(DBCase):
    def files(self):
        """Every file of the data folder the engine writes that could hold memory text."""
        return [p for p in [mem.DB_PATH, mem.DB_PATH.with_name("memory.db-wal"), *mem.paths().backup.glob("*")]
                if p.exists()]

    def holders(self):
        return [p.name for p in self.files() if SECRET.encode() in p.read_bytes()]

    def fts_hits(self):
        n = self.db.execute("select count(*) from chunk_fts where chunk_fts match ?", (SECRET.lower(),)).fetchone()[0]
        if cs.ready(self.db):
            n += self.db.execute("select count(*) from claim_fts where claim_fts match ?",
                                 (SECRET.lower(),)).fetchone()[0]
        return n

    def save_everywhere(self):
        """The memory, an older version merged into it, a claim drawn from it, a snapshot and a JSON backup."""
        old, _ = mem.remember(self.db, f"Therapist: Dr. {SECRET} (older note)")
        mid, _ = mem.remember(self.db, TEXT)
        with self.db:
            self.db.execute("update memories set status = 'merged', merged_into = ? where id = ?", (mid, old))
        cid, _ = cs.add(mem, self.db, TEXT, "owner_typed", vec=fake_embed([TEXT])[0])
        self.assertIsNotNone(cid)
        snap = mem.snapshot(self.db, "before-forget")
        (mem.paths().backup / "memories.json").write_text(json.dumps([{"id": mid, "content": TEXT, "status": "active"}]))
        self.db.execute("pragma wal_checkpoint(TRUNCATE)")
        self.assertIn("memory.db", self.holders())
        self.assertIn(snap.name, self.holders())
        self.assertIn("memories.json", self.holders())
        self.assertGreater(self.fts_hits(), 0)
        return mid, old, cid

    def test_no_model_nothing_is_kept_anywhere(self):
        mid, old, cid = self.save_everywhere()
        with mock.patch.object(mem, "CFG", _with_backend(mem.CFG, "none")):
            reply = mem.forget_reply(self.db, mid)
        self.assertTrue(reply.startswith(f"Deleted #{mid} for good: its text, its search entries, 1 older version"
                                         f" merged into it and 1 fact drawn from it, and the same from 2 backup copies"),
                        reply)
        self.assertIn("This cannot be undone.", reply)
        self.assertNotIn("24 hours", reply)  # no model to sweep with: nothing waits
        self.assertEqual(self.holders(), [])
        self.assertEqual(self.fts_hits(), 0)
        self.assertEqual((self.row(mid)["content"], self.row(old)["content"]), ("", ""))
        self.assertEqual(self.db.execute("select content from claims where id = ?", (cid,)).fetchone()[0], "")
        self.assertFalse(mem.restore(self.db, mid))

    def test_with_a_model_the_words_wait_only_for_the_sweep(self):
        mid, _, _ = self.save_everywhere()
        reply = mem.forget_reply(self.db, mid)
        self.assertIn("held for at most 24 hours", reply)
        self.assertEqual(self.fts_hits(), 0)
        self.assertEqual([r[0] for r in self.db.execute("select memory_id from claim_forgets where words is not null")],
                         [mid])  # the one place they wait
        self.assertEqual([h for h in self.holders() if h.startswith("memory-") or h == "memories.json"], [])
        with mock.patch.object(cs, "sweep", lambda *a, **k: 0):  # the model found nothing more to forget
            cs.run_forgets(mem, self.db, log=lambda *_: None)
        self.assertEqual(self.holders(), [])

    def test_unswept_words_expire_after_a_day(self):
        mid, _, _ = self.save_everywhere()
        mem.forget_reply(self.db, mid)
        later = time.time() + mem.FORGET_WORDS_KEEP + 60
        with mock.patch.object(mem.time, "time", lambda: later):
            mem._expire_forget_words(self.db)
        self.assertEqual(self.holders(), [])
        outcome = self.db.execute("select outcome from claim_forgets where memory_id = ?", (mid,)).fetchone()[0]
        self.assertEqual(json.loads(outcome), "expired")

    def test_the_fingerprint_is_not_the_embedding(self):
        mid, _ = mem.remember(self.db, TEXT)
        vec = mem._vectors(self.db, [mid])[mid]
        mem.forget(self.db, mid)
        kept = np.frombuffer(self.db.execute("select embedding from forgot_vec where rowid = ?", (mid,)).fetchone()[0],
                             dtype=np.float32)
        self.assertEqual(sorted(set(np.round(np.abs(kept) * np.sqrt(mem.DIM), 4))), [1.0])  # signs only
        self.assertLess(float(kept @ vec), 0.2)  # a random rotation's signs: nothing like the embedding itself
        self.assertEqual(mem.forgotten_match(self.db, vec), mid)  # still recognised when said again

    def test_secure_delete_is_on_from_the_start(self):
        self.assertEqual(self.db.execute("pragma secure_delete").fetchone()[0], 1)
        for t in mem._fts_tables(self.db):
            got = self.db.execute(f"select v from {t}_config where k = 'secure-delete'").fetchone()
            self.assertEqual(got[0], 1, t)


class Upgrade(DBCase):
    def test_what_0_1_3_kept_of_a_forget_is_erased_once(self):
        """A store from 0.1.3: a forgotten memory's text kept (redacted), its full embedding in forgot_vec."""
        mid, _ = mem.remember(self.db, TEXT)
        vec = mem._vectors(self.db, [mid])[mid]
        with self.db:
            mem._retire(self.db, mid)  # 0.1.3's forget: text kept
            self.db.execute("insert into forgot_vec(rowid, embedding) values (?, ?)", (mid, vec.tobytes()))
            tidy, _ = mem.remember(self.db, "An old status note the weekly tidy retired")
            mem._retire(self.db, tidy)
            self.db.execute("delete from meta where key = 'forget_erased_v1'")
        self.db.close()
        self.db = mem.connect()
        self.addCleanup(self.db.close)
        self.assertEqual(self.row(mid)["content"], "")
        self.assertNotEqual(self.row(tidy)["content"], "")  # not forgotten: kept, as before
        self.assertEqual(mem.forgotten_match(self.db, vec), mid)
        self.assertNotIn(SECRET.encode(), mem.DB_PATH.read_bytes())


def _with_backend(cfg, backend):
    from dataclasses import replace
    return replace(cfg, llm=replace(cfg.llm, backend=backend))


if __name__ == "__main__":
    unittest.main()
