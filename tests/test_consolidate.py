"""Weekly consolidation: memories it merges or retires leave the search index, like `forget` (d0d881e).

  .venv/bin/python -m unittest -q test_consolidate
"""
import _env  # noqa: F401  (first: the test data folder and user)
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from lflabs_memory import mem


def vec(i):
    v = np.zeros(mem.DIM, dtype=np.float32)
    v[i] = 1
    return v


def fake_embed(texts, query=False):
    # The two "lunch" notes are 0.9 alike: past MERGE_SIM (merged weekly), short of DUP_SIM (no update on save).
    # Everything else gets its own direction.
    def one(t):
        if "weekday lunch" in t.lower():
            return 0.9 * vec(0) + np.sqrt(0.19).astype(np.float32) * vec(1)
        return vec(0) if "lunch" in t.lower() else vec(2 + sum(map(ord, t)) % (mem.DIM - 2))
    return np.stack([one(t) for t in texts]).astype(np.float32)


class Consolidate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.patches = [mock.patch.object(mem, "DB_PATH", Path(self.tmp.name) / "t.db"),
                        mock.patch.object(mem, "embed", fake_embed),
                        mock.patch.object(mem, "rerank", lambda q, docs: [0.0] * len(docs)),
                        mock.patch.object(mem, "activity", None)]
        for p in self.patches:
            p.start()
        self.db = mem.connect()

    def tearDown(self):
        self.db.close()
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def indexed(self, mid):
        chunks = self.db.execute("select id from chunks where memory_id = ?", (mid,)).fetchall()
        fts = self.db.execute("select count(*) from chunk_fts where rowid in (select id from chunks where memory_id = ?)",
                              (mid,)).fetchone()[0]
        vecs = self.db.execute("select count(*) from mem_vec where rowid = ?", (mid,)).fetchone()[0]
        return len(chunks) + fts + vecs

    def llm(self, prompt, schema, system, **k):
        if system == mem.MERGE_SYSTEM:
            return {"content": "Lunch is at noon on weekdays.", "kind": "user", "project": None}
        return {"action": "retire", "content": ""}

    def test_merged_and_retired_memories_leave_the_index(self):
        a, _ = mem.remember(self.db, "Lunch is at noon.")
        b, _ = mem.remember(self.db, "Weekday lunch: noon.")
        stale, _ = mem.remember(self.db, "The old build script is obsolete.")
        keep, _ = mem.remember(self.db, "The deploy target is the staging box.")
        self.assertTrue(all(self.indexed(m) for m in (a, b, stale, keep)))
        with mock.patch.object(mem, "llm_json", self.llm):
            mem.consolidate(self.db, log=lambda m: None)
        status = dict(self.db.execute("select id, status from memories").fetchall())
        self.assertEqual((status[a], status[b], status[stale], status[keep]), ("merged", "merged", "retired", "active"))
        for m in (a, b, stale):
            self.assertEqual(self.indexed(m), 0, f"#{m} is still in the search index")
        merged = self.db.execute("select merged_into from memories where id = ?", (a,)).fetchone()[0]
        self.assertTrue(self.indexed(merged) and self.indexed(keep))
        hits = {h["id"] for h in mem.search(self.db, "lunch at noon")}
        self.assertIn(merged, hits)
        self.assertFalse(hits & {a, b, stale})
        self.assertTrue(mem.restore(self.db, a))  # restore still brings a merged memory back, re-indexed
        self.assertTrue(self.indexed(a))


if __name__ == "__main__":
    unittest.main()
