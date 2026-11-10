"""Meeting facts: tagged meeting:<id>#<n>, never merged with other memories, hard-deleted by forget-source, left out
of the weekly tidy and of the GitHub backup (docs/meeting-notes.md 3.6).

  .venv/bin/python -m unittest -q test_meeting_source
"""
import _env  # noqa: F401  (first: the test data folder and user)
import sys
import unittest
from pathlib import Path
from unittest import mock

from lflabs_memory import mem
from test_redact import DBCase


class MeetingSource(DBCase):
    def test_tagged_and_hard_deleted(self):
        a, how = mem.remember(self.db, "In the Orchard sync on 2026-10-09, Anders said the demo moves to Friday.",
                              "reference", "meetings", "meeting:20261009-1500-ab12#1")
        self.assertEqual(how, "added")
        b, _ = mem.remember(self.db, "In the Orchard sync on 2026-10-09, Robin agreed to send the deck by Thursday.",
                            "reference", "meetings", "meeting:20261009-1500-ab12#2")
        other, _ = mem.remember(self.db, "In another sync, Tomas wants the API beta.", "reference", "meetings",
                                "meeting:20261010-0900-cd34#1")
        keep, _ = mem.remember(self.db, "Robin likes oat milk.")
        self.db.execute("insert into injections values ('s1', ?, 1)", (a,))
        self.assertGreater(self.chunks(a), 0)
        self.assertEqual(mem.forget_source(self.db, "meeting:20261009-1500-ab12#"), 2)
        for mid in (a, b):
            self.assertIsNone(self.row(mid))  # gone, not retired
            self.assertEqual(self.chunks(mid), 0)
            self.assertEqual(self.db.execute("select count(*) from mem_vec where rowid = ?", (mid,)).fetchone()[0], 0)
        self.assertEqual(self.db.execute("select count(*) from injections where memory_id = ?", (a,)).fetchone()[0], 0)
        self.assertEqual(self.db.execute("select count(*) from chunk_fts where chunk_fts match '\"orchard\"'")
                         .fetchone()[0], 0)
        self.assertIsNotNone(self.row(other))
        self.assertIsNotNone(self.row(keep))

    def test_prefix_must_be_a_meeting(self):
        for bad in ("", "meeting:", "meeting:x", "/Users/x/.claude/projects/p/memory/a.md", "meeting:%#", "meeting:_#x"):
            with self.assertRaises(ValueError, msg=bad):
                mem.forget_source(self.db, bad)

    def test_no_merging_across_the_tag(self):
        text = "Anders said the investor demo moves to Friday Oct 16."
        plain, _ = mem.remember(self.db, text)
        with mock.patch.object(mem, "DUP_SIM", -1.0):  # anything counts as a near-duplicate
            tagged, how = mem.remember(self.db, text + " ", "reference", "meetings", "meeting:m1#1")
            self.assertNotEqual(tagged, plain)  # a meeting fact never overwrites another memory
            self.assertEqual(how, "added")
            later, how = mem.remember(self.db, "Something else Robin said later.")
            self.assertNotEqual(later, tagged)  # nor is a meeting fact overwritten by one
            self.assertEqual(self.row(tagged)["content"], text)

    def test_weekly_tidy_skips_meeting_facts(self):
        mem.remember(self.db, "Anders said the demo is Friday.", "reference", "meetings", "meeting:m1#1")
        mem.remember(self.db, "Anders said the demo is on Friday.", "reference", "meetings", "meeting:m2#1")
        with mock.patch.object(mem, "MERGE_SIM", -1.0), mock.patch.object(mem, "llm_json") as llm, \
                mock.patch.object(mem, "STALE_RE", mem.re.compile(".")):
            mem.consolidate(self.db, log=lambda m: None)
        llm.assert_not_called()

    def test_cli_source_is_meeting_only(self):
        import io
        from contextlib import redirect_stdout
        out = io.StringIO()
        with mock.patch.object(sys, "argv", ["mem.py", "add", "x y z", "--source", "/etc/passwd"]), \
                mock.patch.dict(mem.os.environ, {"LFLABS_MEMORY_NOSAVE": "", "LFLABS_MEMORY_CHILD": ""}), \
                redirect_stdout(out), self.assertRaises(SystemExit) as e:
            mem.main()
        self.assertEqual(e.exception.code, 2)
        self.assertIn("meeting:<id>#<n>", out.getvalue())
        self.assertEqual(self.db.execute("select count(*) from memories").fetchone()[0], 0)


class Snapshots(DBCase):
    def test_snapshots_never_hold_meeting_facts(self):
        keep, _ = mem.remember(self.db, "Robin likes oat milk.")
        mem.remember(self.db, "Anders said ZebraQuokka is the code name.", "reference", "meetings", "meeting:m1#1")
        path = mem.snapshot(self.db, "test")
        import sqlite3
        snap = sqlite3.connect(path)
        self.assertEqual(snap.execute("select count(*) from memories where source like 'meeting:%'").fetchone()[0], 0)
        self.assertEqual(snap.execute("select count(*) from memories where id = ?", (keep,)).fetchone()[0], 1)
        snap.close()
        self.assertNotIn(b"zebraquokka", path.read_bytes().lower())  # not even in free pages or the keyword index
        self.assertIsNotNone(self.row(keep))  # the live DB is untouched
        self.assertEqual(self.db.execute("select count(*) from memories where source like 'meeting:%'").fetchone()[0], 1)

    def test_scrub_existing_snapshots(self):
        mem.remember(self.db, "Anders said ZebraQuokka is the code name.", "reference", "meetings", "meeting:m1#1")
        self.db.commit()
        old = self.tmp / "backup" / "memory-2026-10-01-000000-old.db"
        old.parent.mkdir(exist_ok=True)
        import sqlite3
        dst = sqlite3.connect(old)
        self.db.backup(dst)
        dst.close()
        self.assertEqual(mem.scrub_snapshots(), {old.name: 1})
        self.assertNotIn(b"zebraquokka", old.read_bytes().lower())
        self.assertEqual(mem.scrub_snapshots(), {old.name: 0})



if __name__ == "__main__":
    unittest.main()
