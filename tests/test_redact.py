"""Retired memories keep no phone numbers, emails or long numbers, and the backup carries active memories only.

  .venv/bin/python -m unittest -q test_redact
"""
import _env  # noqa: F401  (first: the test data folder and user)
import hashlib
import json
import os
import stat
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from lflabs_memory import mem


def fake_embed(texts, query=False):
    """Deterministic unit vectors from the text: no model, and distinct texts are far apart."""
    out = []
    for t in texts:
        seed = int.from_bytes(hashlib.sha256(t.encode()).digest()[:4], "big")
        v = np.random.default_rng(seed).standard_normal(mem.DIM).astype(np.float32)
        out.append(v / np.linalg.norm(v))
    return out


class Redact(unittest.TestCase):
    def check(self, text, want, counts=None):
        got, c = mem.redact(text)
        self.assertEqual(got, want, text)
        if counts is not None:
            self.assertEqual(c, counts, text)

    def test_emails(self):
        self.check("mail jo.smith+x@mail.sub.example.com today", "mail [email] today", {"email": 1})
        self.check("a@b.example and c_d@e-f.example", "[email] and [email]", {"email": 2})

    def test_full_phone_numbers(self):
        for n in ["(415) 555-1234", "415-555-1234", "415.555.1234", "415 555 1234", "+1 415 555 1234",
                  "1-415-555-1234", "+44 20 7946 0958", "+1.415.555.0199", "+14155551234"]:
            self.check(f"call {n} now", "call [phone] now", {"phone": 1})

    def test_last_four_digits(self):
        for phrase in ["ending 6411", "ending in 6411", "ends in 6411", "ends with 6411", "last 4: 6411",
                       "last four digits 6411", "last-4 6411", "Last 4 digits are 6411", "ending #6411"]:
            got, c = mem.redact(f"phone {phrase}.")
            self.assertEqual(c, {"last4": 1}, phrase)
            self.assertNotIn("6411", got, phrase)
            self.assertIn("[last4]", got, phrase)
        self.check("Visa ••6411 and ***1234 and ...9876", "Visa ••[last4] and ***[last4] and ...[last4]", {"last4": 3})

    def test_long_numbers(self):
        self.check("ticket 48213377, acct 000123456789", "ticket [number], acct [number]", {"number": 2})
        self.check("card 4111 1111 1111 1111 ok", "card [number] ok", {"number": 1})
        self.check("a bare 4155551234 run", "a bare [number] run", {"number": 1})

    def test_ordinary_text_is_left_alone(self):
        for t in ["On 2026-10-08 at 14:30 we shipped v1.2.3", "port 8080, 12345 rows, 5-digit 90210",
                  "commit cf41c94 and PR #412", "+1 vote", "costs $1,234.56", "Python 3.12 and room 101",
                  "an @mention and user@localhost", "the 1990s, 2025-2026 season", "3-2-1 launch"]:
            self.check(t, t, {})

    def test_deterministic_and_idempotent(self):
        t = "Robin (415) 555-1234, robin@example.com, card ending 6411, ticket 99887766"
        once = mem.redact(t)[0]
        self.assertEqual(once, mem.redact(t)[0])
        self.assertEqual(mem.redact(once), (once, {}))


class DBCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="memredact"))
        self.addCleanup(shutil.rmtree, self.tmp, True)  # a 5 MB store each: never left behind
        for p in (mock.patch.object(mem, "DB_PATH", self.tmp / "memory.db"), mock.patch.object(mem, "ROOT", self.tmp),
                  mock.patch.object(mem, "OUTLINE_FILE", self.tmp / "OUTLINE.md"),  # forget_source rewrites it
                  mock.patch.object(mem, "embed", fake_embed),
                  mock.patch.object(mem, "HOLDOUT_MODE", "off"),  # Robin's live switches must not decide test outcomes
                  mock.patch.object(mem, "CLIENT_SCOPES_MODE", "off"), mock.patch.object(mem, "HANDOFF_MODE", "off"),
                  mock.patch.object(mem, "WHATS_UP_MODE", "off"),
                  mock.patch.object(mem, "TALK_PERMS_MODE", "off"), mock.patch.object(mem, "KEYLESS_MODE", "off"),
                  *(mock.patch.object(mem, k, "off") for k in ("NEVER_EMPTY_MODE", "CARDS_MODE", "TRANSLATE_MODE")),
                  *(mock.patch.object(mem, k, "off") for k in mem.LEAN_SWITCHES.values())):
            p.start()
            self.addCleanup(p.stop)
        self.db = mem.connect()
        self.addCleanup(self.db.close)

    def row(self, mid):
        return self.db.execute("select * from memories where id = ?", (mid,)).fetchone()

    def chunks(self, mid):
        return self.db.execute("select count(*) from chunks where memory_id = ?", (mid,)).fetchone()[0]


class Forget(DBCase):
    PRIVATE = "Dentist Dr. Lee: (415) 555-1234, lee@example.com, card ending 6411, booking 77665544."

    def test_forget_deletes_the_text_and_unindexes(self):
        mid, _ = mem.remember(self.db, self.PRIVATE)
        self.assertGreater(self.chunks(mid), 0)
        self.assertTrue(mem.forget(self.db, mid))
        r = self.row(mid)
        self.assertEqual(r["status"], "retired")
        self.assertEqual(r["content"], "")
        self.assertEqual(r["hash"], mem._hash(""))
        self.assertEqual(self.chunks(mid), 0)
        hits = self.db.execute("select count(*) from chunk_fts where chunk_fts match '\"lee\"'").fetchone()[0]
        self.assertEqual(hits, 0)
        self.assertFalse(mem.forget(self.db, mid))  # already retired

    def test_the_forget_tool_says_where_a_merged_memory_went(self):
        a, _ = mem.remember(self.db, "Packing list: two tablets for the long flight.")
        b, _ = mem.remember(self.db, "Packing list for Brazil, merged: tablets, adapters and sunscreen.")
        self.db.execute("update memories set status = 'merged', merged_into = ? where id = ?", (b, a))
        self.assertEqual(mem.forget_reply(self.db, a), f"memory #{a} was merged into #{b}: recall_full {b}, then "
                                                       f"forget #{b} or remember it rewritten without what should go")
        self.assertTrue(mem.forget_reply(self.db, b).startswith(f"Deleted #{b} for good: its text, its search entries and 1 older version merged into it"))
        self.assertEqual(self.row(a)["content"], "")  # the older version merged into it goes too
        self.assertEqual(mem.forget_reply(self.db, b), f"no active memory #{b}")
        self.assertEqual(mem.forget_reply(self.db, 999), "no active memory #999")

    def test_forgetting_is_explained_to_the_brain_and_the_extractor(self):
        # memory-bench diagnosis 2026-10-14: whole memories retired for one detail, mentions left behind, and a
        # "the idea was dropped" note saved right after a forget
        doc = " ".join(mem.FORGET_DOC.split())
        for words in ("recall it first", "every memory that mentions it", "remember a corrected version without it",
                      "Never save a note that something was forgotten"):
            self.assertIn(words, doc)
        self.assertIn("asked Claude to forget", " ".join(mem.EXTRACT_SYSTEM.split()))

    def test_consolidate_retire_redacts_too(self):
        mid, _ = mem.remember(self.db, "No longer true: call 415-555-1234 for the gym. " + "x " * 5)
        with self.db:
            mem._retire(self.db, mid)
        self.assertNotIn("555", self.row(mid)["content"])

    def test_redact_retired_dry_run_then_apply(self):
        mid, _ = mem.remember(self.db, self.PRIVATE)
        keep, _ = mem.remember(self.db, "Active: Robin's number is 415-555-9999")
        with self.db:  # retired the old way, before forget redacted
            self.db.execute("update memories set status='retired' where id=?", (mid,))
            mem._unindex(self.db, mid)
        self.assertEqual(mem.redact_retired(self.db), (1, 1, {"phone": 1, "email": 1, "last4": 1, "number": 1}))
        self.assertEqual(self.row(mid)["content"], self.PRIVATE)  # dry run wrote nothing
        mem.redact_retired(self.db, apply=True)
        self.assertNotIn("555-1234", self.row(mid)["content"])
        self.assertIn("415-555-9999", self.row(keep)["content"])  # active memories are never touched
        self.assertEqual(mem.redact_retired(self.db), (1, 0, {}))

    def test_snapshot_is_private(self):
        mem.remember(self.db, "something")
        p = mem.snapshot(self.db, "pre-redact")
        self.assertTrue(p.name.endswith("-pre-redact.db") and p.parent == self.tmp / "backup")
        self.assertEqual(stat.S_IMODE(os.stat(p).st_mode), 0o600)
        import sqlite3
        self.assertEqual(sqlite3.connect(p).execute("select count(*) from memories").fetchone()[0], 1)


class BackupRestore(DBCase):
    SRC = "/Users/x/.claude/projects/-Users-x/memory/{}.md"

    def test_a_memory_naming_a_token_prefix_restores(self):
        # A memory about the security sweep names the prefix "github_pat_" with no token: not a secret.
        p = self.tmp / "m.json"
        rows = [{"id": 1, "status": "active", "content": "the filter catches github_pat_ tokens", "kind": "project"},
                {"id": 2, "status": "active", "content": "leaked github_pat_" + "11ABCDEFG" + "x" * 70}]
        p.write_text(json.dumps(rows))
        self.assertEqual(mem.import_backup(self.db, p), {"active": 1, "tombstone": 0})
        self.assertIsNone(self.row(2))

    def test_crafted_rows_cannot_stand_in_for_a_file(self):
        rows = [
            # active, but its source_hash isn't its own text's: the source is kept, the hash dropped
            {"id": 1, "status": "active", "content": "evil text", "source": self.SRC.format("a"),
             "source_hash": mem._hash("real file text"), "kind": "user", "project": None, "created": 1, "updated": 1,
             "merged_into": None},
            # a source that isn't a memory file
            {"id": 2, "status": "active", "content": "ok text", "source": "/etc/passwd", "source_hash": mem._hash("ok text"),
             "kind": "<b>", "project": "p", "created": 1, "updated": 1, "merged_into": None},
            # an old backup's retired row with text, not from a file: not restored
            {"id": 3, "status": "retired", "content": "old 415-555-1234", "source": None, "source_hash": None,
             "kind": "note", "project": None, "created": 1, "updated": 1, "merged_into": None},
            # a tombstone without a valid hash
            {"id": 4, "status": "merged", "content": None, "source": self.SRC.format("b"), "source_hash": "zz",
             "created": 1, "updated": 1, "merged_into": 1},
            {"id": 5, "status": "weird", "content": "x"},
            "not a row",
        ]
        p = self.tmp / "crafted.json"
        p.write_text(json.dumps(rows))
        self.assertEqual(mem.import_backup(self.db, p), {"active": 2, "tombstone": 0})
        r1, r2 = self.row(1), self.row(2)
        self.assertEqual((r1["source"], r1["source_hash"]), (self.SRC.format("a"), None))
        self.assertEqual((r2["source"], r2["kind"]), (None, "note"))
        self.assertIsNone(self.row(3))
        self.assertEqual(mem.remember(self.db, "real file text", source=self.SRC.format("a")), (1, "updated"))
        self.assertEqual(self.row(1)["content"], "real file text")


if __name__ == "__main__":
    unittest.main()
