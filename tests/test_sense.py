"""Sensitivity labels (2026-10-18): the daemon labels memories with the local classifier; in "shadow" nothing changes,
in "on" a labelled memory's labels decide what Telegram withholds and the word list covers the rest.

  .venv/bin/python -m unittest -q test_sense
"""
import _env  # noqa: F401  (first: the test data folder and user)
import json
import os
import unittest
from unittest import mock

from lflabs_memory import mem
from test_redact import DBCase
from _fakes import limited_scope  # noqa: E402

LIMITED = limited_scope()  # a restricted reader, as a host's channel

LABELS = ["health", "mental_health", "money", "secret_credential", "government_id", "location_home"]


class FakeModel:
    """Health for "sugar levels" or "clinic", money for "owe": the rest is clean."""
    thresholds = {k: {"threshold": 0.5, "tier": "sensitive"} for k in LABELS}

    def __init__(self):
        self.seen = []

    def classify(self, texts):
        self.seen += texts
        out = []
        for t in texts:
            p = {k: 0.05 for k in LABELS}
            if "sugar levels" in t or "clinic" in t:
                p["health"] = 0.9
            if "owe" in t:
                p["money"] = 0.8
            out.append(p)
        return out


class Sense(DBCase):
    def setUp(self):
        super().setUp()
        self.model = FakeModel()
        for p in (mock.patch.dict(mem._sense, {"tried": True, "model": self.model, "id": "fake1", "error": None,
                                               "flags": None, "mtime": None}),
                  mock.patch.object(mem, "SENSE_MODE", "shadow")):
            p.start()
            self.addCleanup(p.stop)
        self.bank, _ = mem.remember(self.db, "We go to the bank on Friday to open Lucas's school account")
        self.sugar, _ = mem.remember(self.db, "Check my sugar levels before lunch every day")
        self.plain, _ = mem.remember(self.db, "Dinner with Tomas at Casa Ipe on Saturday")

    def row(self, mid):
        r = self.db.execute("select id, content, section, section2 from memories where id = ?", (mid,)).fetchone()
        return {**dict(r), "section": "personal"}

    def withheld_on_telegram(self, mid):
        return not mem._allowed(self.row(mid), (), mem.RESTRICTED[LIMITED])

    def test_off_labels_nothing(self):
        with mock.patch.object(mem, "SENSE_MODE", "off"):
            self.assertEqual(mem.sense_tick(self.db), 0)
        self.assertEqual(self.db.execute("select count(*) from sense").fetchone()[0], 0)
        self.assertFalse(mem._sense_flags_file().exists())

    def test_shadow_labels_but_changes_nothing(self):
        self.assertEqual(mem.sense_tick(self.db), 3)
        self.assertEqual(mem.sense_tick(self.db), 0)  # each memory once
        self.assertEqual(oct(os.stat(mem._sense_flags_file()).st_mode & 0o777), "0o600")
        flags = json.loads(mem._sense_flags_file().read_text())
        self.assertEqual((sorted(flags["labelled"]), flags["withheld"]), (sorted([self.bank, self.sugar, self.plain]),
                                                                          [self.sugar]))
        self.assertNotIn("sugar", mem._sense_flags_file().read_text())  # ids only, never text
        self.assertTrue(self.withheld_on_telegram(self.bank))  # the word list still decides: "bank"
        self.assertFalse(self.withheld_on_telegram(self.sugar))
        self.assertFalse(self.withheld_on_telegram(self.plain))

    def test_on_lets_the_labels_decide(self):
        mem.sense_tick(self.db)
        with mock.patch.object(mem, "SENSE_MODE", "on"):
            self.assertFalse(self.withheld_on_telegram(self.bank))  # the word list's false alarm is gone
            self.assertTrue(self.withheld_on_telegram(self.sugar))  # health with no word on the list
            self.assertFalse(self.withheld_on_telegram(self.plain))
            late, _ = mem.remember(self.db, "Paid the bank fee for the gym")
            self.assertTrue(self.withheld_on_telegram(late))  # not labelled yet: the word list
            self.assertTrue(mem._allowed(self.row(self.sugar), (), ()))  # the desk sees everything

    def test_a_changed_memory_is_labelled_again(self):
        mem.sense_tick(self.db)
        with self.db:
            self.db.execute("update memories set content = ? where id = ?", ("I owe Tomas 40 for dinner", self.plain))
        self.assertIsNone(self.db.execute("select 1 from sense where memory_id = ?", (self.plain,)).fetchone())
        self.assertEqual(mem.sense_tick(self.db), 1)
        self.assertIn(self.plain, json.loads(mem._sense_flags_file().read_text())["withheld"])

    def test_a_deleted_memory_leaves_no_labels(self):
        mem.sense_tick(self.db)
        with self.db:
            self.db.execute("delete from memories where id = ?", (self.sugar,))
        self.assertIsNone(self.db.execute("select 1 from sense where memory_id = ?", (self.sugar,)).fetchone())

    def test_a_new_model_labels_everything_again(self):
        mem.sense_tick(self.db)
        with mock.patch.dict(mem._sense, {"id": "fake2"}):
            self.assertEqual(mem.sense_tick(self.db), 3)
            self.assertEqual(json.loads(mem._sense_flags_file().read_text())["model"], "fake2")

    def test_no_model_files_means_nothing_happens(self):
        with mock.patch.dict(mem._sense, {"tried": False, "model": None, "id": None}):
            self.assertIsNone(mem._sense_model())
            self.assertEqual(mem.sense_tick(self.db), 0)
            self.assertIn("no model", mem.sense_report(self.db))

    def test_the_report_counts_and_never_quotes(self):
        mem.sense_tick(self.db)
        report = mem.sense_report(self.db)
        self.assertIn("3 of 3 active memories labelled", report)
        self.assertIn("only the word list withholds: 1", report)
        self.assertIn("only the labels withhold: 1", report)
        for word in ("sugar", "bank", "Casa"):
            self.assertNotIn(word, report)


if __name__ == "__main__":
    unittest.main()
