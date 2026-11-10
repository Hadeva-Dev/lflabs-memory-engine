"""The user's own life stays theirs (0.1.4, reproduced 2026-11-08): a first-person fact about health, family,
relationships, money or the law, typed in a repo or saved by an app, is filed personal, never projects (which
work-only apps see); a memory the sensitivity model has not read yet is withheld from work-only apps; the model's
credential label needs a credential in the words.

  .venv/bin/python -m unittest -q test_personal_filing
"""
import _env  # noqa: F401  (first: the test data folder and user)
import json
import unittest
from unittest import mock

from lflabs_memory import instant, mem
from test_redact import DBCase

PERSONAL = [
    "my A1C came back at 7.2",
    "my wife and I are separating, keep Friday afternoons free for the lawyer",
    "my therapist moved my appointment to Tuesday",
    "I'm on 50mg of sertraline now",
    "my daughter has a peanut allergy",
    "my salary is going up to 140k in January",
    "we're two months behind on the mortgage",
    "my mom's biopsy results come back Friday",
    "I have a custody hearing next week",
    "my boyfriend and I broke up last night",
]
WORK = [
    "the staging database is Postgres 16 on port 5433",
    "my PR for the billing API needs a review before Friday",
    "we deploy the orchard app on Fridays",
    "my branch is rebased on main and the tests pass",
]


class Detect(unittest.TestCase):
    def test_personal_facts(self):
        for t in PERSONAL:
            self.assertTrue(mem.personal_fact(t), t)

    def test_work_facts_are_not(self):
        for t in WORK:
            self.assertFalse(mem.personal_fact(t), t)


class Filing(DBCase):
    def test_an_apps_save_of_a_personal_fact_is_personal(self):
        for t in PERSONAL:
            mid, reply = mem.app_remember(self.db, t, "project", "orchard", via="codex", scope=None)
            self.assertEqual((self.row(mid)["kind"], self.row(mid)["section"]), ("user", "personal"), t)
            self.assertIn("not apps set to work only", reply)
        for t in WORK[1:]:
            mid, _ = mem.app_remember(self.db, t, None, "orchard", via="codex", scope=None)
            self.assertEqual(self.row(mid)["section"], "projects", t)

    def test_a_typed_personal_fact_in_a_repo_is_personal(self):
        with mock.patch.object(instant, "_typed", lambda *a: True), \
                mock.patch.object(instant, "project_of", lambda m, cwd: ("/repo", "orchard")):
            for t in PERSONAL:
                got = instant.try_save(mem, self.db, f"remember that {t}", "s1", "/repo/src", "desk")
                self.assertIsNotNone(got.id, (t, got.refused))
                r = self.row(got.id)
                self.assertEqual((r["kind"], r["project"], r["section"]), ("user", None, "personal"), t)

    def test_none_of_them_reaches_a_work_only_app(self):
        for t in PERSONAL:
            mem.app_remember(self.db, t, None, "orchard", via="codex", scope=None)
        hide = mem.client_hide("cursor", mem.UNKEYED) if mem.KEYLESS_MODE == "on" else mem.WORK_HIDE
        for r in self.db.execute("select * from memories").fetchall():
            self.assertFalse(mem._allowed(r, (), tuple(mem.WORK_HIDE)), r["content"])


class Labels(DBCase):
    TH = {k: {"threshold": 0.5} for k in mem.SENSE_WITHHELD}

    def test_the_credential_label_needs_a_credential(self):
        probs = {"secret_credential": 0.56}
        self.assertFalse(mem._sense_withheld(probs, self.TH, "the staging database is Postgres 16 on port 5433"))
        self.assertTrue(mem._sense_withheld(probs, self.TH, "the staging database password is hunter22"))
        self.assertTrue(mem._sense_withheld({"secret_credential": 0.56, "money": 0.9}, self.TH, "port 5433"))

    def test_unlabelled_is_withheld_once_a_model_labels_memories(self):
        mid, _ = mem.remember(self.db, "The staging deploy window is Thursday 4pm", "project", "orchard")
        row = self.row(mid)
        with mock.patch.object(mem, "SENSE_MODE", "on"), \
                mock.patch.object(mem, "_sense_flags", lambda: {"labelled": set(), "withheld": set()}):
            self.assertTrue(mem._withheld(row))  # not read yet: withheld from work-only apps
        with mock.patch.object(mem, "SENSE_MODE", "on"), \
                mock.patch.object(mem, "_sense_flags", lambda: {"labelled": {mid}, "withheld": set()}):
            self.assertFalse(mem._withheld(row))
        with mock.patch.object(mem, "SENSE_MODE", "on"), mock.patch.object(mem, "_sense_flags", lambda: None):
            self.assertFalse(mem._withheld(row))  # no model on this Mac: the word list decides, as before



class RealModel(unittest.TestCase):
    def test_the_real_model_on_the_reported_false_positive(self):
        with mock.patch.dict(mem._sense, {"tried": False, "model": None}):  # tried afresh, outside any test store
            model = mem._sense_model()
            why = mem._sense.get("error")
        if model is None:
            self.skipTest(f"no sensitivity model here ({why})")
        t = "the staging database is Postgres 16 on port 5433"
        probs = model.classify([t])[0]
        self.assertFalse(mem._sense_withheld(probs, model.thresholds, t))
        t = "I take sertraline every morning"
        self.assertTrue(mem._sense_withheld(model.classify([t])[0], model.thresholds, t))


if __name__ == "__main__":
    unittest.main()
