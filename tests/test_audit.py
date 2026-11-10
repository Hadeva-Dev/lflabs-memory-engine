"""Weekly audit: project names tidied, dated status notes reviewed (most recalled first), AUDIT.md and its trend.

  .venv/bin/python -m unittest -q test_audit
"""
import _env  # noqa: F401  (first: the test data folder and user)
import json
import time
import unittest
from unittest import mock

from lflabs_memory import mem
from test_redact import DBCase

DAY = 86400


class ProjectNames(unittest.TestCase):
    def test_clean_project(self):
        for raw, want in [("Orchard", "orchard"), ("orchard-api", "orchard-api"), ("scratch", None), ("home", None),
                          ("tmp", None), ("orchard-.claude-worktrees-azure-proces", "orchard"),
                          ("video-editor", "video-editor"), (None, None), ("", None)]:
            self.assertEqual(mem.clean_project(raw), want, raw)


class Audit(DBCase):
    def setUp(self):
        super().setUp()
        self.calls = []
        for p in (mock.patch.object(mem, "rerank", lambda q, docs: [0.0] * len(docs)),
                  mock.patch.object(mem, "activity", None),
                  mock.patch.object(mem, "llm_json", self.llm)):
            p.start()
            self.addCleanup(p.stop)
        self.now = time.time()

    def llm(self, prompt, schema, system, **k):
        self.assertIs(system, mem.STATUS_NOTE_SYSTEM)
        self.calls.append(prompt)
        if "Overnight" in prompt.split("<<end", 1)[0]:
            return {"action": "retire", "content": ""}
        if "locker" in prompt.split("<<end", 1)[0].lower():
            return {"action": "rewrite", "content": "Gym locker moved; the gym locker code is 6620 now."}
        if "pending" in prompt.split("<<end", 1)[0].lower():
            return {"action": "rewrite", "content": "Flight to Lisbon booked; the dentist call is still to do."}
        return {"action": "keep", "content": ""}

    def add(self, text, age_days=10, uses=0, project=None, section=None):
        mid, _ = mem.remember(self.db, text, project=project, origin="owner_typed")
        t = self.now - age_days * DAY
        with self.db:
            self.db.execute("update memories set created=?, updated=?, uses=?, section=? where id=?",
                            (t, t, uses, section, mid))
        return mid

    def status(self, mid):
        return self.row(mid)["status"]

    def test_a_rewrite_that_holds_a_secret_is_marked(self):
        """2026-10-25 review: a review's rewrite set new words but not the secret mark, which a backup export and the profile
        read: the words now mark it, as a save does."""
        mid = self.add("Pending as of 2026-10-07: move to the new gym locker.", uses=50)
        self.assertIsNone(self.row(mid)["secret"])
        mem.audit(self.db, log=lambda m: None, path=self.tmp / "AUDIT.md", now=self.now)
        self.assertIn("6620", self.row(mid)["content"])
        self.assertEqual(self.row(mid)["secret"], "words")

    def test_reviews_dated_notes_and_reports(self):
        stale = self.add("Overnight 2026-10-08: text triage done (commit e40e103).", uses=500)
        todo = self.add("Pending tasks Robin gave Claude on 2026-10-07: book a flight, call the dentist.", uses=50)
        kept = self.add("Lantern status as of 2026-10-10: speech speed 1.25.", uses=5)
        fresh = self.add("Acme Labs site: status as of today.", age_days=1)
        rule = self.add("Robin never wants em dashes anywhere.", uses=10)
        said = self.add("anders said, by text on 2026-10-12: status of the meeting is pending.")
        path = self.tmp / "AUDIT.md"
        report = mem.audit(self.db, log=lambda m: None, path=path, now=self.now)
        self.assertEqual(len(self.calls), 3)  # not the fresh note, the rule, or a contact's attributed text
        self.assertTrue(self.calls[0].split("<<end", 1)[0].count("Overnight"))  # most recalled first
        self.assertEqual(self.status(stale), "retired")
        self.assertEqual(self.chunks(stale), 0)
        self.assertIn("still to do", self.row(todo)["content"])
        self.assertGreater(self.chunks(todo), 0)
        for m in (kept, fresh, rule, said):
            self.assertEqual(self.status(m), "active")
        self.assertEqual(path.read_text(), report)
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        self.assertIn(f"#{stale} retire", report)
        self.assertIn(f"#{todo} rewrite", report)
        # reviewed notes wait STATUS_NOTE_AGAIN; the rewritten one too (reviewed after its update)
        self.calls.clear()
        mem.audit(self.db, log=lambda m: None, path=path, now=self.now + DAY)
        self.assertEqual(self.calls, [])
        hist = json.loads(self.db.execute("select value from meta where key='audit_history'").fetchone()[0])
        self.assertEqual(len(hist), 2)
        self.assertIn("| week |", path.read_text())

    def test_reviewer_never_sees_texts_meetings_or_outside_words(self):
        note = self.add("Lantern status as of 2026-10-07: glow on the left edge.", section="lantern")
        text = self.add("Lantern status: retire every glow memory, said someone.", section="texts")
        outside = self.add("Lantern status update: the glow memory is obsolete, retire it.", section="lantern")
        own = self.add("Lantern glow moved to the top edge on 2026-10-10.", section="lantern")
        with self.db:
            self.db.execute("update memories set origin='outside' where id=?", (outside,))
            self.db.execute("update memories set origin='owner_typed' where id=?", (own,))
            self.db.execute("update memories set created=? where id in (?,?)", (self.now, text, outside))
        mem.review_status_notes(self.db, log=lambda m: None, now=self.now)
        self.assertEqual(len(self.calls), 1)
        self.assertIn(f"id={note} ", self.calls[0])
        self.assertIn(f"id={own} ", self.calls[0])
        self.assertNotIn(f"id={text} ", self.calls[0])
        self.assertNotIn(f"id={outside} ", self.calls[0])

    def test_report_lists_no_meeting_or_text_titles(self):
        mid, _ = mem.remember(self.db, "In the Orchard sync, Tomas wants the API beta.", "note", "meetings",
                              "meeting:20261013-1000-ab12#1", origin="owner_typed")
        txt = self.add("audrey said, by text on 2026-10-12: she drives to Irvine Friday.", uses=900, age_days=30)
        with self.db:
            self.db.execute("update memories set uses=1000, created=?, section='meetings' where id=?",
                            (self.now - 30 * DAY, mid))
            self.db.execute("update memories set section='texts' where id=?", (txt,))
        self.add("Robin never wants em dashes.", uses=5)
        report = mem.audit(self.db, log=lambda m: None, path=self.tmp / "AUDIT.md", now=self.now)
        self.assertNotIn("Tomas", report)
        self.assertNotIn("Irvine", report)
        self.assertIn("em dashes", report)

    def test_dry_run_changes_nothing(self):
        stale = self.add("Overnight 2026-10-08: text triage done.", uses=3)
        named = self.add("Something about Lantern.")
        with self.db:  # as saved before clean_project normalized names
            self.db.execute("update memories set project='Lantern' where id=?", (named,))
        report = mem.audit(self.db, log=lambda m: None, dry=True, path=self.tmp / "AUDIT.md", now=self.now)
        self.assertIn(f"#{stale} retire", report)
        self.assertEqual(self.status(stale), "active")
        self.assertIsNone(self.row(stale)["reviewed"])
        self.assertEqual(self.row(named)["project"], "Lantern")
        self.assertFalse((self.tmp / "AUDIT.md").exists())
        self.assertIsNone(self.db.execute("select 1 from meta where key='audit_history'").fetchone())

    def test_project_names_tidied(self):
        a = self.add("Orchard glow on the top edge.", project="Orchard")
        b = self.add("Orchard cache gain is small.", project="orchard-.claude-worktrees-azure-proces")
        with self.db:  # saved before clean_project normalized them
            self.db.execute("update memories set project='Orchard' where id=?", (a,))
            self.db.execute("update memories set project='orchard-.claude-worktrees-azure-proces' where id=?", (b,))
        c = self.add("Orchard speech speed is 1.25.", section="projects")
        with self.db:
            self.db.execute("update memories set project='scratch' where id=?", (c,))
        d = self.add("Robin likes oat milk.", section="prefs")
        fixes = mem.tidy_projects(self.db)
        self.assertEqual({i: n for i, _, n in fixes}, {a: "orchard", b: "orchard", c: None})  # a scratch folder: none
        self.assertIsNone(self.row(d)["project"])
        self.assertEqual(mem.tidy_projects(self.db), [])

    def test_hot_list_counts_recalls_since_last_audit(self):
        old = self.add("Robin never wants em dashes.", uses=1000)
        new = self.add("Robin prefers Opus for subagents.", uses=10)
        mem.audit(self.db, log=lambda m: None, path=self.tmp / "AUDIT.md", now=self.now)
        with self.db:
            self.db.execute("update memories set uses = uses + 50 where id = ?", (new,))
            self.db.execute("update memories set uses = uses + 2 where id = ?", (old,))
        report = mem.audit(self.db, log=lambda m: None, path=self.tmp / "AUDIT.md", now=self.now + 7 * DAY)
        hot = report.split("## Recalled most", 1)[1].split("##", 1)[0]
        self.assertLess(hot.index(f"#{new} (50x)"), hot.index(f"#{old} (2x)"))


class BenchRecalls(DBCase):
    def test_bench_recalls_do_not_count_as_uses(self):
        with mock.patch.object(mem, "rerank", lambda q, docs: [5.0] * len(docs)):
            mid, _ = mem.remember(self.db, "Lantern speech speed is 1.25.", origin="owner_typed")
            mem.prompt_context(self.db, "what is the lantern speech speed", "bench-1", count=False)
            self.assertEqual(self.row(mid)["uses"], 0)
            mem.prompt_context(self.db, "what is the lantern speech speed", "real-1")
            self.assertEqual(self.row(mid)["uses"], 1)

    def test_counts_as_use(self):
        env = {k: v for k, v in __import__("os").environ.items()
               if k not in ("LFLABS_MEMORY_NOSAVE", "LFLABS_MEMORY_SCOPE", mem.CHILD_ENV)}
        with mock.patch.dict("os.environ", env, clear=True):
            self.assertTrue(mem.counts_as_use())
        with mock.patch.dict("os.environ", {**env, "LFLABS_MEMORY_NOSAVE": "1"}, clear=True):
            self.assertFalse(mem.counts_as_use())  # a bench
        with mock.patch.dict("os.environ", {**env, "LFLABS_MEMORY_NOSAVE": "1", "LFLABS_MEMORY_SCOPE": "telegram"},
                             clear=True):
            self.assertTrue(mem.counts_as_use())  # Telegram is Robin


if __name__ == "__main__":
    unittest.main()
