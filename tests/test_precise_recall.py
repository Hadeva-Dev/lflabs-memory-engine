"""Precise recall excerpts (PRECISE_RECALL_MODE): a recalled memory too long for its room follows excerpts.py's rules
(a statement with the later line that replaces it, never a cut mid-word), shadow sends today's window unchanged and
logs counts only, and the forget, scope, secret and writer guards hold with it on. Temp databases, fake embeddings
and a fake reranker only; no model call.

  .venv/bin/python -m unittest -q test_precise_recall
"""
import _env  # noqa: F401  (first: the test data folder and user)
import json
import os
import queue
import time
import unittest
from unittest import mock

from lflabs_memory import excerpts
from lflabs_memory import mem
import test_clients
import test_recall
import test_secrets
from test_budget import LONG
from test_redact import DBCase
from _fakes import fake_rerank

FILL = " ".join(f"Item {i}: routine filler about builds, deploys and nothing in particular, kept for the record."
                for i in range(30))
# Today's window holds the most of the query's words: the old venue, cut mid-word, without the decision that
# replaces it 1,500 characters later.
OFFSITE = ("Team offsite planning log. The offsite venue is the Harbor Hotel in Lisbon, offsite for 40 people, venue "
           "booked in May. " + FILL + " Decision: the offsite moved to Porto instead of the Harbor Hotel, after the "
           "Lisbon venue cancelled.")
QUERY = "which hotel is the offsite venue"


def cut(text, query, room, mode, ord=0, chunk=None, when="superseded"):
    with mock.patch.object(mem, "PRECISE_RECALL_MODE", mode), mock.patch.object(mem, "RECALL_PRECISE_WHEN", when):
        return mem._recall_excerpt({"content": text, "chunk": chunk or text, "ord": ord}, query, room)


class Rules(unittest.TestCase):
    def test_the_window_drops_the_decision_and_the_precise_excerpt_keeps_it(self):
        old, _ = cut(OFFSITE, QUERY, 1500, "off")
        self.assertIn("Harbor Hotel in Lisbon", old)
        self.assertNotIn("Porto", old)
        kinds = {k for k, _ in excerpts.slice_issues(OFFSITE, [old], mem.plain, mem.date_words)}
        self.assertLessEqual({"superseder_dropped", "mid_word_cut"}, kinds)
        new, more = cut(OFFSITE, QUERY, 1500, "on")
        self.assertLessEqual(len(new), 1500)
        self.assertGreater(more, 0)
        self.assertTrue(new.startswith("Team offsite planning log. The offsite venue is the Harbor Hotel"))
        self.assertTrue(new.endswith("Decision: the offsite moved to Porto instead of the Harbor Hotel, after the "
                                     "Lisbon venue cancelled."))
        text, keep = excerpts.focus(OFFSITE, QUERY, 1500, mem.plain, mem.date_words, fill_past_misfits=True)
        self.assertEqual(text, new)
        self.assertEqual(excerpts.precision_issues(OFFSITE, text, keep, mem.plain, mem.date_words), [])

    def test_the_bench_traps_at_any_room(self):
        for text, query in LONG:
            for room in (200, 300, 400):
                self.assertGreater(len(text), room)
                new, _ = cut(text, query, room, "on", when="always")
                got, keep = excerpts.focus(text, query, room, mem.plain, mem.date_words, fill_past_misfits=True)
                self.assertEqual(new, got)
                self.assertLessEqual(len(new), room, (query, room))
                self.assertEqual(excerpts.precision_issues(text, got, keep, mem.plain, mem.date_words), [],
                                 (query, room))
        mei = cut(LONG[1][0], LONG[1][1], 400, "on", when="always")[0]
        self.assertIn("moved back to Taipei", mei)  # the correction comes with the statement it corrects
        self.assertIn("That is no longer true", mei)

    def test_on_keeps_a_window_that_drops_no_correction(self):
        note = test_recall.long_note("Build box notes: how the staging machine is kept.")
        self.assertEqual(cut(note, "quokka-vault rotation", 1200, "on"), cut(note, "quokka-vault rotation", 1200, "off"))
        self.assertNotEqual(cut(note, "quokka-vault rotation", 1200, "on", when="always"),
                            cut(note, "quokka-vault rotation", 1200, "off"))
        self.assertNotEqual(cut(OFFSITE, QUERY, 1500, "on"), cut(OFFSITE, QUERY, 1500, "off"))

    def test_the_room_left_goes_past_a_unit_that_does_not_fit(self):
        text = ("Trip log for the spring. " + "The train route went through many small towns and valleys, " * 6
                + "and it rained. The hotel was the Miramar in Lisbon.")
        hook, _ = excerpts.focus(text, "spring trip", 200)  # the hook's fill stops at the first misfit
        self.assertEqual(hook, "Trip log for the spring. …")
        new, keep = excerpts.focus(text, "spring trip", 200, fill_past_misfits=True)
        self.assertEqual(new, "Trip log for the spring. … The hotel was the Miramar in Lisbon.")
        self.assertEqual(excerpts.precision_issues(text, new, keep), [])

    def test_a_memory_that_fits_is_whole_in_every_mode(self):
        text = "Robin's offsite is in Porto, 2026-11-04."
        for mode in ("off", "shadow", "on"):
            self.assertEqual(cut(text, QUERY, 1500, mode), (mem.date_words(text), 0))

    def test_shadow_cuts_as_off(self):
        note = test_recall.long_note("Build box notes: how the staging machine is kept.")
        for args in ((OFFSITE, QUERY, 1500), (OFFSITE, "Porto", 300), (note, "quokka-vault rotation", 1200, 3,
                                                                       note[9000:12000])):
            self.assertEqual(cut(*args[:3], "shadow", *args[3:]), cut(*args[:3], "off", *args[3:]))

    def test_the_audit_passes_a_clean_cut(self):
        text = "Garden log. The shed is cedar. It cost 900 dollars. Tomatoes failed in May."
        self.assertEqual(excerpts.slice_issues(text, ["Garden log. The shed is cedar. It cost 900 dollars.…"]), [])
        self.assertEqual([k for k, _ in excerpts.slice_issues(text, ["It cost 900 dollars."])], ["orphaned_lean"])
        self.assertEqual([k for k, _ in excerpts.slice_issues(text, ["Garden log. The sh…"])], ["mid_word_cut"])


class Recall(DBCase):
    def setUp(self):
        super().setUp()
        for p in (mock.patch.object(mem, "rerank", fake_rerank({"offsite": 5.0, "quokka": 5.0})),
                  mock.patch.object(mem, "MEETBRIEF_STATE", self.tmp / "nope.json"),
                  mock.patch.object(mem, "VOICE_SESSION", self.tmp / "none.json"),
                  mock.patch.object(mem, "TELEGRAM_SESSION", self.tmp / "none.json"),
                  mock.patch.dict(os.environ, {"LFLABS_MEMORY_SCOPE": ""}),
                  mock.patch.dict(mem._excerpt_shadow, {"queue": queue.Queue()})):
            p.start()
            self.addCleanup(p.stop)
        self.mid = mem.remember(self.db, OFFSITE, "project", "lantern", origin="owner_typed")[0]
        self.short = mem.remember(self.db, "The offsite budget is 9,000 euros.", "project", "lantern", origin="owner_typed")[0]

    def recall(self, mode, query=QUERY):
        with mock.patch.object(mem, "PRECISE_RECALL_MODE", mode):
            return mem.recall_text(self.db, query, k=8, max_chars=2000)

    def item(self, out, mid=None):
        return next(i for i in json.loads(out) if i["id"] == (mid or self.mid))

    def test_on_sends_the_decision(self):
        item = self.item(self.recall("on"))
        self.assertIn("Porto instead of the Harbor Hotel", item["content"])
        self.assertRegex(item["content"], rf"\(\d+ more chars; recall_full {self.mid} for all\)$")
        self.assertTrue(mem._excerpt_shadow["queue"].empty())

    def test_shadow_sends_today_and_queues_each_cut_memory_once(self):
        off = self.recall("off")
        self.assertTrue(mem._excerpt_shadow["queue"].empty())
        self.assertEqual(self.recall("shadow"), off)
        self.assertNotIn("Porto", self.item(off)["content"])
        q = mem._excerpt_shadow["queue"]
        content, query, room, shown = q.get_nowait()
        self.assertTrue(q.empty())  # the short memory was whole: nothing to compare
        self.assertEqual((content, query), (OFFSITE, QUERY))
        self.assertIn(shown, self.item(off)["content"])

    def test_the_log_holds_counts_only(self):
        self.recall("shadow")
        row = mem.excerpt_one(mem._excerpt_shadow["queue"].get_nowait())
        self.assertEqual(len(row), 10)
        self.assertTrue(all(isinstance(v, (int, float)) for v in row))
        at, room, chars, chars_precise, issues, issues_precise, sup, sup_precise, mid_word, ms = row
        self.assertGreater(issues, 0)
        self.assertGreater(sup, 0)
        self.assertEqual(mid_word, 1)
        self.assertEqual((issues_precise, sup_precise), (0, 0))
        self.assertLessEqual(chars_precise, room)
        with self.db:
            self.db.execute(mem.EXCERPT_LOG_SQL)
            self.db.execute("insert into excerpt_log values (?,?,?,?,?,?,?,?,?,?)", row)
        rep = mem.excerpt_report(self.db)
        self.assertIn("recalled memories cut to fit (last 7 days): 1", rep)
        self.assertIn("a statement without what corrects or replaces it: window 1, as on would send 0", rep)
        for word in ("offsite", "harbor", "porto", "lisbon", "hotel", "venue", "filler"):
            self.assertNotIn(word, rep.lower())

    def test_the_daemon_thread_writes_the_log(self):
        with mock.patch.dict(mem._excerpt_shadow, {"queue": None}):
            mem.excerpt_shadow_thread()
            self.recall("shadow")
            for _ in range(100):
                if self.db.execute("select 1 from sqlite_master where name = 'excerpt_log'").fetchone() and \
                        self.db.execute("select count(*) from excerpt_log").fetchone()[0]:
                    break
                time.sleep(0.05)
            self.assertEqual([tuple(r) for r in self.db.execute("select superseded_precise, mid_word from excerpt_log")],
                             [(0, 1)])

    def test_no_log_yet(self):
        self.assertIn("no excerpt log yet", mem.excerpt_report(self.db))


class Guards(DBCase):
    """With it on, an excerpt only ever selects from a memory already allowed whole."""

    def setUp(self):
        super().setUp()
        for p in (mock.patch.object(mem, "rerank", fake_rerank({"offsite": 5.0})),
                  mock.patch.object(mem, "MEETBRIEF_STATE", self.tmp / "nope.json"),
                  mock.patch.object(mem, "VOICE_SESSION", self.tmp / "none.json"),
                  mock.patch.object(mem, "TELEGRAM_SESSION", self.tmp / "none.json"),
                  mock.patch.dict(os.environ, {"LFLABS_MEMORY_SCOPE": ""}),
                  mock.patch.object(mem, "PRECISE_RECALL_MODE", "on")):
            p.start()
            self.addCleanup(p.stop)

    def ids(self, **kw):
        return [i["id"] for i in json.loads(mem.recall_text(self.db, QUERY, k=8, max_chars=2000, **kw) or "[]")]

    def test_forgotten_never_comes_back(self):
        mid = mem.remember(self.db, OFFSITE, origin="owner_typed")[0]
        self.assertIn(mid, self.ids())
        mem.forget(self.db, mid)
        self.assertEqual(mem.recall_text(self.db, QUERY, k=8, max_chars=2000), "no memories found")

    def test_a_cloud_ais_words_stay_outside_and_quoted_text_stays_marked(self):
        wid = mem.remember(self.db, "Orchard " + OFFSITE, writer="remote-claude", origin="owner_typed")[0]
        said = mem.remember(self.db, "Mia said, by text on 2026-10-09: " + OFFSITE, "reference", "messages",
                            origin="outside")[0]
        items = {i["id"]: i for i in json.loads(mem.recall_text(self.db, QUERY, k=8, max_chars=8000))}
        self.assertEqual(items[wid]["origin"], "from outside")
        self.assertIn("recall_full", items[wid]["content"])
        self.assertTrue(items[said]["content"].startswith(mem.QUOTED_MARK))
        self.assertIn("recall_full", items[said]["content"])
        ctx = mem.prompt_context(self.db, "which hotel is the team offsite venue now", "s1") or ""
        self.assertNotIn(f"#{wid} ", ctx)  # a cloud AI's words never arrive unasked


# The existing recall, secret and client-scope tests, again with it on.
class On:
    def setUp(self):
        super().setUp()
        p = mock.patch.object(mem, "PRECISE_RECALL_MODE", "on")
        p.start()
        self.addCleanup(p.stop)


class RecallOn(On, test_recall.Recall):
    pass


class SecretsNeverVolunteeredOn(On, test_secrets.NeverVolunteered):
    pass


class SecretsGivenWhenHeAsksOn(On, test_secrets.GivenWhenHeAsks):
    pass


class SecretsNotForAnyoneElseOn(On, test_secrets.NotForAnyoneElse):
    pass


class ClientsFilteringOn(On, test_clients.Filtering):
    pass


class ClientsForgetOn(On, test_clients.Forget):
    pass


if __name__ == "__main__":
    unittest.main()
