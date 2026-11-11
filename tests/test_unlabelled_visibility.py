"""A note the user typed reaches a work-only app at once, with or without the background service (2026-11-11).

Live test: "remember that we deploy on Fridays" typed in Claude Code, then asked in Codex, came back empty. A note with
no sensitivity label is kept from work-only apps (mem._withheld: it may be about health or money), and only the daemon
made labels: the save asked it for one (ask_sense) and, when it did not answer, the note waited for the daemon's next
pass (sense_tick, once a minute), or for good with the service stopped. The save now makes the label itself then, with
the same local model. Nothing unlabelled is shown that was hidden before.
Synthetic only: temp stores and repos, a fake embedder and a fake classifier.

  .venv/bin/python -m unittest -q test_unlabelled_visibility
"""
import _env  # noqa: F401  (first: the test data folder and user)
import json
from unittest import mock

from lflabs_memory import instant, mem
from test_instant import Case
from test_sense import FakeModel

WORK = "we deploy on Fridays"
HEALTH = "my sugar levels need checking before every standup"  # no word of the lists: only the classifier knows
SALARY = "my salary is 185k after the raise"


class Unlabelled(Case):
    """Case's store has no daemon (mem.SOCK names no socket). The classifier is a model on disk, and the daemon ran
    once before (sense-flags.json exists), as on any install that has started."""

    def setUp(self):
        super().setUp()
        self.model = FakeModel()
        for p in (mock.patch.dict(mem._sense, {"tried": True, "model": self.model, "id": "fake1", "error": None,
                                               "flags": None, "mtime": None}),
                  mock.patch.object(mem, "SENSE_MODE", "on"),
                  mock.patch.object(mem, "_sense_flags_file", lambda: self.tmp / "sense-flags.json"),
                  mock.patch.object(mem, "CLIENT_SCOPES_MODE", "on"), mock.patch.object(mem, "KEYLESS_MODE", "on"),
                  mock.patch.dict(mem.RESTRICTED)):
            p.start()
            self.addCleanup(p.stop)
        mem.sense_tick(self.db)  # the daemon's first pass, before it stopped

    def cursor(self, query):
        """What Cursor's recall tool answers: a name with no key, so work only."""
        return mem.recall_text(self.db, query, scope=mem.client_scope("cursor", mem.UNKEYED), client="cursor")

    def cursor_start(self):
        return instant.recent_block(mem, self.db, str(self.repo), ("cursor", mem.UNKEYED))

    def labelled(self, mid):
        return self.db.execute("select 1 from sense where memory_id = ?", (mid,)).fetchone() is not None

    def promote(self):
        """The Stop hook found the words in the transcript: the user's own (run_typed)."""
        with self.db:
            self.db.execute("update memories set origin = 'owner_typed' where origin = 'unknown'")

    def no_label_made(self):
        """The classifier cannot run in this process either (its load failed)."""
        return mock.patch.dict(mem._sense, {"model": None})


class TypedWorkNote(Unlabelled):
    def test_visible_to_a_work_only_app_right_after_the_save(self):
        self.hook(f"remember that {WORK}")
        r = self.rows()[0]
        self.assertEqual((r["origin"], r["section"]), ("unknown", "projects"))  # the Stop hook has not run yet
        self.assertTrue(self.labelled(r["id"]))  # by the hook's own process: no daemon answered
        self.assertIn("Fridays", self.cursor("when do we deploy?"))

    def test_and_in_its_start_block_once_the_reply_finished(self):
        self.hook(f"remember that {WORK}")
        self.assertEqual(self.cursor_start(), "")  # still an app's save: never in a start block (held_back)
        self.promote()
        self.assertIn("Fridays", self.cursor_start())

    def test_the_daemon_labels_it_when_it_answers(self):
        asked = []
        with mock.patch.object(mem, "_ask_sense_daemon", lambda ids, wait: asked.append(list(ids)) or True), \
                mock.patch.object(mem, "sense_now", lambda *a: self.fail("labelled twice")):
            self.hook(f"remember that {WORK}")
        self.assertEqual(asked, [[self.rows()[0]["id"]]])

    def test_a_correction_made_at_the_stop_hook_is_labelled_too(self):
        """A typed note that corrects an older one is saved when the reply finishes (_typed_apply): no label was
        asked for it at all before."""
        old, _ = mem.remember(self.db, "we deploy on Thursdays", "project", "orchard", origin="owner_typed")
        mem.sense_now(self.db, [old])
        data = {"prompt": f"remember that {WORK}", "session_id": "s1", "transcript_path": "/tmp/x.jsonl"}
        mem.defer_typed(self.db, data, "desk", "remember-text", json.dumps(
            {"fact": WORK, "kind": "project", "project": "orchard", "via": "desk", "root": None, "replace": old}))
        with mock.patch.object(mem, "typed_in_transcript", lambda *a, **k: True):
            mem.run_typed(self.db, "s1")
        new = self.db.execute("select id from memories where content = ? and status = 'active'", (WORK,)).fetchone()[0]
        self.assertTrue(self.labelled(new))
        self.assertIn("Fridays", self.cursor("when do we deploy?"))

    def test_without_a_label_it_still_waits(self):
        """No daemon and no classifier here: hidden, as before, until one labels it."""
        with self.no_label_made():
            self.hook(f"remember that {WORK}")
            self.promote()
            self.assertFalse(self.labelled(self.rows()[0]["id"]))
            self.assertNotIn("Fridays", self.cursor("when do we deploy?"))
        mem.sense_tick(self.db)  # the daemon's next pass
        self.assertIn("Fridays", self.cursor("when do we deploy?"))


class TypedPersonalNote(Unlabelled):
    def hidden(self, words, query, start=True):
        for when in ("saved", "the user's own"):
            with self.subTest(when=when):
                self.assertNotIn(words[:20], self.cursor(query))
                if start:
                    self.assertNotIn(words[:20], self.cursor_start())
            self.promote()

    def test_a_health_note_the_word_lists_miss_stays_hidden(self):
        self.assertFalse(mem.personal_fact(HEALTH) or any(p.search(HEALTH) for p in mem.WITHHELD_RE))
        self.hook(f"remember that {HEALTH}")
        r = self.rows()[0]
        self.assertEqual(r["section"], "projects")  # filed where a work-only app looks: the label is what hides it
        self.assertIn(r["id"], mem._sense_flags()["withheld"])
        self.hidden(HEALTH, "what about my sugar levels?")

    def test_and_with_no_label_made(self):
        with self.no_label_made():
            self.hook(f"remember that {HEALTH}")
            self.assertFalse(self.labelled(self.rows()[0]["id"]))
            # recall and the start block alike: the user's own unlabelled words wait for their label too (0.1.9;
            # a start block read them by the word lists before, which miss this one)
            self.hidden(HEALTH, "what about my sugar levels?")

    def test_an_unlabelled_health_note_is_not_in_a_start_block_while_a_labelled_work_note_is(self):
        """The user's own typed words, both: the health note (no word of the lists) could get no label, the work
        note has one. A work-only app's start block shows the work note only; recall_full by its id finds nothing."""
        self.assertFalse(mem.personal_fact(HEALTH) or any(p.search(HEALTH) for p in mem.WITHHELD_RE))
        with self.no_label_made():
            self.hook(f"remember that {HEALTH}")
        health = self.rows()[0]["id"]
        self.hook(f"remember that {WORK}")
        self.promote()  # the Stop hook: both are the user's own words now
        self.assertEqual({r["origin"] for r in self.rows()}, {"owner_typed"})
        self.assertEqual((self.labelled(health), self.row(health)["section"]), (False, "projects"))
        with self.no_label_made():  # still none for the health note
            start = self.cursor_start()
            self.assertIn("Fridays", start)
            self.assertNotIn("sugar", start)
            scope = mem.client_scope("cursor", mem.UNKEYED)
            self.assertEqual(mem.full_text(self.db, health, scope=scope), f"no active memory #{health}")
            self.assertIn("sugar", instant.recent_block(mem, self.db, str(self.repo), None))  # a full view has it
        mem.sense_tick(self.db)  # labelled at last: health, so hidden for good
        self.assertIn(health, mem._sense_flags()["withheld"])
        self.assertNotIn("sugar", self.cursor_start())

    def test_a_salary_note_stays_hidden_labelled_or_not(self):
        for label in (True, False):
            with self.subTest(labelled=label):
                with self.db:
                    self.db.execute("delete from memories")
                if label:
                    self.hook(f"remember that {SALARY}")
                else:
                    with self.no_label_made():
                        self.hook(f"remember that {SALARY}")
                r = self.rows()[0]
                self.assertEqual((r["section"], self.labelled(r["id"])), ("personal", label))
                with self.no_label_made() if not label else mock.patch.dict(mem._sense):
                    self.hidden(SALARY, "what is my salary?")


class AppSave(Unlabelled):
    """An app's own save (the remember tool) is held exactly as before."""

    def test_an_unlabelled_app_save_is_hidden_from_work_only_apps(self):
        with self.no_label_made():
            mid, _ = mem.app_remember(self.db, "The orchard cache warms up at six", "project", "orchard", via="cursor")
        self.assertEqual((self.row(mid)["origin"], self.labelled(mid)), ("unknown", False))
        self.assertNotIn("cache warms", self.cursor("when does the cache warm up?"))

    def test_a_labelled_one_is_found_by_recall_but_never_put_in_a_prompt(self):
        mid, said = mem.app_remember(self.db, "The orchard cache warms up at six", "project", "orchard", via="cursor")
        self.assertIn("confirm memory", said)
        self.assertTrue(mem.held_back(self.row(mid)))
        self.assertIn("cache warms", self.cursor("when does the cache warm up?"))
        self.assertEqual(self.cursor_start(), "")

    def test_a_typed_save_is_held_until_the_transcript_shows_it(self):
        self.hook(f"remember that {WORK}")
        self.assertTrue(mem.held_back(self.rows()[0]))  # the label changed nothing of that
