"""Forgetting reaches copies and stays forgotten (2026-10-16): a forgotten nickname had survived in a second copy, and
an old transcript could save a forgotten thing again.

  .venv/bin/python -m unittest -q test_forget
"""
import _env  # noqa: F401  (first: the test data folder and user)
import unittest
from unittest import mock

import numpy as np

from lflabs_memory import mem
from test_redact import DBCase, fake_embed
from _fakes import limited_scope  # noqa: E402

LIMITED = limited_scope()  # a restricted reader, as a host's channel

BASE = fake_embed(["the nickname"])[0]


def near(sim, salt):
    """A unit vector with cosine `sim` to BASE."""
    v = fake_embed([salt])[0]
    v = v - (v @ BASE) * BASE
    v /= np.linalg.norm(v)
    return (sim * BASE + np.sqrt(1 - sim * sim) * v).astype(np.float32)


def embed(texts, query=False):
    """Texts tagged [same] embed as BASE, [similar] at cosine 0.92 to it; the rest as fake_embed."""
    out = []
    for t in texts:
        if "[same]" in t:
            out.append(BASE.copy())
        elif "[similar]" in t:
            out.append(near(0.92, t))
        else:
            out.append(fake_embed([t])[0])
    return out


class Forget(DBCase):
    def setUp(self):
        super().setUp()
        p = mock.patch.object(mem, "embed", embed)
        p.start()
        self.addCleanup(p.stop)

    def status(self, mid):
        return self.row(mid)["status"]

    def test_a_copy_saved_separately_goes_too(self):
        a, _ = mem.remember(self.db, "Odile's childhood nickname is Pipoca [same]")
        b, _ = mem.remember(self.db, "Nickname from childhood: Pipoca [same]", source="meeting:m1#1")  # never merged
        self.assertNotEqual(a, b)
        self.assertTrue(mem.forget_reply(self.db, a).startswith(f"Deleted #{a} and its copies #{b} for good: its text"))
        self.assertEqual((self.status(a), self.status(b)), ("retired", "retired"))
        self.assertEqual((self.row(a)["content"], self.row(b)["content"]), ("", ""))

    def test_a_similar_memory_stays_and_is_named(self):
        a, _ = mem.remember(self.db, "Odile's childhood nickname is Pipoca [same]")
        c, _ = mem.remember(self.db, "Her brother still calls her by the old nickname [similar]")
        reply = mem.forget_reply(self.db, a)
        self.assertEqual(self.status(c), "active")  # alike is not a copy: a newer version may need to stay
        self.assertIn(f"Similar memories still active, check whether they mention what should go: #{c}", reply)

    def test_an_old_transcript_cant_save_it_again(self):
        a, _ = mem.remember(self.db, "Odile's childhood nickname is Pipoca [same]")
        mem.forget(self.db, a)
        mid, why = mem.remember(self.db, "Childhood nickname: Pipoca [same]", origin="unknown")
        self.assertIsNone(mid)
        self.assertEqual(why, f"refused: it restates memory #{a}, which was forgotten")
        for o in ("outside", "guest", "open_mic"):
            self.assertIsNone(mem.remember(self.db, f"Pipoca, from {o} [same]", origin=o)[0])

    def test_saying_it_again_himself_keeps_it(self):
        a, _ = mem.remember(self.db, "Odile's childhood nickname is Pipoca [same]")
        mem.forget(self.db, a)
        mid, action = mem.remember(self.db, "Call me Pipoca again [same]", origin="owner_typed")
        self.assertEqual(action, "added")
        self.assertIsNone(mem.forgotten_match(self.db, BASE))  # his to keep now, from any source
        # an app's restatement leaves the user's own words as they are (0.1.4: mem.KEEPS_ORIGIN)
        self.assertEqual(mem.remember(self.db, "Pipoca, again [same]", origin="unknown"), (mid, "unchanged"))

    def test_restore_cannot_bring_it_back(self):
        a, _ = mem.remember(self.db, "Odile's childhood nickname is Pipoca [same]")
        mem.forget(self.db, a)
        self.assertFalse(mem.restore(self.db, a))  # deleted for good
        self.assertEqual(mem.forgotten_match(self.db, BASE), a)

    def test_unrelated_memories_are_untouched(self):
        a, _ = mem.remember(self.db, "Odile's childhood nickname is Pipoca [same]")
        other, _ = mem.remember(self.db, "The garage code changed in March")
        mem.forget(self.db, a)
        self.assertEqual(self.status(other), "active")
        self.assertEqual(mem.remember(self.db, "Swim class moved to Thursdays", origin="unknown")[1], "added")

    def test_no_text_is_kept_for_the_forgotten(self):
        a, _ = mem.remember(self.db, "Odile's childhood nickname is Pipoca [same]")
        mem.forget(self.db, a)
        cols = [r[1] for r in self.db.execute("pragma table_info(forgot_vec)")]
        self.assertEqual(sorted(c for c in cols if c != "rowid"), ["embedding"])

    def file_as(self, mid, section):
        self.db.execute("update memories set section = ? where id = ?", (section, mid))
        self.db.commit()

    def test_a_restricted_scope_only_forgets_what_it_may_see(self):
        a, _ = mem.remember(self.db, "Odile's childhood nickname is Pipoca [same]")
        hidden, _ = mem.remember(self.db, "Pipoca, said in a meeting [same]", source="meeting:m3#1")
        self.file_as(a, "personal")
        self.file_as(hidden, "people")  # a section the limited reader never sees
        self.assertEqual(mem.forget_reply(self.db, hidden, LIMITED), f"no active memory #{hidden}")
        self.assertEqual(self.status(hidden), "active")
        self.assertTrue(mem.forget_reply(self.db, a, LIMITED).startswith(f"Deleted #{a} for good"))  # its hidden copy is not named
        self.assertEqual(self.status(hidden), "active")  # nor retired from there

    def test_a_restricted_scope_is_not_told_about_hidden_similar_memories(self):
        a, _ = mem.remember(self.db, "Odile's childhood nickname is Pipoca [same]")
        c, _ = mem.remember(self.db, "Her brother still calls her by the old nickname [similar]")
        self.file_as(a, "personal")
        self.file_as(c, "people")
        self.assertNotIn(f"#{c}", mem.forget_reply(self.db, a, LIMITED))

    def test_a_deleted_row_leaves_no_forgotten_vector(self):
        a, _ = mem.remember(self.db, "Said in the meeting [same]", source="meeting:m2#1")
        mem.forget(self.db, a)
        self.assertEqual(mem.forget_source(self.db, "meeting:m2#"), 1)
        self.assertIsNone(mem.forgotten_match(self.db, BASE))


if __name__ == "__main__":
    unittest.main()
