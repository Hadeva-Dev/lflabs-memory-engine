"""Look again (2026-10-18): a prompt that likely wants Robin's memory but gets nothing past the gate is told, in one
line with no memory text, to call `recall` with other words before answering "I don't know".

  .venv/bin/python -m unittest -q test_look_again
"""
import _env  # noqa: F401  (first: the test data folder and user)
import unittest
from unittest import mock

from lflabs_memory import mem
from test_redact import DBCase


def fake_rerank(query, texts):
    return [0.0 for _ in texts]


class LookAgain(DBCase):
    def setUp(self):
        super().setUp()
        p = mock.patch.object(mem, "rerank", fake_rerank)
        p.start()
        self.addCleanup(p.stop)
        self.mid, _ = mem.remember(self.db, "The garage code changed in March to the new keypad", origin="owner_typed")

    def context(self, prompt="what did we decide about the garage", wants=True, kept=None):
        with mock.patch.object(mem, "gate", lambda db, prompt, hits, scope=None, memo=None, **k: kept or []), \
                mock.patch.object(mem, "_wants_memory", lambda *a: wants):
            return mem.prompt_context(self.db, prompt, "s1")

    def test_nothing_strong_but_wanted_says_look_again(self):
        out = self.context()
        self.assertEqual(out, mem.LOOK_AGAIN_LINE)
        self.assertNotIn("garage", out)  # never a memory's words
        self.assertIsNone(self.db.execute("select 1 from injections").fetchone())  # nothing counts as shown

    def test_not_wanted_stays_silent(self):
        self.assertIsNone(self.context(wants=False))

    def test_off_in_the_bench(self):
        with mock.patch.object(mem, "LOOK_AGAIN", False):
            self.assertIsNone(self.context())

    def test_memories_shown_as_before(self):
        hits = mem.search(self.db, "garage code", k=4)
        out = self.context(kept=hits)
        self.assertTrue(out.startswith(f"Recalled from {mem.LABEL} (long-term memory"))
        self.assertNotIn(mem.LOOK_AGAIN_LINE, out)

    def test_short_or_not_asked_prompts_get_nothing(self):
        self.assertIsNone(self.context(prompt="ok thanks"))

    def test_wants_memory_uses_the_needs_memory_model(self):
        g = {"need": {"min": 0.5}}
        with mock.patch.object(mem, "gate_params", lambda: g), mock.patch.object(mem, "GATE", "calibrated"), \
                mock.patch.object(mem, "p_relevant", lambda s, g: 0.1), \
                mock.patch.object(mem, "need_features", lambda *a: []), mock.patch.object(mem, "need_terms", lambda *a: []):
            with mock.patch.object(mem, "need_prob", lambda x, g: 0.7):
                self.assertTrue(mem._wants_memory(self.db, "p", [{"score": 1.0}], None, {}))
            with mock.patch.object(mem, "need_prob", lambda x, g: 0.2):
                self.assertFalse(mem._wants_memory(self.db, "p", [{"score": 1.0}], None, {}))
            self.assertTrue(mem._wants_memory(self.db, "p", [], None, {"need": 0.9}))  # the gate's own answer reused
        with mock.patch.object(mem, "gate_params", lambda: None):
            self.assertFalse(mem._wants_memory(self.db, "p", [], None, {}))


if __name__ == "__main__":
    unittest.main()
