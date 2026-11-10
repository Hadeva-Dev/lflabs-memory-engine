"""The calibrated inject gate (2026-10-15): gate.json's cut and the needs-memory model, with the raw RERANK_MIN cut as
the fallback.

  .venv/bin/python -m unittest -q test_gate
"""
import _env  # noqa: F401  (first: the test data folder and user)
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from lflabs_memory import mem


def hits(*scores):
    return [{"id": i, "score": s} for i, s in enumerate(scores)]


# A need model with one live input (the top candidate's calibrated score): it says "wants memory" only above p1 0.5.
FIXTURE = {"version": 1, "platt": {"a": 1.0, "b": 0.0}, "min_logit": -5.6,
           "need": {"features": [], "mean": [0.5] + [0.0] * 15, "scale": [0.01] + [1.0] * 15,
                    "coef": [1.0] + [0.0] * 15, "intercept": 0.0, "min": 0.5, "strong_logit": -3.0}}


class Gate(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="memgate"))
        self.file = self.tmp / "gate.json"
        self.file.write_text(json.dumps(FIXTURE))
        for p in (mock.patch.object(mem, "GATE_FILE", self.file), mock.patch.object(mem, "GATE", "calibrated"),
                  mock.patch.object(mem, "need_terms", lambda db, prompt: {"n_words": 1})):
            p.start()
            self.addCleanup(p.stop)
        mem._gate_cache[:] = [None, None]
        self.addCleanup(lambda: mem._gate_cache.__setitem__(slice(None), [None, None]))

    def ids(self, hs):
        return [h["id"] for h in mem.gate(None, "what is the garage code", hs)]

    def test_a_missing_gate_file_falls_back_to_the_raw_cut(self):
        self.file.unlink()
        self.assertEqual(self.ids(hits(2.0, -5.9, -6.1)), [0, 1])

    def test_another_version_falls_back_to_the_raw_cut(self):
        self.file.write_text(json.dumps({**FIXTURE, "version": 99}))
        self.assertEqual(self.ids(hits(2.0, -5.9, -6.1)), [0, 1])

    def test_raw_mode_is_the_old_cut(self):
        with mock.patch.object(mem, "GATE", "raw"):
            self.assertEqual(self.ids(hits(-5.9, -6.0, -6.1)), [0, 1])

    def test_a_prompt_that_wants_memory_gets_everything_above_the_cut(self):
        # top score 2.0 -> p1 0.88 -> need model says yes
        self.assertEqual(self.ids(hits(2.0, -5.0, -5.7)), [0, 1])

    def test_a_prompt_that_wants_no_memory_gets_only_strong_matches(self):
        # top score -2.5 -> p1 0.08 -> need model says no: only logits >= -3 pass
        self.assertEqual(self.ids(hits(-2.5, -3.5, -5.0)), [0])

    def test_a_prompt_about_himself_always_gets_its_matches(self):
        with mock.patch.object(mem, "need_terms", side_effect=AssertionError("no need model for first person")):
            got = [h["id"] for h in mem.gate(None, "when's my husband's birthday again", hits(-3.5, -5.0, -5.7))]
        self.assertEqual(got, [0, 1])

    def test_nothing_above_the_cut_skips_the_need_model(self):
        with mock.patch.object(mem, "need_terms", side_effect=AssertionError("no lookups when nothing passes")):
            self.assertEqual(self.ids(hits(-7.0, -8.0)), [])

    def test_the_shipped_gate_loads(self):
        with mock.patch.object(mem, "GATE_FILE", Path(mem.__file__).resolve().parent / "gate.json"):
            mem._gate_cache[:] = [None, None]
            g = mem.gate_params()
        self.assertIsNotNone(g)
        self.assertEqual(len(g["need"]["coef"]), len(mem.need_features("x", [0.1], None, {})))
        self.assertLessEqual(g["need"]["strong_logit"], 0)


class NeedTerms(unittest.TestCase):
    def test_features_are_finite_without_candidates(self):
        x = mem.need_features("hello there", [], None, {})
        self.assertEqual(len(x), 16)
        self.assertTrue(all(abs(v) < 1e6 for v in x))


if __name__ == "__main__":
    unittest.main()
