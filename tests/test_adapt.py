"""The miner, the adapter's bounds, protected-only-stricter, the state file. Synthetic sessions only, never real ones.

    python3 -m unittest -q test_adapt          (from this folder or memory/, wherever adapt.py sits)
"""
import _env  # noqa: F401  (first: the test data folder and user)
import json
import math
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lflabs_memory import adapt  # noqa: E402

PRIOR = {"a": 0.29, "b": -1.86, "H": [[1.08, -0.002], [-0.002, 0.047]], "lam": -7.4, "lam_lo": -10.4,
         "lam_hi": -3.9, "hist": adapt.hist_of([x / 10 for x in range(-150, 80)])}

PLUMBER = "A plumber is coming Thursday 2026-03-12 at 9:30 to fix the dripping bathroom tap. Gerald is sending him."
DENTIST = "Dentist appointment with Dr. Okafor moved to Friday 2026-04-17 at 14:00 at Bayview Clinic."
GYM = "Odile's climbing gym, Cliffhanger Westside, closes at 22:00 on weekdays."


def hook(turn, prompt, cands):
    return {"type": "hook", "turn": turn, "prompt": prompt, "cands": cands}


def cand(mid, score, injected, text="", prot=False):
    return {"id": mid, "score": score, "p": None, "injected": injected, "protected": prot, "text": text}


def labels_of(out):
    return sorted((it["label"], it["id"]) for it in out)


class Miner(unittest.TestCase):
    def test_used(self):
        ev = [{"type": "user", "turn": 1, "text": "when is the plumber coming"},
              hook(1, "when is the plumber coming", [cand(27, 8.3, True, PLUMBER)]),
              {"type": "reply", "turn": 1, "text": "Thursday the 12th at 9:30; Gerald is sending him."}]
        out = adapt.mine(ev)
        self.assertEqual(labels_of(out), [("used", 27)])
        self.assertEqual(out[0]["y"], 1)
        self.assertEqual(out[0]["score"], 8.3)

    def test_unused_is_a_weak_negative(self):
        ev = [hook(1, "what should I cook tonight", [cand(27, -5.0, True, PLUMBER)]),
              {"type": "reply", "turn": 1, "text": "How about a mushroom risotto?"},
              {"type": "reply", "turn": 2, "text": "Or pasta with pesto."}]
        out = adapt.mine(ev)
        self.assertEqual(labels_of(out), [("unused", 27)])
        self.assertEqual(out[0]["y"], 0)
        self.assertLess(out[0]["weight"], adapt.W_USED)

    def test_used_later_in_the_session_is_not_unused(self):
        ev = [hook(1, "anything on this week", [cand(27, 1.0, True, PLUMBER)]),
              {"type": "reply", "turn": 1, "text": "A few things."},
              {"type": "reply", "turn": 4, "text": "Right, Gerald's plumber comes on the 12th at 9:30."}]
        self.assertEqual(adapt.mine(ev), [])

    def test_the_prompts_own_words_do_not_count(self):
        prompt = "is Gerald sending the plumber on the 12th at 9:30"
        ev = [hook(1, prompt, [cand(27, 6.0, True, PLUMBER)]),
              {"type": "reply", "turn": 1, "text": "Yes, Gerald is sending the plumber on the 12th at 9:30."}]
        self.assertEqual(labels_of(adapt.mine(ev)), [("unused", 27)])

    def test_corrected_by_forget(self):
        ev = [hook(1, "when is the dentist", [cand(5, 4.0, True, DENTIST)]),
              {"type": "reply", "turn": 1, "text": "Friday the 17th at 14:00 with Dr. Okafor."},
              {"type": "user", "turn": 2, "text": "no, I cancelled that"},
              {"type": "forget", "turn": 2, "id": 5}]
        out = adapt.mine(ev)
        self.assertEqual(labels_of(out), [("corrected", 5)])
        self.assertEqual(out[0]["y"], 0)
        self.assertGreater(out[0]["weight"], adapt.W_USED)

    def test_corrected_by_the_owners_next_words(self):
        ev = [hook(1, "where do I climb", [cand(9, 3.0, True, GYM)]),
              {"type": "reply", "turn": 1, "text": "At Cliffhanger Westside."},
              {"type": "user", "turn": 2, "text": "that's outdated, Cliffhanger closed, I go to Boulder Barn now"}]
        self.assertEqual(labels_of(adapt.mine(ev)), [("corrected", 9)])

    def test_complaint_about_something_else_is_not_a_correction(self):
        ev = [hook(1, "where do I climb", [cand(9, 3.0, True, GYM)]),
              {"type": "reply", "turn": 1, "text": "At Cliffhanger Westside, open till 22:00."},
              {"type": "user", "turn": 2, "text": "the weather forecast was wrong again"}]
        self.assertEqual(labels_of(adapt.mine(ev)), [("used", 9)])

    def test_miss_by_recall_of_a_below_cut_candidate(self):
        ev = [hook(1, "when's my dentist thing", [cand(27, 2.0, True, PLUMBER), cand(5, -9.0, False)]),
              {"type": "recall", "turn": 1, "ids": [5], "texts": {5: DENTIST}},
              {"type": "reply", "turn": 1, "text": "Friday the 17th at 14:00, Dr. Okafor at Bayview Clinic."}]
        out = {it["id"]: it for it in adapt.mine(ev)}
        self.assertEqual(out[5]["label"], "miss")
        self.assertEqual(out[5]["y"], 1)
        self.assertEqual(out[5]["score"], -9.0)
        self.assertEqual(out[27]["label"], "unused")

    def test_recalled_but_not_used_is_no_label(self):
        ev = [hook(1, "when's my dentist thing", [cand(5, -9.0, False)]),
              {"type": "recall", "turn": 1, "ids": [5], "texts": {5: DENTIST}},
              {"type": "reply", "turn": 1, "text": "I couldn't find anything about that."}]
        self.assertEqual(adapt.mine(ev), [])

    def test_recall_full_counts_as_use(self):
        ev = [hook(1, "tell me everything about the dentist", [cand(5, -9.0, False)]),
              {"type": "recall_full", "turn": 1, "id": 5}]
        self.assertEqual(labels_of(adapt.mine(ev)), [("miss", 5)])

    def test_miss_of_a_memory_that_was_never_a_candidate(self):
        ev = [hook(1, "when's my dentist thing", [cand(27, 2.0, False)]),
              {"type": "recall", "turn": 1, "ids": [5], "texts": {5: DENTIST}},
              {"type": "reply", "turn": 1, "text": "Friday the 17th at 14:00 with Dr. Okafor at Bayview Clinic."}]
        out = adapt.mine(ev)
        self.assertEqual(labels_of(out), [("miss_unretrieved", 5)])
        self.assertIsNone(out[0]["score"])

    def test_miss_by_restatement(self):
        ev = [hook(1, "book the dentist", [cand(5, -8.0, False)]),
              {"type": "user", "turn": 2, "text": "it's Friday the 17th at 2pm with Dr. Okafor"},
              {"type": "remember", "turn": 2, "id": 5, "action": "updated", "via": "tool"},
              {"type": "remember", "turn": None, "id": 77, "action": "updated", "via": "extract"}]
        out = adapt.mine(ev)
        self.assertEqual(labels_of(out), [("miss", 5)])
        self.assertEqual(out[0]["weight"], adapt.W_MISS)

    def test_restatement_seen_only_by_extraction_weighs_less(self):
        ev = [hook(1, "book the dentist", [cand(5, -8.0, False)]),
              {"type": "remember", "turn": None, "id": 5, "action": "unchanged", "via": "extract"}]
        out = adapt.mine(ev)
        self.assertEqual(labels_of(out), [("miss", 5)])
        self.assertLess(out[0]["weight"], adapt.W_MISS)

    def test_a_new_memory_is_not_a_restatement(self):
        ev = [hook(1, "book the dentist", [cand(5, -8.0, False)]),
              {"type": "remember", "turn": 1, "id": 5, "action": "added", "via": "tool"}]
        self.assertEqual(adapt.mine(ev), [])

    def test_no_model_calls_and_no_io(self):
        import builtins
        import subprocess
        real_open, real_run = builtins.open, subprocess.run

        def refuse(*a, **k):
            raise AssertionError("the miner must not do I/O")
        builtins.open, subprocess.run = refuse, refuse
        try:
            adapt.mine([hook(1, "x y z", [cand(1, 0.0, True, PLUMBER)]), {"type": "reply", "turn": 1, "text": "ok"}])
        finally:
            builtins.open, subprocess.run = real_open, real_run


class Events(unittest.TestCase):
    def transcript(self):
        def a(*blocks):
            return {"type": "assistant", "message": {"content": list(blocks)}}

        def u(content, **k):
            return {"type": "user", "message": {"role": "user", "content": content}, **k}
        recall = json.dumps([{"id": 5, "content": DENTIST, "score": -1.0}])
        return [
            u("when's my dentist thing"),
            a({"type": "tool_use", "id": "t1", "name": "mcp__lflabs-memory__recall", "input": {"query": "dentist"}}),
            u([{"type": "tool_result", "tool_use_id": "t1", "content": [{"type": "text", "text": recall}]}]),
            a({"type": "text", "text": "Friday the 17th at 14:00, Dr. Okafor at Bayview Clinic."}),
            u("a background agent finished", origin={"kind": "task-notification"}),
            u("and the plumber"),
            a({"type": "tool_use", "id": "t2", "name": "mcp__lflabs-memory__forget", "input": {"id": 27}}),
            u([{"type": "tool_result", "tool_use_id": "t2", "content": "retired #27"}]),
            a({"type": "text", "text": "Done."}),
            {"type": "assistant", "isSidechain": True, "message": {"content": [{"type": "text", "text": "x"}]}},
        ]

    def test_session_events(self):
        hooks = [{"key": adapt.prompt_key("when's my dentist thing"),
                  "cands": [{"id": 27, "score": 2.0, "p": 0.1, "injected": True, "protected": False},
                            {"id": 5, "score": -9.0, "p": 0.01, "injected": False, "protected": True}]}]
        ev = adapt.session_events(self.transcript(), hooks, {27: PLUMBER, 5: DENTIST}.get, saves=[(5, "updated")])
        kinds = [(e["type"], e["turn"]) for e in ev]
        self.assertEqual(kinds, [("user", 1), ("hook", 1), ("recall", 1), ("reply", 1), ("user", 2), ("forget", 2),
                                 ("reply", 2), ("remember", None)])
        h = ev[1]
        self.assertEqual(h["cands"][0]["text"], PLUMBER)
        self.assertNotIn("text", h["cands"][1])  # a candidate below the cut: its text is never fetched
        out = {it["id"]: it["label"] for it in adapt.mine(ev)}
        self.assertEqual(out, {5: "miss", 27: "corrected"})

    def test_prompt_key_is_not_the_prompt(self):
        k = adapt.prompt_key("my garage code is 4412")
        self.assertEqual(len(k), 16)
        self.assertNotIn("4412", k)
        self.assertEqual(k, adapt.prompt_key("my  garage code is 4412 "))


def state(**cfg):
    return adapt.State(PRIOR, **cfg)


def items(label, scores, y):
    w = {"used": adapt.W_USED, "unused": adapt.W_UNUSED, "miss": adapt.W_MISS, "corrected": adapt.W_CORRECTED}[label]
    return [{"label": label, "score": s, "y": y, "weight": w} for s in scores]


class Bounds(unittest.TestCase):
    def test_shipped_default_is_aci_on_the_calibrated_anchor(self):
        st = state()
        self.assertEqual((st.cfg["anchor"], st.cfg["controller"]), ("calib", "aci"))

    def test_starts_at_the_static_cut(self):
        for anchor in ("quantile", "calib"):
            self.assertAlmostEqual(state(anchor=anchor).cut(), PRIOR["lam"], delta=0.05)

    def test_a_user_who_never_uses_memories(self):
        for anchor, ctl in (("quantile", "aci"), ("calib", "aci"), ("calib", "recal"), ("calib", "balance")):
            st = state(anchor=anchor, controller=ctl)
            rng = random.Random(1)
            for _ in range(3000):
                sc = [rng.uniform(-12, 8) for _ in range(8)]
                st.update(items("unused", sc[:4], 0), hooks=1, missed_turns=0, scores=sc)
            self.assertLessEqual(st.theta, st.hi + 1e-9, (anchor, ctl))
            self.assertGreaterEqual(st.theta, st.lo - 1e-9)
            a0, b0 = PRIOR["a"], PRIOR["b"]
            self.assertTrue(a0 * 0.5 - 1e-9 <= st.a <= a0 * 2 + 1e-9 and abs(st.b - b0) <= 3 + 1e-9, (st.a, st.b))
            if anchor == "calib":  # the cut stays within the bench's cross-validated range, recalibrated or not
                self.assertTrue(PRIOR["lam_lo"] <= st.cut() <= PRIOR["lam_hi"], (ctl, st.cut()))

    def test_a_user_who_wants_everything(self):
        st = state(anchor="calib", controller="recal")
        for _ in range(3000):
            st.update(items("used", [-10.0, -12.0], 1), hooks=1, missed_turns=1, scores=[-10.0, -12.0])
        self.assertGreaterEqual(st.cut(), PRIOR["lam_lo"])
        for anchor in ("quantile", "calib"):
            st = state(anchor=anchor, controller="aci")
            for _ in range(3000):
                st.update(items("used", [5.0], 1) + items("miss", [-11.0], 1), hooks=1, missed_turns=1,
                          scores=[5.0, -11.0])
            self.assertAlmostEqual(st.theta, st.lo, places=6)
            if anchor == "calib":
                self.assertAlmostEqual(st.cut(), PRIOR["lam_lo"], places=6)

    def test_junk_guard_blocks_loosening(self):
        st = state(controller="aci")
        st.junk = 0.99
        t = st.theta
        st.update([], hooks=1, missed_turns=1, scores=[0.0])
        self.assertEqual(st.theta, t)
        self.assertEqual(st.counts["loosen_blocked"], 1)


class ProtectedOnlyStricter(unittest.TestCase):
    def test_protected_cut_never_falls(self):
        rng = random.Random(7)
        for anchor in ("quantile", "calib"):
            st = state(anchor=anchor, controller="aci")
            prev = st.theta_prot
            for _ in range(2000):
                missed = rng.random() < 0.3
                st.update([], hooks=1, missed_turns=int(missed), scores=[rng.uniform(-12, 8)])
                self.assertGreaterEqual(st.theta_prot, prev - 1e-12)
                self.assertGreaterEqual(st.theta_prot, st.theta - 1e-12)
                prev = st.theta_prot

    def test_pick_never_adds_a_protected_memory(self):
        rng = random.Random(3)
        for anchor in ("quantile", "calib"):
            st = state(anchor=anchor, controller="aci")
            for _ in range(500):  # drive the general cut as lenient as it goes
                st.update([], hooks=1, missed_turns=1, scores=[rng.uniform(-12, 8)])
            for _ in range(300):
                lam = PRIOR["lam"]
                cands = [{"score": rng.uniform(-14, 8), "protected": rng.random() < 0.5} for _ in range(8)]
                for c in cands:
                    c["static"] = c["score"] >= lam
                for i in adapt.pick(cands, st):
                    if cands[i]["protected"]:
                        self.assertTrue(cands[i]["static"])
            self.assertLess(st.cut(), PRIOR["lam"])  # the general cut did loosen

    def test_what_is_protected(self):
        self.assertTrue(adapt.protected("My salary went up to 120k"))
        self.assertTrue(adapt.protected("Dentist at Bayview Clinic on Friday"))
        self.assertTrue(adapt.protected("Leo said, by text on 2026-04-17: Tallow is great", section="texts"))
        self.assertTrue(adapt.protected("The plan is X", origin="outside"))
        self.assertTrue(adapt.protected("Standup notes", section="people", section2="meetings"))
        self.assertFalse(adapt.protected("Odile prefers aisle seats", section="prefs", origin="owner_typed"))


class StateFile(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="adapt"))
        self.path = self.dir / "adapt-state.json"

    def test_round_trip(self):
        st = state(controller="aci")
        for _ in range(50):
            st.update(items("unused", [1.0], 0), hooks=1, missed_turns=0, scores=[1.0, -3.0])
        adapt.save(self.path, st)
        self.assertEqual(oct(os.stat(self.path).st_mode & 0o777), "0o600")
        st2 = adapt.load(self.path, PRIOR, controller="aci")
        self.assertAlmostEqual(st2.theta, st.theta)
        self.assertAlmostEqual(st2.cut(), st.cut(), places=3)  # the histogram is saved rounded
        self.assertEqual(st2.counts["unused"], 50)

    def test_bad_files_mean_the_prior(self):
        st = state()
        cases = ["not json", json.dumps({"version": 99}), json.dumps({**st.to_json(), "theta": float("nan")}),
                 json.dumps({**st.to_json(), "prior": {**PRIOR, "lam": -1.0}}),
                 json.dumps({**st.to_json(), "hist": [1.0, 2.0]})]
        for text in cases:
            self.path.write_text(text.replace("NaN", "NaN"))
            st2 = adapt.load(self.path, PRIOR)
            self.assertEqual(st2.theta, st2.theta0, text[:40])
            self.assertEqual(st2.counts["sessions"], 0)

    def test_out_of_bounds_values_are_clamped_on_load(self):
        st = state()
        d = st.to_json()
        d["theta"], d["a"] = st.lo - 50, 100.0
        self.path.write_text(json.dumps(d))
        st2 = adapt.load(self.path, PRIOR)
        self.assertEqual(st2.theta, st2.lo)
        self.assertLessEqual(st2.a, PRIOR["a"] * 2)

    def test_learn_session_end_to_end(self):
        st = state(controller="aci")
        ev = [hook(1, "when's my dentist thing", [cand(27, 2.0, True, PLUMBER), cand(5, -9.0, False)]),
              {"type": "recall", "turn": 1, "ids": [5], "texts": {5: DENTIST}},
              {"type": "reply", "turn": 1, "text": "Friday the 17th at 14:00, Dr. Okafor at Bayview Clinic."}]
        t = st.theta
        out = adapt.learn_session(st, ev)
        self.assertEqual({it["label"] for it in out}, {"miss", "unused"})
        self.assertLess(st.theta, t)  # a detected miss loosens the cut
        self.assertEqual(st.counts["hooks"], 1)
        self.assertTrue(math.isfinite(st.cut()))


if __name__ == "__main__":
    unittest.main()
