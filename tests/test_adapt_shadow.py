"""Learning from use in mem.py (ADAPT_MODE) on the calibrated gate: shadow never changes what is injected, "on" never
adds a protected memory, the hook log holds no text, the needs-memory model runs once, the worker learns from a quiet
session, adapt-reset and adapt-report. Synthetic only.

  .venv/bin/python -m unittest -q test_adapt_shadow
"""
import _env  # noqa: F401  (first: the test data folder and user)
import collections
import json
import os
from unittest import mock

from lflabs_memory import adapt
from lflabs_memory import mem
from test_redact import DBCase
from _fakes import fake_rerank

GATE = {"version": 1, "platt": {"a": 0.3, "b": -1.9}, "min_logit": -6.0}
PRIOR = {"a": 0.3, "b": -1.9, "H": [[1.0, 0.0], [0.0, 0.05]], "lam": -6.0, "lam_lo": -10.0, "lam_hi": -3.0,
         "hist": adapt.hist_of([x / 4 for x in range(-56, 32)]), "m0": 0.03, "u0": 0.66, "fpr": 0.3, "fnr": 0.25}
PLUMBER = "A plumber is coming Thursday 2026-03-12 at 9:30 to fix the dripping tap; Gerald is sending him."
DENTIST = "Dentist appointment with Okafor moved to Friday 2026-04-17 at 14:00 near Bayview."
SALARY = "Robin's salary review: the raise to 140k starts in May, paid through the new payroll."
PROMPT = "what is coming up this week with the plumber dentist salary"


class Shadow(DBCase):
    def setUp(self):
        super().setUp()
        for p in (mock.patch.object(mem, "rerank", fake_rerank({"plumber": 5.0, "Dentist": -7.0, "salary": -7.0})),
                  mock.patch.object(mem, "MEETBRIEF_STATE", self.tmp / "nope.json"),
                  mock.patch.object(mem, "VOICE_SESSION", self.tmp / "none.json"),
                  mock.patch.object(mem, "TELEGRAM_SESSION", self.tmp / "none.json"),
                  mock.patch.object(mem, "ADAPT_STATE", self.tmp / "adapt-state.json"),
                  mock.patch.object(mem, "ADAPT_PRIOR", self.tmp / "adapt-prior.json"),
                  mock.patch.object(mem, "ADAPT_MODE", "shadow"),
                  mock.patch.object(mem, "GATE", "calibrated"),
                  mock.patch.object(mem, "GATE_FILE", self.tmp / "gate.json"),
                  mock.patch.dict(mem._adapt, {"state": None, "queue": None, "mtime": None, "saves": {}}),
                  mock.patch.dict(os.environ, {"LFLABS_MEMORY_SCOPE": ""})):
            p.start()
            self.addCleanup(p.stop)
        self.ids = {name: mem.remember(self.db, text, "project", "lantern", origin="owner_typed")[0]
                    for name, text in (("plumber", PLUMBER), ("dentist", DENTIST), ("salary", SALARY))}
        (self.tmp / "adapt-prior.json").write_text(json.dumps(PRIOR))
        (self.tmp / "gate.json").write_text(json.dumps(GATE))
        mem._gate_cache[:] = [None, None]
        self.addCleanup(lambda: mem._gate_cache.__setitem__(slice(None), [None, None]))

    def lenient(self):
        st = mem.adapt_init(self.db)
        st.theta = st.lo  # as lenient as the bounds allow: the adapted gate wants more than the live one
        return st

    def test_shadow_never_changes_what_is_injected(self):
        live = mem.prompt_context(self.db, PROMPT, "s-live")
        self.assertIn(f"#{self.ids['plumber']}", live)
        self.assertNotIn(f"#{self.ids['dentist']}", live)
        self.lenient()
        shadow = mem.prompt_context(self.db, PROMPT, "s-shadow")
        self.assertEqual(live.replace("s-live", ""), shadow.replace("s-shadow", ""))
        (sid, at, key, cands), = list(mem._adapt["queue"])
        rows = {c["id"]: c for c in json.loads(cands)}
        self.assertTrue(rows[self.ids["plumber"]]["injected"] and rows[self.ids["plumber"]]["adapted"])
        self.assertTrue(rows[self.ids["dentist"]]["adapted"])          # what the adapted gate would add
        self.assertFalse(rows[self.ids["dentist"]]["injected"])
        self.assertTrue(rows[self.ids["salary"]]["protected"])
        self.assertFalse(rows[self.ids["salary"]]["adapted"])          # protected: never more than the live gate
        self.assertEqual(key, adapt.prompt_key(PROMPT))
        for text in (PLUMBER, DENTIST, SALARY, PROMPT):  # the log holds ids, scores and flags, never words
            self.assertNotIn(text[:20], cands)

    def test_on_injects_the_adapted_list_but_no_protected_memory(self):
        self.lenient()
        with mock.patch.object(mem, "ADAPT_MODE", "on"):
            out = mem.prompt_context(self.db, PROMPT, "s-on")
        self.assertIn(f"#{self.ids['dentist']}", out)
        self.assertNotIn(f"#{self.ids['salary']}", out)

    def test_off_and_outside_the_daemon_nothing_is_read_or_logged(self):
        with mock.patch.object(mem, "_adapt_load", side_effect=AssertionError("state read")):
            mem.prompt_context(self.db, PROMPT, "s1")  # no adapt_init: a bench, the CLI, the in-process fallback
        with mock.patch.object(mem, "ADAPT_MODE", "off"):
            self.assertIsNone(mem.adapt_init(self.db))
        self.assertIsNone(mem._adapt["queue"])
        self.assertFalse((self.tmp / "adapt-state.json").exists())

    def test_no_prior_means_no_adaptation(self):
        (self.tmp / "adapt-prior.json").unlink()
        self.assertIsNone(mem.adapt_init(self.db))

    def test_a_prior_for_another_gate_means_no_adaptation(self):
        (self.tmp / "gate.json").write_text(json.dumps({**GATE, "min_logit": -5.0}))
        mem._gate_cache[:] = [None, None]
        self.assertIsNone(mem.adapt_init(self.db))
        with mock.patch.object(mem, "GATE", "raw"):
            self.assertIsNone(mem.adapt_init(self.db))

    def test_the_needs_memory_model_runs_once_and_binds_both_gates(self):
        need = {"features": [], "mean": [0.0] * 16, "scale": [1.0] * 16, "coef": [0.0] * 16, "intercept": -5.0,
                "min": 0.5, "strong_logit": -3.0}  # always "this prompt wants no memory": only strong matches
        (self.tmp / "gate.json").write_text(json.dumps({**GATE, "need": need}))
        mem._gate_cache[:] = [None, None]
        self.lenient()
        calls = []
        real = mem.need_terms
        with mock.patch.object(mem, "need_terms", lambda db, p: calls.append(p) or real(db, p)):
            out = mem.prompt_context(self.db, PROMPT, "s-need")
        self.assertEqual(len(calls), 1)
        self.assertIn(f"#{self.ids['plumber']}", out)  # 5.0: strong
        rows = {c["id"]: c for c in json.loads(mem._adapt["queue"][0][3])}
        self.assertFalse(rows[self.ids["dentist"]]["adapted"])  # -7 clears the adapted cut, not the strong one

    def transcript(self, sid):
        path = self.tmp / f"{sid}.jsonl"
        lines = [{"type": "user", "message": {"role": "user", "content": PROMPT}},
                 {"type": "assistant", "message": {"content": [{"type": "text", "text": "Gerald's plumber comes "
                                                                "Thursday the 12th at 9:30."}]}}]
        path.write_text("".join(json.dumps(x) + "\n" for x in lines))
        with self.db:
            self.db.execute("insert into sessions(session_id, transcript, cwd, activity, final) values (?,?,?,?,1)",
                            (sid, str(path), "/tmp", 0.0))

    def test_the_worker_learns_from_a_quiet_session(self):
        st = mem.adapt_init(self.db)
        mem.prompt_context(self.db, PROMPT, "s-learn")
        self.transcript("s-learn")
        mem._adapt_tick(self.db, lambda m: None)
        self.assertEqual(st.counts["sessions"], 1)
        self.assertEqual(st.counts["used"], 1)  # the plumber memory, named in the reply
        self.assertTrue((self.tmp / "adapt-state.json").exists())
        self.assertEqual(self.db.execute("select count(*), sum(mined) from adapt_hooks").fetchone()[:], (1, 1))
        mem._adapt_tick(self.db, lambda m: None)  # mined once
        self.assertEqual(st.counts["sessions"], 1)

    def test_a_session_still_going_waits(self):
        st = mem.adapt_init(self.db)
        mem.prompt_context(self.db, PROMPT, "s-busy")
        mem._adapt_tick(self.db, lambda m: None)
        self.assertEqual(st.counts["sessions"], 0)
        self.assertEqual(self.db.execute("select sum(mined) from adapt_hooks").fetchone()[0], 0)

    def test_reset_and_report(self):
        st = self.lenient()
        mem.prompt_context(self.db, PROMPT, "s-r")
        self.transcript("s-r")
        mem._adapt_tick(self.db, lambda m: None)
        self.assertLess(mem._adapt_load(PRIOR).cut(), -6.0 + 1e-9)
        report = mem.adapt_report(self.db)
        self.assertIn("hook decisions logged 1", report)
        self.assertIn("protected: live 0, adapted 0", report)
        for text in (PLUMBER, DENTIST, SALARY, PROMPT):
            self.assertNotIn(text[:20], report)
        for i in self.ids.values():
            self.assertNotIn(f"#{i}", report)
        self.assertIn("reset", mem.adapt_reset())
        fresh = mem._adapt_load(PRIOR)
        self.assertAlmostEqual(fresh.cut(), PRIOR["lam"], places=6)
        self.assertEqual(fresh.counts["sessions"], 0)
        mem._adapt_tick(self.db, lambda m: None)  # the daemon picks the reset file up
        self.assertAlmostEqual(mem._adapt["state"].cut(), PRIOR["lam"], places=6)
        del st

    def test_cli(self):
        import contextlib
        import io
        out = io.StringIO()
        with mock.patch.object(mem, "connect", lambda: self.db), mock.patch("sys.argv", ["mem.py", "adapt-reset"]), \
                contextlib.redirect_stdout(out):
            mem.main()
        self.assertIn("reset to the bench prior", out.getvalue())


if __name__ == "__main__":
    import unittest
    unittest.main()
