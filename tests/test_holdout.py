"""The live holdout (HOLDOUT_MODE): off changes nothing; a session's arm is stable and about HOLDOUT_RATE of sessions are
held out; a held-out session gets no memories, claims or look-again line from the hook, and everything else (the
tools, saving, forgetting) is the same; the outcome signals on fixture transcripts; the report holds counts only.
Synthetic only: temp databases, a fake embedder, nothing of Robin's.

  .venv/bin/python -m unittest -q test_holdout
"""
import _env  # noqa: F401  (first: the test data folder and user)
import io
import json
import os
import socket
import stat
import time
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from lflabs_memory import mem
from test_redact import DBCase
from _fakes import fake_rerank

GARAGE = "The garage keypad code changed in March to the new four digit one"
TEA = "Robin drinks oolong tea every afternoon at the desk"
PROMPT = "what did we decide about the garage keypad code"


def iso(t):
    return datetime.fromtimestamp(t, timezone.utc).isoformat().replace("+00:00", "Z")


def user(text, t):
    return {"type": "user", "timestamp": iso(t), "message": {"role": "user", "content": text}}


def reply(text, t, tools=()):
    blocks = ([{"type": "text", "text": text}] if text else []) + [
        {"type": "tool_use", "id": f"tu{i}", "name": n, "input": {"query": "x"}} for i, n in enumerate(tools)]
    return {"type": "assistant", "timestamp": iso(t), "message": {"role": "assistant", "content": blocks}}


def write(path, entries, mode="w"):
    with open(path, mode) as f:
        f.write("".join(json.dumps(e) + "\n" for e in entries))
    return path


class Case(DBCase):
    def setUp(self):
        super().setUp()
        for p in (mock.patch.object(mem, "rerank", fake_rerank({"garage": 5.0, "oolong": 5.0})),
                  mock.patch.object(mem, "gate", lambda db, prompt, hits, scope=None, memo=None, **k: hits),
                  mock.patch.object(mem, "MEETBRIEF_STATE", self.tmp / "nope.json"),
                  mock.patch.object(mem, "VOICE_SESSION", self.tmp / "voice-session.json"),
                  mock.patch.object(mem, "TELEGRAM_SESSION", self.tmp / "telegram-session.json"),
                  mock.patch.object(mem, "SOCK", self.tmp / "mem.sock"),  # never there: the hook works in-process
                  mock.patch.object(mem, "HOLDOUT_MODE", "on"),
                  mock.patch.object(mem, "HOLDOUT_RATE", 0.10),
                  mock.patch.dict(mem._holdout, {"live": True, "salt": None, "arms": {}}),
                  mock.patch.dict(mem._adapt, {"state": None, "queue": None, "mtime": None, "saves": {}}),
                  mock.patch.dict(os.environ, {"LFLABS_MEMORY_SCOPE": ""})):
            p.start()
            self.addCleanup(p.stop)
        self.garage, _ = mem.remember(self.db, GARAGE, "project", "lantern", origin="owner_typed")

    def held(self, rate=1.0):
        """Every new session held out (rate 1) or none (rate 0)."""
        p = mock.patch.object(mem, "HOLDOUT_RATE", rate)
        p.start()
        self.addCleanup(p.stop)

    def ctx(self, prompt=PROMPT, sid="s1", live=True):
        with mock.patch.dict(mem._holdout, {"live": live}):  # live: as the daemon or the hook calls it
            return mem.prompt_context(self.db, prompt, sid)

    def table(self):
        return self.db.execute("select name from sqlite_master where name = 'holdout'").fetchone()


class Off(Case):
    def test_off_changes_nothing(self):
        with mock.patch.object(mem, "HOLDOUT_MODE", "off"), mock.patch.object(mem, "HOLDOUT_RATE", 1.0):
            a = self.ctx(sid="s1", live=True)
            b = self.ctx(sid="s2", live=False)
            self.assertIsNone(mem.holdout_arm(self.db, "s3"))
        self.assertIn(f"#{self.garage}", a)
        self.assertEqual(a, b)
        self.assertFalse((self.tmp / "holdout-salt").exists())  # nothing created
        self.assertIsNone(self.table())

    def test_off_reads_nothing_more_when_saving(self):
        path = write(self.tmp / "t.jsonl", [user("my bike is the blue one in the shed", time.time())])
        mem.hook_activity({"session_id": "s1", "transcript_path": str(path), "cwd": str(self.tmp)}, final=True)
        with mock.patch.object(mem, "HOLDOUT_MODE", "off"), \
                mock.patch.object(mem, "_holdout_note", side_effect=AssertionError("holdout read")), \
                mock.patch.object(mem, "llm_json", lambda *a, **k: {"memories": []}):
            self.assertEqual(mem.pending_sessions(self.db), ["s1"])
            self.assertEqual(mem.extract(self.db, "s1"), [])
        self.assertIsNone(self.table())

    def test_a_bench_or_the_cli_never_holds_out(self):
        self.held(1.0)
        self.assertIn(f"#{self.garage}", self.ctx(live=False))
        self.assertIsNone(self.table())


class Assignment(Case):
    def test_stable_and_about_the_rate(self):
        ids = [f"session-{i}" for i in range(4000)]
        arms = [mem.holdout_arm(self.db, s) for s in ids]
        share = arms.count("held") / len(arms)
        self.assertAlmostEqual(share, 0.10, delta=0.02)
        self.assertEqual(set(arms), {"held", "memory"})
        mem._holdout.update(salt=None, arms={})  # a new process: salt from its file, arms from the table
        self.assertEqual([mem.holdout_arm(self.db, s) for s in ids[:300]], arms[:300])
        with mock.patch.object(mem, "HOLDOUT_RATE", 0.9):  # a later rate never moves a session already seen
            mem._holdout["arms"].clear()
            self.assertEqual([mem.holdout_arm(self.db, s) for s in ids[:300]], arms[:300])

    def test_salt_is_private_and_kept(self):
        mem.holdout_arm(self.db, "s1")
        f = self.tmp / "holdout-salt"
        self.assertEqual(stat.S_IMODE(f.stat().st_mode), 0o600)
        salt = f.read_text()
        mem._holdout.update(salt=None, arms={})
        mem.holdout_arm(self.db, "s2")
        self.assertEqual(f.read_text(), salt)
        self.assertEqual(sorted(p.name for p in self.tmp.iterdir() if p.name.startswith("holdout")), ["holdout-salt"])

    def test_the_table_keeps_a_hash_never_the_id(self):
        mem.holdout_arm(self.db, "session-abc-123")
        (sid, arm, scope), = self.db.execute("select sid, arm, scope from holdout").fetchall()
        self.assertNotIn("session-abc-123", sid)  # (a random salt's hex can hold "abc": about 1 run in 140)
        self.assertEqual(sid, mem._holdout_key("session-abc-123"))
        self.assertEqual(len(sid), 32)
        self.assertIn(arm, ("held", "memory"))
        self.assertEqual(scope, "desk")

    def test_no_session_id_is_never_held_out(self):
        self.held(1.0)
        self.assertIsNone(mem.holdout_arm(self.db, None))
        self.assertIn(f"#{self.garage}", self.ctx(sid=None))
        self.assertIsNone(self.table())

    def test_a_scope_left_out_is_never_held_out(self):
        self.held(1.0)
        (self.tmp / "voice-session.json").write_text(json.dumps({"id": "v1"}))
        with mock.patch.object(mem, "HOLDOUT_SCOPES", ("desk", "telegram")):
            self.assertIn(f"#{self.garage}", self.ctx(sid="v1"))
            self.assertIsNone(mem.holdout_arm(self.db, "v1", "voice"))
        self.assertEqual(mem.holdout_arm(self.db, "v1", "voice"), "held")  # by default every scope is in


class Held(Case):
    def test_held_session_gets_nothing_even_when_memories_match(self):
        self.held(1.0)
        self.assertIsNone(self.ctx())
        self.assertIsNone(self.db.execute("select 1 from injections").fetchone())  # nothing counts as shown
        self.assertEqual(self.db.execute("select uses from memories where id = ?", (self.garage,)).fetchone()[0], 0)

    def test_held_session_gets_no_look_again_or_claims_lines(self):
        claims = mock.Mock(return_value=["- #c1 [claim] the garage code is new"])
        with mock.patch.object(mem, "_claims_context", claims), mock.patch.object(mem, "_wants_memory", lambda *a: True), \
                mock.patch.object(mem, "gate", lambda *a, **k: []):
            self.held(0.0)
            self.assertEqual(self.ctx(sid="mem1"), mem.CONTEXT_HEAD + "- #c1 [claim] the garage code is new")
            with mock.patch.object(mem, "_claims_context", lambda *a: []):
                self.assertEqual(self.ctx(sid="mem2"), mem.LOOK_AGAIN_LINE)
            self.held(1.0)
            claims.reset_mock()
            self.assertIsNone(self.ctx(sid="held1"))
            claims.assert_not_called()
            with mock.patch.object(mem, "_claims_context", lambda *a: []):
                self.assertIsNone(self.ctx(sid="held2"))

    def test_held_session_keeps_only_the_save_nudge(self):
        self.held(1.0)
        self.assertEqual(self.ctx("remember that the garage keypad code changed in March"), mem.REMEMBER_NUDGE)

    def test_not_held_session_is_unchanged(self):
        self.held(0.0)
        on = self.ctx(sid="a")
        with mock.patch.object(mem, "HOLDOUT_MODE", "off"):
            off = self.ctx(sid="b")
        self.assertIn(f"#{self.garage}", on)
        self.assertEqual(on, off)
        self.assertEqual(mem.holdout_arm(self.db, "a"), "memory")

    def test_the_hook_holds_out_in_process(self):
        self.held(1.0)
        mem._holdout["live"] = False  # a fresh hook process: hook_prompt itself turns it on
        out = io.StringIO()
        with redirect_stdout(out):
            mem.hook_prompt({"prompt": PROMPT, "session_id": "s1", "cwd": str(self.tmp)})
        self.assertEqual(out.getvalue(), "")
        self.held(0.0)
        with redirect_stdout(out):
            mem.hook_prompt({"prompt": PROMPT, "session_id": "s2", "cwd": str(self.tmp)})
        self.assertIn(f"#{self.garage}", json.loads(out.getvalue())["hookSpecificOutput"]["additionalContext"])

    def test_the_daemon_holds_out(self):
        self.held(1.0)
        Handler = mem._handler(self.db)
        for sid, held in (("s1", True), ("s2", False)):
            if not held:
                self.held(0.0)
            a, b = socket.socketpair()
            with a, b:
                b.sendall((json.dumps({"prompt": PROMPT, "session_id": sid}) + "\n").encode())
                Handler(a, None, None)
                ctx = json.loads(b.makefile().readline())["context"]
            self.assertIsNone(ctx) if held else self.assertIn(f"#{self.garage}", ctx)

    def test_tools_and_forgetting_are_unaffected(self):
        self.held(1.0)
        self.assertIsNone(self.ctx())
        ids = lambda q: [h["id"] for h in json.loads(mem.recall_text(self.db, q).split("\n(", 1)[0])]  # noqa: E731
        self.assertIn(self.garage, ids("garage keypad code"))
        tea, action = mem.remember(self.db, TEA)
        self.assertEqual(action, "added")
        self.assertIn(tea, ids("oolong tea afternoon"))
        self.assertTrue(mem.forget_reply(self.db, tea).startswith(f"Deleted #{tea} for good"))
        self.assertNotIn(tea, ids("oolong tea afternoon"))

    def test_saving_is_unaffected(self):
        self.held(1.0)
        self.assertIsNone(self.ctx(sid="s1"))
        path = write(self.tmp / "t.jsonl", [user("I drink oolong tea every afternoon at the desk", time.time())])
        mem.hook_activity({"session_id": "s1", "transcript_path": str(path), "cwd": str(self.tmp)}, final=True)
        out = {"memories": [{"content": TEA, "kind": "user", "project": None, "from": [1]}]}
        with mock.patch.object(mem, "llm_json", lambda *a, **k: out):
            self.assertEqual(mem.pending_sessions(self.db), ["s1"])
            (mid, action, _), = mem.extract(self.db, "s1")
        self.assertEqual(action, "added")
        self.assertEqual(self.db.execute("select status from memories where id = ?", (mid,)).fetchone()[0], "active")
        self.assertEqual(self.db.execute("select turns from holdout").fetchone()[0], 1)  # and it was counted

    def test_a_failed_count_never_stops_a_save(self):
        self.held(1.0)
        self.ctx(sid="s1")
        path = write(self.tmp / "t.jsonl", [user("I drink oolong tea every afternoon at the desk", time.time())])
        mem.hook_activity({"session_id": "s1", "transcript_path": str(path), "cwd": str(self.tmp)}, final=True)
        out = {"memories": [{"content": TEA, "kind": "user", "project": None, "from": [1]}]}
        with mock.patch.object(mem, "llm_json", lambda *a, **k: out), \
                mock.patch.object(mem, "_reexplained", side_effect=RuntimeError("boom")), \
                redirect_stdout(io.StringIO()), mock.patch("sys.stderr", io.StringIO()):
            self.assertEqual(len(mem.extract(self.db, "s1")), 1)


class Signals(Case):
    def session(self, entries, sid="s1"):
        mem.holdout_arm(self.db, sid)
        path = write(self.tmp / f"{sid}.jsonl", entries)
        mem.hook_activity({"session_id": sid, "transcript_path": str(path), "cwd": str(self.tmp)}, final=True)
        return path

    def row(self, sid="s1"):
        return mem._holdout_row(self.db, sid)

    def extract(self, sid="s1"):
        with mock.patch.object(mem, "llm_json", lambda *a, **k: {"memories": []}):
            return mem.extract(self.db, sid)

    def test_each_signal_on_a_fixture_transcript(self):
        now = time.time()
        self.session([
            user(GARAGE, now + 60),                                       # he restates a memory saved before: 1
            reply("Noted.", now + 70),
            user(GARAGE, now - 400 * 86400),                               # the same, but before it was saved: 0
            user("No, I meant the side door, I already told you that", now + 120),  # a correction
            reply("I don't know which side door you mean.", now + 130),  # a don't-know
            reply("", now + 140, tools=["mcp__lflabs-memory__recall", "mcp__lflabs-memory__recall_full"]),
            {"type": "user", "timestamp": iso(now + 141), "message": {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu0", "content": "[]"}]}},  # a tool result: not a turn
            reply("I'm not sure who you mean by Sam.", now + 150),     # a second don't-know
            user("what is the garage keypad code", now + 160),            # a question: never re-explaining
            user("thanks, that's all", now + 1860),
        ])
        self.extract()
        r = self.row()
        self.assertEqual((r["turns"], r["owner_lines"], r["reexplained"], r["corrections"], r["dont_know"], r["recalls"]),
                         (5, 5, 1, 1, 2, 1))
        self.assertAlmostEqual(r["last_ts"] - r["first_ts"], 400 * 86400 + 1860, delta=1)

    def test_a_memory_saved_from_his_own_line_is_not_re_explaining(self):
        now = time.time()
        self.session([user(TEA, now + 5)])
        out = {"memories": [{"content": TEA, "kind": "user", "project": None, "from": [1]}]}
        with mock.patch.object(mem, "llm_json", lambda *a, **k: out):
            mem.extract(self.db, "s1")
        self.assertEqual(self.row()["reexplained"], 0)
        self.assertEqual(self.row()["owner_lines"], 1)

    def test_a_retired_or_hidden_memory_does_not_count(self):
        now = time.time()
        mem.forget(self.db, self.garage)
        self.session([user(GARAGE, now + 60)])
        self.extract()
        self.assertEqual(self.row()["reexplained"], 0)

    def test_corrections_and_dont_know_rules(self):
        for t in ("No, the other one", "nope.", "Not that one, the red file", "I meant tuesday",
                  "I already told you this", "that's wrong", "That’s not right", "ok. no, stop", "you forgot the tests"):
            self.assertTrue(mem.CORRECTION_RE.search(t.replace("’", "'")), t)
        for t in ("no em dashes in anything you write", "it's not that hard", "notice the noon meeting",
                  "I know what you meant to say", "add a note"):
            self.assertFalse(mem.CORRECTION_RE.search(t), t)
        for t in ("I don't know who Sam is.", "I do not remember that.", "I couldn't find anything about it",
                  "There's no record of that in memory.", "I don't have any information about your dentist"):
            self.assertTrue(mem.DONT_KNOW_RE.search(t), t)
        for t in ("Let me know if that works.", "I don't think so", "Done: tests pass."):
            self.assertFalse(mem.DONT_KNOW_RE.search(t), t)
        self.assertTrue(mem._question("what's the label?") and mem._question("Where does Mia live"))
        self.assertFalse(mem._question("Do not use em dashes"))

    def test_each_line_counts_once(self):
        now = time.time()
        path = self.session([user(GARAGE, now + 60), reply("I don't know.", now + 61)])
        stats = []
        mem._numbered_slice(path, 0, stats=stats)
        self.assertTrue(mem.holdout_tally(self.db, "s1", stats))
        self.assertFalse(mem.holdout_tally(self.db, "s1", stats))  # the same slice read again (a failed save)
        write(path, [user("and the shed lock too", now + 120)], "a")
        stats = []
        mem._numbered_slice(path, 0, stats=stats)  # the whole file: only the new line is counted
        mem.holdout_tally(self.db, "s1", stats)
        r = self.row()
        self.assertEqual((r["turns"], r["reexplained"], r["dont_know"]), (2, 1, 1))

    def test_tool_chatter_after_a_save_is_counted(self):
        now = time.time() - 3600
        path = self.session([user("look up the shed lock code please", now)])
        self.extract()
        write(path, [reply("", now + 5, tools=["mcp__lflabs-memory__recall"]), reply("I don't know it.", now + 9)], "a")
        with self.db:
            self.db.execute("update sessions set activity = 0")  # gone quiet
        self.assertEqual(mem.pending_sessions(self.db), [])  # no turn of his: skipped past, but counted first
        r = self.row()
        self.assertEqual((r["turns"], r["recalls"], r["dont_know"]), (1, 1, 1))

    def test_a_session_not_in_the_holdout_is_not_counted(self):
        with mock.patch.object(mem, "HOLDOUT_SCOPES", ()):
            self.session([user(GARAGE, time.time() + 60)])
        self.extract()
        self.assertIsNone(self.row())


class Report(Case):
    PRIVATE = "my locker combination is the one from Sam's birthday"

    def test_empty(self):
        self.assertIn("no sessions recorded yet", mem.holdout_report(self.db))

    def test_too_few_sessions_says_so_and_quotes_no_text(self):
        now = time.time()
        for sid in ("alpha-session-1", "beta-session-2", "gamma-session-3"):
            mem.holdout_arm(self.db, sid)
            path = write(self.tmp / f"{sid}.jsonl", [user(GARAGE, now + 60), user(self.PRIVATE, now + 70),
                                                     reply("I don't know Sam's birthday.", now + 80)])
            mem.hook_activity({"session_id": sid, "transcript_path": str(path), "cwd": str(self.tmp)}, final=True)
            with mock.patch.object(mem, "llm_json", lambda *a, **k: {"memories": []}):
                mem.extract(self.db, sid)
        out = mem.holdout_report(self.db)
        self.assertIn("Not enough sessions yet", out)
        keys = [r[0] for r in self.db.execute("select sid from holdout")]
        for text in [GARAGE, self.PRIVATE, "Sam", "birthday", "locker", "garage", "alpha", "beta", "gamma", *keys]:
            self.assertNotIn(text.lower(), out.lower())

    def test_enough_sessions_shows_intervals_and_what_it_takes(self):
        rows = []
        for i in range(400):
            arm = "held" if i % 10 == 0 else "memory"
            rows.append((f"k{i}", arm, "desk", time.time() - 7 * 86400, 10, 6, 8, (3 if arm == "held" else 1) + i % 3,
                         i % 2, i % 4, i % 5, 1000.0, 1000.0 + 60 * (5 + i % 7)))
        mem.holdout_arm(self.db, "make-the-table")
        with self.db:
            self.db.executemany("insert into holdout(sid, arm, scope, at, line, turns, owner_lines, reexplained, "
                                "corrections, dont_know, recalls, first_ts, last_ts) values (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                                rows)
        out = mem.holdout_report(self.db)
        self.assertNotIn("Not enough", out)
        line = next(x for x in out.splitlines() if x.startswith("re-explained what memory held"))
        self.assertTrue(line.endswith("*"), line)  # 3-5 held out against 1-3 with memory: a clear difference
        self.assertIn("95% interval", out)
        self.assertIn("to see a 20% change in re-explaining", out)
        self.assertNotIn("k1", out)


if __name__ == "__main__":
    import unittest
    unittest.main()
