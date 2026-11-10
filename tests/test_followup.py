"""followup-report (measurement only): which follow-up prompts are short, vague or brief, which bring up a name no
memory shown earlier in the session covers, whether the prompt alone or a query with the previous turns would find a
memory not yet shown; held-out sessions left out; the store is opened read-only; the report holds counts only.
Synthetic only: temp databases, a fake embedder and reranker, fixture transcripts, nothing of Robin's.

  .venv/bin/python -m unittest -q test_followup
"""
import _env  # noqa: F401  (first: the test data folder and user)
import json
import sqlite3
import time
from unittest import mock

from lflabs_memory import mem
from test_holdout import iso, reply, user, write
from lflabs_memory import instant
from test_redact import DBCase

HANA = "Hana Sato's desk extension at the district office is 4410"
BUDGET = "The library budget review with finance happens every quarter in the Alder room"
SECRET_WORDS = ("4410", "Alder", "extension", "budget")


def rerank(query, docs):
    """A memory scores high when the query has a word of its first line (names lowercased), else far below any cut."""
    q = set(mem._words(query))
    return [5.0 if q & {w for w in mem._words(d) if len(w) >= 4} - {"about", "what"} else -20.0 for d in docs]


class Case(DBCase):
    def setUp(self):
        super().setUp()
        people = self.tmp / "people.json"
        people.write_text(json.dumps({"people": [{"name": "Hana Sato", "first": "Hana"}]}))
        for p in (mock.patch.object(mem, "rerank", rerank), mock.patch.object(mem, "GATE", "raw"),
                  mock.patch.object(mem, "PEOPLE_FILE", people), mock.patch.object(mem, "_people_re", [None, None]),
                  mock.patch.object(mem, "VOICE_SESSION", self.tmp / "voice-session.json"),
                  mock.patch.object(mem, "TELEGRAM_SESSION", self.tmp / "telegram-session.json"),
                  mock.patch.dict(mem._holdout, {"live": False, "salt": None, "arms": {}})):
            p.start()
            self.addCleanup(p.stop)
        self.hana, _ = mem.remember(self.db, HANA, "reference", origin="owner_typed")
        self.budget, _ = mem.remember(self.db, BUDGET, "project", origin="owner_typed")
        self.t0 = time.time() - 3600

    def session(self, sid, turns, injected=()):
        """turns: (who, text) in order, a minute apart; injected: (memory id, turn index it was injected at)."""
        entries = [(user if who == "owner" else reply)(text, self.t0 + 60 * i) for i, (who, text) in enumerate(turns)]
        path = write(self.tmp / f"{sid}.jsonl", entries)
        with self.db:
            self.db.execute("insert into sessions(session_id, transcript, cwd, activity) values (?,?,?,?)",
                            (sid, str(path), "/tmp", self.t0 + 60 * len(turns)))
            self.db.executemany("insert into injections values (?,?,?)",
                                [(sid, m, self.t0 + 60 * i) for m, i in injected])

    def counts(self):
        per, held, missing = mem.followup_counts(self.db)
        return {k: sum(c[k] for c in per) for k in mem.FOLLOWUP_KEYS}, len(per), held, missing


class Kinds(Case):
    def test_kinds(self):
        k = mem.followup_kind
        self.assertEqual(k("and hers?"), "short")
        self.assertEqual(k("ok"), "short")
        self.assertIsNone(k(""))
        self.assertEqual(k("and what is her extension"), "vague")
        self.assertEqual(k("what about Hana's?"), "brief")
        self.assertEqual(k("what about hana's then"), "brief")  # the people index names her, in any case
        self.assertIsNone(k("can you look through the budget numbers for the third quarter again please"))

    def test_query_adds_the_previous_turns_names_and_words(self):
        q = mem.followup_query("and her extension?", "Is Hana Sato coming to the budget review?",
                               "Yes, Hana said she will. Claude can remind you.")
        self.assertTrue(q.startswith("and her extension?"))
        self.assertIn("hana", q.split())
        self.assertIn("budget", q.split())
        self.assertNotIn("claude", q.split())  # Claude's own name carries nothing
        self.assertEqual(q.split().count("hana"), 1)
        long = mem.followup_query("and?", " ".join(f"word{i}xyz" for i in range(40)))
        self.assertEqual(len(long.split()), 1 + mem.FOLLOWUP_TERMS)

    def test_entities_are_names_the_store_or_people_know(self):
        e = mem.followup_entities(self.db, "what about hana's", "Sure. The Alder room is booked. Path and Config")
        self.assertEqual(e, {"hana", "alder"})  # Path and Config: capitalised in the reply, unknown to the store


class Counts(Case):
    def test_a_short_follow_up_bringing_up_someone_new(self):
        self.session("s1", [("owner", "let's plan the library budget review for next week"),
                            ("claude", "Sure. Hana from finance usually joins it."),
                            ("owner", "her number?")],
                        injected=[(self.budget, 0)])  # the first prompt brought the budget memory, not Hana's
        t, n, held, missing = self.counts()
        self.assertEqual((n, held, missing), (1, 0, 0))
        self.assertEqual((t["prompts"], t["short"], t["short_new"], t["short_with"], t["gain"]), (1, 1, 1, 1, 1))
        self.assertEqual(t["short_alone"], 0)  # never searched today

    def test_nothing_new_when_an_earlier_injection_named_her(self):
        self.session("s1", [("owner", "who handles the district office extension list"),
                            ("claude", "That is Hana."), ("owner", "and hers?")], injected=[(self.hana, 0)])
        t, *_ = self.counts()
        self.assertEqual((t["short"], t["short_new"], t["gain"]), (1, 0, 0))

    def test_a_vague_follow_up_and_a_brief_one(self):
        self.session("s1", [("owner", "let's plan the library budget review for next week"),
                            ("claude", "Sure. Hana from finance usually joins it."),
                            ("owner", "and how do i reach her"),
                            ("claude", "Done."),
                            ("owner", "what about the Alder room")], injected=[(self.budget, 0)])
        t, *_ = self.counts()
        self.assertEqual((t["prompts"], t["vague"], t["vague_new"], t["vague_with"]), (2, 1, 1, 1))
        self.assertEqual(t["vague_alone"], 0)  # "and how do i reach her" names no one: the prompt alone finds nothing
        self.assertEqual((t["brief"], t["brief_new"]), (1, 0))  # Alder is in the budget memory, shown at the start
        self.assertEqual(t["gain"], 1)

    def test_a_long_prompt_is_not_a_follow_up_and_the_first_is_not_counted(self):
        self.session("s1", [("owner", "ok"), ("claude", "Hi."),
                            ("owner", "please go through the whole budget review agenda with me step by step now")])
        t, *_ = self.counts()
        self.assertEqual((t["prompts"], t["short"], t["vague"], t["brief"]), (1, 0, 0, 0))

    def test_held_out_old_and_missing_sessions_are_left_out(self):
        turns = [("owner", "let's plan the library budget review"), ("claude", "Hana joins."), ("owner", "her number?")]
        self.session("held", turns)
        self.session("gone", turns)
        self.session("old", turns)
        (self.tmp / "gone.jsonl").unlink()
        with self.db:
            self.db.execute("update sessions set activity = ? where session_id = 'old'",
                            (time.time() - (mem.FOLLOWUP_DAYS + 1) * 86400,))
        with mock.patch.object(mem, "HOLDOUT_MODE", "on"), mock.patch.object(mem, "HOLDOUT_RATE", 1.0):
            mem.holdout_arm(self.db, "held")
        self.assertEqual(self.counts()[1:], (0, 1, 1))

    def test_pastes_tool_output_and_agents_are_not_his_prompts(self):
        self.session("s1", [("owner", "let's plan the library budget review for next week"),
                            ("claude", "Sure. Hana from finance usually joins it.")])
        path = self.tmp / "s1.jsonl"
        write(path, [{"type": "user", "timestamp": iso(self.t0 + 200), "message": {"role": "user", "content": [
                         {"type": "tool_result", "tool_use_id": "x", "content": "her number?"}]}},
                     {**user("her number?", self.t0 + 260), "origin": {"kind": "agent"}}], mode="a")
        t, *_ = self.counts()
        self.assertEqual(t["prompts"], 0)


class Report(Case):
    def test_counts_only_and_read_only(self):
        self.session("s1", [("owner", "let's plan the library budget review for next week"),
                            ("claude", "Sure. Hana from finance usually joins it."),
                            ("owner", "her number?")], injected=[(self.budget, 0)])
        self.db.commit()
        ro = mem._connect_ro(mem.DB_PATH)
        self.addCleanup(ro.close)
        with self.assertRaises(sqlite3.OperationalError):
            ro.execute("insert into meta values ('x', 'y')")
        before = mem.DB_PATH.read_bytes()
        out = mem.followup_report(ro)
        self.assertEqual(mem.DB_PATH.read_bytes(), before)
        self.assertIn("1 of Robin's prompts", out)
        self.assertIn("Missed today, found with the previous turns: 1 prompts", out)
        for w in (*SECRET_WORDS, "Hana", "hana", "number", "s1"):
            self.assertNotIn(w, out.replace("Robin's", ""))

    def test_cli_never_writes_the_schema(self):
        self.db.close()
        fresh = self.tmp / "store"
        fresh.mkdir()
        sqlite3.connect(fresh / "memory.db").executescript(
            "create table sessions(session_id text, transcript text, cwd text, activity real);"
            "create table memories(id integer primary key, content text);"
            "create table injections(session_id text, memory_id integer, at real);")
        before = (fresh / "memory.db").read_bytes()
        with mock.patch("sys.argv", ["mem.py", "followup-report", "--store", str(fresh)]), \
                mock.patch.object(mem, "ROOT", mem.ROOT), mock.patch("builtins.print") as p:
            mem.main()
        self.assertIn("0 of Robin's prompts", p.call_args[0][0])
        self.assertEqual((fresh / "memory.db").read_bytes(), before)


class PreviousTurns(Case):
    def transcript(self, turns, name="t.jsonl"):
        return write(self.tmp / name, [(user if who == "owner" else reply)(text, self.t0 + 60 * i)
                                       for i, (who, text) in enumerate(turns)])

    def test_the_turn_before_from_the_transcripts_end(self):
        path = self.transcript([("owner", "what is Hana's extension"), ("claude", "Hana Sato's desk extension is 4410."),
                                ("owner", "and the budget review?"),
                                ("claude", "It happens every quarter in the Alder room. Shall I note it?")])
        got = mem.previous_turns(path, "yes do it")
        self.assertEqual(got["user"], "and the budget review?")
        self.assertIn("Alder room", got["reply"])
        self.assertNotIn("4410", got["reply"])

    def test_the_prompt_already_written_is_not_its_own_previous_turn(self):
        path = self.transcript([("owner", "and the budget review?"), ("claude", "Quarterly, in the Alder room."),
                                ("owner", "yes do it")])
        got = mem.previous_turns(path, "yes do it")
        self.assertEqual((got["user"], got["reply"]), ("and the budget review?", "Quarterly, in the Alder room."))

    def test_only_the_end_is_read_and_a_missing_transcript_is_none(self):
        turns = [("owner", f"question number {i} about the orchard " + "detail " * 40) for i in range(40)]
        turns += [("claude", "The last reply, about the Alder room."), ("owner", "a last question?")]
        path = self.transcript(turns)
        got = mem.previous_turns(path, "ok", tail=3000)  # far less than the file: the last turns still come
        self.assertEqual(got["user"], "a last question?")
        self.assertIn("Alder room", got["reply"])
        self.assertIsNone(mem.previous_turns(self.tmp / "gone.jsonl", "ok"))
        self.assertIsNone(mem.previous_turns(self.transcript([], "empty.jsonl"), "ok"))


class Widened(Case):
    PREV = {"user": "what about the orchard budget review", "reply": "The budget review is quarterly, in the Alder room."}

    def test_repo_mode_widens_a_coding_sessions_vague_follow_up_only(self):
        with mock.patch.object(mem, "FOLLOWUP_MODE", "repo"), \
                mock.patch.dict(mem._repo_roots, {"/code/x": "/code/x", "/home/me": None}, clear=True):
            got = mem.followup_widened("yes do it", self.PREV, None, "/code/x")
            self.assertTrue(got.startswith("yes do it ") and "budget" in got and "alder" in got, got)
            self.assertEqual(mem.followup_widened("ok", self.PREV, None, "/code/x"), "ok")  # under three words: never searched
            self.assertEqual(mem.followup_widened("yes do it", self.PREV, None, "/home/me"), "yes do it")  # not a repo
            self.assertEqual(mem.followup_widened("yes do it", self.PREV, None, None), "yes do it")
            self.assertEqual(mem.followup_widened("yes do it", self.PREV, "telegram", "/code/x"), "yes do it")  # a scope
            self.assertEqual(mem.followup_widened("what about Hana's?", self.PREV, None, "/code/x"), "what about Hana's?")
            long = "tell me everything we know about the orchard budget review and the Alder room please"
            self.assertEqual(mem.followup_widened(long, self.PREV, None, "/code/x"), long)  # a full prompt: as it is
            self.assertEqual(mem.followup_widened("yes do it", None, None, "/code/x"), "yes do it")  # no turn before
        with mock.patch.object(mem, "FOLLOWUP_MODE", "off"), mock.patch.dict(mem._repo_roots, {"/code/x": "/code/x"}, clear=True):
            self.assertEqual(mem.followup_widened("yes do it", self.PREV, None, "/code/x"), "yes do it")
        with mock.patch.object(mem, "FOLLOWUP_MODE", "on"), mock.patch.dict(mem._repo_roots, {}, clear=True):
            self.assertIn("budget", mem.followup_widened("yes do it", self.PREV, "telegram", None))  # anywhere

    def test_a_vague_follow_up_that_finds_nothing_is_searched_with_the_turn_before(self):
        prev = {"user": "when is the library budget review", "reply": "The library budget review with finance is quarterly."}
        searches = []
        real = mem.search
        with mock.patch.object(mem, "FOLLOWUP_MODE", "repo"), mock.patch.dict(mem._repo_roots, {"/code/x": "/code/x"}, clear=True), \
                mock.patch.object(mem, "search", lambda db, q, **k: searches.append(q) or real(db, q, **k)):
            self.assertIsNone(mem.prompt_context(self.db, "yes do it", "s9", "/code/x", None, False))
            self.assertEqual(searches, ["yes do it"])  # no turn before: one search, as always
            got = mem.prompt_context(self.db, "yes do it", "s9", "/code/x", None, False, prev=prev)
            self.assertIn("library budget review", got)
            self.assertNotIn("4410", got)  # Hana's extension: the turn before did not bring her up
            self.assertEqual(searches[1], "yes do it")  # the prompt alone first
            self.assertTrue(searches[2].startswith("yes do it ") and "budget" in searches[2], searches)
            del searches[:]  # the prompt alone finds it: one search, the widening is not tried
            got = mem.prompt_context(self.db, "and Hana's extension?", "s9", "/code/x", None, False, prev=prev)
            self.assertIn("4410", got)
            self.assertEqual(len(searches), 1)
            # once per session: the same follow-up again brings nothing new
            self.assertIsNone(mem.prompt_context(self.db, "yes do it", "s9", "/code/x", None, False, prev=prev))

    def test_a_secret_is_never_widened_in(self):
        row = None
        for phrase in ("the gym locker code is 7781", "my bank card PIN is 7781", "the wifi password is hunter7781"):
            mid, _ = mem.remember(self.db, phrase, "reference", origin="owner_typed")
            row = self.db.execute("select * from memories where id = ?", (mid,)).fetchone()
            if mem.is_secret(row):
                break
        else:
            self.skipTest("no phrase counted as a secret")
        prev = {"user": "what is the gym locker code again, the wifi password and my bank card PIN",
                "reply": "The gym locker code, the wifi password and the bank card PIN are in memory; want them?"}
        with mock.patch.object(mem, "FOLLOWUP_MODE", "repo"), mock.patch.dict(mem._repo_roots, {"/code/x": "/code/x"}, clear=True):
            got = mem.prompt_context(self.db, "yes please", "s9", "/code/x", None, False, prev=prev) or ""
        self.assertNotIn("7781", got)

