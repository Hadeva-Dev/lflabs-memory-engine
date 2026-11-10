"""Changing access by talking (TALK_PERMS_MODE, access.py): the deterministic parser and what it
must never match, the one safety rule (narrowing from anywhere, widening only from Robin's own short typed line, else a
pending request only his typed words approve), undo, levels, the access tools, the audit log and notifications (names
only), keyless identity (KEYLESS_MODE), and the handoff phrases. Synthetic only: temp folders, nothing of Robin's, no
model calls, no notification shown (access._post is replaced).

  .venv/bin/python -m unittest -q test_access
"""
import _env  # noqa: F401  (first: the test data folder and user)
import io
import json
import os
import stat
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from lflabs_memory import brand, access
from lflabs_memory import mem
from test_redact import DBCase

U = {*mem.SECTIONS, mem.WITHHELD, "chats"}
WORK_HIDE = tuple(sorted(U - set(mem.WORK_ONLY)))
SECRET_MEMORY = "Orchard: Robin's allergy medication is 10 mg each morning."


def parsed(prompt, speaker="desk"):
    return [{k: v for k, v in i.items() if k != "raw"} for i in access.parse(prompt, speaker)]


class Parser(unittest.TestCase):
    def one(self, prompt, **want):
        got = parsed(prompt)
        self.assertEqual(len(got), 1, (prompt, got))
        for k, v in want.items():
            self.assertEqual(got[0].get(k), v, (prompt, got))
        return got[0]

    def test_the_five_example_sentences(self):
        self.one("Let Cursor see my personal stuff.", kind="change", op="show", t="cursor", what=["personal"])
        self.one("Don't let Codex see my health stuff", kind="change", op="hide", t="codex", what=["health-money"])
        self.one("make Cursor work-only", kind="change", op="level", level="work")
        self.one("what can Codex see?", kind="status", t="codex")
        self.one("undo that", kind="undo", bare=True)

    def test_widening_forms(self):
        for p, what in [("allow Codex to see my past chats", ["chats"]), ("give codex access to personal stuff", ["personal"]),
                        ("show Cursor my contacts and past chats", ["chats", "people"]), ("let codex read my past chats", ["chats"]),
                        ("open up personal stuff to cursor", ["personal"]), ("share my finances with gemini", ["health-money"]),
                        ("please let the Claude app see my contacts", ["people"]), ("let codex see my contacts", ["people"]),
                        ("I want Windsurf to see my background", ["about"]), ("give Codex everything", ["*"])]:
            self.one(p, kind="change", op="show", what=what)

    def test_narrowing_forms(self):
        for p in ["hide past chats from Codex", "stop Codex seeing my contacts", "stop letting cursor read my chat history",
                  "block Cursor from my personal notes", "keep my medical stuff away from codex",
                  "revoke Codex's access to my past chats", "Codex shouldn't see my contacts", "don't share my past chats with Cursor",
                  "take away cursor's access to my old conversations", "no more people notes for codex",
                  "I don't want Gemini seeing my money stuff", "Hide past chats and people notes from codex and cursor."]:
            self.one(p, kind="change", op="hide")

    def test_levels_and_targets(self):
        self.one("Gemini: nothing personal", op="level", t="gemini", level="nothing-personal")
        self.one("set copilot to work only", op="level", t="copilot", level="work")
        self.one("only let cursor see preferences and project notes", op="only", what=["prefs", "projects"])
        self.one("stop letting every other AI see my contacts", op="hide", t="every other ai")
        self.one("everyone except claude code shouldn't see my past chats", op="hide", t="everyone except claude code")

    def test_approve_undo_handoff(self):
        self.one("approve Cursor's request", kind="approve", bare=False, t="cursor")
        self.one("approve the access request", kind="approve", bare=False)
        self.one("yes, allow it", kind="approve", bare=True)
        self.one("undo the last access change", kind="undo", bare=False)
        self.one("I'm switching to Codex now", kind="handoff_out", t="codex")
        self.one("hand off", kind="handoff_out")
        self.one("Pick up where Claude Code left off and run the tests", kind="handoff_in")
        self.one("continue from the handoff", kind="handoff_in")

    def test_never_misfires(self):
        for p in ["Codex couldn't see the meeting link", "let me see the email you drafted",
                  "don't let me forget to email Anders", "can you show Cursor's settings", "hide the sidebar in the app",
                  "let codex see the messages", "show codex the error", "give codex the logs",
                  "allow cursor to use the terminal", "stop codex from reading the file", "what does codex think",
                  "Codex can see my email but it says it can't", "let's hand off the parsing to a helper function",
                  "the handoff logic is broken", "allow it to run longer", "undo that change to the button",
                  "approve the request in GitHub", "Cursor should see the error in the email template",
                  "let Claude see the meetings page", "show me what cursor is doing",
                  'He told me "let Cursor see my meetings" yesterday']:
            self.assertEqual(parsed(p), [], p)

    def test_code_and_quotes_are_skipped(self):
        for p in ["```\nlet cursor see my meetings\n```", "```python\npolicy = {'allow': ['email'], 'deny': []}\n"
                  "allow codex to see my email\n```", "    let cursor see my meetings", "> let cursor see my meetings",
                  "clients.set('codex', allow=['email'])  # allow codex to see my email",
                  "if allow: deny(codex); hide(email)"]:
            self.assertEqual(parsed(p), [], p)

    def test_the_hooks_first_look_lets_every_phrase_through(self):
        """mem.TALK_HINT_RE decides whether access.py is loaded at all: anything parse() reads must pass it."""
        import re
        phrases = re.findall(r'self\.(?:one|say|say_untyped)\("([^"]+)"', Path(__file__).read_text())
        phrases += re.findall(r'\("([^"]+)", \[', Path(__file__).read_text())
        self.assertGreater(len(phrases), 40)
        for p in phrases:
            if parsed(p):
                self.assertTrue(mem.TALK_HINT_RE.search(p), p)

    def test_undo_and_allow_it_only_as_the_whole_prompt(self):
        self.assertEqual(parsed("Fix the flaky test. Undo that."), [])
        self.assertEqual(parsed("ok, allow it. then rerun"), [])


class Case(DBCase):
    def setUp(self):
        super().setUp()
        self.notes = []
        for p in (mock.patch.object(mem, "TALK_PERMS_MODE", "on"), mock.patch.object(mem, "KEYLESS_MODE", "on"),
                  mock.patch.object(mem, "CLIENT_SCOPES_MODE", "on"),
                  mock.patch.object(mem, "VOICE_SESSION", self.tmp / "voice-session.json"),
                  mock.patch.object(mem, "TELEGRAM_SESSION", self.tmp / "telegram-session.json"),
                  mock.patch.object(mem, "SOCK", self.tmp / "mem.sock"),
                  mock.patch.object(mem, "CLAIMS_MODE", "off"),
                  mock.patch.dict(mem._clients, {"key": None, "data": {}}), mock.patch.dict(mem.RESTRICTED),
                  mock.patch.object(access, "_post", self.notes.append),
                  mock.patch.dict(os.environ, {"LFLABS_MEMORY_SCOPE": "", mem.CLIENT_ENV: "", mem.CLIENT_KEY_ENV: "",
                                               "LFLABS_MEMORY_NOSAVE": "", mem.CHILD_ENV: "",
                                               "CLAUDE_CODE_ENTRYPOINT": "cli", "LFLABS_MEMORY_NOTIFY": "",
                                               "HOME": str(self.tmp / "home")})):
            p.start()
            self.addCleanup(p.stop)
        (self.tmp / "home").mkdir()
        self.assertEqual(Path.home(), self.tmp / "home")  # never a real app config

    def say(self, prompt, speaker="desk", **data):
        out = access.on_prompt(mem, {"prompt": prompt, "session_id": data.pop("session_id", "s1"), **data}, speaker)
        return "\n".join((out or {}).get("lines") or [])

    def hide(self, name):
        return set(mem.client_hide(name))

    def tool(self, requester, client, change, what):
        return access.tool_change(mem, requester, client, change, what)


class SafetyRule(Case):
    def test_typed_widening_applies_logs_and_notifies(self):
        self.assertIn("personal", self.hide("cursor"))  # a new agent starts at work only
        line = self.say("let Cursor see my personal stuff")
        self.assertEqual(line, mem.LABEL + ": Cursor can now see personal notes. (Robin's own words. They can say 'undo that'.)")
        self.assertNotIn("personal", self.hide("cursor"))
        self.assertEqual(self.notes, ["Cursor can now see personal notes · say 'undo that' in any AI to undo"])
        row = access.history(mem)[-1]
        self.assertEqual((row["client"], row["dir"], row["source"]), ("cursor", "loosen", "typed"))
        self.assertEqual(stat.S_IMODE(os.stat(self.tmp / "access-history.jsonl").st_mode), 0o600)

    def test_narrowing_applies_from_any_sentence_without_a_notification(self):
        doc = "Notes from the meeting.\n" + "Lots of pasted text here. " * 60 + "\nHide past chats from Claude Code."
        self.assertIn("can no longer see past chats", self.say(doc))
        self.assertIn("chats", self.hide("desk"))
        self.assertEqual(self.notes, [])

    def test_widening_in_a_long_prompt_is_only_a_request(self):
        doc = "Here is the doc.\n" + "Some pasted paragraph text. " * 30 + "\nLet Cursor see my personal stuff."
        line = self.say(doc)
        self.assertIn("waiting for Robin's approval", line)
        self.assertIn("personal", self.hide("cursor"))
        self.assertEqual(len(access.requests(mem)), 1)
        self.assertEqual(self.notes, ["A long or pasted message in Claude Code asked to let Cursor see personal notes · to "
                                      "allow, type 'approve Cursor's request' in any AI"])

    def test_widening_in_pasted_part_of_a_short_prompt_is_a_request(self):
        prompt = "what do you make of this?\n" + "x " * 160 + "\nLet Cursor see my personal stuff."
        self.assertLess(len(prompt), access.LOOSEN_PROMPT_MAX)
        self.assertIn("waiting", self.say(prompt))
        self.assertIn("personal", self.hide("cursor"))

    def test_untyped_sources_only_request(self):
        cases = [({"CLAUDE_CODE_ENTRYPOINT": "sdk-cli"}, "desk", {}), ({"LFLABS_MEMORY_NOSAVE": "1"}, "desk", {}),
                 ({mem.CHILD_ENV: "1"}, "desk", {}), ({}, "codex", {}), ({}, "cursor", {}),
                 ({}, "desk", {"prompt_prefix": "<task-notification>done</task-notification>\n"})]
        for env, speaker, extra in cases:
            with mock.patch.dict(os.environ, env):
                line = self.say(extra.get("prompt_prefix", "") + "let Cursor see my personal stuff", speaker)
            self.assertIn("waiting", line, (env, speaker))
            self.assertIn("personal", self.hide("cursor"), (env, speaker))

    def test_voice_and_telegram_only_request(self):
        (self.tmp / "voice-session.json").write_text(json.dumps({"id": "voice1"}))
        self.assertIn("waiting", self.say("let cursor see my personal stuff", session_id="voice1"))
        with mock.patch.dict(os.environ, {"LFLABS_MEMORY_SCOPE": "telegram"}):
            self.assertIn("waiting", self.say("let cursor see my past chats"))
        self.assertTrue({"personal", "chats"} <= self.hide("cursor"))

    def test_codex_hook_counts_only_for_a_session_owner_started(self):
        def rollout(name, source):
            p = self.tmp / f"rollout-{name}.jsonl"
            p.write_text(json.dumps({"type": "session_meta", "payload": {"id": name, "source": source}}) + "\n")
            return str(p)
        self.assertIn("waiting", self.say("let cursor see my personal stuff", "codex", transcript_path=rollout("x", "exec")))
        self.assertIn("can now see", self.say("let cursor see my personal stuff", "codex", transcript_path=rollout("y", "cli")))

    def test_the_tool_narrows_at_once_and_only_asks_to_widen(self):
        self.assertEqual(self.tool("codex", "cursor", "hide", "chats"), "Cursor: no change.")  # work only already
        out = self.tool("codex", "cursor", "hide", "projects")
        self.assertEqual(out, "Cursor can no longer see project notes.")
        self.assertEqual(access.history(mem)[-1]["source"], "tool:codex")
        self.assertEqual(self.notes, ["Codex hid project notes from Cursor · say 'undo the last access change' in any AI to "
                                      "undo"])
        before = self.hide("cursor")
        for change, what in [("show", "personal"), ("level", "everything"), ("only", "people"), ("show", "everything")]:
            self.assertIn("applies only if they approve it by typing", self.tool("cursor", "cursor", change, what))
        self.assertEqual(self.hide("cursor"), before)
        self.assertIn("Cursor asked to see everything · to allow, type 'approve Cursor's request' in any AI", self.notes)
        self.assertIn("must be", self.tool("cursor", "cursor", "approve", "personal"))  # no way to approve from a tool
        self.assertEqual(self.hide("cursor"), before)

    def test_identical_requests_notify_once(self):
        for _ in range(5):
            self.tool("cursor", "cursor", "show", "personal")
        self.assertEqual(len(access.requests(mem)), 1)
        self.assertEqual(len(self.notes), 1)

    def test_typed_approval_applies_the_request(self):
        self.tool("cursor", "cursor", "show", "personal")
        self.assertIn("approving an access request needs Robin's own", self.say_untyped("approve the access request"))
        self.assertIn("personal", self.hide("cursor"))
        line = self.say("approve Cursor's request")
        self.assertIn("approved; Cursor", line)
        self.assertNotIn("personal", self.hide("cursor"))
        self.assertEqual(access.history(mem)[-1]["source"], "approved")
        self.assertEqual(access.requests(mem), [])

    def say_untyped(self, prompt):
        with mock.patch.dict(os.environ, {"CLAUDE_CODE_ENTRYPOINT": "sdk-py"}):
            return self.say(prompt)

    def test_allow_it_never_applies_and_says_what_would(self):
        self.assertEqual(self.say("allow it"), "")  # nothing waiting: silent ("allow it" is everyday coding talk)
        self.tool("cursor", "cursor", "show", "personal")
        line = self.say("allow it")
        self.assertEqual(line, mem.LABEL + ": waiting for Robin's approval: Cursor would see personal notes. A bare 'allow it' "
                               "approves nothing; they can say 'approve Cursor's request'.")
        self.assertIn("personal", self.hide("cursor"))
        self.assertIn("approved", self.say("approve the access request"))  # the only one waiting: applies
        self.assertNotIn("personal", self.hide("cursor"))

    def test_a_refiled_request_keeps_its_first_time(self):
        self.tool("cursor", "cursor", "show", "personal")
        first = access.requests(mem)[0]["at"]
        time.sleep(0.01)
        self.tool("cursor", "cursor", "show", "personal")
        self.assertEqual([r["at"] for r in access.requests(mem)], [first])
        self.assertEqual(len(self.notes), 1)

    def test_one_agent_cannot_push_out_the_others_requests(self):
        self.tool("codex", "codex", "show", "chats")
        for t in ("cursor", "windsurf", "gemini", "copilot", "chatgpt", "claude app"):
            self.tool("cursor", t, "show", "personal")
        self.assertIn("no agent called", self.tool("cursor", "made-up-agent", "show", "personal"))
        reqs = access.requests(mem)
        self.assertEqual(sum(r["by"] == "cursor" for r in reqs), access.PER_REQUESTER)
        self.assertTrue(any(r["by"] == "codex" for r in reqs))

    def test_requests_expire(self):
        self.tool("cursor", "cursor", "show", "personal")
        self.assertEqual(access.requests(mem, now=time.time() + access.PENDING_TTL + 1), [])

    def test_notifications_and_log_never_hold_memory_text(self):
        mem.remember(self.db, SECRET_MEMORY, "user", None, origin="owner_typed")
        self.say("let cursor see my health stuff")
        self.tool("codex", "cursor", "hide", "health")
        self.tool("cursor", "cursor", "show", "everything")
        blob = " ".join(self.notes) + (self.tmp / "access-history.jsonl").read_text()
        for word in ("allergy", "10 mg", "Orchard"):
            self.assertNotIn(word, blob)
        self.assertNotIn("key_sha256", blob)


class Undo(Case):
    def test_undo_that_reverts_the_last_change(self):
        self.say("let cursor see my personal stuff")
        self.assertIn("undid the last access change", self.say("undo that"))
        self.assertIn("personal", self.hide("cursor"))
        self.assertEqual(access.history(mem)[-1]["undoes"], access.history(mem)[-2]["id"])

    def test_bare_undo_only_when_fresh(self):
        self.say("let cursor see my personal stuff")
        rows = (self.tmp / "access-history.jsonl").read_text().splitlines()
        old = json.loads(rows[-1])
        old["at"] -= 11 * 60
        (self.tmp / "access-history.jsonl").write_text(json.dumps(old) + "\n")
        self.assertEqual(self.say("undo that"), "")  # too old: silent
        self.assertNotIn("personal", self.hide("cursor"))
        self.assertIn("undid", self.say("undo the last access change"))
        self.assertIn("personal", self.hide("cursor"))

    def test_undoing_a_narrowing_needs_typed_words(self):
        self.tool("codex", "desk", "hide", "chats")  # an injected agent narrows Claude Code
        self.assertIn("needs Robin's own typed words", self.say_untyped("undo the last access change"))
        self.assertIn("chats", self.hide("desk"))
        self.assertEqual(self.say("undo that"), "")  # the change was not made in this session: a bare undo is silent
        self.assertIn("undid", self.say("undo the last access change"))
        self.assertEqual(self.hide("desk"), set())

    def say_untyped(self, prompt):
        with mock.patch.dict(os.environ, {"CLAUDE_CODE_ENTRYPOINT": "sdk-py"}):
            return self.say(prompt)

    def test_undo_goes_back_one_change_at_a_time(self):
        self.say("hide past chats from cursor")
        self.say("hide people notes from codex")
        self.say("undo the last access change")
        self.assertIn("people", self.hide("codex"))  # codex was work only already: texts stays hidden
        self.say("undo the last access change")
        self.assertEqual(self.hide("cursor"), set(WORK_HIDE))
        self.assertIn("no access change to undo", self.say("undo the last access change"))


class Levels(Case):
    def test_levels(self):
        self.say("give Codex everything")
        self.assertEqual(self.hide("codex"), set())
        self.say("make Codex work-only")
        self.assertEqual(self.hide("codex"), set(WORK_HIDE))
        self.say("Codex: nothing personal")
        self.assertEqual(self.hide("codex"), set(mem.NOTHING_PERSONAL))
        self.say("make codex nothing")
        self.assertEqual(self.hide("codex"), U)
        self.assertEqual(access.level_name(self.hide("codex"), mem), "nothing")

    def test_hiding_inside_an_allow_list_never_widens(self):
        mem.clients_set("cursor", allow=["prefs"], deny=[])
        self.say("hide preferences from cursor")
        self.assertEqual(self.hide("cursor"), U)

    def test_every_other_ai_and_except(self):
        self.say("give every AI everything")  # widening for all: named agents, never "new agents"
        self.assertEqual(self.hide("codex"), set())
        self.assertEqual(self.hide("unknown"), set(WORK_HIDE))
        self.say("stop letting every other AI see my contacts")
        self.assertNotIn("people", self.hide("desk"))  # the one speaking
        self.assertIn("people", self.hide("codex"))
        self.say("everyone except claude code shouldn't see my past chats")
        self.assertNotIn("chats", self.hide("desk"))
        self.assertTrue({"chats"} <= self.hide("claude-desktop") and {"chats"} <= self.hide("unknown"))

    def test_status_is_read_only(self):
        line = self.say("what can Codex see?")
        self.assertTrue(line.startswith(mem.LABEL + ": Codex is work only: it sees "), line)
        self.assertEqual(access.history(mem), [])
        self.assertIn("Claude Code sees everything", access.status_text(mem, "claude code"))

    def test_off_note_when_scopes_are_off(self):
        with mock.patch.object(mem, "CLIENT_SCOPES_MODE", "off"):
            self.assertIn("takes effect once they are on", self.say("hide past chats from codex"))

    def test_scopes_take_effect_in_recall(self):
        mid = mem.remember(self.db, "Robin's sister Bea moved to Lisbon in the spring.", "note", None)[0]  # personal
        view = mem.client_scope("cursor", True)
        self.assertNotIn(mid, mem._visible(self.db, [mid], view))
        self.say("let cursor see my personal stuff")
        self.assertIn(mid, mem._visible(self.db, [mid], mem.client_scope("cursor", True)))


class Keyless(Case):
    def test_a_declared_name_counts_without_a_key(self):
        self.assertEqual(mem.resolve_client("gemini"), ("gemini", mem.UNKEYED))
        self.assertEqual(mem.resolve_client(hint="codex-mcp-client"), ("codex", mem.UNKEYED))
        self.assertEqual(set(mem.client_hide("gemini")), set(WORK_HIDE))  # new agents start at work only
        self.assertEqual(mem.client_hide("desk"), ())
        self.assertEqual(mem.client_hide("claude-desktop"), ())

    def test_a_client_with_a_key_still_needs_it(self):
        key, _ = mem.clients_add("codex", with_key=True)
        self.assertEqual(mem.resolve_client("codex"), ("codex", False))
        self.assertEqual(mem.resolve_client("codex", key), ("codex", True))

    def test_new_agents_follow_the_unknown_policy(self):
        self.say("give new agents everything")  # widening "new agents" names unknown itself: allowed when typed
        self.assertEqual(set(mem.client_hide("windsurf")), set())  # its policy
        self.assertEqual(set(mem.client_hide(*mem.resolve_client("windsurf"))), set(WORK_HIDE))  # a name alone: capped

    def test_existing_lists_are_kept_as_they_are(self):
        mem._write_clients({"codex": {"deny": ["people"]}})
        self.assertEqual(mem.client_hide("codex"), ("people",))

    def test_add_prints_no_key(self):
        out = io.StringIO()
        with redirect_stdout(out):
            mem._clients_cli(["add", "cursor"])
        self.assertNotIn(mem.CLIENT_KEY_ENV, out.getvalue())
        self.assertIn('"--client", "cursor"', out.getvalue())
        with redirect_stdout(io.StringIO()) as o2:
            mem._clients_cli(["add", "cursor", "--key"])
        self.assertIn(mem.CLIENT_KEY_ENV, o2.getvalue())

    def test_the_products_cli_keys_codex_itself_so_level_says_neither_line(self):
        """VM run, v0.1.8-rc2: `<command> access codex everything` printed the engine's key hint and "not connected
        yet" before its own keying. With KEY_BY_ENV set to the command, `clients level` says neither and keys nothing."""
        (Path(os.environ["HOME"]) / ".codex").mkdir(parents=True, exist_ok=True)  # Codex here, no entry of ours yet

        def run(env):
            with redirect_stdout(io.StringIO()) as o, mock.patch.dict(os.environ, env), \
                    mock.patch.object(access, "human_at_terminal", lambda m, n, **k: None):
                mem._clients_cli(["level", "codex", "everything"])
            mem.clients_put("codex", access._mutation(mem, "level", level="work")[0], "test")  # back, for the next run
            return o.getvalue()
        plain = run({access.KEY_BY_ENV: ""})
        self.assertIn("until its config carries a key", plain)
        self.assertIn("not connected yet", plain)
        by_cli = run({access.KEY_BY_ENV: brand.COMMAND})
        self.assertIn("Codex sees everything", by_cli)
        self.assertNotIn("until its config carries a key", by_cli)
        self.assertNotIn("not connected", by_cli)

    def test_the_variable_never_gets_an_agent_past_the_gate(self):
        before = mem.client_hide("codex")
        with redirect_stdout(io.StringIO()) as o, mock.patch.dict(os.environ, {access.KEY_BY_ENV: brand.COMMAND}), \
                mock.patch.object(mem, "run_by_agent", lambda env=None: True), self.assertRaises(SystemExit):
            mem._clients_cli(["level", "codex", "everything"])
        self.assertIn("an AI app ran this", o.getvalue())
        self.assertEqual(mem.client_hide("codex"), before)

    def test_cli(self):
        def run(*a):
            with redirect_stdout(io.StringIO()) as o, mock.patch.object(access, "human_at_terminal", lambda m, n, **k: None):
                mem._clients_cli(list(a))  # the user at a Terminal, who typed the app's name (test_app_trust)
            return o.getvalue()
        self.assertIn("everything", run("level", "cursor", "everything"))
        self.assertEqual(self.notes[-1], "Cursor can now see health and money, past chats, people and personal notes "
                                         "(changed from the command line) · say 'undo the last access change' in any "
                                         "AI to undo")
        self.tool("codex", "codex", "show", "chats")
        self.assertIn("codex asks", run("requests"))
        self.assertIn("Codex", run("approve"))
        self.assertIn("cli-approve", run("history"))
        self.assertIn("undid it", run("undo"))
        self.assertIn("[work only]", run())


class Handoff(Case):
    def test_hand_off_marks_the_session_and_wakes_the_worker(self):
        with mock.patch.object(mem, "HANDOFF_MODE", "on"), mock.patch.object(mem, "connect", lambda: self.db):
            out = access.on_prompt(mem, {"prompt": "I'm switching to Codex now", "session_id": "h1",
                                         "transcript_path": str(self.tmp / "h1.jsonl"), "cwd": str(self.tmp)})
        self.assertEqual(out["lines"], [mem.LABEL + ": writing a handoff for the next agent."])
        self.assertTrue(out["urgent"])
        self.assertEqual(self.db.execute("select final from sessions where session_id = 'h1'").fetchone()[0], 1)

    def test_nothing_with_handoff_off(self):
        self.assertIsNone(access.on_prompt(mem, {"prompt": "hand off to codex", "session_id": "h2"}))

    def test_pick_up_gives_the_handoff_again(self):
        proj = self.tmp / "orchard"
        proj.mkdir()
        with mock.patch.object(mem, "HANDOFF_MODE", "on"), mock.patch.dict(mem._project_keys, clear=True):
            with self.db:
                mem._handoff_tables(self.db)
                self.db.execute("insert into handoffs values (?, 'codex', 'cx', ?, ?, ?)", (
                    mem.project_key(str(proj)), str(proj), time.time() - 600,
                    json.dumps({"decided": [], "done": [{"text": "Shipped the cache", "section": "prefs",
                                                         "flags": None}], "next": [], "open": []})))
            first = mem.handoff_context(self.db, "c1", str(proj), "desk")
            self.assertIn("Bring it up only where it bears on what Robin asks now.", first)
            self.assertIsNone(mem.handoff_context(self.db, "c1", str(proj), "desk"))
            self.assertIsNone(mem.handoff_context(self.db, "c2", str(proj), "desk"))  # given once per client
            self.assertEqual(mem.handoff_context(self.db, "c2", str(proj), "desk", again=True), first)
            out = access.on_prompt(mem, {"prompt": "pick up where Codex left off", "session_id": "c2"})
            self.assertTrue(out["again"])

    def test_the_hook_puts_the_line_first_and_asks_the_daemon_again(self):
        sent = {}

        def fake_context(db, prompt, session_id=None, cwd=None, scope=None, count=True, client=None, again=False):
            sent["again"] = again
            return "Recalled from long-term memory: something"
        with mock.patch.object(mem, "HANDOFF_MODE", "on"), mock.patch.object(mem, "prompt_context", fake_context), \
                mock.patch.object(mem, "connect", lambda: self.db), redirect_stdout(io.StringIO()) as out:
            mem.hook_prompt({"prompt": "continue from the handoff", "session_id": "c3"})
        ctx = json.loads(out.getvalue())["hookSpecificOutput"]["additionalContext"]
        self.assertTrue(ctx.startswith(mem.LABEL + ": Robin wants to pick up where the last agent left off"))
        self.assertIn("\n\nRecalled from long-term memory", ctx)
        self.assertTrue(sent["again"])

    def test_the_daemon_wakes_the_worker_for_urgent(self):
        mem._wake.clear()
        handler = mem._handler(self.db)
        with mock.patch.object(mem, "prompt_context", lambda *a, **k: None):
            h = handler.__new__(handler)
            h.rfile = io.BytesIO(json.dumps({"prompt": "hand off", "session_id": "x", "urgent": True}).encode() + b"\n")
            h.wfile = io.BytesIO()
            h.handle()
        self.assertTrue(mem._wake.is_set())
        mem._wake.clear()


class RoundOneParser(unittest.TestCase):
    """Adversarial review 2026-10-20, parser lens: widening takes fewer words than narrowing; more natural forms."""

    def test_sandbox_and_memory_talk_never_widens(self):
        for p in ["give Codex full access", "grant Codex full access", "let Codex have full access",
                  "let Cursor use everything", "let Codex use memory", "give Cursor memory", "share memories with Cursor",
                  "give Codex anything", "let Codex see it all", "Cursor: everything", "Codex: full access",
                  "Codex is now everything", "set cursor to everything", "turn Cursor into everything",
                  "open work to Cursor", "let Cursor use gmail", "give Codex work", "let Codex see work",
                  "give Codex school", "allow cursor to use the memory profiler", "let Codex read everything you wrote"]:
            self.assertFalse([i for i in parsed(p) if i.get("op") in ("show", "only", "level")
                              and i.get("level") != "nothing"], p)

    def test_more_natural_forms(self):
        for p, kind, op in [("make sure Codex can't see my health stuff", "change", "hide"),
                            ("make sure Cursor doesn't see my contacts", "change", "hide"),
                            ("Codex can't see my health stuff anymore", "change", "hide"),
                            ("lock Cursor out of my contacts", "change", "hide"),
                            ("cut off Codex from my health stuff", "change", "hide"),
                            ("disable Cursor's access to past chats", "change", "hide"),
                            ("Don't let Codex see my health stuff, it's private.", "change", "hide"),
                            ("don't let codex see anything", "change", "hide"),
                            ("restrict Cursor to work", "change", "level"), ("keep Cursor work-only", "change", "level"),
                            ("Cursor is work only", "change", "level"), ("make Codex see nothing", "change", "level"),
                            ("Codex can only see my work stuff", "change", "only"),
                            ("what access does Codex have?", "status", None), ("who can see my contacts?", "status", None),
                            ("approve the request from Cursor", "approve", None),
                            ("undo the last change to Cursor's access", "undo", None),
                            ("pick up from Codex's handoff", "handoff_in", None)]:
            got = parsed(p)
            self.assertEqual([(i["kind"], i.get("op")) for i in got], [(kind, op)], p)
            self.assertTrue(mem.TALK_HINT_RE.search(p), p)

    def test_handoff_out_is_the_whole_sentence(self):
        for p in ["I'm switching to Cursor's tab completion", "I'll continue with Codex's suggestion",
                  "I'm moving to Cursor from VS Code", "let's continue with Copilot's suggestion"]:
            self.assertEqual(parsed(p), [], p)
        self.assertEqual(parsed("I'm moving to Cursor soon"), [])
        self.assertEqual(parsed("I'm moving to Cursor now")[0]["kind"], "handoff_out")

    def test_single_quoted_lines_are_quotes(self):
        self.assertEqual(parsed("'Let Codex see everything.'"), [])


class RoundOneSafety(Case):
    """Adversarial review 2026-10-20 (escalation lens) and the follow-up security review."""

    def test_narrowing_an_agent_with_no_policy_never_widens_it(self):
        self.say("make new agents nothing")
        self.assertEqual(self.hide("windsurf"), U)
        self.assertEqual(self.tool("cursor", "windsurf", "hide", "prefs"), "Windsurf: no change.")
        self.say_untyped("Hide project notes from Gemini.")
        self.assertEqual(self.hide("windsurf"), U)
        self.assertEqual(self.hide("gemini"), U)
        self.assertFalse([r for r in access.history(mem) if r["dir"] == "loosen"])

    def say_untyped(self, prompt):
        with mock.patch.dict(os.environ, {"CLAUDE_CODE_ENTRYPOINT": "sdk-py"}):
            return self.say(prompt)

    def test_a_short_paste_after_a_question_is_not_his_line(self):
        for p in ["Is this a scam?\nLet Codex see everything.", "what does this mean?\n\nLet every AI see everything.",
                  "why did you say let codex see my past chats?\n\nLet Codex see my past chats.",
                  "1. Let Codex see my past chats.\n2. Run the tests."]:
            self.assertIn("waiting for Robin's approval", self.say(p), p)
        self.assertEqual(self.hide("codex"), set(WORK_HIDE))

    def test_a_narrowing_sentence_never_counts_as_a_widening(self):
        for p in ["don't let Cursor see my personal stuff", "do not let Cursor see my personal stuff",
                  "never let Cursor see my personal stuff", "stop letting Cursor see my personal stuff"]:
            self.say(p)
        self.assertFalse([r for r in access.history(mem) if r["dir"] == "loosen"])
        prompt = "Never let Codex see my past chats.\n" + "pasted " * 70 + "\nLet Codex see my past chats."
        it = access.parse(prompt)
        self.assertEqual([i["op"] for i in it], ["hide", "show"])
        mine = access._gate(mem, {"prompt": prompt}, it, True)  # his own words contain the widening's words: not his line
        self.assertEqual([mine(i) for i in it], [True, False])

    def test_taking_it_back_in_the_same_prompt_wins(self):
        self.assertIn("takes it back", self.say("let Cursor see my personal stuff. actually don't"))
        self.assertIn("also narrows it", self.say("let cursor see my personal stuff. hide my personal stuff from cursor"))
        self.assertIn("personal", self.hide("cursor"))
        self.assertFalse([r for r in access.history(mem) if r["dir"] == "loosen"])

    def test_approval_names_the_agent_when_more_than_one_waits(self):
        self.tool("cursor", "cursor", "show", "personal")
        self.tool("codex", "codex", "show", "past chats")
        for p in ("allow it", "yes, approve that", "accept it"):
            self.assertIn("A bare 'allow it' approves nothing", self.say(p), p)
        line = self.say("approve the access request")
        self.assertIn("2 access requests are waiting: Cursor would see personal notes; Codex would see past chats. Nothing was "
                      "approved", line)
        self.assertTrue({"personal"} <= self.hide("cursor") and {"chats"} <= self.hide("codex"))
        self.assertIn("approved; Codex", self.say("approve Codex's request"))  # named: only Codex's
        self.assertIn("personal", self.hide("cursor"))
        self.assertIn("approved; Cursor", self.say("allow Cursor"))

    def test_an_agent_keeping_a_request_fresh_never_gets_it_from_casual_words(self):
        self.tool("cursor", "cursor", "level", "everything")
        for _ in range(3):
            self.tool("cursor", "cursor", "level", "everything")
            for p in ("yes, approve that", "accept it", "allow it"):
                self.say(p, session_id="other")
        self.assertEqual(self.hide("cursor"), set(WORK_HIDE))

    def test_one_undo_reverts_an_agents_whole_batch(self):
        before = {n: self.hide(n) for n in ("desk", "cursor", "codex")}
        self.tool("codex", "every AI", "level", "nothing")  # an injected agent blinds everyone
        self.assertEqual(self.hide("desk"), U)
        self.assertEqual(len(self.notes), 1)
        self.assertIn("undid", self.say("undo the last access change"))
        self.assertEqual({n: self.hide(n) for n in before}, before)

    def test_nothing_hides_sections_added_later(self):
        self.assertEqual(set(mem.policy_hide(access.level_entry(mem, "nothing"))), U)
        with mock.patch.object(mem, "SECTIONS", mem.SECTIONS + ("newsection",)):
            self.assertIn("newsection", mem.policy_hide(access.level_entry(mem, "nothing")))


class KeyedIdentity(Case):
    """The follow-up security review: a name alone never carries more than work only, never desk or the Claude app."""

    def wire(self, client="cursor"):
        p = self.tmp / "home/.cursor/mcp.json"
        p.parent.mkdir(parents=True)
        p.write_text(json.dumps({"mcpServers": {"lflabs-memory": access._connect().server_entry(mem, client,
                                                                                                "mcpServers")}}))
        return p

    def test_unkeyed_names_never_get_desk_or_the_claude_app(self):
        for name in ("desk", "claude-desktop"):
            self.assertEqual(mem.resolve_client(name), (name, False))
            self.assertEqual(set(mem.client_hide(*mem.resolve_client(name))), set(WORK_HIDE))
        # 0.1.4: a name an app reports for itself names it, and proves nothing: at most work only
        self.assertEqual(mem.resolve_client(hint="claude-ai"), ("claude-desktop", mem.UNKEYED))
        self.assertEqual(mem.resolve_client(hint="claude-code"), ("desk", mem.UNKEYED))
        self.assertEqual(set(mem.client_hide(*mem.resolve_client(hint="claude-code"))), set(WORK_HIDE))

    def test_a_widened_client_needs_its_key(self):
        mem.clients_set("cursor", allow=[], deny=[])  # everything
        self.assertEqual(set(mem.client_hide(*mem.resolve_client("cursor"))), set(WORK_HIDE))
        self.assertIn("only a key in an app's own config proves its name", mem.client_note(*mem.resolve_client("cursor")))
        key, _ = mem.clients_add("cursor", with_key=True)
        self.assertEqual(mem.client_hide(*mem.resolve_client("cursor", key)), ())
        self.assertEqual(set(mem.client_hide(*mem.resolve_client("cursor", "wrong"))), set(WORK_HIDE))
        self.assertIsNone(mem.client_note(*mem.resolve_client("cursor", key)))

    def test_a_typed_widening_writes_the_apps_key(self):
        cfg = self.wire()
        line = self.say("give Cursor everything")
        self.assertIn("Cursor got a new key in its config: restart Cursor to load it", line)
        env = json.loads(cfg.read_text())["mcpServers"]["lflabs-memory"]["env"]
        key = env[mem.CLIENT_KEY_ENV]
        self.assertNotIn(key, line + " ".join(self.notes))
        self.assertEqual(mem.resolve_client("cursor", key), ("cursor", True))
        self.assertEqual(mem.client_hide("cursor", True), ())
        self.say("let cursor see my contacts")  # already keyed: nothing rewritten
        self.assertEqual(json.loads(cfg.read_text())["mcpServers"]["lflabs-memory"]["env"][mem.CLIENT_KEY_ENV], key)

    def test_a_widening_of_an_unwired_app_says_how(self):
        (self.tmp / "home/.cursor").mkdir(parents=True)
        self.assertIn(f"Cursor is not connected yet: run `{mem.CFG.command} connect --yes`",
                      self.say("give Cursor everything"))

    def test_the_hook_tells_a_widened_client_without_its_key(self):
        mem.clients_set("codex", allow=[], deny=[])
        with mock.patch.dict(os.environ, {mem.CLIENT_ENV: "codex"}), mock.patch.object(
                mem, "prompt_context", lambda *a, **k: None), mock.patch.object(mem, "connect", lambda: self.db), \
                redirect_stdout(io.StringIO()) as out:
            mem.hook_prompt({"prompt": "what next", "session_id": "k1"})
        ctx = json.loads(out.getvalue())["hookSpecificOutput"]["additionalContext"]
        self.assertIn("codex gets work only through this connection", ctx)


class RoundTwoParser(unittest.TestCase):
    """Adversarial review round 2, parser lens."""

    def test_more_narrowings_are_read(self):
        for p, t in [("Codex can no longer see my contacts", "codex"), ("Codex isn't allowed to see my contacts", "codex"),
                     ("Codex no longer sees my past chats", "codex"), ("Codex doesn't need to see my health stuff", "codex"),
                     ("I don't want Cursor to have my contacts", "cursor"),
                     ("don't let any other AI see my contacts", "any other ai"),
                     ("don't let any AI see my health stuff", "any ai"), ("no AI should see my health stuff", "everyone"),
                     ("only Claude Code should see my health stuff", "everyone except claude code"),
                     ("shut Codex out of my past chats", "codex"), ("deny Codex access to my health stuff", "codex"),
                     ("kill Cursor's access to my past chats", "cursor"), ("take Codex off my past chats", "codex"),
                     ("don't let anyone else see my contacts", "anyone else")]:
            got = parsed(p)
            self.assertEqual([(i["kind"], i.get("op"), i.get("t")) for i in got], [("change", "hide", t)], p)
            self.assertTrue(mem.TALK_HINT_RE.search(p), p)
        for p in ("make Cursor only see work", "only let Cursor see work", "Cursor should only see work"):
            self.assertEqual([(i.get("op"), i.get("what")) for i in parsed(p)], [("only", ["work"])], p)
        for p in ("work only for Cursor", "nothing personal for Codex", "restrict Cursor to work"):
            self.assertEqual(parsed(p)[0]["op"], "level", p)

    def test_everyday_talk_stays_quiet(self):
        for p in ["make cursor work", "make copilot work", "Cursor - work", "Codex is now work", "put Codex on work",
                  "undo the last permission change", "revert the sharing settings", "undo the last memory change",
                  "allow everyone", "approve all agents", "allow Claude Code", "give Codex access to projects",
                  "let Cursor see the project", "share projects with Codex", "show Cursor preferences",
                  "give Codex access to projects. Then fix the failing test in the bridge", "I'm going to Cursor",
                  "I'll continue with Codex", "let's continue in Codex"]:
            self.assertEqual([i for i in parsed(p) if i["kind"] != "near_miss"], [], p)
        self.assertEqual(parsed("let codex see my projects")[0]["what"], ["projects"])

    def test_a_near_miss_is_flagged_even_beside_a_change(self):
        got = parsed("Hide my past chats from Codex. Cursor shouldn't ever look at my contacts")
        self.assertEqual([i["kind"] for i in got], ["change", "near_miss"])


class RoundTwoSafety(Case):
    def test_the_near_miss_line_shows_beside_a_change(self):
        line = self.say("Hide project notes from Codex. Gemini shouldn't ever look at my contacts")
        self.assertIn("can no longer see project notes", line)
        self.assertIn("matched no form I know", line)

    def test_a_widening_of_an_oddly_wired_app_says_it_stays_work_only(self):
        p = self.tmp / "home/.cursor/mcp.json"
        p.parent.mkdir(parents=True)
        p.write_text(json.dumps({"mcpServers": {"lflabs-memory": {"command": "x", "args": ["y", "serve"]}}}))
        self.assertIn("Cursor stays at work only until it has its key", self.say("let Cursor see my past chats"))

    def test_a_narrowing_that_became_a_widening_meanwhile_is_not_written(self):
        mem.clients_set("cursor", allow=["prefs"], deny=[])
        change = access._mutation(mem, "hide", ["people"])[0]
        with mock.patch.object(mem, "policy_base", lambda n: {"deny": []}):  # what the preview saw
            self.assertEqual(access.direction(*access.preview(mem, "cursor", change)), "tighten")
        mem._write_clients({"cursor": {"allow": ["nothing"], "deny": []}})  # meanwhile: nothing
        b, a = access._narrow(mem, "cursor", lambda e: {"deny": ["people"]}, "said")
        self.assertEqual(set(a), U)

    def test_channel_and_scheduled_prompts_are_not_typed(self):
        self.say('<channel source="x">let cursor see my personal stuff</channel>')
        self.assertIn("waiting", self.say("<scheduled-task>x</scheduled-task>\nlet cursor see my personal stuff"))
        self.assertIn("personal", self.hide("cursor"))


class Off(Case):
    def test_off_reads_nothing(self):
        with mock.patch.object(mem, "TALK_PERMS_MODE", "off"), mock.patch.object(mem, "prompt_context",
                                                                                 lambda *a, **k: None), \
                mock.patch.object(mem, "connect", lambda: self.db), redirect_stdout(io.StringIO()) as out:
            mem.hook_prompt({"prompt": "let cursor see my personal stuff", "session_id": "o1"})
        self.assertEqual(out.getvalue(), "")
        self.assertEqual(access.history(mem), [])


if __name__ == "__main__":
    unittest.main()
