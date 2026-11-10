"""Per-agent scopes (CLIENT_SCOPES_MODE): who is asking (a keyed name from the client's own config, a
reported name as a hint only), each client's policy applied by the one scope filter to the hook, recall, recall_full,
recent and search_chats, forget from any client, the counts-only report, the CLI, and off meaning today's behaviour.
Synthetic only: temp databases, a fake embedder and reranker, nothing of Robin's.

  .venv/bin/python -m unittest -q test_clients
"""
import _env  # noqa: F401  (first: the test data folder and user)
import asyncio
import io
import json
import os
import stat
import threading
import unittest
from contextlib import redirect_stdout
from unittest import mock

from lflabs_memory import connector, mem, models
from test_redact import DBCase
from _fakes import fake_rerank

H = str(mem.HOME)
# Every memory mentions the orchard (the keyword search finds them all); the reranker scores each by its own word.
FACTS = {"projects": ("Orchard deploy notes: restart the daemon after a schema change.", "project", "orchard"),
         "people": ("Bea Reis, Robin's sister: the orchard gate code is changing.", "note", None),
         "reference": ("The orchard invoice lives in the shared Drive folder.", "reference", None),
         "health": ("Orchard week: Robin's allergy medication is 10 mg each morning.", "user", None),
         "personal": ("Orchard trip: Robin flies to Lisbon in November.", "user", None),
         "unfiled": ("Orchard notes from a scratch session.", "note", "scratch")}  # a project note: projects
SCORES = {"deploy notes": 3.0, "gate code": 2.8, "invoice": 2.6, "allergy": 2.4, "Lisbon": 2.2, "scratch": 2.0}
PROMPT = "what do you know about the orchard"


class Case(DBCase):
    def setUp(self):
        super().setUp()
        for p in (mock.patch.object(mem, "rerank", fake_rerank(SCORES)),
                  mock.patch.object(mem, "gate", lambda db, prompt, hits, scope=None, memo=None, **k: hits),
                  mock.patch.object(mem, "MEETBRIEF_STATE", self.tmp / "nope.json"),
                  mock.patch.object(mem, "VOICE_SESSION", self.tmp / "voice-session.json"),
                  mock.patch.object(mem, "TELEGRAM_SESSION", self.tmp / "telegram-session.json"),
                  mock.patch.object(mem, "SOCK", self.tmp / "mem.sock"),  # never there: everything in-process
                  mock.patch.object(mem, "CLAIMS_MODE", "off"),
                  mock.patch.dict(mem._adapt, {"state": None, "queue": None, "mtime": None, "saves": {}}),
                  mock.patch.dict(mem._clients, {"key": None, "data": {}}),
                  mock.patch.dict(mem.RESTRICTED),  # client scopes registered here go away after each test
                  mock.patch.dict(os.environ, {"LFLABS_MEMORY_SCOPE": "", mem.CLIENT_ENV: "", mem.CLIENT_KEY_ENV: ""})):
            p.start()
            self.addCleanup(p.stop)
        self.id = {}
        for name, (content, kind, project) in FACTS.items():
            self.id[name] = mem.remember(self.db, content, kind, project, origin="owner_typed")[0]
        self.db.execute("update memories set section = 'personal' where id in (?, ?)",
                        (self.id["health"], self.id["personal"]))
        self.db.execute("update memories set section = 'people' where id = ?", (self.id["people"],))
        self.db.commit()

    def on(self):
        p = mock.patch.object(mem, "CLIENT_SCOPES_MODE", "on")
        p.start()
        self.addCleanup(p.stop)

    def add(self, name):
        with redirect_stdout(io.StringIO()):
            return mem.clients_add(name)[0]

    def shown(self, text):
        """The fact names whose memory ids appear in a tool's or the hook's answer."""
        return {n for n, mid in self.id.items() if f"#{mid} " in (text or "") or f'"id": {mid},' in (text or "")
                or f"#{mid}\n" in (text or "")}

    def ctx(self, client=None, sid="s1"):
        self.db.execute("delete from injections")
        return mem.prompt_context(self.db, PROMPT, sid, f"{H}/orchard", None, True, client)


ALL = set(FACTS)
RESTRICTED_VIEW = {"projects", "reference"}  # an unknown client: nothing personal (people, personal, health, money,
                                             # chats); "unfiled", filed personal by default, is hidden too


class Identity(Case):
    def test_a_keyed_name_is_verified(self):
        key = self.add("codex")
        self.assertEqual(mem.resolve_client("codex", key), ("codex", True))
        self.assertEqual(mem.resolve_client("Codex ", key), ("codex", True))

    def test_a_bad_or_missing_key_is_not(self):
        self.add("codex")
        self.assertEqual(mem.resolve_client("codex", "not-the-key"), ("codex", False))
        self.assertEqual(mem.resolve_client("codex", None), ("codex", False))
        self.assertEqual(mem.resolve_client("cursor", "anything"), ("cursor", False))  # never registered
        self.assertEqual(mem.resolve_client("../desk", "x"), ("unknown", False))

    def test_a_reported_name_is_a_hint_only(self):
        # names only (0.1.4): a reported name never proves a client, so at most work only
        self.assertEqual(mem.resolve_client(hint="claude-code"), ("desk", False))  # KEYLESS_MODE off here
        self.assertEqual(mem.resolve_client(hint="claude-ai"), ("claude-desktop", False))
        self.assertEqual(mem.resolve_client(hint="codex-mcp-client"), ("codex", False))  # a name for the report
        self.assertEqual(mem.resolve_client(hint="whatever"), ("unknown", False))
        self.assertEqual(mem.resolve_client(), ("unknown", False))
        self.add("desk")  # once desk has a key, a client merely calling itself claude-code is not desk
        self.assertEqual(mem.resolve_client(hint="claude-code"), ("desk", False))

    def test_the_key_is_stored_as_a_hash_in_a_private_file(self):
        key = self.add("codex")
        f = self.tmp / "clients.json"
        self.assertEqual(stat.S_IMODE(f.stat().st_mode), 0o600)
        self.assertNotIn(key, f.read_text())
        self.assertEqual(json.loads(f.read_text())["clients"]["codex"]["key_sha256"], mem._key_hash(key))
        again = self.add("codex")  # a second add rotates it
        self.assertEqual(mem.resolve_client("codex", key), ("codex", False))
        self.assertEqual(mem.resolve_client("codex", again), ("codex", True))

    def test_a_broken_file_counts_as_none(self):
        key = self.add("codex")
        (self.tmp / "clients.json").write_text("{not json")
        self.assertEqual(mem.resolve_client("codex", key), ("codex", False))
        self.assertEqual(mem.resolve_client(hint="claude-code"), ("desk", False))


class Policy(Case):
    def test_defaults(self):
        self.assertEqual(mem.client_hide("desk"), ())
        self.assertEqual(mem.client_hide("claude-desktop"), ())
        self.assertEqual(set(mem.client_hide("anything", False)), {"people", "personal", mem.WITHHELD, "chats"})
        self.add("codex")
        self.assertEqual(mem.client_hide("codex"), mem.client_hide("unknown", False))  # a new client starts restricted

    def test_set_allow_and_deny(self):
        self.add("cursor")
        mem.clients_set("cursor", deny=["people", "health"])
        self.assertEqual(set(mem.client_hide("cursor")), {"people", mem.WITHHELD})
        mem.clients_set("cursor", allow=["projects", "prefs"], deny=[])
        hide = set(mem.client_hide("cursor"))
        self.assertNotIn("projects", hide)
        self.assertTrue({"people", "personal", "reference", "about", mem.WITHHELD, "chats"} <= hide)
        with self.assertRaises(ValueError):
            mem.clients_set("cursor", deny=["nonsense"])

    def test_the_unknown_policy_can_be_set(self):
        mem.clients_set("unknown", deny=["people"])
        self.assertEqual(mem.client_hide("someone", False), ("people",))

    def test_off_registers_nothing(self):
        mem.RESTRICTED.pop("client:unknown", None)  # another test's unknown hook caller (with scopes on)
        self.assertIsNone(mem.client_scope("codex", False))
        self.assertEqual(mem.client_scope("codex", False, "telegram"), "telegram")
        self.assertNotIn("client:unknown", mem.RESTRICTED)


class Filtering(Case):
    def setUp(self):
        super().setUp()
        self.on()
        self.key = self.add("codex")

    def test_prompt_context(self):
        self.assertEqual(self.shown(self.ctx()), ALL)  # Claude Code's hook: desk, everything, as today
        self.assertEqual(self.shown(self.ctx(("codex", True))), RESTRICTED_VIEW)
        self.assertEqual(self.shown(self.ctx(("codex", False))), RESTRICTED_VIEW)  # a bad key: the restricted default
        mem.clients_set("codex", deny=["projects"])
        self.assertEqual(self.shown(self.ctx(("codex", True))), ALL - {"projects"})
        self.assertEqual(self.shown(self.ctx(("codex", False))), RESTRICTED_VIEW)  # its own policy needs its key

    def test_recall_recent_and_full_text(self):
        view = mem.client_scope("codex", True)
        self.assertEqual(self.shown(mem.recall_text(self.db, "orchard", scope=view)), RESTRICTED_VIEW)
        self.assertEqual(self.shown(mem.recent_text(self.db, 20, view)), RESTRICTED_VIEW)
        for name, mid in self.id.items():
            out = mem.full_text(self.db, mid, scope=view)
            self.assertEqual(out.startswith("no active memory"), name not in RESTRICTED_VIEW, name)
        self.assertEqual(mem.recall_text(self.db, "orchard", section="people", scope=view),
                         "the people section is not available here")
        self.assertEqual(self.shown(mem.recall_text(self.db, "orchard")), ALL)

    def test_search_chats_is_hidden_from_a_client_that_may_not_see_chats(self):
        with mock.patch.object(mem, "search_chats", lambda *a, **k: [{"x": 1}]):
            self.assertEqual(mem.chats_text(self.db, "q", scope=mem.client_scope("codex", False)),
                             "past conversations are not available here")
            self.assertEqual(json.loads(mem.chats_text(self.db, "q", scope=mem.client_scope("desk", True))), [{"x": 1}])

    def test_the_daemon_applies_the_policy_serve_sends(self):
        req = {"recall": {"query": "orchard", "k": 8, "client": {"name": "codex", "key": self.key}}}
        self.assertEqual(self.shown(mem._daemon_search(self.db, req)), RESTRICTED_VIEW)
        req = {"recall": {"query": "orchard", "k": 8, "client": {"name": "codex", "key": "wrong"}}}
        self.assertEqual(self.shown(mem._daemon_search(self.db, req)), RESTRICTED_VIEW)
        req = {"recall": {"query": "orchard", "k": 8, "client": {"hint": "claude-code"}}}
        self.assertEqual(self.shown(mem._daemon_search(self.db, req)), RESTRICTED_VIEW)  # a reported name: work only

    def test_a_daemon_that_does_not_scope_clients_is_not_used(self):
        import socket
        sock = socket.socket(socket.AF_UNIX)
        sock.bind(str(mem.SOCK))
        sock.listen(1)
        self.addCleanup(sock.close)

        def answer():  # a daemon from before this change (or with the switch off): full results, no "clients"
            conn, _ = sock.accept()
            conn.makefile().readline()
            conn.sendall(b'{"result": "everything"}\n')
            conn.close()
        threading.Thread(target=answer, daemon=True).start()
        self.assertIsNone(mem._ask_daemon("recall", {"query": "q", "client": {"name": "codex"}}))


class Hook(Case):
    """Another agent's prompt hook (Codex's UserPromptSubmit) names itself in its environment."""

    def run_hook(self, env, claude=True):
        """claude: the payload Claude Code sends (its own transcript, its event name); else none to show it."""
        out = io.StringIO()
        projects = self.tmp / "projects"
        payload = {"prompt": PROMPT, "session_id": "h1", "cwd": f"{H}/orchard"}
        if claude:
            payload.update(transcript_path=str(projects / "-orchard" / "h1.jsonl"), hook_event_name="UserPromptSubmit")
        with mock.patch.dict(os.environ, env), redirect_stdout(out), mock.patch.object(mem, "connect", lambda: self.db), \
                mock.patch.object(mem, "CLAUDE_PROJECTS", projects):
            self.db.execute("delete from injections")
            mem.hook_prompt(payload)
        return self.shown(out.getvalue())

    def test_keyed_restricted_and_claude_code_unchanged(self):
        self.on()
        key = self.add("codex")
        self.assertEqual(self.run_hook({mem.CLIENT_ENV: "codex", mem.CLIENT_KEY_ENV: key}), RESTRICTED_VIEW)
        mem.clients_set("codex", deny=[])
        self.assertEqual(self.run_hook({mem.CLIENT_ENV: "codex", mem.CLIENT_KEY_ENV: key}), ALL)
        self.assertEqual(self.run_hook({mem.CLIENT_ENV: "codex", mem.CLIENT_KEY_ENV: "bad"}), RESTRICTED_VIEW)
        self.assertEqual(self.run_hook({}), ALL)  # Claude Code: desk, on its own transcript
        self.assertEqual(self.run_hook({}, claude=False), RESTRICTED_VIEW)  # no name and nothing to show: work only

    def test_off_ignores_the_environment(self):
        self.assertEqual(self.run_hook({mem.CLIENT_ENV: "codex", mem.CLIENT_KEY_ENV: "bad"}), ALL)


class HookEvidence(Case):
    """v0.1.8-rc2 gate (and 0.1.6, 0.1.7): Codex's prompt hook with LFLABS_MEMORY_CLIENT taken out of its command was
    read as Claude Code's and given a personal note. Who calls is read from what the app itself sends (hook_client):
    Claude Code's whole memory only on Claude Code's own transcript; a name can only narrow that."""

    def setUp(self):
        super().setUp()
        self.on()
        self.projects, self.sessions = self.tmp / "projects", self.tmp / "codex-home" / "sessions"
        for p in (mock.patch.object(mem, "CLAUDE_PROJECTS", self.projects),
                  mock.patch.object(mem, "CODEX_SESSIONS", self.sessions)):
            p.start()
            self.addCleanup(p.stop)

    def claude(self, sid="h1", event="UserPromptSubmit"):
        return {"session_id": sid, "transcript_path": str(self.projects / "-orchard" / f"{sid}.jsonl"),
                "hook_event_name": event, "permission_mode": "default", "cwd": f"{H}/orchard"}

    def codex(self, sid="h1", event="UserPromptSubmit"):
        return {"session_id": sid, "transcript_path": str(self.sessions / "2026" / "11" / "10" /
                                                          f"rollout-2026-11-10T09-00-00-{sid}.jsonl"),
                "hook_event_name": event, "permission_mode": "default", "cwd": f"{H}/orchard", "model": "gpt-5.5"}

    def cursor(self, sid="h1", event="beforeSubmitPrompt"):
        return {"conversation_id": sid, "generation_id": "g1", "hook_event_name": event, "session_id": sid,
                "workspace_roots": [f"{H}/orchard"], "transcript_path": str(self.tmp / "cursor" / f"{sid}.jsonl")}

    def prompt(self, payload, env=None):
        out = io.StringIO()
        with mock.patch.dict(os.environ, env or {}), redirect_stdout(out), \
                mock.patch.object(mem, "connect", lambda: self.db):
            self.db.execute("delete from injections")
            mem.hook_prompt({**payload, "prompt": PROMPT})
        return self.shown(out.getvalue())

    def test_codex_without_a_name_is_codex(self):
        self.assertEqual(self.prompt(self.codex()), RESTRICTED_VIEW)

    def test_codex_claiming_claude_code_is_still_codex(self):
        for name in ("desk", "claude-code"):
            self.assertEqual(self.prompt(self.codex(), {mem.CLIENT_ENV: name}), RESTRICTED_VIEW, name)

    def test_codex_with_its_own_key_gets_what_it_was_given(self):
        key = self.add("codex")
        mem.clients_set("codex", deny=[])
        env = {mem.CLIENT_ENV: "codex", mem.CLIENT_KEY_ENV: key}
        self.assertEqual(self.prompt(self.codex(), env), ALL)
        self.assertEqual(self.prompt(self.codex(), {mem.CLIENT_ENV: "desk", mem.CLIENT_KEY_ENV: key}), RESTRICTED_VIEW)

    def test_claude_code_without_a_name_gets_everything(self):
        self.assertEqual(self.prompt(self.claude()), ALL)
        self.assertEqual(self.prompt(self.claude(), {mem.CLIENT_ENV: "desk"}), ALL)

    def test_claude_code_with_its_folder_moved(self):
        cfg = self.tmp / "claude-cfg"
        payload = {**self.claude(), "transcript_path": str(cfg / "projects" / "-orchard" / "h1.jsonl")}
        self.assertEqual(self.prompt(payload), RESTRICTED_VIEW)  # not where Claude Code keeps them
        self.assertEqual(self.prompt(payload, {"CLAUDE_CONFIG_DIR": str(cfg)}), ALL)  # where it does

    def test_claude_code_may_be_narrowed_by_a_name(self):
        self.assertEqual(self.prompt(self.claude(), {mem.CLIENT_ENV: "cursor"}), RESTRICTED_VIEW)

    def test_nothing_to_show_or_another_apps_shape_is_work_only(self):
        self.assertEqual(self.prompt({"session_id": "h1", "cwd": f"{H}/orchard"}), RESTRICTED_VIEW)
        self.assertEqual(self.prompt(self.cursor()), RESTRICTED_VIEW)
        wrong = {**self.claude(), "transcript_path": str(self.projects / "-orchard" / "another.jsonl")}
        self.assertEqual(self.prompt(wrong), RESTRICTED_VIEW)  # not this session's own transcript
        elsewhere = {**self.claude(), "transcript_path": str(self.tmp / "-orchard" / "h1.jsonl")}
        self.assertEqual(self.prompt(elsewhere), RESTRICTED_VIEW)

    def test_the_payloads_tell_the_apps_apart(self):
        self.assertEqual(mem.hook_app(self.claude()), "desk")
        for ev in ("SessionStart", "Stop", "SessionEnd"):
            self.assertEqual(mem.hook_app(self.claude(event=ev)), "desk", ev)
        self.assertEqual(mem.hook_app(self.codex()), "codex")
        self.assertIsNone(mem.hook_app(self.cursor()))
        self.assertIsNone(mem.hook_app({**self.claude(), "hook_event_name": "beforeSubmitPrompt"}))
        self.assertIsNone(mem.hook_app({}))

    def test_a_session_start_without_claude_codes_transcript_gets_no_claude_code_block(self):
        calls = []
        with mock.patch.object(mem, "INSTANT_MODE", "on"), mock.patch.object(mem, "import_files", lambda db: {}), \
                mock.patch.object(mem, "connect", lambda: self.db), \
                mock.patch.object(mem, "_start_block", lambda *a: calls.append(("start", a[2])) or ""), \
                mock.patch.object(mem, "_recent_block", lambda *a: calls.append(("recent", a[2])) or ""), \
                redirect_stdout(io.StringIO()):
            mem.hook_session_start({**self.codex(event="SessionStart"), "source": "startup"})
            mem.hook_session_start({"session_id": "x", "cwd": f"{H}/orchard", "source": "startup"})
            mem.hook_session_start({**self.claude(event="SessionStart"), "source": "startup"})
        self.assertEqual(calls, [("recent", ("codex", False)), ("recent", ("unknown", False)),
                                 ("start", ("desk", True))])


class Off(Case):
    """CLIENT_SCOPES_MODE off: whatever a client says, every answer is today's, and nothing is counted."""

    def test_identical_answers(self):
        self.add("codex")
        base = self.ctx()
        for who in (("codex", False), ("codex", True), ("unknown", False)):
            self.assertEqual(self.ctx(who), base)
            self.assertEqual(mem.client_scope(*who), None)
        self.assertEqual(self.shown(base), ALL)
        req = {"recall": {"query": "orchard", "k": 8, "client": {"name": "codex", "key": "bad"}}}
        self.assertEqual(mem._daemon_search(self.db, req), mem.recall_text(self.db, "orchard"))
        self.assertIsNone(self.db.execute("select name from sqlite_master where name = 'client_recalls'").fetchone())


class Forget(Case):
    """Forget works from every client, keyed or not: on what that client may see."""

    def test_forget_from_any_client(self):
        self.on()
        key = self.add("codex")
        clients = {"desk": ("desk", True), "codex": ("codex", True), "bad key": ("codex", False),
                   "unknown": ("unknown", False), "claude app": ("claude-desktop", True)}
        for label, who in clients.items():
            mid = mem.remember(self.db, f"Orchard ladder note kept for the {label} client to forget.", "project",
                               "orchard", origin="owner_typed")[0]
            out = mem.forget_reply(self.db, mid, mem.client_scope(*who))
            self.assertTrue(out.startswith(f"Deleted #{mid} for good"), (label, out))
        self.assertTrue(key)

    def test_a_restricted_client_cannot_forget_what_it_cannot_see(self):
        self.on()
        view = mem.client_scope("unknown", False)
        self.assertEqual(mem.forget_reply(self.db, self.id["people"], view), f"no active memory #{self.id['people']}")
        self.assertEqual(self.db.execute("select status from memories where id = ?", (self.id["people"],)).fetchone()[0],
                         "active")


class Report(Case):
    def test_counts_only_never_text(self):
        self.on()
        self.add("codex")
        self.ctx(("codex", True))
        mem.recall_text(self.db, "orchard", scope=mem.client_scope("codex", True), client="codex")
        mem.full_text(self.db, self.id["projects"], client="codex")
        mem.recall_text(self.db, "orchard", client="desk")
        out = mem.clients_report(self.db)
        self.assertIn("codex", out)
        self.assertIn("desk", out)
        self.assertIn(f"#{self.id['projects']} x", out)
        for content, _, _ in FACTS.values():
            for w in content.split():
                if len(w) > 4 and w.lower() not in ("orchard", "notes", "session"):
                    self.assertNotIn(w, out)
        cols = {r[1] for r in self.db.execute("pragma table_info(client_recalls)")}
        self.assertEqual(cols, {"client", "via", "memory_id", "n", "last"})
        n = self.db.execute("select sum(n) from client_recalls where client = 'codex' and memory_id = ?",
                            (self.id["projects"],)).fetchone()[0]
        self.assertEqual(n, 3)  # hook, recall, recall_full

    def test_empty_report(self):
        self.assertIn("nothing recalled", mem.clients_report(self.db))


class Cli(Case):
    def run_cli(self, *args):
        out = io.StringIO()
        from lflabs_memory import access  # the user at a Terminal, who typed the app's name (test_app_trust)
        with redirect_stdout(out), mock.patch.object(access, "human_at_terminal", lambda m, n, **k: None):
            try:
                mem._clients_cli(list(args))
            except SystemExit as e:
                return out.getvalue(), e.code
        return out.getvalue(), 0

    def test_add_prints_a_snippet_with_its_key(self):
        out, code = self.run_cli("add", "codex")
        self.assertEqual(code, 0)
        self.assertIn("[mcp_servers.lflabs-memory]", out)
        self.assertIn('"--client", "codex"', out)
        self.assertIn(mem.CLIENT_KEY_ENV, out)
        key = out.split(f'{mem.CLIENT_KEY_ENV} = "')[1].split('"')[0]
        self.assertEqual(mem.resolve_client("codex", key), ("codex", True))
        js = out.split("other JSON configs:\n")[1].split("\n\nOptional")[0]
        self.assertEqual(json.loads(js)["mcpServers"]["lflabs-memory"]["env"][mem.CLIENT_KEY_ENV], key)
        self.assertIn(f"{mem.CLIENT_ENV}=codex {mem.CLIENT_KEY_ENV}={key} ", out)  # the prompt hook's command

    def test_list_and_set(self):
        self.run_cli("add", "cursor")
        out, code = self.run_cli("set", "cursor", "--deny", "people,health", "--allow", "none")
        self.assertEqual(code, 0)
        self.assertIn("hidden health-money, people", out)
        out, _ = self.run_cli()
        self.assertIn("cursor", out)
        self.assertIn("desk", out)
        self.assertIn("unknown", out)
        self.assertEqual(self.run_cli("set", "cursor", "--deny", "bogus")[1], 2)
        self.assertEqual(self.run_cli("add", "unknown")[1], 2)


class Serve(Case):
    """The real MCP server, in memory: the client's reported name and its config's key decide what the tools return."""

    def tools(self, client_name, declared=None, key=None):
        from mcp.client import Client
        from mcp.server.mcpserver import MCPServer
        import mcp_types
        got = {}

        def capture(self_, *a, **k):
            got["server"] = self_
        env = {mem.CLIENT_ENV: declared or "", mem.CLIENT_KEY_ENV: key or ""}
        with mock.patch.object(MCPServer, "run", capture), mock.patch.object(mem, "connect", lambda: self.db), \
                mock.patch.dict(os.environ, env):
            mem.serve()

        async def call():
            out = {}
            async with Client(got["server"], client_info=mcp_types.Implementation(name=client_name, version="1")) as c:
                names = {t.name for t in (await c.list_tools()).tools}
                for tool, args in (("recall", {"query": "orchard"}), ("recent", {}),
                                   ("recall_full", {"id": self.id["people"]})):
                    r = await c.call_tool(tool, args)
                    out[tool] = r.content[0].text
                out["names"] = names
            return out
        with mock.patch.dict(os.environ, env):
            return asyncio.run(call())

    def test_reported_names_and_keys(self):
        self.on()
        key = self.add("codex")
        desk = self.tools("claude-code")
        self.assertEqual(self.shown(desk["recall"]), RESTRICTED_VIEW)  # a reported name: work only (0.1.4)
        self.assertIn("forget", desk["names"])
        codex = self.tools("codex-mcp-client")  # no key in its config: unknown
        self.assertEqual(self.shown(codex["recall"]), RESTRICTED_VIEW)
        self.assertEqual(self.shown(codex["recent"]), RESTRICTED_VIEW)
        self.assertTrue(codex["recall_full"].startswith("no active memory"))
        self.assertIn("forget", codex["names"])
        mem.clients_set("codex", deny=[])
        keyed = self.tools("codex-mcp-client", "codex", key)
        self.assertEqual(self.shown(keyed["recall"]), ALL)
        liar = self.tools("claude-code", "codex", "wrong")  # a declared name with a bad key: never the hint
        self.assertEqual(self.shown(liar["recall"]), RESTRICTED_VIEW)

    def test_tool_descriptions_name_no_one(self):
        """What every app's AI reads about memory: the generic sections, the user as "the user", nothing of a host's."""
        from mcp.client import Client
        from mcp.server.mcpserver import MCPServer
        import mcp_types
        got = {}
        with mock.patch.object(MCPServer, "run", lambda s, *a, **k: got.update(server=s)), \
                mock.patch.object(mem, "connect", lambda: self.db), mock.patch.object(mem, "TALK_PERMS_MODE", "on"), \
                mock.patch.object(mem, "HANDOFF_MODE", "on"), mock.patch.object(mem, "WHATS_UP_MODE", "on"):
            mem.serve()

        async def descriptions():
            async with Client(got["server"], client_info=mcp_types.Implementation(name="claude-code", version="1")) as c:
                return {t.name: t.description for t in (await c.list_tools()).tools}
        docs = asyncio.run(descriptions())
        self.assertIn("projects, prefs, reference,", docs["recall"])
        for name, doc in docs.items():
            self.assertNotRegex(doc, r"(?i)\b(?:his|him|telegram)\b", name)

    def test_the_server_reports_its_version_and_why_a_model_is_refused(self):
        from mcp.client import Client
        from mcp.server.mcpserver import MCPServer
        import mcp_types
        got = {}
        with mock.patch.object(MCPServer, "run", lambda s, *a, **k: got.update(server=s)), \
                mock.patch.object(mem, "connect", lambda: self.db):
            mem.serve()
        why = "embed model refused: /x/models/embed.onnx is missing"

        async def call():
            async with Client(got["server"], client_info=mcp_types.Implementation(name="cursor", version="1")) as c:
                return c.server_info.version, await c.call_tool("recall", {"query": "orchard"})
        with mock.patch.object(mem, "recall_text", side_effect=models.Refused(why)):
            version, r = asyncio.run(call())
        self.assertEqual(version, connector.engine_version())
        self.assertNotEqual(version, "")
        self.assertTrue(r.is_error)
        self.assertIn(why, r.content[0].text)

    def test_off_is_todays_server(self):
        self.add("codex")
        out = self.tools("codex-mcp-client", "codex", "wrong")
        self.assertEqual(self.shown(out["recall"]), ALL)
        self.assertEqual(out["recall"], mem.recall_text(self.db, "orchard"))
        self.assertEqual(out["names"], {"remember", "recall", "recall_full", "forget", "recent", "search_chats"})


if __name__ == "__main__":
    unittest.main()
