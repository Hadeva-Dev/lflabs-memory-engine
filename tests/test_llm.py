"""The model backend (llm.py): claude-cli, anthropic-api and none; the tier decides the model; the daily budget keeps
what it cannot read queued; none mode still recalls and remembers; credentials are masked before text is sent.
The claude and security binaries are fake scripts on PATH that record their arguments; no real model is called.

  .venv/bin/python -m unittest -q test_llm
"""
import _env  # noqa: F401  (first: the test data folder and user)
import io
import json
import os
import shutil
import stat
import tempfile
import time
import unittest
from contextlib import redirect_stderr
from dataclasses import replace
from pathlib import Path
from unittest import mock

from lflabs_memory import config, llm, mem
from test_redact import DBCase

SCHEMA = {"type": "object", "required": ["memories"], "properties": {"memories": {"type": "array"}}}
EMPTY = {"type": "result", "is_error": False, "structured_output": {"memories": []}, "total_cost_usd": 0.0012,
         "usage": {"input_tokens": 900, "cache_read_input_tokens": 100, "output_tokens": 40}}

FAKE_CLAUDE = """#!/usr/bin/env python3
import json, pathlib, sys
here = pathlib.Path(__file__).parent
if sys.argv[1:] == ["--version"]:
    print((here / "version").read_text().strip())
    sys.exit(0)
n = len(list(here.glob("call-*.json")))
(here / f"call-{n:03d}.json").write_text(json.dumps({"argv": sys.argv[1:], "stdin": sys.stdin.read()}))
print((here / "out").read_text())
"""
FAKE_SECURITY = """#!/bin/sh
echo "$@" >> "$(dirname "$0")/security-calls"
cat "$(dirname "$0")/key"
"""


def write_exec(path, text):
    path.write_text(text)
    path.chmod(0o755)


class Fakes(DBCase):
    """A fake claude (and security) first on PATH, and a config this test can change (llm=...)."""

    def setUp(self):
        super().setUp()
        self.bin = Path(tempfile.mkdtemp(prefix="fakebin"))
        self.addCleanup(shutil.rmtree, self.bin, True)
        write_exec(self.bin / "claude", FAKE_CLAUDE)
        write_exec(self.bin / "security", FAKE_SECURITY)
        self.claude(version="2.1.301 (Claude Code)", out=EMPTY)
        (self.bin / "key").write_text("sk-test\n")
        for p in (mock.patch.dict(os.environ, {"PATH": f"{self.bin}:{os.environ['PATH']}"}),
                  mock.patch.dict(llm._backends, clear=True), mock.patch.dict(llm._down, {"until": 0.0, "why": ""})):
            p.start()
            self.addCleanup(p.stop)
        self.set_llm()

    def claude(self, version=None, out=None):
        if version is not None:
            (self.bin / "version").write_text(version + "\n")
        if out is not None:
            (self.bin / "out").write_text(out if isinstance(out, str) else json.dumps(out))

    def set_llm(self, **kw):
        cfg = config.current()
        p = mock.patch.object(config, "CFG", replace(cfg, llm=replace(config.LLMConfig(), **kw),
                                                     paths=replace(cfg.paths, data=self.tmp)))
        p.start()
        self.addCleanup(p.stop)

    def calls(self):
        return [json.loads(p.read_text()) for p in sorted(self.bin.glob("call-*.json"))]


class CLIBackend(Fakes):
    def test_cli_version_floor(self):
        self.claude(version="2.1.200 (Claude Code)")
        st = llm.status()
        self.assertFalse(st.ok)
        self.assertIn("2.1.301", st.note)
        self.claude(version="2.1.301 (Claude Code)")
        self.assertTrue(llm.status().ok)

    def test_cli_backend_not_logged_in(self):
        self.claude(out="Invalid API key · Please run /login")
        with self.assertRaises(llm.Unavailable) as e:
            llm.llm_json("hi", SCHEMA, "sys", job="extract")
        self.assertIn("/login", str(e.exception))
        n = len(self.calls())
        with self.assertRaises(llm.Unavailable):  # no call is tried again for a while
            llm.llm_json("hi", SCHEMA, "sys", job="extract")
        self.assertEqual(len(self.calls()), n)

    def test_usage_cap_unavailable_with_reset(self):
        self.claude(out={"type": "result", "is_error": True, "result": "Claude AI usage limit reached|resets 5pm"})
        with self.assertRaises(llm.Unavailable) as e:
            llm.llm_json("hi", SCHEMA, "sys", job="extract")
        self.assertIn("5pm", str(e.exception))

    def test_model_text_never_reads_as_a_cap(self):
        """A model's answer that says "usage limit reached" (a transcript can make it) is a failed call, not a cap:
        it must not hold every later slice in the queue."""
        self.claude(out={"type": "result", "is_error": False, "result": "Claude AI usage limit reached, please /login"})
        with self.assertRaises(RuntimeError) as e:
            llm.llm_json("hi", SCHEMA, "sys", job="extract")
        self.assertNotIsInstance(e.exception, llm.Unavailable)
        self.assertIsNone(llm.unavailable(self.tmp / "memory.db"))

    def test_other_failures_stay_runtime_errors(self):
        self.claude(out="Traceback: something broke")
        with self.assertRaises(RuntimeError) as e:
            llm.llm_json("hi", SCHEMA, "sys", job="extract")
        self.assertNotIsInstance(e.exception, llm.Unavailable)

    def test_main_tier_model(self):
        self.set_llm(main_model="sonnet", cheap_model="haiku")
        self.assertEqual(llm.llm_json("hi", SCHEMA, "sys", job="extract"), {"memories": []})
        llm.llm_json("hi", SCHEMA, "sys", job="classify", tier="cheap")
        a, b = self.calls()
        self.assertEqual(a["argv"][a["argv"].index("--model") + 1], "sonnet")
        self.assertEqual(b["argv"][b["argv"].index("--model") + 1], "haiku")
        self.assertEqual(a["stdin"], "hi")
        for flag in ("-p", "--no-session-persistence", "--strict-mcp-config", "--disable-slash-commands"):
            self.assertIn(flag, a["argv"])

    def test_engine_jobs_name_their_tier(self):
        """extract asks for the main model, classify for the cheap one: the models come from config, not the code."""
        self.set_llm(main_model="sonnet", cheap_model="haiku")
        seen = []
        real = llm.llm_json
        with mock.patch.object(llm, "llm_json", lambda *a, **k: seen.append((k["job"], k["tier"])) or real(*a, **k)):
            mem.remember(self.db, "Robin waters the ferns on Sundays.", "note", None)
            self.claude(out={**EMPTY, "structured_output": {"items": []}})
            mem.classify(self.db, log=lambda *a: None)
        self.assertIn(("classify", "cheap"), seen)

    def test_usage_recorded(self):
        llm.llm_json("hi", SCHEMA, "sys", job="extract")
        llm.llm_json("hi", SCHEMA, "sys", job="extract")
        u = llm.today(self.tmp / "memory.db")["extract"]
        self.assertEqual((u.calls, u.in_tok, u.out_tok), (2, 2000, 80))
        self.assertAlmostEqual(u.usd, 0.0024)


class APIBackend(Fakes):
    def setUp(self):
        super().setUp()
        self.set_llm(backend="anthropic-api", main_model="haiku")

    def answer(self, body):
        class R(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False
        return R(json.dumps(body).encode())

    def test_api_key_from_keychain(self):
        self.assertEqual(llm.backend().headers()["x-api-key"], "sk-test")
        args = (self.bin / "security-calls").read_text()
        self.assertIn("find-generic-password -s LF Labs Memory -a anthropic-api-key -w", args)

    def test_api_call_shape_and_usage(self):
        sent = []

        def urlopen(req, timeout):
            sent.append((req, json.loads(req.data)))
            return self.answer({"content": [{"type": "tool_use", "name": "answer", "input": {"memories": []}}],
                                "usage": {"input_tokens": 1000, "output_tokens": 100}})
        with mock.patch.object(llm.urllib.request, "urlopen", urlopen):
            self.assertEqual(llm.llm_json("hi", SCHEMA, "sys", job="extract"), {"memories": []})
        req, body = sent[0]
        self.assertEqual(req.full_url, llm.API_URL)
        self.assertEqual(req.get_header("Anthropic-version"), "2023-06-01")
        self.assertEqual(body["model"], "claude-haiku-5-5")
        self.assertEqual(body["tools"][0]["input_schema"], SCHEMA)
        self.assertEqual(body["tool_choice"], {"type": "tool", "name": "answer"})
        self.assertEqual(body["system"][0]["cache_control"], {"type": "ephemeral"})
        self.assertAlmostEqual(llm.today(self.tmp / "memory.db")["extract"].usd, (1000 * 0.10 + 100 * 0.50) / 1e6)

    def test_api_key_never_logged(self):
        def urlopen(req, timeout):
            raise llm.urllib.error.HTTPError(llm.API_URL, 401, "unauthorized", {}, io.BytesIO(b'{"error": "bad"}'))
        err = io.StringIO()
        with mock.patch.object(llm.urllib.request, "urlopen", urlopen), redirect_stderr(err):
            with self.assertRaises(llm.Unavailable) as e:
                llm.llm_json("hi", SCHEMA, "sys", job="extract")
        self.assertNotIn("sk-test", str(e.exception) + err.getvalue())
        (self.tmp / "config.toml").write_text('[llm]\nbackend = "anthropic-api"\n')
        for f in self.tmp.rglob("*"):
            if f.is_file():
                self.assertNotIn(b"sk-test", f.read_bytes(), f)

    def test_no_key_unavailable(self):
        (self.bin / "key").write_text("")
        with self.assertRaises(llm.Unavailable):
            llm.llm_json("hi", SCHEMA, "sys", job="extract")
        self.assertFalse(llm.status().ok)


def turn(text, t):
    return json.dumps({"type": "user", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t)),
                       "message": {"role": "user", "content": text}}) + "\n"


class Budget(Fakes):
    def queue(self, n):
        old = time.time() - 3600
        for i in range(n):
            path = self.tmp / f"t{i}.jsonl"
            path.write_text(turn(f"the staging database for project {i} listens on port 654{i}", old))
            with self.db:
                self.db.execute("insert into sessions(session_id, transcript, cwd, activity, final) values (?,?,?,?,1)",
                                (f"s{i}", str(path), str(self.tmp), old))

    def tick(self):
        with mock.patch.object(mem, "CLAIMS_MODE", "off"):
            mem._save_tick(self.db, lambda *a: None)

    def test_budget_exhausted_keeps_slices(self):
        self.set_llm(daily_calls=1)
        self.queue(3)
        self.tick()
        self.assertEqual(len(self.calls()), 1)
        self.assertEqual(len(mem.pending_sessions(self.db)), 2)
        self.tick()  # still today: nothing more is sent, nothing dropped
        self.assertEqual(len(self.calls()), 1)
        self.assertEqual(len(mem.pending_sessions(self.db)), 2)
        tomorrow = time.time() + 86400
        with mock.patch.object(llm.time, "time", lambda: tomorrow):
            self.tick()
        self.assertEqual(len(self.calls()), 2)  # the next day: oldest first, one a day at this budget

    def test_budget_zero_means_no_limit(self):
        self.set_llm(daily_calls=0)
        self.queue(3)
        self.tick()
        self.assertEqual(len(self.calls()), 3)
        self.assertEqual(mem.pending_sessions(self.db), [])

    def test_claims_slice_kept_when_unavailable(self):
        self.set_llm(backend="none")
        cs = mem._claims_mod()
        cs.migrate(self.db)
        path = self.tmp / "c.jsonl"
        path.write_text(turn("my bike is the blue one in the shed", time.time()))
        cs.queue(mem._self(), self.db, "s9", str(path), 0, 1, False, True)
        cs.claims_tick(mem._self(), self.db, lambda *a: None)
        self.assertEqual(self.db.execute("select count(*) from claim_slices where done is null").fetchone()[0], 1)


class NoneMode(Fakes):
    def setUp(self):
        super().setUp()
        self.set_llm(backend="none")

    def test_none_mode_degrades(self):
        mem.remember(self.db, "The staging database listens on port 6543.", "project", "orchard")
        old = time.time() - 3600
        path = self.tmp / "t.jsonl"
        path.write_text(turn("remember the cache lives in redis", old))
        with self.db:
            self.db.execute("insert into sessions(session_id, transcript, cwd, activity, final) values ('s1',?,?,?,1)",
                            (str(path), str(self.tmp), old))
        mem._save_tick(self.db, lambda *a: None)
        mem.classify(self.db, log=lambda *a: None)
        self.assertEqual(self.calls(), [])
        self.assertEqual(mem.pending_sessions(self.db), ["s1"])  # kept, for a backend later
        self.assertIn("6543", mem.recall_text(self.db, "staging database port"))
        self.assertTrue(llm.status().ok)
        self.assertIn("automatic saving", llm.status().note)

    def test_none_mode_non_english_keyword_recall(self):
        mem.remember(self.db, "o banco de staging usa a porta 6543", "project", "orchard")
        self.assertIn("6543", mem.recall_text(self.db, "porta 6543"))

    def test_none_mode_remember_and_forget(self):
        mid, action = mem.remember(self.db, "Robin prefers tabs over spaces.", "feedback", None)
        self.assertEqual(action, "added")
        mem.forget(self.db, mid)
        self.assertEqual(self.row(mid)["status"], "retired")
        self.assertEqual(self.calls(), [])


class Defaults(unittest.TestCase):
    def test_plain_install_defaults(self):
        """A stranger's install: Haiku for both tiers on their Claude plan, 200 calls a day, credentials masked."""
        self.assertEqual(config.LLMConfig(), config.LLMConfig("claude-cli", "haiku", "haiku", 200))
        self.assertEqual(config.current().profile.llm, {})
        self.assertEqual(mem.SCRUB_MODE, "on")

    def test_config_toml_wins_over_the_profile(self):
        d = Path(tempfile.mkdtemp(prefix="cfg"))
        self.addCleanup(shutil.rmtree, d, True)
        (d / "config.toml").write_text('[llm]\nbackend = "none"\nmain_model = "sonnet"\n')
        with mock.patch.dict(os.environ, {"LFLABS_MEMORY_HOME": str(d)}):
            got = config.load().llm
        self.assertEqual((got.backend, got.main_model, got.cheap_model), ("none", "sonnet", "haiku"))


class Scrub(unittest.TestCase):
    def test_scrub_for_model_masks_keys(self):
        s = llm.scrub_for_model("key sk-ant-api03-AbCdEfGhIjKlMnOpQrStUv card 4111 1111 1111 1111 password: hunter2")
        self.assertNotIn("sk-ant", s)
        self.assertNotIn("4111", s)
        self.assertNotIn("hunter2", s)

    def test_scrub_shapes(self):
        for secret in ("ghp_" + "a" * 36, "xoxb-1234-5678-abcdefgh", "AKIAABCDEFGHIJKLMNOP", "AIza" + "B" * 35,  # gitleaks:allow (fake)
                       "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U",  # gitleaks:allow (fake)
                       "-----BEGIN RSA PRIVATE KEY-----\nMIIEow\n-----END RSA PRIVATE KEY-----"):
            self.assertNotIn(secret, llm.scrub_for_model(f"here: {secret} done"), secret)
        for said, value in (("the passcode is 8812", "8812"), ("token=abc123def", "abc123def"),
                            ("PIN: 0420", "0420"), ("my secret is 'tulip22'", "tulip22")):
            self.assertNotIn(value, llm.scrub_for_model(said), said)

    def test_scrub_keeps_what_memories_need(self):
        for text in ("call Bea on (415) 555-1234 or bea@example.com", "the staging DB is on port 6543",
                     "order 1234 5678 9012 3456 is not a card", "meeting at 2026-10-28 14:30",
                     "we rotate the token every week", "the password policy needs 12 characters"):
            self.assertEqual(llm.scrub_for_model(text), text)


if __name__ == "__main__":
    unittest.main()


# A fake claude that does what the real one does with an inherited endpoint: send the request there (2026-11-10).
FAKE_CLAUDE_ROUTED = """#!/usr/bin/env python3
import json, os, pathlib, sys, urllib.request
here = pathlib.Path(__file__).parent
if sys.argv[1:] == ["--version"]:
    print((here / "version").read_text().strip())
    sys.exit(0)
seen = {k: v for k, v in os.environ.items() if k.startswith(("ANTHROPIC_", "CLAUDE_CODE_USE_", "AWS_BEARER_", "CLOUD_ML"))}
n = len(list(here.glob("call-*.json")))
(here / f"call-{n:03d}.json").write_text(json.dumps({"argv": sys.argv[1:], "env": seen}))
if os.environ.get("ANTHROPIC_BASE_URL"):
    try:
        urllib.request.urlopen(urllib.request.Request(os.environ["ANTHROPIC_BASE_URL"] + "/v1/messages", data=b"{}",
                                                      method="POST"), timeout=5).read()
    except Exception:
        pass
print((here / "out").read_text())
"""
ROUTING = {"ANTHROPIC_BASE_URL": "http://127.0.0.1:9/", "ANTHROPIC_AUTH_TOKEN": "sk-third-party",
           "ANTHROPIC_API_KEY": "sk-ant-x", "ANTHROPIC_MODEL": "kimi-k2", "ANTHROPIC_SMALL_FAST_MODEL": "kimi-k2",
           "ANTHROPIC_DEFAULT_HAIKU_MODEL": "glm-4.6", "ANTHROPIC_DEFAULT_SONNET_MODEL": "qwen3-coder",
           "ANTHROPIC_DEFAULT_OPUS_MODEL": "deepseek-chat", "ANTHROPIC_CUSTOM_HEADERS": "X-Key: y",
           "CLAUDE_CODE_USE_BEDROCK": "1", "CLAUDE_CODE_USE_VERTEX": "1", "CLAUDE_CODE_SKIP_BEDROCK_AUTH": "1",
           "AWS_BEARER_TOKEN_BEDROCK": "t", "CLOUD_ML_REGION": "us-east5", "VERTEX_REGION_CLAUDE_3_5_HAIKU": "us-east5",
           "ANTHROPIC_VERTEX_PROJECT_ID": "p", "ANTHROPIC_BEDROCK_BASE_URL": "https://b.example"}
KEPT = {"PATH": "/usr/bin:/bin", "HOME": "/Users/x", "LANG": "en_US.UTF-8", "AWS_REGION": "us-west-2",
        "AWS_PROFILE": "work", "GOOGLE_CLOUD_PROJECT": "g", "CLAUDE_CODE_OAUTH_TOKEN": "the-users-plan",
        "CLAUDE_CONFIG_DIR": "/Users/x/.claude"}


class ModelEnv(unittest.TestCase):
    def test_child_env_drops_what_reroutes_claude_code_and_keeps_the_rest(self):
        got = llm.child_env({**ROUTING, **KEPT})
        for k in ROUTING:
            self.assertNotIn(k, got, k)
        for k, v in KEPT.items():
            self.assertEqual(got[k], v, k)
        self.assertEqual(got[llm.CHILD_ENV], "1")

    def test_the_daemons_claude_run_is_unchanged(self):
        base = {"PATH": "/usr/bin:/bin", "HOME": "/Users/x", "LANG": "en_US.UTF-8"}
        self.assertEqual(llm.child_env(base), {**base, llm.CHILD_ENV: "1", **llm.QUIET_ENV})


class HookTranslation(Fakes):
    """The prompt hook's translation runs in the hook's own process, with the environment the user gave Claude Code."""

    def setUp(self):
        super().setUp()
        write_exec(self.bin / "claude", FAKE_CLAUDE_ROUTED)
        self.claude(out={"type": "result", "is_error": False, "structured_output": {"english": "how do I deploy this?"},
                         "total_cost_usd": 0.0001, "usage": {"input_tokens": 10, "output_tokens": 5}})
        import http.server
        import threading
        self.hits = []
        hits = self.hits

        class H(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                hits.append(self.path)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"{}")

            def log_message(self, *a):
                pass
        srv = http.server.HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.shutdown)
        self.url = f"http://127.0.0.1:{srv.server_address[1]}"

    def test_a_translation_never_reaches_an_endpoint_claude_code_was_pointed_at(self):
        from lflabs_memory import facets_live
        env = {"ANTHROPIC_BASE_URL": self.url, "ANTHROPIC_AUTH_TOKEN": "sk-third-party",
               "ANTHROPIC_DEFAULT_HAIKU_MODEL": "kimi-k2"}
        with mock.patch.dict(os.environ, env), mock.patch.dict(facets_live._translated, clear=True), \
                mock.patch.object(mem, "connect", lambda: self.db):
            got = facets_live.hook_translate(mem, "这个服务怎么部署？")
        self.assertIn("how do I deploy this?", got)  # the call was made, through the user's Claude plan
        self.assertEqual(self.hits, [])  # and nothing reached the endpoint
        self.assertEqual(self.calls()[-1]["env"], {})
        # the control: the same fake claude with the hook's own environment does reach it
        import subprocess
        with mock.patch.dict(os.environ, env):
            subprocess.run([str(self.bin / "claude"), "-p"], input="x", capture_output=True, text=True, timeout=30)
        self.assertEqual(self.hits, ["/v1/messages"])


class APIBackendEndpoint(APIBackend):
    def test_the_api_key_backend_never_takes_an_inherited_endpoint(self):
        sent = []

        def urlopen(req, timeout):
            sent.append((req.full_url, req.get_header("X-api-key")))
            return self.answer({"content": [{"type": "tool_use", "input": {"memories": []}}],
                                "usage": {"input_tokens": 1, "output_tokens": 1}})
        with mock.patch.dict(os.environ, {"ANTHROPIC_BASE_URL": "https://kimi.example/anthropic",
                                          "ANTHROPIC_API_KEY": "sk-env", "ANTHROPIC_AUTH_TOKEN": "t"}), \
                mock.patch.object(llm.urllib.request, "urlopen", urlopen):
            llm.llm_json("hi", SCHEMA, "sys", job="extract")
        self.assertEqual(sent, [(llm.API_URL, "sk-test")])  # Anthropic's own URL, the key from the Keychain
