"""The model behind memory's background jobs: reading transcripts, claims, filing, the weekly tidy, handoffs and
translation. Recall, remember and forget never need one.

Every caller names its job and a tier ("main" or "cheap"); config.toml's [llm] says which backend answers and which
model each tier is:

  claude-cli     the user's Claude plan: `claude -p` (Claude Code MIN_CLAUDE or later), models by alias (haiku, sonnet)
  anthropic-api  the user's API key, from the Keychain (never config, logs or backups): one POST to the Messages API
  none           no model: every call raises Unavailable, and the jobs that need one wait (recall still works)

Unavailable (no backend, not logged in, a usage cap, the daily budget spent) is what callers degrade on: the slice
waits in its queue and is read once a model answers again. Any other failure is a RuntimeError, as before.

Daily budget: each call is counted in memory.db's llm_usage table (day, job, calls, tokens, cost); once today's calls
reach daily_calls (0: no limit) the next one raises Unavailable until tomorrow.

scrub_for_model() masks credential shapes (API keys, tokens, private keys, card numbers, a password's value) in text
before it is sent: the engine's own SCRUB_MODE decides where it runs.
"""
import json
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from . import config

MIN_CLAUDE = "2.1.301"  # every flag below was checked on this version
CHILD_ENV = "LFLABS_MEMORY_CHILD"  # set for our own `claude -p` runs so the hooks stay out
API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"
API_MAX_TOKENS = 16000
KEYCHAIN_SERVICE = "LF Labs Memory"
KEYCHAIN_ACCOUNT = "anthropic-api-key"
API_MODELS = {"haiku": "claude-haiku-5-5", "sonnet": "claude-sonnet-5-5", "opus": "claude-opus-5-5"}
PRICES = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.0, 10.0), "claude-opus-5-5": (5.0, 25.0)}  # list
# $ per million tokens, in and out
COOLDOWN = 300  # seconds no call is tried after "not logged in" or a usage cap, unless the cap said when it resets
NONE_NOTE = "no model backend: recall and remember work; automatic saving, claims, handoffs and translation are off"
LOGIN_NOTE = "Claude Code is not logged in: run claude, then /login"
KEY_NOTE = "no Anthropic API key in the Keychain: run setup again"

# What claude -p (or the API) says when the user is logged out, or out of usage for now
LOGIN_RE = re.compile(r"/login\b|not logged in|invalid api key|authentication_error|OAuth token has expired", re.I)
CAP_RE = re.compile(r"usage_limit_reached|rate.?limit|\b429\b|overloaded|usage.?limit|weekly.?limit|"
                    r"hit your [\w-]* ?limit|limit (?:will )?resets?", re.I)
RESET_RE = re.compile(r"resets?\s+(?:at\s+)?([^.\n|]{1,40})", re.I)


class Unavailable(RuntimeError):
    """No model can answer now: no backend, not logged in, a usage cap or the daily budget. Callers degrade."""


@dataclass(frozen=True)
class Usage:
    calls: int = 0
    in_tok: int = 0
    out_tok: int = 0
    usd: float = 0.0


@dataclass(frozen=True)
class BackendStatus:
    name: str
    ok: bool
    version: str | None = None
    logged_in: bool | None = None
    model_main: str | None = None
    model_cheap: str | None = None
    note: str = ""


def _version_tuple(v):
    m = re.search(r"(\d+)\.(\d+)\.(\d+)", v or "")
    return tuple(int(x) for x in m.groups()) if m else None


# Our own `claude -p` runs: the hooks stay out (CHILD_ENV), and Claude Code sends none of its own telemetry, error
# reports or other non-essential traffic for them (review 2026-11-08: its event logging fired on the engine's jobs).
QUIET_ENV = {"CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", "DISABLE_TELEMETRY": "1", "DISABLE_ERROR_REPORTING": "1"}


# What points Claude Code at another endpoint, provider or model: never passed to the engine's own `claude -p` runs, so
# every model call goes where the privacy text says (the user's Claude plan through Claude Code). The prompt hook's
# translation runs in the hook's own process, with the environment the user gave Claude Code: pointed at Kimi,
# DeepSeek, GLM or Qwen (ANTHROPIC_BASE_URL + ANTHROPIC_AUTH_TOKEN) it sent the prompt there (2026-11-10). Such a call
# now fails (no Claude login: Unavailable) and the hook searches the prompt as it is. Only what Claude Code reads to
# route: AWS_REGION, AWS_PROFILE and the like stay, since without its switches Claude Code does not use them.
ROUTING_PREFIXES = ("ANTHROPIC_", "VERTEX_REGION_CLAUDE_")
ROUTING_VARS = {"CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX", "CLAUDE_CODE_USE_FOUNDRY",
                "CLAUDE_CODE_SKIP_BEDROCK_AUTH", "CLAUDE_CODE_SKIP_VERTEX_AUTH", "CLAUDE_CODE_SKIP_FOUNDRY_AUTH",
                "AWS_BEARER_TOKEN_BEDROCK", "CLOUD_ML_REGION"}


def routed(name):
    """Whether this variable sends Claude Code's requests somewhere else (ROUTING_PREFIXES, ROUTING_VARS)."""
    return name.startswith(ROUTING_PREFIXES) or name in ROUTING_VARS


def child_env(base=None):
    """The environment of the engine's own claude -p runs: the caller's, less what reroutes Claude Code (routed)."""
    env = {k: v for k, v in (os.environ if base is None else base).items() if not routed(k)}
    return {**env, CHILD_ENV: "1", **QUIET_ENV}


class ClaudeCLI:
    """The user's Claude plan, through the claude CLI: the same flags the engine always ran."""
    name = "claude-cli"

    def binary(self):
        """The claude command: config.toml's [llm] claude_path (setup finds it in the user's login shell, where the
        background service's own PATH may not reach), else PATH and the folders Claude Code installs into."""
        stored = getattr(config.current().llm, "claude_path", "") or ""
        for p in (stored, shutil.which("claude"), "/opt/homebrew/bin/claude", Path.home() / ".claude/local/claude",
                  Path.home() / ".local/bin/claude"):
            if p and Path(p).exists():
                return str(p)
        return None

    def version(self):
        claude = self.binary()
        if not claude:
            return None
        try:
            out = subprocess.run([claude, "--version"], capture_output=True, text=True, timeout=20)
        except (OSError, subprocess.SubprocessError):
            return None
        m = re.search(r"\d+\.\d+\.\d+", out.stdout)
        return m.group(0) if m else None

    def status(self, ping=False):
        llm = config.current().llm
        base = dict(name=self.name, model_main=llm.main_model, model_cheap=llm.cheap_model)
        version = self.version()
        if not version:
            return BackendStatus(**base, ok=False, note="Claude Code not found: install it, or choose another backend")
        if _version_tuple(version) < _version_tuple(MIN_CLAUDE):
            return BackendStatus(**base, ok=False, version=version,
                                 note=f"Claude Code {version} is too old: {MIN_CLAUDE} or later is needed (claude update)")
        if not ping:
            return BackendStatus(**base, ok=True, version=version)
        try:  # a one-token answer: the one check that shows the login works
            self.json("Say ok.", {"type": "object", "required": ["ok"], "properties": {"ok": {"type": "boolean"}}},
                      "Answer ok true.", model=llm.cheap_model, timeout=120)
        except Unavailable as e:
            return BackendStatus(**base, ok=False, version=version, logged_in=not LOGIN_RE.search(str(e)), note=str(e))
        except RuntimeError as e:
            return BackendStatus(**base, ok=False, version=version, note=str(e))
        return BackendStatus(**base, ok=True, version=version, logged_in=True)

    def json(self, prompt, schema, system, *, model, timeout):
        claude = self.binary() or "claude"
        with tempfile.TemporaryDirectory() as tmp:
            out = subprocess.run(
                [claude, "-p", "--model", model, "--no-session-persistence", "--strict-mcp-config",
                 "--tools", "", "--setting-sources", "", "--disable-slash-commands",
                 "--system-prompt", system, "--output-format", "json", "--json-schema", json.dumps(schema)],
                input=prompt, capture_output=True, text=True, timeout=timeout, cwd=tmp,
                env=child_env())
        try:
            res = json.loads(out.stdout)
        except json.JSONDecodeError:  # no answer at all: the CLI's own words (no model wrote them)
            said = out.stderr.strip()[-500:] or out.stdout[-500:]
            _refused(said)
            raise RuntimeError(f"claude -p failed: {said}")
        if not isinstance(res, dict):
            raise RuntimeError(f"claude -p failed: {out.stdout[-500:]}")
        if res.get("is_error") or not res.get("structured_output"):
            said = str(res.get("result"))[:500]
            if res.get("is_error") is True:  # the CLI's own error, never a model's answer text: words in a transcript
                _refused(said)                # can make a model write "usage limit reached"
            raise RuntimeError(f"claude -p error: {said}")
        u = res.get("usage") or {}
        return res["structured_output"], Usage(
            1, sum(int(u.get(k) or 0) for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")),
            int(u.get("output_tokens") or 0), float(res.get("total_cost_usd") or 0))


class AnthropicAPI:
    """The user's own API key: the Messages API, the answer forced through one tool whose input is the job's schema."""
    name = "anthropic-api"

    def __init__(self):
        self._key = None

    def key(self):
        if self._key is None:
            try:
                out = subprocess.run(["security", "find-generic-password", "-s", KEYCHAIN_SERVICE, "-a",
                                      KEYCHAIN_ACCOUNT, "-w"], capture_output=True, text=True, timeout=20)
            except (OSError, subprocess.SubprocessError):
                return None
            self._key = out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else None
        return self._key

    def headers(self):
        key = self.key()
        if not key:
            raise Unavailable(KEY_NOTE)
        return {"x-api-key": key, "anthropic-version": API_VERSION, "content-type": "application/json"}

    def status(self, ping=False):
        llm = config.current().llm
        base = dict(name=self.name, model_main=API_MODELS.get(llm.main_model, llm.main_model),
                    model_cheap=API_MODELS.get(llm.cheap_model, llm.cheap_model))
        if not self.key():
            return BackendStatus(**base, ok=False, logged_in=False, note=KEY_NOTE)
        if not ping:
            return BackendStatus(**base, ok=True)
        try:
            self.json("Say ok.", {"type": "object", "required": ["ok"], "properties": {"ok": {"type": "boolean"}}},
                      "Answer ok true.", model=llm.cheap_model, timeout=60)
        except RuntimeError as e:
            return BackendStatus(**base, ok=False, logged_in=not LOGIN_RE.search(str(e)), note=str(e))
        return BackendStatus(**base, ok=True, logged_in=True)

    def json(self, prompt, schema, system, *, model, timeout):
        model = API_MODELS.get(model, model)
        body = {"model": model, "max_tokens": API_MAX_TOKENS,
                "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                "messages": [{"role": "user", "content": prompt}],
                "tools": [{"name": "answer", "description": "Give the answer.", "input_schema": schema}],
                "tool_choice": {"type": "tool", "name": "answer"}}
        req = urllib.request.Request(API_URL, data=json.dumps(body).encode(), headers=self.headers(), method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                res = json.loads(r.read())
        except urllib.error.HTTPError as e:
            said = f"{e.code} {e.read()[:500].decode(errors='ignore')}"
            if e.code in (401, 403):
                raise Unavailable(f"the Anthropic API refused the key ({e.code}): run setup again")
            _refused(said)
            raise RuntimeError(f"Anthropic API error: {said}")
        except (urllib.error.URLError, OSError) as e:
            raise RuntimeError(f"Anthropic API unreachable: {e}")
        out = next((c.get("input") for c in res.get("content") or [] if c.get("type") == "tool_use"), None)
        if not isinstance(out, dict):
            raise RuntimeError(f"Anthropic API gave no answer (stop: {res.get('stop_reason')})")
        u = res.get("usage") or {}
        tin = sum(int(u.get(k) or 0) for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
        tout = int(u.get("output_tokens") or 0)
        pin, pout = PRICES.get(model, (0.0, 0.0))
        return out, Usage(1, tin, tout, (tin * pin + tout * pout) / 1e6)


class NoBackend:
    """No model: recall, remember and forget work; what needs a model waits."""
    name = "none"

    def status(self, ping=False):
        return BackendStatus(name=self.name, ok=True, note=NONE_NOTE)

    def json(self, prompt, schema, system, *, model, timeout):
        raise Unavailable(NONE_NOTE)


BACKENDS = {"claude-cli": ClaudeCLI, "anthropic-api": AnthropicAPI, "none": NoBackend}
_backends = {}
_down = {"until": 0.0, "why": ""}  # after "not logged in" or a usage cap: no call is tried until then (this process)
_lock = threading.Lock()


def _refused(said):
    """Raise Unavailable when a failure says the user is logged out or out of usage: no call is tried for a while."""
    if LOGIN_RE.search(said):
        why = LOGIN_NOTE
    elif CAP_RE.search(said):
        reset = RESET_RE.search(said)
        why = "the model's usage limit is reached" + (f" (resets {reset.group(1).strip()})" if reset else "")
    else:
        return
    _down.update(until=time.time() + COOLDOWN, why=why)
    raise Unavailable(why)


def backend():
    """The backend config.toml names, never another: the API is used only when the user chose it (the default is their
    Claude plan), and a name it does not know is refused, not replaced."""
    name = config.current().llm.backend
    if name not in BACKENDS:
        raise Unavailable(f"unknown model backend {name!r} in config.toml (claude-cli, anthropic-api or none)")
    if name not in _backends:
        _backends[name] = BACKENDS[name]()
    return _backends[name]


def status(ping=False):
    """BackendStatus of the configured backend: what doctor prints. ping: one tiny call, to check the login."""
    try:
        return backend().status(ping=ping)
    except Unavailable as e:
        return BackendStatus(name=config.current().llm.backend, ok=False, note=str(e))


def model_for(tier):
    llm = config.current().llm
    return llm.cheap_model if tier == "cheap" else llm.main_model


# ---------- the daily budget ----------

USAGE_SCHEMA = """create table if not exists llm_usage(day text, job text, calls integer, in_tok integer,
  out_tok integer, usd real, primary key(day, job))"""


def _day(now=None):
    return time.strftime("%Y-%m-%d", time.localtime(time.time() if now is None else now))


def _usage_db(db_path):
    db = sqlite3.connect(str(db_path), timeout=5)
    db.execute(USAGE_SCHEMA)
    return db


def today(db_path=None, now=None):
    """{job: Usage} for today."""
    path = Path(db_path or config.current().paths.db)
    if not path.exists():
        return {}
    db = _usage_db(path)
    try:
        return {r[0]: Usage(*r[1:]) for r in db.execute(
            "select job, calls, in_tok, out_tok, usd from llm_usage where day = ?", (_day(now),))}
    finally:
        db.close()


def _record(db_path, job, usage, now=None):
    try:
        db = _usage_db(db_path)
        try:
            with db:
                db.execute("insert into llm_usage values (?,?,?,?,?,?) on conflict(day, job) do update set"
                           " calls = calls + excluded.calls, in_tok = in_tok + excluded.in_tok,"
                           " out_tok = out_tok + excluded.out_tok, usd = usd + excluded.usd",
                           (_day(now), job, usage.calls, usage.in_tok, usage.out_tok, usage.usd))
        finally:
            db.close()
    except sqlite3.Error:  # bookkeeping never costs the answer
        pass


def unavailable(db_path=None, now=None):
    """Why no call would be made now (no backend, a cooldown, the budget spent), or None."""
    cfg = config.current()
    if cfg.llm.backend == "none":
        return NONE_NOTE
    if time.time() < _down["until"]:
        return _down["why"]
    if cfg.llm.daily_calls and sum(u.calls for u in today(db_path, now).values()) >= cfg.llm.daily_calls:
        return f"daily model budget reached ({cfg.llm.daily_calls} calls): the rest waits until tomorrow"
    return None


def llm_json(prompt, schema, system, *, job="other", tier="main", timeout=600, model=None, db_path=None):
    """One structured answer from the configured backend: a dict matching schema. Raises Unavailable when no model can
    answer now (the caller degrades), RuntimeError when the call failed. model: one run's override (the classify CLI's
    --model); else the tier's model from config."""
    db_path = Path(db_path or config.current().paths.db)
    with _lock:
        if why := unavailable(db_path):
            raise Unavailable(why)
    b = backend()
    try:
        out, usage = b.json(prompt, schema, system, model=model or model_for(tier), timeout=timeout)
    except Unavailable:
        raise
    except RuntimeError:
        _record(db_path, job, Usage(calls=1))  # a call that went out counts against the budget
        raise
    _record(db_path, job, usage)
    return out


# ---------- credential scrub ----------

_KEY_RE = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?(?:-----END [A-Z ]*PRIVATE KEY-----|\Z)"  # PEM blocks, whole
    r"|\bsk-(?:ant-)?[A-Za-z0-9_-]{16,}|\bgh[pousr]_[A-Za-z0-9]{20,}|\bgithub_pat_[A-Za-z0-9_]{20,}"
    r"|\bxox[abprs]-[A-Za-z0-9-]{8,}|\bAKIA[0-9A-Z]{16}\b|\bAIza[0-9A-Za-z_-]{30,}|\b[sr]k_live_[A-Za-z0-9]{16,}"
    r"|\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{5,}"  # JWTs
    r"|\bhf_[A-Za-z0-9]{30,}|\bglpat-[A-Za-z0-9_-]{16,}|\bnpm_[A-Za-z0-9]{30,}|\b[0-9]{8,10}:AA[A-Za-z0-9_-]{30,}",
    re.S)
_CARD_RE = re.compile(r"(?<![\d-])(?:\d[ -]?){12,18}\d(?![\d-])")
_VALUE_RE = re.compile(r"(?i)\b(pass(?:word|code|phrase)|pin|secret|token|api[ _-]?key)(s?\s*(?:[:=]|\bis\b)\s*)"
                       r"([\"'`]?)(?!\[redacted\])([^\s\"'`,;]{2,128})\3")


def _luhn(digits):
    s = sum(int(d) if i % 2 == 0 else (2 * int(d) - 9 if int(d) > 4 else 2 * int(d))
            for i, d in enumerate(reversed(digits)))
    return s % 10 == 0


def scrub_for_model(text):
    """text with credential shapes masked: API keys and tokens, private keys, JWTs, card numbers that pass the Luhn
    check, and the value after password / passcode / pin / secret / token. Phones and emails stay: memories need them."""
    text = _KEY_RE.sub("[redacted]", str(text or ""))
    text = _CARD_RE.sub(lambda m: "[redacted]" if 13 <= len(d := re.sub(r"\D", "", m.group(0))) <= 19 and _luhn(d)
                        else m.group(0), text)
    return _VALUE_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}[redacted]", text)
