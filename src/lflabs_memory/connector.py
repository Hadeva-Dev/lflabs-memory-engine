"""lflabs-memory/1: the engine's side of the connector socket (the connector npm package's, brand.COMMAND-connector,
docs/PROTOCOL.md). Apps reach memory through the connector, a stdio MCP server that forwards each tool call here.

One request, one reply: a JSON object and "\\n" in, one JSON line out ({"ok": true, "text": ...} or {"ok": false,
"error": a sentence the AI can show}). Each connection runs in its own thread with its own store connection, so a
slow call here never holds up the prompt hook on the daemon's own socket; the warm models are shared.

Who asks decides what they see, as for the MCP server (mem.client_from_request): a declared name is held to work only
without its key, desk and claude-desktop never count by a declared name, and Claude Code is known by the name it
reports. On top of the connector's own checks: a request of at most 1 MiB, CALL_TIMEOUT per call, recall at most 20
results and 12,000 characters, remember at most 4,000 characters and REMEMBER_PER_HOUR saves an hour per client, and
forget only of ids that client's recall returned in the last day.

The socket is created 0600 in a 0700 folder, after checking that no engine answers there already (a second engine
on one Mac is refused: AnotherEngine).
"""
import json
import os
import re
import socket
import socketserver
import stat
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from importlib.metadata import PackageNotFoundError, version as _dist_version
from pathlib import Path

from . import brand

PROTOCOL = "lflabs-memory/1"
SOCKET_ENV = "LFLABS_MEMORY_SOCKET"  # the connector reads the same variable
MAX_REQUEST = 1 << 20      # bytes
CALL_TIMEOUT = 15.0        # seconds a call may take before the reply says so
RECALL_K = 20              # results a recall returns at most
RECALL_CHARS = 12000       # characters a recall returns at most
REMEMBER_CHARS = 4000      # characters a save holds at most
REMEMBER_PER_HOUR = 30     # saves an hour, per client
FORGET_WINDOW = 86400      # seconds a recalled id stays forgettable by the client that recalled it
KINDS = ("user", "feedback", "project", "reference", "note")
WRONG_PROTOCOL = f"this engine speaks {PROTOCOL}"
ANOTHER = "another memory engine is running"


class AnotherEngine(RuntimeError):
    """An engine already answers on the connector socket."""


def default_path():
    """LFLABS_MEMORY_SOCKET, else ~/Library/Application Support/LF Labs/memory-connector.sock (the connector's)."""
    env = os.environ.get(SOCKET_ENV)
    return Path(env).expanduser() if env else Path.home() / "Library/Application Support/LF Labs/memory-connector.sock"


def engine_version():
    try:
        return _dist_version("lflabs-memory-engine")
    except PackageNotFoundError:
        return "dev"


def answers(path, timeout=2.0):
    """The ping text of an engine answering on path, or None (nothing there, or not an engine)."""
    try:
        with socket.socket(socket.AF_UNIX) as s:
            s.settimeout(timeout)
            s.connect(str(path))
            s.sendall((json.dumps({"protocol": PROTOCOL, "tool": "ping", "args": {}, "client": {}}) + "\n").encode())
            reply = json.loads(s.makefile().readline() or "null")
    except (OSError, ValueError):
        return None
    return reply.get("text") if isinstance(reply, dict) and reply.get("ok") else None


def claim(path):
    """Make path ready for this engine's socket: its folder 0700, and no live engine there (AnotherEngine). A socket
    file nobody answers on (a crashed engine's) is removed; anything else at the path is refused."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    lf = Path.home() / "Library/Application Support/LF Labs"  # the product's own folder, never one a user named
    if path.parent == lf and lf.stat().st_uid == os.getuid():  # existing too (a real install had it at 0755)
        os.chmod(lf, 0o700)
    if answers(path) is not None:
        raise AnotherEngine(f"{ANOTHER} (it answers on {path}): stop it before starting this one")
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return path
    if not stat.S_ISSOCK(st.st_mode) or st.st_uid != os.getuid():
        raise AnotherEngine(f"{path} is not this user's socket: move it away before starting the engine")
    try:  # another engine's socket may still be answering slowly: one more look before removing it
        with socket.socket(socket.AF_UNIX) as s:
            s.settimeout(0.5)
            s.connect(str(path))
        raise AnotherEngine(f"{ANOTHER} (it accepts connections on {path}): stop it before starting this one")
    except OSError:
        pass
    path.unlink()
    return path


class Limits:
    """Per-client saves in the last hour and ids recalled in the last day (this process's memory)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.saves = {}     # client -> [time, ...]
        self.recalled = {}  # client -> {id text: time}

    def save_ok(self, client, now):
        with self.lock:
            got = [t for t in self.saves.get(client, []) if now - t < 3600]
            if len(got) >= REMEMBER_PER_HOUR:
                self.saves[client] = got
                return False
            self.saves[client] = got + [now]
            return True

    def saw(self, client, ids, now):
        with self.lock:
            seen = {i: t for i, t in self.recalled.get(client, {}).items() if now - t < FORGET_WINDOW}
            seen.update({i: now for i in ids})
            self.recalled[client] = seen

    def may_forget(self, client, mid, now):
        with self.lock:
            t = self.recalled.get(client, {}).get(mid)
            return t is not None and now - t < FORGET_WINDOW


def _err(text):
    return {"ok": False, "error": text}


def _str(args, key, most, need=False):
    v = args.get(key)
    if v is None and not need:
        return None
    if not isinstance(v, str) or (need and not v.strip()):
        raise ValueError(f"{key} must be text")
    if len(v) > most:
        raise ValueError(f"{key} is too long (at most {most} characters)")
    return v


def handle(m, req, limits, now=None):
    """The reply to one request (a dict) from the engine module m. Opens its own store connection."""
    now = time.time() if now is None else now
    if not isinstance(req, dict) or req.get("protocol") != PROTOCOL:
        return _err(WRONG_PROTOCOL)
    tool, args = req.get("tool"), req.get("args") or {}
    if not isinstance(args, dict):
        return _err("args must be an object")
    if tool == "ping":
        return {"ok": True, "text": f"{brand.PRODUCT} {engine_version()}"}
    if tool not in ("recall", "remember", "forget", "access_status"):
        return _err(f"unknown tool {str(tool)[:40]!r}: recall, remember, forget, access_status or ping")
    c = req.get("client") if isinstance(req.get("client"), dict) else {}
    c = {k: c.get(k) for k in ("name", "key", "hint")}
    name, verified = m.client_from_request(c) or ("unknown", False)
    scope, label = m.client_scope(name, verified), m.client_label(name, verified)
    who = name or "unknown"  # what the limits count: the name it gave or reported, verified or not
    try:
        if tool == "access_status":
            note = m.client_note(name, verified) if m.TALK_PERMS_MODE == "on" else None
            return {"ok": True, "text": m.own(m._access().status_text(m, name) + ("\n" + note if note else ""))}
        db = m.connect()
        try:
            if tool == "recall":
                query = _str(args, "query", 2000, need=True)
                k = args.get("k", 8)
                k = max(1, min(RECALL_K, k if isinstance(k, int) and not isinstance(k, bool) else 8))
                section = _str(args, "section", 64)
                shown = []  # the ids recall gave back, never read from its text (which can echo the caller's words)
                text = m.recall_text(db, query, k, section, scope, RECALL_CHARS, label, shown=shown)
                limits.saw(who, [str(i).lower() for i in shown], now)
                note = m.client_note(name, verified) if m.TALK_PERMS_MODE == "on" else None
                return {"ok": True, "text": note + "\n\n" + text if note else text}
            if tool == "remember":
                content = _str(args, "content", REMEMBER_CHARS, need=True)
                kind = args.get("kind") or None  # none: filed where this app can see it again (mem.app_kind)
                if kind is not None and kind not in KINDS:
                    return _err(f"kind must be one of {', '.join(KINDS)}")
                project = _str(args, "project", 100)
                if not limits.save_ok(who, now):
                    return _err(f"not saved: at most {REMEMBER_PER_HOUR} saves an hour from one app; try later")
                mid, text = m.app_remember(db, content, kind, project, via=name, scope=scope,  # screened, filed
                                           verified=verified is True)  # its name in a prompt only when keyed
                if mid is None:  # refused: an error the AI can show
                    return _err(text)
                return {"ok": True, "text": text}
            mid = args.get("id")  # forget
            mid = str(mid).strip() if isinstance(mid, (str, int)) and not isinstance(mid, bool) else ""
            if not re.fullmatch(r"[cC]?\d{1,9}", mid):
                return _err("id must be a memory id as recall shows it (digits, or c and digits)")
            mid = mid.lower()
            if not limits.may_forget(who, mid, now):
                return _err(f"not forgotten: an app may forget only what its own recall showed it in the last day "
                            f"(recall first, then forget {mid})")
            # an app's forget: hidden now, deleted after a day unless the user restores it (mem.app_forget)
            held, text = m.app_forget(db, mid if mid.startswith("c") else int(mid), who, scope, now=now)
            return {"ok": True, "text": text} if held else _err(text)
        finally:
            db.close()
    except ValueError as e:
        return _err(str(e))


def _handler(m, limits, pool):
    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            line = self.rfile.readline(MAX_REQUEST + 1)
            if not line:  # connected and left (a liveness probe)
                return
            if len(line) > MAX_REQUEST:
                reply = _err(f"request too large (at most {MAX_REQUEST} bytes)")
            else:
                try:
                    req = json.loads(line)
                except ValueError:
                    reply = _err("the request is not one JSON object")
                else:
                    job = pool.submit(handle, m, req, limits)
                    try:
                        reply = job.result(timeout=CALL_TIMEOUT)
                    except FutureTimeout:  # the work goes on in its thread; the app is told now
                        reply = _err(f"memory took too long to answer (over {CALL_TIMEOUT:g} s); try again")
                    except Exception as e:  # never a traceback to the app: one line here, a sentence there
                        print(f"connector call failed: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
                        reply = _err("memory could not answer that; try again")
            try:
                self.wfile.write((json.dumps(reply) + "\n").encode())
            except (BrokenPipeError, ConnectionResetError):
                pass
    return Handler


class Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


def serve(m, path=None):
    """Start answering lflabs-memory/1 on path (default_path()) in a background thread, for the engine module m.
    Raises AnotherEngine when an engine answers there already. Returns the server (shutdown() stops it)."""
    path = claim(path or default_path())
    pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="connector")
    old = os.umask(0o177)  # created 0600, never readable by anyone else for a moment
    try:
        srv = Server(str(path), _handler(m, Limits(), pool))
    finally:
        os.umask(old)
    os.chmod(path, 0o600)
    threading.Thread(target=srv.serve_forever, name="connector", daemon=True).start()
    return srv
