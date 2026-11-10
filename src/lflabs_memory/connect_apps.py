"""mem.py connect: find the AI apps on this Mac and wire each one to memory (no setup per agent).

Each app's own config gets an MCP server entry that runs `mem.py serve --client <app>`. The name is a label
(KEYLESS_MODE): with it alone an app sees at most "work only". An app the user gave more than that (by saying so, or `mem.py
clients`) needs proof, so connect writes a key into that app's own entry (LFLABS_MEMORY_CLIENT_KEY in its env; clients.json
keeps only the hash) and rotates it when the entry's key is missing or wrong. The user never handles a key. The Claude app's
entry has no --client: it is recognised by the name it reports (claude-ai), as before, and sees everything. Codex also
gets the prompt hook (LFLABS_MEMORY_CLIENT=codex ... hook-prompt) in ~/.codex/hooks.json, which carries handoffs and lets
The user change access by typing in Codex.

  mem.py connect            a dry run: per app, the file it would change and the exact lines it would add
  mem.py connect --yes      apply; every file is copied first (<file>.lflabs-backup-<time>, next to it)
  mem.py connect --only cursor

Safe by construction (adversarial review 2026-10-20):
- only adds, or sets the key in our own entry: an entry named lflabs-memory that runs another mem.py is reported as
  stale and left alone, and nothing else in a file is changed on purpose; a rerun adds nothing;
- a file is changed only when the new text parses and has our entry (checked before anything touches the disk), only
  when it still holds what was read (an app that rewrote it meanwhile wins: nothing is written), never when it is
  read-only, and never when it is not plain JSON (comments, a BOM, duplicate keys) or not valid TOML: then the lines to
  add are printed instead;
- TOML is appended to in the file's own line endings; JSON is loaded and written back whole, keeping keys, their order,
  the indent and the line endings (a compact or odd file comes back pretty-printed; its backup is next to it);
- a symlinked config is written through the link (a hard link is not: the copy at the other name keeps the old text);
- nothing it prints comes from the files and no key is ever printed: only paths and the lines it adds;
- the Claude app rewrites its config when it quits, so while it runs that file is not written: quit it first;
- one odd file never stops the others: each file's problem is reported and the run goes on.
Paths come from each app's own documentation (2026-10): Codex ~/.codex/config.toml [mcp_servers.NAME] and
~/.codex/hooks.json; Cursor ~/.cursor/mcp.json and Windsurf ~/.codeium/windsurf/mcp_config.json and Gemini CLI
~/.gemini/settings.json and the Claude app's claude_desktop_config.json under "mcpServers"; VS Code's user mcp.json under
"servers".
"""
import json
import os
import re
import secrets
import shlex
import shutil
import subprocess
import time
from pathlib import Path

from . import config

SERVER = config.peek().profile.server_name  # our entry's name in every app's config (importing never makes DATA_DIR)
BACKUP = SERVER.removesuffix("-memory") + "-backup"  # <file>.lflabs-backup-<time>
LIB = "Library/Application Support"
NEW_KEY = "<a new key, written with --yes>"  # what a dry run shows in place of a key
# (label, client, folder that shows the app is installed, [(kind, file, key)])
APPS = [("Codex CLI", "codex", ".codex", [("toml", ".codex/config.toml", None), ("hook", ".codex/hooks.json", None)]),
        ("Cursor", "cursor", ".cursor", [("json", ".cursor/mcp.json", "mcpServers")]),
        ("Windsurf", "windsurf", ".codeium/windsurf", [("json", ".codeium/windsurf/mcp_config.json", "mcpServers")]),
        ("Gemini CLI", "gemini", ".gemini", [("json", ".gemini/settings.json", "mcpServers")]),
        ("VS Code (Copilot)", "copilot", f"{LIB}/Code/User", [("json", f"{LIB}/Code/User/mcp.json", "servers")]),
        ("Claude app", "claude-desktop", f"{LIB}/Claude",
         [("json", f"{LIB}/Claude/claude_desktop_config.json", "mcpServers")])]
HINTED = ("claude-desktop",)  # recognised by the name the app reports, never by --client or a key


class Skip(Exception):
    """A file left as it is, with why."""


def claude_running():
    """Whether the Claude desktop app is running (it rewrites its config when it quits)."""
    try:  # ps, not pgrep: pgrep missed it in a sandboxed shell where ps sees it (2026-10-20)
        out = subprocess.run(["ps", "-axo", "comm"], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return True  # unsure: treat as running, so its file is not written
    if out.returncode != 0 or not out.stdout.strip():
        return True
    return any(line.strip().endswith("/Claude.app/Contents/MacOS/Claude") for line in out.stdout.splitlines())


def _paths(m):
    return m.LAUNCHER


def needs_key(m, client):
    """Whether the user gave this client more than work only, so its name alone no longer carries its policy."""
    if client in HINTED or client in ("desk", "unknown"):
        return False
    return not set(m.client_hide(client, True)) >= set(m.WORK_HIDE)


def key_ok(m, client, key):
    want = (m.client_policies().get(client) or {}).get("key_sha256")
    return bool(want and key and m._key_hash(key) == want)


def server_entry(m, client, key, newkey=None):
    py, script = _paths(m)
    entry = {"command": py, "args": [script, "serve"] + ([] if client in HINTED else ["--client", client])}
    if newkey:
        entry["env"] = {m.CLIENT_KEY_ENV: newkey}
    return {"type": "stdio", **entry} if key == "servers" else entry


def hook_command(m, client, newkey=None):
    py, script = _paths(m)
    return (f"{m.CLIENT_ENV}={client} " + (f"{m.CLIENT_KEY_ENV}={shlex.quote(newkey)} " if newkey else "")
            + f"{shlex.quote(py)} {shlex.quote(script)} hook-prompt")


def hook_entry(m, client, newkey=None):
    return {"type": "command", "command": hook_command(m, client, newkey), "timeout": 10}


def toml_block(m, client, newkey=None):
    py, script = _paths(m)
    return (f"[mcp_servers.{SERVER}]\ncommand = {json.dumps(py)}\n"
            f"args = [{json.dumps(script)}, \"serve\", \"--client\", {json.dumps(client)}]\n"
            + (f"env = {{ {m.CLIENT_KEY_ENV} = {json.dumps(newkey)} }}\n" if newkey else ""))


def _read(path):
    """(text with its own line endings, or None for no file). Raises Skip for a file that cannot be read as text."""
    try:
        with open(path, encoding="utf-8", newline="") as f:
            return f.read()
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError) as e:
        raise Skip(f"cannot be read ({type(e).__name__}): left alone")


def _eol(text):
    return "\r\n" if text and "\r\n" in text else "\n"


def _indent(text):
    m = re.search(r"\n([ \t]+)\S", text)
    if not m:
        return 2
    return "\t" if m.group(1).startswith("\t") else len(m.group(1))


def _load_json(text):
    """The object in a JSON config, {} for an empty file. Raises Skip for comments, a BOM, duplicate keys or a
    top-level value that is not an object."""
    if not text or not text.strip():
        return {}
    if text.startswith("﻿"):
        raise Skip("starts with a byte-order mark: left alone; add this by hand")

    def pairs(kv):
        keys = [k for k, _ in kv]
        if len(keys) != len(set(keys)):
            raise Skip("has a key twice: left alone; add this by hand")
        return dict(kv)
    try:
        data = json.loads(text, object_pairs_hook=pairs)
    except ValueError:
        raise Skip("is not plain JSON (comments?): left alone; add this by hand")
    if not isinstance(data, dict):
        raise Skip("is not a JSON object: left alone; add this by hand")
    return data


def _dump(data, original):
    out = json.dumps(data, indent=_indent(original), ensure_ascii=False)
    out = out + "\n" if not original or original.endswith("\n") else out
    return out.replace("\n", "\r\n") if _eol(original) == "\r\n" else out


def _ours(v, script):
    """Whether a server entry runs this memory: its first argument is this mem.py."""
    return isinstance(v, dict) and isinstance(v.get("args"), list) and bool(v["args"]) and v["args"][0] == script


def _container(data, key, what):
    v = data.get(key)
    if v is None and key not in data:
        return {}
    if not isinstance(v, dict):
        raise Skip(f'its "{key}" is not an object: left alone; add {what} by hand')
    return v


def _hook_key(cmd, m):
    mt = re.search(re.escape(m.CLIENT_KEY_ENV) + r"=(\S+)", cmd or "")
    if not mt:
        return None
    try:
        return shlex.split(mt.group(1))[0]
    except ValueError:
        return None


def _with_key(cmd, newkey, m):
    """A hook command with its key set in place: the old key token replaced, or the key put right after the client
    name; everything else in the command (other variables, redirects) kept."""
    tok = f"{m.CLIENT_KEY_ENV}={shlex.quote(newkey)}"
    if re.search(re.escape(m.CLIENT_KEY_ENV) + r"=\S+", cmd):
        return re.sub(re.escape(m.CLIENT_KEY_ENV) + r"=\S+", lambda _: tok, cmd, count=1)
    if re.search(re.escape(m.CLIENT_ENV) + r"=\S+", cmd):
        return re.sub(re.escape(m.CLIENT_ENV) + r"=\S+", lambda mt: mt.group(0) + " " + tok, cmd, count=1)
    return tok + " " + cmd


def _toml_set_key(text, newkey, m):
    """Our [mcp_servers.lflabs-memory] table with its env key set (an inline env table rebuilt with its other keys
    kept, or an env line added). Raises Skip for a shape it does not edit."""
    import tomllib
    by_hand = " (move its env into one inline line, or remove the table and rerun connect)"
    eol = _eol(text)
    s = re.escape(SERVER)
    head = re.search(rf'(?m)^[ \t]*\[[ \t]*mcp_servers\.(?:{s}|"{s}")[ \t]*\][ \t]*\r?$', text)
    if not head:
        raise Skip(f"its {SERVER} table is not a plain [mcp_servers.{SERVER}] header: left alone" + by_hand)
    if re.search(rf'(?m)^[ \t]*\[[ \t]*mcp_servers\.(?:{s}|"{s}")\.env[ \t]*\]', text):
        raise Skip(f"its {SERVER} env is a sub-table: left alone" + by_hand)
    nxt = re.compile(r"(?m)^[ \t]*\[").search(text, head.end())
    end = nxt.start() if nxt else len(text)
    body = text[head.end():end]
    try:
        env = (tomllib.loads(text).get("mcp_servers") or {}).get(SERVER, {}).get("env") or {}
        if not isinstance(env, dict) or not all(isinstance(v, str) for v in env.values()):
            raise Skip(f"its {SERVER} env is not a table of strings: left alone" + by_hand)
        env = {**env, m.CLIENT_KEY_ENV: newkey}
        line = "env = { " + ", ".join(f"{json.dumps(k) if not re.fullmatch(r'[A-Za-z0-9_-]+', k) else k} = "
                                      f"{json.dumps(v)}" for k, v in env.items()) + " }"
    except (TypeError, ValueError, tomllib.TOMLDecodeError):
        raise Skip(f"its {SERVER} env has values I don't write: left alone" + by_hand)
    if re.search(r"(?m)^[ \t]*env[ \t]*=", body):
        if re.search(r"(?m)^[ \t]*env[ \t]*=[ \t]*\{[^\n#]*\}[ \t]*\r?$", body) is None:
            raise Skip(f"its {SERVER} env line spans lines or has a comment: left alone" + by_hand)
        body = re.sub(r"(?m)^[ \t]*env[ \t]*=[^\n]*?(\r?)$", lambda mt: line + mt.group(1), body, count=1)
    else:  # the env line goes right after the table's last line, before any blank lines
        stripped = body.rstrip("\r\n")
        body = stripped + eol + line + eol + (eol if nxt else "")
    return text[:head.end()] + body + text[end:]


def plan_file(m, client, kind, path, key, newkey=None):
    """One file's plan: {"state": add | key | done | stale | error, "note", "snippet", "new", "old"}. newkey: the key
    this app's entries must carry (None: no key needed); an entry whose key already is that key is left alone. Pure:
    reads, never writes."""
    _, script = _paths(m)
    force = False
    try:
        text = _read(path)
        if kind == "toml":
            return _plan_toml(m, client, text, newkey, force, script)
        data = _load_json(text)
        if kind == "hook":
            return _plan_hook(m, client, text, data, path, newkey, force, script)
        servers = _container(data, key, "the entry")
        mine = servers.get(SERVER)
        ours = next((v for v in servers.values() if _ours(v, script)), None)
        if mine is not None and not _ours(mine, script) and ours is None:
            return {"state": "stale", "note": f"its {SERVER} entry runs another mem.py: left alone (remove it and "
                    "rerun to wire this one)"}
        snippet = json.dumps({SERVER: server_entry(m, client, key, newkey)}, indent=2)
        if ours is None:
            new = json.loads(json.dumps(data))
            new.setdefault(key, {})
            if not isinstance(new[key], dict):
                new[key] = {}
            new[key][SERVER] = server_entry(m, client, key, newkey)
            return {"state": "add", "note": f'add {key}."{SERVER}"' + (" with a key" if newkey else ""),
                    "snippet": snippet, "new": _dump(new, text or ""), "old": text}
        have = (ours.get("env") or {}).get(m.CLIENT_KEY_ENV) if isinstance(ours.get("env"), dict) else None
        if newkey and (force or have != newkey):
            new = json.loads(json.dumps(data))
            entry = next(v for v in new[key].values() if _ours(v, script))
            if not isinstance(entry.get("env"), dict):
                entry["env"] = {}
            entry["env"][m.CLIENT_KEY_ENV] = newkey
            return {"state": "key", "note": m.own("give its entry a new key (the user gave it more than work only)"),
                    "snippet": json.dumps({"env": {m.CLIENT_KEY_ENV: newkey}}, indent=2),
                    "new": _dump(new, text or ""), "old": text}
        return {"state": "done", "note": f"already has {SERVER}"}
    except Skip as e:  # the lines to add by hand: with a placeholder, never the key
        shown = NEW_KEY if newkey else None
        return {"state": "error", "note": str(e), "snippet": (toml_block(m, client, shown) if kind == "toml" else
                json.dumps(hook_entry(m, client, shown) if kind == "hook" else
                           {SERVER: server_entry(m, client, key, shown)}, indent=2))}


def _plan_toml(m, client, text, newkey, force, script):
    import tomllib
    data = {}
    if text is not None:
        try:
            data = tomllib.loads(text)
        except tomllib.TOMLDecodeError:
            raise Skip("does not parse as TOML: left alone; add these lines by hand")
    servers = _container(data, "mcp_servers", "the table")
    ours = next((v for v in servers.values() if _ours(v, script)), None)
    if SERVER in servers and not _ours(servers[SERVER], script) and ours is None:
        return {"state": "stale", "note": f"its {SERVER} table runs another mem.py: left alone (remove it and rerun)"}
    if ours is None:
        block = toml_block(m, client, newkey)
        eol = _eol(text)
        new = (text or "") + (eol if text and not text.endswith(("\n", "\r")) else "") + (eol if text else "")
        new += block.replace("\n", eol)
        _check_toml(new)
        return {"state": "add", "note": f"add [mcp_servers.{SERVER}]" + (" with a key" if newkey else ""),
                "snippet": block, "new": new, "old": text}
    have = ours.get("env").get(m.CLIENT_KEY_ENV) if isinstance(ours.get("env"), dict) else None
    if newkey and (force or have != newkey):
        if servers.get(SERVER) is not ours:
            raise Skip(f"its memory table is not named {SERVER}: set the key by hand ({m.CFG.command} clients add NAME "
                       "--key)")
        new = _toml_set_key(text, newkey, m)
        _check_toml(new, newkey, m)
        return {"state": "key", "note": m.own("give its table a new key (the user gave it more than work only)"),
                "snippet": f"env = {{ {m.CLIENT_KEY_ENV} = {json.dumps(newkey)} }}\n", "new": new, "old": text}
    return {"state": "done", "note": f"already has {SERVER}"}


def _check_toml(new, newkey=None, m=None):
    import tomllib
    try:
        d = tomllib.loads(new)
    except tomllib.TOMLDecodeError:
        raise Skip("would not parse as TOML with the table added: left alone; add these lines by hand")
    entry = (d.get("mcp_servers") or {}).get(SERVER)
    if not isinstance(entry, dict) or newkey and (entry.get("env") or {}).get(m.CLIENT_KEY_ENV) != newkey:
        raise Skip("would not hold the table as written: left alone; add these lines by hand")


def _plan_hook(m, client, text, data, path, newkey, force, script):
    import tomllib
    try:  # Codex also takes hooks inline in config.toml: one there already counts (two would both run)
        inline = tomllib.loads(_read(path.parent / "config.toml") or "").get("hooks") or {}
    except (tomllib.TOMLDecodeError, Skip):
        inline = {}
    if script in json.dumps(inline.get("UserPromptSubmit") or []):
        if newkey and not any(_hook_key(c, m) == newkey for c in re.findall(r'"command": "([^"]*)"',
                                                                          json.dumps(inline["UserPromptSubmit"]))):
            raise Skip("its prompt hook is inline in config.toml, where I don't set keys: move it to hooks.json (or "
                       "remove it) and rerun connect")
        return {"state": "done", "note": "already has the prompt hook (in config.toml)"}
    hooks = _container(data, "hooks", "the hook")
    groups = hooks.get("UserPromptSubmit", [])
    if not isinstance(groups, list):
        raise Skip("has hooks in a shape I don't know: left alone; add the hook by hand")
    for gi, g in enumerate(groups):
        for hi, h in enumerate((g.get("hooks") or []) if isinstance(g, dict) else []):
            cmd = h.get("command") if isinstance(h, dict) else None
            if not (isinstance(cmd, str) and script in cmd and "hook-prompt" in cmd):
                continue
            if newkey and (force or _hook_key(cmd, m) != newkey):
                new = json.loads(json.dumps(data))
                new["hooks"]["UserPromptSubmit"][gi]["hooks"][hi]["command"] = _with_key(cmd, newkey, m)
                return {"state": "key", "note": "give the hook a new key (Codex asks you to trust the changed hook "
                        "once: /hooks)", "snippet": json.dumps(hook_entry(m, client, newkey), indent=2),
                        "new": _dump(new, text or ""), "old": text}
            return {"state": "done", "note": "already has the prompt hook"}
    new = json.loads(json.dumps(data))
    if not isinstance(new.get("hooks"), dict):
        new["hooks"] = {}
    new["hooks"].setdefault("UserPromptSubmit", []).append({"hooks": [hook_entry(m, client, newkey)]})
    return {"state": "add", "note": "add a UserPromptSubmit hook (Codex asks you to trust it once: /hooks)",
            "snippet": json.dumps(hook_entry(m, client, newkey), indent=2), "new": _dump(new, text or ""), "old": text}


def write(path, new, old, secret=False):
    """Back up and replace one file: through any symlink, keeping its mode, only if it still holds `old` (what the plan
    read) and is not read-only. A file that will hold a key (secret) is made 0600 (a stricter mode is kept, never
    loosened), and every backup is at most 0600: it may hold an old key or the app's own tokens. The backup's path, or
    None for a new file. Raises Skip, and leaves no temp file."""
    real = Path(os.path.realpath(path))
    mode, backup = 0o600, None
    tmp = real.with_name(real.name + ".lflabs-tmp")
    try:
        if real.exists():
            st = real.stat()
            if not st.st_mode & 0o200:
                raise Skip("is read-only: left alone")
            mode = st.st_mode & 0o777
        if secret:
            mode &= 0o600
        if _read(real) != old:
            raise Skip("changed while I was reading it (the app wrote it?): left alone; run connect again")
        if old is not None:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            backup, n = real.with_name(f"{real.name}.{BACKUP}-{stamp}"), 1
            while backup.exists():
                backup, n = real.with_name(f"{real.name}.{BACKUP}-{stamp}-{n}"), n + 1
            shutil.copy2(real, backup)
            os.chmod(backup, real.stat().st_mode & 0o600)
        real.parent.mkdir(parents=True, exist_ok=True)
        with os.fdopen(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w", encoding="utf-8",
                       newline="") as f:
            f.write(new)
        os.chmod(tmp, mode)
        if _read(real) != old:  # last look: an app that wrote it since wins
            raise Skip("changed while I was writing it: left alone; run connect again")
        os.replace(tmp, real)
    except (OSError, UnicodeEncodeError) as e:
        raise Skip(f"not written ({type(e).__name__}); the file is as it was")
    finally:
        tmp.unlink(missing_ok=True)
    return backup


def _entry_keys(m, kind, path, key):
    """The keys our entries in one file carry now (read only; [] for anything odd)."""
    _, script = _paths(m)
    try:
        text = _read(path)
        if kind == "toml":
            import tomllib
            servers = tomllib.loads(text or "").get("mcp_servers") or {}
            envs = [v.get("env") for v in servers.values() if _ours(v, script)]
            return [e.get(m.CLIENT_KEY_ENV) for e in envs if isinstance(e, dict) and e.get(m.CLIENT_KEY_ENV)]
        data = _load_json(text)
        if kind == "hook":
            groups = (data.get("hooks") or {}).get("UserPromptSubmit") or [] if isinstance(data.get("hooks"), dict) else []
            cmds = [h.get("command") for g in groups if isinstance(g, dict) for h in (g.get("hooks") or [])
                    if isinstance(h, dict) and isinstance(h.get("command"), str) and script in h["command"]]
            return [k for k in (_hook_key(c, m) for c in cmds) if k]
        servers = data.get(key) if isinstance(data.get(key), dict) else {}
        envs = [v.get("env") for v in servers.values() if _ours(v, script)]
        return [e.get(m.CLIENT_KEY_ENV) for e in envs if isinstance(e, dict) and isinstance(e.get(m.CLIENT_KEY_ENV), str)]
    except Exception:
        return []


def _short(path, home):
    s = str(path)
    return "~" + s[len(str(home)):] if s.startswith(str(home) + "/") else s


def wire_app(m, label, client, files, home, apply, out, keys_only=False):
    """Plan (and with apply, write) one app's files. Returns (errors, [paths written], key registered or None).

    A client that needs a key (the user gave it more than work only) gets one in every entry of the app. A key one of its
    entries already holds and that is still registered is reused, so a file that cannot be written never loses a key
    that works (adversarial review round 2). Only when no entry holds a valid key is a new one made, and its hash is
    registered once at least one entry carries it: entries that could not be written were not working anyway."""
    newkey = reused = None
    if needs_key(m, client):
        held = [k for kind, rel, key in files for k in _entry_keys(m, kind, home / rel, key)]
        reused = next((k for k in held if key_ok(m, client, k)), None)
        newkey = reused or secrets.token_urlsafe(24)
    plans = [(kind, home / rel, key, plan_file(m, client, kind, home / rel, key, newkey)) for kind, rel, key in files]
    errors, written = 0, []
    try:
        for i, (kind, path, key, p) in enumerate(plans):
            if keys_only and p["state"] == "add":
                p = {"state": "skip", "note": f"not connected ({m.CFG.command} connect wires it)"}
            if p["state"] in ("add", "key") and client == "claude-desktop" and claude_running():
                p = {"state": "blocked", "note": "the Claude app is running and rewrites this file when it quits: quit "
                     f"it (Cmd+Q), then run `{m.CFG.command} connect --yes --only claude-desktop` from Terminal",
                     "snippet": p.get("snippet")}
            if p["state"] in ("add", "key") and apply:
                try:
                    backup = write(path, p["new"], p["old"], secret=m.CLIENT_KEY_ENV in p["new"])
                    written.append((kind, path))
                    p["note"] = re.sub(r"^(add|give)", lambda mt: {"add": "added", "give": "gave"}[mt.group(1)],
                                       p["note"]) + (f" (backup: {backup.name})" if backup else " (new file)")
                except Skip as e:
                    p = {"state": "error", "note": str(e), "snippet": p.get("snippet")}
            out(f"{label if i == 0 else '':<19}{_short(path, home)}: {p['note']}")
            if p.get("snippet") and (p["state"] in ("error", "blocked") or not apply):
                shown = p["snippet"].replace(newkey, NEW_KEY) if newkey else p["snippet"]
                out("\n".join(" " * 21 + line for line in shown.splitlines()))
            errors += p["state"] == "error"
    finally:  # proof for the wider policy: only the hash is kept (clients.json), and only of a key some entry holds
        if newkey and not reused and written:
            m.clients_add(client, with_key=True, key=newkey)
    return errors, written, bool(newkey and not reused and written)


def main(m, apply=False, only=None, home=None, out=print):
    home = Path(home or Path.home())
    only = (only or "").strip().lower() or None
    say = out
    out = lambda s: say(re.sub(r"(LFLABS_MEMORY_CLIENT_KEY(?:\"?\s*[=:]\s*\"?|=))[A-Za-z0-9_-]{20,}", r"\1" + NEW_KEY,
                               s))  # noqa: E731  (belt and braces: a key never reaches the screen)
    out(f"{m.LABEL}: connecting the AI apps on this Mac" + ("" if apply else
        " (a dry run: nothing is written; add --yes to apply)") + "\n")
    errors, matched = 0, False
    if not only or only in ("desk", "claude-code", "claude code"):
        matched = True
        try:
            settings = _read(home / ".claude/settings.json") or ""
        except Skip:
            settings = ""
        wired = "hook-prompt" in settings and ("mem.py" in settings or "lflabs-memory" in settings)
        out(f"{'Claude Code':<19}" + ("connected (its hooks in ~/.claude/settings.json)" if wired else
                                      "not wired: its hooks go in ~/.claude/settings.json (README: Claude Code)"))
    for label, client, folder, files in APPS:
        if only and only not in (client, label.lower(), label.lower().split()[0]):
            continue
        matched = True
        if not (home / folder).is_dir():
            out(f"{label:<19}not installed")
            continue
        try:
            errors += wire_app(m, label, client, files, home, apply, out)[0]
        except Exception as e:  # one app's surprise never stops the rest
            errors += 1
            out(f"{label:<19}skipped: {type(e).__name__}")
    if not only and (home / LIB / "com.openai.chat").is_dir():
        out(f"{'ChatGPT':<19}no local MCP config to write (its connectors are remote): skipped")
    if not matched:
        out(f"no app called {only!r}; one of: {', '.join(c for _, c, _, _ in APPS)}, claude-code")
        return 1
    out(m.own("\nEach app starts at \"work only\" (the Claude app: everything). An app the user gave more gets a key in its "
              "own entry. Restart an app to load it. To change what one sees, say it in any AI: \"give Cursor "
              "everything\", \"what can Codex see?\"."))
    return 1 if errors else 0


def ensure_key(m, client, home=None):
    """After the user widened a client beyond work only: give its already-wired entries a key (none is added where the app is
    not connected). One plain sentence for the reply, or None when nothing was needed."""
    if not needs_key(m, client):
        return None
    app = next((a for a in APPS if a[1] == client), None)
    if app is None:
        return (f"{client} has no config I know, so it stays at work only until it has a key "
                f"({m.CFG.command} clients add {client} --key)")
    label, _, folder, files = app
    home = Path(home or Path.home())
    if not (home / folder).is_dir():
        return None  # not on this Mac: nothing to key
    lines = []
    errors, written, keyed = wire_app(m, label, client, files, home, True, lines.append, keys_only=True)
    if keyed or written:
        return (f"{label} got a new key in its config: restart {label} to load it"
                + ("; Codex asks to trust the changed hook once (/hooks)" if any(k == "hook" for k, _ in written)
                   else ""))
    if any("not connected" in x for x in lines) and not errors:
        return f"{label} is not connected yet: run `{m.CFG.command} connect --yes`"
    if errors or any("running" in x for x in lines) or not any(key_ok(m, client, k) for kind, rel, key in files
                                                              for k in _entry_keys(m, kind, home / rel, key)):
        return f"{label} stays at work only until it has its key: run `{m.CFG.command} connect --yes`"
    return None
