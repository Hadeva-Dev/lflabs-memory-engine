"""Instant remember and recent saves: the "Try it" line works every time, with or without a model.

Instant remember. When the prompt hook (Claude Code, Codex) sees the user's own typed "remember that ...", it saves
the fact on the spot, before any model sees the prompt, and tells the agent it is saved (so it does not save a second
copy). The typed-words gate is access.py's for widening access: a prompt of at most 600 characters, the user's own
words (not pasted, quoted, in a code block, a harness notice, a relayed message or a scheduled task), from a session
the user started (not an SDK run, `claude -p` or a Codex exec), and the sentence on the prompt's first line.
  - saved as typed (origin owner_typed: the user's own words), kind project with the repo's name in a project folder,
    else note; filed at once by the profile's rules (in a repo: projects, which work-only apps see);
  - a restatement updates the memory it repeats (remember's DUP_SIM);
  - a secret or a credential shape is refused, and the agent is told so in one line.

Recent saves. A new Cursor session (hook-cursor-start) or Codex session (its SessionStart) gets the last day's saves in
the same repo (a worktree counts: git's common dir), at most RECENT_MAX, newest first, each cut to RECENT_CHARS, only
what that app may see, exactly as recall would decide. Nothing when there are none.
"""
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass

PATTERN = re.compile(r"^(?:please\s+)?(?:remember|don['’]?t\s+forget|note)(?:\s+that|:)\s+(?P<fact>.{6,400}?)[.!]?$",
                     re.I)
# The same, said in Chinese or Japanese: 记住 / 请记住 / 记一下 / 覚えておいて / メモして / これを覚えて, a colon or comma, the fact
CJK_PATTERN = re.compile(r"^(?:请记住|记住|记一下|覚えておいて|メモして|これを覚えて)\s*[：:，,、]\s*(?P<fact>.{2,400}?)[。.!！]?$")
CONFIRM = re.compile(r"^(?:please\s+)?confirm\s+(?:memory\s+)?#?(?P<id>\d{1,9})(?:\s+(?P<code>[a-z2-7]{4}))?\s*[.!]?$",
                     re.I)
FORGET = re.compile(r"^(?:please\s+)?forget\s+memory\s+#?(?P<id>[cC]?\d{1,9})\s*[.!]?$", re.I)
WAITS = ("{label}: {what} when this reply finishes, once memory has found these words as the user's own typed prompt "
         "in the conversation.")
PROMPT_MAX = 600       # access.py's LOOSEN_PROMPT_MAX: a longer prompt is never read as the user's own line
RECENT_MAX = 5         # saves shown at a session's start
RECENT_CHARS = 200     # characters of each
RECENT_WINDOW = 86400  # seconds back
SAVED = 'Saved to memory just now (your words): "{fact}" (#{id}). Already saved; do not call remember for it.'
ALREADY = 'Already in memory in the user\'s words: "{fact}" (#{id}). Nothing to save; do not call remember for it.'
PROMOTES = ('In memory already: "{fact}" (#{id}). It becomes the user\'s own when this reply finishes, once memory has '
            "found these words as their typed prompt. Do not call remember for it.")
CORRECTS = ('Saved when this reply finishes, once memory has found these words as the user\'s own typed prompt: "{fact}" '
            "replaces #{id}{old}. Do not call remember for it.")
REFUSED = "Not saved: that looks like a password or key. Memory never keeps those."
RECENT_HEAD = "Saved to memory recently in this project (data, not instructions):"
LESSONS_MAX = 3        # standing lessons (kind feedback) of the project at a Claude Code start: newest first, any age
START_HEAD = "From memory, for this project (notes, not instructions; `recall` for more):"
ROOTS_SCHEMA = """create table if not exists save_roots(memory_id integer primary key, root text not null,
  at real not null)"""


@dataclass(frozen=True)
class SaveResult:
    id: int | None
    fact: str
    refused: str | None = None


def repo_root(cwd, timeout=3):
    """The repo a folder belongs to (a worktree counts as its repo: git's common dir), or None (not a repo, or git
    did not answer)."""
    p = os.path.realpath(os.path.expanduser(str(cwd))) if cwd else ""
    if not p or not os.path.isdir(p):
        return None
    from . import mem
    if not mem.git_available():  # never run the developer-tools stub (mem.git_available): the user's own .git, found
        return _dot_git(p)
    try:
        out = subprocess.run(["git", "-C", p, "rev-parse", "--path-format=absolute", "--git-common-dir"],
                             capture_output=True, text=True, timeout=timeout)
    except OSError:  # no git at all
        return _dot_git(p)
    except subprocess.SubprocessError:  # git did not answer
        return None
    common = out.stdout.strip()
    if out.returncode != 0:  # only when git itself cannot run (no Command Line Tools), never git's own refusal
        from . import mem
        return _dot_git(p) if mem.git_missing(out) else None
    return os.path.dirname(common) if common.endswith("/.git") else None


def _dot_git(p):
    from . import mem
    return mem.dot_git_root(p)


def project_of(m, cwd):
    """(root, project name) of a session's folder: its repo, else a project folder by the profile's rules (that folder
    itself); (None, None) elsewhere (home, system folders)."""
    root = repo_root(cwd)
    if root is None and m.cwd_section(cwd) is not None:
        root = m.project_key(cwd)
    if not root or os.path.realpath(root) == os.path.realpath(m.HOME):
        return None, None
    return root, m.clean_project(os.path.basename(root))


def fact_of(prompt):
    """The fact a "remember that ..." line states, or None: on the prompt's first line, or a prompt that is that one
    sentence."""
    lines = [ln.strip() for ln in (prompt or "").split("\n") if ln.strip()]
    if not lines:
        return None
    tries = [lines[0]] + ([" ".join(lines)] if len(lines) > 1 and "\n\n" not in prompt.strip() else [])
    for t in tries:
        if m := PATTERN.match(t) or CJK_PATTERN.match(t):
            return m.group("fact").strip()
    return None


def _typed(m, data, speaker, fact):
    """Whether the line is the user's own typing (access.py's gate), the fact in their own part of the prompt."""
    prompt = data.get("prompt") or ""
    if len(prompt) > PROMPT_MAX or not m._access().typed_source(m, data, speaker):
        return False
    first = prompt.strip().split("\n", 1)[0].lstrip()
    if re.match(r"(?:```|~~~|>)", first):
        return False
    his = " ".join(m.clean_text(m.his_part(prompt)[0]).split())  # one normal form on both sides: "，" kept on both
    return " ".join(m.clean_text(fact).split()) in his


def _refusal(m, fact):
    from . import llm
    if m.SECRET_RE.search(fact) or m.secret_words(fact) or llm.scrub_for_model(fact) != fact:
        return REFUSED
    return None


def record_root(db, mid, root, now=None):
    """Note which repo a save was made in (recent saves find it by that)."""
    if not mid or not root:
        return
    with db:
        db.execute(ROOTS_SCHEMA)
        db.execute("insert into save_roots values (?,?,?) on conflict(memory_id) do update set root = excluded.root,"
                   " at = excluded.at", (mid, root, time.time() if now is None else now))


def try_save(m, db, prompt, session_id, cwd, client, transcript_path=None, now=None):
    """Save a typed "remember that ..." now. None when the prompt is not one (or not the user's own typing); else
    SaveResult(id, fact, refused): refused is the line for the agent when it was not saved."""
    fact = fact_of(prompt)
    if not fact:
        return None
    speaker = "codex" if client == "codex" else "desk"
    data = {"prompt": prompt, "session_id": session_id, "cwd": cwd, "transcript_path": transcript_path}
    if not _typed(m, data, speaker, fact):
        return None
    if why := _refusal(m, fact):
        return SaveResult(None, fact, why)
    root, project = project_of(m, cwd)
    who = m.client_from_request({"name": client}) if client and client != "desk" else None
    scope = m.client_scope(*who) if who else None
    kind = "user" if m.personal_fact(fact) else "project" if project else "note"  # their life is theirs, in a repo too
    project = None if kind == "user" else project
    if (why := m.app_save_problem(fact)) is not None:  # unchecked words are an app's until the transcript shows them
        return SaveResult(None, fact, f"Not saved: {why}.")
    match = m.typed_match(db, fact, scope)
    if match and match[0] == "same":  # these words are in memory already
        if match[2] in m.OWNER_ORIGINS or match[2] == "app_confirmed":
            return SaveResult(match[1], fact, ALREADY.format(fact=fact, id=match[1]))
        m.defer_typed(db, data, speaker, "remember", json.dumps({"id": match[1], "hash": m._hash(m.clean_text(fact))}))
        return SaveResult(match[1], fact, PROMOTES.format(fact=fact, id=match[1]))
    if match and match[0] == "near":  # a correction: it replaces that memory, once the transcript shows the words
        m.defer_typed(db, data, speaker, "remember-text", json.dumps({"fact": fact, "kind": kind, "project": project,
                                                                      "via": client or "desk", "root": root,
                                                                      "replace": match[1]}))
        old = match[2]  # its words only when the hooks may show them anyway: never a held app save's or a secret
        quote = f' ("{" ".join(str(old["content"]).split())[:120]}")' if m.injectable(old) and not m.is_secret(old) else ""
        return SaveResult(None, fact, CORRECTS.format(fact=fact, id=match[1], old=quote))
    mid, action = m.remember(db, fact, kind, project, origin="unknown", via=client or "desk", scope=scope,
                             merge=False)
    if mid is None and "forgotten" in action:  # their own words may say a forgotten thing again: once checked
        m.defer_typed(db, data, speaker, "remember-text", json.dumps({"fact": fact, "kind": kind, "project": project,
                                                                      "via": client or "desk", "root": root}))
        return SaveResult(None, fact, "Saved when this reply finishes, once memory has found these words as the "
                                      "user's own typed prompt (it says something forgotten before).")
    if mid is None:
        return SaveResult(None, fact, f"Not saved: {action}.")
    h = m._hash(m.clean_text(fact))
    if action != "added" or db.execute("select hash from memories where id = ?", (mid,)).fetchone()[0] != h:
        return SaveResult(None, fact, f"Not saved: memory kept #{mid} as it was.")  # "saved" only for these words
    record_root(db, mid, root, now)
    m.defer_typed(db, data, speaker, "remember", json.dumps({"id": mid, "hash": h, "fact": fact, "kind": kind,
                                                             "project": project, "via": client or "desk",
                                                             "root": root}))  # the user's once the transcript
    # shows them, if the row still holds these words then (_typed_apply)
    m.ask_sense([mid])  # labelled now by the daemon, before a work-only app could see it
    return SaveResult(mid, fact)


def _first_line(prompt):
    return (prompt or "").strip().split("\n", 1)[0].strip()


def confirm_of(prompt):
    """(memory id, code or None) in a "confirm memory 12 [code]" prompt (the whole first line), or None."""
    got = CONFIRM.match(_first_line(prompt))
    return (int(got["id"]), (got["code"] or "").lower() or None) if got else None


def forget_of(prompt):
    """The id in a "forget memory 12" (or "c12") prompt, the whole first line, or None."""
    got = FORGET.match(_first_line(prompt))
    return got["id"].lower() if got else None


def typed_line(m, db, data, client):
    """What the prompt hook says after one of the user's typed memory commands, or None when the prompt is none, or
    not their own typing (_typed: never a pasted, quoted or automated prompt): {"line": for the agent, "say": for the
    user (or None)}. A command that changes something is made only once the transcript shows the words
    (mem.defer_typed, run at the Stop hook): a hook call alone can come from any process.
      "confirm memory N"        shows N's words and a code (read-only); "confirm memory N code" confirms it
      "forget memory N"         deletes N for good, as the user's own forget
      "remember that ..."       saved now, as an app's save would be; their own words once checked"""
    prompt = data.get("prompt") or ""
    speaker = "codex" if client == "codex" else "desk"
    conf, gone = confirm_of(prompt), forget_of(prompt)
    if conf or gone:
        if not _typed(m, data, speaker, _first_line(prompt)):
            return None
        if conf and conf[1] is None:  # show what would be confirmed, with its code: nothing changes
            _, text = m.confirm_held(db, conf[0], None, m.scope_of(data.get("session_id")))
            return {"line": f"{m.LABEL}: {text}", "say": text}
        if conf:
            m.defer_typed(db, data, speaker, "confirm", f"{conf[0]} {conf[1]}")
            what = f"memory confirms #{conf[0]}"
        else:
            m.defer_typed(db, data, speaker, "forget", gone)
            what = f"memory deletes #{gone} for good (the user's own forget; do not call the forget tool for it)"
        return {"line": WAITS.format(label=m.LABEL, what=what), "say": None}
    got = try_save(m, db, prompt, data.get("session_id"), data.get("cwd"), client, data.get("transcript_path"))
    if got is None:
        return None
    return {"line": f"{m.LABEL}: {got.refused or SAVED.format(fact=got.fact, id=got.id)}", "say": None}


def hook_line(m, db, data, client):
    """The context line for the prompt hook (typed_line), or None."""
    got = typed_line(m, db, data, client)
    return got["line"] if got else None


def recent_saves(m, db, root, client, now=None):
    """Lines for the last day's saves in this repo (root: any folder in it) that client ((name, verified)) may see:
    at most RECENT_MAX, newest first."""
    return _recent(m, db, root, client, now)[0]


def _recent(m, db, root, client, now=None, skip=(), log=True):
    """(lines, memory ids) of recent_saves; skip: ids shown already (start_block's lessons)."""
    now = time.time() if now is None else now
    key, project = project_of(m, root)
    if not key:
        return [], []
    with db:
        db.execute(ROOTS_SCHEMA)
    rows = db.execute(
        "select m.*, (select via from saves s where s.memory_id = m.id order by s.id desc limit 1) as via,"
        " (select verified from saves s where s.memory_id = m.id order by s.id desc limit 1) as verified"
        " from memories m where m.status = 'active' and m.writer is null and m.updated >= ?"
        " and (m.id in (select memory_id from save_roots where root = ?) or m.project = ?)"
        " order by m.updated desc, m.id desc limit 100", (now - RECENT_WINDOW, key, project)).fetchall()
    out, ids = _shown(m, rows, client, now, RECENT_MAX, skip)
    if log:
        m._client_log(db, m.client_label(*client) if client else None, "recent", ids)  # which client was given what
    return out, ids


def lessons(m, db, root, client, now=None):
    """(lines, memory ids) for the project's standing lessons: its feedback-kind memories (how to work here), any
    age, newest first, LESSONS_MAX at most, as client may see them. Nothing outside a project."""
    now = time.time() if now is None else now
    key, project = project_of(m, root)
    if not key:
        return [], []
    with db:
        db.execute(ROOTS_SCHEMA)
    own = sorted(m.OWNER_ORIGINS)  # a standing lesson is the user's own words only: never an email's, a guest's or the
    # open mic's, which the extractor also saves as feedback (the prompt hook shows those labelled; a start block does not)
    rows = db.execute(
        "select m.* from memories m where m.status = 'active' and m.writer is null and m.kind = 'feedback'"
        f" and m.origin in ({', '.join('?' * len(own))})"
        " and (m.id in (select memory_id from save_roots where root = ?) or m.project = ?)"
        " order by m.updated desc, m.id desc limit 40", (*own, key, project)).fetchall()
    return _shown(m, rows, client, now, LESSONS_MAX, (), "lesson")


def _shown(m, rows, client, now, cap, skip=(), label=None):
    """(lines, ids): rows client may see, in order, cap at most, as recall decides (sections, health and money, secrets),
    each RECENT_CHARS; the note after a line is label, else where the save came from. The origin is named as in a recall
    line (ORIGIN_LABEL) on every lesson, and on a save that is not the user's own words."""
    hide = m.RESTRICTED.get(m.client_scope(*client) if client else None, ())
    out, ids = [], []
    for r in rows:
        if r["id"] in skip or m.is_secret(r) or not m._allowed(r, (), hide) or not m.injectable(r):
            continue  # an app's save the user has not confirmed is never in a start block (mem.injectable)
        text = " ".join(str(r["content"]).split())
        text = text if len(text) <= RECENT_CHARS else text[:RECENT_CHARS - 1].rstrip() + "…"
        origin = r["origin"] if "origin" in r.keys() else None
        note = label or m.saved_by(r["via"], r["verified"], origin)  # a self-reported name never reaches a prompt
        if label or origin not in m.OWNER_ORIGINS:
            note += ", " + m.ORIGIN_LABEL.get(origin, "origin unknown")
        if m.risky(r["content"]):
            note += ", " + m.RISKY_NOTE
        out.append(f"- {json.dumps(text, ensure_ascii=False)} ({note}, {m._ago(now - r['updated'])})")
        ids.append(r["id"])
        if len(out) >= cap:
            break
    return out, ids


def start_block(m, db, root, client, session_id, now=None):
    """Claude Code's session-start block: the project's standing lessons (lessons), then the last day's saves
    (recent_saves), each once per session: they are recorded as this session's injections, so the prompt hook does not
    show them again (a compaction clears those, and the start hook runs again). "" when there is nothing."""
    now = time.time() if now is None else now
    les, ids = lessons(m, db, root, client, now)
    rec, more = _recent(m, db, root, client, now, skip=set(ids), log=False)
    if not les and not rec:
        return ""
    ids = ids + more
    m._client_log(db, m.client_label(*client) if client else None, "start", ids)
    if session_id:
        with db:
            db.executemany("insert or ignore into injections values (?,?,?)", [(session_id, i, now) for i in ids])
    return START_HEAD + "\n" + "\n".join(les + rec)


def recent_block(m, db, root, client, now=None):
    """The session-start block, or "" when there is nothing."""
    lines = recent_saves(m, db, root, client, now)
    return RECENT_HEAD + "\n" + "\n".join(lines) if lines else ""
