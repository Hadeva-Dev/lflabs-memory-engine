"""Changing what each agent may see by just saying it (mem.py TALK_PERMS_MODE), and the handoff phrases.

The one safety rule. TIGHTENING (an agent sees less) applies at once from any source: a sentence in any prompt, an
agent's access_change tool call, the CLI. WIDENING (an agent sees more) applies only from the user's own typed words, read by
the prompt hook before any model saw them (Claude Code's hook, or Codex's for a session the user started): a sentence of at
most LOOSEN_SENTENCE_MAX characters, equal to one of the sentences of their own part of the prompt (not pasted or
quoted), on the prompt's first line or in a prompt that holds nothing but access sentences, not a list item, in a
prompt under LOOSEN_PROMPT_MAX characters, and not taken back in the same prompt (a narrowing of the same agent, or
"actually don't", wins). Then the user gets a Mac notification ("Cursor can now see meetings · say 'undo that'"). A widening
from anywhere else (an agent's tool call, pasted text, a long prompt, voice or Telegram) never applies by itself: it
becomes a pending request (PENDING_TTL), the user is notified ("Cursor asked to see meetings"), and only their typed approval
applies it, naming the agent ("approve Cursor's request"). No tool can approve, and a bare "allow it" never does: it
only lists what is waiting. Approving and undoing a narrowing are widenings, so they too need their typed words.

A client widened beyond work only needs proof that it is that client (KEYLESS_MODE: a name alone carries at most work
only), so a typed widening also gives the app's wired entry a key (connect_apps.ensure_key); the user never handles it.

The parser is plain code, no model: a strict grammar of a verb, a known agent and a section or level, matched against
whole sentences (fullmatch), so "let me see the email you drafted" or "Codex couldn't see the meeting link" never
match. Widening takes fewer words than narrowing ("full access", "memory", bare "work" never widen). Lines inside code
fences, quoted lines and anything that looks like code are skipped.

Files (all 0600, gitignored, under memory/): clients.json (the policies, mem.py), access-history.jsonl (every change:
time, client, hidden sections before and after, source, session and batch; never memory text), access-requests.json
(pending widenings), clients.lock (one writer at a time). Notifications carry agent and section names only.

Every function takes `m`, the mem module as it runs (as claimstore.py does).
"""
import fcntl
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from . import brand, config

PROFILE = config.current().profile  # its section words (profiles/)
LABEL = PROFILE.label or brand.PRODUCT  # the banners' title

LOOSEN_PROMPT_MAX = 600     # a widening applies only from a prompt this short
LOOSEN_SENTENCE_MAX = 120   # and a sentence this short
PENDING_TTL = 24 * 3600     # a pending widening expires after this
PENDING_MAX = 20            # requests kept at most (the oldest go first)
PER_REQUESTER = 5           # requests one agent (or one kind of message) may have waiting
UNDO_BARE = 10 * 60         # "undo that" undoes only a change this fresh, made in the same session
MAX_INTENTS = 6             # access sentences read from one prompt

# ---------- words ----------

AGENTS = {"codex": ("codex", "codex cli", "the codex app", "codex app", "openai codex"),
          "cursor": ("cursor", "the cursor app", "cursor app"),
          "claude-desktop": ("claude app", "the claude app", "claude desktop", "the claude desktop app",
                             "claude desktop app"),
          "windsurf": ("windsurf",),
          "gemini": ("gemini", "gemini cli"),
          "copilot": ("copilot", "github copilot", "vs code", "vscode", "vs code copilot"),
          "chatgpt": ("chatgpt", "chat gpt", "the chatgpt app"),
          "desk": ("claude code",),
          "unknown": ("new agents", "new ais", "new apps", "unknown agents", "unknown ais", "any new agent",
                      "any new ai", "other apps")}
NAMES = {"desk": "Claude Code", "codex": "Codex", "cursor": "Cursor", "claude-desktop": "the Claude app",
         "windsurf": "Windsurf", "gemini": "Gemini", "copilot": "Copilot", "chatgpt": "ChatGPT",
         "unknown": "new and unknown agents"}
ALL_WORDS = ("everyone", "everybody", "every ai", "every agent", "all ais", "all agents", "all the ais", "all the agents",
             "all my ais", "all my agents", "all of them", "every app", "any ai", "any agent", "anyone", "any of my ais")
OTHER_WORDS = ("every other ai", "every other agent", "all other ais", "all the other ais", "all other agents",
               "all the other agents", "everyone else", "the other ais", "the other agents", "other ais", "other agents",
               "any other ai", "any other agent", "anyone else")
# Section words (WITHHELD is health and money together: one pseudo-section). Strong words need no "my"; weak ones are
# everyday words that name a section only after "my" ("my messages", not "the messages"). "*" is everything, "work" the
# work-only sections. NARROW_ONLY words are read only where they narrow: "don't let Codex see anything" yes, "give Codex
# full access" never (round 1 review: sandbox and repo talk).
SCOPE_STRONG = PROFILE.scope_strong
SCOPE_WEAK = PROFILE.scope_weak
NARROW_ONLY = {"*": ("anything", "any of it", "all of it", "it all", "memory", "memories", "full access"), "work": ("work",)}
# Names the user says to coding agents all day ("give Codex access to orchard", "show Cursor preferences"): they name a section
# for widening only after "my" ("let Codex see my orchard notes"); for narrowing they stay strong (round 2 review).
WIDEN_NEEDS_MY = PROFILE.widen_needs_my
LEVEL_WORDS = {"everything": ("everything",),
               "work": ("work only", "work-only", "only work", "just work", "work stuff only", "only work stuff",
                        "just work stuff", "work stuff"),
               "nothing-personal": ("nothing personal", "no personal stuff", "nothing private"),
               "nothing": ("nothing", "no memory", "no access")}
LEVEL_LABEL = {"everything": "everything", "work": "work only", "nothing-personal": "nothing personal",
               "nothing": "nothing"}
LEVEL_ASK = {"everything": "see everything", "work": "be work only", "nothing-personal": "be nothing personal",
             "nothing": "see nothing"}
SCOPE_LABEL = PROFILE.scope_label
FILLER = {"", "thanks", "thank you", "thx", "ty", "please", "ok", "okay", "cheers", "great", "cool"}


def _alt(words):
    return "(?:" + "|".join(re.escape(w) for w in sorted(set(words), key=len, reverse=True)) + ")"


def _words(*tables):
    return [w for t in tables for ws in t.values() for w in ws]


_AG = _alt(_words(AGENTS))
_AGL = rf"{_AG}(?:(?:, and |, or |, | and | or | & ){_AG})*"
_ALLW = _alt(OTHER_WORDS + ALL_WORDS)
def _t(g="t"):
    return rf"(?P<{g}>{_ALLW}(?: (?:except|but|other than|apart from|besides) {_AGL})?|{_AGL})"


T = _t()
_OWNER = PROFILE.owner or config.current().owner_name
_POSS = "|".join(PROFILE.possessives or ("their",) + ((f"{re.escape(_OWNER.lower())}'s",) if _OWNER else ()))
_DET = rf"(?:(?:my|all|all of|all my|all of my|any|any of|any of my|{_POSS}|of my) )?"
_WEAK_DET = rf"(?:my|all my|all of my|any of my|{_POSS}) "
_TAIL = r"(?: (?:stuff|info|information|data|notes|section|memories|records|things|details|history))?"


def _scope(narrow):
    if narrow:
        strong, weak = _words(SCOPE_STRONG, NARROW_ONLY), _words(SCOPE_WEAK)
    else:
        mine = {k: v for k, v in SCOPE_STRONG.items() if k in WIDEN_NEEDS_MY}
        strong = _words({k: v for k, v in SCOPE_STRONG.items() if k not in WIDEN_NEEDS_MY})
        weak = _words(SCOPE_WEAK, mine)
    one = rf"(?:{_DET}{_alt(strong)}|{_WEAK_DET}{_alt(weak)}){_TAIL}"
    return rf"(?P<s>{one}(?:(?:, and |, or |, | and | or | & | plus ){one})*)"


SW, SN = _scope(False), _scope(True)  # what may widen, what may narrow
L = rf"(?P<l>{_alt(_words(LEVEL_WORDS))})"
LN = rf"(?P<l>{_alt(w for k, ws in LEVEL_WORDS.items() if k != 'everything' for w in ws)})"  # terse forms: never all
LR = rf"(?P<l>{_alt([*(w for k, ws in LEVEL_WORDS.items() if k != 'everything' for w in ws), 'work'])})"  # restrict to
SEE_W = r"(?:see|read|access|view|look at|look through|search|know about)"
SEE = r"(?:see|read|access|view|use|look at|look through|search|have|get|know about|know)"
SEEING = r"(?:seeing|reading|accessing|viewing|using|looking at|searching|having|getting|see|read|access|view|use)"
POSS = r"(?:'s|')?"
END = r"(?: anymore| any more| again| from now on| for now| now| too| as well| today)?"

PATTERNS = [(op, re.compile(rx)) for op, rx in [
    # widen
    ("show", rf"(?:let|allow) {T} (?:to )?{SEE_W}(?: access to)? {SW}{END}"),
    ("show", rf"allow {T} access to {SW}{END}"),
    ("show", rf"i(?: want|'d like| would like) {T} to {SEE_W}(?: access to)? {SW}{END}"),
    ("show", rf"(?:give|grant) {T} (?:back )?(?:read )?(?:access to )?{SW}(?: back)?{END}"),
    ("show", rf"show {T} {SW}{END}"),
    ("show", rf"open (?:up )?{SW} to {T}{END}"),
    ("show", rf"share {SW} with {T}{END}"),
    ("show", rf"(?:unhide|stop hiding) {SW} (?:from|for) {T}{END}"),
    ("show", rf"{T} (?:should be able to|is allowed to|are allowed to|may) {SEE_W} {SW}{END}"),
    # only these
    ("only", rf"only (?:let|allow) {T} (?:to )?{SEE} {SN}{END}"),
    ("only", rf"(?:let|allow) {T} (?:to )?{SEE} only {SN}{END}"),
    ("only", rf"{T} (?:should |can |may )?only (?:see|sees|get|gets) {SN}{END}"),
    ("only", rf"make {T} only see {SN}{END}"),
    # levels
    ("level", rf"make {T}(?: see)? {L}{END}"),
    ("level", rf"(?:set|switch|put|move|change|turn) {T} (?:to|on|into) {LN}{END}"),
    ("level", rf"(?:restrict|limit|keep|lock) {T} (?:to )?{LR}{END}"),
    ("level", rf"{T}(?::| -| should be| is now| is| will be| gets| should get) {LN}{END}"),
    ("level", rf"{LN} for {T}{END}"),
    # narrow
    ("hide", rf"(?:don't|dont|do not|never) (?:let|allow) {T} (?:to )?{SEE}(?: access to)? {SN}{END}"),
    ("hide", rf"i (?:don't|dont|do not) want {T} (?:to {SEE}|{SEEING})(?: access to)? {SN}{END}"),
    ("hide", rf"make sure {T} (?:can't|cannot|can not|doesn't|does not|won't|will not|never) {SEE}(?: access to)? "
             rf"{SN}{END}"),
    ("hide", rf"{T} (?:can't|cannot|can not) {SEE} {SN} (?:anymore|any more)"),
    ("hide", rf"{T} (?:can no longer|no longer|isn't allowed to|is not allowed to|aren't allowed to|are not allowed to|"
             rf"shouldn't be able to|should not be able to|doesn't need to|does not need to|mustn't ever) {SEE}"
             rf"(?: access to)? {SN}{END}"),
    ("hide", rf"{T} no longer (?:sees|gets|has) {SN}{END}"),
    ("hide", rf"{T} (?:doesn't|does not|don't|do not) need (?:access to )?{SN}{END}"),
    ("hide", rf"(?:deny|kill|end) {T}{POSS} access to {SN}{END}"),
    ("hide", rf"deny {T} access to {SN}{END}"),
    ("hide", rf"(?:shut|lock|keep) {T} out of {SN}{END}"),
    ("hide", rf"take {T} off {SN}{END}"),
    ("others", rf"(?:only|just) {T} (?:should|can|may|gets to) see {SN}{END}"),
    ("all", rf"(?:no|no other) (?:ai|agent|app|one) (?:should|may|can) (?:see|read|access|have) {SN}{END}"),
    ("all", rf"(?:nobody|no one) (?:else )?(?:should|may|can) (?:see|read|access|have) {SN}{END}"),
    ("hide", rf"stop (?:letting |allowing )?{T} (?:from )?{SEEING} {SN}{END}"),
    ("hide", rf"hide {SN} from {T}{END}"),
    ("hide", rf"block {T} from (?:{SEEING} )?{SN}{END}"),
    ("hide", rf"block {SN} (?:from|for) {T}{END}"),
    ("hide", rf"keep {SN} (?:away )?from {T}{END}"),
    ("hide", rf"keep {T} (?:away from|out of) {SN}{END}"),
    ("hide", rf"revoke {T}{POSS} access to {SN}{END}"),
    ("hide", rf"revoke {SN} (?:access )?(?:from|for) {T}{END}"),
    ("hide", rf"{T} (?:shouldn't|should not|must not|mustn't|may not) {SEE}(?: access to)? {SN}{END}"),
    ("hide", rf"(?:don't|dont|do not|never) (?:show|give) {T} (?:access to )?{SN}{END}"),
    ("hide", rf"(?:stop sharing|don't share|dont share|do not share|never share) {SN} with {T}{END}"),
    ("hide", rf"(?:take away|remove|drop|cut off|turn off|disable) {T}{POSS} access to {SN}{END}"),
    ("hide", rf"take {SN} away from {T}{END}"),
    ("hide", rf"cut {T} off from {SN}{END}"),
    ("hide", rf"cut off {T} from {SN}{END}"),
    ("hide", rf"no (?:more )?{SN} for {T}{END}"),
]]
STATUS = [re.compile(rx) for rx in [
    rf"what (?:can|does|could|will|do) {T} (?:see|access|read|know|have access to)(?: in (?:my )?memory)?",
    rf"what(?: is|'s| are) {T}{POSS} (?:access|permissions?|scope|level|memory access)",
    rf"what access does {T} have",
    rf"(?:show|tell) me what {T} (?:can see|sees|has access to)",
    rf"(?:can|does) {T} see {SN}",
    rf"(?:who|which agents|which ais) can see {SN}"]]
UNDO_BARE_RE = re.compile(r"(?:undo|revert|reverse) (?:that|it|this)|undo|take that back")
UNDO_RE = re.compile(rf"(?:undo|revert|reverse) (?:the |that |my )?(?:last )?(?:memory )?access "
                     rf"(?:change|setting)s?(?: (?:for|to|on|of) {_t('t1')})?|undo the last change to {_t('t2')}{POSS} access")
APPROVE_BARE_RE = re.compile(r"(?:yes,? )?(?:allow|approve|accept) (?:it|that|this)(?: request)?|(?:allow|approve|accept) "
                             r"the request")
APPROVE_RE = re.compile(rf"(?:allow|approve|accept|grant) (?:{_t('t1')}{POSS} (?:access |memory |permission )?|(?:the |"
                        rf"that |this |the pending |)(?:access |memory |permission ))request|(?:allow|approve|accept|"
                        rf"grant) (?:the |that )?(?:access |memory |permission )?request from {_t('t2')}|(?:allow|approve) "
                        rf"(?P<t3>{_alt(w for k, ws in AGENTS.items() if k not in ('desk', 'unknown') for w in ws)})")
RETRACT_RE = re.compile(r"(?:(?:no|nope|wait|actually|hmm|never ?mind|scratch that|cancel that|forget (?:it|that)|hold "
                        r"on|not yet|on second thought|don't|dont|do not)[,!.]? ?)+(?:do (?:that|it))?")
_A = rf"(?P<t>{_AGL})"
_NOW = r"(?: now| for now| next| for a bit| tomorrow| for the rest)?"
HANDOFF_OUT = [re.compile(rx + _NOW) for rx in [
    rf"(?:i'm|im|i am) (?:switching|moving|jumping|handing (?:this |it )?(?:off|over)|passing (?:this|it))(?: this)?"
    rf"(?: over)? to {_A}",
    rf"(?:i'll|i will|let's|lets) (?:switch|move|hand (?:this |it )?off|pick this up|finish this)(?: over)? (?:in|to|"
    rf"with) {_A}",
    rf"(?:moving|passing|switching|sending|handing) this (?:over )?to {_A}",
    rf"(?:hand|handing) (?:this |it )?(?:off|over)(?: to {_A})?",
    rf"(?:write|make|do|prepare) (?:a |the )?hand ?off(?: note)?(?: for {_A})?",
    rf"hand ?off(?: to {_A})?"]]
HANDOFF_IN = [re.compile(rx) for rx in [
    rf"pick up where (?:{_AGL}|we|you|the last (?:agent|session|ai)|it|they|i) left off\b",
    rf"(?:continue|carry on) where (?:{_AGL}|we|the last (?:agent|session|ai)|it|they) left off\b",
    rf"pick up from (?:{_AGL}){POSS} hand ?off\b",
    r"continue (?:from|with) (?:the |that )?(?:last )?hand ?off\b",
    r"(?:read|check|load|get|show me) the (?:last |latest )?hand ?off\b"]]
# A sentence that names an agent and a section with a narrowing verb, yet matched nothing: one line says so, so the user
# never believes something was hidden when it was not (round 1 review).
_AGW = rf"(?:{_AG}|{_ALLW}|ai|ais|agents?|anyone|nobody|no one)"
NEAR_MISS = re.compile(rf"\b(?:hide|block|revoke|don't let|do not let|never let|stop|restrict|limit|lock|disable|turn "
                       rf"off|cut off|remove|keep|deny|shut|kill|only|don't want|do not want)\b.*\b{_AGW}\b|"
                       rf"\b{_AGW}\b.*\b(?:shouldn't|should not|must not|can't see|no longer|not allowed|isn't allowed|"
                       rf"doesn't need|don't need)\b")
CODE_CHARS = re.compile(r"[`{}\[\]()<>=$\\|#*/@]")
LIST_RE = re.compile(r"\s*(?:\d{1,3}[.)]|[-*•])\s")
LEAD = re.compile(r"^(?:(?:hey|hi|ok|okay|so|and|also|now|then|please|pls|yes|yeah|yep|alright|right|actually|"
                  r"btw|oh|ah|hmm|well|can you|could you|would you|will you|i want you to|i'd like you to|go ahead and"
                  r"|quick one|one more thing)[,:!]?\s+)+")
TRAIL = re.compile(r"(?:,?\s+(?:please|thanks|thank you|thx|ty|it's private|that's private|it's personal|ok|okay|lol))+$")


def _norm(s):
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", s).strip().lower()


def _clean(sentence):
    s = _norm(sentence).strip(" \"'")
    s = re.sub(r"[.!?;,]+$", "", s).strip(" \"'")
    for _ in range(3):
        s = re.sub(r"[.!?;,]+$", "", TRAIL.sub("", LEAD.sub("", s))).strip()
    return s


def _pieces(line):
    return [p.strip() for p in re.split(r"(?<=[.!?;])\s+", line.strip()) if p.strip()]


def sentences(prompt):
    """(raw, cleaned, line number, raw line) for each sentence of a prompt worth reading: not inside a code fence, not
    a quoted or indented line, not code-looking, not wrapped in quotes."""
    fence = False
    for i, line in enumerate((prompt or "").split("\n")):
        if re.match(r"\s*(?:```|~~~)", line):
            fence = not fence
            continue
        if fence or re.match(r"\s*>|(?: {4}|\t)\S", line):
            continue
        for raw in _pieces(line):
            if len(raw) > 300 or CODE_CHARS.search(raw) or re.fullmatch(r'["“\'‘].*["”\'’]', raw):
                continue
            yield raw, _clean(raw), i, line


def _found(text, table):
    """The keys of table whose words occur in text, longest words first."""
    words = {w: k for k, ws in table.items() for w in ws}
    rx = re.compile(r"(?<![\w'-])" + _alt(words) + r"(?![\w'-])")
    return [words[w] for w in rx.findall(text)]


def scopes_of(text, narrow=True):
    """The policy names a scope phrase names: sections, WITHHELD, chats, "*" (everything) or "work" (WORK_ONLY)."""
    tables = (SCOPE_STRONG, SCOPE_WEAK) + ((NARROW_ONLY,) if narrow else ())
    return list(dict.fromkeys(k for t in tables for k in _found(text, t)))


def agents_in(text):
    return list(dict.fromkeys(_found(text or "", AGENTS)))


def targets(m, t, speaker, widen):
    """The client ids a target phrase names: agents, or every known client (registered or named), with "other" leaving
    out the one speaking and "except" those named. "unknown" (new agents) joins "every AI" only for narrowing."""
    if re.match(_ALLW, t or ""):
        other = bool(re.match(_alt(OTHER_WORDS), t))
        rest = re.split(r" (?:except|but|other than|apart from|besides) ", t, maxsplit=1)
        out = set(NAMES) | set(m.client_policies())
        out -= set(agents_in(rest[1])) if len(rest) > 1 else set()
        if other:
            out.discard(speaker)
        if widen:
            out.discard("unknown")
        return sorted(out)
    return agents_in(t)


def parse(prompt, speaker="desk"):
    """What a prompt asks of access and handoffs, in order: dicts with kind (change, status, undo, approve,
    handoff_out, handoff_in, near_miss), the sentence (raw, s cleaned, line, line_text), and for a change: op (show,
    hide, only, level), t (target phrase), what (policy names) or level. Deterministic; nothing here changes anything."""
    out, whole = [], _clean(prompt or "")
    for raw, s, line_no, line in sentences(prompt):
        if len(out) >= MAX_INTENTS:
            break
        hit = None
        for op, rx in PATTERNS:
            if (mt := rx.fullmatch(s)):
                d = mt.groupdict()
                if op == "level":
                    lv = _found(d["l"], LEVEL_WORDS) or (["work"] if d["l"] == "work" else [])
                    hit = {"kind": "change", "op": "level", "t": d["t"], "level": lv[0] if lv else None}
                elif op in ("others", "all"):  # "only Claude Code should see X", "no AI should see X": narrow the rest
                    t = f"everyone except {d['t']}" if op == "others" else "everyone"
                    hit = {"kind": "change", "op": "hide", "t": t, "what": scopes_of(d["s"])}
                else:
                    hit = {"kind": "change", "op": op, "t": d["t"], "what": scopes_of(d["s"], op != "show")}
                break
        if hit is None:
            for rx in STATUS:
                if (mt := rx.fullmatch(s)):
                    hit = {"kind": "status", "t": mt.groupdict().get("t"), "what": scopes_of(mt.groupdict().get("s") or "")}
                    break
        if hit is None and (mt := UNDO_RE.fullmatch(s)):
            hit = {"kind": "undo", "bare": False, "t": next((g for g in mt.groups() if g), None)}
        if hit is None and s == whole and UNDO_BARE_RE.fullmatch(s):
            hit = {"kind": "undo", "bare": True, "t": None}
        if hit is None and (mt := APPROVE_RE.fullmatch(s)):
            hit = {"kind": "approve", "bare": False, "t": next((g for g in mt.groups() if g), None)}
        if hit is None and s == whole and APPROVE_BARE_RE.fullmatch(s):
            hit = {"kind": "approve", "bare": True, "t": None}
        if hit is None:
            for rx in HANDOFF_OUT:
                if (mt := rx.fullmatch(s)):
                    hit = {"kind": "handoff_out", "t": mt.groupdict().get("t")}
                    break
        if hit is None and any(rx.match(s) for rx in HANDOFF_IN):
            hit = {"kind": "handoff_in"}
        if hit is None and len(s) <= LOOSEN_SENTENCE_MAX and NEAR_MISS.search(s) and scopes_of(s):
            hit = {"kind": "near_miss"}
        if hit:
            hit.update(raw=raw, s=s, line=line_no, line_text=line)
            out.append(hit)
    return out


# ---------- policies ----------

def universe(m):
    return {*m.SECTIONS, m.WITHHELD, "chats"}


def level_entry(m, level):
    # nothing is an allow list naming no section: a section added later stays hidden too (round 1 review)
    return {"everything": {"deny": []}, "work": {"allow": list(m.WORK_ONLY), "deny": []},
            "nothing-personal": {"deny": sorted(m.NOTHING_PERSONAL)},
            "nothing": {"allow": ["nothing"], "deny": []}}[level]


def level_name(hide, m):
    """"everything", "work only", "nothing personal" or "nothing" for a hidden set that is exactly a level, else None."""
    hide, u = set(hide), universe(m)
    if not hide:
        return "everything"
    if hide == u - set(m.WORK_ONLY):
        return "work only"
    if hide == set(m.NOTHING_PERSONAL):
        return "nothing personal"
    if hide == u:
        return "nothing"
    return None


def _mutation(m, op, what=None, level=None):
    """(change(entry) -> entry for clients_put, op, level, what). what: policy names ("*" everything, "work" WORK_ONLY)."""
    what = set(what or ())
    if op in ("show", "hide") and "*" in what:
        op, level = "level", "everything" if op == "show" else "nothing"
    if "work" in what:
        what = (what - {"work"}) | set(m.WORK_ONLY)
    what &= universe(m)

    def change(e):
        keep = {k: v for k, v in e.items() if k not in ("allow", "deny")}
        if op == "level":
            return {**keep, **level_entry(m, level)}
        if op == "only":
            return {**keep, "allow": sorted(what), "deny": []}
        deny = {m._policy_name(x) for x in e.get("deny") or []} - {None}
        allow = [a for a in (e.get("allow") or []) if m._policy_name(a) or a == "nothing"]
        if op == "hide":  # deny wins over allow; the allow list is left as it is (emptying it would show everything)
            return {**keep, "deny": sorted(deny | what), **({"allow": allow} if allow else {})}
        new = {**keep, "deny": sorted(deny - what)}
        if allow:
            new["allow"] = sorted((set(allow) - {"nothing"}) | what)
        return new
    return change, op, level, what


def _narrow(m, name, change, source, **log):
    """Apply a change that previewed as a narrowing, under the lock, only if it still narrows then (the policy may have
    changed since the preview). (hidden before, hidden after); equal when nothing was written."""
    def only_narrowing(e):
        new = change(json.loads(json.dumps(e)))
        return new if set(m.policy_hide(new)) >= set(m.policy_hide(e)) else e
    b, a = m.clients_put(name, only_narrowing, source, quiet=True, **log)
    return m.policy_hide(b), m.policy_hide(a)


def direction(before, after):
    b, a = set(before), set(after)
    return "same" if a == b else "tighten" if a >= b else "loosen"


def preview(m, name, change):
    """(hidden before, hidden after) of a change to the policy in force for name, without writing it."""
    e = m.policy_base(name)
    return m.policy_hide(e), m.policy_hide(change(json.loads(json.dumps(e))))


def name_of(name):
    return NAMES.get(name) or str(name).capitalize()


ORDER = ("texts", "email", "meetings", "health-money", "chats", "people", "personal", "school")


def labels(names):
    names = sorted(set(names), key=lambda x: (ORDER.index(x) if x in ORDER else len(ORDER), x))
    words = [SCOPE_LABEL.get(n, n) for n in names]
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1] if words else "nothing"


def _what_words(m, op, what, level):
    if op == "level":
        return LEVEL_LABEL.get(level, level)
    if set(what) >= universe(m):
        return "everything"
    return ("only " if op == "only" else "") + labels(what)


def _join(names):
    names = [name_of(n) for n in names]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


# ---------- history, requests, notifications ----------

def _hist_file(m):
    return m.paths().access_history


def _req_file(m):
    return m.paths().access_requests


def history(m):
    try:
        lines = _hist_file(m).read_text().splitlines()
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if isinstance(d, dict) and isinstance(d.get("id"), int):
            out.append(d)
    return out


def log_change(m, name, before, after, source, undoes=None, session=None, batch=None, quiet=False):
    """Append one change to access-history.jsonl (called by mem.clients_put under the lock). Policies only: names of
    sections, never memory text, never the key's hash. Unless quiet (access.py tells the user itself, once per batch), a
    widening notifies them."""
    hb, ha = m.policy_hide(before), m.policy_hide(after)
    way = direction(hb, ha)
    last = history(m)
    hid = max(int(time.time() * 1000), (last[-1]["id"] + 1) if last else 0)
    strip = lambda e: {k: e[k] for k in ("allow", "deny") if k in e}  # noqa: E731
    row = {"id": hid, "at": time.time(), "client": name, "dir": way, "before": strip(before), "after": strip(after),
           "hidden_before": list(hb), "hidden_after": list(ha), "source": source}
    for k, v in (("undoes", undoes), ("session", session), ("batch", batch)):
        if v:
            row[k] = v
    with os.fdopen(os.open(_hist_file(m), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600), "a") as f:
        f.write(json.dumps(row) + "\n")
    if not quiet and way == "loosen":
        how = " (changed from the command line)" if source.startswith("cli") else ""
        notify(f"{name_of(name)} can now see {labels(set(hb) - set(ha))}{how} · say 'undo the last access change' "
               "in any AI to undo", kind="widened", client=name, sections=sorted(set(hb) - set(ha)), change=hid)
    return row


def _tell(changes, source):
    """One notification for a batch of applied changes: what widened (any source), or what an agent narrowed."""
    wide = [(n, set(b) - set(a)) for n, b, a in changes if set(b) - set(a)]
    if wide:
        same = all(w == wide[0][1] for _, w in wide)
        text = (f"{_join([n for n, _ in wide])} can now see {labels(wide[0][1])}" if same else
                "; ".join(f"{name_of(n)} can now see {labels(w)}" for n, w in wide))
        how = " (approved)" if source == "approved" else ""
        notify(text + how + " · say 'undo that' in any AI to undo", kind="widened", client=",".join(n for n, _ in wide),
               sections=sorted(set().union(*(w for _, w in wide))))
    elif source.startswith("tool:"):
        gone = [(n, set(a) - set(b)) for n, b, a in changes if set(a) - set(b)]
        if gone:
            notify(f"{name_of(source[5:])} hid {labels(set().union(*(g for _, g in gone)))} from "
                   f"{_join([n for n, _ in gone])} · say 'undo the last access change' in any AI to undo",
                   kind="narrowed", client=",".join(n for n, _ in gone), sections=sorted(set().union(*(g for _, g in gone))))


# the user turned macOS banners off (2026-10-20): notices go to a small feed a host's menu-bar app shows instead.
BANNERS = PROFILE.experiments.get("BANNERS", "on")   # off | on (a macOS notification for every widening)
NOTICES = config.current().paths.notices
NOTICES_KEEP = 50          # requests and other notices kept
KEEP_CHANGES = 30 * 86400  # widened and narrowed notices are kept 30 days, however many requests arrive
_meta = {}


def _post(text):
    """Record one notice in the feed (0600; agent and section names only, never memory text), and show a plain Mac
    banner only when BANNERS is on (osascript; the text is an argument, never parsed)."""
    row = {"id": int(time.time() * 1000), "at": time.time(), "text": text, **_meta}
    try:  # under a lock, so two writers never lose a row; a flood of requests never pushes out a widening
        with os.fdopen(os.open(NOTICES.with_suffix(".lock"), os.O_WRONLY | os.O_CREAT, 0o600), "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            rows = []
            for x in (NOTICES.read_text().splitlines() if NOTICES.exists() else []):
                try:
                    d = json.loads(x)
                except ValueError:
                    continue
                if isinstance(d, dict) and isinstance(d.get("at"), (int, float)) and isinstance(d.get("id"), int):
                    rows.append(d)
            rows.append(row)
            keep = [r for r in rows if r.get("kind") in ("widened", "narrowed") and time.time() - r.get("at", 0) < KEEP_CHANGES]
            other = [r for r in rows if r not in keep][-NOTICES_KEEP:]
            rows = sorted(keep + other, key=lambda r: r.get("id", 0))
            fd, tmp = tempfile.mkstemp(dir=NOTICES.parent, prefix=".notices.")
            with os.fdopen(fd, "w") as f:
                f.write("".join(json.dumps(r) + "\n" for r in rows))
            os.chmod(tmp, 0o600)
            os.replace(tmp, NOTICES)
    except Exception:  # a broken feed never stops the change it reports: fall back to appending the row
        try:
            with os.fdopen(os.open(NOTICES, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600), "a") as f:
                f.write(json.dumps(row) + "\n")
        except Exception:
            pass
    if BANNERS != "on":
        return
    subprocess.Popen(["osascript", "-e", "on run argv", "-e", 'display notification (item 1 of argv) with title '
                      '(item 2 of argv)', "-e", "end run", text, LABEL], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, start_new_session=True)


def notify(text, kind="notice", **meta):
    if os.environ.get("LFLABS_MEMORY_NOTIFY") == "off":
        return
    _meta.clear(); _meta.update(kind=kind, **{k: v for k, v in meta.items() if v is not None})
    try:
        _post(text[:220])
    except OSError:
        pass


def requests(m, now=None):
    now = time.time() if now is None else now
    try:
        reqs = json.loads(_req_file(m).read_text()).get("requests") or []
    except (OSError, ValueError, AttributeError):
        reqs = []
    return [r for r in reqs if isinstance(r, dict) and now - float(r.get("at") or 0) < PENDING_TTL]


def _write_requests(m, reqs):
    path = _req_file(m)
    tmp = path.with_name(path.name + ".tmp")
    with os.fdopen(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as f:
        json.dump({"requests": reqs[-PENDING_MAX:]}, f, indent=1)
    os.replace(tmp, path)


def _ask_words(m, r):
    return "see " + _what_words(m, r["op"], set(r.get("what") or ()), r.get("level")) if r["op"] != "level" else \
        LEVEL_ASK.get(r.get("level"), r.get("level"))


def _who(by):
    return name_of(by) if by in NAMES or re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,31}", by or "") else by[:1].upper() + by[1:]


def request(m, by, target, op, what, level):
    """Queue a widening for the user's typed approval and tell them; one identical pending request is refreshed, not
    repeated. Returns the request."""
    key = [target, op, sorted(what), level]
    with m._clients_lock():
        reqs = requests(m)
        same = next((r for r in reqs if r.get("key") == key), None)
        if same:  # the same request again: kept, with its first time (a re-file never makes it look new)
            return same
        mine = [x for x in reqs if x.get("by") == by]
        if len(mine) >= PER_REQUESTER:  # one agent can never push out the others' requests
            reqs.remove(mine[0])
        r = {"id": int(time.time() * 1000), "at": time.time(), "by": by, "target": target, "op": op,
             "what": sorted(what), "level": level, "key": key}
        _write_requests(m, reqs + [r])
    if not same:
        ask, who = _ask_words(m, r), _who(by)
        said = f"{who} asked to {ask}" if by == target else f"{who} asked to let {name_of(target)} {ask}"
        notify(said + f" · to allow, type 'approve {name_of(target)}'s request' in any AI", kind="request",
               client=target, sections=sorted(what), request_id=r["id"])
    return r


def pending_list(m, reqs):
    return "; ".join(f"{name_of(r['target'])} would {_ask_words(m, r)}" for r in reqs)


def approve(m, agent=None, bare=False, speaker=None, source="approved", index=None, now=None, request_id=None):
    """Apply one pending request: index (the CLI), or the requests for agent when named, or the only one waiting for
    "approve the access request". A bare "allow it" / "approve that" never applies anything (round 2 review: an agent
    can keep one request waiting and fresh, then the user's everyday "accept it" in another agent would widen it); it lists
    what is waiting and the words that approve it. ("applied", name, before, after) | ("list", text) | None."""
    now = time.time() if now is None else now
    with m._clients_lock():
        reqs = requests(m, now)
        if index is not None:
            cands = [reqs[index]] if 0 <= index < len(reqs) else []
        elif request_id is not None:  # the menu-bar app's Approve: exactly the request on the row the user clicked
            cands = [r for r in reqs if r.get("id") == request_id]
        else:
            cands = [r for r in reqs if agent is None or r.get("target") == agent]
        if not cands:
            return None
        say = "; ".join(f"'approve {name_of(t)}'s request'" for t in dict.fromkeys(r["target"] for r in cands))
        if bare:
            return "list", (f"waiting for the user's approval: {pending_list(m, cands)}. A bare 'allow it' approves nothing; "
                            f"they can say {say}")
        if len(cands) > 1:
            return "list", (f"{len(cands)} access requests are waiting: {pending_list(m, cands)}. Nothing was approved; "
                            f"they can say {say}")
        pick = cands[0]
        reqs.remove(pick)
        _write_requests(m, reqs)
    change = _mutation(m, pick["op"], pick.get("what"), pick.get("level"))[0]
    b, a = m.clients_put(pick["target"], change, source, quiet=True, batch=int(now * 1000))
    b, a = m.policy_hide(b), m.policy_hide(a)
    _tell([(pick["target"], b, a)], source)
    return "applied", pick["target"], b, a


def _clear_requests(m, name, after):
    """A typed widening that already gives what a pending request asked for answers it."""
    with m._clients_lock():
        reqs = requests(m)
        left = [r for r in reqs if not (r.get("target") == name and r.get("op") == "show"
                                        and set(r.get("what") or ()) <= universe(m) - set(after))]
        if len(left) != len(reqs):
            _write_requests(m, left)


def undo(m, t=None, bare=False, typed=True, source="undo", session=None, now=None):
    """Revert the newest change not yet undone, with every change made with it (one batch); for one client when t names
    it. A bare "undo that" only reverts a change of the last UNDO_BARE made in the same session, else does nothing
    (None). Reverting a narrowing widens, so it needs typed. [(name, hidden before, hidden after)] or a reason string."""
    now = time.time() if now is None else now
    rows = history(m)
    done = {r.get("undoes") for r in rows}
    cands = [r for r in rows if not r.get("undoes") and r["id"] not in done and (t is None or r["client"] == t)]
    if not cands:
        return None if bare else "there is no access change to undo"
    last = cands[-1]
    if bare and (now - last["at"] > UNDO_BARE or not session or last.get("session") != session):
        return None
    batch = [r for r in cands if last.get("batch") and r.get("batch") == last["batch"]] or [last]
    backs = []
    for r in reversed(batch):
        put = lambda e, before=r["before"]: {**{k: v for k, v in e.items() if k not in ("allow", "deny")}, **before}  # noqa: E731
        backs.append((r, put, *preview(m, r["client"], put)))
    if not typed and any(direction(b, a) == "loosen" for *_, b, a in backs):
        return "undoing that would widen access, which needs the user's own typed words"
    stamp, out = int(now * 1000), []
    for r, put, _, _ in backs:
        b, a = m.clients_put(r["client"], put, source, undoes=r["id"], quiet=True, batch=stamp, session=session)
        out.append((r["client"], m.policy_hide(b), m.policy_hide(a)))
    _tell(out, source)
    return out


# ---------- plain-language answers ----------

def describe(m, name, key_hint=True):
    """What this client sees, in one sentence. key_hint False: no word about its missing key (the product's CLI keys it
    next: KEY_BY_ENV)."""
    hide = set(m.client_hide(name, True))
    who, level = name_of(name), level_name(hide, m)
    if not hide:
        out = f"{who} sees everything in memory"
    elif level == "nothing":
        out = f"{who} sees nothing in memory"
    elif level == "work only":
        out = f"{who} is work only: it sees {labels(universe(m) - hide)} and nothing else"
    elif level == "nothing personal" or len(hide) <= len(universe(m)) // 2:
        out = f"{who} sees everything except {labels(hide)}" + (" (nothing personal)" if level else "")
    else:
        out = f"{who} sees only {labels(universe(m) - hide)}"
    if key_hint and m.KEYLESS_MODE == "on" and _connect().needs_key(m, name) and not (
            m.client_policies().get(name) or {}).get("key_sha256"):
        out += f"; until its config carries a key it gets work only (`{brand.COMMAND} access {name}` gives it one)"
    n = sum(r.get("target") == name for r in requests(m))
    return out + (f"; {n} request{'s' if n > 1 else ''} to widen it waiting for the user" if n else "")


def _off_note(m):
    return (" Per-agent scopes are off (CLIENT_SCOPES_MODE), so this takes effect once they are on."
            if m.CLIENT_SCOPES_MODE != "on" else "")


def status_text(m, client):
    names = targets(m, _norm(client), "", False) if client else []
    if not names:
        c = _norm(client or "")
        names = [c] if m.CLIENT_RE.fullmatch(c) else []
    if not names:
        return f"no agent called {client!r}; known: {', '.join(sorted(set(NAMES) | set(m.client_policies())))}"
    return "; ".join(describe(m, n) for n in names) + "." + _off_note(m)


def _connect():
    from . import connect_apps
    return connect_apps


def _keys(m, names):
    """After a widening applied: give each widened app that needs one a key in its wired config. Plain sentences."""
    out = []
    for n in names:
        try:
            if (s := _connect().ensure_key(m, n)):
                out.append(s)
        except Exception as e:  # the widening stands; the app stays at work only until connect gives it its key
            out.append(f"{name_of(n)} stays at work only until it has its key ({type(e).__name__}): run "
                       f"`{brand.COMMAND} access {n}`")
    return out


# ---------- the prompt hook ----------

def typed_source(m, data, speaker):
    """Whether this prompt reached the hook as the user's own typing: Claude Code's hook (desk) outside a background or SDK
    run, or Codex's hook for a session the user started (its rollout's meta); never voice, Telegram, benches, our own
    `claude -p` runs, notices or relayed messages."""
    prompt = data.get("prompt") or ""
    if m.saving_off() or m.scope_of(data.get("session_id")) is not None or m.not_asked(prompt):
        return False
    if m.SYSTEM_MSG_RE.match(prompt) or m.NOT_OWNER_RE.search(prompt) or m.WRAPPER_OPEN_RE.search(prompt) \
            or m.PASTE_MARK_RE.search(prompt) or re.search(r"<(?:channel|scheduled-task|cron)\b", prompt):
        return False
    if speaker == "desk":
        return not os.environ.get("CLAUDE_CODE_ENTRYPOINT", "").startswith("sdk")
    if speaker == "codex":
        path = data.get("transcript_path")
        return bool(path) and m._is_codex(path) and m._codex_desk(m._codex_meta(Path(path)))
    return False


def _his_sentences(m, prompt):
    """The cleaned sentences of the user's own part of a prompt (his_part: no pastes, quotes or added context)."""
    t = prompt or ""
    if (i := t.find(m.CODEX_REQUEST)) >= 0:  # Codex's IDE context comes before this marker
        t = t[i + len(m.CODEX_REQUEST):]
    return {s for _, s, _, _ in sentences(m.his_part(t)[0])}


def _gate(m, data, intents, typed):
    """mine(intent) -> whether a widening sentence counts as the user's own typed line (see the module docstring)."""
    prompt = data.get("prompt") or ""
    if not typed or len(prompt) > LOOSEN_PROMPT_MAX:
        return lambda it: False
    his = _his_sentences(m, prompt)
    said = {it["s"] for it in intents}
    lines = [(i, ln) for i, ln in enumerate(prompt.split("\n")) if ln.strip()]
    first = lines[0][0] if lines else 0
    only_access = all(not re.match(r"\s*(?:```|~~~|>)", ln) and all(_clean(p) in said or _clean(p) in FILLER
                                                                     for p in _pieces(ln)) for _, ln in lines)
    return lambda it: (len(it["raw"]) <= LOOSEN_SENTENCE_MAX and it["s"] in his and not LIST_RE.match(it["line_text"])
                       and (it["line"] == first or only_access))


def _change(m, it, speaker, mine, by, held, session):
    """One change sentence: (line or None, [widened client names]). Narrowing applies; widening applies only from
    The user's own line (mine) and when the prompt neither narrows the same agent nor takes it back (held: names, or "*" for
    a retraction); else a widening that is not theirs is requested, and one they held back is dropped."""
    change, op, level, what = _mutation(m, it["op"], it.get("what"), it.get("level"))
    if op == "level" and not level or op != "level" and not what:
        return None, []
    applied, same, asked, kept = [], [], [], []
    batch = int(time.time() * 1000)
    source = "typed" if mine else "said"
    for name in targets(m, it["t"], speaker, it["op"] in ("show", "only", "level")):
        b, a = preview(m, name, change)
        way = direction(b, a)
        if way == "same":
            same.append(name)
        elif way == "tighten":
            b, a = _narrow(m, name, change, source, batch=batch, session=session)
            if set(a) != set(b):
                applied.append((name, b, a))
        elif mine and ("*" in held or name in held):
            kept.append(name)
        elif mine:
            m.clients_put(name, change, source, quiet=True, batch=batch, session=session)
            applied.append((name, b, a))
            _clear_requests(m, name, a)
        else:
            request(m, by, name, op, what, level)
            asked.append(name)
    _tell(applied, source)
    parts = []
    for name, b, a in applied:
        gone, new = set(a) - set(b), set(b) - set(a)
        lv = level_name(a, m) if op == "level" else None
        if lv:
            parts.append(f"{name_of(name)} is now {lv}" + (f" (it no longer sees {labels(gone)})" if gone and not new
                                                          else ""))
        elif new and not gone:
            parts.append(f"{name_of(name)} can now see {labels(new)}")
        elif gone and not new:
            parts.append(f"{name_of(name)} can no longer see {labels(gone)}")
        else:
            parts.append(f"{name_of(name)} now sees {labels(new)} and no longer {labels(gone)}")
    if same:
        parts.append(_join(same) + " already " + (
            "set that way" if op in ("level", "only") else "sees that" if op == "show" else "can't see that"))
    if kept:
        parts.append(f"nothing widened for {_join(kept)}: the same message " + (
            "takes it back" if "*" in held else "also narrows it"))
    if asked:
        parts.append(f"widening {_join(asked)} ({_what_words(m, op, what, level)}) is waiting for the user's approval, as "
                     "it did not come from their own short typed line")
    widened = [n for n, b, a in applied if set(b) - set(a)]
    if not parts:
        return None, []
    return (f"{m.LABEL}: " + "; ".join(parts) + "."
            + (" (the user's own words. They can say 'undo that'.)" if widened else "") + _off_note(m)), widened


def _mark_handoff(m, data):
    """Queue this session's unsaved part for saving and its handoff now (the worker's urgent path): final = 1."""
    sid, path = data.get("session_id"), data.get("transcript_path")
    if not sid:
        return
    db = m.connect()
    with db:
        if path:
            db.execute("insert into sessions(session_id, transcript, cwd, activity, final) values (?,?,?,?,1)"
                       " on conflict(session_id) do update set final = 1", (sid, path, data.get("cwd"), time.time()))
        else:
            db.execute("update sessions set final = 1 where session_id = ?", (sid,))


# A typed widening is never made from a prompt (security review 2026-11-09: a process running as the user can append a
# forged turn to the session's transcript and fire the hooks, so the transcript is no proof for giving an app more). It
# is answered with the Terminal command that is: there the user types the app's name and clicks Allow in a macOS
# dialog (human_at_terminal). A narrowing still applies at once, from any words.
TERMINAL = ("{label}: giving {who} more access needs the user at a Terminal, where they confirm it in a dialog: "
            "`{cmd}`. Nothing changed.")


def terminal_command(m, it, speaker="desk"):
    """The Terminal command that makes a typed widening, approve or undo (TERMINAL)."""
    if it["kind"] == "approve":
        return "the app", f"{m.CFG.command} clients approve"
    if it["kind"] == "undo":
        return "an app", f"{m.CFG.command} clients undo"
    names = targets(m, it.get("t"), speaker, True) or ["the app"]
    what = set(it.get("what") or ())
    level = it.get("level") if it.get("op") == "level" else "everything" if it.get("op") == "show" and "*" in what \
        else None
    who = ", ".join(name_of(n) for n in names)
    cmds = [f"{brand.COMMAND} access {n} {level}" if level else f"{brand.COMMAND} access {n}" for n in names]
    return who, " ; ".join(cmds)


def on_prompt(m, data, speaker="desk", defer=False):
    """Read one prompt in the hook: apply or queue what it asks of access, and note handoff phrases. {"lines": one
    line each for the context, "urgent": save now, "again": give the last handoff again, "defer": a widening of the
    user's own to make once the transcript shows their words (mem.typed_pending)} or None. defer: the prompt hook's
    call: a narrowing still applies at once; the user's widening, undo or approve waits for the check."""
    prompt = data.get("prompt") or ""
    if not prompt.strip() or m.not_asked(prompt):
        return None
    intents = parse(prompt, speaker)
    if not intents:
        return None
    session = data.get("session_id")
    typed = typed_source(m, data, speaker)
    mine = _gate(m, data, intents, typed)
    held = set()  # what this prompt also narrows (tightening wins), or "*" when it takes widening back
    if any(RETRACT_RE.fullmatch(s) for _, s, _, _ in sentences(prompt)):
        held.add("*")
    for it in intents:
        if it["kind"] == "change" and (it["op"] == "hide" or it.get("level") == "nothing"):
            held |= set(targets(m, it["t"], speaker, False))
    lines, widened = [], []
    out = {"lines": lines}
    coding = m.scope_of(session) is None
    changed = any(it["kind"] == "change" for it in intents)
    for it in intents:
        k, line = it["kind"], None
        narrows = k == "change" and (it["op"] == "hide" or it.get("level") == "nothing")
        if defer and k in ("change", "undo", "approve") and not narrows and mine(it):
            who, cmd = terminal_command(m, it, speaker)  # never from a prompt: the Terminal and its dialog (TERMINAL)
            line = TERMINAL.format(label=m.LABEL, who=who, cmd=cmd)
        elif k == "change":
            own = mine(it)
            by = speaker if own else f"a {'long or pasted ' if typed else ''}message in {name_of(speaker)}"
            line, w = _change(m, it, speaker, own, by, held, session)
            widened += w
        elif k == "status":
            names = targets(m, it["t"], speaker, False) if it.get("t") else sorted(set(NAMES) | set(m.client_policies()))
            line = f"{m.LABEL}: " + "; ".join(describe(m, n) for n in names) + "." + _off_note(m)
        elif k == "undo":
            t = (agents_in(it["t"]) or [None])[0] if it.get("t") else None
            r = undo(m, t, it["bare"], mine(it), session=session)
            if isinstance(r, list):
                line = f"{m.LABEL}: undid the last access change; " + "; ".join(describe(m, n) for n, *_ in r) + "."
                widened += [n for n, b, a in r if set(b) - set(a)]
            elif r:
                line = f"{m.LABEL}: {r}."
        elif k == "approve":
            r = approve(m, (agents_in(it["t"]) or [None])[0] if it.get("t") else None, it["bare"], speaker) \
                if mine(it) or it["bare"] else ("refused", "approving an access request needs the user's own short typed "
                                                "line in Claude Code or Codex") if requests(m) else None
            if r is None:
                line = None if it["bare"] else f"{m.LABEL}: no access request is waiting."
            elif r[0] == "applied":
                line = (f"{m.LABEL}: approved; {describe(m, r[1])} (the user's own words. They can say 'undo that'.)"
                        + _off_note(m))
                widened.append(r[1])
            else:
                line = f"{m.LABEL}: {r[1]}."
        elif k == "handoff_out" and m.HANDOFF_MODE == "on" and coding:
            _mark_handoff(m, data)
            out["urgent"] = True
            line = f"{m.LABEL}: writing a handoff for the next agent."
        elif k == "handoff_in" and m.HANDOFF_MODE == "on" and coding:
            out["again"] = True
            line = (f"{m.LABEL}: The user wants to pick up where the last agent left off; its handoff notes follow when "
                    "there are any (the handoff tool has them too).")
        elif k == "near_miss":
            line = (f"{m.LABEL}: that read like an access change but matched no form I know, so nothing changed "
                    "(say e.g. \"don't let Codex see my texts\").")
        if line and line not in lines:
            lines.append(line)
    if widened and (keys := _keys(m, sorted(set(widened)))):
        lines.append(f"{m.LABEL}: " + "; ".join(keys) + ".")
    lines[:] = [m.own(line) for line in lines]  # written with a placeholder name: the user's own name and pronouns
    return out if lines or out.get("urgent") or out.get("again") else None


# ---------- the MCP tools ----------

STATUS_DOC = """What an AI agent may see in the user's long-term memory, in plain words: everything, work only,
nothing personal, or which sections it never sees. client: an agent (codex, cursor, claude-desktop, windsurf, gemini,
copilot, chatgpt, desk for Claude Code, unknown for new agents); omit it for yourself. Read-only."""
CHANGE_DOC = """Change what an AI agent may see in the user's long-term memory. change: "hide" (narrow: applies at
once) or "show" / "level" / "only" (widen: never applies from here; it becomes a request the user approves by typing,
and they are notified). You cannot approve a request, yours or another's. client: an agent name (codex, cursor,
claude-desktop, windsurf, gemini, copilot, chatgpt, desk, unknown) or "every AI". what: sections such as """ + \
    PROFILE.access_sections + """, or a level: everything, work only, nothing personal, nothing."""
TOOL_OPS = {"hide": "hide", "tighten": "hide", "narrow": "hide", "remove": "hide", "deny": "hide", "block": "hide",
            "revoke": "hide", "show": "show", "loosen": "show", "widen": "show", "allow": "show", "add": "show",
            "grant": "show", "level": "level", "set": "level", "only": "only"}


def tool_change(m, requester, client, change, what):
    """access_change: narrowing applies now (logged as the requesting agent's, and the user is told once); widening is only
    ever a request. Returns one plain line."""
    op = TOOL_OPS.get(_norm(change or ""))
    if not op:
        return "change must be hide, show, level or only"
    w = _clean(what or "")
    level = None
    if re.fullmatch(L, w) or w in ("full access", "all of it"):  # a level: everything, work only, nothing personal...
        level = _found(w, LEVEL_WORDS)[0] if re.fullmatch(L, w) else "everything"
        if op == "hide":  # "hide everything" is the level nothing; hiding a level is not a thing
            if level != "everything":
                return f"to narrow {client} to a level, use change \"level\" with what {what!r}"
            level = "nothing"
        op = "level"
    elif op == "level":
        return f"{what!r} is not a level: everything, work only, nothing personal or nothing"
    elif not scopes_of(w):
        return f"nothing I know in {what!r}: name sections (texts, email, health...) or a level (work only...)"
    c = _norm(client or "")
    known = set(NAMES) | set(m.client_policies())
    names = targets(m, c, requester, op != "hide") if re.match(_ALLW, c) or agents_in(c) else (
        [c] if c in known else [])
    if not names:
        return f"no agent called {client!r}; known: {', '.join(sorted(known))}"
    change_fn, op, level, scopes = _mutation(m, op, scopes_of(w), level)
    if op != "level" and not scopes:
        return f"nothing I know in {what!r}"
    out, applied, batch = [], [], int(time.time() * 1000)
    for name in names:
        b, a = preview(m, name, change_fn)
        way = direction(b, a)
        if way == "same":
            out.append(f"{name_of(name)}: no change")
        elif way == "tighten":
            b, a = _narrow(m, name, change_fn, f"tool:{requester}", batch=batch)
            applied.append((name, b, a))
            out.append(f"{name_of(name)} can no longer see {labels(set(a) - set(b))}" if set(a) != set(b) else
                       f"{name_of(name)}: no change")
        else:
            request(m, requester, name, op, scopes, level)
            out.append(f"asked the user to let {name_of(name)} see more ({_what_words(m, op, scopes, level)}); it "
                       "applies only if they approve it by typing")
    _tell(applied, f"tool:{requester}")
    return "; ".join(out) + "." + _off_note(m)


# ---------- CLI ----------

AGENT_PROCS = ("claude", "codex", "cursor", "cursor helper", "node", "windsurf", "copilot", "aider", "goose")
# The system's own tools, by absolute path and with a minimal environment: never whatever an agent put first on PATH
# (review 2026-11-08: a fake osascript printing "button returned:Allow").
PS, OSASCRIPT, SAFE_ENV = "/bin/ps", "/usr/bin/osascript", {"PATH": "/usr/bin:/bin"}


def agent_ancestor(pid=None, run=None):
    """The name of an AI app among this process's ancestors (/bin/ps, up to launchd), "unknown (...)" when the walk
    cannot be read (fails closed), or None. A secondary refusal only: an agent can clear its env markers and drive a
    pty, but not leave its own process tree. Finding no agent proves nothing about a person; the dialog is the gate."""
    run = run or subprocess.run
    pid, seen = pid or os.getppid(), set()
    while pid != 1:
        if pid in seen or pid <= 0:
            return "unknown (a broken process chain)"
        seen.add(pid)
        try:
            out = run([PS, "-o", "ppid=,comm=", "-p", str(pid)], capture_output=True, text=True, timeout=5,
                      env=SAFE_ENV).stdout
        except (OSError, subprocess.SubprocessError):
            return "unknown (ps failed)"
        parts = (out or "").strip().split(None, 1)
        if len(parts) < 2 or not parts[0].isdigit():
            return "unknown (ps gave nothing readable)"
        name = os.path.basename(parts[1]).lower()
        if any(name == a or name.startswith(a + " ") or name.startswith(a + ".") for a in AGENT_PROCS):
            return name
        pid = int(parts[0])
    return None


def dialog_allows(name, level, run=None):
    """The widening dialog (ask_dialog): "Let <app> see <level> in your memory?"."""
    return ask_dialog(f"Let {name} see {level} in your memory?", "Memory access", run=run)


def ask_dialog(text, title="Memory", run=None):
    """A macOS dialog the user answers with the mouse (an agent's shell cannot click it): True on Allow. False on
    Cancel, after two minutes, or with no GUI session to show it in. The one gate for what only the user may grant:
    widening access (dialog_allows) and trusting memories as their own words (mem.trust_ids)."""
    if run is None and os.environ.get("LFLABS_MEMORY_UNDER_TEST") == "1":
        return False  # a test never puts a real dialog on the user's screen
    run = run or subprocess.run
    text = "".join(c for c in str(text) if c.isprintable() or c == "\n")[:400]

    def quoted(t):  # an AppleScript string: only \\, \" and \n escaped, every other character as it is
        return '"' + t.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'
    script = (f'display dialog {quoted(text)} with title {quoted(str(title))} buttons {{"Cancel", "Allow"}} '
              'default button "Cancel" cancel button "Cancel" giving up after 120 with icon caution')
    try:
        r = run([OSASCRIPT, "-e", script], capture_output=True, text=True, timeout=130, env=SAFE_ENV)
    except (OSError, subprocess.SubprocessError):
        return False
    out = (r.stdout or "").strip()
    return r.returncode == 0 and out == "button returned:Allow, gave up:false"  # exactly the dialog's answer


def human_at_terminal(m, name, ask=input, level="more", dialog=None, ancestor=None):
    """None when the user, at a real Terminal outside any AI app, typed the app's name and then pressed Allow in a
    macOS dialog to let it see more; else why not (review 2026-11-08: env markers and a terminal can be faked by an
    agent with a shell, a dialog it cannot answer). Narrowing never asks."""
    dialog, ancestor = dialog or dialog_allows, ancestor or agent_ancestor
    if m.run_by_agent():
        return "refused: an AI app ran this in its own shell; widening access needs you at a Terminal"
    if (who := ancestor()) is not None:
        return f"refused: this runs inside {who}; widen access from a Terminal window that no AI app started"
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return "refused: widening access asks you to type the app's name, and there is no terminal to ask in"
    try:
        said = ask(f"This lets {name} see {level} of your memory. Type {name} to go ahead: ").strip().lower()
    except EOFError:
        said = ""
    if said != str(name).lower():
        return "not changed"
    if not dialog(name, level):
        return "not changed: it was not allowed in the dialog (it needs a click on Allow, on this Mac's screen)"
    return None


KEY_BY_ENV = "LFLABS_MEMORY_KEY_BY"  # brand.COMMAND: the product's CLI keys the app after `clients level` (cli, level)


def cli(m, args):
    cmd = args[0]
    if cmd == "level":
        if len(args) < 3 or args[2] not in LEVEL_LABEL:
            return f"usage: {m.CFG.command} clients level NAME everything|work|nothing-personal|nothing"
        change = _mutation(m, "level", level=args[2])[0]
        if direction(*preview(m, args[1], change)) == "loosen" and (
                why := human_at_terminal(m, args[1], level=LEVEL_LABEL.get(args[2], args[2]))):
            return why
        b, a = m.clients_put(args[1], change, "cli")
        # The product's CLI (`<command> access`) writes the app's key itself right after this, into the entry it made:
        # this neither keys it nor says it has no key. Only words and this keying are left out; the gate above ran
        # all the same, so the variable gives an agent that sets it nothing (VM run, v0.1.8-rc2, 2026-11-10).
        by_cli = os.environ.get(KEY_BY_ENV) == brand.COMMAND
        keys = _keys(m, [args[1]]) if set(m.policy_hide(b)) - set(m.policy_hide(a)) and not by_cli else []
        return describe(m, args[1], key_hint=not by_cli) + "".join(f"\n{k}" for k in keys)
    if cmd == "history":
        n = int(args[1]) if len(args) > 1 and args[1].isdigit() else 20
        rows = history(m)[-n:]
        if not rows:
            return "no access changes yet"
        out = []
        for r in rows:
            def summ(h):
                return level_name(h, m) or ("hides " + ", ".join(h))
            out.append(f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(r['at']))}  {r['client']:<14} "
                       f"{r['dir']:<8} {summ(r['hidden_before'])} -> {summ(r['hidden_after'])}  ({r['source']}"
                       + (", an undo" if r.get("undoes") else "") + ")")
        return "\n".join(out)
    if cmd == "requests":
        reqs = requests(m)
        return "\n".join(f"{i}  {time.strftime('%m-%d %H:%M', time.localtime(r['at']))}  {r['by']} asks: let "
                         f"{r['target']} {_ask_words(m, r)}" for i, r in enumerate(reqs)) or "no pending requests"
    if cmd in ("approve", "undo"):  # an approval widens; an undo may (it takes back a narrowing too)
        if cmd == "approve":
            reqs = requests(m)
            i = int(args[1]) if len(args) > 1 and args[1].isdigit() else 0
            who = reqs[i]["target"] if 0 <= i < len(reqs) else None
        else:
            last = history(m)[-1:] or [{}]
            who = last[0].get("client")
        if who and (why := human_at_terminal(m, who)):
            return why
    if cmd == "approve":
        r = approve(m, index=int(args[1]) if len(args) > 1 and args[1].isdigit() else None, source="cli-approve")
        if r is None:
            return "nothing to approve"
        if r[0] != "applied":
            return r[1] + f" ({m.CFG.command} clients approve N, N from `{m.CFG.command} clients requests`)"
        return describe(m, r[1]) + "".join(f"\n{k}" for k in _keys(m, [r[1]]))
    if cmd == "undo":
        r = undo(m, source="cli-undo")
        return r if isinstance(r, str) else "undid it; " + "; ".join(describe(m, n) for n, *_ in r)
    return m._clients_cli.__doc__
