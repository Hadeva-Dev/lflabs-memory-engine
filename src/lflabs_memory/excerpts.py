"""Query-focused, verbatim excerpts for the prompt hook's memory lines (EXCERPT_MODE in mem.py; memory-bench
excerpts-precise, 2026-10-21). The selection code is the bench's final version (branch excerpts-precise), moved
here unchanged; only the bench's switch is left out.

mem._excerpt with EXCERPT_MODE off keeps the first HOOK_CHARS (1,000) characters of a memory: a blind head cut, so a
long log is cut mid-sentence before the part the prompt is about. focus() instead picks the memory's sentences and
clauses that match the prompt, verbatim, in their original order, up to a budget, joined with " … " where text was
skipped.

Precision rules (the point of it; precision_issues() checks every one on its output):
- text is only selected, never rewritten: each kept piece is a whole unit (a line, a sentence, or a clause of a long
  sentence) of the same text the head cut would show (plain() and date_words() as now), so nothing is cut mid-word
  or mid-number;
- the memory's subject anchor (its first clause) is always kept, so the excerpt stands alone;
- a later clause of a split sentence brings that sentence's first clause; a unit that opens with a pronoun or a
  continuation word ("It", "This", "Also", "which") brings the unit before it; a line under a "Header:" line brings
  the header: a number or date is never shown without its subject;
- a unit with a negation or a supersession marker ("not", "no longer", "instead", "supersedes", "later decision")
  that shares a name, number or two content words with a kept unit is kept too, and so is a later "Key: value" line
  with the same key: a kept unit never loses what negates or replaces it. When a unit's closure does not fit, the
  unit is skipped rather than shown without it;
- a memory that fits the budget is shown whole, exactly as today; quoted_mark and every scope and sensitivity
  decision are untouched (they act before or around _excerpt, on the whole memory: an excerpt only ever selects
  from one memory that was already allowed whole).
"""
import math
import re
from functools import lru_cache

STOP = frozenset("""a about above after again against all am an and any are as at be because been before being below
between both but by can could did do does doing down during each few for from further had has have having he her here
hers herself him himself his how i if in into is it its itself just me more most my myself nor of off on once only or
other our ours ourselves out over own same she should so some such than that the their theirs them themselves then
there these they this those through to too under until up very was we were what when where which while who whom why
will with would you your yours yourself yourselves also get got let lets s t d ll re ve m o y im ive id dont doesnt
didnt cant wont isnt arent wasnt werent hasnt havent hadnt couldnt shouldnt wouldnt please tell know remember
recall again ok okay hey hi thanks thank need want like just one two thing things anything something""".split())
# Negation, contrast and supersession: a unit carrying one of these may change what a nearby unit means.
NEG_RE = re.compile(r"\b(?:not|no|never|none|nor|neither|cannot|can't|won't|don't|doesn't|didn't|isn't|aren't|"
                    r"wasn't|weren't|hasn't|haven't|hadn't|shouldn't|wouldn't|couldn't|without|except|unless|"
                    r"no longer|instead|rather than|anymore|any more)\b", re.I)
SUPERSEDE_RE = re.compile(r"\b(?:supersed\w*|replac\w*|instead|no longer|not any ?more|later decision|now (?:is|are|"
                          r"uses?|at|in|on|says?|wants?|prefers?|lives?|works?)|changed? (?:to|from)|switched|"
                          r"moved (?:to|from)|updated?|correction|corrected|revised|reversed|overrid\w*|scrapped|"
                          r"dropped|cancel+ed|called off|postponed|rescheduled|pushed (?:back|to)|retired|was wrong|"
                          r"actually|turned out|turns out|after all|decided against|changed (?:my|his|her|their) mind|"
                          r"from now on|going forward|decided|finally|in the end|ended up|settled on|chose|confirmed)\b"
                          r"|\b(?:decision|outcome|result|update|final(?: decision| plan)?)\s*:", re.I)
# A unit that says the one before it is wrong or out of date.
CORRECT_RE = re.compile(r"^(?:[\"'(\[]\s*)?(?:that|this|it|which|these|those)(?:'s|’s| is| was| are| were)?\s+(?:not|no longer|"
                        r"wrong|incorrect|outdated|out of date|old|false|untrue)\b", re.I)
# A unit that opens with one of these leans on the unit before it for its subject.
LEAN_RE = re.compile(r"^(?:[\"'(\[]\s*)?(?:it|its|it's|this|that|these|those|they|them|their|he|she|his|her|him|"
                     r"there|then|also|same|which|who|whose|both|either|neither|such|so|thus|hence|plus|and|or|"
                     r"but|yet|otherwise|meanwhile|afterwards|later|still|too|as well|here)\b", re.I)
# A clause that opens with one of these stays glued to the clause before it.
GLUE_RE = re.compile(r"^(?:not|no|never|nor|but|instead|except|unless|although|though|however|without|rather|"
                     r"only|yet|so|which|who|whose|that|then|and|or|because|since|until|while|if|as|than|"
                     r"plus|including|e\.g\.|i\.e\.)\b", re.I)
ABBR = frozenset("dr mr mrs ms st vs etc eg ie no nos approx jr sr prof inc ltd co dept fig vol mt ft ave rd "
                 "jan feb mar apr jun jul aug sep sept oct nov dec mon tue tues wed thu thur thurs fri sat sun "
                 "a.m p.m e.g i.e u.s u.k".split())
SENT_END = re.compile(r"[.!?…][\"')\]]*\s+")
CLAUSE_CUT = re.compile(r";\s+|\s+[—–]\s+|\s+-\s+")
HEAD_CUT = re.compile(r";\s+|\s+[—–]\s+|\s+-\s+|:\s+|,\s+")
WORD = re.compile(r"[^\W_]+(?:[.'’-][^\W_]+)*", re.U)
NUM = re.compile(r"\d")
KEY_RE = re.compile(r"^\s*(?:[-*•]\s+)?([A-Za-z][\w /&'-]{0,40}?):\s+\S")
BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d{1,2}[.)])\s+")
LONG = 200          # a sentence longer than this is split into clauses
HEAD_LONG = 120     # the memory's first sentence, longer than this, is split for its anchor
GAP = " … "


@lru_cache(maxsize=65536)
def stem(w):
    """A light suffix stripper: "podcasts" and "podcast", "planned", "planning" and "plan" meet."""
    w = w.lower().replace("’", "'")
    if w.endswith("'s"):
        w = w[:-2]
    if len(w) <= 3 or not w.isalpha():
        return w
    for suf, rep in (("ies", "y"), ("sses", "ss"), ("ing", ""), ("edly", ""), ("ed", ""), ("ly", ""), ("es", ""),
                     ("s", "")):
        if w.endswith(suf) and len(w) - len(suf) + len(rep) >= 3 and not (suf == "s" and w[-2] in "sui"):
            w = w[: len(w) - len(suf)] + rep
            break
    if len(w) > 4 and w[-1] == w[-2] and w[-1] not in "lsz":  # planning -> plann -> plan
        w = w[:-1]
    return w[:-1] if len(w) > 4 and w.endswith("e") else w


def terms(text):
    """{stem: weight class} of a text's content words: 3 a number, 2 a capitalised word or one of 7+ letters, 1 other."""
    out = {}
    for m in WORD.finditer(text):
        t = m.group(0)
        low = t.lower().replace("’", "'")
        if low in STOP or (len(low) < 2 and not low.isdigit()):
            continue
        w = 3 if NUM.search(t) else 2 if (t[0].isupper() and m.start() > 0) or len(low) >= 7 else 1
        s = stem(low)
        out[s] = max(out.get(s, 0), w)
    return out


# ---------- units ----------

class Unit:
    __slots__ = ("raw", "text", "line", "sent", "clause", "bullet", "key", "header", "terms", "tset", "strong", "neg",
                 "sup", "lean", "corrects")

    def __init__(self, raw, line, sent, clause, bullet, key, header):
        self.raw, self.line, self.sent, self.clause = raw, line, sent, clause
        self.bullet, self.key, self.header = bullet, key, header
        self.text = ""
        self.terms = terms(raw)
        self.tset = frozenset(self.terms)
        self.strong = frozenset(t for t, w in self.terms.items() if w >= 2)
        self.neg = bool(NEG_RE.search(raw))
        self.sup = bool(SUPERSEDE_RE.search(raw))
        self.lean = bool(LEAN_RE.match(raw.strip()))
        self.corrects = self.lean and (self.sup or bool(CORRECT_RE.match(raw.strip())))


def _sentences(line):
    """A line's sentences, cut after . ! ? only where the next word starts a sentence and the word before is not an
    abbreviation or an initial ("Dr. Mensah", "J. Smith", "e.g. tea", "3.5" never split)."""
    out, start = [], 0
    for m in SENT_END.finditer(line):
        end = m.end()
        nxt = line[end:end + 1]
        if not nxt or not (nxt.isupper() or nxt.isdigit() or nxt in "\"'(["):
            continue
        before = line[start:m.start() + 1].rstrip(".!?…\"')]")
        last = before.split()[-1].lower().lstrip("([\"'").rstrip(".") if before.split() else ""
        if line[m.start()] == "." and (last in ABBR or (len(last) == 1 and last.isalpha())):
            continue
        out.append(line[start:m.end()].strip())
        start = end
    if line[start:].strip():
        out.append(line[start:].strip())
    return out


def _clauses(sent, cut=CLAUSE_CUT, long=LONG):
    """A long sentence cut at "; " and spaced dashes (the memory's first sentence over HEAD_LONG also at ": " and
    ", ", so its subject anchor is short); a clause that opens with a negation, contrast or continuation word, or a
    number, or is short, stays with the one before it."""
    if len(sent) <= long:
        return [sent]
    parts, start = [], 0
    for m in cut.finditer(sent):
        parts.append(sent[start:m.start()] + sent[m.start():m.end()].rstrip())
        start = m.end()
    parts.append(sent[start:])
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if out and (GLUE_RE.match(p) or NUM.match(p) or len(p) < 30 or len(out[-1]) < 30):
            out[-1] = out[-1] + " " + p
        else:
            out.append(p)
    return out


def units(raw_text):
    """The units of a memory's raw text (before plain()): lines, then sentences, then clauses of long sentences."""
    out, header = [], None
    for li, line in enumerate(x for x in raw_text.split("\n") if x.strip()):
        bullet = bool(BULLET_RE.match(line))
        km = KEY_RE.match(line)
        if not bullet:
            header = None
        for si, sent in enumerate(_sentences(line.strip())):
            first = not out and si == 0
            for ci, cl in enumerate(_clauses(sent, HEAD_CUT, HEAD_LONG) if first else _clauses(sent)):
                u = Unit(cl, li, (li, si), ci, bullet and si == 0 and ci == 0,
                         km.group(1).strip().lower() if km and si == 0 and ci == 0 else None,
                         header if bullet else None)
                out.append(u)
        if not bullet and line.rstrip().endswith(":"):
            header = len(out) - 1
    return out


# ---------- what a unit needs ----------

def _strong_overlap(a, b):
    """Two units are about the same thing: a shared number or capitalised/long word, or two shared content words."""
    shared = a.tset & b.tset
    return len(shared) >= 2 or bool(shared & (a.strong | b.strong))


class needs(dict):
    """{unit index: indices it cannot be shown without}, worked out for a unit when first asked (a long memory's
    units are mostly never asked about)."""

    def __init__(self, us):
        super().__init__()
        self.us = us
        self.first_of_sent = {}
        for i, u in enumerate(us):
            self.first_of_sent.setdefault(u.sent, i)
        self.markers = [j for j, v in enumerate(us) if v.sup]
        self.keys = {}
        for j, v in enumerate(us):
            if v.key:
                self.keys.setdefault(v.key, []).append(j)

    def __missing__(self, i):
        us, u, r = self.us, self.us[i], set()
        if u.clause > 0:
            r.add(self.first_of_sent[u.sent])
        if i > 0 and (u.lean or NUM.search(u.raw) and not any(re.search(r"[^\W\d_]{3,}", t) for t in u.terms)):
            r.add(i - 1)  # "It costs 40." or a bare "$400." leans on the unit before it
        if u.header is not None and u.header != i:
            r.add(u.header)
        if u.key:
            r |= {j for j in self.keys[u.key] if j > i}
        if i + 1 < len(us) and us[i + 1].corrects:
            r.add(i + 1)  # "Mei lives in Oslo. That is no longer true." : the correction comes with it
        for j in self.markers:  # a later unit that replaces it ("Later decision: ... instead") comes with it
            if j > i and us[j].sup and _strong_overlap(u, us[j]):
                r.add(j)
        self[i] = r
        return r


def closure(i, req, lens=None, cap=None):
    """Unit i and everything it needs, transitively; None once their text passes cap characters (it cannot fit)."""
    seen, todo, n = {i}, [i], 0
    while todo:
        k = todo.pop()
        if cap is not None:
            n += lens[k] + 1
            if n > cap:
                return None
        for j in req[k]:
            if j not in seen:
                seen.add(j)
                todo.append(j)
    return seen


# ---------- scoring and selection ----------

COVERED = 0.4       # a prompt word already shown by a kept unit is worth this much again,
COVERED_NAME = 0.1  # and a name already shown (the memory's subject, said again) this much


def score_units(us, query, qvec=None, uvecs=None, emb_weight=0.0):
    """Per unit, {prompt word: its value there} (number 3, name or long word 2, other 1, times its rarity in this
    memory), and a flat bonus (meaning, or the header it sits under)."""
    q = terms(query)
    n = len(us)
    df = {}
    for u in us:
        for t in u.terms:
            df[t] = df.get(t, 0) + 1
    hits, bonus = [], []
    for k, u in enumerate(us):
        hits.append({t: max(w, u.terms[t]) * math.log(1 + (n + 1) / (df[t] + 0.5)) for t, w in q.items()
                     if t in u.terms})
        b = 0.0
        if qvec is not None and uvecs is not None:
            b += emb_weight * max(0.0, float(uvecs[k] @ qvec) - 0.5) * 10
        bonus.append(b)
    for k, u in enumerate(us):  # a list's items answer what its header names ("Podcasts I like:")
        if u.header is not None:
            bonus[k] += 0.8 * sum(hits[u.header].values())
    names = {stem(m.group(0)) for u in us for m in WORD.finditer(u.raw) if m.start() > 0 and m.group(0)[0].isupper()}
    return hits, bonus, names & set(q)


def gain(hits, bonus, k, covered, names=frozenset()):
    return bonus[k] + sum(v * ((COVERED_NAME if t in names else COVERED) if t in covered else 1.0)
                          for t, v in hits[k].items())


def render(us, keep):
    """Kept units in their order, " … " where text was skipped, "…" at the end if the memory goes on."""
    out, prev = "", None
    for i in sorted(keep):
        if prev is None:
            out = ("… " if i > 0 else "") + us[i].text
        else:
            out += (" " if i == prev + 1 else GAP) + us[i].text
        prev = i
    return out + (" …" if prev is not None and prev < len(us) - 1 else "")


@lru_cache(maxsize=512)
def _parsed(raw_text, plain, dates):
    """A memory's units and what each needs: the same for every prompt, so a warm process parses a memory once."""
    us = units(raw_text)
    for u in us:
        u.text = dates(plain(u.raw))
    return us, needs(us)


def focus(raw_text, query, budget, plain=lambda s: " ".join(s.split()), dates=lambda s: s, embed=None, emb_weight=0.0,
          fill_by_vector=False, fill_past_misfits=False):
    """(excerpt, kept unit indices or None when the memory is shown whole). raw_text: the memory's text before
    plain(); plain and dates: mem.plain and mem.date_words. fill_past_misfits: the room left is filled with every later
    unit whose closure still fits, not only up to the first that does not (the recall tool's PRECISE_RECALL_MODE)."""
    whole = dates(plain(raw_text))
    if len(whole) <= budget:
        return whole, None
    us, req = _parsed(raw_text, plain, dates)
    if not us:
        return whole[:budget], None
    qvec = uvecs = None
    if embed is not None and (emb_weight or fill_by_vector):
        qvec, uvecs = embed(query, [u.text for u in us])
    hits, bonus, names = score_units(us, query, qvec, uvecs, emb_weight)
    keep = closure(0, req)
    lens = [len(u.text) for u in us]
    last = len(us) - 1

    def size(ks):  # len(render(us, ks)), without building it
        ks = sorted(ks)
        n = (2 if ks[0] > 0 else 0) + (2 if ks[-1] < last else 0)
        for a, b in zip(ks, ks[1:]):
            n += 1 if b == a + 1 else len(GAP)
        return n + sum(lens[k] for k in ks)

    covered = {t for k in keep for t in hits[k]}
    cands = {i for i in range(len(us)) if i not in keep and (hits[i] or bonus[i] > 0)}
    while cands:  # the unit adding the most prompt words not yet shown, with what it needs, while it fits
        best, best_g, best_c = None, 0.0, None
        for i in sorted(cands):
            g = gain(hits, bonus, i, covered, names)
            if g > best_g:
                c = closure(i, req, lens, budget)
                if c is not None and size(keep | c) <= budget:
                    best, best_g, best_c = i, g, c
        if best is None:
            break
        keep = keep | best_c
        covered |= {t for k in best_c for t in hits[k]}
        cands -= keep
    if fill_by_vector and qvec is not None:  # room left: the units nearest the prompt by meaning
        for i in sorted(range(len(us)), key=lambda k: -float(uvecs[k] @ qvec)):
            if i in keep:
                continue
            c = closure(i, req, lens, budget)
            if c is not None and size(keep | c) <= budget:
                keep = keep | c
        return render(us, keep), keep
    for i in range(len(us)):  # room left: the head, in order, as today's cut would show, up to the first misfit
        if i in keep:
            continue
        c = closure(i, req, lens, budget)
        if c is None or size(keep | c) > budget:
            if fill_past_misfits:
                continue
            break
        keep = keep | c
    return render(us, keep), keep


# ---------- the precision audit ----------

def precision_issues(raw_text, excerpt, keep, plain=lambda s: " ".join(s.split()), dates=lambda s: s):
    """Rule violations in one focused excerpt, worked out from the units again (independent of the selection loop):
    a piece that is not verbatim or not cut at a unit boundary, a kept number or leaning unit without what it leans
    on, a kept unit without a later unit that negates or replaces it."""
    issues = []
    if keep is None:
        return issues
    whole = dates(plain(raw_text))
    us = units(raw_text)
    for u in us:
        u.text = dates(plain(u.raw))
    body = excerpt.removeprefix("… ").removesuffix(" …")
    for piece in body.split(GAP):
        at = whole.find(piece)
        if at < 0:
            issues.append(("not_verbatim", piece[:80]))
            continue
        a, b = whole[at - 1:at] if at else "", whole[at + len(piece):at + len(piece) + 1]
        if (a and a.isalnum()) or (b and b.isalnum()):
            issues.append(("mid_word_cut", piece[:80]))
    return issues + unit_issues(us, keep)


def unit_issues(us, keep):
    """The unit rules over a set of shown units (for the head cut: the units it shows whole)."""
    issues = []
    for i in sorted(keep):
        u = us[i]
        if u.lean and i > 0 and i - 1 not in keep:
            issues.append(("orphaned_lean", u.text[:80]))
        if NUM.search(u.raw) and not any(re.search(r"[^\W\d_]{3,}", t) for t in u.terms) and i > 0 \
                and i - 1 not in keep and u.header is None:
            issues.append(("orphaned_number", u.text[:80]))
        if u.clause > 0 and not any(us[j].sent == u.sent and us[j].clause == 0 for j in keep):
            issues.append(("orphaned_clause", u.text[:80]))
        if u.header is not None and u.header not in keep:
            issues.append(("orphaned_bullet", u.text[:80]))
        if i + 1 < len(us) and i + 1 not in keep and us[i + 1].corrects:
            issues.append(("correction_dropped", f"{u.text[:60]} || {us[i + 1].text[:60]}"))
        if u.clause == 0 and any(us[j].sent == u.sent and us[j].clause > 0 and j not in keep and GLUE_RE.match(us[j].raw)
                                 for j in range(i + 1, min(len(us), i + 8))):
            issues.append(("negation_split", u.text[:80]))
        for j in range(i + 1, len(us)):
            v = us[j]
            if j in keep:
                continue
            if v.sup and _strong_overlap(u, v):
                issues.append(("superseder_dropped", f"{u.text[:60]} || {v.text[:60]}"))
            elif u.key and v.key == u.key:
                issues.append(("key_superseded", f"{u.text[:60]} || {v.text[:60]}"))
    return issues


def head_issues(raw_text, shown, cut, plain=lambda s: " ".join(s.split()), dates=lambda s: s):
    """The same checks for today's head cut, on the text it cut (shown, before "…"): a cut mid-word or mid-number,
    and the unit rules over the units it shows whole (a statement shown without what replaces it, ...)."""
    if len(cut) <= len(shown):
        return []
    issues = []
    n = len(shown)
    if cut[n - 1].isalnum() and cut[n].isalnum():
        issues.append(("mid_word_cut", shown[-40:]))
    us = units(raw_text)
    for u in us:
        u.text = dates(plain(u.raw))
    keep = {i for i, u in enumerate(us) if u.text in shown}
    return issues + [x for x in unit_issues(us, keep) if x[0] not in ("orphaned_lean", "orphaned_number")]


def slice_issues(raw_text, pieces, plain=lambda s: " ".join(s.split()), dates=lambda s: s):
    """The same checks for a cut that shows stretches of a memory's text as they are (the recall tool's window, which
    a deep match follows with its first line): a stretch that starts or ends mid-word, and the unit rules over the
    units it shows whole. pieces: the stretches as shown (dates already applied), "…" at either end on or off."""
    whole = dates(plain(raw_text))
    us, _ = _parsed(raw_text, plain, dates)
    issues, shown = [], []
    for piece in pieces:
        p = plain(piece).removeprefix("…").removesuffix("…").strip()
        at = whole.find(p) if p else -1
        if at < 0:
            continue
        shown.append(p)
        a, b = whole[at - 1:at] if at else "", whole[at + len(p):at + len(p) + 1]
        if (a and a.isalnum()) or (b and b.isalnum()):
            issues.append(("mid_word_cut", p[:80]))
    keep = {i for i, u in enumerate(us) if any(u.text in p for p in shown)}
    return issues + unit_issues(us, keep)
