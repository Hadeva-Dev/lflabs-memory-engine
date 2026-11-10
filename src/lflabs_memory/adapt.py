"""Learning from its own user: the prompt hook's gate, corrected from the user's own use. Stdlib only, no model calls.

Three parts, all pure except the state file's load and save:
  mine(events)            one session's events -> labeled items (used, unused, corrected, miss)
  State                   the online calibrator (2-parameter logistic, Bayesian shrinkage to the bench prior) and the
                          threshold controller, with hard bounds; a small versioned JSON
  pick(cands, state)      what the adapted gate would inject; protected memories only ever stricter than the static gate

The design and its evaluation were done on memory-bench dev (the wedge study); they are not part of this package.
"""
import json
import math
import os
import re
import time

VERSION = 1

# ---------- which memories are protected (adaptation may only make them stricter) ----------

# Health, money and codes: the same words as the bench's search lab SENS_RE (what a hook can see, no model).
SENS_RE = re.compile(
    r"\b(doctor|dr\.|clinic|hospital|therap(y|ist)|psychiatr\w*|counsell?(or|ing)|diagnos\w*|medication|medicine|"
    r"prescri\w*|mg\b|dose|surgery|mri|scan|biopsy|pregnan\w*|ivf|fertility|miscarri\w*|anxiety|depress\w*|"
    r"panic|adhd|lorazepam|sertraline|blood (test|pressure)|cholesterol|cancer|tumou?r|std|hiv|"
    r"mental health|bone doctor|physio\w*|injur\w*|"
    r"salary|income|paid|pay(check|slip)?|debt|loan|mortgage|credit (score|card)|bank|account (number|no)|iban|"
    r"routing|savings|invest\w*|tax(es)?|bonus|raise|owe[sd]?|budget|\$\s?\d|\d+\s?(k|dollars|usd|eur|reais)\b|"
    r"password|passcode|pin\b|code is|garage code|wifi|passport|ssn|social security)", re.I)
REVIEW_HIDE = ("meetings", "texts", "email")      # mem.py REVIEW_HIDE
UNTRUSTED_ORIGINS = {"outside", "guest", "open_mic"}  # mem.py UNTRUSTED_ORIGINS


def protected(content, section=None, section2=None, origin=None, sens_re=SENS_RE):
    """A memory the adapted gate may never inject more readily than the static gate does."""
    return bool(sens_re.search(content or "") or section in REVIEW_HIDE or section2 in REVIEW_HIDE
                or (origin or "unknown") in UNTRUSTED_ORIGINS)


# ---------- the "used" signal: a memory's distinctive tokens in the assistant's reply ----------

STOP = set("""a about above after again against all also am an and any are as at be because been before being below
between both but by can could did do does doing down during each few for from further had has have having he her here
hers herself him himself his how i if in into is it its itself just me more most my myself no nor not now of off on
once only or other our ours out over own same she should so some such than that the their theirs them themselves then
there these they this those through to too under until up very was we were what when where which while who whom why
will with would you your yours yourself said says saved since still today tomorrow yesterday week weeks month months
year years day days time times last next first second third really maybe probably usually always never often thing
things something someone anyone everyone people person going want wants wanted like likes liked prefer prefers
preferred think thinks thought know knows knew asked asks tell told keep kept make makes made take takes took get gets
got give gives gave come comes came good great well back again around about remember remembers recalled memory
memories note notes project claude because however although whether either neither every""".split())
MONTHS = set("january february march april may june july august september october november december jan feb mar apr jun "
             "jul aug sep sept oct nov dec".split())
WEEKDAYS = set("monday tuesday wednesday thursday friday saturday sunday".split())
LINE_HEAD_RE = re.compile(r"^\s*-\s*#\d+\s+(\[[^\]]*\]\s*)*(\([^)]*\)\s*)?")  # "- #27 [kind/project] [origin] (saved ..) "
WORD_RE = re.compile(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'’-]*|\d+(?:[.:]\d+)?")


def _norm(w):
    w = w.lower().replace("’", "'")
    w = re.sub(r"'s$", "", w).strip("'-")
    if len(w) > 4 and w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    return w


def _numbers(text):
    """Numbers as values: 2026-03-12 gives 3 and 12 (years are left out: nearly every memory has one)."""
    out = set()
    for m in re.finditer(r"\d+(?:[.:]\d+)?", text):
        v = m.group().lstrip("0") or "0"
        if re.fullmatch(r"(19|20)\d\d", m.group()):
            continue
        out.add(v)
    return out


def tokens(text):
    """Every normalized word and number of a text (what a reply or a prompt is checked against)."""
    return {_norm(w) for w in WORD_RE.findall(text or "") if not w[0].isdigit()} | _numbers(text or "")


def distinctive(text, owner_words=()):
    """A memory's distinctive tokens: numbers, names (capitalised mid-sentence), rare long words. Dates count as their
    numbers; month and weekday names, the owner's name and common words do not count."""
    text = LINE_HEAD_RE.sub("", text or "")
    out = set(_numbers(text))
    owner = {w.lower() for w in owner_words}
    for m in WORD_RE.finditer(text):
        w = m.group()
        if w[0].isdigit():
            continue
        n = _norm(w)
        if n in STOP or n in MONTHS or n in WEEKDAYS or n in owner or len(n) < 3:
            continue
        before = text[:m.start()].rstrip()
        name = w[0].isupper() and before and before[-1] not in ".!?:\n"
        if name or len(n) >= 7:
            out.add(n)
    return out


USED_MIN = 3  # distinctive tokens a reply must share (not counting the prompt's own); see wedge/REPORT.md, used signal


def used_in(memory_text, reply, prompt="", owner_words=(), k=USED_MIN):
    """Whether a reply uses a memory: at least k of its distinctive tokens that the prompt did not already say, or
    every one of them when it has fewer than k (a memory whose one distinctive token is a time or a name)."""
    d = distinctive(memory_text, owner_words) - tokens(prompt)
    if not d:
        return False
    hit = len(d & tokens(reply))
    return hit >= min(k, len(d))


CORRECT_RE = re.compile(r"\b(that'?s (wrong|not right|not true|outdated|old|not me|not mine)|not (true|right|mine|me)|"
                        r"wrong|incorrect|outdated|out of date|no longer|not anymore|isn'?t (true|right|mine)|"
                        r"never said|not my|that was (before|years ago)|old (info|news)|stale)\b", re.I)


# ---------- the miner ----------

W_USED, W_UNUSED, W_CORRECTED, W_MISS, W_MISS_EXTRACT = 1.0, 0.5, 2.0, 1.0, 0.5


def mine(events, owner_words=(), reply_window=1):
    """Labeled items from one session's events, in order. No model calls.

    events (dicts, in order, each with "turn", an int that grows with each of the owner's prompts):
      {"type": "hook", "turn", "prompt", "cands": [{"id", "score", "p", "injected", "protected", "text"}]}
          one prompt hook decision: every candidate the gate saw, with its raw and calibrated score, whether the live
          gate injected it, and its text (text is needed only for injected ones)
      {"type": "user", "turn", "text"}                     the owner's own words (a correction is looked for here)
      {"type": "reply", "turn", "text"}                    the assistant's reply text
      {"type": "recall", "turn", "ids": [...], "texts": {id: text}}   a recall tool result
      {"type": "recall_full", "turn", "id"}                the brain opened one memory whole
      {"type": "forget", "turn", "id"}                     the brain retired a memory
      {"type": "remember", "turn", "id", "action", "via"}  a save that landed on an existing memory (updated or
                                                          unchanged: a restatement); via "tool" or "extract" (turn None)
    Returns [{"label", "hook_turn", "id", "score", "p", "protected", "weight", "y"}]: y 1 or 0 for the calibrator;
    label "miss_unretrieved" (the memory was never a candidate) has no score and only counts.
    """
    hooks = [e for e in events if e.get("type") == "hook"]
    replies = {}
    for e in events:
        if e.get("type") == "reply":
            replies.setdefault(e["turn"], []).append(e.get("text") or "")
    users = {e["turn"]: e.get("text") or "" for e in events if e.get("type") == "user"}
    injected_at = {}   # memory id -> (hook event, cand) where it was injected (first time this session)
    below_at = {}      # memory id -> [(hook event, cand)] where it was a candidate below the cut
    for h in hooks:
        for c in h.get("cands", []):
            if c.get("injected"):
                injected_at.setdefault(c["id"], (h, c))
            else:
                below_at.setdefault(c["id"], []).append((h, c))
    out, done = [], set()

    def item(label, h, c, weight, y):
        out.append({"label": label, "hook_turn": h["turn"], "id": c["id"], "score": c.get("score"), "p": c.get("p"),
                    "protected": bool(c.get("protected")), "weight": weight, "y": y})

    # Corrected: the brain retired an injected memory, or the owner's next words call it wrong and name it.
    corrected = set()
    for e in events:
        if e.get("type") == "forget" and e.get("id") in injected_at:
            h, c = injected_at[e["id"]]
            if e.get("turn", h["turn"]) >= h["turn"]:
                corrected.add(e["id"])
    for mid, (h, c) in injected_at.items():
        if mid in corrected:
            continue
        for t in (h["turn"], h["turn"] + 1):
            text = users.get(t) if t != h["turn"] else None  # the turn's own prompt came before the injection
            if text and CORRECT_RE.search(text) and (
                    f"#{mid}" in text or len(distinctive(c.get("text"), owner_words) & tokens(text)) >= 1):
                corrected.add(mid)
    for mid in sorted(corrected):
        h, c = injected_at[mid]
        item("corrected", h, c, W_CORRECTED, 0)
        done.add(mid)

    # Used / unused: the injected memory's distinctive tokens in a reply of its turn (or the next reply_window turns),
    # else in no reply for the rest of the session.
    last_turn = max([e.get("turn") or 0 for e in events] or [0])
    for mid, (h, c) in injected_at.items():
        if mid in done:
            continue
        near = " ".join(t for k in range(h["turn"], h["turn"] + reply_window + 1) for t in replies.get(k, []))
        if used_in(c.get("text"), near, h.get("prompt", ""), owner_words):
            item("used", h, c, W_USED, 1)
        else:
            rest = " ".join(t for k in range(h["turn"] + reply_window + 1, last_turn + 1) for t in replies.get(k, []))
            if not used_in(c.get("text"), rest, h.get("prompt", ""), owner_words):
                item("unused", h, c, W_UNUSED, 0)
        done.add(mid)

    # Miss by recall: after a hook turn, the brain recalled a memory the hook did not inject and used it.
    for e in events:
        if e.get("type") not in ("recall", "recall_full"):
            continue
        ids = [e["id"]] if e["type"] == "recall_full" else e.get("ids", [])
        for mid in ids:
            if mid in done or mid in injected_at:
                continue
            if e["type"] == "recall":
                text = (e.get("texts") or {}).get(mid) or (e.get("texts") or {}).get(str(mid)) or ""
                reply = " ".join(replies.get(e["turn"], []))
                prompt = next((h.get("prompt", "") for h in hooks if h["turn"] == e["turn"]), "")
                if not used_in(text, reply, prompt, owner_words):
                    continue
            hit = [(h, c) for h, c in below_at.get(mid, []) if h["turn"] <= e["turn"]]
            if hit:
                h, c = hit[-1]
                item("miss", h, c, W_MISS, 1)
            else:
                out.append({"label": "miss_unretrieved", "hook_turn": e["turn"], "id": mid, "score": None, "p": None,
                            "protected": False, "weight": 0.0, "y": 1})
            done.add(mid)

    # Miss by restatement: a save that dedups into a memory that was a below-cut candidate earlier in the session.
    for e in events:
        if e.get("type") != "remember" or e.get("action") not in ("updated", "unchanged"):
            continue
        mid = e.get("id")
        if mid in done or mid in injected_at:
            continue
        t = e.get("turn")
        hit = [(h, c) for h, c in below_at.get(mid, []) if t is None or h["turn"] <= t]
        if hit:
            h, c = hit[-1]
            item("miss", h, c, W_MISS if e.get("via", "tool") == "tool" else W_MISS_EXTRACT, 1)
            done.add(mid)
    return out


# ---------- the online state ----------

def sigmoid(z):
    return 1 / (1 + math.exp(-max(-30.0, min(30.0, z))))


def logit(p):
    p = min(1 - 1e-9, max(1e-9, p))
    return math.log(p / (1 - p))


DEFAULTS = {  # bench-derived numbers; wedge/REPORT.md says where each came from
    "anchor": "calib",     # what the cut is kept in: "calib" (a calibrated probability, clamped to the bench's range of
                           # raw cuts) or "quantile" (the share of the owner's own candidates that pass)
    "controller": "aci",   # none | recal | balance | aci  (recal and balance need anchor "calib"); REPORT.md: aci on
                           # the calibrated anchor is the arm shipped in shadow
    "n0": 300.0,         # the bench prior is worth this many labeled items (recal)
    "forget": 0.999,     # per update, the precision learned from the owner's labels decays toward the prior's by this
    "slope_box": (0.5, 2.0),   # the calibrator's slope stays within these multiples of the prior's
    "icpt_box": 3.0,     # and its intercept within this many logits of the prior's
    "fpr": 0.30,         # P(used signal | the memory was not relevant), measured on bench answers
    "fnr": 0.25,         # P(no used signal | it was relevant)
    "gamma": 0.02,       # aci step (logit units of the anchor)
    "gamma_bal": 0.005,  # balance step
    "d_hat": 0.3,        # assumed miss-detection rate (aci)
    "m0": 0.034,         # per-turn miss rate the bench prior runs at (aci target)
    "junk0": 0.66,       # the unused share the bench prior runs at
    "junk_margin": 0.05,  # the cut may not loosen while the unused share (EWMA) is this far above junk0
    "junk_alpha": 0.02,  # EWMA rate
    "hist_n0": 400.0,    # the bench's candidate-score histogram is worth this many candidates (quantile anchor)
    "hist_decay": 0.999,  # per candidate seen
}
PRIOR_KEYS = ("a", "b", "H", "lam", "lam_lo", "lam_hi", "hist")  # what State takes from a prior file (make_prior.py)


def prior_cfg(prior):
    """The controller's targets a prior file carries (the rest of the config stays DEFAULTS)."""
    return {k: v for k, v in (("m0", prior.get("m0")), ("junk0", prior.get("u0")), ("fpr", prior.get("fpr")),
                              ("fnr", prior.get("fnr"))) if v is not None}


HIST_LO, HIST_STEP, HIST_N = -20.0, 0.25, 160  # candidate-score histogram bins: [-20, 20) by 0.25


def hist_of(scores):
    h = [0.0] * HIST_N
    for s in scores:
        h[min(HIST_N - 1, max(0, int((s - HIST_LO) // HIST_STEP)))] += 1
    n = sum(h) or 1.0
    return [v / n for v in h]


def hist_cut(h, r):
    """The score above which a share r of the histogram's mass lies."""
    total = sum(h)
    want = min(max(r, 0.0), 1.0) * total
    acc = 0.0
    for k in range(HIST_N - 1, -1, -1):
        if h[k] > 0 and acc + h[k] >= want:
            frac = (want - acc) / h[k]
            return HIST_LO + (k + 1 - frac) * HIST_STEP
        acc += h[k]
    return HIST_LO


def pass_share(h, cut):
    """The share of the histogram's mass at or above a score (the inverse of hist_cut)."""
    total = sum(h) or 1.0
    above = 0.0
    for k in range(HIST_N):
        lo = HIST_LO + k * HIST_STEP
        if lo >= cut:
            above += h[k]
        elif lo + HIST_STEP > cut:
            above += h[k] * (lo + HIST_STEP - cut) / HIST_STEP
    return above / total


class State:
    """The adapted gate. theta is the cut's strictness in the anchor's units (higher is stricter): the logit of a
    calibrated probability, or minus the logit of the share of the owner's own candidates that pass. theta_prot, the
    protected cut, is the running maximum of theta. Also: the calibrator's coefficients and precision, the owner's
    candidate-score histogram, the unused-share EWMA, counts and a trajectory. Everything is bounded."""

    def __init__(self, prior, **cfg):
        """prior (the bench fit): a, b (Platt scaling of the raw logit), H (its Fisher information per item), lam (the static
        cut, a raw logit), lam_lo and lam_hi (the cross-validated range of cuts), hist (the bench's candidate-score
        histogram, hist_of())."""
        self.cfg = {**DEFAULTS, **cfg}
        self.prior = dict(prior)
        n0 = self.cfg["n0"]
        H = prior.get("H") or [[1.0, 0.0], [0.0, 0.05]]
        self.H0 = [[H[0][0] * n0, H[0][1] * n0], [H[1][0] * n0, H[1][1] * n0]]
        self.reset()

    def _theta_of_cut(self, cut, a=None, b=None, hist=None):
        if self.cfg["anchor"] == "calib":
            return (self.prior["a"] if a is None else a) * cut + (self.prior["b"] if b is None else b)
        return -logit(pass_share(self.prior["hist"] if hist is None else hist, cut))

    def reset(self):
        p = self.prior
        self.a, self.b = float(p["a"]), float(p["b"])
        self.H = [row[:] for row in self.H0]
        self.hist = [v * self.cfg["hist_n0"] for v in p.get("hist") or hist_of([p["lam"]])]
        self.theta = self.theta_prot = self.theta0 = self._theta_of_cut(p["lam"])
        t1, t2 = self._theta_of_cut(p["lam_lo"]), self._theta_of_cut(p["lam_hi"])
        self.lo, self.hi = min(t1, t2, self.theta0), max(t1, t2, self.theta0)
        self.junk = self.cfg["junk0"]
        self.counts = {k: 0 for k in ("sessions", "hooks", "used", "unused", "corrected", "miss", "miss_unretrieved",
                                      "loosen_blocked")}
        self.trajectory = []

    def p(self, score):
        """The calibrated probability that a candidate is relevant."""
        return sigmoid(self.a * score + self.b)

    def _cut(self, theta):
        if self.cfg["anchor"] == "calib":  # a recalibrated slope or intercept never takes the cut past the bench's range
            return min(max((theta - self.b) / self.a, self.prior["lam_lo"]), self.prior["lam_hi"])
        return hist_cut(self.hist, sigmoid(-theta))

    def cut(self):
        """The raw reranker logit a candidate needs."""
        return self._cut(self.theta)

    def cut_prot(self):
        return self._cut(self.theta_prot)

    def observe(self, scores):
        """The hook's candidate scores (no label needed): the quantile anchor follows the owner's own distribution."""
        f = self.cfg["hist_decay"]
        for s in scores:
            self.hist = [v * f for v in self.hist]
            self.hist[min(HIST_N - 1, max(0, int((s - HIST_LO) // HIST_STEP)))] += 1

    def _clamp(self):
        a0, b0 = self.prior["a"], self.prior["b"]
        s0, s1 = self.cfg["slope_box"]
        self.a = min(max(self.a, a0 * s0), a0 * s1)
        self.b = min(max(self.b, b0 - self.cfg["icpt_box"]), b0 + self.cfg["icpt_box"])
        self.theta = min(max(self.theta, self.lo), self.hi)
        self.theta_prot = max(self.theta_prot, self.theta)  # never falls

    def learn(self, score, y, weight=1.0, noisy=False):
        """One labeled item into the calibrator: an online Newton (Laplace) step from the bench prior."""
        if score is None or weight <= 0:
            return
        x = (float(score), 1.0)
        p = self.p(score)
        if noisy:  # the used signal: P(obs 1) = fpr + (1 - fpr - fnr) p
            fpr, fnr = self.cfg["fpr"], self.cfg["fnr"]
            k = max(1e-6, 1 - fpr - fnr)
            q = min(1 - 1e-6, max(1e-6, fpr + k * p))
            c = k * p * (1 - p)
            dz = c / q if y else -c / (1 - q)
            info = c * c / (q * (1 - q))
        else:
            dz = y - p
            info = p * (1 - p)
        f = self.cfg["forget"]
        H = [[self.H0[i][j] + f * (self.H[i][j] - self.H0[i][j]) + weight * info * x[i] * x[j] for j in range(2)]
             for i in range(2)]
        det = H[0][0] * H[1][1] - H[0][1] * H[1][0]
        if not det > 0 or not math.isfinite(det):
            return
        g = (weight * dz * x[0], weight * dz * x[1])
        self.a += (H[1][1] * g[0] - H[0][1] * g[1]) / det
        self.b += (-H[1][0] * g[0] + H[0][0] * g[1]) / det
        self.H = H
        self._clamp()

    def step(self, delta):
        """Move the cut by delta (positive is stricter); a loosening is refused while the unused share runs high."""
        if delta < 0 and self.junk > self.cfg["junk0"] + self.cfg["junk_margin"]:
            self.counts["loosen_blocked"] += 1
            return
        self.theta += delta
        self._clamp()

    def update(self, labels, hooks=0, missed_turns=0, scores=()):
        """One session: its mined labels (mine()), its number of hook turns, how many of them had a miss, and every
        candidate score its hooks saw (for the quantile anchor)."""
        ctl = self.cfg["controller"]
        self.counts["sessions"] += 1
        self.counts["hooks"] += hooks
        self.observe(scores)
        for it in labels:
            self.counts[it["label"]] = self.counts.get(it["label"], 0) + 1
            if it["label"] in ("used", "unused"):
                self.junk += self.cfg["junk_alpha"] * ((it["label"] == "unused") - self.junk)
        if ctl == "recal":
            for it in labels:
                if it["label"] in ("used", "unused"):
                    self.learn(it["score"], it["y"], it["weight"], noisy=True)
                elif it["label"] == "corrected":
                    self.learn(it["score"], 0, it["weight"])
        elif ctl == "balance":  # are the items just above the cut used as often as the calibrator says?
            fpr, fnr = self.cfg["fpr"], self.cfg["fnr"]
            k = max(1e-6, 1 - fpr - fnr)
            for it in labels:
                if it["label"] in ("used", "unused") and it["score"] is not None:
                    p = self.p(it["score"])
                    if 0 <= logit(p) - self.theta <= 1.5:
                        ytil = (it["y"] - fpr) / k  # an unbiased estimate of its relevance from the noisy signal
                        self.step(-self.cfg["gamma_bal"] * (ytil - p) / max(p * (1 - p), 0.05))
        elif ctl == "aci":  # adaptive conformal: the per-turn miss rate, corrected for detection, held at m0
            d = self.cfg["d_hat"]
            for t in range(hooks):
                err = (1.0 if t < missed_turns else 0.0) / d
                self.step(-self.cfg["gamma"] * (err - self.cfg["m0"]))
        self._clamp()
        self.trajectory.append([round(time.time()), round(self.theta, 4), round(self.cut(), 3),
                                round(self.cut_prot(), 3)])
        del self.trajectory[:-500]

    # ---------- the file ----------

    def to_json(self):
        return {"version": VERSION, "prior": self.prior, "cfg": self.cfg, "a": self.a, "b": self.b, "H": self.H,
                "hist": [round(v, 4) for v in self.hist], "theta": self.theta, "theta_prot": self.theta_prot,
                "junk": self.junk, "counts": self.counts, "trajectory": self.trajectory, "saved": time.time()}

    @classmethod
    def from_json(cls, d, prior, **cfg):
        """The saved state, or a fresh one from the prior when the file is missing, another version, for another
        prior or config, or not finite. Bounds are applied again on load."""
        st = cls(prior, **cfg)
        if (not d or d.get("version") != VERSION or d.get("prior") != json.loads(json.dumps(st.prior))
                or d.get("cfg") != json.loads(json.dumps(st.cfg))):
            return st
        try:
            vals = [float(d[k]) for k in ("a", "b", "theta", "theta_prot", "junk")]
            H = [[float(v) for v in row] for row in d["H"]]
            hist = [float(v) for v in d["hist"]]
        except (KeyError, TypeError, ValueError):
            return st
        if (len(hist) != HIST_N or not all(math.isfinite(v) for v in vals + H[0] + H[1] + hist)
                or min(hist) < 0 or not sum(hist) > 0):
            return st
        st.a, st.b, st.theta, st.theta_prot, st.junk = vals
        st.H, st.hist = H, hist
        st.counts.update({k: int(v) for k, v in (d.get("counts") or {}).items() if isinstance(v, (int, float))})
        st.trajectory = list(d.get("trajectory") or [])[-500:]
        st._clamp()
        return st


def load(path, prior, **cfg):
    try:
        with open(path) as f:
            d = json.load(f)
    except (OSError, ValueError):
        d = None
    return State.from_json(d, prior, **cfg)


def save(path, st):
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(st.to_json(), f)
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


# ---------- the adapted gate ----------

def pick(cands, st):
    """Indices of the candidates the adapted gate passes, best first. cands: the hook's candidates, each {"score" (the
    raw reranker logit), "protected", "static" (whether the static gate passes it)}. A protected candidate passes
    only if the static gate passes it AND it clears the protected cut, so adaptation never makes a protected memory
    more likely to be injected. The caller then applies the hook's own steps (HOOK_K, near-duplicates) and keeps a
    protected one only if the static gate's final list has it too."""
    cut, cut_prot = st.cut(), st.cut_prot()
    out = [i for i, c in enumerate(cands)
           if ((c.get("static") and c["score"] >= cut_prot) if c.get("protected") else c["score"] >= cut)]
    return sorted(out, key=lambda i: -cands[i]["score"])


# ---------- one session's events, from its transcript and the hook log ----------

TOOL_RE = re.compile(r"(?:^|__)(recall|recall_full|forget|remember)$")
SAVED_RE = re.compile(r"^(updated|unchanged) memory #(\d+)$")


def prompt_key(prompt):
    """What the hook log keeps of a prompt: a short hash to find its turn in the transcript, never the words."""
    import hashlib
    return hashlib.sha256(" ".join((prompt or "").split()).encode()).hexdigest()[:16]


def _blocks(d):
    content = (d.get("message") or {}).get("content")
    return [{"type": "text", "text": content}] if isinstance(content, str) else [b for b in content or []
                                                                                   if isinstance(b, dict)]


def _result_text(b):
    c = b.get("content")
    if isinstance(c, str):
        return c
    return "\n".join(x.get("text", "") for x in c or [] if isinstance(x, dict) and x.get("type") == "text")


def session_events(lines, hooks, memory_text, clean=lambda t: t.strip(), saves=()):
    """The events mine() reads, for one session.
    lines: the transcript's JSON lines (dicts), in order. hooks: the hook log's rows for this session, each
    {"key": prompt_key(prompt), "cands": [{"id", "score", "p", "injected", "protected"}]}. memory_text(id): a
    memory's current text, or None (only asked for injected and recalled ones). clean(text): the prompt as the hook
    saw it (mem.py strips the voice clock line). saves: (memory id, action) from the session's extraction, where a
    restated fact landed on an existing memory."""
    by_key = {}
    for h in hooks:
        by_key.setdefault(h["key"], []).append(h)
    events, turn, pending = [], 0, {}
    for d in lines:
        if d.get("isSidechain") or d.get("isMeta") or d.get("type") not in ("user", "assistant"):
            continue
        blocks = _blocks(d)
        if d["type"] == "user":
            results = [b for b in blocks if b.get("type") == "tool_result"]
            for b in results:
                call = pending.pop(b.get("tool_use_id"), None)
                if not call:
                    continue
                name, args = call
                text = _result_text(b)
                if name == "recall":
                    try:
                        items = json.loads(text.split("\n(", 1)[0])
                    except ValueError:
                        continue
                    ids = [it["id"] for it in items if isinstance(it, dict) and isinstance(it.get("id"), int)]
                    events.append({"type": "recall", "turn": turn, "ids": ids,
                                   "texts": {it["id"]: it.get("content", "") for it in items
                                             if isinstance(it, dict) and isinstance(it.get("id"), int)}})
                elif name == "recall_full" and text.startswith("#"):
                    try:
                        events.append({"type": "recall_full", "turn": turn, "id": int(args.get("id"))})
                    except (TypeError, ValueError):
                        pass
                elif name == "forget" and text.startswith("retired #"):
                    events.append({"type": "forget", "turn": turn, "id": int(text.split("#", 1)[1].split()[0])})
                elif name == "remember" and (m := SAVED_RE.match(text.strip())):
                    events.append({"type": "remember", "turn": turn, "id": int(m[2]), "action": m[1], "via": "tool"})
            if results or (d.get("origin") or {}).get("kind", "human") != "human":
                continue
            text = "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text")
            if not text.strip():
                continue
            turn += 1
            prompt = clean(text)
            events.append({"type": "user", "turn": turn, "text": prompt})
            hs = by_key.get(prompt_key(prompt))
            if hs:
                h = hs.pop(0)
                cands = []
                for c in h["cands"]:
                    c = dict(c)
                    if c.get("injected"):
                        c["text"] = memory_text(c["id"]) or ""
                    cands.append(c)
                events.append({"type": "hook", "turn": turn, "prompt": prompt, "cands": cands})
        else:
            for b in blocks:
                if b.get("type") == "text" and b.get("text"):
                    events.append({"type": "reply", "turn": turn, "text": b["text"]})
                elif b.get("type") == "tool_use" and (m := TOOL_RE.search(str(b.get("name", "")))):
                    pending[b.get("id")] = (m[1], b.get("input") or {})
    for mid, action in saves:
        if action in ("updated", "unchanged"):
            events.append({"type": "remember", "turn": None, "id": mid, "action": action, "via": "extract"})
    return events


def learn_session(st, events, owner_words=()):
    """Mine one session and update the state. Returns the labels."""
    labels = mine(events, owner_words)
    hooks = [e for e in events if e["type"] == "hook"]
    missed = len({it["hook_turn"] for it in labels if it["label"] == "miss"})
    scores = [c["score"] for h in hooks for c in h["cands"] if c.get("score") is not None]
    st.update(labels, hooks=len(hooks), missed_turns=missed, scores=scores)
    return labels
