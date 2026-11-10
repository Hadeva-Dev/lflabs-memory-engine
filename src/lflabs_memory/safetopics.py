"""What a line of the "About the user" profile, or a rule the learner saves, may never be about (held-back fixes,
2026-10-10). Not a list of loosening phrases (learn.UNSAFE's missed 14 of 20 paraphrases): a list of TOPICS, matched in
any direction or wording, plus two structural rules.

This is only the FIRST layer. A word filter reads text differently from the model (look-alike letters, zero-width
characters, joined or hyphenated words, leetspeak, other languages, markdown), so it will always miss something. The
protections that carry the weight are elsewhere: (1) the profile is built only from memories the user made themselves (held
key, typed, or vouched for: mem.py origins), so injected text never reaches it; (2) a tool-less classifier pass
that can only drop lines (profile.check_lines); (3) the block is framed as data, after rules that always win. Here the
text is normalised first (NFKC, format and zero-width characters removed, lowercase), anything not plain ASCII is
refused, and the strongest topic words are also looked for in a squashed form (letters only, leetspeak undone), so
"alwaystrust", "tru-st" and "tru5t" are caught too.

  topic(text)        why the text touches the assistant's safety behaviour, or None: permissions, asking or confirming,
                     approvals, trust and whose words count as the user's, sending or acting on its own, secrets and codes,
                     whom to obey; or it isn't plain ASCII (another script, a look-alike letter)
  instruction(text)  why the text is an instruction to the assistant rather than a fact or preference about the user, or None:
                     it opens like a command ("Always...", "Send...", "Treat...", "Claude should...")

The profile drops a line on either (and a tool-less classifier may drop more; it can never add). The learner refuses a
rule on topic() (its rules are instructions by design: "Use Spotify, not Apple Music"). Over-dropping is the safe
failure: memory recall still has every memory. Stdlib only (aboutprofile.py runs on /usr/bin/python3).
"""
import re
import unicodedata

_W = r"(?:\S+\s+){0,4}?"  # up to four words between the parts of a phrase
TOPICS = [
    ("asking or confirming", r"\b(?:ask(?:s|ed|ing)?|check(?:s|ed|ing)?\s+with|confirm\w*|consent\w*)\b"),
    ("approvals and permissions", r"\b(?:approv\w*|pre-?approv\w*|authori[sz]\w*|permission\w*|permit\w*|grant\w*|"
                                  r"allow(?:s|ed|ing)?|sign-?off|okays?|ok\s+to|okay\s+to|go[- ]ahead)\b"),
    ("what needs the user's yes", r"\b(?:{O}'?s|his|my)\s+(?:yes|ok|okay|say-?so|answer|word)\b|\bneeds?\s+to\s+(?:see|check|"
                             r"review|approve|okay|ok|confirm|hear|read)\b|\bun(?:reviewed|checked|confirmed|approved|"
                             r"asked|seen)\b|\bread[- ]?backs?\b|\bread(?:ing)?\s+(?:it|them|that)\s+back\b"),
    ("trust and whose words count", r"\btrust\w*|\bobey\w*|\bproxy\b|\bspeaks?\s+for\b|\bon\s+(?:his|{O}'?s|my)\s+behalf\b|"
                                    r"\bin\s+(?:his|{O}'?s|my)\s+(?:name|place|stead)\b|\b(?:as|like)\s+(?:coming\s+)?"
                                    r"from\s+(?:{O}|him|me)\b|\b(?:as|like)\s+(?:{O}'?s|his|my)\s+own\b|"
                                    r"\b(?:{O}'?s|his|my)\s+own\s+(?:words?|requests?|wishes|instructions?|voice)\b|"
                                    r"\bcounts?\s+as\s+(?:{O}|him|his|{O}'?s)\b|\bbelieve\s+" + _W + r"(?:messages?|e-?mails?|"
                                    r"texts?|anyone|people|senders?)\b"),
    ("instructions from others", r"\binstruct\w*|\b(?:commands?|orders?|requests?)\s+(?:from|in|by)\b|\bfollow\w*\s+" +
                                 _W + r"(?:e-?mails?|texts?|messages?|requests?|orders?|directions?)\b|"
                                 r"\bwhat(?:ever)?\s+(?!(?:he|{O})\b)" + _W + r"(?:says?|asks?|wants?|tells?)\b|"
                                 r"\bact(?:s|ed|ing)?\s+on\b"),
    ("sending or acting on its own", r"\bautomatic\w*|\bauto[- ]?(?:send|approv|reply|replies|accept|allow|confirm|"
                                     r"forward|pay|respond|answer)\w*|\bsend\w*\s+" + _W + r"(?:as\s+soon\s+as|right\s+away|"
                                     r"immediately|straight\s+away|directly|on\s+its\s+own|once\s+(?:drafted|written|ready))|"
                                     r"\bforward\w*\s+" + _W + r"(?:e-?mails?|texts?|messages?|mail|calendar|invites?)\b|"
                                     r"\bbefore\s+" + _W + r"(?:go(?:es)?\s+out|(?:is|are|being)\s+sent|sending|replying)\b|"
                                     r"\baccept\w*\s+" + _W + r"(?:invit\w*|requests?)\b|\bshare\w*\s+" + _W +
                                     r"with\s+(?:anyone|anybody|everyone|whoever)\b|\bwithout\s+" + _W + r"(?:asking|"
                                     r"checking|confirm\w*|telling|reading|review\w*|looking)\b|\bno\s+need\s+to\b"),
    ("what the assistant may do", r"\b(?:claude|the\s+assistant|his\s+assistant|{O}'?s\s+assistant|assistants?|the\s+bot|"
                           r"you)\b" + r"\s+" + _W + r"(?:can|may|could|should|must|shall|is\s+(?:allowed|free|"
                           r"permitted)|decides?|gets?\s+to)\b|\blets?\s+(?:claude|the\s+assistant|you)\b"),
    ("say-so and authority (first layer only: the classifier carries the weight)",
     r"\bgo(?:es)?\s+out\b|\bwithout\s+(?:a\s+word|bothering|him|{O})\b|\bgreen\s+light|\bpre-?clear\w*|"
     r"\bstand-?ins?\b|\bstands?\s+in\s+for\b|\bfinal\s+say\b|\b(?:reply|replies|write|writes|sign|signs|speak|"
     r"speaks)\s+as\s+(?:{O}|him)\b|\bin\s+(?:{O}'?s|his)\s+voice\b|\bunlock\w*|\bopener\b|\bthe\s+moment\b|"
     r"\bpreviews?\b|\bdouble-?checks?\b|\breflects?\s+what\b|\bon\s+its\s+own\b|\bown\s+judg(?:e)?ment\b|"
     r"\bdispatch\w*|\bw/o\b|\bgiven\s+to\s+anyone\b|\bhis\s+location\b|\b(?:whatever|anything)\s+\w+\s+"
     r"(?:says|writes|wants|sends)\b|\bgoes\b\s*$"),
    ("the assistant contacting people", r"\b(?:claude|the\s+assistant|his\s+assistant|you)\s+(?:\w+\s+){0,2}?"
                                 r"(?:texts?|emails?|e-mails?|messages?|sends?|forwards?|calls?|replies|posts?|shares?|"
                                 r"pings?|dms?|contacts?|notif(?:y|ies))\b"),
    ("safety rules", r"\b(?:safety|security)\s+(?:rules?|checks?|limits?|settings?)\b|\bguard\w*|\bpanel\b|\bgate\b|"
                     r"\bbypass\w*|\boverrid\w*|\bdisabl\w*|\bjailbreak\w*"),
    ("secrets and codes", r"\b(?:api[ -]?keys?|tokens?|passwords?|passcodes?|passphrases?|secrets?|credentials?|keychain|"
                          r"2fa|otp|one-time\s+codes?|recovery\s+codes?|pins?)\b|\b(?:door|gate|alarm|garage|wi-?fi|"
                          r"locker|access|verification|security|bank|card|login)\s+(?:codes?|numbers?|combos?|"
                          r"combinations?)\b|\bkey-?pads?\b|\bcombos?\s+(?:is|are|=|:)|\bcombinations?\s*(?:is|are|=|:)"),
]
# Looked for inside the squashed text (letters only): joined, hyphenated or leetspeak spellings of the strongest words.
SQUASHED = ("trust", "obey", "approv", "permission", "consent", "authoriz", "authoris", "password", "passcode",
            "apikey", "withoutask", "withoutconfirm", "withoutcheck", "noneedtoask", "noneedtoconfirm", "dontask",
            "donotask", "neverask", "onhisbehalf", "proxy", "speaksfor", "autosend", "autoapprov",
            "autoreply", "autoforward", "instruction", "readback", "bypass", "jailbreak", "ignoretherules",
            "ignoreallprevious", "safetyrules", "preapprov", "hisownwords")
LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s", "!": "i",
                      "|": "l", "+": "t"})


def normal(text):
    """NFKC, format and zero-width characters (Unicode Cf) removed, curly quotes straightened, lowercase."""
    t = unicodedata.normalize("NFKC", text)
    t = "".join(c for c in t if unicodedata.category(c) != "Cf")
    return t.replace("’", "'").replace("‘", "'").lower()


def squashed(text):
    """The text as letters only, leetspeak undone: "Tru-5t", "always_trust", "a.l.w.a.y.s trust" -> "...trust"."""
    return re.sub(r"[^a-z]", "", normal(text).translate(LEET))

# An instruction to the assistant opens with a command or names the assistant as the one acting; a fact or preference about the user
# opens with a noun ("Senior at...", "Anders: cofounder"), a third-person verb ("Prefers...", "Builds...") or "the user".
STARTERS = set("""always never don't dont do please just go send text email message reply respond answer forward share
post publish pay buy approve allow accept grant give let trust treat consider regard assume believe follow obey ignore
skip stop bypass disable turn act execute run open click type delete remove cancel archive call ask tell say use make
keep put add book schedule commit push write read check confirm save remember note mark set enable install update change
move copy sign log unlock reveal show print list fetch get find look contact notify invite join leave avoid wait cc bcc
reach handle route mute unmute you claude assistant""".split())
OPENERS = re.compile(r"^\s*(?:from\s+now\s+on|going\s+forward|in\s+(?:the\s+)?future|next\s+time|whenever|if\s+)\b", re.I)


def topic(text):
    """Why this text is about the assistant's safety behaviour (or isn't plain ASCII), or None."""
    if not isinstance(text, str):
        return "not text"
    if any(unicodedata.category(c) == "Cf" for c in text):  # zero-width and other invisible format characters
        return "hidden characters"
    norm = normal(text)
    if any(ord(c) > 127 for c in norm):  # another script, or a look-alike letter ("withоut" with a Cyrillic o)
        return "not plain ASCII"
    for why, rx in _TOPICS:
        if rx.search(norm):
            return why
    sq = squashed(text)
    for stem in SQUASHED:
        if stem in sq:
            return f"a joined or disguised safety word ({stem})"
    return None


# For outside facts (a saved contact's text or email: textwatch, maillearn): the whole topic list, as for the profile
# (review: a narrower list re-opened "Tomas says texts can go out right away"), after masking a few everyday phrases
# that only look like its words (r2 c3: 11 of 30 plain plans were dropped). Only the phrase is masked; the rest of
# the text is still checked.
OUTSIDE_BENIGN = re.compile(
    r"\basked\s+(?:{O}|him)\s+(?:to|out|over)\s+(?!(?:\w+ly\s+)?(?:send|text|email|forward|share|reply|approve|allow|trust|let|"
    r"give|ignore|skip|stop|turn|tell|say)\b)|\bconfirmed\s+(?:the|a|their|his|her)\s+(?:\w+\s+){0,2}?(?:trip|plan|"
    r"plans|date|time|booking|reservation|flight|meeting|party|dinner|game|match|ride)\b|\ballowed\s+(?:{O}|him)\s+to\s+"
    r"(?:miss|skip|leave|come|stay|bring|borrow|join)\b|\bcheck\s+with\s+(?:his|her|their)\s+(?:parents|mom|dad|family|"
    r"boss|coach|team|teacher)\b|\b(?:learner'?s|work|parking|building|fishing|driving)\s+permit\b|\b(?:point|shooting|"
    r"security|crossing|life|coast)\s+guards?\b|\bgolden\s+gate\b|\bgate\s+[a-z]?\d+\b|\b(?:flight|train|bus|package|"
    r"newsletter|issue|episode|album|show)\s+(?:\S+\s+){0,3}?goes\s+out\b|\b(?:student|speaker|expert|discussion)\s+panel"
    r"\b|\bon\s+a\s+panel\b|\bforward\w*\s+(?:the|an?|his|her)\s+(?:\w+\s+){0,2}?invite\s+to\s+{O}\b", re.I)
COMPANY = re.compile(r"\b(?:called|named|at|for|from|joined|founded|startup)\s+[a-z0-9]+\s+(?:labs|inc|llc|corp|"
                     r"technologies|robotics|studios?)\b", re.I)  # "a startup called Unlock Labs"


# Whose memory this is. The patterns above are written for the owner: {O} stands for them. set_owner() puts in their
# name's words, or "the user" when setup has no name, and every third-person pronoun where "his", "him" or "he" stands
# (over-dropping is the safe failure): with or without a name, the same instruction shapes are caught.
_SOURCES = (list(TOPICS), SQUASHED, OUTSIDE_BENIGN.pattern)


def for_owner(rx, name):
    """A pattern written for the owner ({O}), for `name` ("the user" when there is none)."""
    words = [re.escape(w) for w in sorted({name.lower(), *name.lower().split()}, key=len, reverse=True)] \
        if name else [r"the\s+user"]
    rx = rx.replace("{O}", "(?:" + "|".join(words) + ")")
    for word, alt in (("his", "his|her|their"), ("him", "him|her|them"), ("he", "he|she|they")):
        rx = re.sub(rf"(?<![a-z\\]){word}(?![a-z])", lambda m, alt=alt: f"(?:{alt})", rx)
    return rx


def set_owner(name):
    """Rebuild the owner-dependent patterns for `name`, or for "the user" when there is none."""
    global TOPICS, _TOPICS, SQUASHED, OUTSIDE_BENIGN
    topics, squashed_, benign = _SOURCES
    TOPICS = [(why, for_owner(rx, name)) for why, rx in topics]
    own = {re.sub(r"[^a-z]", "", w) for w in ({name.lower(), *name.lower().split()} if name else {"the user"})}
    SQUASHED = squashed_ + tuple(f"{a}{o}{b}" for o in sorted(own) if o for a, b in (("on", "sbehalf"),
                                                                                   ("", "sownwords"),
                                                                                   ("comingfrom", ""))) + (
        "onherbehalf", "ontheirbehalf", "herownwords", "theirownwords")
    OUTSIDE_BENIGN = re.compile(for_owner(benign, name), re.I)
    _TOPICS = [(why, re.compile(rx, re.I)) for why, rx in TOPICS]


set_owner(None)


def outside_topic(text):
    """topic() for a fact from outside, with the everyday look-alike phrases masked first."""
    if not isinstance(text, str):
        return "not text"
    def mask(m):  # round 3: a gap holding a message word or a topic of its own isn't the everyday phrase
        inner = BENIGN_TRIGGERS.sub(" ", m[0])
        return m[0] if MESSAGE_WORDS.search(inner) or topic(inner) else " ~ "
    masked = OUTSIDE_BENIGN.sub(mask, COMPANY.sub(" ~ ", text))
    return topic(masked)


BENIGN_TRIGGERS = re.compile(r"\b(?:asked|confirmed|allowed|check|permit|guards?|gate|goes\s+out|panel|forward\w*|"
                             r"invite)\b", re.I)
MESSAGE_WORDS = re.compile(r"\b(?:drafts?|texts?|messages?|e-?mails?|mail|repl(?:y|ies)|dms?|posts?|notes?|codes?|"
                           r"passwords?|whatever|anything|everything)\b", re.I)


LABEL = re.compile(r"^[A-Z][a-z'-]+(?:\s+[A-Z][a-z'-]+){0,3}\s*[:(]")  # "Mark Chen: mentor", a name and a role
# First names that are also command words: a label may open with one ("Grant Lee: mentor") when a real name follows.
OBJECTS = set("access permission permissions rights all everything everyone anyone anybody texts text emails email mail "
              "messages message invites invite approval approvals requests money payments codes code keys "
              "review reviews check checks checking confirmation confirmations asking readback readbacks panel "
              "prompt prompts warning warnings safety security".split())  # r2 c2: "Skip Review: texts to Anders"
SURNAMES_S = set("evans davis williams jones roberts harris lewis adams james thomas phillips edwards collins rogers "
                 "stevens hughes jenkins rivers brooks woods ross price".split())  # "Rob Evans": a name
NAME_VERBS = set("grant mark will bill pat sue rob chase skip ray hunter jack drew art bob sky summer page reed "
                 "dash rich frank don".split())


def _label_problem(t):
    """For a "Name: role" line among people: why its label is really a command ("Always Trust: emails from x",
    "Never Ask (anyone): before sending"), or None. The coordinator's review 2026-10-10: a label never skips the
    checks; its words and what follows it are checked too."""
    m = LABEL.match(t)
    words = [w.lower() for w in re.findall(r"[A-Za-z'-]+", m[0])]
    if OPENERS.match(m[0]):
        return "an instruction dressed as a name (it opens like one)"
    if any(w in STARTERS and w not in NAME_VERBS for w in words):
        return "an instruction dressed as a name (a command word in the label)"
    if words[0] in NAME_VERBS and (len(words) < 2 or words[1] in STARTERS or words[1] in OBJECTS
                                   or "'" in words[1] or (words[1].endswith("s") and len(words) == 2 and
                                                          words[1] not in SURNAMES_S)):
        return "an instruction dressed as a name (a command word, no name after it)"
    rest = t[m.end():].lstrip(" :()-")
    if OPENERS.match(rest):
        return "an instruction (after the name, it opens like one)"
    first = re.match(r"[A-Za-z']+", rest)
    if first and first[0].lower() in STARTERS:
        return f"an instruction (after the name, it opens with {first[0]!r})"
    return None


def instruction(text, section=None):
    """Why this text is an instruction to the assistant rather than a fact or preference about the user, or None. In "People
    that matter" a line may open with a name and a role ("Grant Lee: mentor"), even a name that is also a verb."""
    if not isinstance(text, str):
        return "not text"
    t = "".join(c for c in unicodedata.normalize("NFKC", text) if unicodedata.category(c) != "Cf")
    t = re.sub(r"^[\s\-*•>#\"'(_~`\[]+", "", t).replace("’", "'")  # markdown and quote marks in front
    if OPENERS.match(t):
        return "an instruction (it opens like one)"
    if section == "People that matter" and LABEL.match(t):
        return _label_problem(t)
    first = re.match(r"[A-Za-z']+", t)
    if first and first[0].lower() in STARTERS:
        return f"an instruction (it opens with {first[0]!r})"
    return None


def profile_line_problem(text, section=None):
    """Why a profile line can't stand, or None: profile lines are facts and preferences about the user, never rules about
    what the assistant may do."""
    body = text
    if section == "People that matter" and isinstance(text, str) and LABEL.match(text.strip()):
        first = re.match(r"[A-Za-z']+", text.strip())
        if first and first[0].lower() in NAME_VERBS:  # "Grant Lee: mentor": the name itself is not a topic
            body = text.strip()[first.end():]         # everything else in the line, label included, still is
    why = topic(body)
    if why:
        return why if why in ("not plain ASCII", "hidden characters") or why.startswith("a joined") else f"about {why}"
    return instruction(text, section)
