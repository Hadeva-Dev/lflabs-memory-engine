"""Chinese and Japanese text, judged by Unicode range (no model, no word list of the language): Han, Hiragana and
Katakana. These scripts are written without spaces, so every rule that counts words by spaces or reads words with the
default FTS5 tokenizer sees a whole clause as one word (clean-VM CJK test, 2026-11-09: a Chinese or Japanese question
was never searched by the prompt hook, as "under three words", and keyword search never matched a CJK memory).

  words(text)    words, with a CJK run counted as one word per two characters
  share(text)    how much of a text's letters are CJK, a CJK character weighing as a short Latin word
  terms(text)    the keys keyword search and the gate use: overlapping two-character pieces of each CJK run (bigrams,
                 the usual way to index unspaced text), less pairs of kana that are only grammar, plus the text's
                 Latin words but the commonest
  index_text     terms() as FTS5 text for the cjk keyword tables (chunk_cjk, claim_cjk, convo_cjk)
  fts_query      a CJK query for those tables

Korean is written with spaces, so the default tokenizer already reads it: not included.
"""
import re

_CJK = "々぀-ゟ゠-ヿㇰ-ㇿ㐀-䶿一-鿿豈-﫿ｦ-ﾟ"
RUN_RE = re.compile(f"[{_CJK}\U00020000-\U0002fa1f]+")
_KANA_RE = re.compile("[぀-ゟ]")  # hiragana: particles and endings ("は", "ている"), not content
_LATIN_RE = re.compile(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9'’_.#/+-]*[A-Za-zÀ-ÿ0-9]|[A-Za-zÀ-ÿ]")
# Two-character pieces a question is made of in nearly any memory's words (什么 "what", 我们 "we"): never evidence
STOP = frozenset("什么 怎么 哪里 哪儿 哪些 哪个 多少 我们 你们 他们 这个 那个 这些 那些 一下 可以 应该 是什么 为什么 "
                 "什麼 怎麼 我們 這個 那個 應該".split())
_EN_STOP = frozenset("""the a an and or of to in on at for with by from is are was were be it this that what which who
when where how why do does did can should will would i my me we our you your not no""".split())
CJK_WEIGHT = 3  # a CJK character, in Latin letters, when share() weighs a mixed text (a Han character is about a word)


def has(text):
    return bool(RUN_RE.search(text or ""))


def share(text):
    """CJK letters' share of all letters in the text (0.0 with none), each CJK character weighing CJK_WEIGHT."""
    t = text or ""
    c = sum(len(r) for r in RUN_RE.findall(t))
    if not c:
        return 0.0
    other = sum(1 for ch in RUN_RE.sub("", t) if ch.isalpha())
    return c * CJK_WEIGHT / (c * CJK_WEIGHT + other)


def mostly(text):
    """Written mostly in Han, Hiragana or Katakana (share() over one half)."""
    return share(text) > 0.5


def words(text):
    """len(text.split()), but a CJK run inside a word counts one word per two characters (rounded up)."""
    n = 0
    for w in (text or "").split():
        runs = RUN_RE.findall(w)
        if not runs:
            n += 1
            continue
        rest = RUN_RE.sub(" ", w)  # "APIのタイムアウトは何秒？": "API" is a word too
        n += sum((len(r) + 1) // 2 for r in runs) + len(re.findall(r"[^\W_]+", rest))
    return n


def cjk_terms(text):
    """The bigrams of the text's CJK runs (a one-character run: itself, unless it is kana), less STOP and pairs of
    hiragana."""
    out = set()
    for r in RUN_RE.findall(text or ""):
        if len(r) == 1:
            if not _KANA_RE.match(r):
                out.add(r)
            continue
        for i in range(len(r) - 1):
            b = r[i:i + 2]
            if b in STOP or (_KANA_RE.match(b[0]) and _KANA_RE.match(b[1])):
                continue
            out.add(b)
    return out


def terms(text):
    """cjk_terms plus the text's Latin words (lowercased, two letters or more): "ステージングDB" shares "db"."""
    latin = {w.lower() for w in _LATIN_RE.findall(RUN_RE.sub(" ", text or "")) if len(w) > 1}
    return cjk_terms(text) | (latin - _EN_STOP)


def index_text(text):
    """What a cjk keyword table holds for a text: its CJK bigrams, space-separated; "" without CJK (not indexed)."""
    return " ".join(sorted(cjk_terms(text)))


def fts_query(text):
    """An FTS5 query for the cjk tables: any of the text's CJK bigrams. "" without CJK."""
    return " OR ".join(f'"{t}"' for t in sorted(cjk_terms(text)))


def overlap(query_terms, text):
    """(shared, of): how many of the query's terms the text holds, and how many the query has."""
    return len(query_terms & terms(text)), len(query_terms)


EVIDENCE_MIN = 3        # shared terms that are evidence on their own
EVIDENCE_SHORT = (2, 0.5)  # or this many, when they are at least this share of a short question's terms


def evidence(query_terms, text):
    """Whether a CJK memory shares enough of a CJK question's words to be what it asks about (the gate's test for a
    CJK memory, whose reranker score means nothing: the English cross-encoder scores any CJK pair alike)."""
    s, n = overlap(query_terms, text)
    return s >= EVIDENCE_MIN or (n > 0 and s >= EVIDENCE_SHORT[0] and s / n >= EVIDENCE_SHORT[1])
