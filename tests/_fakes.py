"""Stand-ins shared by the tests."""


def fake_rerank(scores):
    """A reranker that scores each doc by the first key of `scores` it contains (default 0)."""
    return lambda query, docs: [next((v for k, v in scores.items() if k in d), 0.0) for d in docs]


def limited_scope():
    """A reader that never sees people or health and money, nor anything not filed yet: what a host's restricted
    channel is (a host may have one, as for a chat app). Registered once for the run; tests that patch RESTRICTED keep it."""
    from lflabs_memory import mem
    mem.RESTRICTED.setdefault("limited", ("people", mem.WITHHELD))
    if "limited" not in mem.HOST_SCOPES:
        mem.HOST_SCOPES = (*mem.HOST_SCOPES, "limited")
    return "limited"
