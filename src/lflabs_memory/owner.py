"""Whose memory this is: how memory text and the memory prompts name the user.

Every prompt, label and message is written for "the user" (they/their/them). they() puts the owner's name there when
setup has one (config.toml's owner_name), else returns the text as written. Only prose goes through it (prompts,
labels, messages), never a word list. Stdlib only.
"""
import re


def plain_name(n):
    """n if it is a plain name (letters, spaces, apostrophes, hyphens, periods; at most 60 characters, no line
    breaks), else None: it goes into every prompt."""
    n = str(n or "").strip()
    return n if len(n) <= 60 and re.fullmatch(r"[^\W\d_](?:[^\W\d_]|[ .'’-])*", n) and "  " not in n else None


def they(text, name=None):
    """text, written for "the user", for this owner: `name` in place of "the user" (and "The user"); unchanged when
    setup has no name. "USER#1" line labels stay as they are."""
    return re.sub(r"\b[Tt]he user\b", lambda m: name, text) if name else text
