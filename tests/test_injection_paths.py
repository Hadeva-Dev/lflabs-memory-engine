"""Decision 2026-11-09: a memory goes into a prompt without the model asking only when it is the user's own or one they
confirmed (mem.injectable). Every automatic path builds its memory lines with mem._memory_line or instant._shown, which
both refuse anything else; nothing else writes a memory into a prompt. A security review found the secrets path
re-searching after its filter: the check is made at the last step.

  .venv/bin/python -m unittest -q test_injection_paths
"""
import _env  # noqa: F401  (first: the test data folder and user)
import ast
import unittest
from pathlib import Path
from unittest import mock

from lflabs_memory import instant, mem
from test_secrets import Case as SecretsCase

SRC = Path(mem.__file__).parent
# Memory lines written by hand, for the user in Terminal (never into a prompt): the outline file, `audit`, and
# `whats-up --verbose`.
BY_HAND = {("mem.py", "write_outline"), ("mem.py", "audit_report"), ("whatsup.py", "_stale_memories")}


def functions(path):
    text = path.read_text()
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node.name, ast.get_source_segment(text, node) or ""


class OnePlaceDecides(unittest.TestCase):
    def test_only_the_two_checked_builders_write_memory_lines(self):
        builders = set()
        for path in sorted(SRC.glob("*.py")):
            for name, src in functions(path):
                if 'f"- #{' in src or ("json.dumps(text" in src and 'out.append(f"- ' in src):
                    builders.add((path.name, name))
        self.assertEqual(builders - BY_HAND, {("mem.py", "_memory_line"), ("instant.py", "_shown")})
        for path, name in (("mem.py", "_memory_line"), ("instant.py", "_shown")):
            src = dict(functions(SRC / path))[name]
            self.assertIn("injectable(", src, name)

    def test_the_builders_refuse_an_unconfirmed_save(self):
        row = {"id": 1, "kind": "project", "project": None, "content": "The orchard docs live in docs/orchard",
               "origin": "unknown", "section": "projects", "section2": None, "writer": None, "source": None,
               "chunk": "x", "ord": 0, "sim": 0.5, "score": 1.0, "created": 0, "updated": 0}
        with self.assertRaises(ValueError):
            mem._memory_line(row, "where are the docs", frozenset())
        self.assertTrue(mem._memory_line({**row, "origin": "owner_typed"}, "where are the docs", frozenset()))
        self.assertFalse(mem.injectable({**row, "origin": None}))
        self.assertFalse(mem.injectable({**row, "origin": "owner_typed", "writer": "cloud"}))


class UnconfirmedSecret(SecretsCase):
    """A secret-looking app save the user never confirmed, on the path that searches again without the health-and-money
    word list (a host's channel): it never arrives, though the user's own secret still does."""

    def setUp(self):
        super().setUp()
        self.bike = self.add("Robin's bike lock combination is 4417.", origin="unknown")
        self.assertTrue(mem.is_secret(self.row(self.bike)))

    def test_the_re_search_never_brings_it(self):
        view = mem.HOST_SCOPES[0] if mem.HOST_SCOPES else "telegram"
        with mock.patch.object(mem, "HOST_SCOPES", (view,)):
            got = mem._asked_secrets(self.db, "what's my bike lock combination?", [], set(), (mem.WITHHELD,), (),
                                     view)
            self.assertEqual([h["id"] for h in got], [])
            own = mem._asked_secrets(self.db, "What's my locker code?", [], set(), (mem.WITHHELD,), (), view)
            self.assertEqual([h["id"] for h in own], [self.locker])

    def test_no_prompt_brings_it(self):
        for scope in (None, "voice", "telegram"):
            for i, ask in enumerate(("what's my bike lock combination?", "bike lock code?", "remind me of my bike lock")):
                self.assertNotIn("4417", self.ctx(ask, scope, session=f"b-{scope}-{i}"), (scope, ask))
        self.assertIn("7302", self.ctx("What's my locker code?", "telegram", session="own"))


class SetupImportTrust(SecretsCase):
    """Decision 2026-11-09: an interactive setup that imported Claude Code's notes asks once, in the macOS dialog, whether
    they are the user's own; Allow trusts only the notes this import made, Cancel leaves them held."""

    def setUp(self):
        super().setUp()
        src = "/Users/x/.claude/projects/-p/memory/{}.md"
        self.old = mem.remember(self.db, "Orchard deploys from main", "project", source=src.format("old"))[0]
        with self.db:
            self.db.execute("update memories set created = created - 3600 where id = ?", (self.old,))
        self.since = mem.time.time() - 1
        self.new = [mem.remember(self.db, t, "project", source=src.format(i))[0]
                    for i, t in enumerate(("The orchard docs live in docs/orchard", "Orchard uses pnpm, not npm",
                                           "Orchard release builds run on Thursdays", "Orchard CI runs on GitHub"))]
        self.app = mem.app_remember(self.db, "Orchard's staging port is 7311", "project", via="cursor")[0]

    def trust(self, answer):
        asked = []

        def ask(text, title):
            asked.append(text)
            return answer
        got = mem.trust_ids(self.db, self.new + [self.old, self.app], claude_since=self.since, ask=ask)
        return got, asked

    def test_allow_trusts_only_this_imports_notes(self):
        for mid in self.new + [self.old, self.app]:
            self.assertFalse(mem.injectable(self.row(mid)))
        got, asked = self.trust(True)
        self.assertEqual(sorted(got), sorted(self.new))
        self.assertTrue(asked[0].startswith("Trust the 4 notes you imported from Claude Code as your own?"))
        self.assertIn("docs/orchard", asked[0])
        self.assertIn("and 1 more", asked[0])  # the first three shown
        self.assertNotIn(".claude", asked[0])  # never a file name
        for mid in self.new:
            self.assertTrue(mem.injectable(self.row(mid)))
        for mid in (self.old, self.app):  # before this import, or an app's save: still held
            self.assertFalse(mem.injectable(self.row(mid)))

    def test_cancel_leaves_them_held(self):
        got, asked = self.trust(False)
        self.assertEqual(got, [])
        self.assertEqual(len(asked), 1)
        for mid in self.new:
            self.assertFalse(mem.injectable(self.row(mid)))


if __name__ == "__main__":
    unittest.main()
