"""Every prompt the engine sends a model names the owner as setup says, or "the user", and never another name
(a head-to-head test, 2026-11-08, found the extraction prompt's lines labelled with a hard-coded name in 0.1.3).
tests/_prompt_capture.py drives each job that asks a model, in a fresh process whose config.toml names the owner
(Sam) or not, with the model call replaced by a recorder.

  .venv/bin/python -m unittest -q test_owner_prompts
"""
import _env  # noqa: F401  (first: the test data folder and user)
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
JOBS = {"extract", "handoff", "claims-extract", "claims-topics", "claims-pick", "claims-rewrite", "classify", "secrets",
        "merge", "review", "audit", "translate"}
LABELS = {"USER", "CLAUDE", "AGENT", "VOICE", "MIC"}  # the engine's speaker labels ("OPEN MIC#3")
GENERIC = {"classify", "translate", "built:CLASSIFY_SYSTEM", "built:TRANSLATE_SYSTEM"}  # written for "the user" only
# The old hard-coded owner and assistant (bracketed so tools/trace_scan.py doesn't flag this line) and the run's own
# test user, which a prompt built for another owner must not carry over.
NAMES = re.compile(r"(?i)\bh[a]o\b|j[a]rvis|\bRobin\b")


def capture(owner):
    home = Path(tempfile.mkdtemp(prefix="ownerprompts"))
    (home / "config.toml").write_text('version = 1\nprofile = "generic"\n' + (f'owner_name = "{owner}"\n' if owner else ""))
    env = {**os.environ, "LFLABS_MEMORY_HOME": str(home)}
    r = subprocess.run([sys.executable, str(HERE / "_prompt_capture.py")], capture_output=True, text=True, env=env,
                       cwd=str(home), timeout=300)
    assert r.returncode == 0, r.stderr[-3000:]
    return [json.loads(line) for line in r.stdout.splitlines() if line.startswith("{")]


class OwnerPrompts(unittest.TestCase):
    def check(self, owner):
        rows = capture(owner)
        jobs = {r["job"] for r in rows}
        self.assertEqual([r for r in rows if r["job"].startswith("error:")], [])
        self.assertLessEqual(JOBS, jobs)  # every job reached its model call
        for r in rows:
            text = r["system"] + "\n" + r["prompt"]
            self.assertIsNone(NAMES.search(text), (r["job"], NAMES.search(text)))
            self.assertLessEqual(set(re.findall(r"\b([A-Z]{2,})#\d", text)), LABELS, r["job"])
            if owner:
                self.assertIn("the user" if r["job"] in GENERIC else owner, r["system"], r["job"])
            else:
                self.assertNotRegex(text, r"\bSam\b(?! said, by)", r["job"])  # MERGE_SYSTEM's example contact is Sam
                self.assertRegex(r["system"], r"\b[Tt]he user\b", r["job"])
        return rows

    def test_a_named_owner(self):
        rows = self.check("Sam")
        extract = next(r for r in rows if r["job"] == "extract")
        self.assertIn("USER#1: Orchard staging listens", extract["prompt"])
        self.assertIn("OPEN MIC#3 (may not be Sam)", extract["prompt"])
        self.assertIn("Only USER lines (USER#1:, USER#2: ...) are Sam's own words", extract["system"])

    def test_no_name(self):
        rows = self.check(None)
        extract = next(r for r in rows if r["job"] == "extract")
        self.assertIn("OPEN MIC#3 (may not be the user)", extract["prompt"])


if __name__ == "__main__":
    unittest.main()
