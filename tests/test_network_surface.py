"""What in the released engine can reach the network (0.1.4): the model backend the user chose (llm.py: the Anthropic
API with their key, or `claude -p` with their plan) and gh for a repo's pull requests (whatsup.py, only when gh is
installed and logged in). Everything else talks to the apps on this Mac over a Unix socket; 0.1.4 removed the
engine's remote access for cloud AIs. A new network path fails this test until it is listed here and in the privacy
notes.

  .venv/bin/python -m unittest -q test_network_surface
"""
import _env  # noqa: F401  (first: the test data folder and user)
import re
import tomllib
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "lflabs_memory"
NET = re.compile(r"\burllib\.(?:request|error)\b|\bimport urllib\b(?!\.parse)|\bhttp\.client\b|\bhttpx\b|\bimport requests\b|\baiohttp\b|AF_INET|create_connection|"
                 r"start_server|\burlopen\b|\[\s*\"(?:gh|curl|wget|ssh|nc|scp|rsync)\",|\[claude, \"-p\"")
KNOWN = {
    "llm.py": {"urllib", "[claude, \"-p\""},   # api.anthropic.com with the user's key; `claude -p` with their plan
    "whatsup.py": {"[\"gh\","},               # gh pr list for the repo a session is in
}


class NetworkSurface(unittest.TestCase):
    def test_only_the_known_paths_reach_the_network(self):
        found = {}
        for f in sorted(SRC.rglob("*.py")):
            hits = {m.group(0) for m in NET.finditer(f.read_text())}
            hits = {"urllib" if "urllib" in h or h == "urlopen" else re.sub(r"\[\s*", "[", h) for h in hits}
            if hits:
                found[f.name] = hits
        self.assertEqual(found, KNOWN)

    def test_the_anthropic_api_is_the_only_url_called(self):
        urls = set(re.findall(r"https://[\w.-]+", (SRC / "llm.py").read_text()))
        self.assertEqual({u for u in urls if "anthropic" in u or "api" in u}, {"https://api.anthropic.com"})


if __name__ == "__main__":
    unittest.main()
