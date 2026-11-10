"""The product's name is one token (brand.py): no other engine file names it, so a rename touches one line.

  .venv/bin/python -m unittest -q test_brand
"""
import _env  # noqa: F401  (first: the test data folder and user)
import json
import unittest
from pathlib import Path

from lflabs_memory import brand

REPO = Path(brand.__file__).resolve().parent.parent.parent
# models.json names it only in its download URLs, which models.py builds from brand (test_models checks each one)
ALLOWED = {"src/lflabs_memory/brand.py", "tests/test_brand.py", "src/lflabs_memory/models.json"}


class Brand(unittest.TestCase):
    def test_the_name_lives_in_brand_only(self):
        hits = [str(p.relative_to(REPO)) for top in ("src", "tests", "eval", "docs", "skills")
                for p in (REPO / top).rglob("*") if p.is_file() and "__pycache__" not in p.parts
                and str(p.relative_to(REPO)) not in ALLOWED
                and brand.PRODUCT.lower() in p.read_text(errors="ignore").lower()]
        self.assertEqual(hits, [])

    def test_models_json_names_it_only_in_urls(self):
        man = json.loads((REPO / "src/lflabs_memory/models.json").read_text())
        rest = json.dumps({**man, "models": {n: [{k: v for k, v in f.items() if k != "url"} for f in fs]
                                             for n, fs in man["models"].items()}})
        self.assertNotIn(brand.PRODUCT.lower(), rest.lower())


if __name__ == "__main__":
    unittest.main()
