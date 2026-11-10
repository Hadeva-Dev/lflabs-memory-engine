"""The local models (spec 10.2): models.json lists every model file with URL, size and sha256, and the engine refuses
to load a file whose sha256 differs. Hashes are remembered by (path, size, mtime), so a prompt pays a stat, not a read.

  LFLABS_MEMORY_MODELS=~/path/to/models .venv/bin/python -m unittest -q test_models
"""
import _env  # noqa: F401  (first: the test data folder and user)
import hashlib
import importlib.util
import json
import os
import py_compile
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from lflabs_memory import config, mem, models

SPEC_FILES = {"embed": ["model_quantized.onnx", "tokenizer.json"],
              "rerank": ["model_quantized.onnx", "tokenizer.json"],
              "sense": ["friday_sense.py", "model_quantized.onnx", "runtime_config.json", "thresholds.json",
                        "tokenizer.json"]}


def fake_folder(files):
    """A models folder holding {name: {file: bytes}}, and the manifest that describes it."""
    root = Path(tempfile.mkdtemp(prefix="models-"))
    man = {"release": "models-2026-10-28", "models": {}}
    for name, got in files.items():
        (root / name).mkdir()
        man["models"][name] = []
        for f, data in got.items():
            (root / name / f).write_bytes(data)
            man["models"][name].append({"file": f, "url": f"{models.BASE}/models-2026-10-28/{name}-{f}",
                                        "size": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    return root, man


class Manifest(unittest.TestCase):
    def test_every_model_file_is_listed_with_url_size_and_sha256(self):
        man = models.manifest()
        self.assertRegex(man["release"], r"^models-\d{4}-\d{2}-\d{2}$")
        self.assertEqual({n: [f["file"] for f in fs] for n, fs in man["models"].items()}, SPEC_FILES)
        for name, files in man["models"].items():
            for f in files:
                self.assertEqual(set(f), {"file", "url", "size", "sha256"})
                self.assertEqual(f["url"], f"{models.BASE}/{man['release']}/{name}-{f['file']}")
                self.assertRegex(f["sha256"], r"^[0-9a-f]{64}$")
                self.assertGreater(f["size"], 0)

    def test_the_download_host_is_one_line(self):
        src = Path(models.__file__).read_text()
        self.assertEqual(len(re.findall(r"https://", src)), 1, "the base URL lives in BASE only")

    def test_build_writes_the_same_shape(self):
        root, man = fake_folder({"embed": {"model_quantized.onnx": b"w" * 10, "tokenizer.json": b"{}"}})
        (root / "embed" / "MODEL_CARD.md").write_text("not a model file")
        self.assertEqual(models.build(root, "models-2026-10-28", {"embed": ("model_quantized.onnx", "tokenizer.json")}),
                         man)

    def test_the_shipped_list_matches_the_shared_models(self):
        """The bytes in LFLABS_MEMORY_MODELS (the maintainer's copy, the source of truth) are what models.json lists."""
        folder = os.environ.get(config.MODELS_ENV)
        if not folder or not Path(folder, "embed").is_dir():
            self.skipTest(f"set {config.MODELS_ENV}")
        cache = Path(tempfile.mkdtemp()) / "model-hashes.json"
        for name in SPEC_FILES:
            models.verify(name, Path(folder) / name, cache)


class Verify(unittest.TestCase):
    def setUp(self):
        self.root, self.man = fake_folder({"embed": {"model_quantized.onnx": b"weights" * 100,
                                                     "tokenizer.json": b'{"t": 1}'}})
        p = mock.patch.object(models, "manifest", lambda: self.man)
        p.start()
        self.addCleanup(p.stop)
        self.cache = self.root / "model-hashes.json"
        self.d = self.root / "embed"

    def refused(self):
        with self.assertRaises(models.Refused) as cm:
            models.verify("embed", self.d, self.cache)
        msg = str(cm.exception)
        self.assertNotIn("\n", msg)
        self.assertIn("embed", msg)
        return msg

    def test_matching_files_pass(self):
        models.verify("embed", self.d, self.cache)

    def test_a_changed_byte_is_refused(self):
        (self.d / "model_quantized.onnx").write_bytes(b"weighTs" * 100)  # same size
        self.assertIn("model_quantized.onnx", self.refused())

    def test_a_short_file_is_refused(self):
        (self.d / "model_quantized.onnx").write_bytes(b"weights" * 50)  # half-downloaded
        self.assertIn("model_quantized.onnx", self.refused())

    def test_a_missing_file_is_refused(self):
        (self.d / "tokenizer.json").unlink()
        self.assertIn("tokenizer.json", self.refused())

    def test_a_model_the_list_does_not_know_is_refused(self):
        self.man = {**self.man, "models": {}}
        self.refused()

    def test_a_known_file_is_not_read_again(self):
        models.verify("embed", self.d, self.cache)
        self.assertEqual(stat_mode(self.cache), 0o600)
        with mock.patch.object(models, "_sha256", side_effect=AssertionError("read again")):
            models.verify("embed", self.d, self.cache)  # another process: only the cache file and two stats

    def test_a_changed_file_is_read_again(self):
        models.verify("embed", self.d, self.cache)
        f = self.d / "model_quantized.onnx"
        st = f.stat()
        f.write_bytes(b"weighTs" * 100)
        os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
        self.refused()

    def test_a_broken_cache_is_ignored(self):
        self.cache.write_text("{not json")
        models.verify("embed", self.d, self.cache)
        self.assertIn(str(self.d / "tokenizer.json"), json.loads(self.cache.read_text()))

    def test_a_lying_cache_cannot_pass_a_file(self):
        """The cache only remembers what a file hashed to: it is compared with the list every time."""
        models.verify("embed", self.d, self.cache)
        self.man["models"]["embed"][1] = {**self.man["models"]["embed"][1], "sha256": "0" * 64}
        self.refused()


def stat_mode(p):
    return os.stat(p).st_mode & 0o777


class Loaders(unittest.TestCase):
    """mem's loaders check before anything is read into onnxruntime or run."""

    def test_load_refuses_a_tampered_model(self):
        root, man = fake_folder({"embed": {"model_quantized.onnx": b"weights", "tokenizer.json": b"{}"}})
        (root / "embed" / "model_quantized.onnx").write_bytes(b"evil!!!")
        with mock.patch.object(models, "manifest", lambda: man), mock.patch.dict(mem._models, clear=True):
            with self.assertRaises(models.Refused):
                mem._load("embed", root / "embed", 512)
            self.assertEqual(mem._models, {})

    def test_load_checks_the_real_models(self):
        folder = os.environ.get(config.MODELS_ENV)
        if not folder or not Path(folder, "embed").is_dir():
            self.skipTest(f"set {config.MODELS_ENV}")
        with mock.patch.dict(mem._models, clear=True), mock.patch.object(models, "verify",
                                                                        wraps=models.verify) as v:
            mem._load("embed", Path(folder) / "embed", 512)
        self.assertEqual(v.call_args.args[:2], ("embed", Path(folder) / "embed"))

    def test_a_tampered_sense_runtime_never_runs(self):
        marker = Path(tempfile.mkdtemp()) / "ran"
        code = f"open({str(marker)!r}, 'w').close()\n".encode()
        root, man = fake_folder({"sense": {"friday_sense.py": b"# the runtime\n", "thresholds.json": b"{}"}})
        (root / "sense" / "friday_sense.py").write_bytes(code)
        with mock.patch.object(models, "manifest", lambda: man), \
                mock.patch.object(mem, "_sense_dir", lambda: root / "sense"), \
                mock.patch.dict(mem._sense, {"tried": False, "model": None, "id": None, "error": None}):
            self.assertIsNone(mem._sense_model())
            self.assertIn("refused", mem._sense["error"])
        self.assertFalse(marker.exists())

    def sense_folder(self, marker):
        """A sense folder whose runtime records "source" when it runs, beside a planted .pyc of the same size and
        mtime (so its header matches the source) that records "pyc"."""
        good = f"open({str(marker)!r}, 'a').write('source')\nclass FridaySense:\n" \
               f"    def __init__(self, d, threads): self.thresholds = {{}}\n".encode()
        evil = good.replace(b"'source'", b"'pyc!!!'")
        self.assertEqual(len(good), len(evil))
        root, man = fake_folder({"sense": {"friday_sense.py": good, "model_quantized.onnx": b"w" * 10,
                                           "thresholds.json": b"{}"}})
        py = root / "sense" / "friday_sense.py"
        st = py.stat()
        py.write_bytes(evil)
        os.utime(py, ns=(st.st_atime_ns, st.st_mtime_ns))
        py_compile.compile(str(py), cfile=importlib.util.cache_from_source(str(py)), doraise=True)
        py.write_bytes(good)
        os.utime(py, ns=(st.st_atime_ns, st.st_mtime_ns))
        return root, man, py

    def load_sense(self, root, man):
        with mock.patch.object(models, "manifest", lambda: man), \
                mock.patch.object(mem, "_sense_dir", lambda: root / "sense"), \
                mock.patch.object(mem, "paths", lambda: mock.Mock(model_hashes=root / "model-hashes.json")), \
                mock.patch.dict(mem._sense, {"tried": False, "model": None, "id": None, "error": None}):
            return mem._sense_model(), mem._sense["error"]

    def test_a_planted_pyc_never_runs(self):
        """The runtime runs from the bytes that were checked: a .pyc in models/sense/__pycache__ whose header matches
        the source (what the import system trusts) is never read."""
        marker = Path(tempfile.mkdtemp()) / "ran"
        root, man, py = self.sense_folder(marker)
        pyc = Path(importlib.util.cache_from_source(str(py)))
        self.assertTrue(pyc.exists())
        before = pyc.read_bytes()
        model, err = self.load_sense(root, man)
        self.assertIsNone(err)
        self.assertIsNotNone(model)
        self.assertEqual(marker.read_text(), "source")
        self.assertEqual(pyc.read_bytes(), before)  # and no bytecode written
        self.assertEqual(sorted(p.name for p in (root / "sense").rglob("*")),
                         sorted(["__pycache__", pyc.name, "friday_sense.py", "model_quantized.onnx",
                                 "thresholds.json"]))

    def test_the_runtime_is_hashed_as_it_is_read(self):
        """A runtime changed behind the cache (same size and mtime) is caught: its bytes are hashed when read."""
        marker = Path(tempfile.mkdtemp()) / "ran"
        root, man, py = self.sense_folder(marker)
        model, err = self.load_sense(root, man)  # the cache now knows the good source
        self.assertIsNone(err)
        marker.unlink()
        st = py.stat()
        py.write_bytes(py.read_bytes().replace(b"'source'", b"'pyc!!!'"))
        os.utime(py, ns=(st.st_atime_ns, st.st_mtime_ns))
        model, err = self.load_sense(root, man)
        self.assertIsNone(model)
        self.assertIn("refused", err)
        self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
