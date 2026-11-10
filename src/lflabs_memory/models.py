"""The local models (spec 10.2): models.json, shipped in the wheel, lists every file of embed, rerank and sense with its
URL, size and sha256. Setup and update download from those URLs; the engine refuses to load a file whose sha256
differs (verify), so nothing tampered with or half-downloaded is ever run. What each file hashed to is remembered by
(path, size, mtime) in the data folder, so the prompt hook pays a stat per file, not a 34 MB read.

  python -m lflabs_memory.models MODELS_DIR models-<date> > src/lflabs_memory/models.json

models.json: {"release": "models-<date>", "models": {"embed": [{"file", "url", "size", "sha256"}], ...}}
"""
import hashlib
import json
import os
import sys
from pathlib import Path

from . import brand

# Where a models release's assets are downloaded from (flat: <name>-<file>). The public repo that also serves the
# wheels: the engine's own repo is private, so its releases are not reachable for a stranger.
BASE = f"https://github.com/Hadeva-Dev/{brand.COMMAND}/releases/download"
FILES = {"embed": ("model_quantized.onnx", "tokenizer.json"),  # bge-small-en-v1.5, int8
         "rerank": ("model_quantized.onnx", "tokenizer.json"),  # ms-marco-MiniLM-L-6-v2, int8
         "sense": ("friday_sense.py", "model_quantized.onnx", "runtime_config.json", "thresholds.json",
                   "tokenizer.json")}  # the sensitivity classifier and the runtime that came with it
MANIFEST = Path(__file__).resolve().parent / "models.json"


class Refused(RuntimeError):
    """A model file that is missing or does not match models.json: the model is not loaded."""


def manifest():
    return json.loads(MANIFEST.read_text())


def _sha256(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def verify(name, folder, cache):
    """Raise Refused unless every file models.json lists for `name` is in `folder` with its size and sha256. `cache`
    (a JSON file, 0600) maps a path to [size, mtime_ns, sha256]: a file with the same size and mtime is not read."""
    files = manifest()["models"].get(name)
    if not files:
        raise Refused(f"{name} model refused: models.json does not list it")
    try:
        known = json.loads(Path(cache).read_text())
    except (OSError, ValueError):
        known = {}
    if not isinstance(known, dict):
        known = {}
    changed = False
    for f in files:
        p = Path(folder) / f["file"]
        try:
            st = p.stat()
        except OSError:
            raise Refused(f"{name} model refused: {p} is missing; `{brand.COMMAND} setup --models` downloads it") from None
        if st.st_size != f["size"]:
            raise Refused(f"{name} model refused: {p} is {st.st_size} bytes, models.json says {f['size']};"
                          f" `{brand.COMMAND} setup --models` downloads it again")
        got = known.get(str(p))
        if not (isinstance(got, list) and got[:2] == [st.st_size, st.st_mtime_ns]):
            got = known[str(p)] = [st.st_size, st.st_mtime_ns, _sha256(p)]
            changed = True
        if got[2] != f["sha256"]:
            raise Refused(f"{name} model refused: {p} does not match its sha256 in models.json;"
                          f" `{brand.COMMAND} setup --models` downloads it again")
    if changed:
        try:
            tmp = Path(cache).with_name(Path(cache).name + f".{os.getpid()}.tmp")
            with os.fdopen(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as out:
                json.dump(known, out)
            os.replace(tmp, cache)
        except OSError:  # a read-only data folder: the next load reads the files again
            pass


def read(name, folder, file):
    """The bytes of one file models.json lists for `name`, hashed as they are read (never from the cache): code from
    the models folder runs from exactly what was checked. Raises Refused."""
    want = next((f for f in manifest()["models"].get(name) or () if f["file"] == file), None)
    if want is None:
        raise Refused(f"{name} model refused: models.json does not list {file}")
    p = Path(folder) / file
    try:
        data = p.read_bytes()
    except OSError:
        raise Refused(f"{name} model refused: {p} is missing; `{brand.COMMAND} setup --models` downloads it") from None
    if hashlib.sha256(data).hexdigest() != want["sha256"]:
        raise Refused(f"{name} model refused: {p} does not match its sha256 in models.json;"
                      f" `{brand.COMMAND} setup --models` downloads it again")
    return data


def build(folder, release, files=FILES):
    """models.json for the files in `folder` (<name>/<file>), as assets of the release `release`."""
    out = {"release": release, "models": {}}
    for name, names in files.items():
        if not (Path(folder) / name).is_dir():
            continue
        out["models"][name] = [{"file": f, "url": f"{BASE}/{release}/{name}-{f}",
                                "size": (Path(folder) / name / f).stat().st_size,
                                "sha256": _sha256(Path(folder) / name / f)} for f in names]
    return out


if __name__ == "__main__":
    json.dump(build(sys.argv[1], sys.argv[2]), sys.stdout, indent=2)
    print()
