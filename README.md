# lflabs-memory-engine

The memory engine inside [Dawnmere](https://dawnmere.app): one long-term memory on your Mac, shared by your AI apps
(Claude Code, Codex, Cursor and other MCP apps).

Memories live in one SQLite file in your Library folder. Search runs on your Mac: vectors (sqlite-vec), keywords (SQLite
FTS5) and a reranker, with small ONNX models on the CPU. Apps reach it through an MCP server; Claude Code also gets hooks
that recall into each prompt and queue finished sessions for saving.

**You probably want Dawnmere, not this repo.** The Dawnmere installer installs this engine, downloads and checks the
models, starts the background service and connects your apps:

```bash
curl -fsSL https://dawnmere.app/install | sh
```

The installer, the `dawnmere` command, the Claude Code plugin and the connector live in
[Hadeva-Dev/dawnmere](https://github.com/Hadeva-Dev/dawnmere) (MIT). This repo is the engine's source, published so you
can read and check what runs on your Mac: storage, search, the screening of saved text, and what each app may see.

Version 0.1.5, early beta. This repo gets one commit per release.

## License

Source-available under the [Functional Source License, Version 1.1, MIT Future License](LICENSE) (FSL-1.1-MIT).

- You may use, copy, change and share it for any purpose except a competing use: making it available to others in a
  commercial product or service that substitutes for it or offers substantially similar functionality. Internal use,
  non-commercial education and non-commercial research are expressly permitted. `LICENSE` has the exact terms.
- Each version becomes available under the MIT license two years after it was released.

## Requirements

- macOS 14 or later on Apple silicon (onnxruntime ships no wheels for Intel Macs)
- Python 3.12 or 3.13, arm64, and [uv](https://github.com/astral-sh/uv)
- About 60 MB of models, plus 250 MB for the optional sensitivity model

## Run it by itself

```bash
uv tool install --managed-python --python 3.12 .
lflabs-memory --help
```

`--managed-python` makes uv use its own arm64 Python, not an x86_64 one that may be first on your PATH.

Every model file, with its URL, size and sha256, is listed in `src/lflabs_memory/models.json`. The engine checks each
file's size and sha256 before loading it and refuses anything that does not match. `lflabs-memory paths` prints where
the models, the database and the settings go.

```bash
lflabs-memory add "The staging server is staging.example.com" --kind reference
lflabs-memory search "staging server"
lflabs-memory serve          # the MCP server an app runs
lflabs-memory daemon         # the background service: keeps models loaded, saves from finished sessions
```

`lflabs-memory COMMAND --help` shows one command's arguments. How memories are filed into sections, and which sections
each app may see, is in [docs/SECTIONS.md](docs/SECTIONS.md).

Everything the engine writes is in one folder, created readable only by you: `$LFLABS_MEMORY_HOME`, else
`~/Library/Application Support/LF Labs/Memory`.

## Tests

```bash
uv sync
LFLABS_MEMORY_MODELS=~/path/to/models uv run python -m unittest discover -s tests
```

`LFLABS_MEMORY_MODELS` points at a folder holding the models (they are not in git). The suite runs in a temporary data
folder as a plain install whose user is called Robin (`tests/_env.py`); no test touches a real one.

## Bugs and changes

Report bugs in [Dawnmere's issues](https://github.com/Hadeva-Dev/dawnmere/issues/new/choose), with the output of
`dawnmere doctor --report`. For anything bigger than a small fix, open an issue before a pull request. By contributing
you agree that your change is licensed FSL-1.1-MIT.

## Contributors

Hao Lin and Dennis Freyman. This repository starts from a single public commit, so earlier authorship is credited here.

## Security

Please do not open a public issue. Email [hello@dawnmere.app](mailto:hello@dawnmere.app), or use
[Report a vulnerability](https://github.com/Hadeva-Dev/dawnmere/security/advisories/new) on the Dawnmere repo. Say what
you found, how to reproduce it, and which version you ran.
