"""Where the engine keeps its data, and the settings in DATA_DIR/config.toml.

One data folder holds everything the engine writes (the store, client policies, state files, backups); nothing
lives beside the code. The folder is LFLABS_MEMORY_HOME if set, else ~/Library/Application Support/LF Labs/Memory
(created 0700), whatever the profile.

Code-shipped files (gate.json, adapt-prior.json) are not data: mem.py finds them by its own folder.

  CFG = config.current()   the config of this process: set by cli.main(profile=...), else loaded on first use
"""
import os
import sys
import tempfile
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path

from . import profiles
from .owner import plain_name
from .profiles import Profile

HOME_ENV = "LFLABS_MEMORY_HOME"
PROFILE_ENV = "LFLABS_MEMORY_PROFILE"
MODELS_ENV = "LFLABS_MEMORY_MODELS"  # a models folder shared by checkouts and CI runs (read-only)
SOCKET_ENV = "LFLABS_MEMORY_SOCKET"  # the connector socket (connector.py; the npm connector reads it too)
HOST_ENV = "LFLABS_MEMORY_HOST_DIR"  # a host assistant's folder, given to its profile (profiles.load)


# Files in the data folder no backup may copy: per-install secrets. The engine's own snapshots copy memory.db only;
# a host's backup script reads this list from `paths --json` ("never_back_up").
NOT_BACKED_UP = ("source-key.secret", "holdout-salt")


@dataclass(frozen=True)
class Paths:
    data: Path            # DATA_DIR
    models: Path          # data / "models" unless LFLABS_MEMORY_MODELS, config.toml or the profile says otherwise
    logs: Path            # ~/Library/Logs/LF Labs
    sock: Path            # the daemon's socket: data / "engine.sock"

    @property
    def db(self): return self.data / "memory.db"
    @property
    def clients(self): return self.data / "clients.json"
    @property
    def clients_lock(self): return self.data / "clients.lock"
    @property
    def sense_flags(self): return self.data / "sense-flags.json"
    @property
    def model_hashes(self): return self.data / "model-hashes.json"
    @property
    def access_history(self): return self.data / "access-history.jsonl"
    @property
    def access_requests(self): return self.data / "access-requests.json"
    @property
    def notices(self): return self.data / "notices.jsonl"
    @property
    def menu_state(self): return self.data / "menu-state.json"
    @property
    def adapt_state(self): return self.data / "adapt-state.json"
    @property
    def holdout_salt(self): return self.data / "holdout-salt"
    @property
    def source_secret(self): return self.data / "source-key.secret"  # never backed up (NOT_BACKED_UP)
    @property
    def people_tags(self): return self.data / "people-tags.log"
    @property
    def whatsup_cache(self): return self.data / "whatsup-cache.json"
    @property
    def outline(self): return self.data / "OUTLINE.md"
    @property
    def audit(self): return self.data / "AUDIT.md"
    @property
    def backup(self): return self.data / "backup"

    def remote(self, name):
        """A remote-access file (remote-on, remote-front-key, remote-audit.jsonl, ...)."""
        return self.data / name


@dataclass(frozen=True)
class LLMConfig:
    backend: str = "claude-cli"   # claude-cli | anthropic-api | none (llm.py)
    main_model: str = "haiku"
    cheap_model: str = "haiku"
    daily_calls: int = 200        # model calls a day, all jobs together; 0: no limit
    claude_path: str = ""         # the claude command, resolved by setup in the user's login shell; "": look for it


@dataclass(frozen=True)
class Privacy:
    codex_sessions: bool = False
    chat_index: bool = False
    holdout: bool = False
    new_clients: str = "work"


@dataclass(frozen=True)
class Connector:
    enabled: bool = True  # the daemon answers lflabs-memory/1 on the connector socket (connector.py)


@dataclass(frozen=True)
class Config:
    paths: Paths
    profile: Profile = field(default_factory=lambda: profiles.load("generic"))
    llm: LLMConfig = field(default_factory=LLMConfig)
    privacy: Privacy = field(default_factory=Privacy)
    connector: Connector = field(default_factory=Connector)
    owner_name: str | None = None
    launcher: tuple[str, str] = ("", "")    # (python, script) that runs this engine: what client configs and the
                                            # daemon's own child processes call

    @property
    def command(self):
        """What the user types to run this engine, for messages: the launcher's script (lflabs-memory, or mem.py)."""
        return Path(self.launcher[1]).name or "lflabs-memory"

    @property
    def front(self):
        """What the user types for the commands the product CLI also runs (held, trust, forget, restore, search, add),
        for messages: the CLI's command when this is the console script, else the launcher's script (mem.py)."""
        from .brand import COMMAND
        return COMMAND if self.command == "lflabs-memory" else self.command

    def with_data(self, data):
        """This config with another data folder (a bench's or a test's): every data file follows it, models too (the
        folder's own models/, or none), as when ROOT moved."""
        data = Path(data)
        return replace(self, paths=replace(self.paths, data=data, models=data / "models", sock=data / self.paths.sock.name))


def _launcher():
    script = Path(sys.executable).parent / "lflabs-memory"  # the console script installed beside this python
    return sys.executable, str(script)


def _section(raw, key, cls, defaults=None):
    got = raw.get(key, {})
    if not isinstance(got, dict):
        raise ValueError(f"config.toml: [{key}] must be a table")
    known = {k: v for k, v in {**(defaults or {}), **got}.items() if k in cls.__dataclass_fields__}
    return cls(**known)


def _toml(data):
    f = data / "config.toml"
    return tomllib.loads(f.read_text()) if f.exists() else {}


def load(profile=None, create=True):
    """The config for `profile` (default: LFLABS_MEMORY_PROFILE, else config.toml's, else generic): generic, or one a
    host registered (profiles.load, given LFLABS_MEMORY_HOST_DIR). The data folder is LFLABS_MEMORY_HOME if set, else
    the default; it is created (0700) if missing, unless create is False."""
    home = Path.home()
    data = Path(os.environ[HOME_ENV]).expanduser() if os.environ.get(HOME_ENV) else \
        home / "Library/Application Support/LF Labs/Memory"
    raw = _toml(data)
    profile = profile or os.environ.get(PROFILE_ENV) or raw.get("profile") or "generic"
    if not isinstance(profile, str) or not profile.isidentifier():
        raise ValueError(f"unknown profile {profile!r}")
    if create and not data.exists():
        data.mkdir(parents=True, mode=0o700)
        os.chmod(data, 0o700)  # whatever the umask
    models = os.environ.get(MODELS_ENV) or raw.get("models")
    models = Path(models).expanduser() if models else data / "models"
    paths = Paths(data=data, models=models, logs=home / "Library/Logs/LF Labs", sock=data / "engine.sock")
    host = os.environ.get(HOST_ENV)
    prof = profiles.load(profile, Path(host).expanduser() if host else None)
    return Config(paths=paths, profile=prof, llm=_section(raw, "llm", LLMConfig, prof.llm),
                  privacy=_section(raw, "privacy", Privacy), connector=_section(raw, "connector", Connector),
                  owner_name=plain_name(raw.get("owner_name")), launcher=_launcher())


CFG = None


def _under_test():
    """A test run (python -m unittest or pytest, a test_*.py script, or a process one started)."""
    if os.environ.get("LFLABS_MEMORY_UNDER_TEST") == "1":
        return True
    main = sys.modules.get("__main__")
    return ((getattr(main, "__package__", None) or "").split(".")[0] in ("unittest", "pytest")
            or os.path.basename(getattr(main, "__file__", None) or "").startswith("test_"))


def peek():
    """This process's config if loaded, else the config as it would load, without creating the data folder: for code
    that only needs a name from it at import (connect_apps), which the installer imports even when uninstalling."""
    if CFG is not None or _under_test():
        return current()
    return load(create=False)


def current():
    """This process's config, loaded once. A test run with no LFLABS_MEMORY_HOME gets a fresh temp folder (and its
    child processes the same one), so no test can touch a real data folder."""
    global CFG
    if CFG is None:
        if _under_test():
            os.environ["LFLABS_MEMORY_UNDER_TEST"] = "1"  # programs a test starts (a hook, the CLI) inherit it
            if not os.environ.get(HOME_ENV):
                os.environ[HOME_ENV] = tempfile.mkdtemp(prefix="lflabs-memory-test-")
            if not os.environ.get(SOCKET_ENV):  # never the real connector socket either
                os.environ[SOCKET_ENV] = os.path.join(os.environ[HOME_ENV], "memory-connector.sock")
        CFG = load()
    return CFG
