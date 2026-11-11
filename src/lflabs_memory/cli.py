"""The lflabs-memory command: the daemon, hooks, MCP server and one-off commands, all as mem.py's subcommands
(`lflabs-memory --help` lists them). Help, an unknown command and a missing argument are answered here, before
mem.py loads, so none of them opens or creates a store."""
import codecs
import locale
import os
import re
import sys
import tomllib

from . import config
from .brand import COMMAND

USAGE = """usage: lflabs-memory COMMAND [ARGS]

One local long-term memory shared by your AI apps. Data: $LFLABS_MEMORY_HOME, else
~/Library/Application Support/LF Labs/Memory. `lflabs-memory COMMAND --help` for one command.

Memories
  add TEXT [--kind KIND] [--project NAME]   save one memory
  search QUERY [-k N] [--section SECTION]   search by meaning and keywords
  forget ID | restore ID                    retire a memory, or bring it back
  stats | sections | outline [SECTION]      what is saved

Apps
  serve [--client NAME]                     the MCP server (stdio) an AI app starts
  connect [--yes] [--only APP]              wire the AI apps on this Mac (a dry run without --yes)
  clients                                   what each app may see
  daemon                                    keep the models warm and run the background jobs
  hook-prompt | hook-session-start | hook-activity | hook-session-end
                                            Claude Code hooks (JSON on stdin)

This install
  version | paths [--json] | migrate        the version, where everything is, the store brought up to date

Upkeep: import, reindex, classify, consolidate, audit, tidy, purge, outline-file, import-backup, extract,
index-chats, search-chats, whats-up, and the *-report commands."""

# Every command mem.main() runs, with its arguments (`lflabs-memory COMMAND --help`). A command missing here is
# refused as unknown: tests/test_cli.py checks this list against mem.main().
COMMANDS = {
    "add": ("add TEXT [--kind user|feedback|project|reference|note] [--project NAME]: save one memory (TEXT first, "
            "quoted or not)"),
    "search": ("search QUERY [-k N] [--section SECTION] [--outside]: search memories (QUERY first, quoted or not; "
               "8 results)"),
    "forget": "forget ID: delete a memory for good (run by an AI app: hidden, deleted after a day)",
    "held": "held [--count]: app saves kept out of prompts until you confirm them, with the line to type",
    "trust": "trust [--import-since TS] -- ID...: make memories your own words, after you click Allow in a dialog",
    "restore": "restore ID: bring back a memory an app forgot in the last day, or one the weekly tidy retired",
    "purge": "purge ID: delete one memory for good",
    "stats": "stats: how many memories and chunks, and the store's size",
    "sections": "sections: how many memories each section holds",
    "outline": "outline [SECTION]: memories by section",
    "outline-file": "outline-file: write OUTLINE.md in the data folder",
    "serve": "serve [--client NAME]: the MCP server on stdio (NAME: the app it serves)",
    "daemon": "daemon: keep the models warm for the hooks and run the background jobs",
    "connect": "connect [--yes] [--only APP]: wire the AI apps on this Mac to memory (a dry run without --yes)",
    "clients": ("clients | clients set NAME [--allow S,S] [--deny S,S] | clients add NAME [--key] | "
                "clients level NAME everything|work|nothing-personal|nothing | clients history [N] | "
                "clients requests | clients approve [N] | clients undo: what each AI app may see"),
    "version": "version: the engine's version",
    "--version": "--version: the engine's version",
    "paths": "paths [--json]: where this install keeps everything",
    "migrate": "migrate: bring the store's schema up to this version",
    "hook-prompt": "hook-prompt: Claude Code UserPromptSubmit hook (JSON on stdin)",
    "hook-session-start": "hook-session-start: Claude Code SessionStart hook (JSON on stdin)",
    "hook-cursor-start": "hook-cursor-start: Cursor session start hook (JSON on stdin)",
    "hook-activity": "hook-activity: Claude Code Stop hook (JSON on stdin)",
    "hook-session-end": "hook-session-end: Claude Code SessionEnd hook (JSON on stdin)",
    "import": "import: sync Claude Code's file memories into the store",
    "import-backup": "import-backup [FILE]: restore active memories missing here from backup/memories.json",
    "reindex": "reindex: rebuild every memory's search index",
    "classify": "classify [--model MODEL]: file memories that have no section yet",
    "consolidate": "consolidate: merge near-duplicate memories and retire stale ones",
    "audit": "audit [--dry-run] [--limit N]: tidy project names, review dated notes, write AUDIT.md",
    "tidy": "tidy [--limit N]: the weekly job now: import, consolidate, audit",
    "extract": "extract SESSION_ID: save memories from a queued transcript now",
    "index-chats": "index-chats [--limit N] [--since DAYS] [--budget CHUNKS]: index past conversations",
    "search-chats": "search-chats QUERY [-k N] [--since DAYS] [--project NAME]: search past conversations",
    "whats-up": "whats-up [--project DIR] [--verbose]: where you left off in a folder",
    "redact-retired": "redact-retired [--apply]: mask phones, emails and long numbers in retired memories",
    "mark-secrets": "mark-secrets [--apply]: mark memories saved before secrets were marked",
    "scrub-snapshots": "scrub-snapshots: take meeting facts out of the snapshots in backup/",
    "forget-source": "forget-source meeting:ID#: delete every memory from that meeting",
    "tag-people": "tag-people: give memories about people the second section people",
    "untag-people": "untag-people: undo tag-people",
    "vouch": "vouch ID...: ask the user whether these memories are their own words (needs a host)",
    "speech-terms": "speech-terms [--store DIR]: names in memories with counts, as JSON",
    "followup-report": "followup-report [--store DIR]: follow-up prompts memory could have helped with (counts only)",
    **{r: f"{r}: counts only, never a memory's text" for r in (
        "adapt-report", "sense-report", "claims-report", "facets-report", "budget-report", "excerpt-report",
        "holdout-report", "clients-report", "whats-up-report")},
    "adapt-reset": "adapt-reset: put the learning-from-use gate back to its prior",
}
# the commands the product CLI also runs (brand.COMMAND held, ...): their usage lines name that command
FRONT = ("held", "trust", "forget", "restore", "search", "add")
NEEDS_ARG = {"add": "TEXT", "search": "QUERY", "search-chats": "QUERY", "extract": "SESSION_ID"}
# add's and search's options, and whether each takes a value: every word before the first of them is the text
OPTIONS = {"add": {"--kind": True, "--project": True, "--source": True, "--origin": True},
           "search": {"-k": True, "--section": True, "--outside": False}}


def words_and_options(cmd, args):
    """(text, {option: value}) for `add` or `search` from what follows the command: the text is every word before
    the first option, joined by spaces, so it need not be quoted (`add buy milk tomorrow` once saved "buy"). Raises
    ValueError with the reason for an option that is not this command's, or one without its value."""
    known = OPTIONS[cmd]
    n = next((i for i, a in enumerate(args) if a in known or a.startswith("--")), len(args))
    text, rest, opts = " ".join(args[:n]), args[n:], {}
    while rest:
        a = rest.pop(0)
        if a not in known:
            raise ValueError(f"unknown option {a} (put the text in quotes if it is part of it)")
        if not known[a]:
            opts[a] = True
            continue
        if not rest or (a == "-k" and not rest[0].isdigit()):
            raise ValueError(f"{a} needs a {'number' if a == '-k' else 'value'}")
        opts[a] = rest.pop(0)
    return text, opts


def _prog(cmd):
    return COMMAND if cmd in FRONT else "lflabs-memory"


def _refuse(cmd, why):
    print(f"{_prog(cmd)}: {why}\n" + (f"usage: {_prog(cmd)} {COMMANDS[cmd]}" if cmd else USAGE), file=sys.stderr)
    sys.exit(2)


def check(args):
    """Answer --help, an unknown command or a missing argument here (it exits); else return, and mem.py runs it."""
    cmd = args[0] if args else "stats"
    if cmd in ("-h", "--help", "help"):
        print(USAGE)
        sys.exit(0)
    if cmd not in COMMANDS:
        _refuse(None, f"unknown command {cmd!r}\n")
    if any(a in ("-h", "--help") for a in args[1:]):
        print(f"usage: {_prog(cmd)} {COMMANDS[cmd]}")
        sys.exit(0)
    first = args[1] if len(args) > 1 else ""
    if cmd in NEEDS_ARG and (not first or first.startswith("-")):
        _refuse(cmd, f"{cmd} needs {NEEDS_ARG[cmd]} first")
    if cmd in OPTIONS:
        try:
            words_and_options(cmd, args[1:])
        except ValueError as e:
            _refuse(cmd, str(e))
    if cmd in ("forget", "restore") and not re.fullmatch(r"[cC]?\d{1,9}", first):
        _refuse(cmd, f"{cmd} needs a memory id (a number)")


def _locale_encoding():
    """The locale's own encoding (the Terminal's), whatever mode Python is in; "utf-8" when it cannot be told."""
    try:
        return codecs.lookup(locale.getencoding()).name
    except LookupError:
        return "utf-8"


def utf8_everywhere():
    """Under a locale that is not UTF-8 (LANG=zh_CN.GB2312, ja_JP.SJIS, de_DE.ISO8859-1) Python reads and writes
    every file, and the JSON an app sends a hook, in that locale: a prompt in Japanese was a decode error (no recall,
    no typed save), an accented one was saved garbled, and printing a memory the locale cannot write was a traceback.
    The store, the transcripts and the apps all speak UTF-8, so start again in Python's UTF-8 mode (it cannot be
    turned on once Python runs); words typed on the command line in the locale's encoding are passed on as UTF-8.
    Then only a Terminal is written in the locale's encoding, with ? for what it cannot show."""
    enc = _locale_encoding()
    if enc == "utf-8":
        return
    if not sys.flags.utf8_mode and "PYTHONUTF8" not in os.environ and len(sys.orig_argv) > 1:
        def arg(a):
            raw = os.fsencode(a)
            try:
                raw.decode("utf-8")
                return raw  # UTF-8 already (an app's argument)
            except UnicodeDecodeError:
                return raw.decode(enc, "replace").encode("utf-8")
        os.environ["PYTHONUTF8"] = "1"
        os.execv(sys.executable, [sys.executable, *(arg(a) for a in sys.orig_argv[1:])])
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and stream.isatty():
            stream.reconfigure(encoding=enc, errors="replace")


def main(argv=None, profile=None):
    """Run one mem.py command. argv: its arguments (default: the command line's). profile: a host's profile name (the
    host's launcher passes it), else config.toml's or generic. Set before mem.py loads, since its paths are read once;
    the processes it starts inherit it."""
    if profile:
        if config.CFG is not None and config.CFG.profile.name != profile:
            raise RuntimeError(f"the engine already runs the {config.CFG.profile.name} profile, not {profile}")
        os.environ[config.PROFILE_ENV] = profile
    if argv is not None:
        sys.argv = [sys.argv[0], *argv]
    else:  # the command itself (a host's launcher, which passes argv, runs in its own Python as it set it up)
        utf8_everywhere()
    check(sys.argv[1:])
    try:
        from . import mem, models
    except (ValueError, RuntimeError) as e:  # config.toml that is not TOML, a bad value, a profile not installed
        print(f"lflabs-memory: settings not loaded: {'config.toml: ' * isinstance(e, tomllib.TOMLDecodeError)}{e}",
              file=sys.stderr)
        sys.exit(1)
    try:
        return mem.main()
    except models.Refused as e:  # a model file missing, or not the bytes models.json lists: the reason, no traceback
        print(f"lflabs-memory: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
