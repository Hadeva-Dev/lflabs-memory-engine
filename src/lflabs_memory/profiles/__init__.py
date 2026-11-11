"""Profiles: what differs between one install and another. The engine ships one, generic (profiles/generic.py): the
sections, filing rules, access words and prompts any user gets. A host assistant may bring its own through the entry
point `lflabs_memory.profiles: <name> = <module>:<function>`, a function of the host's folder (LFLABS_MEMORY_HOST_DIR,
or None) returning a Profile. The engine ships no host's profile.
"""
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib.metadata import entry_points

from .. import brand
from .hooks import ProfileHooks

GROUP = "lflabs_memory.profiles"


@dataclass(frozen=True)
class Profile:
    name: str
    # Sections (what a memory is about) and how memories are filed into them
    sections: tuple[str, ...]
    section_aliases: dict[str, str]           # other spellings of a section ("preferences" -> prefs)
    project_sections: dict[str, str]          # a memory's project field -> its section
    project_aliases: dict[str, str]           # a project's other spellings
    not_project_slugs: frozenset[str]         # folder names saved as a project that are none
    project_topics: frozenset[str]            # sections a project memory files under; they take a second by kind
    kind_sections: dict[str, str]             # kind -> the second section of a project memory
    default_filing: Callable | None           # (content, kind, project, source) -> (section, section2) for what the
                                              # rules leave unfiled, or None: a model files those (classify)
    cwd_sections: tuple                       # ((folder, section), ...): a session in or under folder; first wins
    code_roots: tuple                         # folders whose every subfolder is a project
    cwd_other: str                            # the section of any other project folder
    # Who sees what
    restricted: dict[str, tuple[str, ...]]    # scope -> sections it never sees
    work_only: tuple[str, ...]                # the "work only" level: an allow list
    nothing_personal: tuple[str, ...]         # the "nothing personal" level: a deny list
    unknown_deny: tuple[str, ...]             # what an unknown client never sees (KEYLESS_MODE off)
    menu_shown: tuple[str, ...]               # sections the menu-bar app may show
    # Changing access by saying it (access.py): section words
    scope_strong: dict[str, tuple[str, ...]]  # name a section by themselves
    scope_weak: dict[str, tuple[str, ...]]    # name a section only after "my"
    widen_needs_my: tuple[str, ...]           # widen only after "my"; narrow by themselves
    scope_label: dict[str, str]               # how a section is named back to the user
    access_sections: str                      # the sections the access_change tool's description lists
    recall_sections: str                      # the sections the recall tool's description lists
    chat_projects: str                        # project names the search_chats tool's description gives as examples
    # Prompts
    classify_system: str                      # the classifier's system prompt (it names every section)
    extract_intro: str | None = None          # the extractor's first sentence (None: the engine's, for any user)
    section_projects: dict = field(default_factory=dict)  # a section -> its project (the audit's naming)
    llm: dict = field(default_factory=dict)  # config.toml [llm] defaults over the plain install's (config.LLMConfig);
                                              # the user's config.toml still wins
    experiments: dict = field(default_factory=dict)  # switch -> value, over the plain install's defaults (mem.py
                                              # _switch): a host's experiments, chat index, Codex reading, banners
    possessives: tuple[str, ...] = ()         # words for the owner's, read as "my" (as in "show Codex her texts");
                                              # () for "their" and the owner's name
    # Whose memory this is, and what the engine is called where the user sees it
    owner: str | None = None                  # what text calls the user (None: config.toml's owner_name, else
                                              # "the user")
    own: Callable | None = None               # text written for the user -> for this owner (None: owner.they)
    server_name: str = brand.COMMAND          # the MCP server's name in every app's config ("lflabs-memory" before
                                              # 0.1.8: connect still knows an entry of ours under it)
    menu_app: str | None = None               # what the host's menu-bar app is called (None: brand.PRODUCT)
    label: str | None = None                  # how messages from memory start ("<label>: ..."), and what the recall
                                              # block calls memory (None: brand.PRODUCT)
    hooks: ProfileHooks = field(default_factory=ProfileHooks)


def load(name, host_dir=None):
    """The profile called name: generic, or the one a host registered. A host profile that is missing or fails to load
    is refused, never run as something else: its trust rules depend on it (without a host's session files, words from
    its channels would read as typed at the desk)."""
    if name == "generic":
        from .generic import PROFILE
        return PROFILE
    for ep in entry_points(group=GROUP, name=name):
        profile = ep.load()(host_dir)
        if not isinstance(profile, Profile):
            raise RuntimeError(f"the {name} profile ({ep.value}) returned {type(profile).__name__}, not a Profile")
        return profile
    raise RuntimeError(f"the {name} profile needs its host's profile (entry point {GROUP}: {name}); none is installed")
