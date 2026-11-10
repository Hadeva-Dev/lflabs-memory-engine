"""What a host assistant adds to the engine. A plain install has none: every field is None and the engine does
without (no voice or Telegram sessions, no people index, no vouching, no menu-bar app check, no activity log).

A host's profile carries its hooks (profiles/__init__.py), from the host's own code. The engine ships none of
that code.
"""
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProfileHooks:
    # Files the host writes and the engine only reads
    voice_session: Path | None = None      # {"id": the voice conversation's session id}
    telegram_session: Path | None = None   # {"id": the Telegram conversation's session id}
    meeting_state: Path | None = None      # {"briefed": {id: {"start": time}}}: meetings with other people
    people_file: Path | None = None        # {"people": [...]}: the host's people index
    vitals_pause: Path | None = None       # {"until": time}: heavy background work waits until then
    # Host code the engine calls
    activity: object | None = None                          # .log(source, kind, text): the host's activity log
    vouch: Callable[[list[int]], int] | None = None         # queue memory ids for the owner to vouch for
    menu_peer_ok: Callable[[object], bool] | None = None    # whether an audit token's process is the host's signed
                                                            # menu-bar app
    refresh_people: Callable[[], object] | None = None      # rebuild people_file; returns a CompletedProcess

