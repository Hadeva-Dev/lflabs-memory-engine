"""Run by test_owner_prompts.py in a fresh process (its config.toml names the owner, or not): drive every job that
asks a model, with the model call (llm.llm_json) replaced by a recorder, and print what each call would send as JSON
lines {"job", "system", "prompt"}. The module-level prompts and the claims prompt builders are printed too."""
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lflabs_memory import claimstore as cs, facets_live, llm, mem  # noqa: E402
from test_claims import bow_embed, overlap_rerank  # noqa: E402

sent = []


def fake(prompt, schema, system, *, job="other", **k):
    sent.append({"job": job, "system": system, "prompt": prompt})
    if job == "extract":
        return {"memories": [{"content": "Orchard: staging listens on port 6502.", "kind": "project",
                              "project": "orchard", "from": [1]}]}
    if job == "claims-extract":
        return {"claims": [], "forget": [1]}
    if job == "claims-topics":
        return {"topics": ["gym locker"]}
    if job == "claims-pick":
        return {"forget": [], "mixed": [1]}
    if job == "claims-rewrite":
        return {"kept": "The owner goes to the gym.", "removed": "the locker number", "sure": True}
    if job == "merge":
        return {"content": "Orchard deploys run on Fridays after the checks.", "kind": "project", "project": "orchard"}
    if job in ("audit", "review"):
        return {"action": "keep", "content": ""}
    if job == "handoff":
        return {"decided": [], "next": [], "done": [], "open": []}
    if job == "translate":
        return {"english": "where is my gym locker"}
    return {"items": []}


tmp = Path(tempfile.mkdtemp(prefix="promptcap"))
patches = [mock.patch.object(llm, "llm_json", fake), mock.patch.object(mem, "DB_PATH", tmp / "memory.db"),
           mock.patch.object(mem, "ROOT", tmp), mock.patch.object(mem, "OUTLINE_FILE", tmp / "OUTLINE.md"),
           mock.patch.object(mem, "embed", bow_embed), mock.patch.object(mem, "rerank", overlap_rerank),
           mock.patch.object(mem, "CLAIMS_MODE", "on"), mock.patch.object(mem, "HANDOFF_MODE", "on"),
           mock.patch.object(mem, "HOLDOUT_MODE", "off"), mock.patch.object(mem, "SENSE_MODE", "off"),
           mock.patch.object(mem, "activity", None),
           mock.patch.dict(os.environ, {"LFLABS_MEMORY_SCOPE": "", "LFLABS_MEMORY_NOSAVE": "", mem.CHILD_ENV: ""})]
patches += [mock.patch.object(mem, n, tmp / n) for n in ("VOICE_SESSION", "TELEGRAM_SESSION", "MEETBRIEF_STATE",
                                                         "PEOPLE_FILE", "SOCK")]
for p in patches:
    p.start()
db = mem.connect()
owner = mem.OWNER


def step(name, fn):
    try:
        fn()
    except Exception as e:  # what was sent before it failed still counts; the failure is printed for the test
        print(json.dumps({"job": "error:" + name, "system": f"{type(e).__name__}: {e}", "prompt": ""}))


# A coding session: the user's lines (one asks to forget), the agent's reply, an open-mic line: extraction, the
# handoff, the claims writer, then the forget sweep's topics, pick and rewrite.
cs.add(mem, db, f"{owner}'s gym locker is number 12 and the gym is on Elm Street.", "owner_typed", kind="user")
path = tmp / "s1.jsonl"
lines = [{"type": "user", "timestamp": "2026-10-16T21:00:00Z", "message": {"content": t}} for t in (
    "Orchard staging listens on port 6502 now, keep the deploy on Fridays",
    "please forget my gym locker number")]
lines.insert(1, {"type": "assistant", "timestamp": "2026-10-16T21:00:05Z",
                 "message": {"content": [{"type": "text", "text": "Noted: staging is on 6502."}]}})
lines.append({"type": "user", "timestamp": "2026-10-16T21:01:00Z",
              "message": {"content": mem._provenance().OPEN_MIC + "Leo starts at Stripe on November 2"}})
path.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
step("hook", lambda: mem.hook_activity({"session_id": "s1", "transcript_path": str(path), "cwd": str(tmp / "orchard")},
                                       final=True))
step("extract", lambda: mem.extract(db, "s1"))
step("claims", lambda: cs.claims_tick(mem, db, log=lambda m: None))
# The background jobs over the store: classify, secrets, consolidate (merge and review), the status-note audit.
for text in ("Orchard deploys run on Fridays after the checks pass.", "Orchard deploys run on Fridays after checks.",
             "The old orchard deploy script is deprecated, replaced by make ship.",
             "The alarm panel is by the back door."):
    mem.remember(db, text, "project", "orchard")
with db:
    db.execute("update memories set section = null, filed_by = null")
step("classify", lambda: mem.classify(db, log=lambda m: None))
step("secrets", lambda: mem.mark_secrets(db, apply=True, log=lambda m: None))
step("consolidate", lambda: mem.consolidate(db, log=lambda m: None))
old = time.time() - 60 * 86400
mid = mem.remember(db, "Orchard status as of 2026-09-01: open items: the launch checklist.", "project", "orchard")[0]
with db:
    db.execute("update memories set created = ?, updated = ?, reviewed = null where id = ?", (old, old, mid))
step("audit", lambda: mem.review_status_notes(db, log=lambda m: None, now=time.time()))
step("translate", lambda: facets_live.translation(mem, db, "onde está o meu cacifo do ginásio"))
for x in sent:
    print(json.dumps(x))
# What the prompt builders and the module-level prompts say, whether or not a job above reached them.
for name in ("system_prompt", "topics_prompt", "pick_prompt", "rewrite_prompt"):
    print(json.dumps({"job": "built:claims-" + name, "system": getattr(cs, name)(owner), "prompt": ""}))
for name in ("EXTRACT_SYSTEM", "MERGE_SYSTEM", "REVIEW_SYSTEM", "CLASSIFY_SYSTEM", "SECRET_SYSTEM", "STATUS_NOTE_SYSTEM",
             "HANDOFF_SYSTEM"):
    print(json.dumps({"job": "built:" + name, "system": getattr(mem, name), "prompt": ""}))
print(json.dumps({"job": "built:TRANSLATE_SYSTEM", "system": facets_live.TRANSLATE_SYSTEM, "prompt": ""}))
