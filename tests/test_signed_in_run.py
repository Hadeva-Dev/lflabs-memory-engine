"""The signed-in run (real Claude Code 2.1.322 on 0.1.8, 2026-11-10): a forgotten fact still reached Codex through
the "Where you left off" notes; Claude Code's own memory file for a typed note was imported as a second copy that the
forget left behind; background reading filed plain work facts as personal, so work-only apps never got them; and a
scripted `claude -p` run was told to call a tool it cannot.

  .venv/bin/python -m unittest -q test_signed_in_run
"""
import _env  # noqa: F401  (first: the test data folder and user)
import json
import os
import tempfile
import time
from pathlib import Path
from unittest import mock

from lflabs_memory import claimstore as cs
from lflabs_memory import mem
from test_claims import Base, Extraction
from test_forget_erase import _with_backend

NOTE = "the build cache lives on the volume named kestrel"
HANDOFF_ITEM = "Saved the build cache location (volume kestrel) to project memory as build-cache-volume.md, indexed in MEMORY.md."
CLAUDE_FILE = """---
name: build-cache-volume
description: Where the build cache lives
type: project
---
The build cache lives on the volume named kestrel.
"""


class Files(Base):
    """No model: the forgotten words are then kept nowhere at all (with one, the claims sweep holds them up to a day,
    as forget_reply says)."""

    def setUp(self):
        super().setUp()
        p = mock.patch.object(mem, "CFG", _with_backend(mem.CFG, "none"))
        p.start()
        self.addCleanup(p.stop)

    def holders(self, word="kestrel"):
        files = [mem.DB_PATH, mem.DB_PATH.with_name("memory.db-wal"), *mem.paths().backup.glob("*")]
        self.db.execute("pragma wal_checkpoint(TRUNCATE)")
        return [f.name for f in files if f.exists() and word.encode() in f.read_bytes().lower()]


class ForgetReachesTheNotes(Files):
    def handoff(self, text=HANDOFF_ITEM):
        with self.db:
            mem._handoff_tables(self.db)
            self.db.execute("insert into handoffs values (?,?,?,?,?,?)", (
                "shop", "desk", "c2287c30", "/repo/shop", time.time(),
                json.dumps({"done": [{"text": text, "section": "projects"}],
                            "open": [{"text": "Check the CI timeout on the shop repo", "section": "projects"}]})))

    def test_a_handoff_note_holding_the_forgotten_words_goes(self):
        mid, _ = mem.remember(self.db, NOTE, "project", "shop", origin="owner_typed")
        self.handoff()
        mem.forget_ids(self.db, mid)
        body = json.loads(self.db.execute("select body from handoffs").fetchone()[0])
        self.assertNotIn("kestrel", json.dumps(body).lower())
        self.assertEqual(body["open"][0]["text"], "Check the CI timeout on the shop repo")  # the rest stays

    def test_and_from_every_backup_copy(self):
        mid, _ = mem.remember(self.db, NOTE, "project", "shop", origin="owner_typed")
        self.handoff()
        mem.snapshot(self.db, "before")
        self.assertTrue(any(n.startswith("memory-") for n in self.holders()))
        mem.forget_ids(self.db, mid)
        self.assertEqual(self.holders(), [])  # memory.db, its log, the snapshot

    def test_a_chat_index_chunk_holding_it_goes(self):
        mid, _ = mem.remember(self.db, NOTE, "project", "shop", origin="owner_typed")
        with self.db:
            self.db.execute("insert into convo_chunks(session_id, ord, ts, text, hash) values (?,?,?,?,?)",
                            ("s1", 0, time.time(), "USER: remember that " + NOTE, "h1"))
            self.db.execute("insert into convo_chunks(session_id, ord, ts, text, hash) values (?,?,?,?,?)",
                            ("s1", 1, time.time(), "USER: what about the orchard deploy?", "h2"))
        mem.forget_ids(self.db, mid)
        left = [r[0] for r in self.db.execute("select text from convo_chunks")]
        self.assertEqual(left, ["USER: what about the orchard deploy?"])

    def test_audit_lines_naming_it_go(self):
        mid, _ = mem.remember(self.db, NOTE, "project", "shop", origin="owner_typed")
        audit = self.tmp / "AUDIT.md"
        audit.write_text(f"# Audit\n- #{mid} rewrote: {NOTE}\n- #{mid + 100} kept: the deploy window\n")
        with mock.patch.object(mem, "AUDIT_FILE", audit):
            mem.forget_ids(self.db, mid)
        self.assertNotIn("kestrel", audit.read_text())
        self.assertIn("the deploy window", audit.read_text())


class ScopeParity(ForgetReachesTheNotes):
    """An app's forget (held a day, then erased by purge_holds) clears only what that app may see (security review
    2026-11-10): a work-only app never deletes a personal handoff note or a chat-index chunk."""

    def setUp(self):
        super().setUp()
        for p in (mock.patch.object(mem, "CLIENT_SCOPES_MODE", "on"), mock.patch.object(mem, "KEYLESS_MODE", "on"),
                  mock.patch.dict(mem.RESTRICTED)):
            p.start()
            self.addCleanup(p.stop)

    def setup_notes(self):
        mid, _ = mem.remember(self.db, NOTE, "project", "shop", origin="owner_typed")
        with self.db:
            self.db.execute("update memories set section = 'projects' where id = ?", (mid,))
            mem._handoff_tables(self.db)
            self.db.execute("insert into handoffs values (?,?,?,?,?,?)", (
                "shop", "desk", "s-personal", "/repo/shop", time.time(),
                json.dumps({"done": [{"text": HANDOFF_ITEM, "section": "personal"},
                                     {"text": "Moved the build cache onto the volume named kestrel",
                                      "section": "projects"}]})))
            self.db.execute("insert into convo_chunks(session_id, ord, ts, text, hash) values (?,?,?,?,?)",
                            ("s1", 0, time.time(), "USER: remember that " + NOTE, "h1"))
        return mid

    def items(self):
        rows = self.db.execute("select body from handoffs").fetchall()
        return [it["text"] for r in rows for v in json.loads(r[0]).values() for it in v]

    def test_a_work_only_apps_forget_leaves_what_it_cannot_see(self):
        mid = self.setup_notes()
        scope = mem.client_scope("cursor", False)
        self.assertTrue(mem.RESTRICTED.get(scope))  # work only
        ids, _ = mem.app_forget(self.db, mid, "cursor", scope)
        self.assertEqual(ids[:1], [mid])
        mem.purge_holds(self.db, now=time.time() + mem.APP_FORGET_HOLD + 60)
        self.assertEqual(self.items(), [HANDOFF_ITEM])  # its own section's item went; the personal one stays
        self.assertEqual(self.db.execute("select count(*) from convo_chunks").fetchone()[0], 1)  # no section: kept

    def test_a_restricted_sessions_forget_leaves_what_it_cannot_see(self):
        mid = self.setup_notes()
        scope = mem.client_scope("cursor", False)
        self.assertEqual(mem.forget_ids(self.db, mid, scope), [mid])
        self.assertEqual(self.items(), [HANDOFF_ITEM])
        self.assertEqual(self.db.execute("select count(*) from convo_chunks").fetchone()[0], 1)

    def test_the_owners_forget_clears_them(self):
        mid = self.setup_notes()
        mem.forget_ids(self.db, mid)
        self.assertEqual(self.items(), [])
        self.assertEqual(self.db.execute("select count(*) from convo_chunks").fetchone()[0], 0)

    def test_two_shared_words_are_not_enough(self):
        self.assertEqual(len(mem._content_words("the orchard window opens Thursday")), 4)  # two shared are half
        self.assertFalse(mem.holds_words("the orchard window opens Thursday",
                                         "USER: the orchard team met on Thursday about lunch plans " * 20))
        self.assertTrue(mem.holds_words(NOTE, HANDOFF_ITEM))


class ClaudeCodesOwnFile(Files):
    def setUp(self):
        super().setUp()
        self.home = Path(tempfile.mkdtemp(prefix="cchome"))
        self.file = self.home / ".claude" / "projects" / "-Users-x-code-shop" / "memory" / "build-cache-volume.md"
        self.file.parent.mkdir(parents=True)
        p = mock.patch("pathlib.Path.home", return_value=self.home)
        p.start()
        self.addCleanup(p.stop)

    def active_with(self, word="kestrel"):
        return [r["id"] for r in self.db.execute("select id, content from memories where status = 'active'")
                if word in (r["content"] or "").lower()]

    def test_the_file_is_the_typed_notes_copy_and_goes_with_it(self):
        mid, _ = mem.remember(self.db, NOTE, "project", "shop", origin="owner_typed")
        self.file.write_text(CLAUDE_FILE)
        mem.import_files(self.db)
        self.assertEqual(self.active_with(), [mid])  # one note, not two
        mem.forget_ids(self.db, mid)
        self.assertEqual(self.active_with(), [])
        self.assertEqual(self.holders(), [])  # "kestrel" in no file
        self.assertTrue(self.file.exists())  # Claude Code's own file is left alone
        mem.import_files(self.db)  # and it is never read in again
        self.file.write_text(CLAUDE_FILE + "\nUpdated by Claude Code.\n")
        mem.import_files(self.db)
        self.assertEqual(self.active_with(), [])

    def test_forgetting_the_file_copy_takes_the_note_too(self):
        self.file.write_text(CLAUDE_FILE)
        mem.import_files(self.db)  # imported before the typed note was there
        fid = self.active_with()[0]
        mid, _ = mem.remember(self.db, NOTE, "project", "shop", origin="owner_typed")
        mem.forget_ids(self.db, fid)
        self.assertEqual(self.active_with(), [])
        self.assertEqual(self.db.execute("select content from memories where id = ?", (mid,)).fetchone()[0], "")

    # Gate 0.1.10 rc1 (false_twin): a long file merely containing a short note's words was linked as its copy, so
    # its other facts left search, and forgetting the note erased the whole file and blocked it from coming back.
    LONG_FILE = """---
name: shop-infra
description: Notes on the shop's infrastructure
type: project
---
The build cache lives on the volume named kestrel. Deploys run on Fridays from the orchid pipeline.
Staging uses Postgres 16 on the heron host. The CDN is Fastly, with a ten-minute TTL on product pages.
The payments webhook retries five times, then pages the on-call engineer through the plover rota.
"""
    THREE_FACTS = """---
name: shop-notes
description: Shop notes
type: project
---
- The build cache lives on the volume named kestrel.
- Deploys run on Fridays from the orchid pipeline.
- Staging uses the heron host.
"""

    def file_row(self):
        return self.db.execute("select id, status, merged_into, content from memories where source = ?",
                               (str(self.file),)).fetchone()

    def searchable(self, mid):
        return self.db.execute("select count(*) from chunks where memory_id = ?", (mid,)).fetchone()[0] > 0

    def assert_own_memory_through_a_forget(self, text):
        mid, _ = mem.remember(self.db, NOTE, "project", "shop", origin="owner_typed")
        self.file.write_text(text)
        mem.import_files(self.db)
        row = self.file_row()
        self.assertEqual((row["status"], row["merged_into"]), ("active", None), "its own memory, not the note's copy")
        self.assertTrue(self.searchable(row["id"]))
        self.assertEqual(mem.file_twins(self.db, mid), [])
        mem.forget_ids(self.db, mid)
        after = self.file_row()
        self.assertEqual(after["status"], "active")
        self.assertIn("heron", after["content"].lower(), "the file's other facts stay")
        self.assertTrue(self.searchable(after["id"]))
        self.file.write_text(text + "\nUpdated by Claude Code.\n")
        self.assertNotIn("refused", mem.remember(self.db, *mem._parse_md(self.file), source=str(self.file))[1])

    def test_a_long_file_holding_a_notes_words_is_not_its_copy(self):
        self.assertTrue(mem.holds_words(NOTE, self.LONG_FILE))  # what linked it before
        self.assert_own_memory_through_a_forget(self.LONG_FILE)

    def test_a_file_of_three_facts_one_the_note_is_not_its_copy(self):
        self.assert_own_memory_through_a_forget(self.THREE_FACTS)

    def test_the_same_fact_links_both_ways(self):
        self.assertTrue(mem.same_fact_file(NOTE, mem._parse_md(self._write(CLAUDE_FILE))[0]))
        self.assertFalse(mem.same_fact_file(NOTE, self.LONG_FILE))
        self.assertFalse(mem.same_fact_file(NOTE, self.THREE_FACTS))
        self.assertFalse(mem.same_fact_file("the build cache lives on the volume named kestrel and staging uses the"
                                            " heron host with Postgres 16 behind the orchid pipeline", CLAUDE_FILE),
                         "a long note is not the copy of a short file either")

    def _write(self, text):
        self.file.write_text(text)
        return self.file

    # Claude Code 2.1.322's own file for the typed note, as the signed-in rerun saw it (2026-11-11): named after the
    # subject, frontmatter, the fact, and who said it when.
    REAL_FILE = """---
name: build-cache-volume-kestrel
description: The build cache lives on the volume named kestrel
type: project
---

The build cache lives on the volume named kestrel.

Stated by the user on 2026-11-11.
"""

    def real_file(self):
        f = self.file.with_name("build-cache-volume-kestrel.md")
        f.write_text(self.REAL_FILE)
        return f

    def test_the_real_file_shape_is_the_notes_copy(self):
        """Compared on its body, frontmatter and "Stated by the user" line left out (twin_texts)."""
        content = mem._parse_md(self.real_file())[0]
        self.assertTrue(content.startswith("build-cache-volume-kestrel - The build cache"))
        self.assertEqual(mem.twin_texts(content)[-1], "The build cache lives on the volume named kestrel.")
        mid, _ = mem.remember(self.db, NOTE, "project", "shop", origin="owner_typed")
        mem.import_files(self.db)
        self.assertEqual(self.active_with(), [mid])

    def test_a_forgotten_subject_is_not_kept_in_the_files_name(self):
        """Signed-in rerun 2026-11-11: after the forget, "kestrel" stayed in memory.db in the imported row's source,
        Claude Code's file name. Now only a hash of the path stays: kestrel is nowhere in memory.db, its log or a
        snapshot, and the file is still never imported again."""
        f = self.real_file()
        mid, _ = mem.remember(self.db, NOTE, "project", "shop", origin="owner_typed")
        mem.import_files(self.db)
        mem.snapshot(self.db, "before-forget")  # a dated copy holding the file's row and its path
        with self.db:  # and a chat titled after it, in the chat index
            self.db.execute("insert into convos(session_id, project, title) values ('s1', 'shop',"
                            " 'Save build cache volume kestrel')")
            self.db.execute("insert into convos(session_id, project, title) values ('s2', 'shop', 'Fix the CI')")
        self.assertIn("memory.db", self.holders())
        mem.forget_ids(self.db, mid)
        self.assertEqual(self.holders(), [])
        rows = self.db.execute("select source, source_hash, source_key from memories where status = 'retired'"
                               " and source_key is not null").fetchall()
        self.assertEqual([(r[0], r[1]) for r in rows], [(None, None)])
        self.assertEqual(rows[0][2], mem.source_key(str(f)))
        self.assertEqual([r[0] for r in self.db.execute("select title from convos order by session_id")],
                         [None, "Fix the CI"])
        f.write_text(self.REAL_FILE + "\nUpdated by Claude Code.\n")
        self.assertEqual(mem.remember(self.db, *mem._parse_md(f), source=str(f))[1],
                         "refused: that file's memory was forgotten")
        mem.import_files(self.db)
        self.assertEqual(self.active_with(), [])
        self.assertEqual(self.holders(), [])

    def everywhere(self, word="kestrel"):
        """Every file under the data folder (backups and memories.json too) that holds the word."""
        self.db.execute("pragma wal_checkpoint(TRUNCATE)")
        root = mem.DB_PATH.parent
        return sorted(str(f.relative_to(root)) for f in root.rglob("*") if f.is_file()
                      and word.encode() in f.read_bytes().lower())

    def test_the_key_is_keyed_and_nothing_under_the_data_folder_keeps_the_subject(self):
        """Security review 2026-11-11: a plain sha256 of a guessable path confirms a guessed subject, and
        backup/memories.json kept the path. The key is an HMAC with this install's secret (outside memory.db and
        every backup), and memories.json is rewritten too."""
        import hashlib
        f = self.real_file()
        mid, _ = mem.remember(self.db, NOTE, "project", "shop", origin="owner_typed")
        mem.import_files(self.db)
        fid = self.db.execute("select id from memories where source = ?", (str(f),)).fetchone()[0]
        backup = mem.paths().backup
        backup.mkdir(parents=True, exist_ok=True)
        (backup / "memories.json").write_text(json.dumps([  # a host's backup: the row as it stood, and an older
            {"id": fid, "status": "merged", "content": "", "source": str(f), "source_hash": mem._hash("x")},
            {"id": 77, "status": "retired", "content": "", "source": str(f.with_name("orchid-db-kestrel.md")),
             "source_hash": mem._hash("y")}]))  # forgotten file's tombstone with its path
        mem.forget_ids(self.db, mid)
        self.assertEqual(self.everywhere(), [])
        key = self.db.execute("select source_key from memories where id = ?", (fid,)).fetchone()[0]
        for guess in (str(f), "memory-file:" + str(f)):
            self.assertNotIn(key, (hashlib.sha256(guess.encode()).hexdigest(),
                                   hashlib.sha256(guess.encode()).hexdigest()[:32]))
        self.assertEqual(len(key), 32)
        secret = mem.paths().source_secret
        self.assertEqual(oct(secret.stat().st_mode & 0o777), "0o600")
        self.assertIn(str(secret), mem.paths_json()["never_back_up"])
        self.assertNotIn(secret.read_bytes(), mem.DB_PATH.read_bytes())
        self.assertEqual(oct((backup / "memories.json").stat().st_mode & 0o777), "0o600")
        f.write_text(self.REAL_FILE + "\nUpdated by Claude Code.\n")
        self.assertEqual(mem.remember(self.db, *mem._parse_md(f), source=str(f))[1],
                         "refused: that file's memory was forgotten")

    def test_a_snapshot_of_an_older_store_is_rekeyed_even_when_the_store_has_nothing_left(self):
        f = self.real_file()
        with self.db:
            self.db.execute("insert into memories(content, kind, source, source_hash, hash, status, created, updated)"
                            " values ('', 'note', ?, ?, ?, 'retired', 0, 0)", (str(f), mem._hash("x"), mem._hash("")))
        mem.snapshot(self.db, "older")
        with self.db:
            mem.key_forgotten_sources(self.db)  # the store itself already done
            self.db.execute("delete from meta where key = 'source_keys_v1'")
        self.assertEqual([h for h in self.everywhere() if h.startswith("backup/")], self.everywhere())
        self.assertTrue(self.everywhere(), "only the snapshot still names it")
        mem._key_forgotten_sources_once(self.db)
        self.assertEqual(self.everywhere(), [])

    def test_an_older_store_keeps_only_a_hash_of_a_forgotten_files_path(self):
        """A store (and its snapshot) from before: the forgotten file's row still names it; once, it is hashed."""
        f = self.real_file()
        with self.db:
            self.db.execute("insert into memories(content, kind, source, source_hash, hash, status, created, updated)"
                            " values ('', 'note', ?, ?, ?, 'retired', 0, 0)", (str(f), mem._hash("x"), mem._hash("")))
            self.db.execute("delete from meta where key = 'source_keys_v1'")
        mem.snapshot(self.db, "older")
        self.assertIn("memory.db", self.holders())
        mem._key_forgotten_sources_once(self.db)
        self.assertEqual(self.holders(), [])
        self.assertEqual(mem.remember(self.db, *mem._parse_md(f), source=str(f))[1],
                         "refused: that file's memory was forgotten")

    def test_a_backups_forgotten_file_comes_back_as_a_hash(self):
        f = self.real_file()
        js = self.file.parent / "memories.json"
        js.write_text(json.dumps([{"id": 90, "status": "retired", "content": "", "source": str(f),
                                   "source_hash": mem._hash("x")},
                                  {"id": 91, "status": "retired", "content": "",
                                   "source_key": mem.source_key("/Users/x/.claude/projects/p/memory/other.md")}]))
        self.assertEqual(mem.import_backup(self.db, js)["tombstone"], 2)
        self.assertEqual(self.db.execute("select count(*) from memories where source is not null").fetchone()[0], 0)
        self.assertEqual(mem.remember(self.db, *mem._parse_md(f), source=str(f))[1],
                         "refused: that file's memory was forgotten")

    def test_a_file_written_after_the_forget_is_not_imported(self):
        mid, _ = mem.remember(self.db, NOTE, "project", "shop", origin="owner_typed")
        mem.forget_ids(self.db, mid)
        self.file.write_text(CLAUDE_FILE)
        mem.import_files(self.db)
        self.assertEqual(self.active_with(), [])


class WorkFactsAreWork(Extraction):
    """Claims from background reading: work facts reach every app, the user's own life stays personal."""
    WORK = ["Robin's team deploys on Fridays and never on Mondays.",
            "The staging database for the shop project is called orchid.",
            "Robin uses pnpm in the shop repo.",
            "Robin's team reviews pull requests every morning.",
            "The build cache lives on the volume named kestrel."]
    PERSONAL = ["Robin's blood pressure medicine is lisinopril.",
                "Robin's sister Bea lives in Lisbon.",
                "Robin is seeing a therapist on Tuesdays.",
                "Robin owes the bank 4000 dollars on the car loan.",
                "Robin and his wife are separating."]

    MIXED = ["Robin's therapist moved Robin's session, so Robin will miss Friday's deploy.",  # personal + work words
             "Robin is interviewing at Stripe next week.",  # employment
             "Robin asked for a raise at the Friday standup.",
             "Robin's manager put Robin on a performance improvement plan for the backend project.",
             "Robin filed an HR complaint about the release manager.",
             "Robin's girlfriend reviews the shop project's copy."]

    def test_each_way(self):
        for fact in self.WORK:
            self.assertEqual(cs.claim_section(mem, fact), "projects", fact)
        for fact in self.PERSONAL:
            self.assertEqual(cs.claim_section(mem, fact), "personal", fact)

    def test_any_personal_or_sensitive_sign_keeps_it_personal(self):
        """Fails closed (security review 2026-11-10): work words never outweigh a personal or sensitive one."""
        for fact in self.MIXED:
            self.assertEqual(cs.claim_section(mem, fact), "personal", fact)
        self.assertEqual(cs.claim_section(mem, "Leo starts at Stripe on November 2."), "personal")  # not work

    def test_the_users_family_by_name_and_relation_is_personal(self):
        """Security review 2026-11-10, second round: the owner by first, last or full name, and family and relationship
        words on the fact as it is, with no "my" needed."""
        with mock.patch.object(mem, "OWNER_NAME", "Robin Vale"), mock.patch.object(mem, "OWNER", "Robin Vale"):
            for fact in ("Robin's mom is in the hospital.", "Vale and his wife are separating.",
                         "Robin Vale's divorce is final in March.", "Vale's sister is getting married in June.",
                         "Meeting with the team about my divorce lawyer."):
                self.assertEqual(cs.claim_section(mem, fact), "personal", fact)
            self.assertEqual(cs.first_person(mem, "Vale's divorce"), "my divorce")
            self.assertEqual(cs.first_person(mem, "Robin Vale's mom"), "my mom")
            for fact in ("Robin's team deploys on Fridays and never on Mondays.",
                         "The staging database for the shop project is called orchid."):
                self.assertEqual(cs.claim_section(mem, fact), "projects", fact)
        for fact in ("Bea's wedding is in June.", "Robin's kids start school on Monday.",
                     "The deploy freeze moved because Robin's dad is in hospital."):
            self.assertEqual(cs.claim_section(mem, fact), "personal", fact)

    def test_a_generic_word_alone_is_not_work(self):
        """"team", "meeting", "project", "review", "class" or "build" also name the user's own life."""
        for fact in ("Robin has a team dinner on Thursday.", "Robin's book club meeting is on Monday.",
                     "Robin's project at home is a treehouse.", "Robin takes a yoga class on Tuesdays.",
                     "Robin wants to build a shed this spring.", "Robin reviews restaurants for fun."):
            self.assertEqual(cs.claim_section(mem, fact), "personal", fact)

    def test_at_the_source(self):
        turns = ["we deploy on Fridays and never on Mondays", "my blood pressure medicine is lisinopril"]
        out = {"forget": [], "claims": [
            {"text": "Robin's team deploys on Fridays and never on Mondays.", "from": [1], "status": "fact",
             "replaces": []},
            {"text": "Robin's blood pressure medicine is lisinopril.", "from": [2], "status": "fact", "replaces": []}]}
        self.slice(turns, out)
        self.assertEqual(self.by_text("deploys")["section"], "projects")
        self.assertEqual(self.by_text("lisinopril")["section"], "personal")

    # the Extraction tests themselves run in test_claims
    for _n in [n for n in dir(Extraction) if n.startswith("test_")]:
        locals()[_n] = None
    del _n


class ScriptedRun(Base):
    def test_a_scripted_run_is_not_asked_to_call_a_tool(self):
        out = mem.prompt_context(self.db, "remember that the staging database is called orchid", "s1", count=False,
                                 headless=True) or ""
        self.assertIn("scripted run", out)
        self.assertNotIn("save it now with", out)
        interactive = mem.prompt_context(self.db, "remember that the staging database is called orchid", "s2",
                                         count=False) or ""
        self.assertIn(mem.REMEMBER_NUDGE, interactive)

    def test_the_hook_says_so_for_claude_p(self):
        seen = {}
        with mock.patch.dict(os.environ, {"CLAUDE_CODE_ENTRYPOINT": "sdk-cli"}), \
                mock.patch.object(mem, "SOCK", self.tmp / "none.sock"), \
                mock.patch.object(mem, "prompt_context", lambda *a, **k: seen.update(k) or None), \
                mock.patch.object(mem, "connect", lambda: self.db):
            mem.hook_prompt({"prompt": "remember that the staging database is called orchid", "session_id": "s3"})
        self.assertTrue(seen.get("headless"))
