"""A store brought up to this version (`migrate`, connect()), and `paths --json` for the programs that read the store.
Synthetic only: temp stores, a fake embedder.

  .venv/bin/python -m unittest -q test_migrate
"""
import _env  # noqa: F401  (first: the test data folder and user)
import hashlib
import io
import json
import os
import subprocess
import sys
from contextlib import redirect_stdout
from unittest import mock

from lflabs_memory import claimstore, config, mem
from test_redact import DBCase

# A 0.1.3 store's names, stood in for by made-up ones: the engine knows the real old names only by their sha256
# (mem.LEGACY_ORIGINS and the rest), so the test puts the hashes of these in their place.
OLD = {"old_key": "owner_key", "old_typed": "owner_typed", "old_vouched": "owner_vouched", "old_desk": "owner_desk"}
OLD_ORIGINS = sorted(OLD)
OLD_LABEL, OLD_HOLDOUT, OLD_CLI_VIA, OLD_MENU_SCOPE = "OLD: ", "old_lines", "oldcli", "old-menu"


class Generic013(DBCase):
    """0.1.4 gives the user's own origins, their chat-index label, the holdout's column and the `add` command's saves
    generic names, once per store; a 0.1.3 store's rows come through with their meaning kept."""

    def setUp(self):
        sha = lambda v: hashlib.sha256(v.encode()).hexdigest()  # noqa: E731
        for k, v in (("LEGACY_ORIGINS", {sha(o): n for o, n in OLD.items()}), ("LEGACY_LABEL", sha(OLD_LABEL)),
                     ("LEGACY_HOLDOUT", sha(OLD_HOLDOUT)), ("LEGACY_CLI_VIA", sha(OLD_CLI_VIA)),
                     ("LEGACY_MENU_SCOPE", sha(OLD_MENU_SCOPE))):
            p = mock.patch.object(mem, k, v)
            p.start()
            self.addCleanup(p.stop)
        super().setUp()

    def old_store(self):
        db = self.db
        with db:
            claimstore.migrate(db)
            for i, o in enumerate(OLD_ORIGINS + ["outside"], 1):
                db.execute("insert into memories(id, content, kind, hash, created, updated, origin) values "
                           "(?, ?, 'note', ?, 1, 1, ?)", (100 + i, f"Robin's note {i}", f"h{i}", o))
                db.execute("insert into claims(id, content, hash, origin, created, updated) values (?, ?, ?, ?, 1, 1)",
                           (200 + i, f"Robin's claim {i}", f"c{i}", o))
            text = "2026-10-01 [orchard]\n" + OLD_LABEL + "which port does staging use\nCLAUDE:\n> 6502"
            db.execute("insert into convo_chunks(id, session_id, ord, ts, text, hash) values (7, 's1', 0, 1, ?, ?)",
                       (text, mem._hash(text)))
            db.execute(f"create table holdout(sid text primary key, arm text not null, scope text not null, at real not "
                       f"null, {OLD_HOLDOUT} integer not null default 0)")
            db.execute(f"insert into holdout(sid, arm, scope, at, {OLD_HOLDOUT}) values ('k', 'memory', 'desk', "
                       f"1, 3)")
            db.execute("insert into saves(memory_id, at, via) values (101, 1, ?)", (OLD_CLI_VIA,))
            db.execute("delete from meta where key = 'names_generic_v1'")

    def test_a_013_store_gets_the_generic_names_once(self):
        self.old_store()
        self.db.close()
        self.db = mem.connect()  # what an update's `migrate` does
        origins = [r[0] for r in self.db.execute("select origin from memories where id > 100 order by id")]
        self.assertEqual(origins, [OLD[o] for o in OLD_ORIGINS] + ["outside"])
        claims = [r[0] for r in self.db.execute("select origin from claims where id > 200 order by id")]
        self.assertEqual(claims, origins)
        self.assertTrue(set(origins[:-1]) <= mem.OWNER_ORIGINS)  # still the user's own words
        text = self.db.execute("select text from convo_chunks where id = 7").fetchone()[0]
        self.assertIn("\nUSER: which port does staging use", text)
        self.assertEqual(self.db.execute("select hash from convo_chunks where id = 7").fetchone()[0], mem._hash(text))
        fts = lambda q: [r[0] for r in self.db.execute("select rowid from convo_fts where convo_fts match ?", (q,))]  # noqa: E731
        self.assertEqual((fts("user"), fts(OLD_LABEL.strip(": ").lower()), fts("staging")), ([7], [], [7]))
        self.assertEqual(self.db.execute("select owner_lines from holdout").fetchone()[0], 3)
        self.assertEqual(self.db.execute("select via from saves where memory_id = 101").fetchone()[0], "cli")
        self.assertEqual(mem.save_source("cli"), "Command line")
        self.db.execute("update memories set origin = ? where id = 101", (OLD_ORIGINS[0],))  # an old process, after
        self.db.commit()
        self.db.close()
        self.db = mem.connect()
        self.assertEqual(self.db.execute("select origin from memories where id = 101").fetchone()[0], OLD_ORIGINS[0])
        self.assertEqual(mem.least_trusted([OLD_ORIGINS[0]]), "unknown")  # an old name left over is not the user's

    def test_a_new_store_is_marked_done(self):
        self.assertTrue(self.db.execute("select 1 from meta where key = 'names_generic_v1'").fetchone())

    def test_add_saves_as_the_command_line(self):
        with mock.patch("sys.argv", ["mem.py", "add", "Orchard: staging listens on 6502", "--project", "orchard"]), \
                redirect_stdout(io.StringIO()):
            mem.main()
        self.assertEqual(self.db.execute("select via from saves order by id desc limit 1").fetchone()[0], "cli")


class Paths(DBCase):
    def test_dependents_read_paths_json(self):
        """A backup script, a host's menu-bar app or a health check read the store's paths from `paths --json`."""
        dst = self.tmp / "elsewhere"
        got = mem.paths_json(config.current().with_data(dst))
        self.assertEqual(got["data"], str(dst))
        self.assertEqual(got["db"], str(dst / "memory.db"))
        for k in ("sock", "connector", "models", "logs", "backup", "profile"):
            self.assertIn(k, got)

    def test_paths_cli(self):
        env = {**os.environ, "LFLABS_MEMORY_HOME": str(self.tmp)}
        out = subprocess.run([sys.executable, "-m", "lflabs_memory.cli", "paths", "--json"], capture_output=True,
                             text=True, env=env, timeout=120)
        got = json.loads(out.stdout)
        self.assertEqual(got["db"], str(self.tmp / "memory.db"))
        self.assertEqual(got["profile"], "generic")

    def test_version_and_schema_migrate_commands(self):
        env = {**os.environ, "LFLABS_MEMORY_HOME": str(self.tmp)}
        run = lambda *a: subprocess.run([sys.executable, "-m", "lflabs_memory.cli", *a], capture_output=True,  # noqa
                                        text=True, env=env, timeout=120)
        v = run("--version")
        self.assertEqual(v.returncode, 0)
        self.assertRegex(v.stdout.strip(), r"^\d+\.\d+\.\d+")
        m = run("migrate")
        self.assertEqual((m.returncode, m.stdout.strip()), (0, "schema up to date"))


if __name__ == "__main__":
    import unittest
    unittest.main()
