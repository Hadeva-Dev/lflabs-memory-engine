"""A first start imports Claude Code's notes on its own (the daemon's worker) while setup runs `import` too: the two
must never fail each other (VM run with 300 notes, 2026-11-10: UNIQUE constraint failed: memories.source, 5 of 5).

  .venv/bin/python -m unittest -q test_import_race
"""
import _env  # noqa: F401  (first: the test data folder and user)
import tempfile
import threading
import time
from pathlib import Path
from unittest import mock

from lflabs_memory import mem
from test_redact import DBCase, fake_embed


def notes(home, n):
    d = home / ".claude" / "projects" / "-orchard" / "memory"
    d.mkdir(parents=True)
    for i in range(n):
        (d / f"note{i}.md").write_text(f"---\ntype: project\n---\nThe orchard service number {i} listens on port {7000 + i}\n")
    return d


class Race(DBCase):
    def test_a_file_the_other_import_just_wrote_is_unchanged(self):
        src = "/Users/x/.claude/projects/-p/memory/a.md"
        other = mem.connect()
        self.addCleanup(other.close)
        state = {"done": False}

        def embed(texts, query=False):  # the other import lands while this one is embedding
            if not state["done"]:
                state["done"] = True
                with mock.patch.object(mem, "embed", fake_embed):
                    self.assertEqual(mem.remember(other, "Orchard deploys from main (as the file read a moment ago)",
                                                  "project", source=src)[1], "added")  # not a near copy: the insert meets it
            return fake_embed(texts, query)
        with mock.patch.object(mem, "embed", embed):
            mid, action = mem.remember(self.db, "Orchard deploys from main", "project", source=src)
        self.assertEqual(action, "unchanged")
        self.assertEqual(self.db.execute("select count(*) from memories where source = ?", (src,)).fetchone()[0], 1)
        self.assertEqual(self.db.execute("select id from memories where source = ?", (src,)).fetchone()[0], mid)

    def test_two_imports_at_once_both_finish(self):
        home = Path(tempfile.mkdtemp(prefix="race-home"))
        notes(home, 40)

        def slow(texts, query=False):
            time.sleep(0.005)  # the embedding's moment, where the two imports overlap
            return fake_embed(texts, query)
        results, errors = [], []

        def run():
            db = mem.connect()
            try:
                results.append(mem.import_files(db))
            except Exception as e:  # what setup's `import` reported to the user
                errors.append(e)
            finally:
                db.close()
        with mock.patch("pathlib.Path.home", return_value=home), mock.patch.object(mem, "embed", slow):
            threads = [threading.Thread(target=run) for _ in range(2)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(60)
        self.assertEqual(errors, [])
        self.assertEqual(len(results), 2)
        self.assertEqual(sum(r["added"] for r in results), 40)  # each note once, whichever import wrote it
        self.assertEqual(tuple(self.db.execute("select count(distinct source), count(*) from memories").fetchone()), (40, 40))

    def test_threads_making_the_source_secret_at_once_all_get_the_same_one(self):
        """The two imports above failed about 1 run in 15 (FileNotFoundError): both threads wrote the secret to one
        temp file name, and the second linked it after the first had removed it."""
        path = mem.paths().source_secret
        for _ in range(20):
            path.unlink(missing_ok=True)
            mem._SOURCE_SECRET.pop(str(path), None)
            start, got, errors = threading.Barrier(8), [], []

            def make():
                try:
                    start.wait(10)
                    got.append(mem._source_secret())
                except Exception as e:
                    errors.append(e)
            threads = [threading.Thread(target=make) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(30)
            self.assertEqual(errors, [])
            self.assertEqual(set(got), {path.read_bytes()})
            self.assertEqual(len(path.read_bytes()), 32)
            self.assertEqual([f.name for f in path.parent.glob(path.name + ".*")], [])  # no temp file left

    def test_a_refused_note_never_stops_the_import(self):
        home = Path(tempfile.mkdtemp(prefix="race-home"))
        d = notes(home, 2)
        (d / "secret.md").write_text("---\ntype: reference\n---\ntoken sk-" + "a" * 30 + "\n")
        with mock.patch("pathlib.Path.home", return_value=home):
            got = mem.import_files(self.db)
        self.assertEqual(got["added"], 2)

    def test_held_says_them_for_many(self):
        for i in range(2):
            mem.app_remember(self.db, f"The orchard nightly batch {i} runs at {i + 1}am", "project", via="cursor")
        self.assertIn("until you confirm them", mem.held_list(self.db))
        with self.db:
            self.db.execute("delete from memories where id = (select max(id) from memories)")
        self.assertIn("until you confirm it ", mem.held_list(self.db))


if __name__ == "__main__":
    import unittest
    unittest.main()
