"""The memory daemon's nightly rebuild of the host's people index (a host's people code, through
ProfileHooks.refresh_people): when it runs, that it runs it, and that a plain install has none.

  .venv/bin/python -m unittest -q test_people_refresh
"""
import _env  # noqa: F401  (first: the test data folder and user)
import os
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from lflabs_memory import mem


def at(hour, minute):
    return time.mktime((2026, 10, 8, hour, minute, 0, 0, 0, -1))


class Due(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.file = Path(self.tmp.name) / "people.json"
        self.built = []
        build = lambda: self.built.append(1) or SimpleNamespace(returncode=0, stdout="12 people: ...\n", stderr="")  # noqa: E731
        for p in (mock.patch.object(mem, "PEOPLE_FILE", self.file), mock.patch.object(mem, "_people_tried", [0.0]),
                  mock.patch.object(mem, "HOST", replace(mem.HOST, refresh_people=build)),
                  mock.patch.object(mem, "_on_ac", lambda: self.ac)):
            p.start()
            self.addCleanup(p.stop)
        self.ac = True

    def tearDown(self):
        self.tmp.cleanup()

    def test_missing_index_is_built_at_night_only(self):
        self.assertTrue(mem.people_due(at(2, 0)))
        self.assertTrue(mem.people_due(at(23, 0)))
        self.assertFalse(mem.people_due(at(14, 0)))

    def test_a_fresh_index_waits_a_day(self):
        self.file.write_text("{}")
        os.utime(self.file, (at(2, 0), at(2, 0)))
        self.assertFalse(mem.people_due(at(5, 0)))          # later the same night
        self.assertTrue(mem.people_due(at(2, 0) + 86400))   # the next night

    def test_never_on_battery_and_not_twice_an_hour(self):
        self.ac = False
        self.assertFalse(mem.people_due(at(2, 0)))
        self.ac = True
        mem._people_tried[0] = at(1, 30)
        self.assertFalse(mem.people_due(at(2, 0)))
        self.assertTrue(mem.people_due(at(2, 31)))

    def test_refresh_runs_the_builder_and_logs_counts(self):
        logs = []
        mem.refresh_people(logs.append)
        self.assertEqual(self.built, [1])
        self.assertEqual(logs, ["people index: 12 people: ..."])
        self.assertGreater(mem._people_tried[0], 0)

    def test_a_plain_install_has_no_index_to_build(self):
        with mock.patch.object(mem, "HOST", replace(mem.HOST, refresh_people=None)):
            self.assertFalse(mem.people_due(at(2, 0)))
        with mock.patch.object(mem, "PEOPLE_FILE", None):
            self.assertFalse(mem.people_due(at(2, 0)))
            self.assertEqual(mem.people_links(), set())


if __name__ == "__main__":
    unittest.main()
