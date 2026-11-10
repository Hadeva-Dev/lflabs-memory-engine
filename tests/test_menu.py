"""A host's menu-bar app's socket commands: status, the newest saves with private ones masked in the
daemon, and forget from the menu. Temp DB and temp socket only; no model calls.

  .venv/bin/python -m unittest -q test_menu
"""
import _env  # noqa: F401  (first: the test data folder and user)
import json
import os
import re
import socket
import socketserver
import stat
import threading
import time
from dataclasses import replace
from unittest import mock

from lflabs_memory import mem
from lflabs_memory import access
from test_access import Case
from test_forget import embed as same_embed
from test_redact import DBCase


def shown(out):
    """The rows the menu shows with their words: private rows carry a label only, never text."""
    for r in out["saves"]:
        if r.get("private"):
            assert "text" not in r and r["label"], r
    return [r for r in out["saves"] if not r.get("private")]


class Menu(DBCase):
    def setUp(self):
        super().setUp()
        for name in ("VOICE_SESSION", "TELEGRAM_SESSION", "MEETBRIEF_STATE", "PEOPLE_FILE"):
            p = mock.patch.object(mem, name, self.tmp / name)
            p.start()
            self.addCleanup(p.stop)
        for p in (mock.patch.object(mem, "SOCK", self.tmp / "m.sock"), mock.patch.object(mem, "activity", None),
                  mock.patch.dict(mem._saving, {"sid": None, "since": None}),
                  mock.patch.dict(mem._sense, {"model": None, "flags": None, "mtime": None, "id": None}),
                  mock.patch.object(mem, "peer_is_menu", lambda tok: True)):  # the signed app (Peer tests the check)
            p.start()
            self.addCleanup(p.stop)
        self.server = socketserver.UnixStreamServer(str(mem.SOCK), mem._handler(self.db))
        self.addCleanup(self.server.server_close)

    def ask(self, req):
        """One request over the temp socket, as the app sends it: the reply line, parsed."""
        t = threading.Thread(target=self.server.handle_request)
        t.start()
        with socket.socket(socket.AF_UNIX) as c:
            c.connect(str(mem.SOCK))
            c.sendall((json.dumps(req) + "\n").encode())
            raw = c.makefile().readline()
        t.join(5)
        return raw, json.loads(raw)

    def save(self, text, via="desk", **kw):
        """A save filed under lantern (by its project), as most work saves are; an unfiled one is private (below)."""
        mid, action = mem.remember(self.db, text, via=via, **{"project": "lantern", **kw})
        self.assertIn(action, ("added", "updated"))
        return mid

    # --- status ---

    def test_status_counts_saves_and_says_when_the_worker_is_saving(self):
        _, r = self.ask({"menu": "status"})
        self.assertEqual(r["result"], {"saving": False, "seq": 0, "at": None, "notice": False})
        self.save("Robin likes short replies in the terminal.")
        mem._saving.update(sid="s1", since=time.time())
        _, r = self.ask({"menu": "status"})
        self.assertTrue(r["result"]["saving"])
        self.assertEqual(r["result"]["seq"], 1)
        self.save("Robin ships on Fridays.", via="codex")
        mem._saving.update(sid=None, since=None)
        _, r = self.ask({"menu": "status"})
        self.assertEqual((r["result"]["saving"], r["result"]["seq"]), (False, 2))

    def test_a_remember_without_via_and_an_unchanged_one_are_not_saves(self):
        mem.remember(self.db, "Imported from a memory file.")  # import, restore, tests: no via, not logged
        mid = self.save("Robin likes short replies in the terminal.")
        mem.remember(self.db, "Robin likes short replies in the terminal.", via="desk")  # unchanged
        self.assertEqual(mem.menu_status(self.db)["seq"], 1)
        self.assertEqual([s["id"] for s in mem.recent_saves(self.db)["saves"]], [mid])

    def test_the_worker_marks_saving_while_it_extracts(self):
        seen = []

        class Stop(Exception):
            pass

        def extract(db, sid):
            seen.append(mem.menu_status(db)["saving"])
            return []
        with self.db:
            self.db.execute("insert or replace into meta values ('consolidated', ?)", (str(time.time()),))
        with mock.patch.object(mem, "connect", lambda: self.db), \
                mock.patch.object(mem, "pending_sessions", lambda db: ["s1"]), \
                mock.patch.object(mem, "extract", extract), \
                mock.patch.object(mem, "_adapt_tick", lambda db, log: None), \
                mock.patch.object(mem, "_chat_tick", lambda db, log: None), \
                mock.patch.object(mem, "sense_tick", lambda db, log: 0), \
                mock.patch.object(mem, "_claims_ready", lambda db: False), \
                mock.patch.object(mem, "people_due", lambda: False), \
                mock.patch.object(mem, "tag_people", lambda db: []), \
                mock.patch.object(mem, "outline_due", lambda: False), \
                mock.patch.object(mem, "_wake", mock.Mock(wait=mock.Mock(side_effect=Stop))):
            with self.assertRaises(Stop):
                mem._worker()
        self.assertEqual(seen, [True])
        self.assertFalse(mem.menu_status(self.db)["saving"])  # cleared after, even when nothing was saved

    # --- the list ---

    def test_saves_come_newest_first_with_their_source_and_one_row_per_memory(self):
        a = self.save("Robin likes short replies in the terminal.", via="desk")
        b = self.save("Marigold is the new name of the memory layer.", via="codex")
        c = self.save("Pear Lane pricing stays at nine dollars a month.", via="voice")
        real = mem.embed  # the fake embedding is random per text: the restatement gets a's vector, so it updates a
        same = lambda ts, query=False: real(["Robin likes short replies in the terminal." if x == "Robin likes very "  # noqa: E731
                                             "short replies." else x for x in ts], query)
        with mock.patch.object(mem, "embed", same):  # a restatement updates a: it moves to the top, once
            self.assertEqual(mem.remember(self.db, "Robin likes very short replies.", via="cursor"), (a, "updated"))
        out = mem.recent_saves(self.db)
        self.assertEqual([s["id"] for s in out["saves"]], [a, c, b])
        self.assertEqual([s["source"] for s in out["saves"]], ["Cursor", "Voice", "Codex"])
        self.assertEqual(out["saves"][0]["text"], "Robin likes very short replies.")
        self.assertEqual(out["private"], 0)
        self.assertEqual(out["seq"], 4)

    def test_only_the_last_hour_and_three_rows(self):
        ids = [self.save(f"Fact number {i} about the build pipeline, kept for later sessions {i * 7919}.")
               for i in range(5)]
        p1 = self.put("Robin's new prescription is 20 mg of the blue medication.")
        p2 = self.put("Tomas said, by text on 2026-10-19: the spare key is under the planter.")
        _, r = self.ask({"menu": "saves", "n": 50})  # asking for more changes nothing
        self.assertEqual([x["id"] for x in r["result"]["saves"]], [p2, p1, ids[4]])  # private rows: a label, no words
        shown(r["result"])
        self.assertEqual(r["result"]["private"], 2)  # every private save of the hour, not only among three
        with self.db:  # an hour and more ago: no longer listed, nor counted
            self.db.execute("update saves set at = at - 3601 where memory_id in (?, ?, ?)", (ids[4], ids[3], p2))
        out = mem.recent_saves(self.db)
        self.assertEqual(([x["id"] for x in out["saves"]], out["private"]), ([p1, ids[2], ids[1]], 1))
        with self.db:
            self.db.execute("update saves set at = at - 3601")
        out = mem.recent_saves(self.db)
        self.assertEqual((out["saves"], out["private"]), ([], 0))  # nothing new: the empty line
        self.assertEqual(self.ask({"menu": "forget", "id": ids[0]})[1]["result"], {"error": "no such save"})
        self.assertNotIn(p1, [x["id"] for x in out["saves"]])

    def test_a_forgotten_or_merged_memory_leaves_the_list(self):
        a = self.save("Robin likes short replies in the terminal.")
        b = self.save("Marigold is the new name of the memory layer.")
        mem.forget(self.db, a)
        with self.db:
            self.db.execute("update memories set status = 'merged', merged_into = ? where id = ?", (a, b))
        self.assertEqual(shown(mem.recent_saves(self.db)), [])

    def test_text_is_the_first_line_cut_short_and_redacted(self):
        self.save("## Build notes\nThe second line never reaches the menu.")
        self.save("Call the venue at (415) 555-1234 about the long-awaited spring tournament schedule and parking")
        out = mem.recent_saves(self.db)["saves"]
        self.assertEqual(out[1]["text"], "Build notes")
        t = out[0]["text"]
        self.assertLessEqual(len(t), mem.MENU_TEXT)
        self.assertTrue(t.endswith("…"))
        self.assertNotIn("555", t)
        self.assertIn("[phone]", t)

    def test_unknown_via_is_kept_as_a_plain_name(self):
        self.save("Robin likes short replies in the terminal.", via="Windsurf")
        self.save("Marigold is the new name of the memory layer.", via="no such client!")
        self.assertEqual([s["source"] for s in mem.recent_saves(self.db)["saves"]], ["Other", "Windsurf"])

    # --- private saves ---

    PRIVATE = {
        "health words": ("Robin's new prescription is 20 mg of the blue medication.", {}),
        "money words": ("Robin's salary at the lab is 4000 a month.", {}),
        "a quoted text": ("Tomas said, by text on 2026-10-19: the spare key is under the planter.", {}),
        "the email section": ("The landlord confirmed the inspection window.", {"section": "chats"}),
        "the people section": ("Maya's sister moved to Lisbon in June.", {"section": "people"}),
        "a second section": ("Maya runs the Tuesday standup.", {"section": "lantern", "section2": "people"}),
        "a meeting source": ("They agreed to ship the beta in November.", {"source": "meeting:abc#1"}),
        "someone else's words": ("The page said the event moved to Saturday.", {"origin": "outside"}),
        "open mic": ("Someone nearby said the code is 4471.", {"origin": "open_mic"}),
        "not filed yet": ("Priya is moving in with her boyfriend Dan in March.", {"project": None}),
        "personal as second": ("Robin's cousin Lena starts at the university in the fall.", {"section2": "personal"}),
        "about Robin himself": ("Robin is going through a breakup and has not told anyone.", {"kind": "user"}),
        "a section outside work": ("Robin keeps his passport in the blue folder.", {"section": "personal"}),
    }

    def put(self, text, section=None, section2=None, source=None, origin=None, **kw):
        mid = self.save(text, source=source, origin=origin, **kw)
        with self.db:
            self.db.execute("update memories set section = coalesce(?, section), section2 = coalesce(?, section2)"
                            " where id = ?", (section, section2, mid))
        return mid

    def test_private_saves_are_counted_never_shown(self):
        for i, (why, (text, kw)) in enumerate(self.PRIVATE.items()):
            with self.subTest(why):
                with self.db:
                    self.db.execute("delete from saves")
                mid = self.put(text, **kw)
                public = self.save(f"Build pipeline uses uv now, step {i}.")  # a public one after it
                raw, r = self.ask({"menu": "saves", "n": 5})
                out = r["result"]
                self.assertEqual(out["private"], 1)
                self.assertEqual([s["id"] for s in shown(out)], [public])  # no words of it either
                for word in re.findall(r"[A-Za-z]{5,}", text):  # no word of it leaves the daemon
                    self.assertNotIn(word.lower(), raw.lower(), f"{why}: {raw}")

    def test_sensitivity_labels_mark_a_save_private(self):
        mid = self.save("Robin's flight lands at 6 and Lena picks him up.")
        other = self.save("Marigold is the new name of the memory layer.")
        self.assertEqual(mem.recent_saves(self.db)["private"], 0)  # no label model on this Mac: the rules alone
        flags = self.tmp / "sense-flags.json"  # the daemon's withheld list (sense_tick writes it)
        flags.write_text(json.dumps({"model": "m1", "labelled": [mid, other], "withheld": [mid]}))
        self.assertEqual(mem.recent_saves(self.db)["private"], 1)
        flags.unlink()
        mem._sense.update(flags=None, mtime=None)
        (self.tmp / "models/sense").mkdir(parents=True)  # a label model: text waits for its labels
        (self.tmp / "models/sense/model_quantized.onnx").write_bytes(b"x")
        self.assertEqual(mem.recent_saves(self.db)["private"], 2)  # not loaded yet: nothing labelled
        model = mock.Mock(thresholds={"others_private": {"threshold": 0.5}, "health": {"threshold": 0.5}})
        mem._sense.update(model=model, id="m1")
        with self.db:  # a label the handoff keeps private (others_private), at the model's own threshold
            self.db.execute("insert into sense values (?, ?, 'm1', 0)", (mid, json.dumps({"others_private": 0.9})))
            self.db.execute("insert into sense values (?, ?, 'm1', 0)", (other, json.dumps({"others_private": 0.1})))
        out = mem.recent_saves(self.db)
        self.assertEqual((out["private"], [s["id"] for s in shown(out)]), (1, [other]))
        mem._sense.update(id="m2")  # labels from another model version do not count: both wait again
        self.assertEqual(mem.recent_saves(self.db)["private"], 2)

    def test_a_private_memory_stays_private_when_restated(self):
        with mock.patch.object(mem, "embed", same_embed):
            for i, (kw, again) in enumerate((
                    ({"kind": "user"}, {"kind": "note"}),                       # about Robin, restated as a note
                    ({"section": "chats"}, {}),                                 # an email's, restated in a session
                    ({"origin": "outside"}, {"origin": "unknown"}))):           # a page's, through the remember tool
                with self.subTest(kw=kw):
                    with self.db:
                        self.db.execute("delete from saves")
                        self.db.execute("update memories set status = 'retired'")
                    with mock.patch.object(mem, "DUP_SIM", 2.0):
                        mid = self.put(f"Robin's brother Tom lives in Oslo, fact {i} [same]", **kw)
                    self.assertEqual(mem.recent_saves(self.db)["private"], 1)
                    new, action = mem.remember(self.db, f"Robin's brother Tom lives in Oslo now, {i} [same]",
                                               via="desk", **{"project": "lantern", **again})
                    self.assertEqual((new, action), (mid, "updated"))
                    out = mem.recent_saves(self.db)
                    self.assertEqual((out["private"], shown(out)), (1, []))
                    self.assertTrue(next(r for r in out["saves"] if r["id"] == mid)["private"])  # its row: a label only

    def test_a_private_row_has_a_label_can_be_held_to_read_and_forgotten(self):
        public = self.save("Marigold is the new name of the memory layer.", via="codex")
        pills = self.put("Robin's new prescription is 20 mg of the blue medication.")
        text = self.put("Tomas said, by text on 2026-10-19: the spare key is under the planter.", section="people")
        out = mem.recent_saves(self.db)
        rows = {r["id"]: r for r in out["saves"]}
        self.assertEqual((rows[pills]["label"], rows[text]["label"]), ("Health or money", "About someone"))
        self.assertNotIn("text", rows[pills])
        _, r = self.ask({"menu": "reveal", "id": pills})  # held: its words, for the signed app only
        self.assertIn("prescription", r["result"]["text"])
        self.assertEqual(self.ask({"menu": "reveal", "id": public})[1]["result"], {"error": "no such save"})  # shown
        self.assertEqual(self.ask({"menu": "reveal", "id": 9999})[1]["result"], {"error": "no such save"})
        self.assertEqual(mem.menu_request(self.db, {"menu": "reveal", "id": pills}, lambda: False),
                         {"error": f"only the {mem.MENU_APP} app may do that"})
        self.assertEqual(self.ask({"menu": "forget", "id": pills})[1]["result"], {"forgotten": [pills]})
        self.assertNotIn(pills, [r["id"] for r in mem.recent_saves(self.db)["saves"]])
        with self.db:  # more than an hour ago: no longer on the menu, so neither readable nor forgettable from it
            self.db.execute("update saves set at = at - 3601 where memory_id = ?", (text,))
        self.assertEqual(self.ask({"menu": "reveal", "id": text})[1]["result"], {"error": "no such save"})
        self.assertEqual(self.ask({"menu": "forget", "id": text})[1]["result"], {"error": "no such save"})

    def test_private_is_kept_on_the_memory_for_good(self):
        mid = self.put("Robin's brother Tom lives in Oslo.", section="people")
        self.assertEqual(mem.recent_saves(self.db)["private"], 1)
        with self.db:  # the save log pruned, the memory moved to a shown section, no label model on this Mac
            self.db.execute("delete from saves")
            self.db.execute("update memories set section = 'lantern', section2 = null where id = ?", (mid,))
        self.save("Marigold is the new name of the memory layer.")
        mem.remember(self.db, "Robin's brother Tom lives in Oslo.", via="desk")  # unchanged: not a new save
        with self.db:
            self.db.execute("insert into saves(memory_id, at, via) values (?, ?, 'desk')", (mid, time.time()))
        self.assertEqual(mem.recent_saves(self.db)["private"], 1)  # menu_private remembered it
        self.assertTrue(mem.menu_is_private(self.db, mid))

    def test_a_private_label_from_any_model_sticks(self):
        mid = self.save("Robin's flight lands at 6 and Lena picks him up.")
        (self.tmp / "models/sense").mkdir(parents=True)
        (self.tmp / "models/sense/model_quantized.onnx").write_bytes(b"x")
        model = mock.Mock(thresholds={"others_private": {"threshold": 0.5}},
                          classify=lambda texts: [{"others_private": 0.9 if "Lena" in t else 0.0} for t in texts])
        with mock.patch.object(mem, "_sense_model", lambda: model), mock.patch.object(mem, "_write_sense_flags",
                                                                                       lambda db, m: None):
            mem._sense.update(id="m1")
            self.assertEqual(mem.sense_tick(self.db, log=lambda *a: None), 1)  # the labeller marks it
        mem._sense.update(model=None, id=None)
        (self.tmp / "models/sense/model_quantized.onnx").unlink()  # the model gone, or replaced: still private
        self.assertTrue(mem.menu_is_private(self.db, mid))
        self.assertEqual(mem.recent_saves(self.db)["private"], 1)

    def test_a_merge_of_a_private_memory_is_private(self):
        a = self.put("Robin's brother Tom lives in Oslo.", section="people")
        b = self.save("Tom, Robin's brother, lives in Oslo, Norway.")
        self.assertTrue(mem.menu_is_private(self.db, a))
        with self.db:
            self.db.execute("update memories set status = 'merged', merged_into = ? where id = ?", (b, a))
        self.assertTrue(mem.menu_is_private(self.db, b))
        self.assertEqual(shown(mem.recent_saves(self.db)), [])

    def test_forget_refuses_a_copy_that_was_ever_private(self):
        with mock.patch.object(mem, "embed", same_embed), mock.patch.object(mem, "DUP_SIM", 2.0):
            a = self.save("The standup moved to 10 [same]")
            b = self.save("The standup moved to 10 on Tuesdays [same]")
        with self.db:
            self.db.execute("insert into menu_private values (?, 0)", (b,))  # private once, looks public now
        with mock.patch.object(mem, "embed", same_embed):
            self.assertIn("error", self.ask({"menu": "forget", "id": a})[1]["result"])
        self.assertEqual(self.row(b)["status"], "active")

    def test_a_fresh_save_is_not_private_for_good(self):
        mid = mem.remember(self.db, "Marigold is the new name of the memory layer.", via="desk")[0]  # not filed yet
        self.assertEqual(mem.recent_saves(self.db)["private"], 1)
        with self.db:
            self.db.execute("update memories set section = 'lantern' where id = ?", (mid,))  # classify, a minute on
        self.assertEqual([s["id"] for s in mem.recent_saves(self.db)["saves"]], [mid])

    # --- forget ---

    def test_forget_retires_a_listed_memory_and_its_copies(self):
        a = self.save("Robin likes short replies in the terminal.")
        _, r = self.ask({"menu": "forget", "id": a})
        self.assertEqual(r["result"], {"forgotten": [a]})
        self.assertEqual(self.row(a)["status"], "retired")
        self.assertEqual(self.row(a)["content"], "")  # deleted for good: restore cannot bring it back
        self.assertFalse(mem.restore(self.db, a))
        _, r = self.ask({"menu": "forget", "id": a})
        self.assertEqual(r["result"], {"error": "no such save"})

    def test_forget_refuses_what_the_menu_does_not_show(self):
        old = mem.remember(self.db, "Imported from a memory file, never a save.")[0]
        private = self.put("Robin's new prescription is 20 mg of the blue medication.")
        for i in range(mem.MENU_SAVES + 2):
            self.save(f"Fact number {i} about the build pipeline, kept for later sessions {i * 7919}.")
        listed_once = self.db.execute("select memory_id from saves order by id limit 1 offset 1").fetchone()[0]
        first = self.db.execute("select memory_id from saves order by id limit 1").fetchone()[0]
        self.assertEqual(first, private)
        for mid in (old, private, listed_once, 999, "1", True, None, 1.0):  # listed once: now 5th, off the screen
            with self.subTest(mid=mid):
                _, r = self.ask({"menu": "forget", "id": mid})
                self.assertEqual(r["result"], {"error": "no such save"})
        self.assertEqual(self.row(old)["status"], "active")
        self.assertEqual(self.row(private)["status"], "active")

    def test_forget_refuses_when_a_private_copy_would_go_too(self):
        with mock.patch.object(mem, "embed", same_embed):  # [same]: one vector, so each is the other's exact copy
            with mock.patch.object(mem, "DUP_SIM", 2.0):  # saved as two memories all the same
                a = self.save("Maya runs the Tuesday standup [same]")
                b = self.put("Maya runs the Tuesday standup, her number is in my phone [same]", section="people")
            _, r = self.ask({"menu": "forget", "id": a})
        self.assertIn("error", r["result"])
        self.assertEqual((self.row(a)["status"], self.row(b)["status"]), ("active", "active"))

    def test_forget_refuses_a_copy_private_by_its_labels_only(self):
        (self.tmp / "models/sense").mkdir(parents=True)
        (self.tmp / "models/sense/model_quantized.onnx").write_bytes(b"x")
        mem._sense.update(model=mock.Mock(thresholds={"child_related": {"threshold": 0.5}}), id="m1")
        with mock.patch.object(mem, "embed", same_embed), mock.patch.object(mem, "DUP_SIM", 2.0):
            a = self.save("The school pickup moved to 3:15 [same]")
            b = self.save("The school pickup moved to 3:15 on Fridays [same]")
        with self.db:
            self.db.execute("insert into sense values (?, ?, 'm1', 0)", (a, json.dumps({"child_related": 0.1})))
            self.db.execute("insert into sense values (?, ?, 'm1', 0)", (b, json.dumps({"child_related": 0.9})))
        with mock.patch.object(mem, "embed", same_embed):
            self.assertIn("error", self.ask({"menu": "forget", "id": a})[1]["result"])
        self.assertEqual(self.row(b)["status"], "active")

    def test_forget_is_logged_as_owners_click(self):
        a = self.save("Robin likes short replies in the terminal.")
        log = mock.Mock()
        with mock.patch.object(mem, "activity", mock.Mock(log=log)):
            self.ask({"menu": "forget", "id": a})
        log.assert_called_once_with("memory", "action", f"Robin forgot memory #{a} from the {mem.MENU_APP} menu")

    def test_an_unknown_menu_request_changes_nothing(self):
        a = self.save("Robin likes short replies in the terminal.")
        _, r = self.ask({"menu": "delete", "id": a})
        self.assertEqual(r["result"], {"error": "unknown request"})
        self.assertEqual(self.row(a)["status"], "active")

    # --- where saves come from ---

    def test_extract_logs_where_each_save_came_from(self):
        cases = (("desk", "desk.jsonl", None), ("codex", "rollout-2026-10-20T10-00-00-x.jsonl", None),
                 ("voice", "voice.jsonl", "VOICE_SESSION"), ("telegram", "tg.jsonl", "TELEGRAM_SESSION"))
        for i, (via, name, marker) in enumerate(cases):
            with self.subTest(via=via):
                sid = f"s{i}"
                path = self.tmp / name
                path.write_text(json.dumps({"type": "user", "message": {"content": f"note {i}"}}) + "\n")
                if marker:
                    getattr(mem, marker).write_text(json.dumps({"id": sid}))
                mem.hook_activity({"session_id": sid, "transcript_path": str(path), "cwd": str(self.tmp)}, final=True)
                out = {"memories": [{"content": f"Saved fact number {i} for the menu test.", "kind": "note",
                                     "project": None, "from": [1]}]}
                got = {}
                real = mem.remember

                def spy(*a, **kw):
                    got["via"] = kw.get("via")
                    return real(*a, **kw)
                with mock.patch.object(mem, "llm_json", return_value=out), \
                        mock.patch.object(mem, "search", return_value=[]), \
                        mock.patch.object(mem, "remember", spy), \
                        mock.patch.object(mem, "_codex_as_claude", lambda p, lines: lines), \
                        mock.patch.dict(mem.os.environ, {"LFLABS_MEMORY_SCOPE": ""}):
                    mem.extract(self.db, sid)
                self.assertEqual(got.get("via"), via)


class Peer(DBCase):
    """Only the host's signed menu app may change anything; anyone may read the masked status, saves and notices."""

    def test_a_caller_that_is_not_the_app_cannot_forget_or_act(self):
        mid, _ = mem.remember(self.db, "Robin likes short replies in the terminal.", "note", "lantern", via="desk")
        sock = self.tmp / "p.sock"
        with mock.patch.object(mem, "SOCK", sock), \
                socketserver.UnixStreamServer(str(sock), mem._handler(self.db)) as server:
            def ask(req):
                t = threading.Thread(target=server.handle_request)
                t.start()
                with socket.socket(socket.AF_UNIX) as c:
                    c.connect(str(sock))
                    c.sendall((json.dumps(req) + "\n").encode())
                    out = json.loads(c.makefile().readline())["result"]
                t.join(5)
                return out
            for req in ({"menu": "forget", "id": mid}, {"menu": "notice", "id": 1, "action": "undo"},
                        {"menu": "notice", "id": 1, "action": "allow"}, {"menu": "notice", "id": 1, "action": "dismiss"},
                        {"menu": "seen"}, {"menu": "saves"}, {"menu": "status"}, {"menu": "notices"}):
                self.assertEqual(ask(req), {"error": f"only the {mem.MENU_APP} app may do that"})  # reads too: no one else needs them
        self.assertEqual(self.row(mid)["status"], "active")

    def test_the_check_is_by_audit_token_against_the_apps_signature(self):
        a, b = socket.socketpair(socket.AF_UNIX)
        with a, b:
            tok = a.getsockopt(0, 6, 32)
            self.assertEqual(len(tok), 32)
            self.assertFalse(mem.peer_is_menu(tok))  # this python, not the signed app
        self.assertFalse(mem.peer_is_menu(None))
        self.assertFalse(mem.peer_is_menu(b"\0" * 32))  # no such process
        def broken(tok, req):
            raise OSError
        with mock.patch.object(mem, "HOST", replace(mem.HOST, menu_peer_ok=broken)):
            self.assertFalse(mem.peer_is_menu(b"\1" * 32))  # fails closed
        with mock.patch.object(mem, "HOST", replace(mem.HOST, menu_peer_ok=None)):
            self.assertFalse(mem.peer_is_menu(b"\1" * 32))  # a plain install: no menu app to trust
        seen = []
        with mock.patch.object(mem, "HOST", replace(mem.HOST, menu_peer_ok=lambda tok: seen.append(tok) or True)), \
                mock.patch.dict(mem.__dict__, {"_menu_token": [None]}):
            self.assertTrue(mem.peer_is_menu(b"\1" * 32))
            self.assertTrue(mem.peer_is_menu(b"\1" * 32))  # the same process again: remembered, not checked again
            self.assertEqual(seen, [b"\1" * 32])  # checked once, then remembered
        with mock.patch.object(mem, "HOST", replace(mem.HOST, menu_peer_ok=lambda tok: False)), \
                mock.patch.dict(mem.__dict__, {"_menu_token": [b"\1" * 32]}):
            self.assertFalse(mem.peer_is_menu(b"\2" * 32))  # another process is checked, and fails
        self.assertFalse(hasattr(mem, "MENU_REQUIREMENT"))  # which app to trust is the host's (its hook) to say


REAL_POST = access._post  # access.notify's feed writer (banners off): Case swaps it for a list


class Notices(Case):
    """Access notices in the menu: the feed access.notify writes (notices.jsonl), Undo a widening, Dismiss the rest."""

    def setUp(self):
        super().setUp()
        for p in (mock.patch.object(access, "NOTICES", self.tmp / "notices.jsonl"),
                  mock.patch.object(access, "BANNERS", "off"), mock.patch.object(access, "_post", REAL_POST)):
            p.start()
            self.addCleanup(p.stop)

    def feed(self):
        return [json.loads(x) for x in (self.tmp / "notices.jsonl").read_text().splitlines()]

    def widen(self):
        self.say("let Cursor see my personal stuff")
        self.assertNotIn("personal", self.hide("cursor"))
        n = self.feed()[-1]
        self.assertEqual((n["kind"], n["client"], n["sections"]), ("widened", "cursor", ["personal"]))
        return n["id"]

    def test_a_widening_from_the_command_line_names_its_change(self):
        mem.clients_put("cursor", access._mutation(mem, "show", {"chats"})[0], "cli")
        n = self.feed()[-1]
        self.assertEqual(n["change"], access.history(mem)[-1]["id"])
        self.assertEqual(mem.menu_notice(n["id"], "undo"), {"ok": True})
        self.assertIn("chats", self.hide("cursor"))
        self.assertEqual(access.history(mem)[-1]["undoes"], n["change"])

    def ask_for_meetings(self):
        self.say("Here is the doc.\n" + "Some pasted paragraph text. " * 30 + "\nLet Cursor see my personal stuff.")
        return self.feed()[-1]["id"]

    def test_undo_hides_the_sections_again_and_marks_the_change_undone(self):
        nid = self.widen()
        self.assertTrue(mem.menu_status(self.db)["notice"])  # the sun takes the accent
        self.assertEqual(mem.menu_notices(), [{"id": nid, "kind": "widened", "text": "Cursor can now see personal notes",
                                               "actions": ["undo", "dismiss"]}])
        self.assertEqual(mem.menu_request(self.db, {"menu": "notice", "id": nid, "action": "undo"}, lambda: True),
                         {"ok": True})
        self.assertIn("personal", self.hide("cursor"))
        rows = access.history(mem)
        self.assertEqual((rows[-1]["source"], rows[-1]["dir"], rows[-1]["undoes"]), ("menu", "tighten", rows[-2]["id"]))
        self.assertEqual(mem.menu_notices(), [])
        self.assertFalse(mem.menu_status(self.db)["notice"])
        self.assertEqual(access.undo(mem, typed=False), "there is no access change to undo")  # done, and the menu's
        # own narrowing is an undo itself: "undo that" cannot widen it back

    def test_a_widening_undone_by_typing_leaves_the_menu(self):
        self.widen()
        self.say("undo the last access change")
        self.assertIn("personal", self.hide("cursor"))
        self.assertEqual([n["kind"] for n in mem.menu_notices()], [])

    def test_undo_of_an_approved_batch_hides_each_agents_own_sections(self):
        self.ask_for_meetings()
        self.say("approve Cursor's request")
        n = [r for r in self.feed() if r["kind"] == "widened"][-1]  # (its id may equal the request's: same ms)
        self.assertEqual((n["client"], n.get("change")), ("cursor", None))
        self.assertNotIn("personal", self.hide("cursor"))
        nid = next(x["id"] for x in mem.menu_notices() if x["kind"] == "widened")  # its own number in the menu
        self.assertEqual(mem.menu_notice(nid, "undo"), {"ok": True})
        self.assertIn("personal", self.hide("cursor"))

    def test_notices_of_the_same_millisecond_get_their_own_numbers(self):
        t = time.time()
        (self.tmp / "notices.jsonl").write_text("".join(json.dumps(r) + "\n" for r in (
            {"id": 50, "at": t, "kind": "narrowed", "text": "Codex hid email from Cursor"},
            {"id": 50, "at": t, "kind": "notice", "text": "Something else"})))
        self.assertEqual([(n["id"], n["kind"]) for n in mem.menu_notices()], [(51, "notice"), (50, "narrowed")])
        self.assertEqual(mem.menu_notice(51, "dismiss"), {"ok": True})
        self.assertEqual([n["id"] for n in mem.menu_notices()], [50])

    def test_undo_never_widens_and_needs_a_real_agent(self):
        before = self.hide("cursor")  # work only: meetings hidden already
        with open(self.tmp / "notices.jsonl", "a") as f:
            f.write(json.dumps({"id": 5, "at": time.time(), "kind": "widened", "text": "Cursor can now see personal notes",
                                "client": "cursor", "sections": ["personal"]}) + "\n")
            f.write(json.dumps({"id": 6, "at": time.time(), "kind": "widened", "text": "x", "client": "../x",
                                "sections": ["*"]}) + "\n")
        self.assertEqual(mem.menu_notice(5, "undo"), {"ok": True})  # already hidden: nothing left to undo
        self.assertEqual(self.hide("cursor"), before)
        self.assertEqual(mem.menu_notice(6, "undo"), {"error": "nothing to undo"})

    def test_opening_the_menu_clears_the_accent_and_keeps_the_line(self):
        nid = self.widen()
        self.assertEqual(mem.menu_request(self.db, {"menu": "seen"}, lambda: True), {"ok": True})
        self.assertFalse(mem.menu_status(self.db)["notice"])
        self.assertEqual([n["id"] for n in mem.menu_notices()], [nid])
        self.assertEqual(mem.menu_notice(nid, "dismiss"), {"ok": True})
        self.assertEqual(mem.menu_notices(), [])
        self.assertNotIn("personal", self.hide("cursor"))  # dismissing a widening keeps it
        self.assertEqual(stat.S_IMODE(os.stat(self.tmp / "menu-state.json").st_mode), 0o600)

    def test_a_request_can_be_dismissed(self):
        nid = self.ask_for_meetings()
        self.assertFalse(mem.menu_status(self.db)["notice"])  # only a widening that happened takes the accent
        shown = mem.menu_notices()
        self.assertEqual([(n["id"], n["kind"], n["actions"]) for n in shown], [(nid, "request", ["approve", "dismiss"])])
        self.assertTrue(shown[0]["text"].endswith("Cursor to see personal notes"), shown[0]["text"])
        self.assertEqual((shown[0]["target"], shown[0]["what"]), ("cursor", ["personal"]))
        self.assertEqual(mem.menu_notice(nid, "allow"), {"error": "no such notice"})
        self.assertEqual(mem.menu_notice(nid, "dismiss"), {"ok": True})
        self.assertEqual(access.requests(mem), [])  # dropped from the waiting list
        self.assertIsNone(access.approve(mem, "cursor"))
        self.assertIn("personal", self.hide("cursor"))

    def approve(self, nid, now, **over):
        row = next(n for n in mem.menu_notices(now) if n["id"] == nid)
        req = {"menu": "notice", "id": nid, "action": "approve", "target": row["target"], "what": row["what"],
               "level": row["level"], **over}
        with mock.patch.object(mem.time, "time", lambda: now):
            return mem.menu_request(self.db, req, lambda: True)

    def test_one_click_approve_gives_exactly_the_row_and_logs_the_click(self):
        nid = self.ask_for_meetings()
        t = time.time()
        mem.menu_notices(t)  # the row is shown
        self.assertEqual(self.approve(nid, t + 0.7), {"ok": True})
        self.assertNotIn("personal", self.hide("cursor"))
        self.assertEqual(access.history(mem)[-1]["source"], "menu-click")
        self.assertEqual(access.requests(mem), [])
        self.assertEqual([n["kind"] for n in mem.menu_notices()], ["widened"])  # "Cursor can now see personal notes · Undo"

    def test_approve_refuses_a_too_soon_click_a_changed_row_and_a_stranger(self):
        nid = self.ask_for_meetings()
        t = time.time()
        mem.menu_notices(t)
        self.assertEqual(self.approve(nid, t + 0.2), {"error": "too soon"})  # nothing slides under a click
        self.assertEqual(self.approve(nid, t + 0.7, what=["personal", "people"]), {"error": "that request changed: look again"})
        self.assertEqual(self.approve(nid, t + 0.7, target="codex"), {"error": "that request changed: look again"})
        self.assertEqual(self.approve(nid, t + 0.7, what="personal"), {"error": "that request changed: look again"})
        self.assertEqual(mem.menu_request(self.db, {"menu": "notice", "id": nid, "action": "approve", "target": "cursor",
                                                    "what": ["personal"], "level": None}, lambda: False),
                         {"error": f"only the {mem.MENU_APP} app may do that"})
        self.assertIn("personal", self.hide("cursor"))  # none of those gave anything
        self.assertEqual(len(access.requests(mem)), 1)

    def test_approve_of_a_request_answered_or_expired_elsewhere_does_nothing(self):
        nid = self.ask_for_meetings()
        t = time.time()
        row = next(n for n in mem.menu_notices(t) if n["id"] == nid)
        access._write_requests(mem, [])  # dismissed or approved elsewhere meanwhile
        req = {"menu": "notice", "id": nid, "action": "approve", "target": row["target"], "what": row["what"],
               "level": row["level"]}
        with mock.patch.object(mem.time, "time", lambda: t + 1):
            self.assertIn("error", mem.menu_request(self.db, req, lambda: True))
        self.assertIn("personal", self.hide("cursor"))

    def test_a_request_answered_elsewhere_leaves_the_menu(self):
        self.ask_for_meetings()
        self.assertEqual(len(mem.menu_notices()), 1)
        self.say("approve Cursor's request")
        self.assertEqual([n["kind"] for n in mem.menu_notices()], ["widened"])  # the request went; its widening came

    def test_old_odd_and_many_notices(self):
        rows = [{"id": 1, "at": time.time() - mem.NOTICE_TTL - 5, "kind": "widened", "text": "Cursor can now see past chats",
                 "client": "cursor"},  # a day old: gone
                {"id": True, "at": time.time(), "kind": "widened", "text": "x"},
                {"id": 2, "at": time.time(), "kind": "deleted", "text": "x"},
                {"id": 3, "at": "now", "kind": "notice", "text": "x"},
                {"id": 4, "at": time.time(), "kind": "notice", "text": 5}]
        rows += [{"id": 10 + i, "at": time.time(), "kind": "narrowed", "text": f"Codex hid email from Cursor <b>{i}</b>"}
                 for i in range(5)]
        (self.tmp / "notices.jsonl").write_text("not json\n" + "".join(json.dumps(r) + "\n" for r in rows))
        out = mem.menu_notices()
        self.assertEqual([n["id"] for n in out], [14, 13, 12])
        self.assertNotIn("<", out[0]["text"])
        self.assertEqual(out[0]["actions"], ["dismiss"])
        self.assertFalse(mem.menu_status(self.db)["notice"])


if __name__ == "__main__":
    import unittest
    unittest.main()
