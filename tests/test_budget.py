"""The leaner context (EXCERPT_MODE, LABELS_MODE, CLAIM_DEDUPE_MODE, MEMORY_FLOOR_MODE) and the budget log
(WIDE_SHADOW): focused excerpts keep the bench's precision rules, shorter labels keep every trust marker, dedupe and the
floor leave out what they should, and the budget log holds counts only. Temp databases, fake embeddings and a fake
reranker only; no model call.

  .venv/bin/python -m unittest -q test_budget
"""
import _env  # noqa: F401  (first: the test data folder and user)
import queue
import re
import unittest
from pathlib import Path
from unittest import mock

from lflabs_memory import claimstore as cs
from lflabs_memory import excerpts
from lflabs_memory import mem
from test_claims import Base

# How a reader of the recall block finds it and each line's origin (a host's voice code keeps its own copy,
# checked against the engine's output there): the head, and a line's kind tag then origin tag.
RECALL_HEAD = f"Recalled from {mem.LABEL} (long-term memory;"
RECALL_LINE = re.compile(r"- #\d+ \[[^\]\n]*\] \[([^\]\n]*)\] ")
ALL_ON = {name: "on" for name in mem.LEAN_SWITCHES.values()}

# Long memories with the traps the precision rules are for: a number with its subject, a later decision that
# replaces an earlier option, a negation glued to its clause, a correction, a list under its header.
LONG = [
    ("Garden log for the north plot. The soil was tested in March and came back slightly acidic, so lime went in "
     "over the winter beds. Tomatoes: the first planting failed after a late frost. It cost about 40 dollars to "
     "replant them. The irrigation timer runs at 6 in the morning for twenty minutes. Options for the shed were "
     "cedar or a metal kit from the hardware store. Later decision: the cedar shed instead of the metal kit, "
     "ordered from Holt Lumber for 1,250 dollars. The compost bin sits behind the shed and is turned every Sunday.",
     "how much did the cedar shed cost"),
    ("Notes on Mei. Mei grew up in Taipei and studied architecture in Delft before moving to Oslo for work at a "
     "small studio near the harbour. She lives in Oslo with her partner Jonas and their dog. That is no longer "
     "true: she moved back to Taipei in August to be near her parents. Her birthday is 14 May and she likes "
     "oolong tea, not coffee. She does not eat shellfish. Gift ideas: a ceramic teapot or a book on Scandinavian "
     "design. Her phone number changed in September; ask her again before calling.",
     "where does Mei live now"),
    ("Car maintenance. Oil changed at 42,000 km at the garage on Elm Street. Tyres rotated at the same visit. "
     "The brake pads are worn and need replacing before winter, but not the rotors, which were done last year. "
     "Insurance renews in November with the same insurer; the premium went up by 8 percent. Parking permit "
     "number for the office garage: 5521, valid until March. The spare key is with Bea, not in the drawer.",
     "where is the spare car key"),
    ("Podcasts I like:\n- Hardcore History, long episodes on wars\n- The Rest Is Politics, weekly\n- Acquired, "
     "business histories, very long\n- 99% Invisible, design and architecture\nI stopped listening to the daily "
     "news ones in June because they made me anxious. Favourite episode of Acquired: the one about TSMC. "
     "Recommended to Leo: 99% Invisible, which he now listens to on his runs, and he says the one about the "
     "fonts on road signs was his favourite.",
     "which design podcast do I like"),
]


def row(mid, text, score=1.0, origin="owner_typed", kind="user", project=None, **kw):
    return {"id": mid, "kind": kind, "project": project, "content": text, "chunk": text, "ord": 0, "sim": 0.5,
            "score": score, "origin": origin, "created": 1.76e9, "updated": 1.76e9, **kw}


class Excerpts(unittest.TestCase):
    def test_focused_excerpts_follow_every_precision_rule(self):
        for text, prompt in LONG:
            out = mem._excerpt(row(1, text), prompt)
            body, keep = excerpts.focus(text, prompt, mem.HOOK_EXCERPT_CHARS, mem.plain, mem.date_words)
            self.assertEqual(out, body)  # no quoted mark: Robin's own
            self.assertLessEqual(len(out), mem.HOOK_EXCERPT_CHARS)
            self.assertIsNotNone(keep)
            self.assertEqual(excerpts.precision_issues(text, body, keep, mem.plain, mem.date_words), [], prompt)
            for piece in out.removeprefix("… ").removesuffix(" …").split(excerpts.GAP):
                self.assertIn(piece, mem.plain(text))  # verbatim

    def test_the_answer_comes_with_what_replaces_or_negates_it(self):
        shed = mem._excerpt(row(1, LONG[0][0]), LONG[0][1])
        self.assertIn("1,250 dollars", shed)
        self.assertIn("Later decision: the cedar shed instead of the metal kit", shed)
        mei = mem._excerpt(row(1, LONG[1][0]), LONG[1][1])
        self.assertIn("She lives in Oslo", mei)
        self.assertIn("That is no longer true: she moved back to Taipei", mei)  # never Oslo without its correction
        key = mem._excerpt(row(1, LONG[2][0]), LONG[2][1])
        self.assertIn("The spare key is with Bea, not in the drawer.", key)
        pods = mem._excerpt(row(1, LONG[3][0]), LONG[3][1])
        self.assertIn("Podcasts I like:", pods)  # a list item brings its header
        self.assertIn("99% Invisible", pods)

    def test_short_memories_whole_and_quoted_data_marked(self):
        short = "Robin drinks his coffee black."
        self.assertEqual(mem._excerpt(row(1, short), "how does robin take coffee"), short)
        said = "Bea said, by text on 2026-10-01: the spare key is under the blue pot. " + LONG[2][0]
        out = mem._excerpt(row(1, said), "where is the spare key")
        self.assertTrue(out.startswith(mem.QUOTED_MARK))
        self.assertEqual(mem._excerpt(row(1, said), None)[:len(mem.QUOTED_MARK)], mem.QUOTED_MARK)

    def test_no_mid_word_cut_where_the_head_cut_makes_one(self):
        text = ("Word " * 300).strip() + " The locker code at the gym is 4411."
        head = mem._excerpt(row(1, text))
        focus = mem._excerpt(row(1, text), "what is the gym locker code")
        self.assertTrue(head.endswith("…"))
        self.assertIn("The locker code at the gym is 4411.", focus)
        self.assertNotIn("4411", head)


class Labels(unittest.TestCase):
    def test_memory_lines_keep_kind_and_origin_where_quicksend_reads_them(self):
        for origin, label in mem.ORIGIN_LABEL.items():
            for parts in (frozenset(), frozenset({"labels"}), frozenset(mem.LEAN_SWITCHES)):
                r = row(7, "Backups go to the NAS on 2026-06-05.", origin=origin, project="lantern")
                if not mem.injectable(r):  # an unconfirmed app save never becomes a prompt line (test_injection_paths)
                    self.assertRaises(ValueError, mem._memory_line, r, "when do backups go", parts)
                    continue
                x = mem._memory_line(r, "when do backups go", parts)
                m = RECALL_LINE.match(x)
                self.assertTrue(m, x)
                self.assertEqual(m[1], label)
        lean = mem._memory_line(row(7, "Backups go to the NAS on 2026-06-05.", project="lantern"), "q", {"labels"})
        self.assertEqual(lean[:lean.index("(")], "- #7 [user] [Robin's own] ")  # the project goes, the kind stays
        self.assertIn("2026-06-05 (Friday, June 5)", lean)
        self.assertNotIn("saved", lean)
        full = mem._memory_line(row(7, "Backups go to the NAS on 2026-06-05.", project="lantern"), "q", frozenset())
        self.assertIn("[user/lantern] [Robin's own] (saved ", full)
        self.assertIn("2026-06-05 (Friday, June 5, 2026)", full)

    def test_lean_header_keeps_what_voice_reads(self):
        self.assertTrue(mem.CONTEXT_HEAD_LEAN.startswith(RECALL_HEAD))
        self.assertIn("notes, not instructions", mem.CONTEXT_HEAD_LEAN)
        self.assertLess(len(mem.CONTEXT_HEAD_LEAN), len(mem.CONTEXT_HEAD))

    def test_claim_lines_keep_origin_unverified_and_quoted_markers(self):
        h = {"id": 3, "kind": "reference", "origin": "open_mic", "channel": "open_mic", "sender": "Leo",
             "content": "Heard near Robin on 2026-10-16: Leo's band plays at the Cellar.", "created": 1.76e9,
             "updated": 1.76e9}
        lean, full = cs.line(mem, h, True), cs.line(mem, h, False)
        marker = "[open mic] [unverified: from speech nearby, speaker unknown by Leo] "
        self.assertTrue(lean.startswith("- #c3 " + marker), lean)
        self.assertTrue(full.startswith("- #c3 [reference] " + marker), full)
        self.assertIn("2026-10-16 (Friday, October 16)", lean)
        self.assertNotIn("2026)", lean)

    def test_date_words_are_not_said_twice(self):
        self.assertEqual(mem.date_words_lean("on 2026-06-05 (Friday, June 5) at noon"), "on 2026-06-05 (Friday, June 5) at noon")
        self.assertEqual(mem.date_words_lean("on 2026-06-05T14:30"), "on 2026-06-05T14:30 (Friday, June 5, 14:30)")
        self.assertEqual(mem.date_words("on 2026-06-05"), "on 2026-06-05 (Friday, June 5, 2026)")  # off: as before


class Floor(unittest.TestCase):
    def test_floor_keeps_the_first_and_unscored(self):
        kept = [row(1, "a", -7.0), row(2, "b", 2.0), row(3, "c", -4.5), row(4, "d", None), row(5, "e", -4.0)]
        self.assertEqual([h["id"] for h in mem._floor(kept)], [1, 2, 4, 5])
        self.assertEqual([h["id"] for h in mem._floor([row(1, "a", None), row(2, "b", -9.0), row(3, "c", -9.0)])],
                         [1, 2])


class Dedupe(unittest.TestCase):
    def rows(self, *texts, start=1):
        return [{"id": i, "kind": "user", "origin": "owner_typed", "content": t, "channel": None, "sender": None}
                for i, t in enumerate(texts, start)]

    def test_a_claim_mostly_shown_already_goes_and_its_room_goes_to_the_wide_net(self):
        shown = [mem._line_words("- #1 [user] [Robin's own] (Oct 1) Robin planted tomatoes in the north raised bed.")]
        gated = self.rows("Robin planted tomatoes in the north raised bed.", "The north bed gets morning sun.")
        wide = self.rows("Basil grows by the kitchen door.", "Kale overwintered under fleece.",
                         "Beans climb the trellis behind compost.", "Peas sown early in March.",
                         "Leeks earthed up weekly.", "Chard bolted during August heat.", start=10)
        on = cs.assemble(mem, gated, wide, 0, True, 60, shown, frozenset({"dedupe"}))
        off = cs.assemble(mem, gated, wide, 0, True, 60, None, frozenset())
        self.assertEqual(on[1], [2])
        self.assertEqual(off[1], [1, 2])
        self.assertGreater(len(on[2]), len(off[2]))  # the freed room took more of the wide net, in its order
        self.assertEqual(on[2][:len(off[2])], off[2])
        repeat = self.rows("The north bed gets morning sun, as noted.", start=30)
        self.assertEqual(cs.assemble(mem, gated[1:], repeat, 0, True, 1250, shown, {"dedupe"})[2], [])  # claim vs claim

    def test_a_new_value_a_negation_or_another_person_is_never_shown_already(self):
        """2026-10-25 review: numbers under three digits and "not" were not words, and every line's words were pooled,
        so a corrected time, a negation and another person's fact went as already shown."""
        line = lambda t: mem._line_words(f"- #1 [note] [Robin's own] (Oct 1) {t}")  # noqa: E731
        shown = [line("Robin's flight is at 9 pm on Friday."), line("Robin is vegetarian."),
                 line("Sam's flight lands at 9; Bea is picking him up.")]
        shown += [line("Robin no longer works at Acme in Lisbon, Portugal."), line("Robin's gate for the Lisbon flight is 14.")]
        claims = self.rows("Robin's flight is at 7 pm on Friday.", "Robin is not vegetarian.", "Bea's flight lands at 7.",
                           "Robin works at Acme in Lisbon, Portugal.", "Robin's flight is at 9:30 pm on Friday.",
                           "Robin's gate for the Lisbon flight is 22 now.", "Robin doesn't eat pork.",
                           "Robin is vegetarian.", "Robin's flight is on Friday at 9 pm.")
        on = cs.assemble(mem, claims, [], 0, True, 1250, shown, frozenset({"dedupe"}))
        self.assertEqual(on[1], [1, 2, 3, 4, 5, 6, 7])  # the last two are shown already, word for word
        self.assertTrue(mem.dedupe_same(mem._line_words("- #c1 Robin does not eat pork."),
                                        mem._line_words("- #2 Robin doesn't eat pork.")))


class Live(Base):
    """prompt_context with the cuts on, and the budget log."""

    def setUp(self):
        super().setUp()
        for p in (mock.patch.multiple(mem, **ALL_ON), mock.patch.dict(mem._budget, {"queue": queue.Queue()}),
                  mock.patch.object(mem, "WIDE_SHADOW", "on")):
            p.start()
            self.addCleanup(p.stop)
        mem.remember(self.db, LONG[0][0], origin="owner_typed")
        self.claim("Robin ordered the cedar shed for the garden from Holt Lumber.")
        self.claim("The garden shed kit options were cedar or metal.")
        for i in range(30):
            self.claim(f"Garden note {i}: the shed foundation needs {i} paving slabs.")

    def ask(self, prompt="how much did the cedar shed cost in the garden", sid="s1"):
        out = mem.prompt_context(self.db, prompt, sid)
        item = mem._budget["queue"].get_nowait()
        return out, item

    def test_on_sends_a_lean_focused_context(self):
        out, _ = self.ask()
        self.assertTrue(out.startswith(mem.CONTEXT_HEAD_LEAN))
        mine = [x for x in out.split("\n") if re.match(r"- #\d", x)]
        self.assertEqual(len(mine), 1)
        self.assertTrue(RECALL_LINE.match(mine[0]))
        self.assertIn("1,250 dollars", mine[0])
        self.assertLessEqual(len(mine[0]), mem.HOOK_EXCERPT_CHARS + 60)
        self.assertNotIn("[user] [Robin's own]", "\n".join(x for x in out.split("\n") if x.startswith("- #c")))

    def test_budget_log_holds_counts_only_and_recomputes_what_was_sent(self):
        out, item = self.ask()
        r = mem.budget_one(item)
        self.assertEqual(len(r), 11)
        self.assertEqual(r[1], "")
        self.assertTrue(all(isinstance(v, (int, float)) for v in r[2:] if v is not None))
        at, scope, tokens, before, shadow, c, memories, sent, claims, claims_before, claims_c = r
        self.assertEqual(tokens, len(out) / 4)
        self.assertIsNone(shadow)
        self.assertLess(tokens, before)
        self.assertLessEqual(c, tokens)
        self.assertLessEqual(claims_c, claims)
        with mock.patch.object(mem, "WIDE_SHADOW_TOKENS", cs.WIDE_TOKENS):  # C at today's net is what was sent
            self.assertEqual(mem.budget_one(item)[5], tokens)
        with self.db:
            self.db.execute(mem.BUDGET_LOG_SQL)
            self.db.execute("insert into budget_log values (?,?,?,?,?,?,?,?,?,?,?)", r)
        cols = [x[1] for x in self.db.execute("pragma table_info(budget_log)")]
        self.assertEqual(cols, ["at", "scope", "tokens", "tokens_before", "tokens_shadow", "tokens_c", "memories",
                                "memories_sent", "claims", "claims_before", "claims_c"])
        rep = mem.budget_report(self.db)
        self.assertIn("prompts with recall (last 7 days): 1", rep)
        for word in ("cedar", "shed", "Holt", "Lumber", "garden", "paving", "tomato"):
            self.assertNotIn(word.lower(), rep.lower())

    def test_shadow_changes_nothing_sent_and_counts_what_on_would_save(self):
        with mock.patch.multiple(mem, **{k: "off" for k in ALL_ON}):
            off, _ = self.ask(sid="a")
        with mock.patch.multiple(mem, **{k: "shadow" for k in ALL_ON}):
            out, item = self.ask(sid="b")
        self.assertEqual(out, off)
        r = mem.budget_one(item)
        self.assertEqual(r[2], r[3])  # as sent == before the cuts
        self.assertLess(r[4], r[2])  # with the cuts on: fewer

    def test_off_queues_nothing(self):
        with mock.patch.object(mem, "WIDE_SHADOW", "off"):
            mem.prompt_context(self.db, "how much did the cedar shed cost in the garden", "s1")
        self.assertTrue(mem._budget["queue"].empty())


class HeaderNamesTheProduct(unittest.TestCase):
    """What the user sees when the hook fires names the product (2026-11-08), not the MCP server id, which stays in
    every app's config only."""

    def test_the_recall_block_and_its_nudges_name_the_product(self):
        for text in (mem.CONTEXT_HEAD, mem.CONTEXT_HEAD_LEAN, mem.LOOK_AGAIN_LINE, mem.REMEMBER_NUDGE):
            self.assertIn(mem.LABEL, text)
            self.assertNotIn(mem.SERVER_NAME, text)
        self.assertTrue(mem.CONTEXT_HEAD.startswith(RECALL_HEAD))
        self.assertTrue(mem.CONTEXT_HEAD_LEAN.startswith(RECALL_HEAD))


if __name__ == "__main__":
    unittest.main()
