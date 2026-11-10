"""speech-terms (2026-10-25): the names in Robin's memories for the voice's name list, never a secret or health.

  .venv/bin/python -m unittest -q test_speech_terms
"""
import _env  # noqa: F401  (first: the test data folder and user)
import unittest

from lflabs_memory import mem
from test_redact import DBCase


class SpeechTerms(DBCase):
    def terms(self):
        return {t["term"]: t["memories"] for t in mem.speech_terms(self.db)}

    def test_names_and_pairs_mid_sentence_with_counts(self):
        mem.remember(self.db, "Robin is building Marigold, a menu bar app, with Kenji.")
        mem.remember(self.db, "The Marigold demo is on Friday; ask Zofia Pilar about the safe.")
        t = self.terms()
        self.assertEqual(t["Marigold"], 2)
        self.assertEqual(t["Kenji"], 1)
        self.assertIn("Zofia Pilar", t)
        self.assertNotIn("The", t)  # a sentence's first word is not a name
        self.assertNotIn("Robin is", t)

    def test_secrets_and_health_are_left_out(self):
        mem.remember(self.db, "Robin's Zorblat password is hunter2.")
        mem.remember(self.db, "Robin saw Dr. Quillan about his ADHD medication.")
        t = self.terms()
        self.assertNotIn("Zorblat", t)
        self.assertNotIn("Quillan", t)

    def test_a_cloud_ais_words_are_left_out(self):
        mem.remember(self.db, "Robin met Brundlefly at the Marigold demo.", writer="remote-chatgpt")
        mem.remember(self.db, "Robin demoed Marigold to Kenji.")
        t = self.terms()
        self.assertNotIn("Brundlefly", t)
        self.assertEqual(t["Marigold"], 1)

    def test_retired_memories_are_left_out(self):
        mid = mem.remember(self.db, "Robin tried Glimmerly once.")
        mid = mid if isinstance(mid, int) else self.db.execute("select max(id) from memories").fetchone()[0]
        self.db.execute("update memories set status = 'retired' where id = ?", (mid,))
        self.assertNotIn("Glimmerly", self.terms())

    def test_only_owners_own_memories(self):
        """A cloud AI's memory (writer) and a meeting's fact (other people's words) never feed the name list: it
        biases what a host's voice hears and goes to its speech service as keyterms (security review, 2026-10-25)."""
        mem.remember(self.db, "Robin should text Zarnoff Quell about the wire.", writer="remote-chatgpt")
        mem.remember(self.db, "In the meeting, Vexley Brann said the launch slips.", source="meeting:m1#1")
        mem.remember(self.db, "Priya Ostrander said, by text on 2026-10-20: ask Quorvin about it.")
        mem.remember(self.db, "Robin should call Brixtonn Vale back.", origin="outside")
        mem.remember(self.db, "Robin should pay Ondrecz Fall.", origin="open_mic")
        mem.remember(self.db, "Robin is building Marigold with Kenji.")
        for i, sec in enumerate(("texts", "email", "meetings")):
            self.db.execute("insert into memories(content, hash, created, updated, section) values (?, ?, 1, 1, ?)",
                            (f"Tell Robin that Pellandra Moss{i} moved.", f"h{i}", sec))
        t = self.terms()
        self.assertIn("Kenji", t)
        for gone in ("Zarnoff", "Zarnoff Quell", "Vexley", "Vexley Brann", "Quorvin", "Brixtonn", "Ondrecz", "Pellandra", "Pellandra Moss0", "Moss1", "Moss2"):
            self.assertNotIn(gone, t)


if __name__ == "__main__":
    unittest.main()
