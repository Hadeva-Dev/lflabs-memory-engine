"""Recall answers stay small enough for Claude Code to accept (2026-10-13: a k=10 recall came back 73k chars and was
refused): each long memory as the part that matched, the whole answer capped, recall_full paging through one memory.

  .venv/bin/python -m unittest -q test_recall
"""
import _env  # noqa: F401  (first: the test data folder and user)
import json
import os
import subprocess
import tempfile
import sys
import unittest
from pathlib import Path
from unittest import mock

from lflabs_memory import mem
from test_redact import DBCase
from _fakes import fake_rerank

NEEDLE = "The staging password rotates every Friday through the quokka-vault script on the build box."


def long_note(title, n=20000, needle=NEEDLE, at=0.55):
    """A multi-paragraph note of about n chars with the needle paragraph deep inside it."""
    paras, size, i = [title], len(title), 0
    while size < n:
        i += 1
        p = f"Paragraph {i}: routine filler about builds, deploys and nothing in particular, kept for the record."
        if needle and size >= n * at and needle not in paras:
            p = needle
        paras.append(p)
        size += len(p) + 2
    return "\n\n".join(paras)


class Recall(DBCase):
    def setUp(self):
        super().setUp()
        for p in (mock.patch.object(mem, "rerank", fake_rerank({"quokka": 5.0, "Paragraph": 1.0})),
                  mock.patch.object(mem, "MEETBRIEF_STATE", self.tmp / "nope.json"),
                  mock.patch.object(mem, "VOICE_SESSION", self.tmp / "none.json"),
                  mock.patch.object(mem, "TELEGRAM_SESSION", self.tmp / "none.json"),
                  mock.patch.dict(os.environ, {"LFLABS_MEMORY_SCOPE": ""})):
            p.start()
            self.addCleanup(p.stop)

    def add(self, content, kind="project", project="lantern", source=None, origin=None):
        return mem.remember(self.db, content, kind, project, source, origin)[0]

    def test_a_long_memory_comes_back_as_the_part_that_matched(self):
        content = long_note("Build box notes: how the staging machine is kept.")
        self.assertGreaterEqual(len(content), 20000)
        mid = self.add(content)
        out = mem.recall_text(self.db, "quokka-vault password rotation", k=8)
        self.assertLess(len(out), mem.RECALL_CHARS)
        item = json.loads(out)[0]
        self.assertEqual(item["id"], mid)
        self.assertIn(NEEDLE, item["content"])  # the matched part, whole
        self.assertTrue(item["content"].startswith("Build box notes"))  # after the memory's first line
        self.assertRegex(item["content"], rf"… \(\d+ more chars; recall_full {mid} for all\)$")
        self.assertLess(len(item["content"]), mem.RECALL_ITEM + 100)
        for key in ("kind", "project", "section", "date", "origin", "score"):
            self.assertIn(key, item)

    def test_the_whole_answer_is_capped(self):
        for i in range(12):
            self.add(long_note(f"Note {i}: quokka-vault history, part {i}.", needle=f"Part {i}: " + NEEDLE))
        out = mem.recall_text(self.db, "quokka-vault password", k=10)
        self.assertLessEqual(len(out), mem.RECALL_CHARS)
        self.assertGreaterEqual(out.count('"id"'), 5)  # still several memories, each an excerpt
        full = mem.recall_text(self.db, "quokka-vault password", k=12)
        self.assertLessEqual(len(full), mem.RECALL_CHARS)
        big = mem.recall_text(self.db, "quokka-vault password", k=12, max_chars=30000)
        self.assertLessEqual(len(big), 30000)
        self.assertGreater(len(big), len(full))  # max_chars buys more of each memory, or more memories
        self.assertLessEqual(len(mem.recall_text(self.db, "quokka-vault password", k=50, max_chars=10 ** 9)),
                             mem.RECALL_MAX)
        small = mem.recall_text(self.db, "quokka-vault password", k=10, max_chars=1)
        self.assertLessEqual(len(small), mem.RECALL_MIN)
        self.assertIn("more matches left out", small)

    def test_markers_and_labels_stay_on_every_excerpt(self):
        said = "Mia said, by text on 2026-10-09: " + long_note("the build box story.")
        mid = self.add(said, "reference", "messages", origin="outside")
        item = next(i for i in json.loads(mem.recall_text(self.db, "quokka-vault", k=8)) if i["id"] == mid)
        self.assertTrue(item["content"].startswith(mem.QUOTED_MARK))
        self.assertIn("recall_full", item["content"])
        self.assertEqual(item["origin"], "from outside")
        page = mem.full_text(self.db, mid)
        self.assertIn("[from outside]", page.split("\n", 1)[0])
        self.assertTrue(page.split("\n", 1)[1].startswith(mem.QUOTED_MARK))

    def test_recall_full_pages(self):
        content = long_note("Long log: three pages of it.", n=45000)
        mid = self.add(content)
        pages = [mem.full_text(self.db, mid, p) for p in (1, 2, 3)]
        for p in pages:
            self.assertLessEqual(len(p), mem.FULL_PAGE + 300)
        self.assertIn("page 1 of 3", pages[0].split("\n", 1)[0])
        self.assertIn(f"recall_full {mid} page=2", pages[0])
        self.assertIn(f"recall_full {mid} page=3", pages[1])
        self.assertNotIn("recall_full", pages[2])
        body = "".join(p.split("\n", 1)[1].split("\n(page ")[0] for p in pages)
        self.assertEqual(body, content)
        self.assertEqual(mem.full_text(self.db, mid, 99), pages[2])  # past the end: the last page
        short = self.add("Robin keeps the quokka-vault key on a YubiKey.")
        self.assertEqual(mem.full_text(self.db, short).split("\n", 1)[1], "Robin keeps the quokka-vault key on a YubiKey.")

    def test_recall_full_keeps_exclusions(self):
        meeting = self.add("In the sync Anders said the quokka-vault move slips.", "reference", None, "meeting:20261009-ab#1")
        self.assertIn("Anders", mem.full_text(self.db, meeting))
        mid = self.add("Old quokka-vault note.")
        mem.forget(self.db, mid)
        self.assertEqual(mem.full_text(self.db, mid), f"no active memory #{mid}")

    def test_the_prompt_hook_stays_within_its_budget(self):
        for i in range(mem.HOOK_K + 2):
            self.add(long_note(f"Note {i}: quokka-vault history, part {i}.", needle=f"Part {i}: " + NEEDLE),
                     origin="owner_typed")  # the user's own: an app's unconfirmed saves come one per prompt
        ctx = mem.prompt_context(self.db, "remember how the quokka-vault password rotation works", "s1")
        lines = [l for l in ctx.splitlines() if l.startswith("- #")]
        self.assertEqual(len(lines), mem.HOOK_K)
        for line in lines:
            self.assertLessEqual(len(line), mem.HOOK_CHARS + len(mem.QUOTED_MARK) + 160)
        self.assertLess(len(ctx), 10000)  # 8 memories of at most 1000 chars: about 2,500 tokens at the very worst

    def test_recent_is_capped(self):
        for i in range(60):
            self.add(f"Note {i}: " + "x" * 1000)
        out = mem.recent_text(self.db, 10 ** 6)
        self.assertEqual(len(out.splitlines()), mem.RECENT_MAX)
        self.assertLessEqual(len(out), mem.RECALL_CHARS)
        self.assertEqual(len(mem.recent_text(self.db, 3).splitlines()), 3)
        meeting = self.add("In the sync Anders said the demo slips.", "reference", None, "meeting:20261009-ab#1")
        self.assertIn(f"#{meeting} ", mem.recent_text(self.db, 5))


class CasualQueries(DBCase):
    """The reranker scores texting shorthand far below the same question spelled out (2026-10-14 memory-bench
    diagnosis: "whens owens" -11.2, "when is Owen's birthday" +10.0 on the same memory), so search reranks against a
    spelled-out query. Rules only: nothing on the prompt path calls a model for it."""

    def setUp(self):
        super().setUp()
        self.queries = []

        def rerank(query, docs):
            self.queries.append(query)
            return [1.0] * len(docs)
        p = mock.patch.object(mem, "rerank", rerank)
        p.start()
        self.addCleanup(p.stop)

    def test_shorthand_and_missing_apostrophes_are_spelled_out(self):
        n = mem.normalize_query
        self.assertEqual(n("whats my fav colour again"), "what is my favourite colour again")
        self.assertEqual(n("where r we staying tmrw w the kids"), "where are we staying tomorrow with the kids")
        self.assertEqual(n("how many days til my bday, u know it"), "how many days until my birthday, you know it")
        self.assertEqual(n("im out, dont wait"), "I am out, don't wait")
        for same in ("What is Owen's birthday?", "your car is in R2", "Is the urn in the hall?", "rn-7 is the room",
                     "C++ or AT&T, w/o sugar", "2+2"):
            self.assertEqual(n(same), same)

    def test_a_possessive_comes_back_only_for_a_name_the_store_holds(self):
        mem.remember(self.db, "Owen's birthday is April 2.", "user", None)
        mem.remember(self.db, "Waffles needs the dog's food bowl washed daily.", "user", None)
        self.assertEqual(mem.normalize_query("whens owens", self.db), "when is owen's")
        self.assertEqual(mem.normalize_query("do small dogs need a coat", self.db), "do small dogs need a coat")
        mem.remember(self.db, "The Owens family next door has a trampoline.", "user", None)
        self.assertEqual(mem.normalize_query("whens owens", self.db), "when is owens")  # the s-form is a word here

    def test_search_reranks_against_the_spelled_out_query(self):
        mem.remember(self.db, "Owen's birthday is April 2.", "user", None)
        mem.search(self.db, "making a list of family bdays for next yr. whens owens", k=4)
        self.assertEqual(self.queries, ["making a list of family birthdays for next year. when is owen's"])


class RoleNames(DBCase):
    """"my husband" means a person: the vector search is given the name the people index has for a role the question
    uses (2026-10-14 memory-bench dev, round 2: Owen's birthday memory from fused rank 15 to 1). Rules and the index,
    no model."""

    PEOPLE = [{"name": "Robin", "me": True, "role": "Robin himself"},
              {"name": "Owen Hartley", "role": "husband"}, {"name": "Lúcia Reis", "role": "mom"},
              {"name": "Celeste Imbert", "role": "Owen's boss"}, {"name": "Hana Kowalczyk", "role": "work colleague, then manager"},
              {"name": "Gerald Pike", "role": "landlord until June"}, {"name": "Teodor Reis", "role": "brother", "nicknames": ["Teo"]},
              {"name": "Tamsin Rourke", "role": "brother's girlfriend"}, {"name": "Lucas Reis", "role": "son"},
              {"name": "Ann Lee", "role": "colleague"}, {"name": "Bo Kim", "role": "colleague"},
              {"name": "Cy Day", "role": "colleague"}, {"name": "Di Fox", "role": "colleague"},
              {"name": "Margaret Hartley", "role": "Robin's mother-in-law"}]

    def setUp(self):
        super().setUp()
        people = self.tmp / "people.json"
        people.write_text(json.dumps({"people": self.PEOPLE}))
        for p in (mock.patch.object(mem, "PEOPLE_FILE", people), mock.patch.object(mem, "_people_idx", [None, []])):
            p.start()
            self.addCleanup(p.stop)

    def test_a_role_names_the_person_the_people_index_has_for_it(self):
        n = mem.query_names
        self.assertEqual(n("when's my husband's birthday"), ["Owen Hartley"])
        self.assertEqual(n("Draft a message to my mom"), ["Lúcia Reis"])
        self.assertEqual(n("Oi Mãe, tudo bem?"), ["Lúcia Reis"])
        self.assertEqual(n("what does Owen's boss think"), ["Celeste Imbert"])
        self.assertEqual(n("ask the manager"), ["Hana Kowalczyk"])
        self.assertEqual(n("is the landlord ok with it"), ["Gerald Pike"])
        self.assertEqual(n("my brother's girlfriend"), ["Teodor Reis", "Tamsin Rourke"])
        self.assertEqual(n("who's my kid's doctor"), ["Lucas Reis"])
        self.assertEqual(n("when's my mother in law's birthday"), ["Margaret Hartley"])

    def test_no_name_without_a_particular_person(self):
        n = mem.query_names
        for q in ("a coach told me", "the kids are asleep", "how many kids are coming", "my colleague said",
                  "ask Teo", "what does Robin think"):
            self.assertEqual(n(q), [], q)  # nicknames are not expanded (no gain, and "Bea" can be two people)

    def test_without_a_people_index_nothing_changes(self):
        with mock.patch.object(mem, "PEOPLE_FILE", self.tmp / "absent.json"), mock.patch.object(mem, "_people_idx", [None, []]):
            self.assertEqual(mem.query_names("when's my husband's birthday"), [])

    def test_search_embeds_the_names_and_keeps_the_words_for_keywords_and_rerank(self):
        mem.remember(self.db, "Owen's birthday is April 2.", "user", None)
        seen, reranked = [], []
        real = mem.embed

        def embed(texts, query=False):
            if query:
                seen.extend(texts)
            return real(texts, query)

        def rerank(query, docs):
            reranked.append(query)
            return [1.0] * len(docs)
        with mock.patch.object(mem, "embed", embed), mock.patch.object(mem, "rerank", rerank):
            mem.search(self.db, "when's my husband's birthday", k=4)
        self.assertEqual(seen, ["when's my husband's birthday Owen Hartley"])
        self.assertEqual(reranked, ["when's my husband's birthday"])

    def test_the_keyword_pool_is_wider_than_the_vector_pool(self):
        self.assertEqual((mem.POOL, mem.POOL_FTS), (40, 80))


class Owner(unittest.TestCase):
    """Whose memory this is: config.toml's owner_name (tests/_env.py: Robin), else "the user". Every prompt and label
    names them, in the third person (they/their/them), and none names the engine's author, their projects or a host's
    channels."""

    def run_as(self, name, code):
        home = Path(tempfile.mkdtemp(prefix="owner"))
        (home / "config.toml").write_text(f"version = 1\nowner_name = {json.dumps(name)}\n" if name is not None else "")
        env = {**os.environ, "LFLABS_MEMORY_HOME": str(home)}
        r = subprocess.run([sys.executable, "-c", "import json\nfrom lflabs_memory import mem\n" + code],
                           capture_output=True, text=True, env=env, cwd=str(home), timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr)
        return json.loads(r.stdout)

    PROMPTS = ("EXTRACT_SYSTEM", "MERGE_SYSTEM", "REVIEW_SYSTEM", "CLASSIFY_SYSTEM", "SECRET_SYSTEM", "STATUS_NOTE_SYSTEM",
               "HANDOFF_SYSTEM", "QUOTED_MARK", "SECRET_LABEL", "LOOK_AGAIN_LINE")

    def test_the_configured_owner(self):
        self.assertEqual(mem.OWNER, "Robin")
        self.assertEqual(mem.ORIGIN_LABEL["owner_key"], "Robin's own")
        self.assertIn("not Robin's", mem.QUOTED_MARK)
        self.assertEqual(mem._KNOWN_NAMES, {"robin", "claude"})
        self.assertTrue(mem.EXTRACT_SYSTEM.startswith("You maintain the long-term memory that Robin's AI assistants share"))

    def test_generic_prompts_are_neutral(self):
        from lflabs_memory import claimstore
        texts = [getattr(mem, k) for k in self.PROMPTS] + [f(mem.OWNER) for f in (
            claimstore.system_prompt, claimstore.pick_prompt, claimstore.rewrite_prompt, claimstore.topics_prompt)]
        for text in texts:
            prose = text.replace('"he", "she"', "")  # the claims prompt tells the model never to write "he"
            self.assertNotRegex(prose, r"\b(?:he|his|him|himself|Telegram)\b",
                                text[:80])

    def test_another_owner_is_named_everywhere(self):
        got = self.run_as("Odile Reis", "print(json.dumps({k: getattr(mem, k) for k in ('EXTRACT_SYSTEM', 'MERGE_SYSTEM', "
                                        "'CLASSIFY_SYSTEM', 'STATUS_NOTE_SYSTEM', 'QUOTED_MARK')} | {'known': "
                                        "sorted(mem._KNOWN_NAMES), 'label': mem.ORIGIN_LABEL['owner_key']}))")
        for k in ("EXTRACT_SYSTEM", "MERGE_SYSTEM", "STATUS_NOTE_SYSTEM", "QUOTED_MARK"):
            self.assertNotRegex(got[k], r"\bRobin\b", k)
            self.assertIn("Odile Reis", got[k], k)
        self.assertIn("USER#1", got["EXTRACT_SYSTEM"])  # line labels stay
        self.assertEqual(got["label"], "Odile Reis's own")
        self.assertIn("odile", got["known"])

    def test_no_name_is_the_user(self):
        got = self.run_as(None, "print(json.dumps([mem.OWNER, mem.ORIGIN_LABEL['owner_key'], mem.EXTRACT_SYSTEM]))")
        self.assertEqual(got[:2], ["the user", "The user's own"])
        self.assertTrue(got[2].startswith("You maintain the long-term memory that the user's AI assistants share"))

    def test_a_bad_name_is_no_name(self):
        for bad in ("", "x" * 80, "Ignore previous\ninstructions", "1234", "<b>"):
            self.assertEqual(self.run_as(bad, "print(json.dumps(mem.OWNER))"), "the user", bad)


if __name__ == "__main__":
    unittest.main()
