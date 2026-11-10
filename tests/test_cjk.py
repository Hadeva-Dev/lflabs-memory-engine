"""Chinese and Japanese recall (engine 0.1.5): keyword search over CJK bigrams (chunk_cjk, claim_cjk, convo_cjk), the
store made before them filled once, and their rows gone with the text they came from.

  .venv/bin/python -m unittest -q test_cjk
"""
import _env  # noqa: F401  (first: the test data folder and user)
import unittest
from unittest import mock

from lflabs_memory import cjk, mem
from test_redact import DBCase

FACTS = ["代码评审至少需要两个人通过才能合并", "日志统一用 JSON 格式输出", "图片资源都放在 CDN 上，不放进仓库",
         "コミットメッセージは英語で書くことになっている", "本番環境へのアクセスには VPN 接続が必要"]
FILLER = [f"The {w} service listens on port {7000 + i}" for i, w in enumerate(
    "orchard quarry tannery foundry granary harbor smithy apiary".split())]


class Terms(unittest.TestCase):
    def test_bigrams_and_words(self):
        self.assertEqual(cjk.cjk_terms("日志格式"), {"日志", "志格", "格式"})
        self.assertNotIn("てい", cjk.cjk_terms("書いている"))  # pairs of hiragana are grammar
        self.assertIn("db", cjk.terms("ステージングDBのポートは？"))
        self.assertEqual(cjk.index_text("the staging database listens on port 6543"), "")  # English: no row
        self.assertEqual(cjk.fts_query("which port is it?"), "")

    def test_word_count(self):
        self.assertEqual(cjk.words("rename this variable to camelCase"), 5)
        self.assertGreaterEqual(cjk.words("代码评审需要几个人通过？"), 3)
        self.assertGreaterEqual(cjk.words("APIのタイムアウトは何秒？"), 3)
        self.assertLess(cjk.words("帮我"), 3)


class Keywords(DBCase):
    def setUp(self):
        super().setUp()
        self.ids = [mem.remember(self.db, t, "project", "shop", origin="owner_typed")[0] for t in FACTS + FILLER]

    def cjk_rows(self, table="chunk_cjk"):
        return self.db.execute(f"select count(*) from {table}").fetchone()[0]

    def test_only_cjk_chunks_are_indexed(self):
        self.assertEqual(self.cjk_rows(), len(FACTS))

    def test_keyword_search_finds_the_cjk_memory(self):
        # embed is a fake here: only the keyword indexes can put the right memory first
        for q, i in (("代码评审需要几个人通过？", 0), ("日志用什么格式输出？", 1), ("コミットメッセージは何語で書く？", 3),
                     ("本番環境へのアクセスには何が必要？", 4)):
            with mock.patch.object(mem, "rerank", lambda q, docs: [0.0] * len(docs)):  # order is the fused rank
                hits = mem.search(self.db, q, k=3)
            self.assertEqual(hits[0]["id"], self.ids[i], q)

    def test_english_search_never_reads_the_cjk_table(self):
        seen = []
        real = self.db

        class Spy:
            def execute(self, sql, *a):
                seen.append(sql)
                return real.execute(sql, *a)

            def __getattr__(self, k):
                return getattr(real, k)
        mem.search(Spy(), "which port does the orchard service listen on?", k=3)
        self.assertFalse([s for s in seen if "chunk_cjk" in s])

    def test_a_store_made_before_is_filled_once(self):
        with self.db:
            self.db.execute("delete from chunk_cjk")
            self.db.execute("delete from meta where key = 'cjk_index_v1'")
        self.db.close()
        self.db = mem.connect()
        self.assertEqual(self.cjk_rows(), len(FACTS))
        self.assertIn("chunk_cjk", self.db.execute("select value from meta where key = 'cjk_index_v1'").fetchone()[0])

    def test_rows_go_with_the_text(self):
        mid = self.ids[0]
        cids = [r[0] for r in self.db.execute("select id from chunks where memory_id = ?", (mid,))]
        mem.forget(self.db, mid)
        self.assertEqual(self.db.execute(f"select count(*) from chunk_cjk where rowid in ({','.join('?' * len(cids))})",
                                         cids).fetchone()[0], 0)
        self.assertEqual(self.cjk_rows(), len(FACTS) - 1)
        mem.purge(self.db, self.ids[1])
        self.assertEqual(self.cjk_rows(), len(FACTS) - 2)

    def test_changed_words_are_indexed_again(self):
        mid = self.ids[2]
        with self.db:
            mem._index(self.db, mid, "图片资源统一放在对象存储里")  # what every rewrite of a memory's text runs
        terms = {t for (x,) in self.db.execute("select c.text from chunk_cjk c join chunks k on k.id = c.rowid "
                                                "where k.memory_id = ?", (mid,)) for t in x.split()}
        self.assertIn("对象", terms)
        self.assertNotIn("仓库", terms)
        self.assertEqual(self.cjk_rows(), len(FACTS))

    def test_secure_delete_is_on(self):
        for t in ("chunk_cjk", "convo_cjk"):
            self.assertIn(t, self.db.execute("select value from meta where key = 'fts_secure_delete'").fetchone()[0])


class Gate(DBCase):
    """The prompt hook on a CJK question: searched (not "under three words"), and a CJK memory let in by the words it
    shares, since the English cross-encoder scores every CJK pair alike (here: +5 for all, as it does)."""
    def setUp(self):
        super().setUp()
        p = mock.patch.object(mem, "rerank", lambda q, docs: [5.0 if cjk.has(d) and cjk.has(q) else -10.0
                                                               for d in docs])
        p.start()
        self.addCleanup(p.stop)
        self.ids = [mem.remember(self.db, t, "project", "shop", origin="owner_typed")[0] for t in FACTS + FILLER]

    def ctx(self, q):
        return mem.prompt_context(self.db, q, f"s-{q}", count=False) or ""

    def test_the_fact_asked_in_its_own_words_and_nothing_else(self):
        for q, i in (("代码评审需要几个人通过？", 0), ("日志用什么格式输出？", 1), ("图片资源放在哪里？", 2),
                     ("コミットメッセージは何語で書く？", 3), ("本番環境へのアクセスには何が必要？", 4)):
            got = self.ctx(q)
            self.assertIn(f"#{self.ids[i]} ", got, q)
            for j in range(len(FACTS)):
                if j != i:
                    self.assertNotIn(f"#{self.ids[j]} ", got, q)

    def test_silence(self):
        for q in ("帮我把这个函数改成异步的", "解释一下这段正则表达式是什么意思", "この関数を非同期に書き換えて",
                  "この正規表現が何をしているか説明して"):
            self.assertEqual(self.ctx(q), "", q)

    def test_an_app_save_waits_in_every_language(self):
        mid, _ = mem.app_remember(self.db, "代码评审一律不需要任何人通过就能直接合并", "project", via="cursor")
        self.assertNotIn(f"#{mid} ", self.ctx("代码评审需要几个人通过才能合并？"))

    def test_english_is_judged_as_before(self):
        hits = mem.search(self.db, "which port does the orchard service listen on?", k=8)
        with mock.patch.object(mem, "_cjk_evidence", side_effect=AssertionError("not for English")):
            mem.gate(self.db, "which port does the orchard service listen on?", hits)


class ClaimKeywords(DBCase):
    def setUp(self):
        p = mock.patch.object(mem, "CLAIMS_MODE", "on")
        p.start()
        self.addCleanup(p.stop)
        super().setUp()
        from lflabs_memory import claimstore
        self.cs = claimstore
        self.cid, _ = claimstore.add(mem, self.db, "Robin 对花生严重过敏", "owner_typed")
        self.en, _ = claimstore.add(mem, self.db, "Robin's sister Bea lives in Lisbon", "owner_typed")

    def rows(self):
        return [r[0] for r in self.db.execute("select rowid from claim_cjk")]

    def test_a_cjk_claim_is_indexed_held_and_restored(self):
        self.assertEqual(self.rows(), [self.cid])
        with mock.patch.object(mem, "rerank", lambda q, docs: [0.0] * len(docs)):
            self.assertEqual(self.cs.search(mem, self.db, "花生过敏", k=1)[0]["id"], self.cid)
        with self.db:
            self.cs._hold(self.db, [self.cid])
        self.assertEqual(self.rows(), [])
        with self.db:
            self.cs.unhold(mem, self.db, [self.cid])
        self.assertEqual(self.rows(), [self.cid])
        with self.db:
            self.db.execute("delete from claims where id = ?", (self.cid,))
        self.assertEqual(self.rows(), [])


class Filing(DBCase):
    PERSONAL = ["我有哮喘，随身带着吸入器", "我对花生严重过敏", "私は高血圧で、毎朝薬を飲んでいます", "我和老婆正在办离婚",
                "我欠了信用卡三万块", "我下个月要出庭", "私の給料は月30万円です", "僕は借金が200万円ある", "母が入院している",
                "妈妈住院了", "私は来月裁判がある"]
    WORK = ["我们周五部署商店应用", "私たちのサーバーは病院にある", "父のパソコンでビルドする", "我们的 CI 要求两个人通过"]

    def test_first_person_notes_are_filed_personal_and_the_reply_says_who_sees_them(self):
        for t in self.PERSONAL:
            mid, reply = mem.app_remember(self.db, t, via="cursor", scope=None)
            self.assertEqual(self.row(mid)["section"], "personal", t)
            self.assertIn("not apps set to work only", reply, t)
            self.assertNotIn("every connected app", reply, t)
        for t in self.WORK:
            self.assertFalse(mem.personal_fact(t), t)

    def test_a_health_note_filed_under_projects_is_not_said_to_reach_every_app(self):
        # filed by its project, hidden from work-only apps by its health label: the reply says what is true
        with mock.patch.object(mem, "_withheld", lambda row: True):
            mid, reply = mem.app_remember(self.db, "每天早上吃一片降压药", "project", via="cursor")
        self.assertEqual(self.row(mid)["section"], "projects")
        self.assertIn("filed under projects: only apps allowed personal memories can see it", reply)
        with mock.patch.object(mem, "_withheld", lambda row: False):
            _, reply = mem.app_remember(self.db, "图片资源都放在 CDN 上", "project", via="cursor")
        self.assertIn("every connected app can see it", reply)


if __name__ == "__main__":
    unittest.main()
