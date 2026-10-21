import sys, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import build_corpus


class TestCorpus(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chunks = build_corpus.build(str(ROOT / "data/dsa_en.html"))
        cls.arts = [c for c in cls.chunks if c["type"] == "article"]

    def test_all_93_articles_and_156_recitals(self):
        self.assertEqual(sorted({c["article"] for c in self.arts}), list(range(1, 94)))
        self.assertEqual(sum(c["type"] == "recital" for c in self.chunks), 156)

    def test_article_metadata(self):
        a1 = [c for c in self.arts if c["article"] == 1]
        self.assertEqual(a1[0]["title"], "Subject matter")
        self.assertTrue(a1[0]["chapter"].startswith("CHAPTER I"))
        self.assertTrue(a1[0]["url"].endswith("#art_1"))
        self.assertTrue(a1[0]["text"].startswith("1. The aim of this Regulation"))

    def test_no_empty_or_huge_chunks(self):
        self.assertTrue(all(len(c["text"]) > 20 for c in self.chunks))
        self.assertLess(max(len(c["text"]) for c in self.arts), 4000)

    def test_definitions_split_into_points(self):
        self.assertGreater(sum(c["article"] == 3 for c in self.arts), 20)


if __name__ == "__main__":
    unittest.main()
