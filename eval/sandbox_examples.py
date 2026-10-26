"""Precompute keyword (BM25) vs vector (all-MiniLM-L6-v2) top-3 articles for the 25 test questions,
for the comparison sandbox on malavnaik.com. Uses the same documents, tokenizer and model as eval/evaluate.py."""
import json, os, re

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

chunks = [json.loads(l) for l in open("data/chunks.jsonl", encoding="utf-8")]
questions = json.load(open("eval/questions.json"))
arts = [c for c in chunks if c["type"] == "article"]
doc = lambda c: f"Article {c['article']} {c['title']}. {c['text']}"
tok = lambda s: re.findall(r"[a-z0-9]+", s.lower())

bm25 = BM25Okapi([tok(doc(c)) for c in arts])
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
E = model.encode([doc(c) for c in arts], normalize_embeddings=True, batch_size=64)
rankers = {"keyword": lambda q: np.argsort(-bm25.get_scores(tok(q))),
           "vector": lambda q: np.argsort(-(E @ model.encode(q, normalize_embeddings=True)))}


def top_articles(order, k=3):
    """Best chunk of each article, in rank order (same rule as the evaluation)."""
    out, seen = [], set()
    for i in order:
        c = arts[i]
        if c["article"] in seen:
            continue
        seen.add(c["article"])
        text = c["text"] if len(c["text"]) <= 240 else c["text"][:237].rsplit(" ", 1)[0] + "…"
        out.append({"article": c["article"], "title": c["title"], "url": c["url"], "paragraph": text})
        if len(out) == k:
            return out


examples = []
for item in questions:
    row = {"q": item["q"], "expected": item["articles"]}
    for name, rank in rankers.items():
        row[name] = top_articles(rank(item["q"]))
    examples.append(row)
hits = {n: sum(any(a["article"] in r["expected"] for a in r[n]) for r in examples) for n in rankers}
print("top-3 hits out of", len(examples), hits)
assert hits["keyword"] == 17 and hits["vector"] == 21, "should match the published evaluation (68% and 84% in the top 3)"
os.makedirs("out", exist_ok=True)
json.dump({"source": "malavnaik99/eu-dsa-search eval/sandbox_examples.py; text: EUR-Lex, © European Union",
           "hits_top3": hits, "examples": examples}, open("out/sandbox.json", "w"), ensure_ascii=False, separators=(",", ":"))
