# %% [markdown]
# # Searching the EU Digital Services Act: keyword vs semantic vs hybrid
# A portfolio project by [Malav Naik](https://malavnaik99.github.io). **Not legal advice.**
#
# The DSA (Regulation (EU) 2022/2065) is split into 638 chunks (each paragraph of the 93 articles, plus 156 recitals). For 25 hand-written questions,
# each with the article(s) that answer it, three ways of finding the right article are compared:
# - **BM25:** classic keyword search
# - **Semantic:** the small open embedding model `all-MiniLM-L6-v2` (the same one the browser demo runs)
# - **Hybrid:** both rankings combined with reciprocal rank fusion
#
# Results are scored per article (a chunk's article counts once, at its best rank): recall@1/3/5 and mean reciprocal rank (MRR).
# Source text: EUR-Lex, © European Union.

# %%
import json, os, re
import numpy as np
from rank_bm25 import BM25Okapi

DATA = os.environ.get("DATA", "data")
chunks = [json.loads(l) for l in open(f"{DATA}/chunks.jsonl", encoding="utf-8")]
questions = json.load(open(os.environ.get("QUESTIONS", "eval/questions.json")))
arts = [c for c in chunks if c["type"] == "article"]
doc = lambda c: f"Article {c['article']} {c['title']}. {c['text']}"
tok = lambda s: re.findall(r"[a-z0-9]+", s.lower())
print(len(arts), "article chunks,", len(questions), "questions")

# %%
def to_articles(order):
    """Chunk ranking -> article ranking (first time each article appears)."""
    seen = []
    for i in order:
        a = arts[i]["article"]
        if a not in seen:
            seen.append(a)
    return seen

bm25 = BM25Okapi([tok(doc(c)) for c in arts])
def rank_bm25(q):
    return list(np.argsort(-bm25.get_scores(tok(q))))

DENSE = os.environ.get("DENSE", "1") == "1"
if DENSE:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    E = model.encode([doc(c) for c in arts], normalize_embeddings=True, batch_size=64)
    def rank_dense(q):
        return list(np.argsort(-(E @ model.encode(q, normalize_embeddings=True))))

def rank_hybrid(q, k=60):
    score = {}
    for order in (rank_bm25(q), rank_dense(q)):
        for r, i in enumerate(order):
            score[i] = score.get(i, 0) + 1 / (k + r + 1)
    return sorted(score, key=score.get, reverse=True)

# %%
def evaluate(rank):
    rows = []
    for item in questions:
        found = to_articles(rank(item["q"]))
        pos = min((found.index(a) + 1 for a in item["articles"] if a in found), default=None)
        rows.append(pos)
    n = len(rows)
    return {f"recall@{k}": round(sum(p is not None and p <= k for p in rows) / n, 3) for k in (1, 3, 5)} | \
           {"mrr": round(sum(1 / p for p in rows if p) / n, 3)}

methods = {"bm25": rank_bm25} | ({"semantic": rank_dense, "hybrid": rank_hybrid} if DENSE else {})
results = {name: evaluate(f) for name, f in methods.items()}
print(json.dumps(results, indent=1))
os.makedirs("out", exist_ok=True)
json.dump(results, open("out/eval.json", "w"), indent=1)

# %% [markdown]
# ## Misses of the best method
# Which questions the best method doesn't place in its top 3, and what it found instead. Useful for seeing *why* search fails.

# %%
best = max(results, key=lambda m: results[m]["mrr"])
for item in questions:
    found = to_articles(methods[best](item["q"]))[:3]
    if not set(found) & set(item["articles"]):
        print(f"- {item['q']}\n  expected {item['articles']}, got {found}")
