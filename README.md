# Ask the EU Digital Services Act

Type a question in plain English and get the articles of the EU Digital Services Act (Regulation (EU) 2022/2065) most likely to answer it, quoted and linked. **Not legal advice:** it only quotes the law, it never writes answers of its own.

- **Demo** (runs in your browser): https://huggingface.co/spaces/Malavnaik/ask-the-dsa
- **Dataset** (638 chunks + 25 test questions): https://huggingface.co/datasets/Malavnaik/eu-dsa-articles
- **Evaluation notebook:** https://www.kaggle.com/code/malavnaik/searching-the-eu-digital-services-act
- **Project page:** https://malavnaik99.github.io/projects/ask-the-dsa/

## How it works
1. `fetch-law.yml` saves the official English text from EUR-Lex to `data/dsa_en.html`.
2. `build_corpus.py` splits it into one chunk per article paragraph (long lists split into their points) plus the recitals, keeping article number, title, chapter and link. `tests/` checks the parser.
3. `eval/evaluate.py` compares keyword (BM25), semantic (`all-MiniLM-L6-v2`) and hybrid search on the 25 questions in `eval/questions.json`.
4. `space/embed.mjs` embeds the article chunks with the browser version of the same model and re-runs the evaluation with it; `space/index.html` is the static demo.

## Results (25 questions, article chunks)
| | Keyword | Semantic | Hybrid | Semantic, browser model |
|---|---|---|---|---|
| Right article ranked first | 0.48 | 0.52 | 0.56 | 0.56 |
| In top 3 | 0.68 | 0.84 | 0.80 | 0.72 |
| In top 5 | 0.72 | 0.92 | 0.84 | 0.92 |
| MRR | 0.599 | 0.675 | 0.701 | 0.686 |

The demo shows three articles, so it uses semantic search (best top-3 score).

## Workflows
All run on demand from the Actions tab: `build.yml` (tests + dataset), `evaluate.yml`, `space.yml` (demo), `run-notebook.yml` (Kaggle). Secrets: `HF_TOKEN`, `KAGGLE_API_TOKEN`. Cost: €0 (free tiers only).

## Licence and source
Source: [EUR-Lex](https://eur-lex.europa.eu/eli/reg/2022/2065/oj/eng), © European Union; EUR-Lex documents may be reused with attribution. Only the Official Journal text is authentic. By [Malav Naik](https://malavnaik99.github.io).
