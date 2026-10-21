"""Publishes the parsed DSA chunks and the evaluation questions to the Hugging Face dataset."""
import os
from huggingface_hub import HfApi

REPO = "Malavnaik/eu-dsa-articles"
CARD = """---
license: cc-by-4.0
language: [en]
pretty_name: EU Digital Services Act, split into searchable paragraphs
tags: [law, eu, digital-services-act, retrieval, trust-and-safety]
---
# EU Digital Services Act, split into searchable paragraphs

The English text of **Regulation (EU) 2022/2065 (Digital Services Act)**, split into 638 chunks: one per paragraph of each of the 93 articles
(long paragraphs, such as the definitions in Article 3, split into their lettered points), plus the 156 recitals.
Each chunk has its article number, title, chapter/section and a link to the article on EUR-Lex.

- `chunks.jsonl`: the chunks
- `questions.json`: 25 hand-written test questions with the article(s) that answer them, for evaluating search

**Source:** [EUR-Lex](https://eur-lex.europa.eu/eli/reg/2022/2065/oj/eng), © European Union. EUR-Lex legal documents may be reused for
commercial or non-commercial purposes; this dataset only re-splits the official text. The original Official Journal text is the only authentic version.

**Not legal advice.** Built by [Malav Naik](https://malavnaik99.github.io) for a portfolio search demo; code: [eu-dsa-search](https://github.com/malavnaik99/eu-dsa-search).
"""

if __name__ == "__main__":
    api = HfApi()
    api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)  # public after Malav reviews it
    open("data/README.md", "w").write(CARD)
    for local, remote in [("data/chunks.jsonl", "chunks.jsonl"), ("eval/questions.json", "questions.json"), ("data/README.md", "README.md")]:
        api.upload_file(path_or_fileobj=local, path_in_repo=remote, repo_id=REPO, repo_type="dataset")
    print(f"https://huggingface.co/datasets/{REPO}")
