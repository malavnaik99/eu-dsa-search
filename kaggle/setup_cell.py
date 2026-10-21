# %%
# On Kaggle: install BM25 and fetch the chunks and questions from the public Hugging Face dataset.
import os, subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rank_bm25"], check=True)
from huggingface_hub import hf_hub_download
os.environ["DATA"] = os.path.dirname(hf_hub_download("Malavnaik/eu-dsa-articles", "chunks.jsonl", repo_type="dataset"))
os.environ["QUESTIONS"] = hf_hub_download("Malavnaik/eu-dsa-articles", "questions.json", repo_type="dataset")

