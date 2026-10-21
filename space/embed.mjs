// Embeds every article chunk with the exact model and settings the browser demo uses (transformers.js, quantized
// all-MiniLM-L6-v2), so search results in the demo match these vectors. Also re-runs the 25-question evaluation
// with this browser model and prints the scores. Usage: node embed.mjs ../data/chunks.jsonl ../eval/questions.json
import { pipeline } from "@huggingface/transformers";
import fs from "node:fs";

const [chunksPath, questionsPath] = process.argv.slice(2);
const chunks = fs.readFileSync(chunksPath, "utf8").trim().split("\n").map(JSON.parse).filter(c => c.type === "article");
const embed = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2", { dtype: "q8" });
const vec = async t => (await embed(t, { pooling: "mean", normalize: true })).data;

const dim = 384, all = new Float32Array(chunks.length * dim);
for (let i = 0; i < chunks.length; i++) {
  const c = chunks[i];
  all.set(await vec(`Article ${c.article} ${c.title}. ${c.text}`), i * dim);
}
fs.writeFileSync("embeddings.bin", Buffer.from(all.buffer));
fs.writeFileSync("chunks.json", JSON.stringify(chunks.map(c => ({ a: c.article, t: c.title, c: c.chapter, p: c.paragraph, x: c.text }))));

// Same evaluation as eval/evaluate.py, with the browser model.
const qs = JSON.parse(fs.readFileSync(questionsPath, "utf8"));
const ranks = [];
for (const q of qs) {
  const v = await vec(q.q), scores = [];
  for (let i = 0; i < chunks.length; i++) { let s = 0; for (let d = 0; d < dim; d++) s += v[d] * all[i * dim + d]; scores.push([s, chunks[i].article]); }
  const seen = []; for (const [, a] of scores.sort((x, y) => y[0] - x[0])) if (!seen.includes(a)) seen.push(a);
  const pos = Math.min(...q.articles.map(a => seen.indexOf(a) + 1 || Infinity));
  ranks.push(pos);
}
const r = k => +(ranks.filter(p => p <= k).length / ranks.length).toFixed(3);
const result = { "recall@1": r(1), "recall@3": r(3), "recall@5": r(5), mrr: +(ranks.reduce((s, p) => s + (isFinite(p) ? 1 / p : 0), 0) / ranks.length).toFixed(3) };
fs.writeFileSync("eval_browser_model.json", JSON.stringify(result));
console.log("Browser model on the 25 questions:", result, "chunks:", chunks.length);
