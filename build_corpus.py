"""Splits the official DSA text (data/dsa_en.html) into searchable chunks: one per paragraph (or per lettered point,
for very long paragraphs such as the definitions in Article 3), plus one per recital. Writes data/chunks.jsonl."""
import json, re, warnings
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)  # the official file is XHTML

SRC = "data/dsa_en.html"
URL = "https://eur-lex.europa.eu/eli/reg/2022/2065/oj/eng"
LONG = 1200  # characters; longer paragraphs are split into their lettered points so each fits the embedding model


def text(el):
    return re.sub(r"\s+", " ", el.get_text(" ", strip=True)).strip()


def heading(div):
    """'CHAPTER III – Due diligence obligations… / Section 2 – …' for an article's chapter (and section, if any)."""
    parts = []
    for anc in reversed(list(div.parents)):
        if anc.name == "div" and re.fullmatch(r"cpt_[IVX]+(\.sct_\d+)?", anc.get("id", "")):
            t1 = anc.find("p", class_="oj-ti-section-1", recursive=False)
            t2 = anc.find("p", class_="oj-ti-section-2") if t1 else None
            parts.append(f"{text(t1)} – {text(t2).capitalize()}" if t1 and t2 else text(t1) if t1 else "")
    return " / ".join(p for p in parts if p)


def article_chunks(div):
    num = int(div["id"].split("_")[1])
    title = text(div.find("p", class_="oj-sti-art"))
    base = {"type": "article", "article": num, "title": title, "chapter": heading(div), "url": f"{URL}#art_{num}"}
    paras = div.find_all("div", id=re.compile(r"^\d{3}\.\d{3}$"), recursive=False)
    for p in paras or [div]:  # an article without numbered paragraphs is one paragraph
        n = str(int(p["id"].split(".")[1])) if p is not div else ""
        full = " ".join(text(x) for x in p.find_all(["p", "table"], recursive=False) if "oj-normal" in x.get("class", ["oj-normal"]))
        points = p.find_all("table", recursive=False)
        if len(full) <= LONG or not points:
            yield {**base, "paragraph": n, "text": full}
            continue
        lead = " ".join(text(x) for x in p.find_all("p", class_="oj-normal", recursive=False))
        for t in points:
            yield {**base, "paragraph": n, "text": f"{lead[:200]} … {text(t)}"}


def recital_chunks(soup):
    for div in soup.find_all("div", id=re.compile(r"^rct_\d+$")):
        num = int(div["id"].split("_")[1])
        cells = div.find_all("td")
        body = text(cells[-1]) if cells else text(div)
        yield {"type": "recital", "article": None, "recital": num, "title": f"Recital {num}", "chapter": "",
               "paragraph": "", "url": f"{URL}#rct_{num}", "text": body}


def build(path=SRC):
    soup = BeautifulSoup(open(path, encoding="utf-8").read(), "lxml")
    chunks = [c for div in soup.find_all("div", id=re.compile(r"^art_\d+$")) for c in article_chunks(div)]
    chunks += list(recital_chunks(soup))
    for i, c in enumerate(chunks):
        c["id"] = i
    return chunks


if __name__ == "__main__":
    chunks = build()
    with open("data/chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    arts = {c["article"] for c in chunks if c["type"] == "article"}
    print(f"{len(chunks)} chunks: {len(arts)} articles, {sum(c['type'] == 'recital' for c in chunks)} recitals")
