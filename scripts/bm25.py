"""Pure-Python BM25 over the CR chunks — the retrieval that can leave this Mac.

Section 21.141 tested a lexical hybrid as a way to FIX dense retrieval and
reported it negative: BM25 found the same questions dense did (6 both, 0
dense-only, 0 BM25-only at k=5), so it added nothing as a complement. That
finding stands.

What it also established, and what matters here, is that BM25 is not WORSE:

    cited-rule recall, 21 benchmark questions citing a CR rule
    k=3   dense 24%   BM25 29%
    k=5   dense 29%   BM25 29%
    k=10  dense 29%   BM25 38%

Dense retrieval costs `mlx-embeddings`, which pins `mlx-metal` and is
Apple-Silicon-only, so it cannot run on fly.io at all. BM25 is stdlib. Equal
recall at zero portability cost makes it the obvious choice for a deployed
service — a negative result from one framing turning out to be the enabling
result for another.

Deliberately NOT a drop-in replacement for `rag.retrieve`: every stored
number was produced with the dense index, and swapping the retriever under
them would silently change what they describe. **Nothing imports this
module.** It exists so 21.141's central negative claim is reproducible rather
than resting on a prototype that was thrown away:

    python scripts/bm25.py --compare-recall

Section 21.149 built a deployable service on it and then reverted that
service; the measurement is what survives.
"""

import math
import re
from collections import Counter
from pathlib import Path

_TOKEN_RE = re.compile(r"[a-z0-9.]+")

K1 = 1.5
B = 0.75


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumerics, keeping dots so `702.19b` survives as one token.

    Rule ids are the highest-signal terms in this corpus and splitting them on
    the dot would turn `702.19b` into `702` and `19b`, which match almost
    everything and nothing respectively.
    """
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """Okapi BM25. Built once at startup; queried per request.

    Scoring is the standard formulation. It is written out rather than pulled
    from a dependency because the deployable image's whole virtue is having
    almost none, and this is twenty lines.
    """

    def __init__(self, docs: list[dict], text_key: str = "text"):
        self.docs = docs
        self.tokens = [tokenize(d.get(text_key, "")) for d in docs]
        self.n = len(docs)
        self.avgdl = (sum(len(t) for t in self.tokens) / self.n) if self.n else 0.0
        df: Counter = Counter()
        for toks in self.tokens:
            df.update(set(toks))
        self.idf = {
            term: math.log(1 + (self.n - count + 0.5) / (count + 0.5))
            for term, count in df.items()
        }
        self.tf = [Counter(t) for t in self.tokens]

    def search(self, query: str, k: int = 3) -> list[dict]:
        if not self.n or k <= 0:
            return []
        q = tokenize(query)
        scored = []
        for i, tf in enumerate(self.tf):
            dl = len(self.tokens[i])
            score = 0.0
            for term in q:
                freq = tf.get(term)
                if not freq:
                    continue
                score += self.idf.get(term, 0.0) * freq * (K1 + 1) / (
                    freq + K1 * (1 - B + B * dl / self.avgdl))
            if score > 0:
                scored.append((score, i))
        scored.sort(key=lambda si: (-si[0], si[1]))
        return [self.docs[i] for _, i in scored[:k]]


def load_index(chunks_path: Path) -> BM25Index:
    import json
    docs = [json.loads(line) for line in
            chunks_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return BM25Index(docs)


def _compare_recall() -> None:
    """Reproduce Section 21.141's dense-vs-BM25 table on the live corpora.

    Prints both retrievers' cited-rule recall AND the overlap, because the
    headline finding is not that BM25 is better — it is that the two find the
    SAME questions, which is what killed the lexical-hybrid hypothesis. Recall
    alone cannot show that; the split can.
    """
    import sys
    from common import CHUNKS_PATH, read_jsonl
    from common import REPO_ROOT

    bench = REPO_ROOT / "data" / "gold" / "card_ruling_candidates.jsonl"
    recs = [r for r in read_jsonl(bench) if r.get("cr_rule_citations")]
    index = load_index(CHUNKS_PATH)
    print(f"{len(recs)} benchmark questions cite a CR rule; {index.n} chunks indexed\n")

    try:
        from mlx_embeddings import load as load_embedder
        from rag import MODEL_ID, retrieve
        embed = load_embedder(MODEL_ID)
    except Exception as exc:                      # noqa: BLE001 - reported, not raised
        embed = None
        print(f"dense retrieval unavailable ({type(exc).__name__}); BM25 only\n",
              file=sys.stderr)

    print(f"{'k':>3}  {'BM25':>6}  {'dense':>6}")
    for k in (3, 5, 10):
        b = sum(1 for r in recs if set(r["cr_rule_citations"]) &
                {i for h in index.search(r["question"], k=k) for i in h.get("rule_ids", [])})
        row = f"{k:>3}  {b}/{len(recs)} = {b / len(recs):.0%}"
        if embed is not None:
            d = sum(1 for r in recs if set(r["cr_rule_citations"]) &
                    {i for h in retrieve(r["question"], k=k, model_and_tokenizer=embed)
                     for i in h.get("rule_ids", [])})
            row += f"   {d}/{len(recs)} = {d / len(recs):.0%}"
        print(row)

    if embed is None:
        return
    k = 5
    bm, dn = set(), set()
    for r in recs:
        want = set(r["cr_rule_citations"])
        if want & {i for h in index.search(r["question"], k=k) for i in h.get("rule_ids", [])}:
            bm.add(r["id"])
        if want & {i for h in retrieve(r["question"], k=k, model_and_tokenizer=embed)
                   for i in h.get("rule_ids", [])}:
            dn.add(r["id"])
    print(f"\noverlap at k={k}: both {len(bm & dn)}, BM25-only {len(bm - dn)}, "
          f"dense-only {len(dn - bm)}, neither {len(recs) - len(bm | dn)}")
    print("BM25-only + dense-only == 0 is Section 21.141's finding: a lexical "
          "hybrid adds nothing,\nbecause the two retrievers fail on the same questions.")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compare-recall", action="store_true",
                    help="reproduce Section 21.141's dense-vs-BM25 comparison")
    args = ap.parse_args()
    if args.compare_recall:
        _compare_recall()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
