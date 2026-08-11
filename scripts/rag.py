"""Stand up retrieval-augmented generation over the chunked rules.

Implements README Section 6.1: embed every chunk with a small local
embedding model, store the vectors, and retrieve the most relevant
chunks for a query at inference time. Do this before any fine-tuning —
it's the RAG-only baseline (Section 9.4) and the grounding source for
SFT data generation (Section 7.2).

448 chunks is small enough that brute-force cosine similarity over an
in-memory array is instant; a FAISS/Chroma index would be pure overhead
at this corpus size.

Usage:
    python scripts/rag.py index                                  # embed chunks.jsonl -> chunk_embeddings.npz
    python scripts/rag.py query "when are SBAs checked?" [--k 5]
"""

import argparse
import json
from pathlib import Path

import numpy as np
from mlx_embeddings import generate, load

MODEL_ID = "mlx-community/all-MiniLM-L6-v2-4bit"
CHUNKS_PATH = Path("data/processed/chunks.jsonl")
INDEX_PATH = Path("data/processed/chunk_embeddings.npz")


def load_chunks(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def embed_texts(model, tokenizer, texts: list[str], batch_size: int = 32) -> np.ndarray:
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        output = generate(model, tokenizer, texts=batch)
        vectors.append(np.array(output.text_embeds))
    return np.concatenate(vectors, axis=0)


def build_index(chunks_path: Path, index_path: Path) -> None:
    chunks = load_chunks(chunks_path)
    model, tokenizer = load(MODEL_ID)
    embeddings = embed_texts(model, tokenizer, [c["text"] for c in chunks])
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        index_path,
        embeddings=embeddings,
        chunk_ids=np.array([c["chunk_id"] for c in chunks]),
        model_id=MODEL_ID,
    )
    print(f"embedded {len(chunks)} chunks ({embeddings.shape[1]}-dim) -> {index_path}")


def retrieve(
    query: str, k: int, chunks_path: Path = CHUNKS_PATH, index_path: Path = INDEX_PATH,
    model_and_tokenizer: tuple | None = None,
) -> list[dict]:
    if not index_path.exists():
        raise FileNotFoundError(f"{index_path} not found — run `python scripts/rag.py index` first")
    stored = np.load(index_path, allow_pickle=True)
    embeddings, chunk_ids = stored["embeddings"], stored["chunk_ids"]

    # Loading the embedding model takes real time, so callers doing many
    # retrievals in a loop (e.g. build_reddit_eval.py) should load it once
    # and pass it in rather than paying that cost on every call.
    model, tokenizer = model_and_tokenizer if model_and_tokenizer else load(MODEL_ID)
    query_vec = embed_texts(model, tokenizer, [query])[0]
    query_vec = query_vec / np.linalg.norm(query_vec)

    scores = embeddings @ query_vec  # both sides pre-normalized -> cosine similarity
    top_k = np.argsort(-scores)[:k]

    chunks_by_id = {c["chunk_id"]: c for c in load_chunks(chunks_path)}
    return [{**chunks_by_id[str(chunk_ids[i])], "score": float(scores[i])} for i in top_k]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    idx_p = sub.add_parser("index", help="Embed all chunks and build the vector store")
    idx_p.add_argument("--chunks", type=Path, default=CHUNKS_PATH)
    idx_p.add_argument("--out", type=Path, default=INDEX_PATH)

    q_p = sub.add_parser("query", help="Retrieve the top-k most relevant chunks for a query")
    q_p.add_argument("text")
    q_p.add_argument("--k", type=int, default=5)
    q_p.add_argument("--chunks", type=Path, default=CHUNKS_PATH)
    q_p.add_argument("--index", type=Path, default=INDEX_PATH)

    args = parser.parse_args()

    if args.command == "index":
        build_index(args.chunks, args.out)
    else:
        for r in retrieve(args.text, args.k, args.chunks, args.index):
            print(f"[{r['score']:.3f}] {r['chunk_id']} ({r['section_title']})")
            print(r["text"][:300].replace("\n", " "))
            print()


if __name__ == "__main__":
    main()
