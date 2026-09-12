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
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import CHUNKS_PATH, INDEX_PATH, read_jsonl  # noqa: F401  (re-exported; eval.py imports from here)

MODEL_ID = "mlx-community/all-MiniLM-L6-v2-4bit"


def _mlx():
    """mlx_embeddings, imported on use rather than at module scope.

    Two reasons, both practical. `chat_server` imports MODEL_ID from here
    through `retrieve_hybrid` and must be able to build its argument parser
    without loading a model; and `load_index` below is pure numpy structural
    validation, which is testable on any machine — but only if importing this
    module does not require Apple Silicon first.
    """
    from mlx_embeddings import generate, load
    return generate, load


def fingerprint(chunks: list[dict]) -> str:
    """Identify the exact chunk set an index was built from.

    Re-running chunk.py with different budgets rewrites chunks.jsonl while
    leaving a stale .npz in place. Chunk ids are positional, so the stale
    vectors still *resolve* — retrieval silently returns text that was
    never embedded. Comparing a content hash turns that into an error.
    """
    h = hashlib.sha256()
    for c in chunks:
        h.update(c["chunk_id"].encode())
        h.update(c["text"].encode())
    return h.hexdigest()


def load_chunks(path: Path) -> list[dict]:
    return read_jsonl(path, missing_ok=False)


def embed_texts(model, tokenizer, texts: list[str], batch_size: int = 32) -> np.ndarray:
    generate, _ = _mlx()
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        output = generate(model, tokenizer, texts=batch)
        vectors.append(np.array(output.text_embeds))
    return np.concatenate(vectors, axis=0)


def build_index(chunks_path: Path, index_path: Path) -> None:
    _, load = _mlx()
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
        chunks_fingerprint=fingerprint(chunks),
    )
    print(f"embedded {len(chunks)} chunks ({embeddings.shape[1]}-dim) -> {index_path}")


REBUILD = "rebuild with `python scripts/rag.py index`"


def load_index(index_path: Path = INDEX_PATH) -> tuple[np.ndarray, list[str], str | None, str | None]:
    """Load the vector store and check its SHAPE. No model, no pickle.

    `allow_pickle=True` is gone. It was never needed -- this index holds a
    float array, a string array and two scalar strings, all of which load
    natively -- and `np.load` with pickles enabled executes arbitrary code
    from the file while it is being parsed. The index is a build artifact
    that sits in `data/processed/` and is rebuilt by a script, so nothing was
    exploiting it; it was simply a loaded gun pointed at the retrieval path,
    which is now the CHAT SURFACE's retrieval path as well as the eval one.
    Anything an index legitimately contains still loads. Anything that needs a
    pickle to load is not an index this repo built.

    The structural checks exist because the failure they catch is silent.
    Chunk ids here are POSITIONAL, so a mismatched array does not raise -- it
    resolves to the wrong text and retrieval quietly returns chunks that were
    never embedded, which is the same class of bug `fingerprint` was written
    for one level down.

    Returned rather than raised into the caller's face as a tuple so
    `retrieve` can do the corpus-dependent half; split out as its own function
    so the shape checks are testable without Apple Silicon.
    """
    if not index_path.exists():
        raise FileNotFoundError(f"{index_path} not found — run `python scripts/rag.py index` first")

    # Pickle-free. A crafted file that needs pickles now fails to load instead
    # of executing on load.
    stored = np.load(index_path)

    for key in ("embeddings", "chunk_ids"):
        if key not in stored:
            raise ValueError(f"{index_path} has no {key!r} array — {REBUILD}")

    embeddings = stored["embeddings"]
    chunk_ids = stored["chunk_ids"]

    if embeddings.ndim != 2:
        raise ValueError(
            f"{index_path} embeddings are {embeddings.ndim}-D, expected 2-D "
            f"(rows x dims) — {REBUILD}")
    if not np.issubdtype(embeddings.dtype, np.floating):
        raise ValueError(
            f"{index_path} embeddings are {embeddings.dtype}, expected a float "
            f"array — {REBUILD}")
    if chunk_ids.ndim != 1:
        raise ValueError(
            f"{index_path} chunk_ids is {chunk_ids.ndim}-D, expected 1-D — {REBUILD}")
    if embeddings.shape[0] != chunk_ids.shape[0]:
        # The silent one: positional ids against the wrong number of vectors.
        raise ValueError(
            f"{index_path} has {embeddings.shape[0]} embedding rows and "
            f"{chunk_ids.shape[0]} chunk ids. Ids are positional, so this "
            f"resolves to the WRONG text rather than failing — {REBUILD}")
    if embeddings.shape[0] == 0:
        raise ValueError(f"{index_path} is empty — {REBUILD}")
    if not np.isfinite(embeddings).all():
        raise ValueError(
            f"{index_path} contains NaN or infinite embeddings; every cosine "
            f"score against them would be NaN and `argsort` would rank them "
            f"arbitrarily — {REBUILD}")
    norms = np.linalg.norm(embeddings, axis=1)
    if not (norms > 0).all():
        raise ValueError(
            f"{index_path} contains {int((norms <= 0).sum())} zero-length "
            f"embedding(s); cosine similarity is undefined for them — {REBUILD}")

    stored_model = str(stored["model_id"]) if "model_id" in stored else None
    stored_fp = (str(stored["chunks_fingerprint"])
                 if "chunks_fingerprint" in stored else None)
    return embeddings, [str(c) for c in chunk_ids], stored_model, stored_fp


def retrieve(
    query: str, k: int, chunks_path: Path = CHUNKS_PATH, index_path: Path = INDEX_PATH,
    model_and_tokenizer: tuple | None = None,
) -> list[dict]:
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise ValueError(f"k must be an integer, got {k!r}")
    if k < 0:
        # `argsort(...)[:k]` with a negative k silently returns all but the
        # last |k| results instead of failing, i.e. the WORST matches.
        raise ValueError(f"k must be >= 0, got {k}")

    embeddings, chunk_ids, stored_model, stored_fp = load_index(index_path)
    chunks = load_chunks(chunks_path)

    if stored_model and stored_model != MODEL_ID:
        raise ValueError(
            f"{index_path} was built with {stored_model}, but this run uses {MODEL_ID}. "
            f"Vectors from different models are not comparable — rebuild with "
            f"`python scripts/rag.py index`."
        )
    if stored_fp and stored_fp != fingerprint(chunks):
        raise ValueError(
            f"{index_path} is stale: {chunks_path} has changed since it was embedded. "
            f"Rebuild with `python scripts/rag.py index`."
        )

    # Every stored id must name a chunk that still exists. The fingerprint
    # above catches a CHANGED corpus, but it is skipped entirely on an index
    # built before that field existed, and this is the check that keeps the
    # lookup below from a KeyError -- or, worse, from resolving against a
    # corpus the vectors do not describe.
    chunks_by_id = {c["chunk_id"]: c for c in chunks}
    missing = [cid for cid in chunk_ids if cid not in chunks_by_id]
    if missing:
        raise ValueError(
            f"{index_path} names {len(missing)} chunk id(s) that are not in "
            f"{chunks_path} (first: {missing[0]!r}) — {REBUILD}")

    # Loading the embedding model takes real time, so callers doing many
    # retrievals in a loop (e.g. build_reddit_eval.py) should load it once
    # and pass it in rather than paying that cost on every call.
    _, load = _mlx()
    model, tokenizer = model_and_tokenizer if model_and_tokenizer else load(MODEL_ID)
    query_vec = embed_texts(model, tokenizer, [query])[0]
    query_norm = np.linalg.norm(query_vec)
    if not np.isfinite(query_norm) or query_norm == 0:
        # A degenerate query vector makes every score NaN or zero, and the
        # ranking that comes back is then arbitrary rather than empty.
        raise ValueError(
            f"the embedding of this query has norm {query_norm!r}; it cannot "
            f"be compared against the index")
    query_vec = query_vec / query_norm

    scores = embeddings @ query_vec  # both sides pre-normalized -> cosine similarity
    # Bounded: asking for more than exists is not an error, but slicing past
    # the end must not be mistaken for having found that many.
    top_k = np.argsort(-scores)[:min(k, len(chunk_ids))]

    return [{**chunks_by_id[chunk_ids[i]], "score": float(scores[i])} for i in top_k]


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
