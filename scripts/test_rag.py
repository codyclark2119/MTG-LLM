"""The vector index's structural validation, without a model.

`rag.retrieve` sits on TWO paths now: the eval harness it was written for, and
the chat surface's rules retrieval through `retrieve_hybrid`. That is why the
index loader is worth testing on its own — a malformed index does not raise
here, it resolves to the wrong text, because chunk ids are positional.

No MLX. `rag` imports `mlx_embeddings` lazily (see `rag._mlx`), so everything
below runs on any machine on synthetic `.npz` files built in a temp directory.
The retrieval arithmetic itself still needs a real embedder and is not in
scope here; what is asserted is that a bad index is REFUSED before it can be
used, and that a good one is accepted.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import rag

FAILED = 0


def check(label: str, got, want) -> None:
    global FAILED
    if got != want:
        print(f"  FAIL {label}\n    got  {got!r}\n    want {want!r}")
        FAILED += 1


def refuses(label: str, path: Path, exc=ValueError, **kw) -> None:
    """`load_index` must raise, and the message must name the rebuild."""
    global FAILED
    try:
        rag.load_index(path, **kw)
    except exc as e:
        if "rebuild" not in str(e).lower():
            print(f"  FAIL {label}: raised, but the message does not say how "
                  f"to fix it: {e}")
            FAILED += 1
        return
    except Exception as e:  # noqa: BLE001 - the point is that it is the RIGHT one
        print(f"  FAIL {label}: raised {type(e).__name__}, wanted {exc.__name__}")
        FAILED += 1
        return
    print(f"  FAIL {label}: accepted a malformed index")
    FAILED += 1


def write_index(tmp: Path, name: str = "idx.npz", **arrays) -> Path:
    path = tmp / name
    np.savez(path, **arrays)
    # np.savez appends .npz when the name lacks it; ours already has it.
    return path


def good_arrays(n: int = 4, dims: int = 3) -> dict:
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(n, dims)).astype(np.float32)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    return {"embeddings": emb,
            "chunk_ids": np.array([f"c{i}" for i in range(n)]),
            "model_id": rag.MODEL_ID,
            "chunks_fingerprint": "deadbeef"}


def test_valid_index_loads() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        path = write_index(tmp, **good_arrays())
        emb, ids, model, fp = rag.load_index(path)
        check("embeddings come back 2-D", emb.ndim, 2)
        check("all four rows", emb.shape[0], 4)
        check("ids come back as str, not numpy scalars",
              [type(i) for i in ids], [str] * 4)
        check("ids are the ones stored", ids, ["c0", "c1", "c2", "c3"])
        check("the model id survives", model, rag.MODEL_ID)
        check("the corpus fingerprint survives", fp, "deadbeef")

        # The two provenance fields are optional on an old index and must not
        # be invented when absent.
        bare = good_arrays()
        del bare["model_id"], bare["chunks_fingerprint"]
        _, _, model2, fp2 = rag.load_index(write_index(tmp, "bare.npz", **bare))
        check("a pre-provenance index reports no model", model2, None)
        check("...and no fingerprint", fp2, None)


def test_malformed_indexes_are_refused() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        missing = tmp / "nope.npz"
        raised = None
        try:
            rag.load_index(missing)
        except FileNotFoundError as e:
            raised = e
        check("a missing index is a FileNotFoundError", raised is not None, True)

        a = good_arrays(); del a["embeddings"]
        refuses("no embeddings array", write_index(tmp, "a.npz", **a))

        a = good_arrays(); del a["chunk_ids"]
        refuses("no chunk_ids array", write_index(tmp, "b.npz", **a))

        # THE SILENT ONE. Ids are positional, so a count mismatch resolves to
        # the wrong text rather than failing.
        a = good_arrays()
        a["chunk_ids"] = np.array(["c0", "c1"])
        refuses("more embedding rows than chunk ids", write_index(tmp, "c.npz", **a))

        a = good_arrays()
        a["embeddings"] = a["embeddings"][0]           # 1-D
        refuses("1-D embeddings", write_index(tmp, "d.npz", **a))

        a = good_arrays()
        a["embeddings"] = a["embeddings"][:, :, None]  # 3-D
        a["chunk_ids"] = np.array(["c0", "c1", "c2", "c3"])
        refuses("3-D embeddings", write_index(tmp, "e.npz", **a))

        a = good_arrays()
        a["embeddings"] = np.array([["a", "b"], ["c", "d"]])
        a["chunk_ids"] = np.array(["c0", "c1"])
        refuses("string embeddings", write_index(tmp, "f.npz", **a))

        a = good_arrays()
        a["embeddings"] = np.zeros((0, 3), dtype=np.float32)
        a["chunk_ids"] = np.array([], dtype="<U2")
        refuses("an empty index", write_index(tmp, "g.npz", **a))

        for label, bad in (("NaN", np.nan), ("+inf", np.inf), ("-inf", -np.inf)):
            a = good_arrays()
            a["embeddings"] = a["embeddings"].copy()
            a["embeddings"][1, 0] = bad
            refuses(f"{label} in the embeddings", write_index(tmp, "h.npz", **a))

        # Zero-length vector: cosine similarity is undefined, and `argsort`
        # would rank the resulting NaN arbitrarily rather than dropping it.
        a = good_arrays()
        a["embeddings"] = a["embeddings"].copy()
        a["embeddings"][2] = 0.0
        refuses("a zero-length embedding", write_index(tmp, "i.npz", **a))


def test_an_index_needing_pickle_is_refused() -> None:
    """The headline of this change: `allow_pickle=True` is gone.

    `np.load(..., allow_pickle=True)` executes arbitrary code out of the file
    while parsing it. Nothing was exploiting it — the index is a local build
    artifact — but it sat on the retrieval path, which is now the chat
    surface's path as well as the eval harness's. The real index never needed
    it: it holds a float array, a string array and two scalar strings.

    So the test is in two halves: a file that REQUIRES a pickle must fail to
    load, and the repository's own index must still load without one.
    """
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        a = good_arrays()
        # An object array can only be serialised by pickling it.
        a["chunk_ids"] = np.array([{"not": "a chunk id"}] * 4, dtype=object)
        path = tmp / "pickled.npz"
        np.save(tmp / "obj.npy", a["chunk_ids"], allow_pickle=True)
        np.savez(path, **a)

        raised = None
        try:
            rag.load_index(path)
        except Exception as e:  # noqa: BLE001
            raised = e
        check("an index that needs a pickle to load is refused",
              raised is not None, True)
        check("...and it is refused by numpy's pickle guard, not by chance",
              "pickle" in str(raised).lower() if raised else "", True)

    # And the real one still loads, pickle-free.
    from common import INDEX_PATH
    if INDEX_PATH.exists():
        emb, ids, model, fp = rag.load_index(INDEX_PATH)
        check("the repository's own index loads without pickles",
              emb.ndim == 2 and emb.shape[0] == len(ids), True)
        check("...and it is the model this code expects", model, rag.MODEL_ID)
    else:
        print(f"  note: {INDEX_PATH} absent, skipped the real-index half")


def test_retrieve_validates_k_before_touching_a_model() -> None:
    """A bad k must be refused, and refused before an embedder is loaded.

    `np.argsort(-scores)[:k]` with a negative k does not fail — it returns all
    but the last |k| results, i.e. quietly drops the BEST matches while looking
    like a successful retrieval. That these raise without MLX present is itself
    the assertion: the checks run before `rag._mlx()`.
    """
    for bad in (-1, -5):
        raised = None
        try:
            rag.retrieve("q", bad)
        except ValueError as e:
            raised = e
        check(f"k={bad} is refused", raised is not None, True)
        check(f"...and the message names k (k={bad})",
              "k must be >= 0" in str(raised) if raised else "", True)

    for bad in (1.5, "3", None, True):
        raised = None
        try:
            rag.retrieve("q", bad)
        except ValueError as e:
            raised = e
        check(f"k={bad!r} is refused as a non-integer", raised is not None, True)


def test_ids_must_resolve_to_the_current_corpus() -> None:
    """An id that names no chunk is an error, not a KeyError at the last line.

    Reached without a model: the corpus check runs before `rag._mlx()`.
    """
    import json

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        a = good_arrays()
        del a["chunks_fingerprint"]      # so the stale-corpus check does not fire first
        index = write_index(tmp, "idx.npz", **a)

        chunks = tmp / "chunks.jsonl"
        # Only two of the four ids exist in the corpus.
        chunks.write_text("".join(
            json.dumps({"chunk_id": cid, "text": "t", "section_title": "s"}) + "\n"
            for cid in ("c0", "c1")))

        raised = None
        try:
            rag.retrieve("q", 2, chunks_path=chunks, index_path=index)
        except ValueError as e:
            raised = e
        check("an id that names no chunk is refused", raised is not None, True)
        check("...and the message says how many and names one",
              "2 chunk id(s)" in str(raised) if raised else "", True)

        # A mismatched MODEL id is still caught, unchanged by this work.
        b = good_arrays()
        b["model_id"] = "mlx-community/some-other-embedder"
        del b["chunks_fingerprint"]
        other = write_index(tmp, "other.npz", **b)
        raised = None
        try:
            rag.retrieve("q", 2, chunks_path=chunks, index_path=other)
        except ValueError as e:
            raised = e
        check("an index from another embedder is still refused",
              "not comparable" in str(raised) if raised else "", True)


def main() -> None:
    test_valid_index_loads()
    test_malformed_indexes_are_refused()
    test_an_index_needing_pickle_is_refused()
    test_retrieve_validates_k_before_touching_a_model()
    test_ids_must_resolve_to_the_current_corpus()
    if FAILED:
        print(f"\n{FAILED} check(s) failed")
        raise SystemExit(1)
    print("index structural validation passed (no model loaded)")


if __name__ == "__main__":
    main()
