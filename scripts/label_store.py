"""Data layer for the gold-set web UI: load, validate, and write records.

Split out of the server so the labeling view, the new-record form, and any
future consumer share one implementation of "what a valid gold record is"
and one atomic writer. The rules that matter live here rather than in a
request handler: at least two key points, a category from the Section 7.1
vocabulary, citations that resolve against the pinned CR, and card names
that resolve against the Oracle pool.
"""

import json
import os
import shutil
import sys
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import (CR_VERSION, GOLD_CANDIDATES_PATH, GOLD_PATH, RULES_PATH,
                    is_hand_authored, read_jsonl, write_jsonl_atomic)

CANDIDATES_PATH = GOLD_CANDIDATES_PATH
REJECTED_PATH = Path("data/gold/rejected.jsonl")

# Re-exported, not redefined: one vocabulary, in common.py (Section 21.97).
# `webui` imports them from here and keeps working.
from common import CATEGORIES, DIFFICULTIES  # noqa: E402,F401
DIFFICULTY_RANK = {"advanced": 0, "intermediate": 1, "basic": 2}


# read_jsonl / write_jsonl_atomic now live in common.py — see the note there
# on the five divergent copies this replaced.


class Store:
    def __init__(self, gold_path: Path, candidates_path: Path, rules_path: Path,
                 category: str | None, skip_cards: bool):
        self.lock = threading.Lock()
        self.gold_path = gold_path
        self.category_filter = category

        self.gold = read_jsonl(gold_path)
        self.candidates = read_jsonl(candidates_path)
        self.rejected = {r["id"] for r in read_jsonl(REJECTED_PATH)}
        self.rule_text = {r["rule_id"]: r["text"] for r in read_jsonl(rules_path)}

        self.card_index = None
        if not skip_cards:
            try:
                from card_lookup import CardIndex
                self.card_index = CardIndex()
            except Exception as e:  # card corpus missing -> degrade, don't die
                print(f"  card index unavailable ({e}); card checks disabled")

        self._backed_up = False
        self._reindex()

    # -- indexing ---------------------------------------------------------

    def _reindex(self) -> None:
        self.by_id = {r["id"]: r for r in self.gold}
        self.cand_by_id = {c["id"]: c for c in self.candidates}

    _authored = staticmethod(is_hand_authored)  # one definition, in common.py

    def queue(self) -> list[dict]:
        """Ordered work list: stale gold rubrics first, then thin categories."""
        items = []
        for r in self.gold:
            if not self._authored(r):
                items.append({"id": r["id"], "kind": "fix", "category": r.get("category"),
                              "difficulty": r.get("difficulty"), "done": False})
        fixes = len(items)

        have = Counter(r["category"] for r in self.gold if self._authored(r))
        pending = Counter(i["category"] for i in items)
        pool = [c for c in self.candidates
                if c["id"] not in self.by_id and c["id"] not in self.rejected]
        if self.category_filter:
            pool = [c for c in pool if c["category"] == self.category_filter]

        by_cat: dict[str, list[dict]] = {}
        for c in pool:
            by_cat.setdefault(c["category"], []).append(c)
        for v in by_cat.values():
            v.sort(key=lambda c: (DIFFICULTY_RANK.get(c["difficulty"], 3), c["rulesguru_id"]))

        # Round-robin, always refilling whichever category is furthest behind.
        taken: list[dict] = []
        while any(by_cat.values()):
            cat = min(
                (c for c in by_cat if by_cat[c]),
                key=lambda c: (have[c] + pending[c] + sum(1 for t in taken if t["category"] == c), c),
            )
            rec = by_cat[cat].pop(0)
            taken.append({"id": rec["id"], "kind": "new", "category": rec["category"],
                          "difficulty": rec["difficulty"], "done": False})

        done = [{"id": r["id"], "kind": "done", "category": r.get("category"),
                 "difficulty": r.get("difficulty"), "done": True}
                for r in self.gold if self._authored(r)]
        return items + taken + done, fixes

    def stats(self) -> dict:
        authored = [r for r in self.gold if self._authored(r)]
        return {
            "gold_total": len(self.gold),
            "authored": len(authored),
            "stale": len(self.gold) - len(authored),
            "rejected": len(self.rejected),
            "by_category": {c: sum(1 for r in authored if r.get("category") == c) for c in CATEGORIES},
            "candidates_left": sum(
                1 for c in self.candidates if c["id"] not in self.by_id and c["id"] not in self.rejected
            ),
        }

    # -- record detail ----------------------------------------------------

    def detail(self, rid: str) -> dict | None:
        rec = self.by_id.get(rid) or self.cand_by_id.get(rid)
        if rec is None:
            return None
        in_gold = rid in self.by_id
        cites = list(rec.get("rule_citations") or [])
        return {
            "id": rid,
            "question": rec.get("question", ""),
            "answer": rec.get("answer", ""),
            "draft": (self.cand_by_id.get(rid) or rec).get("key_points") or [],
            "key_points": rec.get("key_points") if in_gold and self._authored(rec) else [],
            "common_errors": rec.get("common_errors") if in_gold and self._authored(rec) else [],
            "category": rec.get("category"),
            "difficulty": rec.get("difficulty"),
            "cards": rec.get("cards") or [],
            "rule_citations": cites,
            "unresolved": rec.get("rule_citations_unresolved") or [],
            "url": rec.get("rulesguru_url", ""),
            "tags": rec.get("rulesguru_tags") or [],
            "level": rec.get("rulesguru_level"),
            "complexity": rec.get("rulesguru_complexity"),
            "notes": rec.get("notes", ""),
            "authored": in_gold and self._authored(rec),
            "in_gold": in_gold,
            "rules": self.check_rules(cites),
            "card_checks": self.check_cards(rec.get("cards") or []),
        }

    def check_rules(self, ids: list[str]) -> list[dict]:
        out = []
        for rid in ids:
            text = self.rule_text.get(rid)
            out.append({"id": rid, "ok": text is not None,
                        "text": text or "does not resolve against the pinned CR"})
        return out

    def check_cards(self, names: list[str]) -> list[dict]:
        if self.card_index is None:
            return [{"name": n, "ok": None, "how": "unchecked", "resolved": n} for n in names]
        out = []
        for n in names:
            card, how = self.card_index.resolve(n)
            out.append({"name": n, "ok": card is not None and how == "exact",
                        "how": how, "resolved": card["name"] if card else None,
                        "text": (card or {}).get("text", "")[:400]})
        return out

    # -- writes -----------------------------------------------------------

    def _backup_once(self) -> None:
        if self._backed_up or not self.gold_path.exists():
            self._backed_up = True
            return
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        dest = self.gold_path.with_name(f"{self.gold_path.stem}.backup-{stamp}.jsonl")
        shutil.copy2(self.gold_path, dest)
        print(f"  backed up gold set -> {dest}")
        self._backed_up = True

    def save(self, rid: str, payload: dict, author: str) -> dict:
        kp = [s.strip() for s in payload.get("key_points", []) if s.strip()]
        ce = [s.strip() for s in payload.get("common_errors", []) if s.strip()]
        if len(kp) < 2:
            return {"ok": False, "error": "Need at least 2 key points — one point cannot "
                                          "separate a partly-correct answer from a wrong one."}

        with self.lock:
            self._backup_once()
            base = self.by_id.get(rid) or self.cand_by_id.get(rid)
            if base is None:
                return {"ok": False, "error": f"unknown record {rid}"}

            rec = dict(base)
            rec.update(
                key_points=kp,
                common_errors=ce,
                category=payload.get("category") or rec.get("category"),
                difficulty=payload.get("difficulty") or rec.get("difficulty"),
                rule_citations=[s.strip() for s in payload.get("rule_citations", []) if s.strip()],
                rubric_source=(f"hand-authored ({author}); decomposed from the source's verified "
                               f"answer, not judge-reviewed" if author else
                               "hand-authored; decomposed from the source's verified answer, "
                               "not judge-reviewed"),
                needs_rubric_review=False,
                needs_review=True,
                labeled_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            )
            if payload.get("notes", "").strip():
                rec["notes"] = payload["notes"].strip()

            if rid in self.by_id:
                self.gold[self.gold.index(self.by_id[rid])] = rec
            else:
                self.gold.append(rec)
            write_jsonl_atomic(self.gold_path, self.gold)
            self._reindex()
        return {"ok": True}

    def slug(self, question: str) -> str:
        """Stable id from the question text, de-duplicated against the set."""
        import re as _re

        base = "jq-" + _re.sub(r"[^a-z0-9]+", "-", question.lower()).strip("-")[:44].rstrip("-")
        rid, n = base, 2
        while rid in self.by_id:
            rid = f"{base}-{n}"
            n += 1
        return rid

    def validate_new(self, p: dict) -> list[str]:
        """Everything a machine can check about a hand-entered record.

        Returned as a list so the form can show every problem at once
        rather than making the contributor fix them one round trip at a time.
        """
        problems = []
        if not (p.get("question") or "").strip():
            problems.append("Question is required.")
        if not (p.get("answer") or "").strip():
            problems.append("Answer is required.")

        kp = [s for s in (p.get("key_points") or []) if s.strip()]
        if len(kp) < 2:
            problems.append("At least 2 key points — one point cannot separate a "
                            "partly-correct answer from a wrong one.")
        if p.get("category") not in CATEGORIES:
            problems.append(f"Category must be one of: {', '.join(CATEGORIES)}")
        if p.get("difficulty") not in DIFFICULTIES:
            problems.append(f"Difficulty must be one of: {', '.join(DIFFICULTIES)}")
        if not (p.get("source") or "").strip():
            problems.append("Source is required — e.g. judge:LX, so provenance is never guessed later.")

        cites = [s.strip() for s in (p.get("rule_citations") or []) if s.strip()]
        if not cites:
            problems.append("At least one CR citation — an uncited ruling cannot be "
                            "checked against the pinned rules.")
        bad = [r["id"] for r in self.check_rules(cites) if not r["ok"]]
        if bad:
            problems.append(f"These rule ids do not resolve against the pinned CR: {', '.join(bad)}")

        unresolved = [c["name"] for c in self.check_cards(
            [s.strip() for s in (p.get("cards") or []) if s.strip()]) if c["ok"] is False]
        if unresolved:
            problems.append(f"Card names did not resolve exactly: {', '.join(unresolved)}")

        rid = (p.get("id") or "").strip()
        if rid and rid in self.by_id:
            problems.append(f"Id {rid} already exists.")
        return problems

    def create(self, p: dict, author: str) -> dict:
        problems = self.validate_new(p)
        if problems:
            return {"ok": False, "problems": problems}

        clean = lambda k: [s.strip() for s in (p.get(k) or []) if s.strip()]  # noqa: E731
        with self.lock:
            self._backup_once()
            rid = (p.get("id") or "").strip() or self.slug(p["question"])
            rec = {
                "id": rid,
                "question": p["question"].strip(),
                "paraphrases": clean("paraphrases"),
                "answer": p["answer"].strip(),
                "key_points": clean("key_points"),
                "common_errors": clean("common_errors"),
                "rule_citations": clean("rule_citations"),
                "cards": clean("cards"),
                "category": p["category"],
                "difficulty": p["difficulty"],
                "source": p["source"].strip(),
                "cr_version": CR_VERSION,
                # Hand-entered records are authored by definition — there is no
                # machine draft behind them to be mistaken for a rubric.
                "rubric_source": f"hand-authored ({author})" if author else "hand-authored",
                "needs_rubric_review": False,
                "needs_review": True,
                "labeled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            for opt in ("format_context", "notes", "verified_by"):
                if (p.get(opt) or "").strip():
                    rec[opt] = p[opt].strip()

            self.gold.append(rec)
            write_jsonl_atomic(self.gold_path, self.gold)
            self._reindex()
        return {"ok": True, "id": rid}

    def reject(self, rid: str, reason: str) -> dict:
        with self.lock:
            rows = read_jsonl(REJECTED_PATH)
            if rid not in {r["id"] for r in rows}:
                rows.append({"id": rid, "reason": reason,
                             "at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
                write_jsonl_atomic(REJECTED_PATH, rows)
            self.rejected.add(rid)
        return {"ok": True}
