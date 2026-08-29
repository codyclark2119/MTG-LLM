"""Turn recorded XMage board snapshots into position DRAFTS.

WHY THIS EXISTS

Every board in `positions.jsonl` is machine-drafted (Section 21.98), and that
qualifies the entire gameplay track — Gates 1-3, blunder rate, the arm ranking,
37% protocol precision, every judge-vs-human kappa. A rules engine then found
the boards are mechanically sound far more often than feared (30 of 32,
Section 21.101), which settles legality and leaves the real objection standing:
a generated board has no reason to be a board anyone would ever face.

Boards that arose in a played game do. `MagicLlmPositionDataCollector` (in the
XMage checkout) writes one JSON snapshot per step of a real game; this reads
them and emits position drafts.

WHAT THIS DOES NOT DO

It does not write the gold set, and it never can — `--out` defaults to a
candidates file. Promotion stays local and reviewed, which is what makes "gold"
mean *a person looked at this* (the rubric server's first invariant).

It does not fill `legal_actions`. The engine can answer that, but the path that
asks it is already built and validated (`xmage_export.py` emits a JUnit test per
board, `xmage_diff.py` reads the answer back, 30 of 32 agree). A second way to
compute one field is how two copies drift, so a draft carries no
`legal_actions` and the existing path supplies them.

It does not fill `key_points` or `common_errors`. Those are the rubric, they are
what a person authors in the form, and a machine-drafted rubric is half of what
21.98 is a caveat about. A draft with an empty rubric is honest; a draft with a
generated one would put the caveat straight back.

SELECTION IS THE HARD PART

A twelve-turn game yields ~150 snapshots and almost none of them are worth
asking about. `--interesting` applies the cheapest defensible filter — a board
where the active player holds cards and has untapped lands, at a step where
something can be done — and it is a *shortlist*, not a judgement. 21.93 is the
standard a board has to meet (one question, ~1.5 plays) and no filter here can
check it.

Usage:
    python scripts/gameplay/import_game.py --snapshots ~/code/mage/Mage.Tests/magicllmPositions
    python scripts/gameplay/import_game.py --snapshots DIR --interesting --limit 20
    python scripts/gameplay/import_game.py --snapshots DIR --as you=PlayerA
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from common import (PHASE_STEPS, POSITIONS_PATH, CR_VERSION,  # noqa: E402
                    PROTOCOL_ERRORS, read_jsonl, render_position,
                    write_jsonl_atomic)

# XMage enum name -> the phase spelling this project stores. Both sides of this
# come from `common.PHASE_STEPS`, which mirrors `mage.constants.PhaseStep`, so
# there is no mapping table to drift (Section 21.102) — the collector writes
# `step.name()` and this reads the same row back.
_STEP_TO_PHASE = {row[0]: row[3] for row in PHASE_STEPS}

# Steps in which no player receives priority, so there is no decision to ask
# about. Untap and cleanup take no actions at all; the damage steps resolve
# without a window in the ordinary case.
_NO_PRIORITY = ("UNTAP", "CLEANUP", "COMBAT_DAMAGE", "FIRST_COMBAT_DAMAGE")


def load_snapshots(root: Path) -> list[dict]:
    """Every snapshot under a recording directory, in file then line order."""
    if root.is_file():
        return read_jsonl(root, missing_ok=True)
    out: list[dict] = []
    for path in sorted(root.rglob("*.jsonl")):
        rows = read_jsonl(path, missing_ok=True)
        for row in rows:
            row.setdefault("_source_file", path.name)
        out.extend(rows)
    return out


def split_records(snaps: list[dict]) -> tuple[list[dict], dict[str, dict]]:
    """(board snapshots, game_id -> end record).

    The collector writes one `record: game_end` line per game alongside the
    boards. It is not a board and must never become a position — it has no
    battlefield, and `to_position` would emit an empty one.
    """
    boards, ends = [], {}
    for s in snaps:
        if s.get("record") == "game_end":
            ends[s.get("game_id")] = s
        else:
            boards.append(s)
    return boards, ends


def end_summary(end: dict) -> str:
    """One line describing how a game finished, for the import report.

    A CLOCK loss is not a game loss (Section 21.118). The boards leading to one
    are real and usable; the outcome is not evidence about the line that was
    played, and a game that ends on the clock stops mid-turn with both players
    alive rather than at a resolution.

    **`winner` is NOT read**, because it is written too early to be true. The
    engine sets it in `findWinnersAndLosers()`, which also calls `player.won()`
    — and the first real end record had `winner: "Game is a draw"` with one
    player dead at -1 life, flagged `lost`, and the survivor NOT flagged `won`.
    That combination is only possible if `onGameEnd` fired before the winner was
    decided (Section 21.120).

    So the outcome is derived from the fields that ARE set by then: `lost`,
    `left`, `quit`, the timeout flags and life totals. Deriving from what is
    true beats reading a field that is merely present.

    **LIFE, not `left`, separates a real loss from an abandoned game.** `left` is
    true for the loser of a played-out game too — a player leaves the table after
    losing — so it discriminates nothing. A loser at positive life did not die.

    Known limit: a loss by DECKING or poison also leaves the loser at positive
    life and would be reported as abandoned here. Nothing in the end record
    separates them; the last board's `library_count` could, and is not read
    (Section 21.121).
    """
    if not end:
        return "no end record — a recording made before the collector wrote one"
    players = end.get("players") or []
    turn = end.get("turn")
    clock = [p["name"] for p in players if p.get("timer_timeout") or p.get("idle_timeout")]
    quit_ = [p["name"] for p in players if p.get("quit")]
    lost = [p["name"] for p in players if p.get("lost")]
    alive = [p["name"] for p in players if not p.get("lost")]

    if clock:
        return (f"ended on the CLOCK at turn {turn} ({', '.join(clock)} timed out)"
                " — boards are real, the OUTCOME is not")
    if quit_:
        return f"ended by concession at turn {turn} ({', '.join(quit_)} quit)"
    if len(lost) == 1 and len(alive) == 1:
        dead = next(p for p in players if p.get("lost"))
        # LIFE is the discriminator, not `left`. Confirmed against three
        # recordings the user could identify: the genuine loss had the loser at
        # -1, the concession at 20, and an ABANDONED game at 19. `left` was true
        # on all three, including the real one, because a player leaves the
        # table after losing normally (Section 21.121).
        if dead.get("life", 0) > 0:
            return (f"ABANDONED at turn {turn} — {lost[0]} is flagged lost at "
                    f"{dead['life']} life, having taken no lethal damage. Not a "
                    "played-out result; the boards are real, the outcome is not.")
        return (f"played out to turn {turn} — {alive[0]} won, {lost[0]} lost "
                f"at {dead['life']} life")
    if not lost:
        return f"ended at turn {turn} with nobody eliminated — outcome unclear"
    return f"ended at turn {turn}; lost: {', '.join(lost)}"


def to_position(snap: dict, sides: dict[str, str], index: int) -> dict:
    """One snapshot -> one position draft in this project's schema.

    `sides` maps an XMage player name to `you` or `opp`. Which player the
    position is asked FROM is a choice, not a fact about the snapshot: the
    recording holds both hands, and a position states one hand and the other as
    a count.
    """
    you_name = next((n for n, s in sides.items() if s == "you"), None)

    players: dict[str, dict] = {}
    for p in snap.get("players") or []:
        side = sides.get(p.get("name"))
        if side is None:
            continue
        info: dict = {"life": p.get("life"), "library_count": p.get("library_count")}
        if side == "you":
            info["hand"] = list(p.get("hand") or [])
        else:
            # The opponent's hand is recorded and deliberately reduced to a
            # count here. The board is what the answering player can see, and a
            # position that hands over the opponent's cards is asking a
            # different question than the one a player faces.
            info["hand_count"] = len(p.get("hand") or [])
        players[side] = info

    battlefield = []
    for perm in snap.get("battlefield") or []:
        side = sides.get(perm.get("controller"))
        if side is None:
            continue
        entry: dict = {"controller": side, "card": perm.get("card")}
        # A token is carried through as a permanent and marked, because it IS on
        # the board and the answer may depend on it — but it is not a card, and
        # `validate_position` checks every name against Oracle. The first real
        # recording produced "Treasure Token", "Hero Token" and "Everywhere";
        # the third is the land token Overlord of the Hauntwoods creates, which
        # is why detecting tokens by the word "Token" in the name does not work
        # and the engine's own `isToken()` is recorded instead (Section 21.106).
        if perm.get("token"):
            entry["token"] = True
            # The characteristics needed to REBUILD it, since a token cannot be
            # looked up by name (796 token classes, five of them Treefolk).
            # `token_rules` rides along so the export can tell a vanilla token —
            # rebuildable exactly — from one whose abilities it would silently
            # drop (Section 21.113).
            for k in ("token_types", "token_subtypes", "token_colors", "token_rules"):
                if perm.get(k):
                    entry[k] = list(perm[k])
        # State `addCard(name)` cannot reproduce, carried so the export can
        # REFUSE the board rather than rebuild it wrong (Section 21.116).
        for k in ("counters", "attachments", "chosen"):
            if perm.get(k):
                entry[k] = perm[k]
        for k in ("face_down", "transformed"):
            if perm.get(k):
                entry[k] = True
        if perm.get("phased_in") is False:
            entry["phased_in"] = False
        if perm.get("tapped"):
            entry["tapped"] = True
        if perm.get("attacking"):
            entry["attacking"] = True
        if perm.get("sick"):
            entry["sick"] = True
        # A land has no meaningful P/T and the collector reports 0/0 for one.
        # Carried only when non-zero, so a rendered board does not claim a
        # Mountain is a 0/0.
        if perm.get("power") or perm.get("toughness"):
            entry["pt"] = f"{perm.get('power')}/{perm.get('toughness')}"
        battlefield.append(entry)

    graveyards = {}
    for p in snap.get("players") or []:
        side = sides.get(p.get("name"))
        if side and p.get("graveyard"):
            graveyards[side] = list(p["graveyard"])

    step = snap.get("step") or ""
    return {
        "id": f"pos-recorded-{snap.get('game_id', '')[:8]}-{index:04d}",
        # Deliberately unset. A category is a claim about what the board ASKS,
        # and nothing here has looked at that; `positions.py` rejects an empty
        # one, which is the intended outcome for an unreviewed draft.
        "category": "",
        "difficulty": "",
        "format": "modern",
        "turn": snap.get("turn"),
        "phase": _STEP_TO_PHASE.get(step, step.lower()),
        "active_player": "you" if snap.get("active_player") == you_name else "opp",
        "priority": "you" if snap.get("priority_player") == you_name else "opp",
        "players": players,
        "battlefield": battlefield,
        "graveyards": graveyards,
        "key_points": [],
        "common_errors": [],
        "rule_citations": [],
        "cr_version": CR_VERSION,
        "source": "xmage-recording",
        "rubric_source": "unauthored",
        # What the draft came from, so a promoted position can be traced to the
        # game and step it was taken from.
        "recording": {"game_id": snap.get("game_id"), "step": step,
                      "file": snap.get("_source_file", "")},
    }


def is_interesting(snap: dict, you_name: str) -> bool:
    """A cheap shortlist filter. NOT a judgement that the board asks a question.

    Three conditions, each of which only removes boards that certainly have
    nothing to ask: a step with no priority, an empty hand, and no untapped
    land. A board passing all three may still be trivial — 21.93's standard
    (one question, ~1.5 plays) is not machine-checkable, and pretending it is
    would put a generator back in the loop under a different name.
    """
    if (snap.get("step") or "") in _NO_PRIORITY:
        return False
    you = next((p for p in snap.get("players") or []
                if p.get("name") == you_name), None)
    if not you or not (you.get("hand") or []):
        return False
    untapped = [p for p in snap.get("battlefield") or []
                if p.get("controller") == you_name and not p.get("tapped")]
    return bool(untapped)



def export_rubric_tasks(drafts: list[dict], out_path: Path) -> int:
    """Write recorded boards as tasks the DEPLOYED rubric form can author.

    The form already exists. `rubric_server.py`'s `/rubric` view collects
    `key_points` and `common_errors` against a `question`, and its CSS is
    already `pre-wrap`, so a rendered board displays as a board. Reusing it
    beats a fourth authoring view: it is deployed, it has been used, and its
    submission path, attribution and duplicate handling are all tested.

    So a board becomes a task by RENDERING it into `question` — the same
    `render_position` the model reads, which means the author and the model see
    the same board and a rubric cannot be written against a different one.

    `draft` is deliberately EMPTY. The rules flow ships a machine sentence-split
    of the answer to argue with; a recorded board has no answer yet, and 14.6
    measured machine drafts at r = +0.30 against hand-written +0.62. Shipping a
    generated draft here would put 21.98's caveat straight back into the one
    part of the pipeline that was going to escape it.

    Self-contained, like every other task file: no gold set, no candidates, no
    corpora (Section 21.125).

    EACH TASK DECLARES ITSELF A POSITION (`kind: "position"`), because reusing
    the rules form silently handed the author the wrong three things (21.126).
    A rules task has a verified answer to write a rubric against; a board has
    none, so the "this is correct, do not re-adjudicate" card asserted an empty
    string was correct. A rules `key_points` is a set of claims; a position's is
    the correct LINE. And a position's `common_errors` is unioned with
    `PROTOCOL_ERRORS` before either the judge or the human form sees it
    (`eval_positions` line 552, `adjudicate.task_for`), so an author who cannot
    see those ten entries writes an eleventh that duplicates one — and since the
    judge returns error NUMBERS, the duplicate lands in the STRATEGY half of the
    split `score_run` reads, where no parser confirms it. 21.75 was the judge
    having a rubric the form lacked; this is the form needing a rubric the judge
    already has.

    They ride ON THE TASK rather than being imported by the form, so the file
    stays the self-contained thing `rubric_server` is built to read, and a task
    exported today keeps showing the ten entries it was authored against even
    after `PROTOCOL_ERRORS` grows.
    """
    tasks = []
    for d in drafts:
        tasks.append({
            "id": d["id"],
            # The author sets these; the board cannot state what it ASKS.
            "category": d.get("category") or "",
            "difficulty": d.get("difficulty") or "",
            "question": render_position(d),
            "answer": "",
            "draft": [],
            "url": "",
            "in_gold": False,
            "card_slots": {},
            "kind": "position",
            "protocol_errors": list(PROTOCOL_ERRORS),
            # Shown so the author knows whether the board is even scorable yet.
            # A board with no list scores every correct play ILLEGAL and fires
            # protocol entry 3 (verified by running, 21.126) — the rubric is
            # still valid, because `render_position` ignores the field, but the
            # board is not promotable until the engine path fills it in.
            "legal_actions": list(d.get("legal_actions") or []),
        })
    # The FILE kind stays "rubric" and the TASK kind is "position". They answer
    # different questions: the file's picks which form and which submission
    # shape (a position rubric is collected by `/rubric` exactly as a rules one
    # is), the task's picks the framing inside that form. Declaring the file a
    # position made `load_tasks` refuse it outright — caught by running the
    # server, not by `--help`, which is the whole reason that rule exists.
    payload = {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "kind": "rubric", "count": len(tasks), "tasks": tasks}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    return len(tasks)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshots", type=Path,
                    help="directory the collector wrote, or one .jsonl file")
    ap.add_argument("--export-from", type=Path, metavar="CANDIDATES",
                    help="re-export tasks.json from an EXISTING candidates file "
                         "instead of re-deriving from snapshots. Use this whenever "
                         "the form changes under work already in progress: ids stay "
                         "byte-identical, so submissions still key (Section 21.126). "
                         "Re-deriving cannot promise that — --as and --interesting "
                         "both change which boards get which number.")
    ap.add_argument("--out", type=Path,
                    default=POSITIONS_PATH.parent / "position_candidates_recorded.jsonl",
                    help="candidates file; NEVER the gold set")
    ap.add_argument("--as", dest="as_side", default=None,
                    help="which XMage player is `you`, e.g. you=PlayerA "
                         "(default: the first player by name)")
    ap.add_argument("--interesting", action="store_true",
                    help="shortlist only — see is_interesting, it is not a judgement")
    ap.add_argument("--limit", type=int, default=0, help="keep at most N drafts")
    ap.add_argument("--export-rubric-tasks", type=Path, metavar="OUT",
                    help="write the drafts as a tasks.json the deployed "
                         "rubric form can author key_points/common_errors against")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # Re-export only. Reads what was already written, touches no snapshot, and
    # writes no candidates file — the boards are not being re-derived, only the
    # form's copy of them refreshed.
    if args.export_from:
        if not args.export_rubric_tasks:
            raise SystemExit("--export-from needs --export-rubric-tasks OUT")
        drafts = read_jsonl(args.export_from)
        if not drafts:
            raise SystemExit(f"no candidates in {args.export_from}")
        n = export_rubric_tasks(drafts, args.export_rubric_tasks)
        have = sum(1 for d in drafts if d.get("legal_actions"))
        print(f"{n} board(s) re-exported -> {args.export_rubric_tasks}")
        print(f"  ids unchanged, so submissions already collected still key")
        print(f"  legal_actions present on {have}/{n}"
              + ("" if have == n else "  <- the rest cannot be SCORED yet; "
                                      "the rubric work is still valid"))
        return

    if not args.snapshots:
        raise SystemExit("--snapshots is required (or --export-from to re-export)")
    snaps = load_snapshots(args.snapshots)
    if not snaps:
        raise SystemExit(f"no snapshots under {args.snapshots}")
    snaps, ends = split_records(snaps)
    if not snaps:
        raise SystemExit(f"only end records under {args.snapshots}, no boards")

    names = sorted({p.get("name") for s in snaps for p in (s.get("players") or [])
                    if p.get("name")})
    if args.as_side:
        if "=" not in args.as_side:
            raise SystemExit("--as takes you=<PlayerName>")
        _, you_name = args.as_side.split("=", 1)
        if you_name not in names:
            raise SystemExit(f"{you_name!r} is not in this recording: {names}")
    else:
        you_name = names[0] if names else ""
    sides = {n: ("you" if n == you_name else "opp") for n in names}

    print(f"{len(snaps)} snapshots from {args.snapshots}")
    for gid in sorted({s.get("game_id") for s in snaps}):
        print(f"  game {str(gid)[:8]}: {end_summary(ends.get(gid))}")
    print(f"  players: {', '.join(f'{n} -> {s}' for n, s in sides.items())}")

    kept = [s for s in snaps if not args.interesting or is_interesting(s, you_name)]
    if args.interesting:
        print(f"  {len(kept)}/{len(snaps)} pass the shortlist filter "
              "(a shortlist, not a judgement — see the docstring)")
    if args.limit:
        kept = kept[:args.limit]

    drafts = [to_position(s, sides, i) for i, s in enumerate(kept, 1)]

    by_phase: dict[str, int] = {}
    for d in drafts:
        by_phase[d["phase"]] = by_phase.get(d["phase"], 0) + 1
    print(f"\n{len(drafts)} drafts")
    for phase, n in sorted(by_phase.items(), key=lambda kv: -kv[1]):
        print(f"  {n:4}  {phase}")
    print("\nEvery draft is missing, on purpose:")
    print("  legal_actions  -> xmage_export.py + xmage_diff.py (the validated path)")
    print("  key_points / common_errors -> authored in the rubric form")
    print("  category / difficulty      -> a claim about what the board asks")

    if args.export_rubric_tasks:
        n = export_rubric_tasks(drafts, args.export_rubric_tasks)
        print(f"\n{n} board(s) -> {args.export_rubric_tasks}")
        print("Serve with: python scripts/rubric_server.py --tasks "
              f"{args.export_rubric_tasks}  then open #/rubric")
    if args.dry_run:
        print(f"\n(dry run — {args.out} not written)")
        return
    write_jsonl_atomic(args.out, drafts)
    print(f"\n-> {args.out}")
    print("This is a CANDIDATES file. Nothing promotes it to the gold set but a "
          "person reviewing it.")


if __name__ == "__main__":
    main()
