"""Load and validate board positions.

A position is a gold record that happens to describe a board instead of a
question. It carries the same rubric fields — `key_points` for the correct
line, `common_errors` for the blunders — so `eval.py`'s V3 rubric judge scores
it with no changes at all: `score_one_question` routes on `key_points`, and
`rubric_correctness` already returns `errors_made`, which IS the blunder
signal.

What positions need that questions don't is structural validation. A rules
question is prose and can only be wrong; a position can be *malformed* — a
card that doesn't exist, a legal action that won't parse, a controller that
isn't a player — and every one of those silently corrupts an eval run rather
than showing up as a bad score.

    python scripts/gameplay/positions.py                    # validate the set
    python scripts/gameplay/positions.py --render pos-0001  # see what the model sees
    python scripts/gameplay/positions.py --ingest drafts.jsonl --dry-run
    python scripts/gameplay/positions.py --ingest drafts.jsonl --author "Cody Clark"
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from actions import Action, parse_line  # noqa: E402
from common import (  # noqa: E402,F401  (lint_common_errors re-exported for webui)
    CR_VERSION,
    POSITIONS_PATH,
    build_position_messages,
    lint_common_errors,
    read_jsonl,
    render_position,
    write_jsonl_atomic,
)

# Kept separate from label_store.CATEGORIES on purpose. Those eight drive
# stratified sampling and validation for the rules gold set, and an in-flight
# n~100 comparison depends on that set's composition — adding gameplay
# categories there would mix positions into it and corrupt the measurement.
POSITION_CATEGORIES = [
    "mulligan", "land sequencing", "combat math", "blocking",
    "removal timing", "trigger ordering", "race vs stabilize",
]
DIFFICULTIES = ["basic", "intermediate", "advanced"]
PLAYERS = ("you", "opp")


def load_positions(path: Path = POSITIONS_PATH) -> list[dict]:
    return read_jsonl(path)


def position_card_names(pos: dict) -> list[str]:
    """Every card name a position references, for Oracle validation.

    Stack entries are free text ("Lightning Strike (targeting Bears)") so only
    the leading name is taken; everything else is a plain name field.
    """
    names: list[str] = []
    for p in pos.get("battlefield") or []:
        if p.get("card"):
            names.append(p["card"])
        names.extend(p.get("attachments") or [])
    names.extend(pos["players"]["you"].get("hand") or [])
    for zone in ("graveyards", "exile"):
        for side in PLAYERS:
            names.extend((pos.get(zone) or {}).get(side) or [])
    for entry in pos.get("stack") or []:
        names.append(entry.split("(")[0].strip())
    return [n for n in (x.strip() for x in names) if n]


# A permanent as a human types it on a phone:
#   "Mountain x3"                     three untapped Mountains
#   "Grizzly Bears 2/2 tapped"        with current power/toughness
#   "Serra Angel 4/4 tapped attacking"
#   "Llanowar Elves 1/1 sick"
# Flags are suffix words so the card name stays at the front where it reads
# naturally and where name validation can find it.
_PERM_FLAGS = {"tapped": "tapped", "attacking": "attacking",
               "sick": "summoning_sick", "summoning-sick": "summoning_sick"}
_PT_RE = re.compile(r"^\d+/\d+$")
_COUNT_RE = re.compile(r"^x(\d+)$", re.I)


def parse_permanent_line(line: str, controller: str) -> list[dict]:
    """Parse one battlefield line into permanents. Returns [] for a blank line."""
    tokens = line.replace(",", " ").split()
    if not tokens:
        return []
    count, pt, flags, name_parts = 1, None, {}, []
    for tok in tokens:
        low = tok.lower()
        if (m := _COUNT_RE.match(tok)):
            count = max(1, min(int(m.group(1)), 40))
        elif _PT_RE.match(tok):
            pt = tok
        elif low in _PERM_FLAGS:
            flags[_PERM_FLAGS[low]] = True
        else:
            name_parts.append(tok)
    name = " ".join(name_parts).strip()
    if not name:
        return []
    out = []
    for _ in range(count):
        p = {"controller": controller, "card": name, "tapped": flags.get("tapped", False)}
        if pt:
            p["pt"] = pt
        for key in ("attacking", "summoning_sick"):
            if flags.get(key):
                p[key] = True
        out.append(p)
    return out


def position_from_form(f: dict) -> dict:
    """Turn the authoring form's flat fields into a position record.

    Kept server-side so the browser never constructs the schema: the form can
    change shape without the stored records changing shape, and the same
    function backs both the live preview and the save.
    """
    def rows(key):
        return [s.strip() for s in (f.get(key) or "").splitlines() if s.strip()]

    def num(key, default=0):
        try:
            return int(str(f.get(key, "")).strip() or default)
        except ValueError:
            return default

    battlefield = []
    for line in rows("you_battlefield"):
        battlefield.extend(parse_permanent_line(line, "you"))
    for line in rows("opp_battlefield"):
        battlefield.extend(parse_permanent_line(line, "opp"))

    pos = {
        "id": (f.get("id") or "").strip(),
        "format": (f.get("format") or "standard").strip(),
        "turn": num("turn"),
        "phase": (f.get("phase") or "").strip(),
        "active_player": f.get("active_player") or "you",
        "priority": f.get("priority") or "you",
        "players": {
            "you": {"life": num("you_life", 20), "hand": rows("you_hand"),
                    "library_count": num("you_library", 0)},
            "opp": {"life": num("opp_life", 20), "hand_count": num("opp_hand_count", 0),
                    "library_count": num("opp_library", 0)},
        },
        "battlefield": battlefield,
        "graveyards": {"you": rows("you_graveyard"), "opp": rows("opp_graveyard")},
        "stack": rows("stack"),
        "known_information": rows("known_information"),
        "legal_actions": rows("legal_actions"),
        "answer": (f.get("answer") or "").strip(),
        "key_points": rows("key_points"),
        "common_errors": rows("common_errors"),
        "rule_citations": [s for s in (f.get("rule_citations") or "").replace(",", " ").split() if s],
        "category": f.get("category") or "",
        "difficulty": f.get("difficulty") or "",
        "source": (f.get("source") or "").strip(),
        "cr_version": CR_VERSION,
    }
    if f.get("mana_available"):
        pos["mana_available"] = f["mana_available"].strip()
    if f.get("deck_note"):
        pos["deck_note"] = f["deck_note"].strip()
    return pos


def next_position_id(existing: list[dict], category: str) -> str:
    """pos-<category-slug>-NNNN, stable and collision-free."""
    slug = re.sub(r"[^a-z0-9]+", "-", (category or "position").lower()).strip("-")
    taken = {p.get("id") for p in existing}
    n = 1
    while f"pos-{slug}-{n:04d}" in taken:
        n += 1
    return f"pos-{slug}-{n:04d}"


def append_position(pos: dict, path: Path = POSITIONS_PATH) -> None:
    """Append one position, rewriting the file atomically.

    Same reasoning as label_store.write_jsonl_atomic: a half-written file
    costs hand-authored work, and positions are the most expensive records
    in the project to author.
    """
    write_jsonl_atomic(path, load_positions(path) + [pos])


def validate_position(pos: dict, card_index=None,
                      rule_ids: set[str] | None = None) -> list[str]:
    """Everything a machine can check. Returns every problem, not just the first."""
    problems: list[str] = []
    rid = pos.get("id") or "<no id>"

    for field in ("id", "turn", "phase", "players", "answer", "key_points",
                  "category", "difficulty", "source"):
        if not pos.get(field) and pos.get(field) != 0:
            problems.append(f"missing required field: {field}")

    if pos.get("category") not in POSITION_CATEGORIES:
        problems.append(f"category must be one of: {', '.join(POSITION_CATEGORIES)}")
    if pos.get("difficulty") not in DIFFICULTIES:
        problems.append(f"difficulty must be one of: {', '.join(DIFFICULTIES)}")
    if pos.get("cr_version") and pos["cr_version"] != CR_VERSION:
        problems.append(f"cr_version {pos['cr_version']} != pinned {CR_VERSION}")

    players = pos.get("players") or {}
    for side in PLAYERS:
        if side not in players:
            problems.append(f"players.{side} is required")
        elif "life" not in players[side]:
            problems.append(f"players.{side}.life is required")

    for p in pos.get("battlefield") or []:
        if p.get("controller") not in PLAYERS:
            problems.append(f"battlefield entry {p.get('card')!r} has controller "
                            f"{p.get('controller')!r}; must be 'you' or 'opp'")

    if pos.get("active_player") and pos["active_player"] not in PLAYERS:
        problems.append(f"active_player must be one of {PLAYERS}")
    if pos.get("priority") and pos["priority"] not in PLAYERS:
        problems.append(f"priority must be one of {PLAYERS}")

    # The rubric. Two key points minimum for the same reason the rules gold set
    # requires it: one point cannot separate a partly-correct line from a wrong
    # one. common_errors is required here, unlike the rules set, because it IS
    # the blunder list — a position without it cannot contribute to the gate.
    kp = [s for s in (pos.get("key_points") or []) if s.strip()]
    if len(kp) < 2:
        problems.append(f"needs >=2 key points, has {len(kp)}")
    ce = [s for s in (pos.get("common_errors") or []) if s.strip()]
    if not ce:
        problems.append("needs >=1 common_error — blunder rate is measured from "
                        "errors_made, so a position with no listed blunder cannot "
                        "contribute to Gate 3")

    # Legal actions must parse, or the closed arm silently degrades: an
    # unparseable entry can never be matched, so every model answer naming it
    # would be scored illegal.
    for legal in pos.get("legal_actions") or []:
        parsed = parse_line(legal)
        if not isinstance(parsed, Action):
            reason = parsed.reason if parsed else "not an action at all"
            problems.append(f"legal_action {legal!r} does not parse: {reason}")

    if card_index is not None:
        # Same rule as label_store.check_cards: an exact hit only. A position is
        # rendered verbatim into the prompt, so a name that only resolves by
        # prefix or fuzzy match would put a string in front of the model that
        # doesn't name a real card.
        for name in sorted(set(position_card_names(pos))):
            card, how = card_index.resolve(name)
            if card is None:
                problems.append(f"card not found in Oracle: {name!r} ({how})")
            elif how != "exact":
                problems.append(f"card {name!r} only resolves by {how} "
                                f"— use the exact name {card['name']!r}")

    if rule_ids is not None:
        for cite in pos.get("rule_citations") or []:
            if cite not in rule_ids:
                problems.append(f"rule id does not resolve against the pinned CR: {cite}")

    return [f"[{rid}] {p}" for p in problems]


def ingest(drafts: list[dict], existing: list[dict], author: str = "",
           card_index=None, rule_ids: set[str] | None = None,
           dry_run: bool = True, path: Path = POSITIONS_PATH) -> int:
    """Promote reviewed drafts into the position set. Returns the count added.

    The batch analogue of the `#/position` form, and the same trade the rubric
    loop makes: drafting is cheap, reviewing is the expensive part, so the tool
    exists to make a reviewed batch land safely rather than to make drafting
    faster.

    Three refusals, each one a measurement the loop could otherwise corrupt:

    - **An id that already exists is refused, never rewritten.** Positions are
      never edited through here. `author_rubrics.py` needed a rewrite path
      because a rubric is a patch onto a candidate question; a position is the
      whole record, so a "rewrite" is a different board wearing an old id, and
      every score already filed under that id would then describe a board that
      no longer exists.
    - **A record carrying `seed_note` is refused.** The seed fixtures are
      plumbing verification and say so in their own text (Section 14.6); the
      one way they become gate evidence is by being copied into this file.
    - **Nothing is written unless every record validates.** A partial batch
      leaves the set in a state no one reviewed.
    """
    by_id = {p.get("id") for p in existing}
    problems: list[str] = []
    accepted: list[dict] = []
    seen: set[str] = set()

    for i, pos in enumerate(drafts):
        rid = pos.get("id") or f"<record {i}>"
        if pos.get("seed_note"):
            problems.append(f"[{rid}] carries seed_note — machine-drafted fixtures are "
                            f"not gate evidence and must stay in positions_seed.jsonl")
            continue
        if rid in by_id:
            problems.append(f"[{rid}] already exists — positions are appended, never "
                            f"rewritten; give the new board a new id")
            continue
        if rid in seen:
            problems.append(f"[{rid}] appears twice in the draft file")
            continue
        seen.add(rid)
        pos = dict(pos)
        pos.setdefault("cr_version", CR_VERSION)
        pos.setdefault("graveyards", {"you": [], "opp": []})
        # Attribution rides on the record when it carries one, so a file can
        # hold several authors — the same rule that lets eval.py --compare
        # break agreement down per author for the rules gold set.
        if not pos.get("rubric_source"):
            pos["rubric_source"] = f"hand-authored ({author})" if author else "hand-authored"
        problems.extend(validate_position(pos, card_index, rule_ids))
        accepted.append(pos)

    for pos in accepted:
        for w in lint_common_errors(pos):
            print(f"  !!  [{pos['id']}] {w}")

    if problems:
        print(f"\n{len(problems)} problem(s) — nothing written:")
        for p in problems:
            print(f"  - {p}")
        raise SystemExit(1)

    if dry_run:
        print(f"\nDRY RUN — {path} not modified. {len(accepted)} position(s) would be added:")
        for pos in accepted:
            print(f"\n### {pos['id']}  [{pos['category']} / {pos['difficulty']}]  "
                  f"{pos['rubric_source']}")
            print(render_position(pos))
            print(f"  -> {pos['answer']}")
            for kp in pos.get("key_points") or []:
                print(f"    KP  {kp}")
            for ce in pos.get("common_errors") or []:
                print(f"    CE  {ce}")
        print(f"\n{len(accepted)} position(s) parsed. Re-run without --dry-run to apply.")
        return 0

    write_jsonl_atomic(path, existing + accepted)
    print(f"\npositions {len(existing)} -> {len(existing) + len(accepted)}  "
          f"added {len(accepted)}  pre-existing modified 0")
    return len(accepted)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--positions", type=Path, default=POSITIONS_PATH)
    parser.add_argument("--ingest", type=Path, default=None,
                        help="promote reviewed position drafts from a jsonl file")
    parser.add_argument("--dry-run", action="store_true",
                        help="with --ingest, render every incoming board and write nothing")
    parser.add_argument("--author", default="",
                        help="recorded in rubric_source when a draft carries none")
    parser.add_argument("--render", default=None,
                        help="print the rendered board (and prompt) for one position id")
    parser.add_argument("--closed", action="store_true",
                        help="with --render, show the closed-arm prompt instead of the open one")
    parser.add_argument("--no-cards", action="store_true",
                        help="skip Oracle name validation (faster; loses the main check)")
    args = parser.parse_args()

    positions = load_positions(args.positions)
    # An empty set is fatal for validation but is the normal starting state for
    # --ingest: the first reviewed batch is what creates positions.jsonl.
    if not positions and not args.ingest:
        raise SystemExit(f"no positions in {args.positions}")

    if args.render:
        match = next((p for p in positions if p["id"] == args.render), None)
        if not match:
            raise SystemExit(f"no position with id {args.render}")
        messages = build_position_messages(match, closed=args.closed)
        for m in messages:
            print(f"--- {m['role']} ---\n{m['content']}\n")
        print("--- expected line ---")
        print(match["answer"])
        return

    card_index = None
    if not args.no_cards:
        from card_lookup import CardIndex
        card_index = CardIndex()
    rule_ids = None
    try:
        from common import load_rule_ids
        rule_ids = load_rule_ids()
    except Exception as exc:  # rules corpus not built yet — skip, don't fail
        print(f"(skipping citation checks: {exc})")

    if args.ingest:
        # A missing file used to read as an empty list and report "0 positions
        # parsed", exiting 0. A typo'd filename then looked exactly like a
        # successful promotion of nothing, which is the quietest way to lose a
        # batch of hand-authored work — the author believes it landed.
        if not args.ingest.exists():
            raise SystemExit(f"no such file: {args.ingest}")
        drafts = read_jsonl(args.ingest)
        if not drafts:
            raise SystemExit(f"{args.ingest} contains no records — nothing to ingest")
        ingest(drafts, positions, author=args.author,
               card_index=card_index, rule_ids=rule_ids,
               dry_run=args.dry_run, path=args.positions)
        return

    all_problems = []
    for pos in positions:
        all_problems.extend(validate_position(pos, card_index, rule_ids))

    by_cat = Counter(p["category"] for p in positions)
    by_diff = Counter(p["difficulty"] for p in positions)
    print(f"{len(positions)} positions in {args.positions}")
    print("  by category:  " + ", ".join(f"{c}: {n}" for c, n in sorted(by_cat.items())))
    print("  by difficulty: " + ", ".join(f"{d}: {n}" for d, n in sorted(by_diff.items())))

    # Token budget: the 36GB training window is 2048 tokens, and a position
    # plus card text plus rules chunks approaches it. Reported in characters
    # here to avoid loading a tokenizer just to validate.
    lengths = [len(render_position(p)) for p in positions]
    print(f"  rendered board: {min(lengths)}-{max(lengths)} chars "
          f"(mean {sum(lengths) / len(lengths):.0f})")

    n_closed = sum(1 for p in positions if p.get("legal_actions"))
    print(f"  {n_closed}/{len(positions)} carry legal_actions (usable in the closed arm)")

    if all_problems:
        print(f"\n{len(all_problems)} problem(s):")
        for p in all_problems:
            print(f"  - {p}")
        raise SystemExit(1)
    print("\nall positions valid")


if __name__ == "__main__":
    main()
