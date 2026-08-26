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

from actions import (Action, DECLARATIONS, PHASE_NAMES,  # noqa: E402
                     parse_line)
from common import (  # noqa: E402
    parse_permanent_line,
    position_from_form,  # noqa: E402,F401  (lint_common_errors re-exported for webui)
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
    # Stages 3 and 5 of the gameplay curriculum, which had no category and so
    # could not be authored at all: "payment" is correct mana tapped and
    # sorcery-speed timing, "closing the turn" is the postcombat main phase the
    # set has never contained a position for (Section 21.71).
    "payment", "closing the turn",
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

    problems.extend(timing_problems(pos, card_index))
    problems.extend(mana_problems(pos, card_index))
    # A stored reference line is a claim that this is the 100%-correct answer,
    # and it is the one claim in a position a machine can settle. Checked here
    # as well as at ingest so it is re-verified whenever anything else changes
    # — a rubric entry added later (PROTOCOL_ERRORS is append-only) can make a
    # previously clean reference fail, and nothing else would notice (21.85).
    if pos.get("reference_actions"):
        problems.extend(f"reference_actions: {pr}" for pr in
                        check_reference(pos, pos["reference_actions"], card_index))

    return [f"[{rid}] {p}" for p in problems]


# Steps in which only instant-speed spells may be cast. A sorcery-speed spell
# needs your own main phase with an empty stack (307.1).
# Steps where a sorcery-speed spell CANNOT be cast — i.e. every step that is not
# a main phase. Named for what it means rather than for combat: it has always
# included upkeep, draw, end step, cleanup and the opening hand, so a reader
# checking "does this cover upkeep?" against the old combat-flavoured name would
# have concluded it did not. One name, two meanings, in the direction that reads
# as a coverage gap where there is none (Section 21.59).
_NON_MAIN_STEPS = ("upkeep", "draw", "beginning of combat", "declare attackers",
                   "declare blockers", "combat damage", "end of combat", "end step",
                   "cleanup", "opening hand")


def timing_problems(pos: dict, card_index=None) -> list[str]:
    """Catch a `legal_action` that could not legally be taken in this step.

    Found the hard way. A board was drafted offering `CAST Pacifism` during the
    opponent's declare-attackers step, and everything passed: the card name
    resolves against Oracle, the action parses, the rule ids check out. Pacifism
    is an Aura, so it is sorcery-speed and cannot be cast then at all — the
    position asked for a decision that could not be made.

    Nothing else validates this. `validate_position` checks that a card EXISTS
    and that an action PARSES; neither is a claim about when it may be played.
    A position offering an illegal play is worse than a malformed one, because
    it looks correct all the way to the scores.

    Deliberately narrow: only CAST is checked, and only for the clear case of a
    non-instant without flash outside a main phase. Activated abilities carry
    their own timing restrictions in their text and are not parsed here.
    """
    if card_index is None:
        return []
    phase = (pos.get("phase") or "").lower()
    if not any(step in phase for step in _NON_MAIN_STEPS):
        return []
    out = []
    for line in pos.get("legal_actions") or []:
        parsed = parse_line(line)
        if not isinstance(parsed, Action) or parsed.verb != "CAST" or not parsed.args:
            continue
        card, how = card_index.resolve(parsed.args[0])
        if card is None or how != "exact":
            continue  # the name check already reports this
        text, type_line = card.get("text") or "", card.get("type_line") or ""
        if "Instant" in type_line or "Flash" in text:
            continue
        out.append(f"legal_action {line!r} is sorcery-speed ({type_line}) but the "
                   f"position is in {pos.get('phase')!r} — it could not be cast then")
    return out


_MANA_SYMBOL_RE = re.compile(r"\{([^}]+)\}")
_COLOURS = ("W", "U", "B", "R", "G", "C")


def parse_mana(cost: str) -> tuple[dict[str, int], int] | None:
    """(coloured requirements, generic) for a mana string, or None if unsupported.

    Returns None rather than guessing on hybrid, Phyrexian, X, or snow symbols.
    A cost this cannot represent must not be silently treated as free — that
    would turn "we do not model this" into "this is affordable", which is the
    direction that reads as a pass (Section 21.59).
    """
    if not cost:
        return None
    need: dict[str, int] = {}
    generic = 0
    for sym in _MANA_SYMBOL_RE.findall(cost):
        s = sym.upper()
        if s.isdigit():
            generic += int(s)
        elif s in _COLOURS:
            need[s] = need.get(s, 0) + 1
        else:
            return None          # hybrid, Phyrexian, X, snow — not modelled
    return need, generic


def protocol_findings(pos: dict, parsed, card_index=None) -> dict[int, bool]:
    """Which `common.PROTOCOL_ERRORS` this answer actually commits, by 1-based index.

    The checker side of the six appended rubric entries. Every one is decidable
    from the board and the parse, so when the judge charges error N a machine can
    say whether N is true — per-error precision with no human and no second
    judge (Section 21.70).

    `None` for an entry means "not decidable here": indices 5, 6 and 8 need the
    card index and index 4 needs the position to state a phase. A check that
    cannot run must not be reported as "the judge was wrong", which is the
    one-sided-control mistake of 21.43 in a new place.
    """
    from actions import legality

    out: dict[int, bool | None] = {}
    # "Only passes" is a blunder when there was something to do and the CORRECT
    # answer when there was not. Those two readings agree on every board that
    # enumerates a play and come apart on the one kind that does not — stage 3's
    # payment positions, where the whole test is recognising that nothing is
    # affordable. Without this, the right answer to
    # `sample-stage3-payment-0005` fired entry 1 and scored valid_turn=False.
    #
    # Conditioned on the position's own list, not on what the arm was shown: the
    # open arm never sees `legal_actions`, but the board still has them, and
    # this is a question about the board. An ABSENT list is left alone — that is
    # "not stated", not "nothing is legal", the same distinction `phase_problems`
    # draws (Section 21.85).
    no_play_board = "legal_actions" in pos and not (pos.get("legal_actions") or [])
    out[1] = False if no_play_board else parsed.only_pass
    out[2] = parsed.degenerate
    out[3] = not legality(parsed, pos.get("legal_actions") or []).get("all_legal")
    out[4] = bool(phase_problems(pos, parsed.actions)) if pos.get("phase") else None
    kind = (payment_kind(pos, parsed.actions, card_index)
            if card_index is not None
            and any(a.verb == "TAP" for a in parsed.actions) else "undecidable")
    out[5] = None if kind == "undecidable" else (kind == "short")
    out[6] = None if kind == "undecidable" else (kind == "excess")
    out[7] = bool(battlefield_cast_problems(pos, parsed.actions))
    # 8-10, appended with the rubric entries in Section 21.83. Each was measured
    # as a diagnostic first and promoted only once it had a frequency and a
    # reviewer note behind it.
    #
    # 8 needs the card index to know whether the spell is a creature that does
    # not say "target", so it is undecidable without one — the same treatment 5
    # and 6 get, and for the same reason: a check that cannot run must not be
    # reported as the judge being wrong (21.43).
    out[8] = (bool(targeting_problems(pos, parsed.actions, card_index))
              if card_index is not None else None)
    # 9 and 10 read only the answer's own lines, so they are always decidable.
    out[9] = bool(wasted_tap_problems(pos, parsed.actions))
    out[10] = bool(missing_phase_problems(pos, parsed.actions))
    return out


def phase_problems(pos: dict, actions) -> list[str]:
    """A `PHASE` declaration that disagrees with the board it was given.

    The verbose grammar asks the model to state when it thinks it is. That is
    worth asking only if it is checked, and it is checkable against
    `pos["phase"]` with no judge and no card data.

    This is the one thing the enumerated-`legal_actions` check structurally
    cannot see. Section 21.59 measured 24 of 251 answers taking an action
    impossible in the stated phase and found **all 24 already caught** — because
    a phase-illegal action is simply not in the list. But a model can take a
    perfectly legal action while believing it is a different step, and nothing
    about the action reveals that. The declaration does.

    Matching is loose on purpose: the rendered board says "opponent's declare
    attackers" and an answer saying "declare attackers" means the same step. A
    false mismatch would manufacture a finding out of phrasing, which is the
    failure this repo has had from `_PLAYER_PREP` and from substring verbs.
    Returns [] when nothing was declared — silence is "not stated", not "agreed".
    """
    want = (pos.get("phase") or "").lower()
    if not want:
        return []
    out = []
    for a in actions:
        if a.verb != "PHASE" or not a.args:
            continue
        said = a.args[0].lower()
        # Agree if either names the other's step, so possessives and turn
        # ownership ("your", "opponent's") never decide the comparison.
        if said in want or want in said:
            continue
        shared = [st for st in PHASE_NAMES if st in said and st in want]
        if shared:
            continue
        out.append(f"PHASE {a.args[0]!r}: the position is in {pos.get('phase')!r}")
    return out


_ADD_RE = re.compile(r"\{T\}[^:\n]*:\s*Add ([^.\n]*)\.", re.I)


def mana_abilities(card: dict) -> list[str] | None:
    """Which mana a land can produce, as a list of `{X}`-style strings, or None.

    Scryfall's `produced_mana` is not in this corpus, so the abilities are read
    out of the oracle text (`{T}: Add {G}.`, `{T}: Add {W} or {B}.`). Returns
    None when no `{T}: Add` clause is found, which must NOT be read as "produces
    nothing" — a land whose ability this cannot parse is a land this cannot
    check, and treating it as producing nothing would flag every correct tap of
    it (Section 21.60).
    """
    clauses = _ADD_RE.findall(card.get("text") or "")
    if not clauses:
        return None
    out = []
    for clause in clauses:
        for option in re.split(r"\bor\b", clause, flags=re.I):
            syms = _MANA_SYMBOL_RE.findall(option)
            if syms:
                out.append("".join(f"{{{x.upper()}}}" for x in syms))
    return out or None


def tap_problems(pos: dict, actions, card_index=None) -> list[str]:
    """Declared taps that a permanent could not have produced, or does not exist.

    The verbose grammar (21.60) asks the model to say `TAP Forest FOR {G}`
    rather than leaving mana implicit. That declaration is only worth asking for
    if it is checked, and it is checkable without a judge: the battlefield says
    which permanents are untapped and the oracle text says what each can add.

    Three problems, each a different claim:
      * tapping something not on your battlefield
      * tapping something already tapped
      * naming mana the permanent cannot produce

    Silent when the card index is absent or the ability does not parse, and the
    caller reports that as coverage rather than as a pass.
    """
    if card_index is None:
        return []
    untapped: dict[str, int] = {}
    for b in pos.get("battlefield") or []:
        if b.get("controller") != "you" or b.get("tapped"):
            continue
        untapped[b["card"].lower()] = untapped.get(b["card"].lower(), 0) + 1
    used: dict[str, int] = {}
    out = []
    for a in actions:
        if a.verb != "TAP" or len(a.args) < 2:
            continue
        name, mana = a.args[0], a.args[1]
        key = name.lower()
        on_bf = any(b["card"].lower() == key and b.get("controller") == "you"
                    for b in pos.get("battlefield") or [])
        if not on_bf:
            out.append(f"TAP {name}: you control no such permanent")
            continue
        used[key] = used.get(key, 0) + 1
        if used[key] > untapped.get(key, 0):
            out.append(f"TAP {name}: only {untapped.get(key, 0)} untapped copies, "
                       f"tapped {used[key]} times")
            continue
        card, how = card_index.resolve(name)
        if card is None or how != "exact":
            continue
        can = mana_abilities(card)
        if can is None:
            continue
        want = "".join(f"{{{x.upper()}}}" for x in _MANA_SYMBOL_RE.findall(mana)) or mana
        if want not in can:
            out.append(f"TAP {name} FOR {mana}: it can only add {' or '.join(can)}")
    return out


def payment_problems(pos: dict, actions, card_index=None) -> list[str]:
    """Declared taps that do not pay for the spells declared alongside them.

    `tap_problems` checks each tap on its own — is it yours, is it untapped, can
    it add that colour. It never adds them up, so a reviewer found the gap the
    moment the verbose grammar produced one: *"It taps excess mana as lightning
    strike costs 1 generic mana and 1 red mana"* — three lands tapped for a
    two-mana spell, every individual tap correct (Section 21.66).

    Compares the declared pool against the summed cost of the `CAST` actions in
    the same answer. Two findings, deliberately separate: **short** means the
    declaration cannot pay for the plays, **excess** means mana was floated for
    nothing. Only the colours a cost actually demands are checked against the
    pool; generic is paid from whatever is left.

    Silent unless the answer declares at least one tap AND casts at least one
    spell — an answer that declares nothing is not over-tapping, it is a
    different finding that `only_pass` and the prompt already cover. Silent too
    when any cost involved is one `parse_mana` refuses to model, because a
    partial total is worse than none.
    """
    if card_index is None:
        return []
    taps = [a for a in actions if a.verb == "TAP" and len(a.args) >= 2]
    casts = [a for a in actions if a.verb == "CAST" and a.args]
    if not taps or not casts:
        return []

    pool: dict[str, int] = {}
    for a in taps:
        parsed = parse_mana(a.args[1])
        if parsed is None:
            return []
        coloured, generic = parsed
        for c, n in coloured.items():
            pool[c] = pool.get(c, 0) + n
        if generic:                      # "TAP Foo FOR {2}" — colourless amount
            pool["*"] = pool.get("*", 0) + generic

    need: dict[str, int] = {}
    generic_total = 0
    for a in casts:
        card, how = card_index.resolve(a.args[0])
        if card is None or how != "exact":
            return []                    # an unknown card cannot be costed
        parsed = parse_mana(card.get("mana_cost") or "")
        if parsed is None:
            return []
        coloured, generic = parsed
        for c, n in coloured.items():
            need[c] = need.get(c, 0) + n
        generic_total += generic

    out = []
    have_total = sum(pool.values())
    short = [f"{n - pool.get(c, 0)}x{{{c}}}" for c, n in need.items() if pool.get(c, 0) < n]
    if short:
        out.append(f"declared taps are short {', '.join(short)} for the spells cast")
    cost_total = sum(need.values()) + generic_total
    if have_total < cost_total:
        out.append(f"declared {have_total} mana but cast {cost_total} mana of spells")
    elif have_total > cost_total:
        out.append(f"declared {have_total} mana for {cost_total} mana of spells "
                   f"({have_total - cost_total} floated for nothing)")
    return out


def payment_kind(pos: dict, actions, card_index=None) -> str | None:
    """"short", "excess", or None — which way the declared taps miss.

    A reviewer drew the line and it is a real one: under-tapping means the spell
    cannot be cast at all, and over-tapping is a legal play a real player makes.
    One rubric entry covering both asked the judge to charge two different
    mistakes with one number (Section 21.70).

    "short" wins when both are somehow reported, because an unpayable cost is
    the more serious finding and the two are not independent.
    """
    probs = payment_problems(pos, actions, card_index)
    if not probs:
        return None
    if any("short" in p or "but cast" in p for p in probs):
        return "short"
    return "excess" if any("floated" in p for p in probs) else None


def battlefield_cast_problems(pos: dict, actions) -> list[str]:
    """Casting a permanent that is already on the battlefield.

    From a reviewer's note: *"It is not seeing Serra Angel as already on the
    field so it is attempting to cast it."* Caught today only because the string
    is absent from `legal_actions`, which reports "not a legal action" and gives
    no reason — and would not catch it at all on a position that happened to
    enumerate the card for another purpose.

    Needs no card data: the board says what is in play and what is in hand. A
    card in BOTH is fine — a second copy is castable — so the check is "on the
    battlefield and not in hand", which is the only unambiguous case.
    """
    hand = {c.lower() for c in (pos.get("players", {}).get("you", {}).get("hand") or [])}
    mine = {b["card"].lower() for b in (pos.get("battlefield") or [])
            if b.get("controller") == "you"}
    out = []
    for a in actions:
        if a.verb != "CAST" or not a.args:
            continue
        name = a.args[0]
        low = name.lower()
        if low in mine and low not in hand:
            out.append(f"CAST {name}: it is already on your battlefield, not in hand")
    return out


def wasted_tap_problems(pos: dict, actions) -> list[str]:
    """Mana declared and then not spent on anything.

    From four reviewer notes: *"Taps lands for no reason, declares no
    blockers"*, *"it pointlessly taps the forests on the field before passing
    wasting the mana"*, *"Taps mana unnecessarily"*. Measured at **11 of 78**
    answers (14%), stable at 15% on the pre-protocol run (Section 21.83).

    `PROTOCOL_ERRORS` entry 6 is over-tapping and does **not** cover this. Its
    wording — *"add up to MORE than the spells it casts require"* — presumes
    there are spells, and `payment_problems` is silent when nothing is cast, so
    entry 6 fires on **0 of the 11**. Three of them have no entry firing at all.

    The mana empties at end of step, so tapping without spending is never a way
    to hold up a trick: the correct way to represent holding removal is to leave
    the lands untapped. That makes this unambiguous rather than a judgement call,
    which is why a parser can decide it.

    Needs no card data — the answer's own lines are the whole evidence.
    """
    taps = [a for a in actions if a.verb == "TAP"]
    if not taps:
        return []
    if any(a.verb in ("CAST", "PLAY") for a in actions):
        return []
    named = ", ".join(a.args[0] for a in taps if a.args)
    return [f"TAP {named}: mana declared and never spent — it empties at end of "
            "step, so nothing is held up by it"]


def missing_phase_problems(pos: dict, actions) -> list[str]:
    """Plays made without ever declaring which step they are made in.

    `phase_problems` checks a declaration against the board and deliberately
    returns [] when none was made — *"silence is 'not stated', not 'agreed'"*.
    That was right while the grammar merely allowed a PHASE line. Since 21.60 the
    prompt *requires* one ("Open with PHASE"), so silence is now a contract
    violation rather than an abstention, and the reviewer flagged it three times:
    *"doesnt declare the change to the declare attackers phase"*.

    Measured at **10 of 78** (13%), and 14% on the pre-protocol run.

    An answer whose only action is PASS is exempt: it makes no play, entry 1
    already describes it, and demanding a phase declaration from a player who
    does nothing would double-charge 21.58's do-nothing case.
    """
    if any(a.verb == "PHASE" for a in actions):
        return []
    plays = [a for a in actions if a.verb not in DECLARATIONS and a.verb != "PASS"]
    if not plays:
        return []
    return [f"{len(plays)} play(s) made with no PHASE line: the answer never says "
            "which step it is acting in"]


def targeting_problems(pos: dict, actions, card_index=None) -> list[str]:
    """A creature spell cast with a TARGET it does not have.

    From four separate reviewer notes naming the same mistake — *"casting
    Grizzly Bears as a non-creature spell targeting Centaur Courser"*, *"Serra
    Angel gets recasted targeting creatures like a non-creature spell"*, and
    twice for Ambush Viper. Measured at **6 of 78** stored answers (Section
    21.80). A creature spell does not target on cast; the model is treating it
    like removal.

    Today this is convicted only as `PROTOCOL_ERRORS` entry 3, "names a play
    that is not available", because `CAST Ambush Viper TARGET Centaur Courser`
    does not string-match the `CAST Ambush Viper` in `legal_actions`. That fires
    for the wrong reason and gives no reason back — and the reviewer who met it
    ticked **`not_covered`**, because "the play is unavailable" is not what went
    wrong. The play was right; the operand was invented.

    **Misses rather than guesses**, the rule `find_players` follows. It fires
    only when the card is a Creature, is not an Aura (auras DO target on cast,
    rule 303.4c, and their text says "Enchant" rather than "target"), and its
    oracle text contains no "target" at all — so a creature whose ETB trigger
    targets is skipped, even though the SPELL still does not target. A missed
    case costs a diagnostic; a false one would convict a correct play.

    Returns [] with no card index, like `mana_problems` and `payment_problems`,
    so a caller without card data reports nothing rather than crediting.
    """
    if card_index is None:
        return []
    out = []
    for a in actions:
        if a.verb not in ("CAST", "PLAY") or len(a.args) < 2:
            continue
        name = a.args[0]
        card, _how = card_index.resolve(name)
        if card is None:
            continue
        types = (card.get("type_line") or card.get("type") or "")
        text = (card.get("text") or card.get("oracle_text") or "")
        if "Creature" not in types or "Aura" in types:
            continue
        if "target" in text.lower():
            continue
        out.append(f"CAST {name} TARGET {a.args[1]}: a creature spell does not "
                   "target on cast")
    return out


def mana_problems(pos: dict, card_index=None) -> list[str]:
    """A `legal_action` the position could not actually pay for.

    `timing_problems` asks whether a play is legal *when*; this asks whether it
    is payable *at all*. Nothing checked it: a board could offer
    `CAST Doom Blade` with `{G}{G}` available and every existing check passes —
    the card resolves, the action parses, the step allows an instant.

    Only fires when the position states `mana_available` (8 of 24 do) and the
    cost is one `parse_mana` can represent. Both silences are deliberate and
    both are reported by the caller rather than swallowed: a position with no
    stated pool is not a position with no mana, and a cost this cannot parse is
    not a cost of zero.
    """
    if card_index is None:
        return []
    have = parse_mana(pos.get("mana_available") or "")
    if have is None:
        return []
    pool, pool_generic = have
    total = sum(pool.values()) + pool_generic
    out = []
    for line in pos.get("legal_actions") or []:
        parsed = parse_line(line)
        if not isinstance(parsed, Action) or parsed.verb != "CAST" or not parsed.args:
            continue
        card, how = card_index.resolve(parsed.args[0])
        if card is None or how != "exact":
            continue
        cost = parse_mana(card.get("mana_cost") or "")
        if cost is None:
            continue
        need, generic = cost
        short = [c for c, n in need.items() if pool.get(c, 0) < n]
        if short or sum(need.values()) + generic > total:
            out.append(
                f"legal_action {line!r} costs {card.get('mana_cost')} but the position "
                f"states only {pos.get('mana_available')} available")
    return out


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
        # A draft from the web form carries no id: the form cannot see the gold
        # set, so it cannot number without risking a collision between two
        # authors. Numbering happens HERE, where the existing set is in hand —
        # the same reason promotion is local and reviewed at all.
        if not pos.get("id"):
            pos["id"] = next_position_id(existing + accepted, pos.get("category") or "")
        pos.setdefault("cr_version", CR_VERSION)
        pos.setdefault("graveyards", {"you": [], "opp": []})
        # Attribution rides on the record when it carries one, so a file can
        # hold several authors — the same rule that lets eval.py --compare
        # break agreement down per author for the rules gold set.
        if not pos.get("rubric_source"):
            pos["rubric_source"] = f"hand-authored ({author})" if author else "hand-authored"
        elif pos["rubric_source"].startswith("draft sample") and author:
            pos["rubric_source"] = f"hand-authored ({author}); reviewed draft"
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


def check_reference(pos: dict, lines: list[str], card_index=None) -> list[str]:
    """Why this reference line is not a 100%-correct answer, if it is not.

    A reference line claims to be the perfect output for a board, so it is
    checkable in a way no ordinary claim about gold data is: run it through the
    same parser and the same `PROTOCOL_ERRORS` checks every arm answer goes
    through, and a correct line must come out clean on all of them.

    That is what makes this field worth collecting. Section 21.55's trap is a
    field a human fills in that nothing reads; this one is refused at the door
    if it does not hold up, so a stored reference is a line a machine has
    already agreed with.

    Checks, in the order a failure is most likely:

    - it parses at all, with no `ParseFailure`;
    - every play is in `legal_actions` (skipped when the board enumerates none,
      where PASS alone is the whole correct answer);
    - `protocol_findings` fires nothing decidable. An undecidable entry is not
      a failure — the 21.43 rule that a check which cannot run must not be
      reported as a verdict.

    Returns [] when the line is good, so it reads like `validate_position`.
    """
    from actions import legality, parse_output

    if not lines:
        return ["reference is empty"]
    parsed = parse_output("\n".join(lines))
    problems = [f"unparseable: {f.line!r}" for f in parsed.failures]

    legal = pos.get("legal_actions") or []
    if legal:
        info = legality(parsed, legal)
        for bad in info.get("illegal") or []:
            problems.append(f"not a legal play here: {bad!r}")
    else:
        # An empty `legal_actions` means the board intentionally has no play, so
        # PASS is the whole correct answer and anything else contradicts the
        # position. PASS is excluded explicitly: `ParsedOutput.plays` keeps it,
        # because it is the protocol terminator rather than a declaration
        # (16.13), so counting `plays` here would refuse the only correct line.
        made = [a for a in parsed.plays if a.verb != "PASS"]
        if made:
            problems.append("board lists no legal plays, but the reference makes "
                            f"{len(made)}")

    for n, fired in protocol_findings(pos, parsed, card_index).items():
        if fired:
            problems.append(f"commits PROTOCOL_ERRORS entry {n}")
    return problems


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
    parser.add_argument("--ingest-references", type=Path, default=None,
                        metavar="FILE",
                        help="promote reference lines from a submissions log "
                             "into positions.jsonl (parser-validated)")
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

    if args.ingest_references:
        # Promotion is local and reviewed, the invariant the rubric form has
        # always kept: the deployed server appends a claim, and this is where a
        # claim becomes gold — after a parser has agreed with it (21.85).
        existing = load_positions(args.positions)
        by_id = {p["id"]: p for p in existing}
        rows = [r for r in read_jsonl(args.ingest_references, missing_ok=True)
                if r.get("kind") == "reference"]
        # Newest row per (record, author) wins, so a correction supersedes the
        # line it corrects without anyone editing the log.
        newest: dict[tuple, dict] = {}
        for r in sorted(rows, key=lambda r: r.get("submitted") or ""):
            newest[(r.get("record_id"), r.get("author"))] = r
        # card_index is built above and is None under --no-cards, which makes
        # `protocol_findings` entries 5, 6 and 8 undecidable rather than false —
        # so a reference imported without card data is checked less thoroughly,
        # not credited (21.43).
        accepted, refused, unchanged = [], [], 0
        for (rid, author), row in sorted(newest.items()):
            pos = by_id.get(rid)
            if pos is None:
                refused.append((rid, author, ["no such position"]))
                continue
            lines = row.get("reference_actions") or []
            problems = check_reference(pos, lines, card_index)
            if problems:
                refused.append((rid, author, problems))
                continue
            if (pos.get("reference_actions") or []) == lines:
                unchanged += 1
                continue
            accepted.append((rid, author, lines))
        print(f"{len(rows)} reference submissions, {len(newest)} after "
              f"(record, author) dedup")
        print(f"  {len(accepted)} accepted, {len(refused)} REFUSED, "
              f"{unchanged} already on file unchanged")
        for rid, author, problems in refused:
            print(f"  refused {rid} ({author or '?'}):")
            for pr in problems:
                print(f"      {pr}")
        if not accepted:
            return
        if args.dry_run:
            for rid, author, lines in accepted:
                print(f"  would set {rid} <- {len(lines)} actions ({author or '?'})")
            print("\nDRY RUN — re-run without --dry-run to write")
            return
        for rid, author, lines in accepted:
            by_id[rid]["reference_actions"] = lines
            by_id[rid]["reference_source"] = (f"hand-authored ({author})"
                                              if author else "hand-authored")
        write_jsonl_atomic(args.positions, existing)
        print(f"wrote {len(accepted)} reference lines -> {args.positions}")
        return

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

    # What the timing and mana checks could NOT look at. Both return [] when
    # their input is absent, and "no problems found" over a set they never
    # examined is the shape of a passing check that checked nothing — the same
    # failure as a coverage-free gate verdict (21.14) and a one-sided control
    # (21.43). Stated as coverage, next to the verdict it qualifies.
    n_mana = sum(1 for p in positions if p.get("mana_available"))
    print(f"  {n_mana}/{len(positions)} state mana_available "
          f"(the affordability check sees only these)")
    if card_index is None:
        print("  ! no card index loaded — the timing and affordability checks are "
              "SKIPPED, not passed")

    if all_problems:
        print(f"\n{len(all_problems)} problem(s):")
        for p in all_problems:
            print(f"  - {p}")
        raise SystemExit(1)
    print("\nall positions valid")


if __name__ == "__main__":
    main()
