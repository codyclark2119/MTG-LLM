"""Self-checking tests for the action grammar.

No pytest in the env, so these are plain asserts behind a runner:

    python scripts/gameplay/test_actions.py

The cases that matter most are the messy ones. A parser that only handles
`CAST Lightning Strike TARGET Grizzly Bears` would report a low legal-action
rate for a model that plays perfectly but writes markdown, and Gate 1 would
fail for a formatting reason dressed up as a capability finding.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from actions import Action, ParseFailure, legality, match_to_legal, parse_line, parse_output  # noqa: E402

CASES: list[tuple[str, str | None]] = [
    # (input line, expected canonical key or None for "prose")
    ("PASS", "PASS"),
    ("pass the turn", "PASS"),
    ("PLAY Mountain", "PLAY Mountain"),
    ("- PLAY Mountain", "PLAY Mountain"),
    ("1. PLAY Mountain", "PLAY Mountain"),
    ("**PLAY Mountain**", "PLAY Mountain"),
    ("* `PLAY Mountain`", "PLAY Mountain"),
    ("Action: PLAY Mountain", "PLAY Mountain"),
    ("Step 2: PLAY Mountain.", "PLAY Mountain"),
    ("CAST Lightning Strike", "CAST Lightning Strike"),
    ("CAST Lightning Strike TARGET Grizzly Bears", "CAST Lightning Strike TARGET Grizzly Bears"),
    ("cast Lightning Strike targeting Grizzly Bears", "CAST Lightning Strike TARGET Grizzly Bears"),
    ("CAST Fireball TARGET Bear, Elf", "CAST Fireball TARGET Bear, Elf"),
    ("CAST Fireball TARGET Bear and Elf", "CAST Fireball TARGET Bear, Elf"),
    ("ACTIVATE Llanowar Elves: {T}: Add {G}", "ACTIVATE Llanowar Elves: {T}: Add {G}"),
    ("ATTACK Monastery Swiftspear", "ATTACK Monastery Swiftspear"),
    ("ATTACK WITH Monastery Swiftspear", "ATTACK Monastery Swiftspear"),
    ("ORDER TRIGGERS Bolt, Draw", "ORDER TRIGGERS Bolt, Draw"),
    ("MULLIGAN", "MULLIGAN"),
    ("KEEP", "KEEP"),
    ("KEEP BOTTOM Mountain", "KEEP BOTTOM Mountain"),
    ("KEEP, bottoming Mountain and Shock", "KEEP BOTTOM Mountain, Shock"),
    ("Keeping this hand.", None),
    ("Mulliganing here is right.", None),
    # Prose — must be ignored, never a failure.
    ("Here is my line for this turn:", None),
    ("I think the best play is to hold up mana.", None),
    ("", None),
    # Present participles: the verb must be a WHOLE WORD. Each of these once
    # parsed as a phantom action ("Blocking the Bears with Elves" even came out
    # with its operands inverted), which would have inflated the action count
    # and sunk the legality rate for any model that explains its reasoning.
    ("Attacking with Swiftspear is correct.", None),
    ("Blocking the Bears with Elves would be bad.", None),
    ("Casting Shock here is wrong.", None),
    ("Playing a land first is right.", None),
    ("Passing is fine.", None),
    ("Activating the Elves taps it.", None),
    # ...but the real verbs still parse when followed by punctuation or nothing.
    ("PASS.", "PASS"),
    ("ATTACK Bear.", "ATTACK Bear"),
]

FAILURES = [
    ("CAST", "CAST with no argument"),
    ("ACTIVATE Llanowar Elves", "ACTIVATE without a colon"),
    ("ATTACK", "ATTACK with nothing named"),
    ("BLOCK Grizzly Bears", "BLOCK with only one creature"),
    ("ORDER TRIGGERS Bolt", "ORDER TRIGGERS with one trigger"),
    ("CAST Fireball TARGET", "TARGET with nothing after it"),
]


def check(label: str, got, want) -> bool:
    if got == want:
        return True
    print(f"  FAIL {label}\n       got  {got!r}\n       want {want!r}")
    return False


def main() -> None:
    failed = 0

    for line, want in CASES:
        result = parse_line(line)
        got = result.key() if isinstance(result, Action) else (
            None if result is None else f"PARSE FAILURE: {result.reason}")
        failed += not check(f"parse_line({line!r})", got, want)

    for line, why in FAILURES:
        result = parse_line(line)
        failed += not check(f"{why}: {line!r}", isinstance(result, ParseFailure), True)

    # --- The inversion trap ---------------------------------------------
    # These two lines mean the SAME play in English and OPPOSITE plays if the
    # operands are read positionally. Getting it wrong would silently invert
    # every blocking position in the eval, so it is asserted directly.
    canonical = parse_line("BLOCK Llanowar Elves -> Grizzly Bears")
    natural = parse_line("BLOCK Grizzly Bears with Llanowar Elves")
    failed += not check("BLOCK canonical", canonical.key(), "BLOCK Llanowar Elves -> Grizzly Bears")
    failed += not check("BLOCK natural phrasing agrees", natural.key(), canonical.key())
    failed += not check("blocker is the Elves", canonical.args[0], "Llanowar Elves")
    failed += not check("attacker is the Bears", canonical.args[1], "Grizzly Bears")

    # --- Attacker order is not meaningful --------------------------------
    a1 = parse_line("ATTACK Bear, Elf")
    a2 = parse_line("ATTACK Elf, Bear")
    failed += not check("attack order-insensitive", a1.key(), a2.key())

    # --- Trigger order IS meaningful -------------------------------------
    t1 = parse_line("ORDER TRIGGERS Bolt, Draw")
    t2 = parse_line("ORDER TRIGGERS Draw, Bolt")
    failed += not check("trigger order preserved", t1.key() != t2.key(), True)

    # --- Whole-output parsing --------------------------------------------
    messy = """Here's my line:

```
1. PLAY Mountain
2. **CAST Lightning Strike** targeting Grizzly Bears
```

Then I attack:
- ATTACK WITH Monastery Swiftspear
- PASS

That should be lethal next turn.
"""
    out = parse_output(messy)
    failed += not check("messy output -> 4 actions", len(out.actions), 4)
    failed += not check("messy output -> 0 failures", len(out.failures), 0)
    failed += not check("messy output ok", out.ok, True)
    failed += not check("prose ignored, not failed", len(out.ignored), 3)
    failed += not check(
        "messy output keys", out.keys(),
        ["PLAY Mountain", "CAST Lightning Strike TARGET Grizzly Bears",
         "ATTACK Monastery Swiftspear", "PASS"])

    # A malformed action must NOT hide in the ignored bucket.
    bad = parse_output("I'll think about it.\nCAST\nPASS")
    failed += not check("malformed action is a failure", len(bad.failures), 1)
    failed += not check("malformed output not ok", bad.ok, False)
    failed += not check("prose still ignored", len(bad.ignored), 1)

    # Pure prose yields no actions and is therefore not ok.
    prose = parse_output("I would hold priority and see what happens.")
    failed += not check("pure prose -> no actions", prose.actions, [])
    failed += not check("pure prose not ok", prose.ok, False)

    # --- Bracket meta-syntax the model copies out of the grammar ---------
    # These three forms all appeared in the first real run. Each one turned a
    # correct play into an unmatchable string, which the report counted as the
    # model naming an illegal action.
    for line in ("CAST Lightning Strike [ TARGET Grizzly Bears]",
                 "CAST Lightning Strike [TARGET Grizzly Bears]",
                 "CAST Lightning Strike [Target: Grizzly Bears]"):
        got = parse_line(line)
        failed += not check(f"bracket form {line!r}", got.key(),
                            "CAST Lightning Strike TARGET Grizzly Bears")

    # --- Degenerate repetition -------------------------------------------
    # The v2 adapter emitted PASS 190 times on a board state. That is one
    # action followed by a decode failure, not 190 actions.
    loop = parse_output("CAST Shock TARGET Bear\n" + "PASS\n" * 100)
    failed += not check("repetition collapsed to 2 actions", len(loop.actions), 2)
    failed += not check("collapse counted", loop.repeats_collapsed, 99)
    failed += not check("flagged degenerate", loop.degenerate, True)
    normal = parse_output("PLAY Mountain\nPASS")
    failed += not check("normal output not degenerate", normal.degenerate, False)

    # --- Matching against an enumerated legal set ------------------------
    legal = ["PLAY Mountain", "CAST Lightning Strike TARGET Grizzly Bears", "PASS"]
    failed += not check(
        "phrasing-independent match",
        match_to_legal(parse_line("cast Lightning Strike targeting Grizzly Bears"), legal),
        "CAST Lightning Strike TARGET Grizzly Bears")
    failed += not check(
        "illegal action does not match",
        match_to_legal(parse_line("CAST Shock TARGET Grizzly Bears"), legal), None)

    lg = legality(parse_output("PLAY Mountain\nCAST Shock TARGET Grizzly Bears"), legal)
    failed += not check("legality counts", (lg["n_actions"], lg["n_legal"], lg["all_legal"]),
                        (2, 1, False))
    failed += not check("illegal listed", lg["illegal"], ["CAST Shock TARGET Grizzly Bears"])

    # --- The grammar the model is TOLD matches the one that is PARSED -----
    # ACTION_GRAMMAR lives in common.py (it is a prompt) and the parser lives
    # here. A verb added to one and not the other is an action the model is
    # instructed to produce and then scored as malformed for producing — which
    # would read as a capability failure. Assert they agree instead of asking
    # a comment to keep them in sync.
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from common import ACTION_GRAMMAR  # noqa: PLC0415

    # Match each grammar line the way the parser does, rather than re-deriving
    # the rule: some verbs have more than one documented form (KEEP / KEEP
    # BOTTOM, CAST / CAST TARGET), so the check is that every line resolves to
    # a known verb and every known verb is documented — not a line-count match.
    from actions import VERBS  # noqa: PLC0415

    documented, unmatched = set(), []
    for gline in (line for line in ACTION_GRAMMAR.splitlines() if line.strip()):
        upper = gline.upper()
        verb = next((v for v in sorted(VERBS, key=len, reverse=True)
                     if upper.startswith(v)), None)
        (documented.add(verb) if verb else unmatched.append(gline.split("  ")[0]))
    failed += not check("every grammar line names a known verb", unmatched, [])
    failed += not check("every parser verb is documented", set(VERBS) - documented, set())

    total = len(CASES) + len(FAILURES) + 29
    if failed:
        print(f"\n{failed} check(s) FAILED")
        raise SystemExit(1)
    print(f"all checks passed ({total} assertions)")


if __name__ == "__main__":
    main()
