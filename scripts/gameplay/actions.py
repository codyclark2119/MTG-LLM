"""The action grammar: what a model is allowed to say when it takes a turn.

Rules answers are prose and get judged as prose. A *play* has to be compared
mechanically — was it legal, was it the same play the rubric describes — so
the model's output needs a shape a machine can read.

The grammar:

    PLAY <card>                       a land drop
    CAST <card> [TARGET <a>, <b>]     a spell, with targets if it has them
    ACTIVATE <permanent>: <ability>   an activated ability
    ATTACK <creature>[, <creature>]   declare attackers, all at once
    BLOCK <blocker> -> <attacker>     one assignment per line
    ORDER TRIGGERS <a>, <b>           in the order they should RESOLVE
    MULLIGAN                          ship the opening hand
    KEEP [BOTTOM <a>, <b>]            keep it, naming any cards put on the bottom
    PASS                              yield priority / end the turn

Two design rules earn their keep here:

**Liberal on input.** Real model output arrives wrapped in markdown bullets,
numbered lists, bold, code fences, and a sentence of throat-clearing. Rejecting
that would measure formatting compliance, not play quality. So scaffolding is
stripped before parsing.

**Never silently drop a line.** Every line lands in exactly one of three
buckets: an `Action`, a `ParseFailure`, or `ignored` prose. The distinction
between the last two is deliberate — a line that OPENS WITH A KNOWN VERB and
then fails to parse is a real failure (the model meant to act and malformed
it), while a line that never claimed to be an action is just narration. Folding
those together would either punish models for explaining themselves or hide
malformed actions inside the "prose" bucket, and both corrupt Gate 1.

Syntax only. Card names are not checked against Oracle here — that is the
position validator's job (see build_position.py), and keeping the split clean
means a parse failure always means "malformed", never "unknown card".
"""

import re
from dataclasses import dataclass, field

VERBS = ("PLAY", "CAST", "ACTIVATE", "ATTACK", "BLOCK", "ORDER TRIGGERS",
         "MULLIGAN", "KEEP", "PASS")

# Markdown/list scaffolding the model wraps its answer in. Stripped, not failed.
_SCAFFOLD_RE = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s*")
_BOLD_RE = re.compile(r"\*\*|__|`")
# "Action: CAST ..." / "Play: ATTACK ..." — a label the model adds to its line.
_LABEL_RE = re.compile(r"^\s*(?:action|play|step|move)\s*\d*\s*:\s*", re.I)
_FENCE_RE = re.compile(r"^\s*```")


@dataclass(frozen=True)
class Action:
    """One parsed play. `key()` is the canonical form used for comparison."""

    verb: str
    args: tuple = ()
    raw: str = ""

    def key(self) -> str:
        if not self.args:
            return self.verb  # PASS, MULLIGAN, KEEP-at-seven
        if self.verb == "KEEP":
            return "KEEP BOTTOM " + ", ".join(sorted(self.args))
        if self.verb == "ATTACK":
            # Attacker order is not meaningful — sort so two orderings of the
            # same attack compare equal.
            return "ATTACK " + ", ".join(sorted(self.args))
        if self.verb == "BLOCK":
            return f"BLOCK {self.args[0]} -> {self.args[1]}"
        if self.verb == "ACTIVATE":
            return f"ACTIVATE {self.args[0]}: {self.args[1]}"
        if self.verb == "CAST" and len(self.args) > 1:
            return f"CAST {self.args[0]} TARGET " + ", ".join(self.args[1:])
        # ORDER TRIGGERS keeps its order — that is the whole content of the play.
        return f"{self.verb} " + ", ".join(self.args)

    def card_names(self) -> list[str]:
        """Names this action refers to, for validation against Oracle."""
        if self.verb == "ACTIVATE":
            return list(self.args[:1])  # the ability text is not a card name
        if self.verb in ("PLAY", "CAST", "ATTACK", "BLOCK", "KEEP"):
            return list(self.args)
        return []


@dataclass(frozen=True)
class ParseFailure:
    raw: str
    reason: str


@dataclass
class ParsedOutput:
    actions: list[Action] = field(default_factory=list)
    failures: list[ParseFailure] = field(default_factory=list)
    ignored: list[str] = field(default_factory=list)
    # How many consecutive duplicates were collapsed. A model that emits PASS
    # 190 times has taken one action and then degenerated; counting that as 190
    # actions makes "actions per answer" a measure of looping rather than of
    # play. The count is kept rather than discarded because the looping itself
    # is a real finding — the v2 adapter does exactly this on board states.
    repeats_collapsed: int = 0

    @property
    def ok(self) -> bool:
        """Gate 1's per-output criterion: at least one action, nothing malformed."""
        return bool(self.actions) and not self.failures

    @property
    def degenerate(self) -> bool:
        """True when the output was mostly repetition rather than a line of play."""
        return self.repeats_collapsed >= 5

    def keys(self) -> list[str]:
        return [a.key() for a in self.actions]


def _norm(s: str) -> str:
    """Collapse whitespace and drop trailing punctuation from a name."""
    return re.sub(r"\s+", " ", s).strip().strip(".,;")


def _split_list(s: str) -> tuple[str, ...]:
    """Split 'a, b and c' into names, tolerating 'and' as a separator."""
    parts = re.split(r",|\band\b", s)
    return tuple(p for p in (_norm(x) for x in parts) if p)


def parse_line(line: str) -> Action | ParseFailure | None:
    """Parse one line. Returns None for prose that never claimed to be an action."""
    text = _BOLD_RE.sub("", line)
    text = _SCAFFOLD_RE.sub("", text)
    text = _LABEL_RE.sub("", text)
    # Square brackets are meta-syntax, never part of a card name, and models
    # copy them out of a grammar that uses them to mark optional operands
    # ("CAST Lightning Strike [ TARGET Grizzly Bears]"). The grammar no longer
    # uses brackets (see common.ACTION_GRAMMAR), but stripping them here keeps
    # a correct play from being scored illegal on phrasing alone.
    text = text.replace("[", " ").replace("]", " ")
    text = text.strip()
    if not text:
        return None

    upper = text.upper()
    # The verb must be a whole word. A bare `startswith` matches the present
    # participles models narrate with — "Attacking with Swiftspear is correct"
    # parsed as ATTACK("ing with Swiftspear is correct"), and "Blocking the
    # Bears with Elves" parsed as a BLOCK with its operands inverted. Those are
    # phantom actions manufactured from prose: they inflate the action count and
    # sink the legality rate, which would have shown up as a capability finding
    # rather than the parser bug it is.
    verb = next(
        (v for v in VERBS
         if upper.startswith(v) and (len(text) == len(v) or not text[len(v)].isalpha())),
        None,
    )
    if verb is None:
        return None  # narration, not a malformed action

    body = _norm(text[len(verb):])

    if verb == "PASS":
        # "PASS" and "PASS priority" / "PASS the turn" are the same play.
        return Action("PASS", (), line)

    if verb == "MULLIGAN":
        return Action("MULLIGAN", (), line)

    if verb == "KEEP":
        # London mulligan: keeping at fewer than seven means naming what goes
        # back, so the bottomed cards are part of the play and not a footnote.
        m = re.split(r"\bBOTTOM(?:ING)?\b", body, maxsplit=1, flags=re.I)
        return Action("KEEP", _split_list(m[1]) if len(m) > 1 else (), line)

    if not body:
        return ParseFailure(line, f"{verb} with no argument")

    if verb == "PLAY":
        return Action("PLAY", (body,), line)

    if verb == "CAST":
        # TARGET / TARGETING / "targeting" all appear in real output.
        # "TARGET x", "targeting x", "Target: x" all occur in real output.
        m = re.split(r"\bTARGET(?:ING|S)?\b\s*:?", body, maxsplit=1, flags=re.I)
        name = _norm(m[0])
        if not name:
            return ParseFailure(line, "CAST with no card name")
        targets = _split_list(m[1]) if len(m) > 1 else ()
        if len(m) > 1 and not targets:
            return ParseFailure(line, "CAST ... TARGET with no target named")
        return Action("CAST", (name, *targets), line)

    if verb == "ACTIVATE":
        if ":" not in body:
            return ParseFailure(line, "ACTIVATE needs '<permanent>: <ability>'")
        perm, ability = body.split(":", 1)
        perm, ability = _norm(perm), _norm(ability)
        if not perm or not ability:
            return ParseFailure(line, "ACTIVATE needs both a permanent and an ability")
        return Action("ACTIVATE", (perm, ability), line)

    if verb == "ATTACK":
        body = re.sub(r"^\s*WITH\b\s*", "", body, flags=re.I)  # "ATTACK WITH x"
        names = _split_list(body)
        if not names:
            return ParseFailure(line, "ATTACK with no creature named")
        return Action("ATTACK", names, line)

    if verb == "BLOCK":
        # Both orderings occur in the wild and they mean OPPOSITE things:
        #   "BLOCK <blocker> -> <attacker>"   (canonical)
        #   "BLOCK <attacker> WITH <blocker>" (natural English)
        # Silently getting this backwards would invert every blocking position
        # in the eval, so each form is matched explicitly and neither is a
        # fallback for the other.
        if "->" in body:
            blocker, attacker = body.split("->", 1)
        elif re.search(r"\bWITH\b", body, re.I):
            attacker, blocker = re.split(r"\bWITH\b", body, maxsplit=1, flags=re.I)
        else:
            return ParseFailure(line, "BLOCK needs '<blocker> -> <attacker>' or "
                                      "'<attacker> with <blocker>'")
        blocker, attacker = _norm(blocker), _norm(attacker)
        if not blocker or not attacker:
            return ParseFailure(line, "BLOCK needs both a blocker and an attacker")
        return Action("BLOCK", (blocker, attacker), line)

    if verb == "ORDER TRIGGERS":
        # The operand order is RESOLUTION order, which is what
        # `common.ACTION_GRAMMAR` promises the model. It is deliberately not
        # the order the triggers go on the stack — the two are reverses of
        # each other (405.2), so a position authored against one convention
        # and scored against the other marks the correct play wrong. Stated
        # here as well as in the grammar because POSITIONS.md sends authors of
        # `legal_actions` to this file.
        names = _split_list(body)
        if len(names) < 2:
            return ParseFailure(line, "ORDER TRIGGERS needs at least two triggers")
        return Action("ORDER TRIGGERS", names, line)

    return ParseFailure(line, f"unhandled verb {verb}")  # unreachable


def parse_output(text: str) -> ParsedOutput:
    """Parse a full model response into actions, failures, and ignored prose."""
    out = ParsedOutput()
    in_fence = False
    for line in (text or "").splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if not line.strip():
            continue
        result = parse_line(line)
        if result is None:
            out.ignored.append(line.strip())
        elif isinstance(result, Action):
            # Collapse a run of the same action: repeating it is not doing it
            # more than once, and unbounded repetition is a decode failure.
            if out.actions and out.actions[-1].key() == result.key():
                out.repeats_collapsed += 1
            else:
                out.actions.append(result)
        else:
            out.failures.append(result)
    return out


def match_to_legal(action: Action, legal_actions: list[str]) -> str | None:
    """Return the legal action this one corresponds to, or None.

    Both sides go through the same parser, so `ATTACK a, b` matches
    `ATTACK b, a` and `BLOCK x -> y` matches `BLOCK y with x`. Comparing raw
    strings instead would make the closed arm measure phrasing.

    PASS always matches, whether or not the position enumerates it — see
    Section 16.13. `GAMEPLAY_SYSTEM_PROMPT` ends with "End with a single PASS",
    so every well-formed answer contains one; scoring it against
    `legal_actions` made the protocol terminator compete with the game action
    of the same name. It cost Gate 1: `pos-seed-0006` is a mulligan decision,
    which happens before any player has priority (103.4), so its
    `legal_actions` are exactly ["MULLIGAN", "KEEP"] — correctly, since there
    is no priority to pass. The model answered `KEEP` / `PASS`, doing precisely
    what it was told, and the mandated terminator scored illegal.

    Declining to act is also always available in the game sense, so this is not
    merely a protocol escape hatch. An all-PASS dodge is still caught, by the
    gate that matters: it hits no key points and scores as a blunder.
    """
    want = action.key().casefold()
    if want == "pass":
        return "PASS"
    for legal in legal_actions:
        parsed = parse_line(legal)
        if isinstance(parsed, Action) and parsed.key().casefold() == want:
            return legal
    return None


def legality(parsed: ParsedOutput, legal_actions: list[str]) -> dict:
    """How much of an output names an action from the enumerated legal set."""
    matched = [a for a in parsed.actions if match_to_legal(a, legal_actions)]
    return {
        "n_actions": len(parsed.actions),
        "n_legal": len(matched),
        "n_failures": len(parsed.failures),
        "all_legal": bool(parsed.actions) and len(matched) == len(parsed.actions),
        "illegal": [a.key() for a in parsed.actions if not match_to_legal(a, legal_actions)],
    }
