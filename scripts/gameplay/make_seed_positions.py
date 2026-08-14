"""Emit the seed position fixtures.

    python scripts/gameplay/make_seed_positions.py

These are MACHINE-DRAFTED and exist to exercise the plumbing end to end — the
renderer, the action parser, the validator, and the scoring path — not to feed
the gate. Section 14.6 measured hand-authored rubrics beating machine drafts
(inter-judge r +0.30 -> +0.62), so they live in their own file and never touch
data/gold/positions.jsonl. Every record carries `seed_note`, and the position
report prints a warning whenever a seed is present in a scored run.

They are kept as code rather than as hand-edited JSONL so that a schema change
(the `attacking` flag was added after the first draft modelled attackers as
stack entries) can be re-applied by re-running this.
"""

import json
from pathlib import Path

SOURCE = "seed:claude"
NOTE = ("machine-drafted plumbing fixture — verifies the renderer, parser and "
        "eval path end to end. NOT gate material: Section 14.6 showed machine "
        "rubrics score materially worse than hand-authored ones, so Gate 3 "
        "needs judge-authored positions in data/gold/positions.jsonl.")


def land(name, controller, n=1, tapped=False):
    return [{"controller": controller, "card": name, "tapped": tapped} for _ in range(n)]


def creature(name, controller, pt, tapped=False, sick=False, counters=None, attach=None,
             attacking=False):
    d = {"controller": controller, "card": name, "pt": pt, "tapped": tapped}
    if attacking:
        d["attacking"] = True
    if sick:
        d["summoning_sick"] = True
    if counters:
        d["counters"] = counters
    if attach:
        d["attachments"] = attach
    return [d]


P = []

# 1 --- combat math: remove the blocker first, and prowess pays for it -------
P.append({
    "id": "pos-seed-0001", "format": "standard",
    "turn": 4, "phase": "precombat main", "active_player": "you", "priority": "you",
    "players": {
        "you": {"life": 18, "hand": ["Lightning Strike", "Mountain"], "library_count": 26},
        "opp": {"life": 14, "hand_count": 3, "library_count": 27},
    },
    "battlefield": (land("Mountain", "you", 3) + creature("Monastery Swiftspear", "you", "1/2")
                    + creature("Grizzly Bears", "opp", "2/2") + land("Island", "opp", 3)),
    "mana_available": "{R}{R}{R}",
    "legal_actions": [
        "PLAY Mountain",
        "CAST Lightning Strike TARGET Grizzly Bears",
        "ATTACK Monastery Swiftspear",
        "PASS",
    ],
    "answer": ("Cast Lightning Strike on Grizzly Bears in your main phase, then attack "
               "with Monastery Swiftspear for 2."),
    "key_points": [
        "Lightning Strike kills Grizzly Bears before combat, clearing the only blocker",
        "Prowess triggers on Lightning Strike, so Swiftspear attacks as a 2/3 and deals 2",
    ],
    "common_errors": [
        "Attacks with Monastery Swiftspear into the untapped Grizzly Bears, losing it for nothing",
        "Leaves Grizzly Bears alive, pointing Lightning Strike at the opponent instead",
    ],
    "rule_citations": ["702.108a"],
    "category": "combat math", "difficulty": "intermediate",
})

# 2 --- blocking: forced block, and which blocker ---------------------------
P.append({
    "id": "pos-seed-0002", "format": "standard",
    "turn": 9, "phase": "declare blockers", "active_player": "opp", "priority": "you",
    "players": {
        "you": {"life": 5, "hand": [], "library_count": 18},
        "opp": {"life": 12, "hand_count": 2, "library_count": 20},
    },
    "battlefield": (creature("Wall of Omens", "you", "0/4") + creature("Centaur Courser", "you", "3/3")
                    + land("Plains", "you", 4)
                    + creature("Serra Angel", "opp", "4/4", tapped=True, attacking=True)
                    + creature("Grizzly Bears", "opp", "2/2", tapped=True, attacking=True)
                    + land("Forest", "opp", 5)),
    "legal_actions": [
        "BLOCK Centaur Courser -> Grizzly Bears",
        "BLOCK Wall of Omens -> Grizzly Bears",
        "PASS",
    ],
    "answer": ("Block Grizzly Bears with Centaur Courser. Neither creature can block Serra "
               "Angel, so you take 4 and end at 1."),
    "key_points": [
        "You must block Grizzly Bears — 4 from Serra Angel plus 2 from the Bears is lethal at 5 life",
        "Block with Centaur Courser rather than Wall of Omens: 3 power kills the 2/2 and the 3/3 survives",
        "Neither blocker has flying or reach, so Serra Angel's 4 damage is unavoidable",
    ],
    "common_errors": [
        "Tries to block the flier: assigns Wall of Omens or Centaur Courser to Serra Angel",
        "Uses Wall of Omens as the blocker, which cannot kill a 2/2",
        "Declines to block and dies to 6 total damage at 5 life",
    ],
    "rule_citations": ["509.1a", "510.1a"],
    "category": "blocking", "difficulty": "basic",
})

# 3 --- removal timing: don't spend premium removal on a bad creature -------
P.append({
    "id": "pos-seed-0003", "format": "standard",
    "turn": 3, "phase": "precombat main", "active_player": "you", "priority": "you",
    "players": {
        "you": {"life": 20, "hand": ["Doom Blade", "Swamp"], "library_count": 30},
        "opp": {"life": 20, "hand_count": 5, "library_count": 29},
    },
    "battlefield": (land("Swamp", "you", 3) + creature("Grizzly Bears", "opp", "2/2")
                    + land("Forest", "opp", 3)),
    "mana_available": "{B}{B}{B}",
    "legal_actions": ["PLAY Swamp", "CAST Doom Blade TARGET Grizzly Bears", "PASS"],
    "answer": ("Play Swamp and pass, holding Doom Blade. Grizzly Bears is not worth premium "
               "instant-speed removal this early."),
    "key_points": [
        "Hold Doom Blade rather than spending it on a vanilla 2/2",
        "Play the land and pass with mana open, so the removal can answer a real threat on their turn",
    ],
    "common_errors": [
        "Casts Doom Blade on Grizzly Bears, spending the answer on the least threatening permanent",
        "Passes without playing a land, falling a turn behind on mana for no reason",
    ],
    "rule_citations": [],
    "category": "removal timing", "difficulty": "intermediate",
})

# 4 --- land sequencing: only one land casts the spell ----------------------
P.append({
    "id": "pos-seed-0004", "format": "standard",
    "turn": 2, "phase": "precombat main", "active_player": "you", "priority": "you",
    "players": {
        "you": {"life": 20, "hand": ["Forest", "Island", "Llanowar Elves", "Counterspell"],
                "library_count": 33},
        "opp": {"life": 20, "hand_count": 6, "library_count": 33},
    },
    "battlefield": land("Island", "you", 1) + land("Mountain", "opp", 2),
    "legal_actions": [
        "PLAY Forest", "PLAY Island", "CAST Llanowar Elves", "PASS",
    ],
    "answer": "Play Forest, then cast Llanowar Elves with the {G}.",
    "key_points": [
        "Play Forest, not Island — it is the only green source, and without it Llanowar Elves is stranded",
        "Cast Llanowar Elves this turn; the whole point of the card is the turn of acceleration",
    ],
    "common_errors": [
        "Plays Island, leaving no green source and stranding Llanowar Elves in hand",
        "Holds Llanowar Elves to keep mana up for Counterspell, which two lands cannot cast",
    ],
    "rule_citations": [],
    "category": "land sequencing", "difficulty": "intermediate",
})

# 5 --- race vs stabilize: you cannot win the race, so don't try ------------
P.append({
    "id": "pos-seed-0005", "format": "standard",
    "turn": 8, "phase": "precombat main", "active_player": "you", "priority": "you",
    "players": {
        "you": {"life": 6, "hand": ["Lightning Strike"], "library_count": 15},
        "opp": {"life": 7, "hand_count": 1, "library_count": 16},
    },
    "battlefield": (land("Mountain", "you", 3) + creature("Centaur Courser", "you", "3/3")
                    + creature("Grizzly Bears", "opp", "2/2")
                    + creature("Elite Vanguard", "opp", "2/1") + land("Plains", "opp", 4)),
    "mana_available": "{R}{R}{R}",
    "legal_actions": [
        "CAST Lightning Strike TARGET Grizzly Bears",
        "CAST Lightning Strike TARGET Elite Vanguard",
        "ATTACK Centaur Courser",
        "PASS",
    ],
    "answer": ("Lightning Strike the Grizzly Bears and keep Centaur Courser home as a blocker. "
               "You cannot win the race, so cut their clock instead."),
    "key_points": [
        "You cannot win the race: Centaur Courser plus Lightning Strike is 6 damage against 7 life",
        "Kill Grizzly Bears, cutting the incoming clock from 4 damage per turn to 2",
        "Keep Centaur Courser back as a blocker — at 6 life you cannot trade it away in an attack",
    ],
    "common_errors": [
        "Attacks with Centaur Courser, leaving no blocker while at 6 life",
        "Points Lightning Strike at the opponent's face, which neither kills them nor slows the clock",
    ],
    "rule_citations": [],
    "category": "race vs stabilize", "difficulty": "advanced",
})

# 6 --- mulligan: six lands is not a keep ----------------------------------
P.append({
    "id": "pos-seed-0006", "format": "standard",
    "turn": 0, "phase": "opening hand, on the play", "active_player": "you", "priority": "you",
    "players": {
        "you": {"life": 20, "library_count": 53,
                "hand": ["Mountain", "Mountain", "Mountain", "Mountain", "Mountain",
                         "Mountain", "Lightning Strike"]},
        "opp": {"life": 20, "hand_count": 7, "library_count": 53},
    },
    "battlefield": [],
    "legal_actions": ["MULLIGAN", "KEEP"],
    "answer": "Mulligan. Six lands and one Lightning Strike is effectively a one-card hand.",
    "key_points": [
        "Mulligan — six lands and a single spell cannot function as a keep",
        "The extra lands are dead draws, so this hand is one relevant card and a pile of Mountains",
    ],
    "common_errors": [
        "Keeps because the hand has lands and is not a mulligan for colour or count",
        "Keeps on the reasoning that a six-lander guarantees hitting land drops",
    ],
    "rule_citations": [],
    "category": "mulligan", "difficulty": "basic",
})

# 7 --- combat math: the double block that loses two for one ----------------
P.append({
    "id": "pos-seed-0007", "format": "standard",
    "turn": 6, "phase": "declare blockers", "active_player": "opp", "priority": "you",
    "players": {
        "you": {"life": 15, "hand": [], "library_count": 22},
        "opp": {"life": 17, "hand_count": 3, "library_count": 21},
    },
    "battlefield": (creature("Grizzly Bears", "you", "2/2") + creature("Elite Vanguard", "you", "2/1")
                    + land("Forest", "you", 4)
                    + creature("Centaur Courser", "opp", "3/3", tapped=True, attacking=True)
                    + land("Forest", "opp", 4)),
    "legal_actions": [
        "BLOCK Grizzly Bears -> Centaur Courser",
        "BLOCK Elite Vanguard -> Centaur Courser",
        "PASS",
    ],
    "answer": "Take the 3. Any block here loses more than it gains at 15 life.",
    "key_points": [
        "Do not block — Centaur Courser's 3 toughness survives either single blocker",
        "At 15 life, 3 damage is cheaper than any creature you would lose blocking it",
    ],
    "common_errors": [
        "Double blocks, letting the attacker assign 2 to Grizzly Bears and 1 to Elite Vanguard and kill both",
        "Chump blocks with Elite Vanguard, trading a creature for 3 damage that does not threaten anything",
        "Blocks with Grizzly Bears, which dies without killing the Courser",
    ],
    "rule_citations": ["510.1c"],
    "category": "combat math", "difficulty": "advanced",
})

# 8 --- removal timing: use it now, the threat is the clock ----------------
P.append({
    "id": "pos-seed-0008", "format": "standard",
    "turn": 7, "phase": "declare attackers", "active_player": "opp", "priority": "you",
    "players": {
        "you": {"life": 10, "hand": ["Doom Blade"], "library_count": 19},
        "opp": {"life": 4, "hand_count": 0, "library_count": 20},
    },
    "battlefield": (land("Swamp", "you", 4)
                    + creature("Serra Angel", "opp", "4/4", tapped=True, attacking=True)
                    + land("Plains", "opp", 5)),
    "mana_available": "{B}{B}{B}{B}",
    "legal_actions": ["CAST Doom Blade TARGET Serra Angel", "PASS"],
    "answer": "Doom Blade the Serra Angel now. It is white, so it is a legal target, and it is their only threat.",
    "key_points": [
        "Doom Blade the Serra Angel — it is nonblack, so it is a legal target",
        "Kill it now: with an empty opponent hand it is the entire clock, and taking 4 puts you at 6 for nothing",
    ],
    "common_errors": [
        "Takes 4 while holding removal that cleanly answers the only threat on the board",
        "Believes Doom Blade cannot target Serra Angel, or holds it for a better target that an empty hand cannot produce",
    ],
    "rule_citations": [],
    "category": "removal timing", "difficulty": "intermediate",
})

for p in P:
    p.setdefault("graveyards", {"you": [], "opp": []})
    p["source"] = SOURCE
    p["seed_note"] = NOTE
    p["rubric_source"] = "machine-drafted (seed fixture)"

out = Path("data/gold/positions_seed.jsonl")
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as f:
    for p in P:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")
print(f"{len(P)} seed positions -> {out}")
