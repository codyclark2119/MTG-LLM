#!/usr/bin/env python3
"""One-time exact branch migration for the gold-set k-rules serving decision.

Deleted before merge. This applies only the already-observed decision from the
precommitted 99-question replication: flat k=3 becomes the shipped default;
`auto` remains implemented as an explicit experimental override.
"""

from pathlib import Path
import re


def sub_once(path: str, pattern: str, repl: str, flags=0) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    new, n = re.subn(pattern, repl, text, count=1, flags=flags)
    if n != 1:
        raise SystemExit(f"{path}: expected exactly one replacement, got {n}")
    p.write_text(new, encoding="utf-8")


# common.py: settle the profile from the gold replication and bump identity.
sub_once(
    "scripts/common.py",
    r"# `k_rules=AUTO_K_RULES` IS A CHANGE FROM THE FLAT k=3 21\.158 SET,.*?Flip it back with `--k-rules 3`\.\n",
    """# The 99-question gold replication settles the provisional routing choice.\n# On the matched 32B card+rulings arm, k=0 lost 14-20 with 65 ties (mean\n# delta -0.131, two-sided exact sign-test p=0.392) and fabricated citations\n# rose from 4/99 at k=3 to 21/99 at k=0. The precommitted decision rule\n# required a significant positive correctness effect AND no fabrication\n# increase; k=0 failed both. Flat k=3 therefore returns as the shipped\n# default. `AUTO_K_RULES` and `route_k_rules` remain available for explicit\n# research overrides; the experiment rejects the router as DEFAULT, not as\n# a capability. See eval/experiments/k_rules_gold99_result.md.\n""",
    re.S,
)
sub_once("scripts/common.py", r'profile_id="chat-32b-routed-v2"', 'profile_id="chat-32b-flat-k3-v3"')
sub_once("scripts/common.py", r"k_rules=AUTO_K_RULES,", "k_rules=K_RULES_NO_CARDS,")
sub_once("scripts/common.py", r'CHAT_SERVING_PROFILE_FINGERPRINT = "[0-9a-f]{12}"',
         'CHAT_SERVING_PROFILE_FINGERPRINT = "6f17affbeded"')

# chat_server.py: current-state prose and CLI help must describe the shipped policy.
sub_once(
    "scripts/chat_server.py",
    r"`--k-rules` defaults to \*\*`auto`\*\*, the per-question router \(Section 21\.156\):.*?whichever way it is later settled\.\n",
    """`--k-rules` defaults to **`3`**. The provisional `auto` default was tested\non the full 99-question gold set after Section 21.165 required replication.\nIt did not replicate: k=0 lost 14-20 with 65 ties (mean delta -0.131, p=0.392)\nand fabricated citations rose from 4/99 at k=3 to 21/99 at k=0. The\nprecommitted rule therefore restores flat k=3 for the shipped 32B profile.\n`--k-rules auto` remains available as an explicit experimental override, and\n`k_rules_used` remains recorded per answer.\n""",
    re.S,
)
sub_once(
    "scripts/chat_server.py",
    r'"service ran the 7B, where it measures -0\.02 and takes fabricated "\n\s*"citations 0/53 to 5/53\. On the 32B it costs 3/53 to 7/53 "\n\s*"fabrications, which 21\.158 called a bad trade for a rules bot and "\n\s*"21\.165 marks provisional — pass `3` for the flat grounded setting\. "',
    '"gold replication: k=0 lost 14-20 with 65 ties and raised fabricated "\n                         "citations from 4/99 to 21/99. Flat k=3 is the shipped setting; "\n                         "pass `auto` only to reproduce the experimental router. "',
)
sub_once(
    "scripts/chat_server.py",
    r"python scripts/chat_server\.py --k-rules 3\s+# flat k=3, ignoring the router",
    "python scripts/chat_server.py --k-rules auto       # experimental per-question router",
)

# The drift gate now pins the replicated 32B policy, not the provisional one.
sub_once(
    "scripts/test_chat_server.py",
    r'# 21\.144 \+0\.25 on the k=0 branch \(15/6/32, p=0\.078\), reproduced in\n\s*# 21\.158\'s own table; the k=3 branch is 21\.155/21\.157\.\n\s*"mlx-community/Qwen2\.5-32B-Instruct-4bit": AUTO_K_RULES,',
    '# Gold99 replication: k=0 lost 14-20 with 65 ties (p=0.392) and\n        # fabricated 21/99 citations versus 4/99 at k=3.\n        "mlx-community/Qwen2.5-32B-Instruct-4bit": K_RULES_NO_CARDS,',
)

# CLAUDE.md current-state policy: retain the historical 53q result, record that
# the required replication reversed the serving conclusion.
sub_once(
    "CLAUDE.md",
    r"\*\*`k_rules` defaults to `auto` on the 32B, and that is a judgement call\.\*\*.*?`k_rules_used` keeps the branches separable in the ratings either way\.\n",
    """**`k_rules` defaults to flat `3` on the 32B; the provisional `auto`\npolicy failed its required gold-set replication.** The earlier 53-question\ncard benchmark favored k=0 by +0.25 (15/6/32, p=0.078) but also raised\nfabricated citations 3/53 -> 7/53. The precommitted 99-question replication\nreversed the correctness direction (14 k=0 wins / 20 k=3 wins / 65 ties, mean\ndelta -0.131, p=0.392) and widened the grounding cost to **21/99 fabricated\ncitations at k=0 versus 4/99 at k=3**. The replication rule therefore restores\nflat k=3 as the shipped default. `--k-rules auto` remains an explicit research\noverride, and `k_rules_used` keeps treatments separable in stored ratings.\n""",
    re.S,
)

print("flat k=3 serving reconciliation applied")
