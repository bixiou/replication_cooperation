"""coop_far_td_draws.py: the farsighted count of Table 1's entries over several draws (2026-11-05).

Logic. The 24 tie-free symmetric perturbations of the traveler's dilemmas with M = 4..9 and r in {3/2, 2, 5/2, 3}, built
as in coop_table_B.perturbed_tds (player 1's payoff at (x, y) plus k(x, y)/1000, k a uniformly random ranking of the
profiles; player 2's by symmetry). For each seed, count the games in which the largest consistent set with joint moves
(LCS2) contains an asymmetric profile, and the games in which it differs from {(z, z)}. Seed 7 is the draw of
coop_table_B.out; seeds 1..30 show the range.
"""
import random
from pathlib import Path

import coop_table_B as t

OUT = Path(__file__).with_suffix(".out")
_lines = []


def say(x=""):
    print(x)
    _lines.append(str(x))


def counts(seed):
    games = [G for G in t.perturbed_tds(random.Random(seed)) if t.cp(*G)]
    asym = other = 0
    for U, sh in games:
        E = t.CON["LCS2"](U, sh)
        asym += any(s[0] != s[1] for s in E)
        other += E != frozenset([t.zz(U, sh)])
    return len(games), asym, other


tally = {}
for seed in [7] + [s for s in range(1, 31) if s != 7]:
    n, a, o = counts(seed)
    tally[a] = tally.get(a, 0) + 1
    say(f"seed {seed}: games with copying pays {n}; LCS2 contains an asymmetric profile in {a}; differs from {{(z,z)}} in {o}")
say(f"distribution of the asymmetric count over the 30 seeds: {dict(sorted(tally.items()))}")
OUT.write_text("\n".join(_lines) + "\n")
