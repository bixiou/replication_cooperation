"""coop_pg3_path.py: a shortest sequence of verdict blocks reaching {CCC} in the three-player binary public goods
game, u_i = 1 - x_i + a * sum x_j, a = 3/4 and a = 11/20 (1 = contribute, 0 = free ride), with the guarantees used
at each step; and the ex ante payoffs of reasoning in turn against full contribution."""
import itertools
import sys
from fractions import Fraction as Fr
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from coop_pg_n import game, step, guarantee  # noqa: E402

OUT = Path(__file__).with_suffix(".out")
lines = [__doc__.splitlines()[0]]
n = 3
for a in (Fr(3, 4), Fr(11, 20)):
    U = game(n, a)
    blocks = [B for r in range(1, n + 1) for B in itertools.combinations(range(n), r)]
    S = frozenset(U)
    parent = {S: None}
    frontier = [S]
    target = frozenset({(1, 1, 1)})
    while frontier and target not in parent:
        nxt = []
        for E in frontier:
            for B in blocks:
                F = step(U, E, B)
                if F not in parent:
                    parent[F] = (E, B)
                    nxt.append(F)
        frontier = nxt
    path = []
    E = target
    while parent[E] is not None:
        P, B = parent[E]
        path.append((P, B, E))
        E = P
    path.reverse()
    lines.append(f"a = {a}: shortest path to {{CCC}} has {len(path)} steps")
    for P, B, E in path:
        g = {i + 1: guarantee(U, P, i) for i in B}
        lines.append(f"  from {len(P)} profiles {sorted(''.join('FC'[x] for x in s) for s in P)}, block "
                     f"{tuple(i + 1 for i in B)}, guarantees {g} -> {sorted(''.join('FC'[x] for x in s) for s in E)}")
    exante = (2 * (2 * a) + (1 + 2 * a)) / 3
    lines.append(f"  ex ante payoff under reasoning in turn (two contributors, free rider uniform): {exante}; "
                 f"full contribution: {3 * a}")
OUT.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
