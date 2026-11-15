"""coop_pg_weighted_scan.py: exact PTE limit of the weighted public goods game for every a in (0, 1)
(2026-11-05).

Logic. Three players; player i contributes x_i in {0, w_i}, w = (1, 9/10, 4/5); u_i = 1 - x_i + a * sum_j x_j. Every
comparison of two payoffs of the same player is linear in a, so the PTE limit is constant between consecutive
breakpoints a = (x_i - x_i') / (X - X') (X the total contribution). The script lists the breakpoints in (0, 1),
computes the limit exactly (Fractions) at each breakpoint and at the midpoint of each interval, and reports the
values of a in (0, 1) at which the game has ties.
"""
import itertools
from fractions import Fraction as Fr
from pathlib import Path

OUT = Path(__file__).with_suffix(".out")
_lines = []
W = (Fr(1), Fr(9, 10), Fr(4, 5))
P = list(itertools.product((0, 1), repeat=3))


def say(x=""):
    print(x)
    _lines.append(str(x))


def pay(s, a):
    X = sum(W[j] * s[j] for j in range(3))
    return tuple(1 - W[i] * s[i] + a * X for i in range(3))


def guarantee(U, E, i):
    return max(min(U[s][i] for s in E if s[i] == t) for t in (0, 1) if any(s[i] == t for s in E))


def pte(a):
    U = {s: pay(s, a) for s in P}
    E = frozenset(P)
    while E:
        m = [guarantee(U, E, i) for i in range(3)]
        F = frozenset(s for s in E if all(U[s][i] >= m[i] for i in range(3)))
        if F == E:
            break
        E = F
    return sorted(E)


def ties(a):
    U = {s: pay(s, a) for s in P}
    return any(len({U[s][i] for s in P}) < len(P) for i in range(3))


bps = set()
for i in range(3):
    for s, t in itertools.combinations(P, 2):
        dx = W[i] * s[i] - W[i] * t[i]
        dX = sum(W[j] * (s[j] - t[j]) for j in range(3))
        if dX != 0:
            a = dx / dX
            if 0 < a < 1:
                bps.add(a)
bps = sorted(bps)
say(f"breakpoints in (0,1): {[str(b) for b in bps]}")
pts = [Fr(0) + bps[0] / 2]
for k, b in enumerate(bps):
    pts.append(b)
    pts.append((b + (bps[k + 1] if k + 1 < len(bps) else Fr(1))) / 2)
for a in pts:
    say(f"a = {str(a):>8} ({float(a):.4f}): ties {ties(a)}; limit {pte(a)}")
say(f"values of a in (0,1) with ties: {[str(a) for a in bps if ties(a)]}")
OUT.write_text("\n".join(_lines) + "\n")
