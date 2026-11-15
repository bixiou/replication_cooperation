"""Three counts of paper/cooperation/cooperation.tex: a Wilson interval (Section S6), invariant reachable rest points
of symmetric 3x3 games (Section S5), and reachable rest points outside IR(G) (Section 5.3).

Logic.
A. The 95% Wilson interval of the farsighted sample rate 1,245 of 3,000 (coop_table_B.out, line 35; Online
   Appendix S6, seed 20261104).
B. Online Appendix S5: among the 362,880 symmetric 3x3 games in which player 1's payoffs rank the nine profiles
   (u2(x, y) = u1(y, x)), count the games whose perfectly transparent equilibrium limit is empty and, among them,
   those with a nonempty reachable rest point that the exchange (x, y) -> (y, x) maps to itself. A reachable rest
   point is a nonempty self-consistent set reachable from S by applying the verdicts of any nonempty blocks of
   players (coop_device.reachable).
C. Section 5.3: a nonempty reachable rest point need not lie in IR(G), the profiles giving each player at least
   her pure security level. Count, among the 576 strict 2x2 games, those with a nonempty reachable rest point
   containing a profile outside IR(G), and print the example (1,1), (2,4) / (3,3), (4,2) with its reachable rest
   points and the resolved perfect prediction under both orders.
"""
import itertools
import math

from coop_standalone import profiles, pte, ir
from coop_device import reachable
from coop_sequential import jacobi_resolved

# A. Wilson interval
k, n, z = 1245, 3000, 1.959963984540054
p = k / n
centre = (p + z * z / (2 * n)) / (1 + z * z / n)
half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
print(f'A. farsighted sample: {k}/{n} = {p:.4f}; 95% Wilson interval [{centre - half:.3f}, {centre + half:.3f}]')

# B. symmetric 3x3 ranking games
P3 = profiles((3, 3))
n_games = n_empty = n_inv = 0
for perm in itertools.permutations(range(1, 10)):
    u1 = dict(zip(P3, perm))
    U = {(x, y): (u1[(x, y)], u1[(y, x)]) for (x, y) in P3}
    n_games += 1
    if pte(U, (3, 3)):
        continue
    n_empty += 1
    _, rests = reachable(U, (3, 3))
    if any(tuple(sorted((y, x) for (x, y) in E)) == E for E in rests):
        n_inv += 1
print(f'B. symmetric 3x3 ranking games: {n_games}; empty limit: {n_empty}; of these, with a nonempty reachable rest'
      f' point mapped to itself by the exchange: {n_inv}')

# C. reachable rest points outside IR(G), strict 2x2 games
P2 = profiles((2, 2))
perms = list(itertools.permutations(range(1, 5)))
n_out = 0
for q1 in perms:
    for q2 in perms:
        U = {s: (q1[j], q2[j]) for j, s in enumerate(P2)}
        R = set(ir(U, (2, 2)))
        _, rests = reachable(U, (2, 2))
        if any(not set(E) <= R for E in rests):
            n_out += 1
print(f'C. strict 2x2 games: {len(perms) ** 2}; with a nonempty reachable rest point not contained in IR(G): {n_out}')
EX = {(0, 0): (1, 1), (0, 1): (2, 4), (1, 0): (3, 3), (1, 1): (4, 2)}
_, rests = reachable(EX, (2, 2))
print(f'   example (1,1), (2,4) / (3,3), (4,2): IR(G) = {ir(EX, (2, 2))}; nonempty reachable rest points'
      f' {sorted(rests)}')
for order in [(0, 1), (1, 0)]:
    try:
        print(f'   resolved perfect prediction, order {order}: {jacobi_resolved(EX, (2, 2), order)}')
    except TypeError as e:
        print(f'   resolved perfect prediction: signature differs ({e})')
        break
