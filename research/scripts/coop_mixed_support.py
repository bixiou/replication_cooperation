"""coop_mixed_support.py: the games W and G* of the Online Appendix of paper/cooperation/cooperation.tex (Section S8,
"Pareto optimality and equal treatment"), in exact arithmetic.

Run: python3 research/scripts/coop_mixed_support.py > research/scripts/coop_mixed_support.out

Logic.
- Mixed security level of a player in a two-player game: v = max over her mixed strategies x of the minimum over the
  other's pure strategies j of x.A_j. The maximum of this linear program is attained at a vertex of
  {x >= 0, sum x = 1, t <= x.A_j for all j}; every vertex solves sum x = 1 together with n of the constraints
  x_k = 0 or x.A_j = t (n = number of own strategies), so we enumerate these choices, solve each square system with
  Fractions, keep the feasible solutions and take the largest t.
- W (2x3, rows a, a'; columns b, b', b''): no ties, the Pareto-optimal pure profiles, the mixed security levels, and
  whether a Pareto-optimal pure profile gives each player at least her mixed security level. A mixed profile's support is
  the product of the players' supports, so a support made of Pareto-optimal profiles is a single one of them here.
- G* (symmetric 3x3, player 1's payoffs given, player 2's at (x,y) equal to player 1's at (y,x)): no ties, copying
  pays (whenever the players choose differently, the one who gets less gets more by copying the other's strategy),
  the best symmetric profile and its payoff, the uniform mixture's payoff against each pure strategy, and the exact
  mixed security level.
"""
from fractions import Fraction as F
from itertools import combinations, product


def solve(rows, rhs):
    # Gaussian elimination over the rationals; None if singular
    n = len(rows)
    m = [list(r) + [b] for r, b in zip(rows, rhs)]
    for c in range(n):
        piv = next((r for r in range(c, n) if m[r][c] != 0), None)
        if piv is None:
            return None
        m[c], m[piv] = m[piv], m[c]
        for r in range(n):
            if r != c and m[r][c] != 0:
                f = m[r][c] / m[c][c]
                m[r] = [a - f * b for a, b in zip(m[r], m[c])]
    return [m[r][n] / m[r][r] for r in range(n)]


def security(A):
    # A[k][j]: the player's payoff with her strategy k against the other's strategy j
    n, J = len(A), len(A[0])
    cons = [('x', k) for k in range(n)] + [('col', j) for j in range(J)]
    best = None
    for chosen in combinations(cons, n):
        rows, rhs = [[F(1)] * n + [F(0)]], [F(1)]  # unknowns x_0..x_{n-1}, t
        for kind, idx in chosen:
            if kind == 'x':
                rows.append([F(int(k == idx)) for k in range(n)] + [F(0)])
            else:
                rows.append([F(A[k][idx]) for k in range(n)] + [F(-1)])
            rhs.append(F(0))
        sol = solve(rows, rhs)
        if sol is None:
            continue
        x, t = sol[:n], sol[n]
        if min(x) < 0 or any(sum(x[k] * A[k][j] for k in range(n)) < t for j in range(J)):
            continue
        if best is None or t > best[0]:
            best = (t, x)
    return best


def fmt(v):
    return str(v) if not isinstance(v, (list, tuple)) else '(' + ', '.join(fmt(a) for a in v) + ')'


print('== W ==')
rows, cols = ['a', "a'"], ['b', "b'", "b''"]
W = {('a', 'b'): (4, 4), ('a', "b'"): (1, 6), ('a', "b''"): (9, 5),
     ("a'", 'b'): (2, 8), ("a'", "b'"): (3, 3), ("a'", "b''"): (5, 1)}
for i in range(2):
    vals = [W[s][i] for s in W]
    print(f'player {i + 1}: payoffs {vals}, no ties: {len(set(vals)) == len(vals)}')
po = [s for s in W if not any(all(W[o][i] >= W[s][i] for i in range(2)) and W[o] != W[s] for o in W)]
print('Pareto-optimal pure profiles:', ', '.join(f'({r},{c}) paying {W[(r, c)]}' for r, c in po))
v1, x1 = security([[W[(r, c)][0] for c in cols] for r in rows])
v2, x2 = security([[W[(r, c)][1] for r in rows] for c in cols])
print(f'mixed security levels: player 1 {v1} (rows {fmt(x1)}), player 2 {v2} (columns {fmt(x2)})')
for r, c in po:
    u = W[(r, c)]
    print(f'({r},{c}) gives player 1 {u[0]} {">=" if u[0] >= v1 else "<"} {v1} and player 2 {u[1]} '
          f'{">=" if u[1] >= v2 else "<"} {v2}')
prods = [(R, C) for R in range(1, 3) for C in range(1, 4)]
ok = []
for R, C in prods:
    for rs in combinations(rows, R):
        for cs in combinations(cols, C):
            if all(s in po for s in product(rs, cs)):
                ok.append((rs, cs))
print('supports (products of the players\' supports) made only of Pareto-optimal profiles:',
      ', '.join('{' + ','.join(rs) + '} x {' + ','.join(cs) + '}' for rs, cs in ok))
print('each such support is a single profile outside IR^m:',
      all(len(rs) == 1 and len(cs) == 1 and (W[(rs[0], cs[0])][0] < v1 or W[(rs[0], cs[0])][1] < v2)
          for rs, cs in ok))

print()
print('== G* ==')
S = ['a', 'b', 'c']
A1 = {'a': [-1, 10, -5], 'b': [-3, -2, 11], 'c': [12, -4, 0]}
u1 = {(x, y): A1[x][S.index(y)] for x in S for y in S}
u2 = {(x, y): u1[(y, x)] for x in S for y in S}
print('player 1 payoffs (rows a, b, c):', [A1[s] for s in S])
print('no ties: player 1', len(set(u1.values())) == 9, '; player 2', len(set(u2.values())) == 9)
pays = True
for x, y in product(S, S):
    if x == y:
        continue
    if u1[(x, y)] < u2[(x, y)]:  # player 1 is worse off; copying means playing y
        gain = u1[(y, y)] > u1[(x, y)]
        who, before, after = 1, u1[(x, y)], u1[(y, y)]
    else:
        gain = u2[(x, x)] > u2[(x, y)]
        who, before, after = 2, u2[(x, y)], u2[(x, x)]
    pays &= gain
    print(f'  ({x},{y}) pays ({u1[(x, y)]},{u2[(x, y)]}); player {who} copies: {before} -> {after}')
print('copying pays:', pays)
best = max(S, key=lambda z: u1[(z, z)])
print(f'best symmetric profile ({best},{best}) gives each player {u1[(best, best)]}')
unif = [sum(F(A1[x][S.index(y)], 3) for x in S) for y in S]
print('uniform mixture against a, b, c:', [str(v) for v in unif], 'so it guarantees', min(unif))
v, x = security([A1[s] for s in S])
print(f'exact mixed security level {v} = {float(v):.4f}, attained by {fmt(x)}')
print(f'({best},{best}) in IR^m:', u1[(best, best)] >= v)
