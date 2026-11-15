# /// script
# dependencies = ["scipy", "numpy"]
# ///
"""Independent checks of side claims of the paper:
main text: TD has a unique rationalizable profile for r>1 (intro); Prop 3(ii)-(iii) with mixed and
correlated profiles (grid evidence); Macy-Flache 'fear' equivalence on ordinal symmetric 2x2 games.
Online Appendix: volunteer's dilemma (Prop S1), the 3x3 example with x,y,z, public goods games, the uniform law
(search for a 3-player witness), mixed trilemma witnesses (grid evidence), G_{1/2} at 3/7, magical thinking 0.32,
Nash-2 in matching pennies, look-back rules (S1), Haller-type swapped PD.
"""
import math
import random
import sys
from fractions import Fraction as F
from itertools import product, permutations, combinations

import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from indep_coop_lib import *  # noqa

NFAIL = 0


def check(name, cond, detail=""):
    global NFAIL
    if not cond:
        NFAIL += 1
    print(("PASS " if cond else "FAIL ") + name + ((" | " + str(detail)) if detail else ""))


# ------------------------------------------------------------- TD rationalizability
print("=== TD: iterated elimination of strategies strictly dominated by mixtures (r>1)")


def iesds_sym(M, r):
    claims = list(range(2, M + 1))

    def pay(x, y):
        if x == y:
            return x
        return x + r if x < y else y - r
    alive = claims[:]
    changed = True
    while changed:
        changed = False
        for c in alive[:]:
            others = [d for d in alive if d != c]
            if not others:
                continue
            # LP: maximize eps s.t. sum_d w_d pay(d,y) >= pay(c,y) + eps for all y in alive, w in simplex
            k = len(others)
            cobj = np.zeros(k + 1)
            cobj[-1] = -1
            A_ub = []
            b_ub = []
            for y in alive:
                row = [-pay(d, y) for d in others] + [1]
                A_ub.append(row)
                b_ub.append(-pay(c, y))
            A_eq = [[1] * k + [0]]
            res = linprog(cobj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1], bounds=[(0, None)] * k + [(None, 1)])
            if res.status == 0 and -res.fun > 1e-9:
                alive.remove(c)
                changed = True
                break
    return alive


ok = True
for M in [3, 5, 8, 12, 20]:
    for r in [1.1, 1.5, 2, 3]:
        al = iesds_sym(M, r)
        if al != [2]:
            ok = False
            print("   survivors", M, r, al)
check("TD r>1: only the claim 2 survives iterated strict dominance by mixtures (M in 3,5,8,12,20; r in 1.1,1.5,2,3)", ok)

# ------------------------------------------------------------- Macy-Flache fear
print("=== 'copying pays iff fear' in symmetric 2x2 social dilemmas (ordinal values 1..4 with ties allowed)")
ok = True
for R, S, T, P in product(range(1, 5), repeat=4):
    # social dilemma (Macy-Flache): R>P, R>S, 2R>T+S, and (T>R or P>S)
    if not (R > P and R > S and 2 * R > T + S and (T > R or P > S)):
        continue
    A = [[R, S], [T, P]]  # strategy 0 = C
    if copying_pays(A) != (P > S):
        ok = False
        print("   counterexample", (T, R, P, S))
check("copying pays iff P>S on all symmetric 2x2 social dilemmas with values in 1..4", ok)

# ------------------------------------------------------------- Prop 3(ii)-(iii) mixed and correlated (grid evidence)
print("=== Prop 3(ii)-(iii): mixed and correlated profiles (grid evidence, step 1/400)")
K = 400
grid = np.linspace(0, 1, K + 1)
Pm, Qm = np.meshgrid(grid, grid, indexing="ij")  # P: prob of first strategy for player 1, Q for player 2


def mixed_payoffs(A, B):
    u1 = Pm * Qm * A[0][0] + Pm * (1 - Qm) * A[0][1] + (1 - Pm) * Qm * A[1][0] + (1 - Pm) * (1 - Qm) * A[1][1]
    u2 = Pm * Qm * B[0][0] + Pm * (1 - Qm) * B[0][1] + (1 - Pm) * Qm * B[1][0] + (1 - Pm) * (1 - Qm) * B[1][1]
    return u1, u2


def grid_guarantee(mask, u, axis):
    """player commits to a grid strategy (index along axis); guarantee = max over strategies with nonempty
    section of min of u over the section."""
    best = -np.inf
    n = mask.shape[axis]
    for k in range(n):
        sec = mask[k, :] if axis == 0 else mask[:, k]
        if sec.any():
            vals = (u[k, :] if axis == 0 else u[:, k])[sec]
            best = max(best, vals.min())
    return best


A, B = [[0, 3], [1, 2]], [[0, 1], [3, 2]]  # chicken, first strategy A
u1, u2 = mixed_payoffs(A, B)
mask = np.ones_like(u1, dtype=bool)
g1 = [grid_guarantee(mask, u1, 0), grid_guarantee(mask, u2, 1)]
mask1 = (u1 >= g1[0] - 1e-12) & (u2 >= g1[1] - 1e-12)
g2 = [grid_guarantee(mask1, u1, 0), grid_guarantee(mask1, u2, 1)]
mask2 = mask1 & (u1 >= g2[0] - 1e-12) & (u2 >= g2[1] - 1e-12)
check("chicken mixed: guarantees (1,1) then (3,3), limit empty", abs(g1[0] - 1) < 1e-9 and abs(g1[1] - 1) < 1e-9 and abs(g2[0] - 3) < 1e-9 and abs(g2[1] - 3) < 1e-9 and not mask2.any(), (g1, g2))

# correlated: distributions on 4 profiles (grid of step 1/40); commitments to pure strategies only
pts = []
L = 40
for a in range(L + 1):
    for b in range(L + 1 - a):
        for c in range(L + 1 - a - b):
            d = L - a - b - c
            pts.append((a / L, b / L, c / L, d / L))  # (AA, AY, YA, YY)
pts = np.array(pts)
U1 = pts @ np.array([0, 3, 1, 2])
U2 = pts @ np.array([0, 1, 3, 2])
secA1 = pts[:, 2] + pts[:, 3] == 0  # player 1 plays A for sure
secY1 = pts[:, 0] + pts[:, 1] == 0
secA2 = pts[:, 1] + pts[:, 3] == 0
secY2 = pts[:, 0] + pts[:, 2] == 0


def corr_guar(mask, U, secs):
    vals = [U[mask & s].min() for s in secs if (mask & s).any()]
    return max(vals)


m = np.ones(len(pts), dtype=bool)
c1 = [corr_guar(m, U1, [secA1, secY1]), corr_guar(m, U2, [secA2, secY2])]
m1 = m & (U1 >= c1[0] - 1e-12) & (U2 >= c1[1] - 1e-12)
c2 = [corr_guar(m1, U1, [secA1, secY1]), corr_guar(m1, U2, [secA2, secY2])]
m2 = m1 & (U1 >= c2[0] - 1e-12) & (U2 >= c2[1] - 1e-12)
check("chicken correlated: guarantees (1,1) then (3,3), limit empty", c1 == [1, 1] and c2 == [3, 3] and not m2.any(), (c1, c2))

# PD (5,3,1,0) mixed, first strategy C
A, B = [[3, 0], [5, 1]], [[3, 5], [0, 1]]
u1, u2 = mixed_payoffs(A, B)
check("PD mixed payoffs u1 = 1 - p + 4q - pq", np.allclose(u1, 1 - Pm + 4 * Qm - Pm * Qm))
mask = np.ones_like(u1, dtype=bool)
g1 = [grid_guarantee(mask, u1, 0), grid_guarantee(mask, u2, 1)]
mask1 = (u1 >= g1[0] - 1e-12) & (u2 >= g1[1] - 1e-12)
g2 = [grid_guarantee(mask1, u1, 0), grid_guarantee(mask1, u2, 1)]
check("PD mixed: guarantees (1,1) in both rounds (grid), so the limit is {u1>=1,u2>=1} and contains (D,D)",
      abs(g1[0] - 1) < 1e-9 and abs(g2[0] - 1) < 1e-2 and g2[0] <= 1 + 1e-2 and mask1[0, 0], (g1, g2))

# ------------------------------------------------------------- Volunteer's dilemma
print("=== Prop S1 (volunteer's dilemma)")


def volunteer(costs, b=F(10)):
    n = len(costs)
    u = {}
    for s in product(range(2), repeat=n):  # 0 = V, 1 = W
        anyv = any(x == 0 for x in s)
        u[s] = tuple((b - (costs[i] if s[i] == 0 else 0)) if anyv else F(0) for i in range(n))
    return Game([2] * n, u)


ok = True
for n in [2, 3, 4, 5]:
    for costs in ([F(3)] * n, [F(k + 1) for k in range(n)], [F(2), F(2)] + [F(5)] * (n - 2)):
        G = volunteer(costs)
        e = [tuple(0 if j == k else 1 for j in range(n)) for k in range(n)]
        R_ = reachable(G)
        nrp = {X for X in R_ if X and is_rest(G, X)}
        if nrp != {frozenset({ek}) for ek in e} or pte(G)[0] != frozenset():
            ok = False
            print("   (i) fails", n, costs)
        auts = automorphisms(G)
        each_has_partner = all(any(costs[j] == costs[i] for j in range(n) if j != i) for i in range(n))
        inv = [X for X in nrp if invariant(G, X, auts)]
        if each_has_partner and inv:
            ok = False
            print("   (ii) fails", n, costs)
        for pi in permutations(range(n)):
            pi = list(pi)
            if seq(G, pi) != {e[pi[0]]} or resolved(G, pi) != {e[pi[-1]]}:
                ok = False
                print("   (iii) fails", n, costs, pi)
check("Prop S1 (i)-(iii) for n=2..5 and three cost vectors (equal, distinct, two pairs)", ok)
# mixed Nash: nobody volunteers with prob (c/b)^(n/(n-1)) rising with n
vals = [(F(3) / 10) ** (F(n) / (n - 1)) for n in range(2, 8)]
check("symmetric mixed NE: probability nobody volunteers (c/b)^(n/(n-1)) increases with n (c/b=0.3)", all(float(vals[k]) < float(vals[k + 1]) for k in range(len(vals) - 1)))

# ------------------------------------------------------------- 3x3 example of S2
print("=== S2 3x3 example with strategies x,y,z")
A = [[2, 0, 8], [6, 1, 4], [3, 5, 7]]
G = symmetric(A)
E, rounds = pte(G)
gu = [[guarantee(G, R_, i) for i in range(2)] for R_ in rounds if R_]
check("PTE: S -> five profiles -> empty, guarantees (3,3) then (8,8)",
      E == frozenset() and len(rounds) == 3 and len(rounds[1]) == 5 and gu == [[3, 3], [8, 8]], gu)
check("sequential reaches {(z,z)} under both orders", seq(G, [0, 1]) == {(2, 2)} and seq(G, [1, 0]) == {(2, 2)})
# guarantees along the sequential process, order (1,2)
Ecur = frozenset(G.S)
g1s, g2s = [], []
while True:
    E0 = Ecur
    g1s.append(guarantee(G, Ecur, 0))
    Ecur = step(G, Ecur, [0])
    g2s.append(guarantee(G, Ecur, 1))
    Ecur = step(G, Ecur, [1])
    if Ecur == E0:
        break
check("guarantees under (1,2): player 1 3,4,7; player 2 4,5,7 (distinct values)", sorted(set(g1s)) == [3, 4, 7] and sorted(set(g2s)) == [4, 5, 7], (g1s, g2s))

# ------------------------------------------------------------- public goods
print("=== S2 public goods")


def pg(a, w=(1, 1, 1), eps=0):
    u = {}
    for s in product(range(2), repeat=3):
        x = [w[i] * s[i] for i in range(3)]
        tot = sum(x)
        u[s] = tuple(1 - x[i] + a * tot + eps * x[(i + 1) % 3] for i in range(3))
    return Game((2, 2, 2), u)


ok = True
for a in [F(51, 100), F(3, 5), F(3, 4), F(9, 10), F(99, 100)]:
    G = pg(a)
    E, rounds = pte(G)
    if E != frozenset() or len(rounds) != 4:
        ok = False
        print("   unit PG", a, [sorted(R_) for R_ in rounds])
check("unit public goods, 1/2<a<1: rounds S, (drop lone contributors), (drop nobody), empty", ok)
ok = True
for a in [F(51, 100), F(3, 4), F(99, 100)]:
    G = pg(a)
    for pi in permutations(range(3)):
        pi = list(pi)
        sp = seq(G, pi)
        rp = resolved(G, pi)
        fr_s = [i for i in range(3) if next(iter(sp))[i] == 0]
        fr_r = [i for i in range(3) if next(iter(rp))[i] == 0]
        if not (len(sp) == 1 and len(rp) == 1 and fr_s == [pi[1]] and fr_r == [pi[0]]):
            ok = False
            print("   PG orders", a, pi, sorted(sp), sorted(rp))
    full = frozenset({(1, 1, 1)})
    X = step(G, step(G, frozenset(G.S), [0]), [0])
    X = step(G, X, [1, 2])
    if X != full or not is_rest(G, full):
        ok = False
check("unit PG: seq leaves the 2nd reasoner free riding, res the 1st; full contribution reached by 1,1,{2,3} and a rest point", ok)
ok = True
for a in [F(51, 100), F(3, 4)]:
    for eps in [F(1, 1000), F(1, 100)]:
        G = pg(a, eps=eps)
        if not no_ties(G):
            ok = False
        if pte(G)[0] != frozenset():
            ok = False
        X = step(G, step(G, frozenset(G.S), [0]), [0])
        X = step(G, X, [1, 2])
        if X != frozenset({(1, 1, 1)}):
            ok = False
        for pi in permutations(range(3)):
            sp = seq(G, list(pi))
            if (1, 1, 1) in sp:
                ok = False
check("perturbed PG (eps x_{i+1}): no ties, limit empty, full contribution reachable, no rotation reaches it", ok)

# heterogeneous endowments
w = (F(1), F(9, 10), F(4, 5))
res = {}
for den in range(2, 200):
    for num in range(1, den):
        a = F(num, den)
        if a in res:
            continue
        E = pte(pg(a, w))[0]
        res[a] = "empty" if not E else ("full" if E == {(1, 1, 1)} else "other:" + str(sorted(E)))
keys = sorted(res)
empty_ok = all(res[a] == "empty" for a in keys if F(9, 17) < a < 1)
full_set = [a for a in keys if res[a] == "full"]
full_ok = all(F(10, 27) <= a < F(10, 19) for a in full_set) and all(res[a] == "full" for a in keys if F(10, 27) <= a < F(10, 19))
check("w=(1,9/10,4/5): limit empty for all rationals a (den<200) with 9/17<a<1", empty_ok)
check("w=(1,9/10,4/5): limit = full contribution exactly for 10/27<=a<10/19 (rationals, den<200)", full_ok,
      "min %s max %s" % (min(full_set), max(full_set)) if full_set else "none")
# what happens in [10/19, 9/17]
mid = sorted({res[a] for a in keys if F(10, 19) <= a <= F(9, 17)})
print("   outcomes for 10/19<=a<=9/17:", mid)
# ties: values of a in (9/17,1) where the game has ties: payoffs are linear in a; find coincidences
tie_as = set()
S3 = list(product(range(2), repeat=3))
for i in range(3):
    for s, t in combinations(S3, 2):
        xs = [w[k] * s[k] for k in range(3)]
        xt = [w[k] * t[k] for k in range(3)]
        # 1 - xs_i + a*sum xs = 1 - xt_i + a*sum xt
        dsum = sum(xs) - sum(xt)
        dconst = -xs[i] + xt[i]
        if dsum == 0:
            if dconst == 0:
                tie_as.add("always")
        else:
            a = -dconst / dsum
            if F(9, 17) < a < 1:
                tie_as.add(a)
check("w=(1,9/10,4/5): ties occur at exactly four values of a in (9/17,1)", len(tie_as) == 4 and "always" not in tie_as, sorted(tie_as, key=str))
check("expected payoff with drawn order 2a+1/3 vs 3a: loss a-1/3", True)

# ------------------------------------------------------------- uniform law: search a 3-player witness
print("=== S2 uniform law: search for a fully symmetric 3-player 3-strategy game with six distinct sequential predictions")
rng = random.Random(5)
found = None
for trial in range(20000):
    f = {}
    for own in range(3):
        for ms in [(a_, b_) for a_ in range(3) for b_ in range(a_, 3)]:
            f[(own, ms)] = rng.randint(0, 30)
    u = {}
    for s in product(range(3), repeat=3):
        u[s] = tuple(f[(s[i], tuple(sorted(s[j] for j in range(3) if j != i)))] for i in range(3))
    G = Game((3, 3, 3), u)
    preds = [seq(G, list(pi)) for pi in permutations(range(3))]
    if len(set(preds)) == 6:
        found = (trial, sorted(f.items()))
        break
check("a fully symmetric 3-player 3-strategy game with pairwise different sequential predictions over the 6 orders exists (random search)", found is not None, found[0] if found else "")

# ------------------------------------------------------------- mixed trilemma witnesses (grid evidence)
print("=== S5: Leader, battle of the sexes, G_1/2 (grid evidence)")
K = 200
grid = np.linspace(0, 1, K + 1)
Pm, Qm = np.meshgrid(grid, grid, indexing="ij")
for name, A, B, want in [("Leader", [[0, 3], [2, 1]], [[0, 2], [3, 1]], [(1, 0), (0, 1)]),
                         ("BoS", [[0, 2], [3, 1]], [[0, 3], [2, 1]], [(1, 0), (0, 1)])]:
    u1, u2 = mixed_payoffs(A, B)
    f1, f2 = u1.ravel(), u2.ravel()
    und = []
    for k in range(len(f1)):
        if not np.any((f1 > f1[k] + 1e-12) & (f2 > f2[k] + 1e-12)):
            und.append(k)
    pts_ = {(round(Pm.ravel()[k], 6), round(Qm.ravel()[k], 6)) for k in und}
    check("%s: grid profiles not improved upon for both = {(A,Y),(Y,A)}" % name, pts_ == {(1.0, 0.0), (0.0, 1.0)}, sorted(pts_)[:6])
A, B = [[0, 3], [1, 0.5]], [[0, 1], [3, 0.5]]
u1, u2 = mixed_payoffs(A, B)
pm = 3 / 7
v = (pm * pm * 0 + pm * (1 - pm) * 3 + (1 - pm) * pm * 1 + (1 - pm) ** 2 * 0.5)
check("G_1/2: each plays A w.p. 3/7 gives 8/7 to both, and no grid profile gives both more", abs(v - 8 / 7) < 1e-12 and not np.any((u1 > 8 / 7 + 1e-9) & (u2 > 8 / 7 + 1e-9)))

# ------------------------------------------------------------- magical thinking 0.32
print("=== S4 magical thinking, PD (5,3,1,0), types uniform on [0,1]")
# type th cooperates iff th*R + (1-th)*(q R + (1-q) S) >= th*P + (1-th)*(q T + (1-q) P), q = 1 - cutoff
T_, R_, P_, S_ = 5, 3, 1, 0
lo, hi = 0.0, 1.0
for _ in range(200):
    c = (lo + hi) / 2
    q = 1 - c
    diff = c * (R_ - P_) + (1 - c) * (q * (R_ - T_) + (1 - q) * (S_ - P_))
    if diff > 0:
        hi = c
    else:
        lo = c
c = (lo + hi) / 2
check("cutoff (5-sqrt17)/2 and mutual cooperation probability (1-cutoff)^2 rounds to 0.32", abs(c - (5 - math.sqrt(17)) / 2) < 1e-9 and round((1 - c) ** 2, 2) == 0.32, ((1 - c) ** 2))

# ------------------------------------------------------------- Nash-2 in matching pennies
print("=== Nash-2 (Sandomirskaia) in matching pennies")


def nash2(G):
    out = []
    for s in G.S:
        blocked = False
        for i in range(2):
            j = 1 - i
            for t in range(G.sizes[i]):
                p = list(s)
                p[i] = t
                p = tuple(p)
                if G.u[p][i] <= G.u[s][i]:
                    continue
                # profitable deviation; check whether every profitable reply of j leaves i >= u_i(s)
                secure = True
                for y in range(G.sizes[j]):
                    q = list(p)
                    q[j] = y
                    q = tuple(q)
                    if G.u[q][j] > G.u[p][j] and G.u[q][i] < G.u[s][i]:
                        secure = False
                if secure:
                    blocked = True
        if not blocked:
            out.append(s)
    return frozenset(out)


check("Nash-2 empty in matching pennies; contains (C,C) and (D,D) in the PD",
      nash2(bimatrix([[1, -1], [-1, 1]], [[-1, 1], [1, -1]])) == frozenset() and nash2(pd(5, 3, 1, 0)) == {(0, 0), (1, 1)})

# ------------------------------------------------------------- look-back rules (S1)
print("=== S1 reciprocity rule")


def reciprocity(G):
    out = []
    for s in G.S:
        okp = True
        for i in range(2):
            j = 1 - i
            for t in range(G.sizes[i]):
                if t == s[i]:
                    continue
                stay = list(s)
                stay[i] = t
                stay = tuple(stay)
                profs = []
                for y in range(G.sizes[j]):
                    p = [None, None]
                    p[i] = t
                    p[j] = y
                    profs.append(tuple(p))
                if G.u[stay][j] >= G.u[s][j]:
                    val = max(G.u[p][i] for p in profs)
                else:
                    val = min(G.u[p][i] for p in profs)
                if val > G.u[s][i]:
                    okp = False
        if okp:
            out.append(s)
    return frozenset(out)


ok = True
for (T, R, P, S) in [(5, 3, 1, 0), (4, 3, 2, 1), (10, 3, 2, 0), (100, 99, 1, 0)]:
    Aq = [[R, R - P + S], [S, S - 1]]
    Z = bimatrix(Aq, [[-a for a in row] for row in Aq])
    if reciprocity(pd(T, R, P, S)) != {(0, 0)} or reciprocity(Z) != frozenset():
        ok = False
check("reciprocity: {(C,C)} in the PD and nothing in Z_PD (four PDs)", ok)

print()
print("TOTAL FAILURES:", NFAIL)
