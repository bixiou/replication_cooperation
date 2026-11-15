# Independent recomputation, for the paper, of
# (A) the formal virtual bargaining equilibrium (VBE) of Melkonyan, Zeitoun
#     and Chater (2022, Appendix, eqs. (2)-(5)) in pure strategies, two players:
#       w_i(s) = min{ u_i(s), sup_{y in BR_j(s_i)} u_i(s_i, y) }
#     (the "sup" is their tie-break among the opponent's best responses; the
#     variant "inf" is also run), feasible agreements = pure Nash equilibria
#     of the game with payoffs w, status quo w_i^m = min over feasible
#     agreements of w_i, VBE = argmax over feasible agreements of the
#     product of (w_i - w_i^m);
# (B) mixed-strategy claims: Proposition S7 (Leader, battle of the sexes),
#     the G_{1/2} remark, Proposition 3(iii), the games W and G* of Section
#     S8, the prisoner's dilemma (10,3,1,0) with a third pure strategy, and
#     the rising-demands path in Leader (maximin of the two payoffs over
#     mixed profiles), all on fine grids or in exact arithmetic.
import itertools
from fractions import Fraction as Fr
import numpy as np

def out(*a):
    print(*a, flush=True)

# ------------------------------------------------------------------ (A) VBE
def vbe(u1, u2, n1, n2, tie="sup"):
    U = [u1, u2]
    def w(i, s):
        a, b = s
        if i == 0:
            vals = [u2[(a, y)] for y in range(n2)]; m = max(vals)
            br = [y for y in range(n2) if u2[(a, y)] == m]
            alt = [u1[(a, y)] for y in br]
        else:
            vals = [u1[(x, b)] for x in range(n1)]; m = max(vals)
            br = [x for x in range(n1) if u1[(x, b)] == m]
            alt = [u2[(x, b)] for x in br]
        best = max(alt) if tie == "sup" else min(alt)
        return min(U[i][s], best)
    S = [(a, b) for a in range(n1) for b in range(n2)]
    W = {s: (w(0, s), w(1, s)) for s in S}
    feas = [s for s in S if all(W[s][0] >= W[(x, s[1])][0] for x in range(n1)) and all(W[s][1] >= W[(s[0], y)][1] for y in range(n2))]
    if not feas:
        return W, feas, None, []
    sq = (min(W[s][0] for s in feas), min(W[s][1] for s in feas))
    prod = {s: (W[s][0] - sq[0]) * (W[s][1] - sq[1]) for s in feas}
    mx = max(prod.values())
    return W, feas, sq, [s for s in feas if prod[s] == mx]

def two(rows):
    n1, n2 = len(rows), len(rows[0])
    u1 = {(a, b): rows[a][b][0] for a in range(n1) for b in range(n2)}
    u2 = {(a, b): rows[a][b][1] for a in range(n1) for b in range(n2)}
    return u1, u2, n1, n2

def guarantee(u, E, i, n):
    best = None
    for t in range(n):
        vals = [u[s] for s in E if s[i] == t]
        if vals:
            m = min(vals); best = m if best is None or m > best else best
    return best

def self_consistent(u1, u2, n1, n2, E):
    m1 = guarantee(u1, E, 0, n1); m2 = guarantee(u2, E, 1, n2)
    return all(u1[s] >= m1 and u2[s] >= m2 for s in E), (m1, m2)

out("== (A) Formal VBE, pure strategies ==")
games = {
    "PD (5,3,1,0)": [[(3, 3), (0, 5)], [(5, 0), (1, 1)]],
    "G_1/10": [[(0, 0), (3, 1)], [(1, 3), (Fr(1, 10), Fr(1, 10))]],
    "G_1/2": [[(0, 0), (3, 1)], [(1, 3), (Fr(1, 2), Fr(1, 2))]],
    "G_9/10": [[(0, 0), (3, 1)], [(1, 3), (Fr(9, 10), Fr(9, 10))]],
    "chicken": [[(0, 0), (3, 1)], [(1, 3), (2, 2)]],
    "Aumann chicken": [[(0, 0), (7, 2)], [(2, 7), (6, 6)]],
    "Leader": [[(0, 0), (3, 2)], [(2, 3), (1, 1)]],
    "BoS": [[(0, 0), (2, 3)], [(3, 2), (1, 1)]],
}
for name, rows in games.items():
    u1, u2, n1, n2 = two(rows)
    for tie in ["sup", "inf"]:
        W, feas, sq, sol = vbe(u1, u2, n1, n2, tie)
        sc = self_consistent(u1, u2, n1, n2, sol) if sol else None
        out("%-15s tie=%s worst payoffs %s feasible %s status quo %s VBE %s self-consistent %s" % (
            name, tie, {s: tuple(str(x) for x in W[s]) for s in W}, feas, sq and tuple(str(x) for x in sq), sol, sc))

def TD(M, r):
    cl = list(range(2, M + 1)); n = len(cl)
    u1 = {}; u2 = {}
    for a in range(n):
        for b in range(n):
            x, y = cl[a], cl[b]
            if x == y:
                u1[(a, b)] = Fr(x); u2[(a, b)] = Fr(x)
            elif x < y:
                u1[(a, b)] = x + r; u2[(a, b)] = x - r
            else:
                u1[(a, b)] = y - r; u2[(a, b)] = y + r
    return u1, u2, n, n, cl

out("Traveler's dilemma, claims 2..M: VBE (sup and inf tie-breaks); code T = top claims only, L = lowest only, B = both")
summary = {}
mism = 0
for r in [Fr(5, 4), Fr(3, 2), Fr(2), Fr(5, 2), Fr(3), Fr(7, 2), Fr(5), Fr(6)]:
    line = []
    for M in range(3, 21):
        u1, u2, n1, n2, cl = TD(M, r)
        codes = []
        for tie in ["sup", "inf"]:
            W, feas, sq, sol = vbe(u1, u2, n1, n2, tie)
            sc = sorted((cl[a], cl[b]) for a, b in sol)
            code = "T" if sc == [(M, M)] else "L" if sc == [(2, 2)] else "B" if sc == [(2, 2), (M, M)] else str(sc)
            codes.append(code)
        assert codes[0] == codes[1]
        pred = "T" if M > r + 3 else "L" if M < r + 3 else "B"
        if codes[0] != pred:
            mism += 1
        line.append("%d:%s" % (M, codes[0]))
    out("  r=%s  %s" % (r, " ".join(line)))
out("  mismatches with the rule (T if M>r+3, L if M<r+3, B if M=r+3):", mism)

# ------------------------------------------------------------------ (B) mixed
out("== (B) Mixed profiles ==")
def pay2(rows):
    # mixed payoffs of a 2x2 game; p = prob of row 0, q = prob of column 0
    (a, b), (c, d) = rows[0][0], rows[0][1]
    (e, f), (g, h) = rows[1][0], rows[1][1]
    def u(p, q):
        return (p * q * a + p * (1 - q) * c + (1 - p) * q * e + (1 - p) * (1 - q) * g,
                p * q * b + p * (1 - q) * d + (1 - p) * q * f + (1 - p) * (1 - q) * h)
    return u

N = 1001
P, Q = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N), indexing="ij")
for name in ["Leader", "BoS"]:
    u = pay2(games[name])
    U1, U2 = u(P, Q)
    # profiles strictly improved upon for both by (1,0) or (0,1)
    a1, a2 = u(1.0, 0.0); b1, b2 = u(0.0, 1.0)
    dom = ((U1 < a1) & (U2 < a2)) | ((U1 < b1) & (U2 < b2))
    notdom = np.argwhere(~dom)
    out(name, "grid 1001^2: profiles not strictly improved upon for both by (A,Y) or (Y,A):", [(P[i, j], Q[i, j]) for i, j in notdom][:6], "count", len(notdom))
    out("   max u1", U1.max(), "attained at", [(P[i, j], Q[i, j]) for i, j in np.argwhere(U1 == U1.max())][:3], "; max u2", U2.max(), "at", [(P[i, j], Q[i, j]) for i, j in np.argwhere(U2 == U2.max())][:3])
    mm = np.minimum(U1, U2)
    out("   max over grid of min(u1,u2):", mm.max(), "at", [(P[i, j], Q[i, j]) for i, j in np.argwhere(mm >= mm.max() - 1e-12)][:4])

u = pay2(games["G_1/2"])
U1, U2 = u(P, Q); mm = np.minimum(U1, U2)
out("G_1/2: u at (3/7,3/7) exact:", tuple(str(x) for x in u(Fr(3, 7), Fr(3, 7))), "; grid max of min(u1,u2):", mm.max(), "8/7 =", 8 / 7,
    "; grid points giving both more than 8/7:", int(((U1 > 8 / 7 + 1e-12) & (U2 > 8 / 7 + 1e-12)).sum()))
# refine: local search around candidates for max min
best = (0, None)
for i, j in np.argwhere(mm >= mm.max() - 1e-3):
    p0, q0 = P[i, j], Q[i, j]
    for p in np.linspace(max(0, p0 - 0.002), min(1, p0 + 0.002), 81):
        for q in np.linspace(max(0, q0 - 0.002), min(1, q0 + 0.002), 81):
            v = min(u(p, q))
            if v > best[0]:
                best = (v, (p, q))
out("   refined max of min(u1,u2):", best)

# Proposition 3(iii): PD (5,3,1,0), p,q = prob of cooperating
def upd(p, q):
    return (1 - p + 4 * q - p * q, 1 - q + 4 * p - p * q)
chk = []
for p in [Fr(k, 20) for k in range(0, 21)]:
    qs = [Fr(k, 2000) for k in range(0, 2001)]
    sec = [q for q in qs if upd(p, q)[0] >= 1 and upd(p, q)[1] >= 1]
    lo = min(sec); hi = max(sec)
    chk.append((str(p), str(lo), str(min(upd(p, q)[0] for q in sec))))
out("Prop 3(iii): for p on a grid, (p, smallest q in section on grid 1/2000, min u1 on section):", chk[:6], "...", chk[-2:])
out("   exact: u1(p, p/(4-p)) =", [str(upd(Fr(k, 10), Fr(k, 10) / (4 - Fr(k, 10)))[0]) for k in range(0, 11)])

# W and G* (Section S8): mixed security levels by LP over supports (2 or 3 strategies)
def mixed_security(A):
    # A[x][y]: payoff of the player choosing x against y; max_sigma min_y
    best = None
    k = len(A)
    import itertools as it
    # vertices: equalize payoffs against subsets of columns; brute force exact via Fractions
    cols = range(len(A[0]))
    cands = []
    for supp in it.chain.from_iterable(it.combinations(range(k), r) for r in range(1, k + 1)):
        for ys in it.combinations(cols, len(supp)):
            # solve sum_x s_x A[x][y] = v for y in ys, sum s = 1
            n = len(supp)
            M = [[Fr(A[x][y]) for x in supp] + [Fr(-1)] for y in ys] + [[Fr(1)] * n + [Fr(0)]]
            rhs = [Fr(0)] * n + [Fr(1)]
            # gaussian elimination
            m = [row[:] + [rhs[i]] for i, row in enumerate(M)]
            sz = n + 1; ok = True
            for c in range(sz):
                piv = next((r for r in range(c, sz) if m[r][c] != 0), None)
                if piv is None:
                    ok = False; break
                m[c], m[piv] = m[piv], m[c]
                for r in range(sz):
                    if r != c and m[r][c] != 0:
                        f = m[r][c] / m[c][c]
                        m[r] = [m[r][j] - f * m[c][j] for j in range(sz + 1)]
            if not ok:
                continue
            sol = [m[r][sz] / m[r][r] for r in range(sz)]
            s = sol[:n]
            if any(x < 0 for x in s):
                continue
            mix = [Fr(0)] * k
            for idx, x in zip(supp, s):
                mix[idx] = x
            val = min(sum(mix[x] * A[x][y] for x in range(k)) for y in cols)
            cands.append((val, mix))
    return max(cands, key=lambda t: t[0])

W1 = [[4, 1, 9], [2, 3, 5]]
W2t = [[4, 8], [6, 3], [5, 1]]   # player 2's payoffs, rows = her strategies b,b',b'', cols = a,a'
out("W: mixed security levels", str(mixed_security(W1)[0]), str(mixed_security(W2t)[0]))
Gs = [[-1, 10, -5], [-3, -2, 11], [12, -4, 0]]
out("G*: mixed security level", str(mixed_security(Gs)[0]), "with", [str(x) for x in mixed_security(Gs)[1]], "; uniform mixture guarantees", str(min(Fr(sum(Gs[x][y] for x in range(3)), 3) for y in range(3))))

# PD (10,3,1,0) with a third pure strategy m: cooperate with prob 9/10; PTE rounds in pure strategies
def mix_pay(p, q):
    # p, q: prob of cooperating; PD (T,R,P,S) = (10,3,1,0)
    T_, R_, P_, S_ = 10, 3, 1, 0
    return (p * q * R_ + p * (1 - q) * S_ + (1 - p) * q * T_ + (1 - p) * (1 - q) * P_,
            p * q * R_ + q * (1 - p) * S_ + (1 - q) * p * T_ + (1 - p) * (1 - q) * P_)
st = [Fr(1), Fr(0), Fr(9, 10)]   # C, D, m
U = {(a, b): mix_pay(st[a], st[b]) for a in range(3) for b in range(3)}
E = set(U); hist = []
while True:
    g = (guarantee({s: U[s][0] for s in U}, E, 0, 3), guarantee({s: U[s][1] for s in U}, E, 1, 3))
    F = {s for s in E if U[s][0] >= g[0] and U[s][1] >= g[1]}
    hist.append((tuple(str(x) for x in g), sorted(F)))
    if F == E or not F:
        break
    E = F
ties = len(set(U[s][0] for s in U)) < 9 or len(set(U[s][1] for s in U)) < 9
out("PD (10,3,1,0) with m: ties", ties, "; PTE rounds (guarantees, set):", hist, "; (m,m) pays", str(U[(2, 2)][0]))

# Rising demands in Leader: the demands rise together while (A,Y) and (Y,A) meet them; stall at the
# largest common level, which the grid above gives as max min(u1,u2) = 2, attained at (1,0) and (0,1) only.
u = pay2(games["Leader"])
v = max(min(u(p, 0.0)[0], u(p, 1.0)[0]) for p in np.linspace(0, 1, 100001))
out("Leader: mixed security level (grid 1e-5):", v, "; union {(A,Y),(Y,A)} guarantees: player 1 commits to A -> u1(A,Y) =", u(1, 0)[0], ", (Y,A) gives her", u(0, 1)[0])
