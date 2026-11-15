"""Checks of claims of the paper on local counterfactual rules, the traveler's dilemma, chicken and rules
that look back.

Part 1. Theorem 2 (local counterfactuals), strengthened form. In a 2x2 game, a local rule is described, for each
   answerer j, by alpha_j (switch probability when her previous strategy is her preferred one in an opposed
   continuation) and beta_j (when it is her less preferred one). For a prisoner's dilemma (T,R,P,S), the game
   Z_PD = [[R, R-P+S], [S, S-1]] (player 1's payoffs, zero-sum). Claims checked in exact arithmetic:
   (a) mutual defection is ruled out in the PD iff max(alpha_1, alpha_2) > (P-S)/(R-S);
   (b) Z_PD is empty iff alpha_1 > (P-S)/(R-S); Z_PD with the players exchanged is empty iff alpha_2 > (P-S)/(R-S).
   Hence ruling out mutual defection empties Z_PD or its exchanged copy, whatever the betas.
Part 2. The perfectly transparent equilibrium in the traveler's dilemma (integer claims 2..M, reward r > 0):
   the thresholds follow c_1 = 2, c_{k+1} = min(M, c_k + ceil(r)), and the limit is {(M,M)}.
Part 3. Chicken: the credible alpha-core (outsiders best reply to a deviating coalition) is empty; the last
   nonempty iterate of the PTE fails its own test (T(E) is empty).
Part 4. Rules that look back: reciprocity in the PD and Z_PD; the mutual-gain rule with the defaults worst answer,
   best reply and staying put, in the PD and the TD.
"""
import itertools
import math
import random
from fractions import Fraction as F

random.seed(20261030)


# ---------- Part 1 ----------
def value_dev(u_dev, u_ans, prev, alpha, beta):
    """Expected payoff of the deviator in a two-strategy continuation. u_dev[y], u_ans[y] for y in {0,1};
    prev = answerer's previous strategy. Returns None if the continuation is not opposed-strict (not used here)."""
    other = 1 - prev
    if u_ans[prev] == u_ans[other] or u_dev[prev] == u_dev[other]:
        return None
    ans_pref = prev if u_ans[prev] > u_ans[other] else other
    opposed = (u_dev[ans_pref] < u_dev[1 - ans_pref])
    if not opposed:
        return None
    p = alpha if ans_pref == prev else beta
    return (1 - p) * u_dev[prev] + p * u_dev[other]


def predicted(U1, U2, a, b):
    """U1[x][y], U2[x][y]; a = (alpha_1, beta_1) for answerer 1, b = (alpha_2, beta_2) for answerer 2.
    Returns the set of predicted pure profiles, or raises if some continuation is not opposed-strict."""
    out = []
    for x, y in itertools.product(range(2), range(2)):
        ok = True
        # player 1 deviates to 1-x, player 2 answers (previous y)
        t = 1 - x
        v = value_dev([U1[t][0], U1[t][1]], [U2[t][0], U2[t][1]], y, *b)
        assert v is not None
        if v > U1[x][y]:
            ok = False
        # player 2 deviates to 1-y, player 1 answers (previous x)
        t = 1 - y
        v = value_dev([U2[0][t], U2[1][t]], [U1[0][t], U1[1][t]], x, *a)
        assert v is not None
        if v > U2[x][y]:
            ok = False
        if ok:
            out.append((x, y))
    return out


def pd(T, R, P, S):
    # strategy 0 = C, 1 = D
    U1 = [[R, S], [T, P]]
    U2 = [[R, T], [S, P]]
    return U1, U2


def zpd(T, R, P, S):
    U1 = [[R, R - P + S], [S, S - 1]]
    U2 = [[-v for v in row] for row in U1]
    return U1, U2


def swap(U1, U2):
    return [list(r) for r in zip(*U2)], [list(r) for r in zip(*U1)]


print('Part 1. Local counterfactuals: mutual defection ruled out in the PD versus Z_PD empty (exact arithmetic)')
grid = [F(k, 20) for k in range(21)]
n_games = n_cases = 0
viol = 0
for _ in range(300):
    S = F(random.randint(-20, 20), random.randint(1, 5))
    P = S + F(random.randint(1, 30), random.randint(1, 5))
    R = P + F(random.randint(1, 30), random.randint(1, 5))
    T = R + F(random.randint(1, 30), random.randint(1, 5))
    thr = (P - S) / (R - S)
    G = pd(T, R, P, S)
    Z = zpd(T, R, P, S)
    Zs = swap(*Z)
    # dominance in Z_PD: row 0 for player 1, column 1 for player 2
    assert all(Z[0][0][y] > Z[0][1][y] for y in range(2)) and all(Z[1][x][1] > Z[1][x][0] for x in range(2))
    n_games += 1
    for _ in range(40):
        a = (random.choice(grid + [thr]), random.choice(grid))
        b = (random.choice(grid + [thr]), random.choice(grid))
        n_cases += 1
        dd_out = (1, 1) not in predicted(*G, a, b)
        z_empty = predicted(*Z, a, b) == []
        zs_empty = predicted(*Zs, a, b) == []
        if dd_out != (max(a[0], b[0]) > thr):
            viol += 1
        if z_empty != (a[0] > thr) or zs_empty != (b[0] > thr):
            viol += 1
        if dd_out and not (z_empty or zs_empty):
            viol += 1
print(f'   {n_games} random PDs (rational payoffs), {n_cases} rule pairs (alpha, beta on a 1/20 grid plus the threshold):'
      f' violations of (a), (b) and of "DD ruled out => Z_PD or its exchanged copy empty": {viol}')
for T, R, P, S in [(5, 3, 1, 0), (10, 3, 1, 0), (4, 3, 2, 1)]:
    Z = zpd(T, R, P, S)
    print(f'   PD {(T, R, P, S)}: threshold (P-S)/(R-S) = {F(P - S, R - S)}; Z_PD player 1 payoffs {Z[0]};'
          f' anonymous rule alpha = beta = 1: PD prediction {predicted(*pd(T, R, P, S), (1, 1), (1, 1))},'
          f' Z_PD prediction {predicted(*Z, (1, 1), (1, 1))}')
# a counterexample without anonymity: answerer 1 (0,1), answerer 2 (1,1)
G = pd(5, 3, 1, 0)
Z = zpd(5, 3, 1, 0)
print(f'   non-anonymous rule (alpha_1,beta_1)=(0,1), (alpha_2,beta_2)=(1,1): PD {predicted(*G, (0, 1), (1, 1))},'
      f' Z_PD {predicted(*Z, (0, 1), (1, 1))}, exchanged Z_PD {predicted(*swap(*Z), (0, 1), (1, 1))}'
      '  [profiles as (row, column), 0 = C or a]')


# ---------- Part 2 ----------
def td(M, r):
    claims = list(range(2, M + 1))

    def pay(x, y):
        if x == y:
            return (x, x)
        if x < y:
            return (x + r, x - r)
        return (y - r, y + r)
    return claims, pay


def pte_thresholds(M, r):
    claims, pay = td(M, r)
    E = [(x, y) for x in claims for y in claims]
    ths = []
    while True:
        m = []
        for i in range(2):
            best = None
            for t in claims:
                sec = [s for s in E if s[i] == t]
                if sec:
                    w = min(pay(*s)[i] for s in sec)
                    best = w if best is None else max(best, w)
            m.append(best)
        ths.append(tuple(m))
        E2 = [s for s in E if all(pay(*s)[i] >= m[i] for i in range(2))]
        if E2 == E or not E2:
            return ths, E2
        E = E2


print('\nPart 2. PTE in the traveler\'s dilemma: thresholds against c1 = 2, c_{k+1} = min(M, c_k + ceil(r))')
bad = 0
tot = 0
for M in range(3, 31):
    for r in [F(1, 4), F(1, 2), F(1), F(3, 2), F(2), F(5, 2), F(3), F(5), F(7), F(10), F(25)]:
        ths, lim = pte_thresholds(M, r)
        pred = [2]
        while pred[-1] < M:
            pred.append(min(M, pred[-1] + math.ceil(r)))
        got = [t[0] for t in ths]
        # the last threshold is repeated when the set stabilizes
        if got[:len(pred)] != pred or lim != [(M, M)] or any(t[0] != t[1] for t in ths):
            bad += 1
        tot += 1
print(f'   {tot} (M, r) pairs, M = 3..30, r in {{1/4,...,25}}: mismatches with the recursion or limit != {{(M,M)}}: {bad}')
ths, lim = pte_thresholds(12, F(2))
print(f'   example M = 12, r = 2: thresholds {[int(t[0]) for t in ths]}, limit {lim}')

# ---------- Part 3 ----------
print('\nPart 3. Chicken (0 = aggressive, 1 = yield): payoffs (0,0), (3,1), (1,3), (2,2)')
C1 = [[0, 3], [1, 2]]
C2 = [[0, 1], [3, 2]]
core = []
for x, y in itertools.product(range(2), range(2)):
    blocked = False
    # player 1 alone deviates, player 2 best replies (worst best reply for player 1, credible version)
    for t in range(2):
        if t != x:
            br = [z for z in range(2) if C2[t][z] == max(C2[t])]
            if min(C1[t][z] for z in br) > C1[x][y]:
                blocked = True
    for t in range(2):
        if t != y:
            br = [z for z in range(2) if C1[z][t] == max(C1[w][t] for w in range(2))]
            if min(C2[z][t] for z in br) > C2[x][y]:
                blocked = True
    # grand coalition: a profile giving both strictly more
    if any(C1[a][b] > C1[x][y] and C2[a][b] > C2[x][y] for a in range(2) for b in range(2)):
        blocked = True
    if not blocked:
        core.append((x, y))
print(f'   credible alpha-core (outsiders best reply): {core}')
E = [(x, y) for x in range(2) for y in range(2)]
U = (C1, C2)
hist = []
while True:
    m = []
    for i in range(2):
        m.append(max(min(U[i][s[0]][s[1]] for s in E if s[i] == t) for t in range(2) if any(s[i] == t for s in E)))
    E2 = [s for s in E if all(U[i][s[0]][s[1]] >= m[i] for i in range(2))]
    hist.append((list(E), tuple(m)))
    if not E2 or E2 == E:
        break
    E = E2
for Ek, mk in hist:
    print(f'   set {Ek}: thresholds {mk}')
print(f'   last nonempty iterate {E}; applying the test to it leaves {E2}')


# ---------- Part 4 ----------
def lookback(U, strategies, rule, default=None):
    """Two-player pure profiles; U[i][(x, y)]. rule 'recip': the answerer gives the deviator's best answer if the
    deviation (answerer staying put) did not lower the answerer's payoff, else the worst. rule 'mutual': the answerer
    picks, among answers that leave both strictly better than at the previous profile, the best for the deviator;
    if none, the default answer ('worst' for the deviator, 'best' = answerer's best reply (worst for the deviator
    among them), 'stay')."""
    S1, S2 = strategies
    out = []
    for s in itertools.product(S1, S2):
        ok = True
        for i in range(2):
            j = 1 - i
            Sj = S2 if j == 1 else S1
            for t in (S1 if i == 0 else S2):
                if t == s[i]:
                    continue

                def prof(y):
                    return (t, y) if i == 0 else (y, t)
                stay = prof(s[j])
                if rule == 'recip':
                    if U[j][stay] >= U[j][s]:
                        v = max(U[i][prof(y)] for y in Sj)
                    else:
                        v = min(U[i][prof(y)] for y in Sj)
                else:
                    good = [y for y in Sj if U[i][prof(y)] > U[i][s] and U[j][prof(y)] > U[j][s]]
                    if good:
                        v = max(U[i][prof(y)] for y in good)
                    elif default == 'worst':
                        v = min(U[i][prof(y)] for y in Sj)
                    elif default == 'best':
                        b = max(U[j][prof(y)] for y in Sj)
                        v = min(U[i][prof(y)] for y in Sj if U[j][prof(y)] == b)
                    else:
                        v = U[i][stay]
                if v > U[i][s]:
                    ok = False
        if ok:
            out.append(s)
    return out


def as_dict(U1, U2):
    return [{(x, y): U1[x][y] for x in range(2) for y in range(2)}, {(x, y): U2[x][y] for x in range(2) for y in range(2)}]


print('\nPart 4. Rules that look back at the previous profile (0 = C or a)')
for T, R, P, S in [(5, 3, 1, 0), (10, 3, 1, 0)]:
    Gd = as_dict(*pd(T, R, P, S))
    Zd = as_dict(*zpd(T, R, P, S))
    print(f'   PD {(T, R, P, S)}: reciprocity {lookback(Gd, ([0, 1], [0, 1]), "recip")}; Z_PD reciprocity '
          f'{lookback(Zd, ([0, 1], [0, 1]), "recip")}')
    for d in ['worst', 'best', 'stay']:
        print(f'      mutual gain, default {d}: PD {lookback(Gd, ([0, 1], [0, 1]), "mutual", d)}')
for M, r in [(6, 2), (12, 2), (12, 5)]:
    claims, pay = td(M, r)
    Ud = [{(x, y): pay(x, y)[i] for x in claims for y in claims} for i in range(2)]
    for d in ['worst', 'best', 'stay']:
        print(f'   TD M = {M}, r = {r}, mutual gain, default {d}: {lookback(Ud, (claims, claims), "mutual", d)}')
