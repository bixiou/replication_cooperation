"""Shared functions of the independent checks of the typology of the paper (Table 1).

Logic summary.
A two-player finite game is a pair of payoff matrices A (player 1, rows) and B
(player 2, columns), lists of lists of ints or Fractions. A profile is a pair
(r, c). Every concept of Table 1 that is defined on pure profiles is coded
directly from its definition in the paper (Section 2 for U(G), IR(G),
Pareto optimality, copying pays; Section 5 for the verdicts, the perfectly
transparent equilibrium rounds and the truncation; Section 6 for resolved and
sequential perfect prediction; Section 7 for the strong equilibrium, the pure
alpha-core, the credible alpha-core, Berge, the bargaining selections; S7 for
Nash-2, the optimin, the measure of mutual advantage, team reasoning,
superrationality, the perfect cooperative equilibrium in pure strategies;
S6 for Chwe's largest consistent set and farsighted stable sets).
The test scripts import these functions and compare their outputs with each
table entry. The functions are implemented separately from the paper's own programs.
"""
from fractions import Fraction
from itertools import permutations, product, combinations

INF = float("inf")


# ---------------------------------------------------------------- basics
def dims(A):
    return len(A), len(A[0])


def profiles(A):
    m, n = dims(A)
    return [(r, c) for r in range(m) for c in range(n)]


def pay(A, B, s):
    return (A[s[0]][s[1]], B[s[0]][s[1]])


def u(A, B, i, s):
    return A[s[0]][s[1]] if i == 0 else B[s[0]][s[1]]


def security(A, B):
    m, n = dims(A)
    v1 = max(min(A[r][c] for c in range(n)) for r in range(m))
    v2 = max(min(B[r][c] for r in range(m)) for c in range(n))
    return v1, v2


def has_no_ties(A, B):
    m, n = dims(A)
    a = [A[r][c] for r in range(m) for c in range(n)]
    b = [B[r][c] for r in range(m) for c in range(n)]
    return len(set(a)) == len(a) and len(set(b)) == len(b)


def U_set(A, B):
    m, n = dims(A)
    best_row = [max(A[r][c] for c in range(n)) for r in range(m)]
    best_col = [max(B[r][c] for r in range(m)) for c in range(n)]
    out = set()
    for (r, c) in profiles(A):
        ok = all(best_row[r2] <= A[r][c] for r2 in range(m) if r2 != r)
        ok = ok and all(best_col[c2] <= B[r][c] for c2 in range(n) if c2 != c)
        if ok:
            out.add((r, c))
    return out


def IR_set(A, B):
    v1, v2 = security(A, B)
    return {s for s in profiles(A) if A[s[0]][s[1]] >= v1 and B[s[0]][s[1]] >= v2}


def pareto_optimal(A, B, s):
    a, b = pay(A, B, s)
    for t in profiles(A):
        x, y = pay(A, B, t)
        if x >= a and y >= b and (x > a or y > b):
            return False
    return True


def strictly_improved(A, B, s):
    a, b = pay(A, B, s)
    return any(A[t[0]][t[1]] > a and B[t[0]][t[1]] > b for t in profiles(A))


# ------------------------------------------------------- symmetric games
def is_symmetric(A, B):
    m, n = dims(A)
    return m == n and all(B[x][y] == A[y][x] for x in range(m) for y in range(n))


def best_symmetric(A):
    m = len(A)
    vals = [A[x][x] for x in range(m)]
    best = max(vals)
    zs = [x for x in range(m) if vals[x] == best]
    return zs


def copying_pays(A, B):
    m = len(A)
    for x in range(m):
        for y in range(m):
            if x == y:
                continue
            a, b = A[x][y], B[x][y]
            if a < b and not (A[y][y] > a):
                return False
            if b < a and not (B[x][x] > b):
                return False
    return True


# ------------------------------------------------------- Nash and Berge
def br1(A, c):
    m, _ = dims(A)
    best = max(A[r][c] for r in range(m))
    return [r for r in range(m) if A[r][c] == best]


def br2(B, r):
    _, n = dims(B)
    best = max(B[r][c] for c in range(n))
    return [c for c in range(n) if B[r][c] == best]


def nash_pure(A, B):
    return {(r, c) for (r, c) in profiles(A) if r in br1(A, c) and c in br2(B, r)}


def berge(A, B):
    # player 1's payoff maximised by player 2's choice given row, and conversely
    m, n = dims(A)
    out = set()
    for (r, c) in profiles(A):
        if A[r][c] == max(A[r][c2] for c2 in range(n)) and B[r][c] == max(B[r2][c] for r2 in range(m)):
            out.add((r, c))
    return out


# --------------------------------------------- coalitional concepts (2 players)
def strong_eq(A, B):
    m, n = dims(A)
    out = set()
    for (r, c) in profiles(A):
        a, b = A[r][c], B[r][c]
        if any(A[r2][c] > a for r2 in range(m)):
            continue
        if any(B[r][c2] > b for c2 in range(n)):
            continue
        if strictly_improved(A, B, (r, c)):
            continue
        out.add((r, c))
    return out


def alpha_core(A, B):
    m, n = dims(A)
    out = set()
    for (r, c) in profiles(A):
        a, b = A[r][c], B[r][c]
        # coalition {1}: a row that beats a against every answer
        if any(min(A[r2][cc] for cc in range(n)) > a for r2 in range(m) if r2 != r):
            continue
        if any(min(B[rr][c2] for rr in range(m)) > b for c2 in range(n) if c2 != c):
            continue
        if strictly_improved(A, B, (r, c)):
            continue
        out.add((r, c))
    return out


def credible_alpha_core(A, B):
    """Two players; a singleton deviation is valued at the worst of the other's best replies."""
    m, n = dims(A)
    out = set()
    for (r, c) in profiles(A):
        a, b = A[r][c], B[r][c]
        if any(min(A[r2][y] for y in br2(B, r2)) > a for r2 in range(m) if r2 != r):
            continue
        if any(min(B[x][c2] for x in br1(A, c2)) > b for c2 in range(n) if c2 != c):
            continue
        if strictly_improved(A, B, (r, c)):
            continue
        out.add((r, c))
    return out


# ------------------------------------------------------- Nash-2 (two players)
def nash2(A, B, weak=True):
    """Sandomirskaia: a profitable deviation is secure if every profitable reply of the
    other leaves the deviator at least (weak=True) her initial payoff. A profile is a
    Nash-2 equilibrium if no player has a secure profitable deviation."""
    m, n = dims(A)
    out = set()
    for (r, c) in profiles(A):
        a, b = A[r][c], B[r][c]
        ruled = False
        for r2 in range(m):
            if r2 == r or A[r2][c] <= a:
                continue
            replies = [c2 for c2 in range(n) if c2 != c and B[r2][c2] > B[r2][c]]
            if all((A[r2][c2] >= a) if weak else (A[r2][c2] > a) for c2 in replies):
                ruled = True
                break
        if not ruled:
            for c2 in range(n):
                if c2 == c or B[r][c2] <= b:
                    continue
                replies = [r2 for r2 in range(m) if r2 != r and A[r2][c2] > A[r][c2]]
                if all((B[r2][c2] >= b) if weak else (B[r2][c2] > b) for r2 in replies):
                    ruled = True
                    break
        if not ruled:
            out.add((r, c))
    return out


# ------------------------------------------------------- optimin (pure)
def optimin(A, B):
    """pi_i(s): the minimum of i's payoff at s and at the profiles reached by one
    profitable unilateral deviation of the other; optimin = profiles whose pi is
    not Pareto-dominated by the pi of another profile."""
    m, n = dims(A)
    pi = {}
    for (r, c) in profiles(A):
        p1 = min([A[r][c]] + [A[r][c2] for c2 in range(n) if B[r][c2] > B[r][c]])
        p2 = min([B[r][c]] + [B[r2][c] for r2 in range(m) if A[r2][c] > A[r][c]])
        pi[(r, c)] = (p1, p2)
    out = set()
    for s in pi:
        dom = any(pi[t][0] >= pi[s][0] and pi[t][1] >= pi[s][1] and pi[t] != pi[s] for t in pi)
        if not dom:
            out.add(s)
    return out


# ------------------------------------------- comparisons of profiles
def mg(A, B):
    v1, v2 = security(A, B)
    ir = IR_set(A, B)
    best = max(min(A[r][c] - v1, B[r][c] - v2) for (r, c) in ir)
    return {(r, c) for (r, c) in ir if min(A[r][c] - v1, B[r][c] - v2) == best}


def nb(A, B):
    v1, v2 = security(A, B)
    ir = IR_set(A, B)
    best = max((A[r][c] - v1) * (B[r][c] - v2) for (r, c) in ir)
    return {(r, c) for (r, c) in ir if (A[r][c] - v1) * (B[r][c] - v2) == best}


def ma_security(A, B):
    v1, v2 = security(A, B)
    M1 = max(max(row) for row in A)
    M2 = max(max(row) for row in B)
    if M1 == v1 or M2 == v2:
        return set()
    ir = IR_set(A, B)

    def f(s):
        return min(Fraction(A[s[0]][s[1]] - v1) / (M1 - v1), Fraction(B[s[0]][s[1]] - v2) / (M2 - v2))
    best = max(f(s) for s in ir)
    return {s for s in ir if f(s) == best}


def team(A, B):
    best = max(A[r][c] + B[r][c] for (r, c) in profiles(A))
    return {(r, c) for (r, c) in profiles(A) if A[r][c] + B[r][c] == best}


def superrational(A, B):
    return {(z, z) for z in best_symmetric(A)}


def pce_pure(A, B):
    """Perfect cooperative equilibrium with pure commitments: leader value = max over own
    strategies of the best, for the leader, of the follower's best replies."""
    m, n = dims(A)
    L1 = max(max(A[r][y] for y in br2(B, r)) for r in range(m))
    L2 = max(max(B[x][c] for x in br1(A, c)) for c in range(n))
    return {(r, c) for (r, c) in profiles(A) if A[r][c] >= L1 and B[r][c] >= L2}


# ------------------------------------------------ perfect prediction
def guarantee(A, B, E, i):
    best = None
    secs = {}
    for s in E:
        secs.setdefault(s[i], []).append(u(A, B, i, s))
    for t, vals in secs.items():
        w = min(vals)
        if best is None or w > best:
            best = w
    return best


def verdict(A, B, E, i):
    if not E:
        return frozenset()
    g = guarantee(A, B, E, i)
    return frozenset(s for s in E if u(A, B, i, s) >= g)


def simultaneous(A, B, E):
    if not E:
        return frozenset()
    g = [guarantee(A, B, E, i) for i in (0, 1)]
    return frozenset(s for s in E if u(A, B, 0, s) >= g[0] and u(A, B, 1, s) >= g[1])


def pte_rounds(A, B):
    E = frozenset(profiles(A))
    rounds = [E]
    while True:
        E2 = simultaneous(A, B, E)
        if E2 == E:
            return rounds
        rounds.append(E2)
        if not E2:
            return rounds
        E = E2


def pte_limit(A, B):
    return set(pte_rounds(A, B)[-1])


def truncation(A, B):
    rs = pte_rounds(A, B)
    nonempty = [E for E in rs if E]
    return set(nonempty[-1])


def rpp(A, B, order):
    E = frozenset(profiles(A))
    while True:
        E2 = simultaneous(A, B, E)
        if E2:
            if E2 == E:
                return set(E)
            E = E2
        else:
            for i in order:
                E = verdict(A, B, E, i)


def spp(A, B, order):
    E = frozenset(profiles(A))
    while True:
        E0 = E
        for i in order:
            E = verdict(A, B, E, i)
        if E == E0:
            return set(E)


def union_orders(A, B):
    return rpp(A, B, (0, 1)) | rpp(A, B, (1, 0))


def is_rest_point(A, B, E):
    return all(verdict(A, B, frozenset(E), i) == frozenset(E) for i in (0, 1))


def completely_self_consistent(A, B, E):
    E = frozenset(E)
    if not E:
        return False
    g = [guarantee(A, B, E, i) for i in (0, 1)]
    return E == frozenset(s for s in profiles(A) if u(A, B, 0, s) >= g[0] and u(A, B, 1, s) >= g[1])


def self_consistent(A, B, E):
    E = frozenset(E)
    if not E:
        return True
    g = [guarantee(A, B, E, i) for i in (0, 1)]
    return all(u(A, B, 0, s) >= g[0] and u(A, B, 1, s) >= g[1] for s in E)


# ------------------------------------------------ Chwe: indirect domination, LCS
def moves(A, s, joint):
    """(coalition, target) pairs: single movers change their own strategy; the pair may
    move to any other profile when joint moves are allowed."""
    m, n = dims(A)
    r, c = s
    out = []
    for r2 in range(m):
        if r2 != r:
            out.append(((0,), (r2, c)))
    for c2 in range(n):
        if c2 != c:
            out.append(((1,), (r, c2)))
    if joint:
        for t in profiles(A):
            if t != s:
                out.append(((0, 1), t))
    return out


def indirect_dom(A, B, joint):
    """dom[b] = set of profiles a with a << b (a indirectly dominated by b)."""
    P = profiles(A)
    mv = {s: moves(A, s, joint) for s in P}
    dom = {}
    for b in P:
        R = set()
        changed = True
        while changed:
            changed = False
            for x in P:
                if x in R:
                    continue
                for S, y in mv[x]:
                    if all(u(A, B, i, x) < u(A, B, i, b) for i in S) and (y == b or y in R):
                        R.add(x)
                        changed = True
                        break
        dom[b] = R
    return dom


def lcs(A, B, joint):
    P = profiles(A)
    dom = indirect_dom(A, B, joint)
    # dominators[d] = {e : d << e}
    dominators = {d: {e for e in P if d in dom[e]} for d in P}
    mv = {s: moves(A, s, joint) for s in P}
    Y = set(P)
    while True:
        Y2 = set()
        for a in P:
            ok = True
            for S, d in mv[a]:
                cands = ({d} & Y) | (dominators[d] & Y)
                if not any(not all(u(A, B, i, a) < u(A, B, i, e) for i in S) for e in cands):
                    ok = False
                    break
            if ok:
                Y2.add(a)
        if Y2 == Y:
            return Y
        Y = Y2


def farsighted_stable_sets(A, B, joint):
    P = profiles(A)
    dom = indirect_dom(A, B, joint)
    out = []
    for k in range(1, len(P) + 1):
        for Y in combinations(P, k):
            Ys = set(Y)
            internal = not any(a in dom[b] for a in Ys for b in Ys)
            external = all(any(a in dom[b] for b in Ys) for a in P if a not in Ys)
            if internal and external:
                out.append(Ys)
    return out


# ------------------------------------------------------- game families
def strict_2x2_games():
    """576 games: each player ranks the four profiles strictly (ranks 1..4)."""
    games = []
    for p1 in permutations(range(1, 5)):
        for p2 in permutations(range(1, 5)):
            A = [[p1[0], p1[1]], [p1[2], p1[3]]]
            B = [[p2[0], p2[1]], [p2[2], p2[3]]]
            games.append((A, B))
    return games


def all_2x2_games_0123():
    games = []
    for vals in product(range(4), repeat=8):
        A = [[vals[0], vals[1]], [vals[2], vals[3]]]
        B = [[vals[4], vals[5]], [vals[6], vals[7]]]
        games.append((A, B))
    return games


def traveler(M, r):
    claims = list(range(2, M + 1))
    k = len(claims)
    A = [[None] * k for _ in range(k)]
    B = [[None] * k for _ in range(k)]
    for i, x in enumerate(claims):
        for j, y in enumerate(claims):
            if x == y:
                A[i][j] = B[i][j] = Fraction(x)
            elif x < y:
                A[i][j] = Fraction(x) + r
                B[i][j] = Fraction(x) - r
            else:
                A[i][j] = Fraction(y) - r
                B[i][j] = Fraction(y) + r
    return A, B, claims


def pd(T, R, P, S):
    # strategies: 0 = C, 1 = D
    A = [[R, S], [T, P]]
    B = [[R, T], [S, P]]
    return A, B
