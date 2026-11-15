"""Mixed and correlated profiles as a weaker existence criterion (2026-10-30).

Question: does a single-logic concept have cooperation, existence in mixed strategies, individual reasoning,
generality and self-consistency? Each part checks one route.

Conventions. Two players, 2x2 games; p and q are the probabilities of each player's FIRST strategy. Fourny's
guarantee of a set E: m_i(E) = sup over commitments with a nonempty section of the inf of u_i over the section.
Mixed: E is a set of product profiles (p, q), a commitment is a mixed strategy. Correlated: E is a set of
distributions on S, a commitment is a pure strategy, its section the distributions of E carried by that strategy.
In a game with the player swap as symmetry, an invariant fixed point E = {u >= m(E)} has m_1 = m_2 = c, so it is
E(c) = {u_1 >= c, u_2 >= c} with m(E(c)) = (c, c).

Part 1. Complete self-consistency with mixed and with correlated profiles: for symmetric games, scan c and report
   the c at which E(c) is a fixed point (m_1(E(c)) <= c + 1e-9; mixed commitments on a grid of step 1/1000).
Part 2. The trilemma with mixed profiles (Leader game; Battle of the Sexes; G_e as a contrast): exact identities
   used by the hand proof, and a grid check of the product profiles that are not strictly Pareto-dominated.
   Also the mixed and correlated Fourny rounds in Leader, the PD and chicken.
Part 3. Theorem 2 (local counterfactuals) with mixed predictions: beliefs about the answer are averaged over the
   answerer's previous mixed strategy; deviations are pure. Grid of profiles and of the switching probabilities
   (alpha_j when the answerer's previous strategy is her preferred one, beta_j otherwise). Claim: Z_PD has a mixed
   prediction iff alpha_1 <= theta; the PD with alpha = beta = 1 keeps a region of mixed profiles near (C,C).
Part 4. Nash equilibria of transformed games (existence in mixed strategies): the linear transforms
   u'_i = l_ii u_i + l_ij u_j that make (C,C) an equilibrium of every PD need l_ii = 0 (pure altruism); the
   altruistic equilibrium in the PD, the TD (the claim M beats every lower claim) and Z_PD (strictly dominated
   strategies, a payoff below the security level); altruism constrained by the security level keeps (D,D).
Part 5. The commitment game behind the mutual-commitment PTE: at {CC, DD} in the PD it is a stag hunt, so Nash
   equilibria or rationalizable commitments in place of Fourny's best commitment keep mutual defection.
"""
import itertools
from fractions import Fraction as Fr


def swap(A):
    """Player 2's payoffs in a symmetric game: u2(x, y) = u1(y, x)."""
    return [[A[y][x] for y in range(2)] for x in range(2)]


def sym(aa, ay, ya, yy):
    A = [[aa, ay], [ya, yy]]
    return A, swap(A)


GAMES = {
    'PD (5,3,1,0)': sym(3, 0, 5, 1),
    'PD (10,3,1,0)': sym(3, 0, 10, 1),
    'chicken': sym(0, 3, 1, 2),
    'G_e, e = 1/2': sym(0, 3, 1, Fr(1, 2)),
    'Leader': sym(0, 3, 2, 1),
    'Battle of the sexes': sym(0, 2, 3, 1),
}


def u(M, p, q):
    return p * q * M[0][0] + p * (1 - q) * M[0][1] + (1 - p) * q * M[1][0] + (1 - p) * (1 - q) * M[1][1]


def interval(lines, lo=0.0, hi=1.0, tol=1e-12):
    """{x in [lo, hi] : a + b x >= c for every (a, b, c)}, or None."""
    for a, b, c in lines:
        if b > tol:
            lo = max(lo, (c - a) / b)
        elif b < -tol:
            hi = min(hi, (c - a) / b)
        elif a < c - tol:
            return None
    return (lo, hi) if lo <= hi + tol else None


def floats(M):
    return [[float(v) for v in r] for r in M]


def m_mixed(A, B, c, N=1000):
    """Fourny's guarantees of E(c) with mixed commitments on a grid; None when every section is empty."""
    out = []
    for who in range(2):
        best = None
        for k in range(N + 1):
            x = k / N
            if who == 0:
                lin = [(u(M, x, 0.0), u(M, x, 1.0) - u(M, x, 0.0)) for M in (A, B)]
            else:
                lin = [(u(M, 0.0, x), u(M, 1.0, x) - u(M, 0.0, x)) for M in (A, B)]
            iv = interval([(a, b, c) for a, b in lin])
            if iv is None:
                continue
            a, b = lin[who]
            w = min(a + b * iv[0], a + b * iv[1])
            best = w if best is None or w > best else best
        out.append(best)
    return out


def m_corr(A, B, c):
    """Fourny's guarantees of {mu in Delta(S) : u(mu) >= (c, c)} with pure commitments."""
    out = []
    for who in range(2):
        best = None
        for t in range(2):
            pts = [(A[t][y], B[t][y]) for y in range(2)] if who == 0 else [(A[x][t], B[x][t]) for x in range(2)]
            lin = [(pts[1][k], pts[0][k] - pts[1][k]) for k in range(2)]  # weight w on pts[0]
            iv = interval([(a, b, c) for a, b in lin])
            if iv is None:
                continue
            a, b = lin[who]
            w = min(a + b * iv[0], a + b * iv[1])
            best = w if best is None or w > best else best
        out.append(best)
    return out


def maxmin_mixed(A, B, N=1000):
    return max(min(u(A, i / N, j / N), u(B, i / N, j / N)) for i in range(N + 1) for j in range(N + 1))


def maxmin_corr(A, B):
    pts = [(A[x][y], B[x][y]) for x in range(2) for y in range(2)]
    best = max(min(v) for v in pts)
    for s, t in itertools.combinations(pts, 2):
        d1, d2 = s[0] - t[0], s[1] - t[1]
        if d1 != d2:
            w = (t[1] - t[0]) / (d1 - d2)
            if 0 <= w <= 1:
                best = max(best, w * s[0] + (1 - w) * t[0])
    return best


def runs(cs, flags):
    """Maximal runs of consecutive grid values of c with flag True, as (first, last)."""
    out, start, prev = [], None, None
    for c, f in zip(cs + [None], flags + [False]):
        if f and start is None:
            start = c
        if not f and start is not None:
            out.append((round(start, 4), round(prev, 4)))
            start = None
        prev = c
    return out


def part1():
    print('Part 1. Invariant fixed points E(c) = {u_1 >= c, u_2 >= c} of complete self-consistency')
    for name, (A, B) in GAMES.items():
        Af, Bf = floats(A), floats(B)
        hi_m, hi_c = maxmin_mixed(Af, Bf), maxmin_corr(Af, Bf)
        lo = min(min(r) for r in Af) - 0.5
        cs = [lo + k * (max(hi_m, hi_c) - lo) / 200 for k in range(201)] + [hi_m, hi_c, 1.0, 3.0]
        cs = sorted(set(round(c, 9) for c in cs if c <= max(hi_m, hi_c) + 1e-12))
        fm, fc, gm, gc = [], [], [], []
        for c in cs:
            mm = m_mixed(Af, Bf, c) if c <= hi_m + 1e-12 else [None, None]
            mc = m_corr(Af, Bf, c) if c <= hi_c + 1e-12 else [None, None]
            fm.append(mm[0] is not None and mm[1] is not None and mm[0] <= c + 1e-9 and mm[1] <= c + 1e-9)
            fc.append(mc[0] is not None and mc[1] is not None and mc[0] <= c + 1e-9 and mc[1] <= c + 1e-9)
            gm.append(mm[0] is not None and mm[0] < c - 1e-9)
            gc.append(mc[0] is not None and mc[0] < c - 1e-9)
        print(f'   {name}: largest symmetric payoff c with E(c) nonempty: mixed {hi_m:.4f} (grid 1/1000),'
              f' correlated {float(hi_c):.4f}')
        print(f'      c-ranges of fixed points ({len(cs)} values of c): mixed {runs(cs, fm)}; correlated {runs(cs, fc)}')
        print(f'      m < c with a nonempty section (should not happen): mixed {sum(gm)}, correlated {sum(gc)}')
    A, B = GAMES['PD (5,3,1,0)']
    print('   PD (5,3,1,0), c = P = 1, exact: section of a commitment p and its guarantee')
    for p in [Fr(0), Fr(1, 4), Fr(1, 2), Fr(1)]:
        # u_1(p, q) = 1 - p + 4q - pq increases in q; u_2(p, q) = 1 - q + 4p - pq decreases in q
        lo = p / (4 - p)
        hi = min(Fr(1), (4 * p) / (1 + p))
        print(f'      p = {p}: q in [{lo}, {hi}], inf u_1 = u_1(p, {lo}) = {u(A, p, lo)}, u_2 there = {u(B, p, lo)}')
    print('      so E(1) = {u >= (1, 1)}, which contains mutual defection (p = q = 0), is a fixed point')


def part2():
    print('\nPart 2. The trilemma with mixed profiles')
    A, B = GAMES['Leader']
    q = Fr(3, 7)
    print('   Leader (AA (0,0), AY (3,2), YA (2,3), YY (1,1)); u_1(p,q) = 1 + 2p + q - 4pq, u_2(p,q) = 1 + p + 2q - 4pq')
    print(f'   identities at q = 3/7: u_1(1,q) = {u(A, 1, q)} = 3(1-q), u_2(1,q) = {u(B, 1, q)} = 2(1-q),'
          f' u_1(0,q) = {u(A, 0, q)} = 1 + q;'
          f' formula check on a 11x11 grid: {all(u(A, Fr(i, 10), Fr(j, 10)) == 1 + 2 * Fr(i, 10) + Fr(j, 10) - 4 * Fr(i * j, 100) for i in range(11) for j in range(11))}')
    for name in ['Leader', 'Battle of the sexes', 'G_e, e = 1/2', 'chicken']:
        A, B = GAMES[name]
        Af, Bf = floats(A), floats(B)
        N = 100
        prof = [(i / N, j / N) for i in range(N + 1) for j in range(N + 1)]
        vals = [(u(Af, p, q), u(Bf, p, q)) for p, q in prof]
        und = [pq for pq, v in zip(prof, vals)
               if not any(w[0] > v[0] + 1e-12 and w[1] > v[1] + 1e-12 for w in vals)]
        sym_und = [p for p, q in und if p == q]
        print(f'   {name}: grid profiles (step 1/{N}) not strictly dominated by a grid profile: {len(und)};'
              f' first and last {und[:2]} ... {und[-2:]}; symmetric ones: {sym_und}')
    print('   Fourny rounds with mixed and correlated profiles (player 1\'s guarantee per round)')
    for name in ['Leader', 'PD (5,3,1,0)', 'chicken']:
        A, B = GAMES[name]
        Af, Bf = floats(A), floats(B)
        for label, mfun, top in [('mixed', lambda c: m_mixed(Af, Bf, c), maxmin_mixed(Af, Bf)),
                                 ('correlated', lambda c: m_corr(Af, Bf, c), maxmin_corr(Af, Bf))]:
            c, path = -1e9, []
            for _ in range(8):
                if c > top + 1e-12:
                    path.append('empty')
                    break
                m = mfun(c)
                if m[0] is None:
                    path.append('no nonempty section: no constraint, set unchanged')
                    break
                path.append(round(float(m[0]), 4))
                if abs(m[0] - c) < 1e-9:
                    path.append('fixed')
                    break
                c = m[0]
            print(f'      {name}, {label}: {path} (largest symmetric payoff {float(top):.4f})')


def opposed_answer(M_i, M_j, row_is_i, t, prev_prob, alpha, beta):
    """Probability that the answerer j plays her FIRST strategy after i deviates to pure t, with beliefs averaged
    over j's previous mixed strategy (prev_prob = probability of her first strategy). The continuation must have
    opposed strict rankings of j's two strategies."""
    if row_is_i:
        ui, uj = [M_i[t][0], M_i[t][1]], [M_j[t][0], M_j[t][1]]
    else:
        ui, uj = [M_i[0][t], M_i[1][t]], [M_j[0][t], M_j[1][t]]
    assert (uj[0] - uj[1]) * (ui[0] - ui[1]) < 0, 'continuation not opposed'
    pref = 0 if uj[0] > uj[1] else 1  # j's preferred strategy
    on_pref = prev_prob if pref == 0 else 1 - prev_prob  # j's previous strategy is her preferred one
    p_pref = on_pref * (1 - alpha) + (1 - on_pref) * beta
    return p_pref if pref == 0 else 1 - p_pref


def predicted(A, B, p, q, al, be):
    """Mixed profile (p, q) survives the local-counterfactual test with pure deviations t != own strategy.
    al = (alpha_1, alpha_2), be = (beta_1, beta_2); the answerer's parameters are used."""
    u1, u2 = u(A, p, q), u(B, p, q)
    for t in range(2):
        if not ((p == 1 and t == 0) or (p == 0 and t == 1)):
            y = opposed_answer(A, B, True, t, q, al[1], be[1])
            if A[t][0] * y + A[t][1] * (1 - y) > u1:
                return False
        if not ((q == 1 and t == 0) or (q == 0 and t == 1)):
            x = opposed_answer(B, A, False, t, p, al[0], be[0])
            if B[0][t] * x + B[1][t] * (1 - x) > u2:
                return False
    return True


def part3():
    print('\nPart 3. Theorem 2 with mixed predictions (beliefs averaged over the answerer\'s previous mixture)')
    grid = [Fr(k, 20) for k in range(21)]
    for T, R, P, S in [(5, 3, 1, 0), (10, 3, 1, 0), (4, 3, 2, 0)]:
        th = Fr(P - S, R - S)
        Z = [[R, R - P + S], [S, S - 1]]
        Zb = [[-v for v in r] for r in Z]
        PDA = [[R, S], [T, P]]
        PDB = swap(PDA)
        params = sorted(set([Fr(0), Fr(1, 10), th, th + Fr(1, 100), Fr(1, 2), Fr(9, 10), Fr(1)]))
        bad, total, kinds = 0, 0, {}
        for a1, a2, b1, b2 in itertools.product(params, repeat=4):
            surv = [(p, q) for p in grid for q in grid if predicted(Z, Zb, p, q, (a1, a2), (b1, b2))]
            total += 1
            if (len(surv) > 0) != (a1 <= th):
                bad += 1
            kind = 'none' if not surv else ('only (a,b\')' if surv == [(1, 0)] else 'also mixed')
            key = ('alpha_1 = 0' if a1 == 0 else ('0 < alpha_1 <= theta' if a1 <= th else 'alpha_1 > theta'), kind)
            kinds[key] = kinds.get(key, 0) + 1
        pd_mirror = [(p, q) for p in grid for q in grid if predicted(PDA, PDB, p, q, (1, 1), (1, 1))]
        dd_out = all(not predicted(PDA, PDB, 0, 0, (a1, a2), (b1, b2)) for a1 in params for a2 in params
                     for b1 in params for b2 in params if max(a1, a2) > th)
        print(f'   PD {(T, R, P, S)}, theta = {th}: {total} parameter vectors (alpha_1, alpha_2, beta_1, beta_2) in'
              f' {[str(x) for x in params]}^4; "Z_PD has a mixed prediction iff alpha_1 <= theta" fails in {bad}')
        print(f'      counts by (alpha_1, what Z_PD keeps on the grid of step 1/20): {sorted(kinds.items())}')
        print(f'      PD itself with alpha = beta = 1: mixed predictions on the grid {pd_mirror}; mutual defection'
              f' ruled out whenever max(alpha_1, alpha_2) > theta: {dd_out}')


def nash_2x2(A, B):
    """Nash equilibria (p, q) of a 2x2 bimatrix game (row payoffs A, column payoffs B) with isolated equilibria."""
    eqs = []
    for x, y in itertools.product(range(2), range(2)):
        if A[x][y] >= A[1 - x][y] and B[x][y] >= B[x][1 - y]:
            eqs.append((Fr(1 - x), Fr(1 - y)))
    dA = A[0][0] - A[0][1] - A[1][0] + A[1][1]
    dB = B[0][0] - B[1][0] - B[0][1] + B[1][1]
    if dA != 0 and dB != 0:
        q = Fr(A[1][1] - A[0][1]) / dA
        p = Fr(B[1][1] - B[1][0]) / dB
        if 0 < p < 1 and 0 < q < 1:
            eqs.append((p, q))
    return eqs


def td(M, r):
    """Traveler's dilemma, claims 2..M: u1[x][y] for claims x of player 1 and y of player 2."""
    cl = list(range(2, M + 1))
    return cl, [[x if x == y else (x + r if x < y else y - r) for y in cl] for x in cl]


def part4():
    print('\nPart 4. Nash equilibria of transformed games')
    for l11, l12 in [(1, 1), (1, 3), (1, 10), (0, 1)]:
        fails = [T for T in [5, 10, 50, 1000] if l11 * (T - 3) > l12 * (3 - 0)]
        print(f'   weights (l_ii, l_ij) = ({l11}, {l12}): PDs (T,3,1,0), T in {{5, 10, 50, 1000}}, where (C,C) is not'
              f' an equilibrium of the transformed game: {fails}')
    for T, R, P, S in [(5, 3, 1, 0), (10, 3, 1, 0), (100, 3, 2, 1)]:
        A = [[R, S], [T, P]]
        print(f'   PD {(T, R, P, S)}: altruistic equilibria (p, q = probabilities of C) {nash_2x2(swap(A), A)}')
    ok, cases = True, 0
    for M in range(3, 13):
        for r in [Fr(1, 2), 1, 2, 5]:
            cl, u1 = td(M, r)
            n = len(cl)
            u2 = [[u1[y][x] for y in range(n)] for x in range(n)]
            for k in range(n - 1):
                for y in range(n):
                    d = u2[-1][y] - u2[k][y]
                    cases += 1
                    if (y >= k and d <= 0) or (y < k and d != 0):
                        ok = False
    print(f'   TD, M = 3..12, r in {{1/2, 1, 2, 5}} ({cases} comparisons): for player 1 maximizing player 2\'s payoff,'
          f' the claim M gives strictly more than a claim k < M against every y >= k and as much against y < k: {ok}')
    R, P, S = 3, 1, 0
    Z = [[R, R - P + S], [S, S - 1]]
    Zb = [[-v for v in r] for r in Z]
    eq = nash_2x2(Zb, Z)
    sec1 = max(min(r) for r in Z)
    print(f'   Z_PD for (.,3,1,0): altruistic equilibria {eq} (p = prob of a, q = prob of b); player 1 gets'
          f' {[u(Z, p, q) for p, q in eq]}, her security level is {sec1}; a\' and b are strictly dominated')
    for T, R, P, S in [(5, 3, 1, 0), (10, 3, 1, 0)]:
        A = [[R, S], [T, P]]
        feas = [p for p in [Fr(k, 10) for k in range(11)] if u(A, p, 0) >= P]
        print(f'   PD {(T, R, P, S)}, altruism under the security constraint: against q = 0 the strategies p on the'
              f' grid 1/10 with u_1 >= P are {feas}, so (D,D) is an equilibrium')


def part5():
    print('\nPart 5. The commitment game of the mutual-commitment PTE at E_1 = {CC, DD} in the PD')
    for T, R, P, S in [(5, 3, 1, 0), (10, 3, 1, 0)]:
        A = [[R, S], [T, P]]
        E = [(0, 0), (1, 1)]
        sec = [min(A[x][y] for x2, y in E if x2 == x) for x in range(2)]
        pi1 = [[min(sec[t], A[t][y]) for y in range(2)] for t in range(2)]
        print(f'   PD {(T, R, P, S)}: player 1\'s commitment payoffs min(section minimum, u_1(t, y)), rows and'
              f' columns C, D: {pi1}; Nash equilibria (p, q = probabilities of C) {nash_2x2(pi1, swap(pi1))};'
              f' Fourny\'s best commitment (largest section minimum): {"CD"[sec.index(max(sec))]}')


if __name__ == '__main__':
    part1()
    part2()
    part3()
    part4()
    part5()
