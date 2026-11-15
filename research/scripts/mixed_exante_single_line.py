"""mixed_exante_single_line.py: checks of a single line of reasoning judged by expected payoffs
throughout.

Run: uv run --no-project --with numpy python research/scripts/mixed_exante_single_line.py [parts]
     (parts: any of A B C D E F G H; default all). Output recorded in mixed_exante_single_line.out.

Logic (summary written before the code).
- Part A (exact). One-strategy players and single welfare criteria.
  G2: players 1, 2 play the zero-sum 2x2 game u1 = [[3,-1],[-2,1]] (no ties), player 3 has one strategy and gets
  0, 1, 2, 3 at (a,b), (a,b'), (a',b), (a',b') (no ties). Exact v^m, the cross IR^m = {p = 3/7} U {q = 2/7}, gains of
  the movers (0,0) on it, player 3's payoff along both arms; two members with the same movers' payoffs and different
  player-3 payoffs: the leximin over the movers predicts a Pareto-dominated profile.
  G1': same shape, players 1 and 2 have constant payoff 0 (ties), player 3 as in G2: B = every mixed profile.
  G3 (no ties, three players, player 3 dummy): (a,b) is the bliss profile of the movers but the leximin over all
  players' gains prefers (a',b'): the all-player leximin excludes B, so it is not unilateral (M).
- Part B (exact). The attainable gain vectors used by the welfarist impossibility: in G1' every (0,0,z), z in [0,3];
  in G2 the gain vectors on IR^m are (0,0,z), z in [5/7, 19/7].
- Part C (exact, Fractions). Ex ante perfect prediction on the mixed extension of prisoner's dilemmas (symmetric and
  asymmetric payoffs): the round from all mixed profiles gives E(P); in E(c) (c between P and R) the section of each
  mixed commitment p is an interval [q_lo, q_hi] of the other's cooperation probability; worst answer for the
  deviator (Fourny) and best answer for the answerer (credible) both give the deviator exactly c; hence gamma = c and
  E(c) is a rest set. Generous answers (best for the deviator within the section) give gamma = max over the state.
  Laplace answers (uniform over the section) from all profiles: gamma_i = max_p u_i(p, 1/2), compared with what
  any profile can give both. Checked on a grid of rational p (k/120) for several c and several PDs.
- Part D (exact endpoints, numerical frontier). PD (10,3,1,0): the Pareto-optimal mixed profiles inside IR^m are the
  segment p + q = 4/3, p in [1/3, 1] (grid nondominated check, 401 x 401). Each commitment has one answer on it,
  so perfect prediction on this set gives gamma = (23/3, 23/3) and an empty simultaneous step.
- Part E (numerical). Nash product over IR^m against the best symmetric mixed profile on symmetric 2x2 games in the
  domain U* (a strictly dominant strategy d with u(d,d') > u(d',d)), random payoffs (seed 20261107): does the Nash
  product maximizer differ from (sigma*, sigma*)? Grid 401 x 401 plus local refinement.
- Part F (exact, one grid). Rong-Halpern M-PCE (BU_i = sup over i's mixed strategies and over the Nash equilibria
  of the others' residual game, Rong-Halpern arXiv 1412.6722, text before Definition 2.8): in G2, BU = (1/7, -1/7,
  13/7) and the maximizers of min_i (u_i - BU_i) include (3/7, 2/7), Pareto-dominated by (3/7, 0). Three players
  without a dummy: an M-PCE gives player 3 less than her mixed security level
  (exact payoffs at the candidate, grid 101^3 for the maximum). With a dummy counted: B is not inside M-PCE.
- Part G (numerical). Matching answers in 2x2 games without ties: each player, committing to p, expects the other
  to answer so that their normalized gains (u_i - v_i^m) / (max u_i - v_i^m) are equal. In 2x2 games each p has at
  most one matching answer, so the rounds started at IR^m end at E* = {sigma in IR^m: both normalized gains >= e*},
  e* the largest common gain on the equal-gain curve. Count games in which no equal-gain profile lies in IR^m, and in
  which the max-min of normalized gains exceeds e* (then E* contains a Pareto-dominated profile).
- Part H (numerical). Hurwicz answers: a commitment is valued at alpha times its worst plus (1 - alpha) times its
  best payoff over its section. States stay threshold sets E(c) (from all profiles), so the rounds are a sequence of
  thresholds c_k, c_(k+1) = gamma(E(c_k)). 2x2 games: sections computed exactly in the answerer's probability for a
  grid of the committer's probability (2001 points). Stops: convergence (|step| < 1e-10), overshoot (no profile
  meets gamma: the simultaneous step is empty, truncation keeps E(c_k)), or the iteration cap. Named games, random
  2x2 games without ties, symmetric 2x2 games in U*; alpha in (0.9, 0.99). Records: overshoot at round 0 (truncation
  then predicts all mixed profiles, outside IR^m), whether the final state is thin (every member gives about c*,
  hence Pareto-optimal), and the distance from c* to the leximin gain point of IR^m (grid).
"""
import random
import sys
from fractions import Fraction as Fr

import numpy as np

ONE, ZERO = Fr(1), Fr(0)


def bil(M, p, q):
    """Expected payoff in a 2x2 game, M[i][j] at (row i, column j), p = prob(row 0), q = prob(column 0)."""
    return (p * q * M[0][0] + p * (1 - q) * M[0][1] + (1 - p) * q * M[1][0]
            + (1 - p) * (1 - q) * M[1][1])


def vm_row(M):
    """Exact mixed security level of the row player of 2x2 matrix M (her payoffs). Returns (value, p)."""
    cands = [ZERO, ONE]
    # intersection of the two column lines: p*M00 + (1-p)*M10 = p*M01 + (1-p)*M11
    den = (M[0][0] - M[1][0]) - (M[0][1] - M[1][1])
    if den != 0:
        p = (M[1][1] - M[1][0]) / den
        if 0 <= p <= 1:
            cands.append(p)
    best = None
    for p in cands:
        w = min(p * M[0][0] + (1 - p) * M[1][0], p * M[0][1] + (1 - p) * M[1][1])
        if best is None or w > best[0]:
            best = (w, p)
    return best


def transpose(M):
    return [[M[0][0], M[1][0]], [M[0][1], M[1][1]]]


def part_A():
    print("== Part A: one-strategy players and single welfare criteria (exact) ==")
    U1 = [[Fr(3), Fr(-1)], [Fr(-2), Fr(1)]]
    U2 = [[-x for x in r] for r in U1]
    U3 = [[Fr(0), Fr(1)], [Fr(2), Fr(3)]]
    v1, p1 = vm_row(U1)
    v2, q2 = vm_row(transpose(U2))  # player 2 chooses columns: rows of the transpose
    print("G2 payoffs (a,b),(a,b'),(a',b),(a',b'):", [(U1[i][j], U2[i][j], U3[i][j]) for i in (0, 1) for j in (0, 1)])
    print("no ties:", all(len({M[i][j] for i in (0, 1) for j in (0, 1)}) == 4 for M in (U1, U2, U3)))
    print("v1^m =", v1, "at p =", p1, "; v2^m =", v2, "at q =", q2, "; v3^m =", min(min(r) for r in U3))
    # u1 - v1 = k (p - 3/7)(q - 2/7)
    k = U1[0][0] - U1[0][1] - U1[1][0] + U1[1][1]
    ok = all(bil(U1, Fr(a, 7), Fr(b, 7)) - v1 == k * (Fr(a, 7) - p1) * (Fr(b, 7) - q2)
             for a in range(8) for b in range(8))
    print("u1 - v1 = %s (p - %s)(q - %s) on the 1/7 grid: %s" % (k, p1, q2, ok))
    print("so IR^m = {u1 = v1} = {p = 3/7} U {q = 2/7} (u2 = -u1), movers' gains (0,0) on it")
    for (p, q) in [(Fr(3, 7), ZERO), (Fr(3, 7), ONE), (ZERO, Fr(2, 7)), (ONE, Fr(2, 7)), (Fr(3, 7), Fr(2, 7)),
                   (Fr(11, 14), Fr(2, 7))]:
        print("  (p,q) = (%s,%s): u = (%s, %s, %s)" % (p, q, bil(U1, p, q), bil(U2, p, q), bil(U3, p, q)))
    print("u3 on arm p=3/7: 15/7 - q:", all(bil(U3, Fr(3, 7), Fr(b, 9)) == Fr(15, 7) - Fr(b, 9) for b in range(10)))
    print("u3 on arm q=2/7: 19/7 - 2p:", all(bil(U3, Fr(a, 9), Fr(2, 7)) == Fr(19, 7) - 2 * Fr(a, 9) for a in range(10)))
    print("=> leximin over the movers predicts the whole cross, including (3/7,1) with u = (1/7,-1/7,8/7),")
    print("   Pareto-dominated by (3/7,0) with u = (1/7,-1/7,15/7): PO_b fails in a game without ties.")
    print("   Leximin over all players: maximizes u3 on the cross: (0,2/7) with u3 = 19/7 (single profile).")
    # G3: no ties, dummy, all-player leximin excludes the bliss profile
    A1 = [[Fr(4), Fr(1)], [Fr(0), Fr(3)]]
    A2 = [[Fr(4), Fr(0)], [Fr(1), Fr(3)]]
    A3 = [[Fr(0), Fr(1)], [Fr(2), Fr(3)]]
    w1, _ = vm_row(A1)
    w2, _ = vm_row(transpose(A2))
    print("G3 payoffs (a,b),(a,b'),(a',b),(a',b'):", [(A1[i][j], A2[i][j], A3[i][j]) for i in (0, 1) for j in (0, 1)])
    print("no ties:", all(len({M[i][j] for i in (0, 1) for j in (0, 1)}) == 4 for M in (A1, A2, A3)))
    print("v^m =", (w1, w2, ZERO), "; B = {(a,b)} (both movers get 4, their maximum, only there)")
    g_ab = sorted([A1[0][0] - w1, A2[0][0] - w2, A3[0][0]])
    g_bb = sorted([A1[1][1] - w1, A2[1][1] - w2, A3[1][1]])
    print("sorted gains: (a,b)", g_ab, " (a',b')", g_bb, " -> (a',b') leximin-better:", g_bb > g_ab)
    print("=> the all-player leximin excludes the bliss profile (a,b): not unilateral (M), in a game without ties.")
    print("   Nash product over all players: (a,b) gives", (A1[0][0] - w1) * (A2[0][0] - w2) * A3[0][0],
          ", (a',b') gives", (A1[1][1] - w1) * (A2[1][1] - w2) * A3[1][1])
    print()


def part_B():
    print("== Part B: attainable gain vectors for the welfarist impossibility (exact) ==")
    U3 = [[Fr(0), Fr(1)], [Fr(2), Fr(3)]]
    print("G1': movers' payoffs constant 0, so B = every mixed profile and the movers' gains are (0,0) everywhere;")
    vals = sorted({bil(U3, Fr(a, 4), Fr(b, 4)) for a in range(5) for b in range(5)})
    print("  player 3's payoff (v3^m = 0) ranges over [0,3] (bilinear, continuous): sample", vals[0], "..", vals[-1])
    print("G2: on IR^m player 3's payoff ranges over [8/7,15/7] (arm p=3/7) and [5/7,19/7] (arm q=2/7),")
    print("  so the attainable gain vectors are (0,0,z), z in [5/7,19/7], a subset of G1''s.")
    print("  Any rule choosing from the attainable gain vectors with Sen's contraction property chooses all of")
    print("  [5/7,19/7] in G2, since it chooses all of [0,3] in G1'; so it predicts (3/7,1), dominated.")
    print()


def pd_payoffs(T1, R1, P1, S1, T2, R2, P2, S2):
    U1 = [[R1, S1], [T1, P1]]   # rows: C, D for player 1; columns: C, D for player 2
    U2 = [[R2, T2], [S2, P2]]
    return U1, U2


def section(U1, U2, p, c1, c2):
    """Interval of q (player 2's prob of C) with u1(p,q) >= c1 and u2(p,q) >= c2, or None."""
    # u1(p,q) = a1 + b1 q with b1 > 0; u2(p,q) = a2 + b2 q with b2 < 0
    a1 = bil(U1, p, ZERO)
    b1 = bil(U1, p, ONE) - a1
    a2 = bil(U2, p, ZERO)
    b2 = bil(U2, p, ONE) - a2
    assert b1 > 0 and b2 < 0
    lo = max(ZERO, (c1 - a1) / b1)
    hi = min(ONE, (c2 - a2) / b2)
    if lo > hi:
        return None
    return lo, hi


def part_C():
    print("== Part C: ex ante perfect prediction on the mixed extension of prisoner's dilemmas (exact) ==")
    pds = [(10, 3, 1, 0, 10, 3, 1, 0), (5, 3, 1, 0, 5, 3, 1, 0), (6, 3, 1, 0, 6, 3, 1, 0),
           (10, 3, 1, 0, 7, 5, 2, -1), (4, 3, 2, 1, 9, 2, 1, -3)]
    grid = [Fr(k, 120) for k in range(121)]
    for pd in pds:
        T1, R1, P1, S1, T2, R2, P2, S2 = [Fr(x) for x in pd]
        U1, U2 = pd_payoffs(T1, R1, P1, S1, T2, R2, P2, S2)
        # round from all mixed profiles: Fourny inf over q of u1(p,q) = u1(p,0); credible answer is q=0 (D dominant)
        g1_four = max(min(bil(U1, p, ZERO), bil(U1, p, ONE)) for p in grid)
        g1_cred = max(bil(U1, p, ZERO) for p in grid)
        g1_gen = max(max(bil(U1, p, ZERO), bil(U1, p, ONE)) for p in grid)
        # Laplace answers: the answer is uniform on the section; from all profiles the section is [0,1], mean q = 1/2
        g1_lap = max(bil(U1, p, Fr(1, 2)) for p in grid)
        g2_lap = max(bil(transpose(U2), p, Fr(1, 2)) for p in grid)
        none_meets = all(not (bil(U1, a, b) >= g1_lap and bil(U2, a, b) >= g2_lap) for a in grid[::4] for b in grid[::4])
        line = "PD %s: from all profiles gamma_1 Fourny %s, credible %s (P1 = %s); generous %s (T1 = %s)" % (
            pd, g1_four, g1_cred, P1, g1_gen, T1)
        print(line)
        print("  Laplace answers from all profiles: gamma = (%s, %s); no profile of the 1/30 grid meets it: %s;"
              " max over pure profiles of u1 + u2 = %s" % (
                  g1_lap, g2_lap, none_meets, max(U1[i][j] + U2[i][j] for i in (0, 1) for j in (0, 1))))
        cs = [(P1, P2), ((P1 + R1) / 2, (P2 + R2) / 2), (R1 - Fr(1, 10), R2 - Fr(1, 10))]
        for (c1, c2) in cs:
            vals_four, vals_cred, vals_gen, nonempty = [], [], [], 0
            for p in grid:
                sec = section(U1, U2, p, c1, c2)
                if sec is None:
                    continue
                nonempty += 1
                lo, hi = sec
                vals_four.append(min(bil(U1, p, lo), bil(U1, p, hi)))   # u1 increasing in q: min at lo
                vals_cred.append(bil(U1, p, lo))                          # u2 decreasing in q: answerer picks lo
                vals_gen.append(bil(U1, p, hi))
            # player 2: by the symmetric computation with roles exchanged
            V1 = transpose(U2)
            V2 = transpose(U1)
            vals2 = []
            for p in grid:
                sec = section(V1, V2, p, c2, c1)
                if sec is None:
                    continue
                vals2.append(bil(V1, p, sec[0]))
            print("  E(c), c = (%s, %s): commitments with answers %d/121; gamma_1 Fourny = %s, credible = %s,"
                  " all equal c1: %s; gamma_2 (both rules) = %s; generous gamma_1 = %s" % (
                      c1, c2, nonempty, max(vals_four), max(vals_cred),
                      all(v == c1 for v in vals_four) and all(v == c1 for v in vals_cred), max(vals2), max(vals_gen)))
    print("=> E(P) is reached in one round and is a rest set for every verdict; (D,D) is in it.")
    print()


def part_D():
    print("== Part D: Pareto-filtered rounds in the PD (10,3,1,0) ==")
    U1, U2 = pd_payoffs(*[Fr(x) for x in (10, 3, 1, 0, 10, 3, 1, 0)])
    N = 401
    ps = np.linspace(0, 1, N)
    P, Q = np.meshgrid(ps, ps, indexing="ij")
    f = lambda M: (P * Q * float(M[0][0]) + P * (1 - Q) * float(M[0][1]) + (1 - P) * Q * float(M[1][0])
                   + (1 - P) * (1 - Q) * float(M[1][1]))
    u1, u2 = f(U1), f(U2)
    ir = (u1 >= 1 - 1e-12) & (u2 >= 1 - 1e-12)
    pts = np.stack([u1[ir], u2[ir], P[ir], Q[ir]], axis=1)
    order = np.lexsort((-pts[:, 1], -pts[:, 0]))
    pts = pts[order]
    nd, best2 = [], -np.inf
    for r in pts:
        if r[1] > best2 + 1e-12:
            nd.append(r)
            best2 = r[1]
    nd = np.array(nd)
    dev = np.abs(nd[:, 2] + nd[:, 3] - 4 / 3)
    print("grid %dx%d: %d nondominated points in IR^m; max |p+q-4/3| = %.4f (grid step %.4f)" % (
        N, N, len(nd), dev.max(), 1 / (N - 1)))
    for (p, q) in [(Fr(1, 3), ONE), (Fr(2, 3), Fr(2, 3)), (ONE, Fr(1, 3))]:
        print("  (p,q) = (%s,%s): u = (%s, %s)" % (p, q, bil(U1, p, q), bil(U2, p, q)))
    print("on p + q = 4/3: u1 = 13 - 18p + 6p^2 (decreasing on [1/3,1]), u2 = -1/3 + 2p + 6p^2 (increasing)")
    print("check at p = k/30:", all(bil(U1, Fr(k, 30), Fr(4, 3) - Fr(k, 30)) == 13 - 18 * Fr(k, 30) + 6 * Fr(k, 30) ** 2
                                     and bil(U2, Fr(k, 30), Fr(4, 3) - Fr(k, 30)) == Fr(-1, 3) + 2 * Fr(k, 30)
                                     + 6 * Fr(k, 30) ** 2 for k in range(10, 31)))
    print("=> each commitment p has one answer q = 4/3 - p on the set, gamma = (23/3, 23/3), the simultaneous step")
    print("   is empty; truncation keeps the segment, player 1 first keeps (1/3,1) paying (23/3,1), the union keeps")
    print("   both endpoints: none is {(2/3,2/3)}.")
    print()


def part_E(n_games=300, seed=20261107):
    print("== Part E: Nash product over IR^m vs (sigma*, sigma*) on symmetric 2x2 games in U* (numerical) ==")
    rng = random.Random(seed)
    N = 401
    ps = np.linspace(0, 1, N)
    P, Q = np.meshgrid(ps, ps, indexing="ij")
    differ, checked, maxgap = [], 0, 0.0
    while checked < n_games:
        vals = rng.sample(range(0, 1000), 4)
        A, B, C, D = [v / 100 for v in vals]   # u1(a,a), u1(a,b), u1(b,a), u1(b,b); a dominant iff A > C and B > D
        if not (A > C and B > D and B > C):
            continue
        checked += 1
        M1 = [[A, B], [C, D]]
        u1 = P * Q * A + P * (1 - Q) * B + (1 - P) * Q * C + (1 - P) * (1 - Q) * D
        u2 = Q * P * A + Q * (1 - P) * B + (1 - Q) * P * C + (1 - Q) * (1 - P) * D
        vm = float(vm_row([[Fr(A), Fr(B)], [Fr(C), Fr(D)]])[0])
        fs = ps * ps * A + ps * (1 - ps) * (B + C) + (1 - ps) ** 2 * D
        s = ps[np.argmax(fs)]
        ir = (u1 >= vm - 1e-12) & (u2 >= vm - 1e-12)
        prod = np.where(ir, (u1 - vm) * (u2 - vm), -np.inf)
        i, j = np.unravel_index(np.argmax(prod), prod.shape)
        # local refinement around the grid maximum
        best = (prod[i, j], ps[i], ps[j])
        for _ in range(3):
            lo_p, hi_p = max(0, best[1] - 0.01), min(1, best[1] + 0.01)
            lo_q, hi_q = max(0, best[2] - 0.01), min(1, best[2] + 0.01)
            pp = np.linspace(lo_p, hi_p, 81)
            qq = np.linspace(lo_q, hi_q, 81)
            PP, QQ = np.meshgrid(pp, qq, indexing="ij")
            v1 = PP * QQ * A + PP * (1 - QQ) * B + (1 - PP) * QQ * C + (1 - PP) * (1 - QQ) * D
            v2 = QQ * PP * A + QQ * (1 - PP) * B + (1 - QQ) * PP * C + (1 - QQ) * (1 - PP) * D
            pr = np.where((v1 >= vm) & (v2 >= vm), (v1 - vm) * (v2 - vm), -np.inf)
            a, b = np.unravel_index(np.argmax(pr), pr.shape)
            if pr[a, b] >= best[0]:
                best = (pr[a, b], pp[a], qq[b])
        fstar = s * s * A + s * (1 - s) * (B + C) + (1 - s) ** 2 * D
        gap = best[0] - (fstar - vm) ** 2
        maxgap = max(maxgap, gap)
        if abs(best[1] - best[2]) > 2e-3 and gap > 1e-6:
            differ.append(((A, B, C, D), round(best[1], 4), round(best[2], 4), round(s, 4), gap))
    print("games checked: %d; Nash product maximizer asymmetric and strictly above (f*-v)^2: %d; max gap %.2e" % (
        checked, len(differ), maxgap))
    for d in differ[:5]:
        print("  witness (A,B,C,D) =", d[0], "maximizer (p,q) ~", (d[1], d[2]), "sigma* ~", d[3], "gap", d[4])
    print()


def part_F():
    print("== Part F: Rong-Halpern M-PCE in G2 (exact, BU from Nash equilibria of the residual game) ==")
    U1 = [[Fr(3), Fr(-1)], [Fr(-2), Fr(1)]]
    U3 = [[Fr(0), Fr(1)], [Fr(2), Fr(3)]]
    BU1, BU2 = Fr(1, 7), Fr(-1, 7)          # zero-sum residual: the leader gets the value
    BU3 = bil(U3, Fr(3, 7), Fr(2, 7))         # unique Nash equilibrium of the movers' zero-sum game
    print("BU = (%s, %s, %s)" % (BU1, BU2, BU3))
    print("min(u1 - 1/7, u2 + 1/7) <= 0 with equality iff u1 = 1/7 (cross); so max of min_i(u_i - BU_i) is 0, reached")
    print("on the cross where u3 >= 13/7: {(3/7,q): q <= 2/7} U {(p,2/7): p <= 3/7}")
    for (p, q) in [(Fr(3, 7), ZERO), (Fr(3, 7), Fr(2, 7)), (ZERO, Fr(2, 7))]:
        print("  (%s,%s): u3 = %s, min gain = %s" % (p, q, bil(U3, p, q),
                                                   min(bil(U1, p, q) - BU1, -bil(U1, p, q) - BU2, bil(U3, p, q) - BU3)))
    print("=> the M-PCE (3/7,2/7) is Pareto-dominated by the M-PCE (3/7,0): not every M-PCE is Pareto-optimal.")
    # (2) three players, no dummy: M-PCE outside IR^m
    # players 1, 2 choose a or b, player 3 chooses X or Y. Under Y: (a,a) -> (2,1,1), (b,b) -> (1,2,1), mismatch
    # (0,0,1). Under X: (a,a) -> (2,2,1/2), else (0,0,0).
    def pay3(p, q, r):   # p, q = prob of a for players 1, 2; r = prob of X
        aa, bb = p * q, (1 - p) * (1 - q)
        u1 = (1 - r) * (2 * aa + 1 * bb) + r * 2 * aa
        u2 = (1 - r) * (1 * aa + 2 * bb) + r * 2 * aa
        u3 = (1 - r) * 1 + r * Fr(1, 2) * aa if isinstance(p, Fr) else (1 - r) + r * 0.5 * aa
        return u1, u2, u3
    BU = (Fr(2), Fr(2), Fr(1))
    vm3 = (Fr(0), Fr(0), Fr(1))
    print("three players: BU = (2,2,1) (player 1 commits to a, player 3's dominant Y, player 2 answers a; symmetric for"
          " player 2; player 3's Y gives 1); v^m = (0,0,1) (Y guarantees 1 to player 3; the others can be held to 0)")
    u = pay3(ONE, ONE, Fr(2, 3))
    print("  (a, a, 2/3 X + 1/3 Y): u = (%s, %s, %s), min_i (u_i - BU_i) = %s" % (
        u[0], u[1], u[2], min(u[i] - BU[i] for i in range(3))))
    n = 101
    g = np.linspace(0, 1, n)
    P, Q, R = np.meshgrid(g, g, g, indexing="ij")
    u1, u2, u3 = pay3(P, Q, R)
    f = np.minimum(np.minimum(u1 - 2, u2 - 2), u3 - 1)
    i = np.unravel_index(np.argmax(f), f.shape)
    print("  grid %d^3: max of min_i (u_i - BU_i) = %.4f at (p,q,r) = (%.2f, %.2f, %.2f); player 3 gets %.4f there" % (
        n, f[i], g[i[0]], g[i[1]], g[i[2]], u3[i]))
    top = f >= f.max() - 1e-9
    print("  every grid maximizer gives player 3 less than v_3^m = 1: %s" % bool(np.all(u3[top] < 1)))
    # (3) dummy counted: coordination game with a dummy who prefers (b,b)
    print("coordination (a,a) -> (2,2), (b,b) -> (1,1), mismatch (0,0); dummy gets 0, 10 at (b,b), 0: BU = (2,2,10);"
          " B = {(a,a)} with min_i (u_i - BU_i) = %s, while (b,b) has %s: B is not inside M-PCE" % (
              min(2 - 2, 2 - 2, 0 - 10), min(1 - 2, 1 - 2, 10 - 10)))
    print()


def part_G(n_games=2000, seed=20261107):
    print("== Part G: matching answers (equal normalized gains) in 2x2 games without ties (numerical) ==")
    rng = random.Random(seed)
    N = 801
    ps = np.linspace(0, 1, N)
    P, Q = np.meshgrid(ps, ps, indexing="ij")
    no_eq, po_fail, examples = 0, 0, []
    for _ in range(n_games):
        a = rng.sample(range(100), 4)
        b = rng.sample(range(100), 4)
        M1 = [[Fr(a[0]), Fr(a[1])], [Fr(a[2]), Fr(a[3])]]
        M2 = [[Fr(b[0]), Fr(b[1])], [Fr(b[2]), Fr(b[3])]]
        v1 = float(vm_row(M1)[0])
        v2 = float(vm_row(transpose(M2))[0])
        m1, m2 = max(a), max(b)
        if m1 <= v1 + 1e-12 or m2 <= v2 + 1e-12:
            continue
        f = lambda M: (P * Q * float(M[0][0]) + P * (1 - Q) * float(M[0][1]) + (1 - P) * Q * float(M[1][0])
                       + (1 - P) * (1 - Q) * float(M[1][1]))
        g1 = (f(M1) - v1) / (m1 - v1)
        g2 = (f(M2) - v2) / (m2 - v2)
        ir = (g1 >= -1e-12) & (g2 >= -1e-12)
        # equal-gain curve: sign changes of g1 - g2 between grid neighbours (both directions), inside IR^m
        d = g1 - g2
        cm = (np.sign(d[:, :-1]) != np.sign(d[:, 1:])) | (d[:, :-1] == 0)
        cm2 = (np.sign(d[:-1, :]) != np.sign(d[1:, :])) | (d[:-1, :] == 0)
        eq_vals = []
        mm = np.minimum(g1, g2)
        sel = cm & ir[:, :-1] & ir[:, 1:]
        if sel.any():
            eq_vals.append(np.max(np.minimum(mm[:, :-1], mm[:, 1:])[sel]))
        sel2 = cm2 & ir[:-1, :] & ir[1:, :]
        if sel2.any():
            eq_vals.append(np.max(np.minimum(mm[:-1, :], mm[1:, :])[sel2]))
        if not eq_vals:
            no_eq += 1
            continue
        e_star = max(eq_vals)
        maxmin = np.max(np.where(ir, mm, -np.inf))
        if maxmin > e_star + 5e-3:
            po_fail += 1
            if len(examples) < 3:
                i, j = np.unravel_index(np.argmax(np.where(ir, mm, -np.inf)), mm.shape)
                examples.append((a, b, round(e_star, 4), round(maxmin, 4), (round(ps[i], 3), round(ps[j], 3)),
                                 round(float(g1[i, j]), 4), round(float(g2[i, j]), 4)))
    print("games: %d; no equal-gain profile in IR^m: %d; max-min of normalized gains above e* (+5e-3): %d" % (
        n_games, no_eq, po_fail))
    for ex in examples:
        print("  player 1 payoffs (a,b),(a,b'),(a',b),(a',b') =", ex[0], "player 2 =", ex[1],
              "e* ~", ex[2], "max-min ~", ex[3], "at", ex[4], "gains", (ex[5], ex[6]))
    print()


def _sections(Ma, Mb, xs, c_self, c_other):
    """For a grid xs of the committer's probability, the interval [lo, hi] of the answerer's probability y with
    Ma(x,y) >= c_self and Mb(x,y) >= c_other (Ma: committer's payoffs, Mb: answerer's, both as functions
    x*y*M00 + x*(1-y)*M01 + (1-x)*y*M10 + (1-x)*(1-y)*M11). Returns lo, hi, ok, and the committer's payoff
    at lo and hi."""
    lo = np.zeros_like(xs)
    hi = np.ones_like(xs)
    for M, c in ((Ma, c_self), (Mb, c_other)):
        a = xs * M[0][1] + (1 - xs) * M[1][1]
        b = xs * M[0][0] + (1 - xs) * M[1][0] - a
        with np.errstate(divide="ignore", invalid="ignore"):
            r = (c - a) / b
        lo = np.where(b > 1e-14, np.maximum(lo, r), lo)
        hi = np.where(b < -1e-14, np.minimum(hi, r), hi)
        flat = np.abs(b) <= 1e-14
        lo = np.where(flat & (a < c - 1e-12), 2.0, lo)
    ok = lo <= hi + 1e-12
    a = xs * Ma[0][1] + (1 - xs) * Ma[1][1]
    b = xs * Ma[0][0] + (1 - xs) * Ma[1][0] - a
    return lo, hi, ok, a + b * lo, a + b * hi


def hurwicz_run(M1, M2, alpha, n_grid=2001, cap=20000, tol=1e-10):
    """M1, M2: 2x2 payoff matrices (row player, column player), entries [row][col]. Returns (status, c, k)."""
    xs = np.linspace(0, 1, n_grid)
    T1 = [[M1[0][0], M1[1][0]], [M1[0][1], M1[1][1]]]   # transposes: the column player as committer
    T2 = [[M2[0][0], M2[1][0]], [M2[0][1], M2[1][1]]]
    c = np.array([-1e18, -1e18])
    for k in range(cap):
        lo, hi, ok, w_lo, w_hi = _sections(M1, M2, xs, c[0], c[1])
        if not ok.any():
            return "empty", c, k
        v1 = alpha * np.minimum(w_lo, w_hi) + (1 - alpha) * np.maximum(w_lo, w_hi)
        g1 = np.max(v1[ok])
        lo2, hi2, ok2, z_lo, z_hi = _sections(T2, T1, xs, c[1], c[0])
        v2 = alpha * np.minimum(z_lo, z_hi) + (1 - alpha) * np.maximum(z_lo, z_hi)
        g2 = np.max(v2[ok2]) if ok2.any() else c[1]
        g = np.maximum(c, np.array([g1, g2]))
        _, _, okn, _, _ = _sections(M1, M2, xs, g[0], g[1])
        if not okn.any():
            return ("overshoot0" if k == 0 else "overshoot"), c, k
        if np.max(g - c) < tol:
            return "converged", g, k
        c = g
    return "cap", c, k


def _payoff_grid(M, P, Q):
    return P * Q * M[0][0] + P * (1 - Q) * M[0][1] + (1 - P) * Q * M[1][0] + (1 - P) * (1 - Q) * M[1][1]


def thin_and_leximin(M1, M2, c, n=401):
    ps = np.linspace(0, 1, n)
    P, Q = np.meshgrid(ps, ps, indexing="ij")
    u1, u2 = _payoff_grid(M1, P, Q), _payoff_grid(M2, P, Q)
    inE = (u1 >= c[0] - 1e-9) & (u2 >= c[1] - 1e-9)
    spread = max(np.max(u1[inE] - c[0]), np.max(u2[inE] - c[1])) if inE.any() else np.nan
    v1 = float(vm_row([[Fr(x).limit_denominator(10**6) for x in r] for r in M1])[0])
    v2 = float(vm_row(transpose([[Fr(x).limit_denominator(10**6) for x in r] for r in M2]))[0])
    g1, g2 = u1 - v1, u2 - v2
    ir = (g1 >= -1e-12) & (g2 >= -1e-12)
    mm = np.where(ir, np.minimum(g1, g2), -np.inf)
    top = mm >= mm.max() - 1e-9
    mx = np.where(top, np.maximum(g1, g2), -np.inf)
    i, j = np.unravel_index(np.argmax(mx), mx.shape)
    lex = np.array([u1[i, j], u2[i, j]])
    return spread, lex, (v1, v2)


def part_H(seed=20261107):
    print("== Part H: Hurwicz answers (ex ante perfect prediction, numerical) ==")
    named = {
        "PD (10,3,1,0)": ([[3, 0], [10, 1]], [[3, 10], [0, 1]]),
        "PD (5,3,1,0)": ([[3, 0], [5, 1]], [[3, 5], [0, 1]]),
        "BoS (A,A)=0,(A,Y)=(2,3),(Y,A)=(3,2),(Y,Y)=1": ([[0, 2], [3, 1]], [[0, 3], [2, 1]]),
        "Leader (A,Y)=(3,2),(Y,A)=(2,3)": ([[0, 3], [2, 1]], [[0, 2], [3, 1]]),
        "Chicken (A,A)=0,(A,Y)=(3,1),(Y,A)=(1,3),(Y,Y)=2": ([[0, 3], [1, 2]], [[0, 1], [3, 2]]),
        "Matching pennies": ([[1, -1], [-1, 1]], [[-1, 1], [1, -1]]),
        "Stag hunt (4,4),(0,3),(3,0),(3,3)": ([[4, 0], [3, 3]], [[4, 3], [0, 3]]),
    }
    for alpha in (0.9, 0.99):
        print("alpha = %s" % alpha)
        for name, (M1, M2) in named.items():
            st, c, k = hurwicz_run(M1, M2, alpha)
            spread, lex, vm = thin_and_leximin(M1, M2, c)
            print("  %-48s %-10s rounds %5d  c = (%.4f, %.4f)  spread of E(c) %.4f  leximin point (%.4f, %.4f)"
                  "  v^m (%.3f, %.3f)" % (name, st, k, c[0], c[1], spread, lex[0], lex[1], vm[0], vm[1]))
    M1, M2 = named["BoS (A,A)=0,(A,Y)=(2,3),(Y,A)=(3,2),(Y,Y)=1"]
    for alpha in (0.9, 0.99):
        st, c, k = hurwicz_run(M1, M2, alpha)
        p = 3 / 8
        u = (_payoff_grid(M1, p, p), _payoff_grid(M2, p, p))
        print("BoS, alpha = %s: final state E(c), c = (%.6f, %.6f); (3/8, 3/8) pays (%.4f, %.4f), in E(c): %s, and is"
              " Pareto-dominated by (A,Y) paying (2, 3)" % (alpha, c[0], c[1], u[0], u[1],
                                                           u[0] >= c[0] and u[1] >= c[1]))
    rng = random.Random(seed)
    for alpha in (0.9, 0.99):
        stats = {"converged": 0, "overshoot0": 0, "overshoot": 0, "cap": 0, "empty": 0}
        thin, far = 0, []
        for _ in range(300):
            a = rng.sample(range(100), 4)
            b = rng.sample(range(100), 4)
            M1 = [[a[0], a[1]], [a[2], a[3]]]
            M2 = [[b[0], b[1]], [b[2], b[3]]]
            st, c, k = hurwicz_run(M1, M2, alpha, n_grid=1001, cap=5000)
            stats[st] += 1
            if st == "converged":
                spread, lex, vm = thin_and_leximin(M1, M2, c, n=201)
                if spread < 0.5:
                    thin += 1
                far.append(float(np.max(np.abs(c - lex))))
        print("random 2x2 without ties (300, payoffs 0..99), alpha = %s: %s; converged with E(c*) thin (spread < 0.5):"
              " %d; max |c* - leximin point| over converged: %.2f, median %.2f" % (
                  alpha, stats, thin, max(far) if far else float("nan"), float(np.median(far)) if far else float("nan")))
    for alpha in (0.9, 0.99):
        stats = {"converged": 0, "overshoot0": 0, "overshoot": 0, "cap": 0, "empty": 0}
        sym_ok, n_u = 0, 0
        while n_u < 100:
            vals = rng.sample(range(100), 4)
            A, B, C, D = vals
            if not (A > C and B > D and B > C):
                continue
            n_u += 1
            M1 = [[A, B], [C, D]]
            M2 = [[A, C], [B, D]]
            st, c, k = hurwicz_run(M1, M2, alpha, n_grid=1001, cap=5000)
            stats[st] += 1
            ps = np.linspace(0, 1, 100001)
            fstar = np.max(ps * ps * A + ps * (1 - ps) * (B + C) + (1 - ps) ** 2 * D)
            if st in ("converged", "cap") and abs(c[0] - fstar) < 1e-2 and abs(c[1] - fstar) < 1e-2:
                sym_ok += 1
        print("symmetric 2x2 in U* (100), alpha = %s: %s; final thresholds within 0.01 of (f*, f*): %d" % (
            alpha, stats, sym_ok))
    print()


if __name__ == "__main__":
    parts = sys.argv[1:] or list("ABCDEFGH")
    for name in parts:
        globals()["part_" + name]()
