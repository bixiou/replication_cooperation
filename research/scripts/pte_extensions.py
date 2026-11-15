"""Extensions of Fourny's perfectly transparent equilibrium (PTE): ties, strict versus weak inequality, mixed and
correlated profiles, and two variants that always exist (2026-10-29).

Logic.
1. Pure PTE (Fourny 2020, Definitions 4 to 6): E_0 = S; m_i(E) = max over strategies t_i whose section
   E(t_i) = {s in E : s_i = t_i} is nonempty of min_{s in E(t_i)} u_i(s); E_{k+1} = {s in E_k : u_i(s) >= m_i(E_k)
   for every i} (weak inequality, as in Fourny's formula). The PTE is the limit when it is a singleton (no ties);
   with ties, we call the limit set the PTE set and say that it exists when it is nonempty.
   Strict variant: u_i(s) > m_i(E_k). Every nonempty E contains, for each i, a profile at which u_i = m_i(E), so the
   strict iteration loses at least one profile per round and ends empty: it never exists (checked here).
2. Two variants that always exist: (a) truncation, the last nonempty iterate; (b) fixed points of
   T(E) = {s in S : u_i(s) >= m*_i(E)}, where m*_i values a strategy with an empty section at its worst case over S
   (with Fourny's convention, which ignores such strategies, every singleton {s} with no weak Pareto improvement is a
   fixed point, including (C,D) in the prisoner's dilemma). We enumerate all fixed points on small games.
3. Mixed profiles (2x2 games): E is a set of product profiles (p, q); m_i(E) = sup_p inf_{q : (p,q) in E} u_1(p, q)
   for player 1, symmetrically for 2. Computed on a grid of step 1/N (numerical evidence; the paper gives the exact
   values for chicken and the battle of the sexes by hand).
4. Correlated profiles: player 1 committing to a pure strategy faces the distributions on her row that survive; the
   round-2 threshold in chicken is computed exactly (one linear program per pure strategy, here by hand in closed form).
5. Games: prisoner's dilemma (5,3,1,0); traveler's dilemma (claims 2..M, reward r); chicken as in Fourny (2020,
   Figure 24); battle of the sexes in general position; matching pennies; Z, the 2x2 zero-sum game with strictly
   dominant strategies used in the cooperation paper; random games with and without ties.
6. Credible joint set with complete knowledge and no opacity arc (main paper): J* = {s : u_i(s) >= L_i}, L_i the
   leader value; computed for chicken to compare with the PTE's round-2 thresholds.
"""
import itertools
import numpy as np

rng = np.random.default_rng(20261030)


def profiles(U):
    return list(itertools.product(*(range(k) for k in U.shape[1:])))


def m_values(U, E, impossible='ignore', strict_sections=None):
    """Thresholds m_i(E). impossible='ignore' (Fourny) or 'worst' (a strategy with an empty section is worth its
    worst payoff over S)."""
    n = U.shape[0]
    m = []
    for i in range(n):
        best = -np.inf
        for t in range(U.shape[1 + i]):
            sec = [s for s in E if s[i] == t]
            if sec:
                v = min(U[(i,) + s] for s in sec)
            elif impossible == 'worst':
                v = min(U[(i,) + s] for s in profiles(U) if s[i] == t)
            else:
                continue
            best = max(best, v)
        m.append(best)
    return m


def pte_path(U, strict=False):
    E = profiles(U)
    path = [E]
    while True:
        if not E:
            return path
        m = m_values(U, E)
        if strict:
            F = [s for s in E if all(U[(i,) + s] > m[i] for i in range(U.shape[0]))]
        else:
            F = [s for s in E if all(U[(i,) + s] >= m[i] for i in range(U.shape[0]))]
        if F == E:
            return path
        E = F
        path.append(E)


def pte(U, strict=False):
    return pte_path(U, strict)[-1]


def truncated(U):
    path = pte_path(U)
    nonempty = [E for E in path if E]
    return nonempty[-1]


def T_op(U, E, impossible):
    m = m_values(U, E, impossible)
    return [s for s in profiles(U) if all(U[(i,) + s] >= m[i] for i in range(U.shape[0]))]


def fixed_points(U, impossible, max_profiles=12):
    S = profiles(U)
    if len(S) > max_profiles:
        raise ValueError('too many profiles to enumerate all subsets')
    out = []
    for r in range(1, len(S) + 1):
        for E in itertools.combinations(S, r):
            if sorted(T_op(U, list(E), impossible)) == sorted(E):
                out.append(list(E))
    return out


def two(u1, u2):
    return np.array([u1, u2], float)


PD = two([[3, 0], [5, 1]], [[3, 5], [0, 1]])  # 0 = C, 1 = D
CHICKEN = two([[0, 3], [1, 2]], [[0, 1], [3, 2]])  # Fourny (2020, Figure 24): rows C, D; columns A, B
BOS = two([[3, 0], [1, 2]], [[2, 1], [0, 3]])  # battle of the sexes in general position: (B,B)=(3,2), (S,S)=(2,3)
BOS_TIES = two([[2, 0], [0, 1]], [[1, 0], [0, 2]])
MP = two([[1, -1], [-1, 1]], [[-1, 1], [1, -1]])
Z = two([[4, 3], [2, 1]], [[-4, -3], [-2, -1]])


def td(M, r):
    claims = list(range(2, M + 1))
    k = len(claims)
    U = np.zeros((2, k, k))
    for a, x in enumerate(claims):
        for b, y in enumerate(claims):
            if x == y:
                U[0, a, b] = U[1, a, b] = x
            elif x < y:
                U[0, a, b], U[1, a, b] = x + r, x - r
            else:
                U[0, a, b], U[1, a, b] = y - r, y + r
    return U, claims


def leader_values(U):
    """Two players, credible reactions with complete knowledge: L_i = max_t min over the follower's best replies."""
    L = []
    for i in range(2):
        j = 1 - i
        best = -np.inf
        for t in range(U.shape[1 + i]):
            def prof(y):
                return (t, y) if i == 0 else (y, t)
            vals_j = [U[(j,) + prof(y)] for y in range(U.shape[1 + j])]
            br = [y for y, v in enumerate(vals_j) if v == max(vals_j)]
            best = max(best, min(U[(i,) + prof(y)] for y in br))
        L.append(best)
    return L


def show(name, U, lab=None):
    lab = lab or (lambda s: s)
    path = pte_path(U)
    print(f'{name}:')
    print('   weak (Fourny) path sizes', [len(E) for E in path], '; limit', [lab(s) for s in path[-1]])
    sp = pte_path(U, strict=True)
    print('   strict path sizes', [len(E) for E in sp], '; limit', [lab(s) for s in sp[-1]])
    print('   truncation (last nonempty iterate)', [lab(s) for s in truncated(U)])
    if len(profiles(U)) <= 12:
        for imp in ('ignore', 'worst'):
            fps = fixed_points(U, imp)
            print(f'   fixed points of T, impossible strategies {imp}:', [[lab(s) for s in E] for E in fps])


print('Part 1. Pure profiles: Fourny weak, strict, truncation, fixed points')
show('Prisoner\'s dilemma (5,3,1,0), 0 = C', PD, lambda s: ''.join('CD'[x] for x in s))
show('Chicken (Fourny Figure 24), rows C/D, columns A/B', CHICKEN, lambda s: 'CD'[s[0]] + 'AB'[s[1]])
print('   chicken: round-2 thresholds', m_values(CHICKEN, pte_path(CHICKEN)[1]), '; leader values (credible, K complete)',
      leader_values(CHICKEN))
show('Battle of the sexes, general position', BOS)
print('   BoS: leader values', leader_values(BOS))
show('Battle of the sexes with ties (2,1),(0,0),(0,0),(1,2)', BOS_TIES)
show('Matching pennies', MP)
show('Z: zero-sum, strictly dominant strategies (row 0, column 1)', Z)
for M, r in [(6, 2), (12, 2), (12, 5), (20, 3)]:
    U, claims = td(M, r)
    path = pte_path(U)
    lim = [(claims[a], claims[b]) for a, b in path[-1]]
    tr = [(claims[a], claims[b]) for a, b in truncated(U)]
    # singleton fixed points of T with worst-case impossible strategies: {s} is fixed iff T({s}) = {s}
    single = [(claims[s[0]], claims[s[1]]) for s in profiles(U) if sorted(T_op(U, [s], 'worst')) == [s]]
    print(f'Traveler M={M} r={r}: weak path sizes {[len(E) for E in path]}; limit {lim}; truncation {tr}; '
          f'singleton fixed points (impossible = worst) {single}')

print()
print('Part 2. Existence on random games (payoffs i.i.d. uniform integers; "ties" = range 0..3, else distinct ranks)')
for shape, N in [((2, 2), 4000), ((3, 3), 4000), ((2, 2, 2), 2000), ((4, 4), 1000)]:
    n = len(shape)
    for ties in (False, True):
        ex_w = ex_s = sing = tr_ok = fp_any = 0
        for _ in range(N):
            if ties:
                U = rng.integers(0, 4, size=(n,) + shape).astype(float)
            else:
                U = np.stack([rng.permutation(int(np.prod(shape))).reshape(shape) for _ in range(n)]).astype(float)
            lim = pte(U)
            ex_w += bool(lim)
            sing += len(lim) == 1
            ex_s += bool(pte(U, strict=True))
            tr_ok += bool(truncated(U))
            if int(np.prod(shape)) <= 8:
                # a fixed point with worst-case impossible strategies always exists: some IR Pareto class
                fp_any += any(sorted(T_op(U, [s], 'worst')) == [s] for s in profiles(U)) or bool(
                    [1 for s in profiles(U) if sorted(T_op(U, [t for t in profiles(U) if (U[:, t[0], t[1]] if n == 2 else U[(slice(None),) + t]).tolist() == (U[:, s[0], s[1]] if n == 2 else U[(slice(None),) + s]).tolist()], 'worst')) == sorted([t for t in profiles(U) if (U[:, t[0], t[1]] if n == 2 else U[(slice(None),) + t]).tolist() == (U[:, s[0], s[1]] if n == 2 else U[(slice(None),) + s]).tolist()])])
        fp_txt = f'; a fixed point (IR Pareto class) found in {fp_any}' if int(np.prod(shape)) <= 8 else ''
        print(f'{n} players {shape}, {"ties" if ties else "no ties"}, {N} games: weak PTE set nonempty in {ex_w} '
              f'(singleton in {sing}); strict nonempty in {ex_s}; truncation nonempty in {tr_ok}{fp_txt}')

print()
print('Part 3. Mixed profiles on a grid (2x2), step 1/400, thresholds lowered by the grid tolerance')
print('Exact, prisoner\'s dilemma (5,3,1,0): u_1(p,q) = 1 - p + 4q - pq (p, q = probabilities of C). Round 1: m = (1,1).')
print('  The section of p in E_1 is q in [p/(4-p), 4p/(1+p)]; u_1 increases in q, so its infimum is u_1(p, p/(4-p)) = 1')
print('  for every p: m stays (1,1) and E_1 = {u_1 >= 1, u_2 >= 1} is the limit. It contains (D,D) and (C,C).')
for pp in [0, 0.25, 0.5, 1]:
    qq = pp / (4 - pp)
    print(f'  check p={pp}: q={qq:.4f}, u_1 = {1 - pp + 4*qq - pp*qq:.6f}, q <= 4p/(1+p): {qq <= 4*pp/(1+pp) + 1e-12}')


def mixed_path(U, N=400, rounds=30):
    grid = np.linspace(0, 1, N + 1)
    P, Q = np.meshgrid(grid, grid, indexing='ij')  # P = prob of row 0, Q = prob of column 0

    def u(i):
        A = U[i]
        return P * Q * A[0, 0] + P * (1 - Q) * A[0, 1] + (1 - P) * Q * A[1, 0] + (1 - P) * (1 - Q) * A[1, 1]
    u1, u2 = u(0), u(1)
    E = np.ones_like(P, dtype=bool)
    out = []
    for _ in range(rounds):
        if not E.any():
            out.append(('empty', None))
            break
        # player 1: sup over p (rows of the grid) of inf over q in the section
        rows = [u1[a][E[a]].min() for a in range(N + 1) if E[a].any()]
        cols = [u2[:, b][E[:, b]].min() for b in range(N + 1) if E[:, b].any()]
        m = (max(rows), max(cols))
        # tolerance of the grid: the infimum over a section is approximated within the payoff range times 2/N,
        # so thresholds are lowered by that amount (without it, grid rounding raises m every round)
        tol = 2 * (U.max() - U.min()) / N
        F = E & (u1 >= m[0] - tol) & (u2 >= m[1] - tol)
        out.append((round(m[0], 4), round(m[1], 4), int(F.sum())))
        if (F == E).all():
            break
        E = F
    return out


for name, U in [('Prisoner\'s dilemma', PD), ('Chicken', CHICKEN), ('Battle of the sexes, general position', BOS),
                ('Battle of the sexes with ties', BOS_TIES), ('Matching pennies', MP), ('Z', Z)]:
    print(f'{name}: rounds (m_1, m_2, surviving grid points)', mixed_path(U))

print()
print('Part 4. Correlated profiles, chicken, round 2: player 1 committing to C faces the distributions on row C '
      'with u_2 >= 1 (round-1 threshold), i.e. mu(CB) * 1 >= 1, so mu = CB and u_1 = 3 = her largest payoff; '
      'by symmetry m = (3,3), while u_1 + u_2 <= 4 at every distribution: no correlated profile survives.')
sums = CHICKEN[0] + CHICKEN[1]
print('   chicken: largest u_1 + u_2 over profiles', sums.max())
