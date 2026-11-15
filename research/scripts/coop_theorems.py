"""Checks for the paper on cooperation in the prisoner's and traveler's dilemmas (2026-10-30).

Logic.
1. Theorem (dominance): a concept that excludes a profile only through a profitable unilateral deviation, valued at
   no more than the better of "the others stay put" and "the others play strategies not strictly dominated in the
   game that remains", keeps every profile of strictly dominant strategies. Checked on the response models of the
   main paper (stay put, worst case, best replies) on random games with dominant strategies.
2. Theorem (local counterfactuals): two players; the answer of j to a deviation of i to t_i is a distribution over
   j's strategies that depends only on the continuation game G|t_i (both payoff functions at the profiles (t_i, .))
   and on j's status quo s_j, and is invariant to relabeling and to separate positive affine transformations of the
   two payoff functions. In a 2x2 continuation with strictly opposed preferences the rule is one number per type:
   type a = j's status quo is j's top (switching helps i), type b = j's status quo is j's bottom (switching hurts i).
   (a) We enumerate the pure rules and the set-valued rules with min or max aggregation and keep those that give
       (C,C) alone in the prisoner's dilemma (5,3,1,0) and in a family of prisoner's dilemmas.
   (b) For distributions: (C,C) alone in every PD forces switching with probability 1 in both types (shown on a
       family of PDs that approaches the limits).
   (c) The surviving rules leave the 2x2 zero-sum game Z (u_1 = [[4,3],[2,1]], strictly dominant strategies) with
       no solution; we also count, over all 576 strict ordinal 2x2 games, those left empty by "always switch".
   (d) For a single PD and distributions, a search over zero-sum 2x2 games finds an empty one for each pair of
       switching probabilities on a grid.
   (e) Rules that also consult the status quo profile: a reciprocity rule (reward a deviation that weakly raised
       the answerer's payoff, punish otherwise) is also empty on Z; the rule "answer so that both gain relative to
       the status quo when possible" is a pairwise joint deviation and keeps (M-1, M) in the traveler's dilemma.
3. Collective selections: Nash product and maximin of the gains over the pure security levels (pure maximin) on
   the dilemmas, chicken, battle of the sexes, matching pennies, Z, and random n-player asymmetric games (existence
   is immediate; uniqueness rates).
4. Joint deviations: alpha-core (a coalition blocks if it has a joint strategy whose worst case over the others'
   strategies gives every member strictly more), credible core for two players (non-members answer with best
   replies, worst one for the coalition), strong Nash equilibrium (others stay put). On the PD and the TD.
5. Berge equilibrium (pure): each player's payoff is maximized by the others' choices given her own strategy.
"""
import itertools
import numpy as np

rng = np.random.default_rng(20261030)


def profiles(U):
    return list(itertools.product(*(range(k) for k in U.shape[1:])))


def two(u1, u2):
    return np.array([u1, u2], float)


def pd(T, R, P, S):
    return two([[R, S], [T, P]], [[R, T], [S, P]])  # 0 = C, 1 = D


PD = pd(5, 3, 1, 0)
Z = two([[4, 3], [2, 1]], [[-4, -3], [-2, -1]])
CHICKEN = two([[0, 3], [1, 2]], [[0, 1], [3, 2]])
BOS = two([[3, 0], [1, 2]], [[2, 1], [0, 3]])
MP = two([[1, -1], [-1, 1]], [[-1, 1], [1, -1]])


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


def prof(i, t, y):
    return (t, y) if i == 0 else (y, t)


# ---------------------------------------------------------------- 1. dominance
def dominant_profile(U):
    n = U.shape[0]
    d = []
    for i in range(n):
        found = None
        for a in range(U.shape[1 + i]):
            ok = True
            for b in range(U.shape[1 + i]):
                if b == a:
                    continue
                for s in profiles(U):
                    if s[i] != a:
                        continue
                    t = list(s)
                    t[i] = b
                    if not U[(i,) + s] > U[(i,) + tuple(t)]:
                        ok = False
            if ok:
                found = a
        if found is None:
            return None
        d.append(found)
    return tuple(d)


def kept_by_models(U, s):
    """Two players: is s kept when deviations are valued at staying put, at the worst case, at the worst best reply?"""
    res = {}
    for model in ('stay', 'worst', 'bestreply'):
        ok = True
        for i in range(2):
            j = 1 - i
            for t in range(U.shape[1 + i]):
                if t == s[i]:
                    continue
                if model == 'stay':
                    v = U[(i,) + prof(i, t, s[j])]
                else:
                    ys = range(U.shape[1 + j])
                    if model == 'bestreply':
                        vj = [U[(j,) + prof(i, t, y)] for y in ys]
                        ys = [y for y in ys if vj[y] == max(vj)]
                    v = min(U[(i,) + prof(i, t, y)] for y in ys)
                if v > U[(i,) + s]:
                    ok = False
        res[model] = ok
    return res


print('Part 1. Dominant-strategy profiles under the response models (stay put, worst case, worst best reply)')
count = kept = 0
for _ in range(20000):
    shape = tuple(rng.integers(2, 5, size=2))
    U = rng.integers(0, 10, size=(2,) + shape).astype(float)
    d = dominant_profile(U)
    if d is None:
        continue
    count += 1
    kept += all(kept_by_models(U, d).values())
print(f'random two-player games with a strictly dominant-strategy profile: {count}; kept under all three models: {kept}')
print('PD (5,3,1,0): (D,D) kept:', kept_by_models(PD, (1, 1)))

# ---------------------------------------------------------------- 2. local counterfactuals
print()
print('Part 2. Local counterfactual rules on 2x2 continuations with strictly opposed preferences')


def continuation_type(U, i, t, sj):
    j = 1 - i
    other = 1 - sj
    a_sq, a_sw = U[(i,) + prof(i, t, sj)], U[(i,) + prof(i, t, other)]
    b_sq, b_sw = U[(j,) + prof(i, t, sj)], U[(j,) + prof(i, t, other)]
    if a_sw > a_sq and b_sq > b_sw:
        return 'a'  # status quo is j's top, switching helps i
    if a_sq > a_sw and b_sw > b_sq:
        return 'b'  # status quo is j's bottom, switching hurts i
    return None  # not strictly opposed


def solve_rule(U, rule, agg='dist'):
    """rule: dict type -> probability of switching (agg='dist'), or type -> set of answers {'stay','switch'} with
    agg in ('min','max'). Returns the profiles passing every test, or None if some continuation is not opposed."""
    out = []
    for s in profiles(U):
        ok = True
        for i in range(2):
            j = 1 - i
            t = 1 - s[i]
            typ = continuation_type(U, i, t, s[j])
            if typ is None:
                return None
            v_stay = U[(i,) + prof(i, t, s[j])]
            v_sw = U[(i,) + prof(i, t, 1 - s[j])]
            if agg == 'dist':
                v = (1 - rule[typ]) * v_stay + rule[typ] * v_sw
            else:
                vals = [v_stay if a == 'stay' else v_sw for a in rule[typ]]
                v = min(vals) if agg == 'min' else max(vals)
            if v > U[(i,) + s]:
                ok = False
        if ok:
            out.append(s)
    return out


pds = [pd(5, 3, 1, 0), pd(7, 3, 1, 0), pd(4, 3, 2, 0), pd(10, 3, 2.9, 0), pd(3.1, 3, 0.1, 0), pd(100, 3, 1, 0)]
CC = [(0, 0)]
print('pure rules (switch probability 0 or 1 per type) giving (C,C) alone:')
for qa, qb in itertools.product((0, 1), repeat=2):
    rule = {'a': qa, 'b': qb}
    good1 = solve_rule(PD, rule) == CC
    goodall = all(solve_rule(G, rule) == CC for G in pds)
    print(f'   switch in type a: {qa}, in type b: {qb}: (C,C) alone in PD (5,3,1,0): {good1}; in the family: {goodall};'
          f' Z solutions: {solve_rule(Z, rule)}')
subsets = [('stay',), ('switch',), ('stay', 'switch')]
for agg in ('min', 'max'):
    for Ra, Rb in itertools.product(subsets, repeat=2):
        rule = {'a': Ra, 'b': Rb}
        if all(solve_rule(G, rule, agg) == CC for G in pds):
            print(f'   set-valued, {agg} aggregation, type a {Ra}, type b {Rb}: (C,C) alone in the family; '
                  f'Z solutions: {solve_rule(Z, rule, agg)}')
print('distributions: largest switch probability in type a that fails some PD of the family, smallest in type b:')
grid = np.linspace(0, 1, 101)
ok_pairs = [(qa, qb) for qa in grid for qb in grid if all(solve_rule(G, {'a': qa, 'b': qb}) == CC for G in pds)]
print(f'   pairs on the grid giving (C,C) alone in the whole family: {[(round(a, 2), round(b, 2)) for a, b in ok_pairs]}')
ok1 = [(qa, qb) for qa in grid for qb in grid if solve_rule(PD, {'a': qa, 'b': qb}) == CC]
print(f'   in PD (5,3,1,0) alone: {len(ok1)} pairs; switch probability in type a ranges over '
      f'[{min(a for a, b in ok1):.2f}, {max(a for a, b in ok1):.2f}], in type b over '
      f'[{min(b for a, b in ok1):.2f}, {max(b for a, b in ok1):.2f}]')
# (d) for each pair allowed by the single PD, look for an empty zero-sum 2x2 game
emp = 0
for qa, qb in ok1:
    found = False
    for _ in range(3000):
        A = rng.permutation(np.concatenate([rng.uniform(-50, 50, 4)]))
        A = A.reshape(2, 2)
        G = two(A, -A)
        res = solve_rule(G, {'a': qa, 'b': qb})
        if res == []:
            found = True
            break
    emp += found
print(f'   zero-sum 2x2 game with no solution found (random search, 3000 tries) for {emp} of the {len(ok1)} pairs')

# all strict ordinal 2x2 games under "always switch"
cnt = empty = opposed_all = empty_opposed = 0
for p1 in itertools.permutations(range(1, 5)):
    for p2 in itertools.permutations(range(1, 5)):
        G = two(np.array(p1).reshape(2, 2), np.array(p2).reshape(2, 2))
        cnt += 1
        res = solve_rule(G, {'a': 1, 'b': 1})
        if res is None:
            continue
        opposed_all += 1
        empty_opposed += res == []
print(f'strict ordinal 2x2 games: {cnt}; with every continuation strictly opposed: {opposed_all}; of these, '
      f'empty under "always switch": {empty_opposed}')


# (e) rules that consult the status quo profile
def solve_status_quo_rule(U, kind):
    out = []
    for s in profiles(U):
        ok = True
        for i in range(2):
            j = 1 - i
            for t in range(U.shape[1 + i]):
                if t == s[i]:
                    continue
                ys = list(range(U.shape[1 + j]))
                ui = [U[(i,) + prof(i, t, y)] for y in ys]
                uj = [U[(j,) + prof(i, t, y)] for y in ys]
                if kind == 'reciprocity':
                    helped = U[(j,) + prof(i, t, s[j])] >= U[(j,) + s]
                    v = max(ui) if helped else min(ui)
                else:  # pairwise joint deviation: the best answer for i among those that leave j weakly better
                    good = [ui[y] for y in ys if uj[y] >= U[(j,) + s]]
                    v = max(good) if good else min(ui)
                if v > U[(i,) + s]:
                    ok = False
        if ok:
            out.append(s)
    return out


print('reciprocity rule on Z:', solve_status_quo_rule(Z, 'reciprocity'), '; on the PD:',
      solve_status_quo_rule(PD, 'reciprocity'))
for M, r in [(6, 2), (12, 2), (12, 5)]:
    U, claims = td(M, r)
    lab = lambda xs: [(claims[a], claims[b]) for a, b in xs]
    print(f'TD M={M} r={r}: reciprocity rule {lab(solve_status_quo_rule(U, "reciprocity"))}; '
          f'pairwise joint-deviation rule {lab(solve_status_quo_rule(U, "joint"))}')
print('pairwise joint-deviation rule on the PD:', solve_status_quo_rule(PD, 'joint'), '; on Z:',
      solve_status_quo_rule(Z, 'joint'))


# ---------------------------------------------------------------- 3. collective selections
def security(U):
    n = U.shape[0]
    out = []
    for i in range(n):
        best = -np.inf
        for t in range(U.shape[1 + i]):
            best = max(best, min(U[(i,) + s] for s in profiles(U) if s[i] == t))
        out.append(best)
    return out


def selections(U):
    n = U.shape[0]
    m = security(U)
    S = profiles(U)
    gains = {s: [U[(i,) + s] - m[i] for i in range(n)] for s in S}
    ir = [s for s in S if min(gains[s]) >= 0]
    mm = max(min(gains[s]) for s in ir)
    maximin = [s for s in ir if min(gains[s]) == mm]
    npv = max(np.prod(gains[s]) for s in ir)
    nash = [s for s in ir if np.prod(gains[s]) == npv]
    return m, maximin, nash


print()
print('Part 3. Collective selections over pure profiles with the security levels (pure maximin) as threat point')
for name, U in [('PD (5,3,1,0)', PD), ('chicken', CHICKEN), ('battle of the sexes', BOS), ('matching pennies', MP),
                ('Z', Z)]:
    m, mx, nb = selections(U)
    print(f'{name}: security {m}; maximin of gains {mx}; Nash product {nb}')
for M, r in [(6, 2), (12, 2), (12, 5), (30, 3)]:
    U, claims = td(M, r)
    m, mx, nb = selections(U)
    lab = lambda xs: [(claims[a], claims[b]) for a, b in xs]
    print(f'TD M={M} r={r}: security {m}; maximin of gains {lab(mx)}; Nash product {lab(nb)}')
for shape, N in [((3, 4), 2000), ((2, 3, 4), 1000), ((2, 2, 2, 3), 300)]:
    n = len(shape)
    um = un = 0
    for _ in range(N):
        U = rng.uniform(0, 1, size=(n,) + shape)
        m, mx, nb = selections(U)
        um += len(mx) == 1
        un += len(nb) == 1
    print(f'{n} players, asymmetric strategy sets {shape}, {N} games with continuous payoffs: always nonempty; '
          f'maximin of gains unique in {um}, Nash product unique in {un}')


# ---------------------------------------------------------------- 4. joint deviations
def blocked(U, s, model):
    n = U.shape[0]
    for size in range(1, n + 1):
        for C in itertools.combinations(range(n), size):
            others = [k for k in range(n) if k not in C]
            for tC in itertools.product(*(range(U.shape[1 + k]) for k in C)):
                if all(tC[x] == s[k] for x, k in enumerate(C)):
                    continue

                def full(y):
                    p = list(s)
                    for x, k in enumerate(C):
                        p[k] = tC[x]
                    for x, k in enumerate(others):
                        p[k] = y[x]
                    return tuple(p)
                if model == 'stay':
                    ys = [tuple(s[k] for k in others)]
                else:
                    ys = list(itertools.product(*(range(U.shape[1 + k]) for k in others)))
                    if model == 'bestreply' and others:
                        # two players only: the non-member best replies; worst best reply for the coalition
                        k = others[0]
                        vk = [U[(k,) + full(y)] for y in ys]
                        ys = [y for y, v in zip(ys, vk) if v == max(vk)]
                if all(min(U[(k,) + full(y)] for y in ys) > U[(k,) + s] for k in C):
                    return True
    return False


print()
print('Part 4. Joint deviations: alpha-core (worst case), credible core (best replies), strong Nash (stay put)')
for name, U, lab in [('PD (5,3,1,0)', PD, lambda s: ''.join('CD'[x] for x in s))] + \
        [(f'TD M={M} r={r}', td(M, r)[0], (lambda c: (lambda s: (c[s[0]], c[s[1]])))(td(M, r)[1]))
         for M, r in [(6, 2), (12, 2), (12, 5)]]:
    for model in ('worst', 'bestreply', 'stay'):
        sol = [lab(s) for s in profiles(U) if not blocked(U, s, model)]
        print(f'{name}, {model}: {sol if len(sol) <= 12 else str(len(sol)) + " profiles, top: " + str(sol[-6:])}')


# ---------------------------------------------------------------- 5. Berge
def berge(U):
    out = []
    for s in profiles(U):
        ok = True
        for i in range(2):
            j = 1 - i
            if max(U[(i,) + prof(i, s[i], y)] for y in range(U.shape[1 + j])) > U[(i,) + s]:
                ok = False
        if ok:
            out.append(s)
    return out


print()
print('Part 5. Berge equilibrium (pure)')
print('PD:', berge(PD), '; matching pennies:', berge(MP), '; chicken:', berge(CHICKEN), '; Z:', berge(Z))
for M, r in [(6, 2), (12, 5)]:
    U, claims = td(M, r)
    print(f'TD M={M} r={r}:', [(claims[a], claims[b]) for a, b in berge(U)])
