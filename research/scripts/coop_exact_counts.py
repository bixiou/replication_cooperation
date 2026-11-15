"""Exact counts and checks of claims of the paper on finite games. Pure profiles throughout.

Part 1: exact rates. The 576 strict 2x2 games are the pairs of strict rankings of the four profiles,
one ranking per player; the 65,536 games with ties give each player a payoff in {0,1,2,3} at each profile. Counted:
games whose PTE limit is nonempty; among the strict ones, games in which some order of sequential perfect prediction
misses a nonempty limit; among those with ties, nonempty limits that are not a single profile.
Part 2: without indifference, every nonempty self-consistent set is a single profile (all 15 nonempty
subsets of each of the 576 strict games).
Part 3: the unilateral column of Table 1, through Lemma 1 (U(G) <= phi(G) <= IR(G)). In the game G0
with payoffs (1,1) at (a,b), (2,1) at (a,b'), (0,0) at (a',b) and at (a',b'), U(G0) contains (a,b); we list which
selections exclude it. On random games, the pure alpha-core, the strong equilibrium and (two players) the credible
alpha-core lie between U(G) and IR(G).
Part 4: the random-label patch (draw an order of the profiles uniformly; predict the PTE limit if
nonempty, else the payoff class of the first Pareto-optimal individually rational profile) in chicken and G_e, and
the reachable sets and rest points of both games, the empty set included.
Part 5 (blocker B1): with W(own, other) = min(own, other), mutual cooperation and mutual defection are both equilibria
of every transformed prisoner's dilemma (random PDs).
"""
import itertools
import random
from fractions import Fraction as Fr

from coop_standalone import profiles, m_of, pte, ir, uset, selfcons, rand_game, CHICKEN

random.seed(20261103)
SH = (2, 2)


def verdict(U, E, i, n):
    m = m_of(U, E, n)[i]
    return [s for s in E if U[s][i] >= m]


def sequential(U, shape, order):
    n = len(shape)
    E = profiles(shape)
    while True:
        start = E
        for i in order:
            E = verdict(U, E, i, n)
        if E == start:
            return E


def pareto_optimal(U, P, s, n):
    return not any(all(U[x][i] >= U[s][i] for i in range(n)) and any(U[x][i] > U[s][i] for i in range(n))
                   for x in P)


# ---------- Part 1 ----------
def part1():
    print('Part 1. Exact rates on 2x2 games.')
    P = profiles(SH)
    strict = nonempty = miss = 0
    for r1 in itertools.permutations(range(4)):
        for r2 in itertools.permutations(range(4)):
            U = {s: (r1[k], r2[k]) for k, s in enumerate(P)}
            strict += 1
            L = pte(U, SH)
            if L:
                nonempty += 1
                if any(sorted(sequential(U, SH, o)) != sorted(L) for o in [(0, 1), (1, 0)]):
                    miss += 1
    print(f'  strict games: {strict}; PTE limit nonempty: {nonempty}; some SPP order misses the nonempty limit: {miss}')
    total = ne = multi = 0
    for vals in itertools.product(range(4), repeat=8):
        U = {s: (vals[2 * k], vals[2 * k + 1]) for k, s in enumerate(P)}
        total += 1
        L = pte(U, SH)
        if L:
            ne += 1
            if len(L) > 1:
                multi += 1
    print(f'  games with payoffs in 0..3: {total}; PTE limit nonempty: {ne}; nonempty and more than one profile: {multi}')


# ---------- Part 2 ----------
def part2():
    print('Part 2. Without indifference, nonempty self-consistent sets are single profiles.')
    P = profiles(SH)
    checked = bad = 0
    for r1 in itertools.permutations(range(4)):
        for r2 in itertools.permutations(range(4)):
            U = {s: (r1[k], r2[k]) for k, s in enumerate(P)}
            for k in range(1, 5):
                for E in itertools.combinations(P, k):
                    checked += 1
                    if selfcons(U, list(E), 2) and len(E) > 1:
                        bad += 1
    print(f'  (game, nonempty subset) pairs checked: {checked}; self-consistent with more than one profile: {bad}')


# ---------- Part 3 ----------
def security(U, shape):
    return m_of(U, profiles(shape), len(shape))


def nash_product(U, shape):
    v, I = security(U, shape), ir(U, shape)
    val = {s: 1 for s in I}
    for s in I:
        for i in range(len(shape)):
            val[s] *= U[s][i] - v[i]
    b = max(val.values())
    return [s for s in I if val[s] == b]


def maximin_gains(U, shape):
    v, I = security(U, shape), ir(U, shape)
    val = {s: min(U[s][i] - v[i] for i in range(len(shape))) for s in I}
    b = max(val.values())
    return [s for s in I if val[s] == b]


def max_sum(U, shape):
    P = profiles(shape)
    b = max(sum(U[s]) for s in P)
    return [s for s in P if sum(U[s]) == b]


def optimin(U, shape):
    """Pure version: u^min_i(s) is the least of u_i(s) and of i's payoffs after the profitable unilateral deviations
    of the other players; the optimin profiles have Pareto-undominated u^min vectors."""
    n, P = len(shape), profiles(shape)
    umin = {}
    for s in P:
        vec = []
        for i in range(n):
            w = U[s][i]
            for j in range(n):
                if j == i:
                    continue
                for t in range(shape[j]):
                    x = s[:j] + (t,) + s[j + 1:]
                    if U[x][j] > U[s][j]:
                        w = min(w, U[x][i])
            vec.append(w)
        umin[s] = tuple(vec)
    return [s for s in P if not any(all(umin[x][i] >= umin[s][i] for i in range(n)) and umin[x] != umin[s]
                                    for x in P)]


def bu(U, shape):
    """Rong and Halpern's BU_i for two players, pure: best payoff of i over her strategies and the other's best
    replies."""
    out = []
    for i in range(2):
        j = 1 - i
        best = None
        for t in range(shape[i]):
            rows = [s for s in profiles(shape) if s[i] == t]
            br = max(U[s][j] for s in rows)
            val = max(U[s][i] for s in rows if U[s][j] == br)
            best = val if best is None or val > best else best
        out.append(best)
    return out


def pce(U, shape):
    b = bu(U, shape)
    return [s for s in profiles(shape) if all(U[s][i] >= b[i] for i in range(2))]


def max_pce(U, shape):
    b, P = bu(U, shape), profiles(shape)
    val = {s: min(U[s][i] - b[i] for i in range(2)) for s in P}
    m = max(val.values())
    return [s for s in P if val[s] == m]


def mutual_advantage(U, shape):
    """Karpus and Radzvilas, eqs. (3)-(5), with the security levels as reference points, pure profiles."""
    n, P = len(shape), profiles(shape)
    v, I = security(U, shape), ir(U, shape)
    top = [max(U[s][i] for s in P) for i in range(n)]
    if any(top[i] == v[i] for i in range(n)):
        return []
    val = {s: min(Fr(U[s][i] - v[i], top[i] - v[i]) for i in range(n)) for s in I}
    b = max(val.values())
    return [s for s in I if val[s] == b]


def coalitions(n):
    for k in range(1, n + 1):
        yield from itertools.combinations(range(n), k)


def alpha_core(U, shape):
    n, P = len(shape), profiles(shape)
    out = []
    for s in P:
        blocked = False
        for C in coalitions(n):
            others = [j for j in range(n) if j not in C]
            for tC in itertools.product(*(range(shape[i]) for i in C)):
                if all(tC[k] == s[i] for k, i in enumerate(C)):
                    continue
                ok = True
                for y in itertools.product(*(range(shape[j]) for j in others)):
                    x = [None] * n
                    for k, i in enumerate(C):
                        x[i] = tC[k]
                    for k, j in enumerate(others):
                        x[j] = y[k]
                    if not all(U[tuple(x)][i] > U[s][i] for i in C):
                        ok = False
                        break
                if ok:
                    blocked = True
                    break
            if blocked:
                break
        if not blocked:
            out.append(s)
    return out


def strong_eq(U, shape):
    n, P = len(shape), profiles(shape)
    out = []
    for s in P:
        blocked = False
        for C in coalitions(n):
            for tC in itertools.product(*(range(shape[i]) for i in C)):
                x = list(s)
                for k, i in enumerate(C):
                    x[i] = tC[k]
                x = tuple(x)
                if x != s and all(U[x][i] > U[s][i] for i in C):
                    blocked = True
                    break
            if blocked:
                break
        if not blocked:
            out.append(s)
    return out


def credible_alpha_core(U, shape):
    """Two players: a single deviator is answered by every best reply of the other (blocked if she gains against all
    of them); the pair blocks if both gain."""
    P = profiles(shape)
    out = []
    for s in P:
        blocked = any(all(U[x][i] > U[s][i] for i in range(2)) for x in P if x != s)
        for i in range(2):
            j = 1 - i
            for t in range(shape[i]):
                if t == s[i] or blocked:
                    continue
                rows = [x for x in P if x[i] == t]
                br = max(U[x][j] for x in rows)
                if all(U[x][i] > U[s][i] for x in rows if U[x][j] == br):
                    blocked = True
        if not blocked:
            out.append(s)
    return out


def part3():
    print('Part 3. The unilateral column (Lemma 1).')
    G0 = {(0, 0): (1, 1), (0, 1): (2, 1), (1, 0): (0, 0), (1, 1): (0, 0)}
    print(f'  G0: U = {uset(G0, SH)}, IR = {ir(G0, SH)}, security levels {security(G0, SH)}, BU {bu(G0, SH)}')
    for name, f in [('Nash product', nash_product), ('maximin of gains', maximin_gains), ('sum of payoffs', max_sum),
                    ('optimin (pure)', optimin), ('PCE (pure)', pce), ('maximum PCE (pure)', max_pce),
                    ('mutual advantage (security reference)', mutual_advantage), ('alpha-core', alpha_core),
                    ('strong equilibrium', strong_eq), ('credible alpha-core', credible_alpha_core)]:
        pred = f(G0, SH)
        print(f'    {name}: {pred}; contains (a,b) = (0,0): {(0, 0) in pred}')
    for shape, concepts, N in [((2, 2), [alpha_core, strong_eq, credible_alpha_core], 2000),
                               ((3, 3), [alpha_core, strong_eq, credible_alpha_core], 1000),
                               ((2, 2, 2), [alpha_core, strong_eq], 500)]:
        for ties in (False, True):
            fails = {f.__name__: 0 for f in concepts}
            nonemptyU = 0
            for _ in range(N):
                U = rand_game(shape, ties)
                Us, I = set(uset(U, shape)), set(ir(U, shape))
                nonemptyU += bool(Us)
                for f in concepts:
                    pred = set(f(U, shape))
                    if not (Us <= pred <= I):
                        fails[f.__name__] += 1
            print(f'  {shape} {"ties" if ties else "rankings"}: {N} games ({nonemptyU} with U nonempty); '
                  f'failures of U <= phi <= IR: {fails}')


# ---------- Part 4 ----------
def reachable(U, shape):
    n = len(shape)
    start = tuple(sorted(profiles(shape)))
    seen, todo = {start}, [start]
    while todo:
        E = list(todo.pop())
        for B in coalitions(n):
            F = set(E)
            for i in B:
                F &= set(verdict(U, E, i, n))
            F = tuple(sorted(F))
            if F not in seen:
                seen.add(F)
                todo.append(F)
    rest = [E for E in seen if all(sorted(verdict(U, list(E), i, n)) == list(E) for i in range(n))]
    return seen, rest


def label_patch(U, shape, lam):
    n, P = len(shape), profiles(shape)
    L = pte(U, shape)
    if L:
        return sorted(L)
    I = ir(U, shape)
    s = next(x for x in lam if x in I and pareto_optimal(U, P, x, n))
    return sorted(x for x in P if U[x] == U[s])


def part4():
    print('Part 4. Random-label patch and reachable rest points.')
    Ge = {(0, 0): (Fr(0), Fr(0)), (0, 1): (Fr(3), Fr(1)), (1, 0): (Fr(1), Fr(3)), (1, 1): (Fr(1, 2), Fr(1, 2))}
    for name, U in [('chicken', CHICKEN), ('G_e, e = 1/2', Ge)]:
        seen, rest = reachable(U, SH)
        print(f'  {name}: reachable sets {sorted(seen, key=len)}; rest points {sorted(rest, key=len)}')
        counts = {}
        for lam in itertools.permutations(profiles(SH)):
            pred = tuple(label_patch(U, SH, lam))
            counts[pred] = counts.get(pred, 0) + 1
        print(f'    label patch over the 24 orders of profiles: {counts}')
        for pred in counts:
            print(f'    {list(pred)}: completely self-consistent '
                  f'{sorted(pred) == sorted(s for s in profiles(SH) if all(U[s][i] >= m_of(U, list(pred), 2)[i] for i in range(2)))}'
                  f'; reachable rest point {pred in rest}')


# ---------- Part 5 ----------
def part5():
    print('Part 5. W(own, other) = min(own, other) in transformed prisoner\'s dilemmas.')
    both = 0
    N = 2000
    for _ in range(N):
        T, R, P_, S = sorted(random.sample(range(-50, 51), 4), reverse=True)
        W = min
        cc = W(R, R) >= W(T, S)
        dd = W(P_, P_) >= W(S, T)
        both += cc and dd
    print(f'  {N} random PDs: (C,C) and (D,D) both equilibria in {both}')


if __name__ == '__main__':
    part1()
    part2()
    part3()
    part4()
    part5()
