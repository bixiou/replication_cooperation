"""Standalone cooperative concepts, the link with the credible joint set J*, and a trilemma
(2026-10-30). Pure profiles; Fourny's guarantee m_i(E) = max over t_i with E(t_i) nonempty of min over E(t_i) of u_i.

Logic.
Part 1. PTE versus the main paper's credible joint set with complete knowledge and no opacity (two players).
   J*: L_i = max_t min over the other's best replies to t of u_i(t, y) (leader value); J* = {s : u(s) >= L}.
   Iterated J* (IJ): X_0 = S, X_{k+1} = {s in X_k : u(s) >= L(X_k)}, where L_i(X) uses as answers to t the
   other's best replies among the profiles of X(t) (credible answers restricted to the remaining set).
   The PTE does the same with the worst answer in X(t) instead of the best replies. Compare the limits.
Part 2. Trilemma. For a game, the symmetries are the relabelings of players and strategies that leave the payoffs
   unchanged; a concept with generality predicts a union of orbits. Enumerate all nonempty unions of orbits and
   keep the self-consistent ones (u_i(s) >= m_i(E) for all s in E). Claim: in the symmetric game
   G_e = AA (0,0), AY (3,1), YA (1,3), YY (e,e), 0 < e < 1, the only invariant self-consistent sets are {AA} and
   {YY}, both strictly Pareto-dominated by AY. Hence no concept has existence, generality, self-consistency and
   weak efficiency (some predicted profile not strictly Pareto-dominated). Also: frequency of such "forced
   inefficiency" among random symmetric games.
Part 3. Candidate standalone rules, evaluated on random games for existence, self-consistency, individual
   reasoning (U(G) <= phi(G) <= IR(G), Lemma 1), weak efficiency, and on the dilemmas:
   PTE limit; truncation; patchwork phi^P (PTE limit else iterated security set); PROP, proportional raising
   (thresholds c move toward m(E) by the largest common fraction lambda in [0,1] that some profile of E meets;
   E shrinks to the profiles meeting the new thresholds; stop when nothing changes).
Part 4. Pareto: a singleton PTE limit is never strictly Pareto-dominated (proof: a dominating profile would survive
   every round); check, and count set-valued limits that contain a dominated profile.
"""
import itertools
import random
from fractions import Fraction as Fr

random.seed(20261030)


def profiles(shape):
    return list(itertools.product(*(range(k) for k in shape)))


def m_of(U, E, n):
    out = []
    for i in range(n):
        best = None
        for t in sorted(set(s[i] for s in E)):
            w = min(U[s][i] for s in E if s[i] == t)
            best = w if best is None or w > best else best
        out.append(best)
    return out


def selfcons(U, E, n):
    if not E:
        return False
    mm = m_of(U, E, n)
    return all(U[s][i] >= mm[i] for s in E for i in range(n))


def pte_rounds(U, shape):
    n = len(shape)
    E = profiles(shape)
    rounds = [E]
    while True:
        mm = m_of(U, E, n)
        F = [s for s in E if all(U[s][i] >= mm[i] for i in range(n))]
        if F == E:
            return rounds, E
        if not F:
            return rounds, []
        E = F
        rounds.append(E)


def pte(U, shape):
    return pte_rounds(U, shape)[1]


def truncation(U, shape):
    rounds, lim = pte_rounds(U, shape)
    return lim if lim else rounds[-1]


def security_set(U, shape):
    n = len(shape)
    A = [list(range(k)) for k in shape]
    while True:
        newA = []
        for i in range(n):
            w = {t: min(U[s][i] for s in itertools.product(*A) if s[i] == t) for t in A[i]}
            b = max(w.values())
            newA.append([t for t in A[i] if w[t] == b])
        if newA == A:
            return list(itertools.product(*A))
        A = newA


def patch(U, shape):
    L = pte(U, shape)
    return L if L else security_set(U, shape)


def prop(U, shape):
    n = len(shape)
    E = profiles(shape)
    c = [min(U[s][i] for s in E) for i in range(n)]
    for _ in range(10000):
        mm = m_of(U, E, n)
        gap = [mm[i] - c[i] for i in range(n)]
        lam = Fr(0)
        for s in E:
            # largest lambda in [0,1] with u_i(s) >= c_i + lambda * gap_i for all i
            l = Fr(1)
            for i in range(n):
                if gap[i] > 0:
                    l = min(l, Fr(U[s][i] - c[i]) / gap[i])
            lam = max(lam, l)
        newc = [c[i] + lam * gap[i] for i in range(n)]
        F = [s for s in E if all(U[s][i] >= newc[i] for i in range(n))]
        if F == E and newc == c:
            return E
        E, c = F, newc
    raise RuntimeError('no convergence')


def ir(U, shape):
    n = len(shape)
    v = m_of(U, profiles(shape), n)
    return [s for s in profiles(shape) if all(U[s][i] >= v[i] for i in range(n))]


def uset(U, shape):
    n = len(shape)
    P = profiles(shape)
    best = [{t: max(U[s][i] for s in P if s[i] == t) for t in range(shape[i])} for i in range(n)]
    return [s for s in P if all(best[i][t] <= U[s][i] for i in range(n) for t in range(shape[i]) if t != s[i])]


def dominated(U, s, P, n):
    return any(all(U[x][i] > U[s][i] for i in range(n)) for x in P)


def efficient(U, E, shape):
    P = profiles(shape)
    return any(not dominated(U, s, P, len(shape)) for s in E)


# ---------- Part 1 ----------
def best_replies_in(U, X, i, t):
    j = 1 - i
    sec = [s for s in X if s[i] == t]
    if not sec:
        return []
    b = max(U[s][j] for s in sec)
    return [s for s in sec if U[s][j] == b]


def leader(U, X, i, shape):
    vals = []
    for t in range(shape[i]):
        br = best_replies_in(U, X, i, t)
        if br:
            vals.append(min(U[s][i] for s in br))
    return max(vals)


def jstar(U, shape):
    P = profiles(shape)
    L = [leader(U, P, i, shape) for i in range(2)]
    return [s for s in P if all(U[s][i] >= L[i] for i in range(2))], L


def ij(U, shape):
    X = profiles(shape)
    while True:
        L = [leader(U, X, i, shape) for i in range(2)]
        F = [s for s in X if all(U[s][i] >= L[i] for i in range(2))]
        if F == X:
            return X
        if not F:
            return []
        X = F


def rand_game(shape, ties):
    n = len(shape)
    P = profiles(shape)
    if ties:
        return {s: tuple(random.randint(0, 3) for _ in range(n)) for s in P}
    perms = [random.sample(range(len(P)), len(P)) for _ in range(n)]
    return {s: tuple(perms[i][k] for i in range(n)) for k, s in enumerate(P)}


def pd_game(T, R, P_, S):
    return {(0, 0): (R, R), (0, 1): (S, T), (1, 0): (T, S), (1, 1): (P_, P_)}


def td_game(M, r):
    U = {}
    for x in range(2, M + 1):
        for y in range(2, M + 1):
            U[(x - 2, y - 2)] = (x, x) if x == y else ((x + r, x - r) if x < y else (y - r, y + r))
    return U, (M - 1, M - 1)


CHICKEN = {(0, 0): (0, 0), (0, 1): (3, 1), (1, 0): (1, 3), (1, 1): (2, 2)}

# ---------- Part 2 ----------
def automorphisms(U, shape):
    n = len(shape)
    P = profiles(shape)
    auts = []
    for pi in itertools.permutations(range(n)):
        if any(shape[pi[i]] != shape[i] for i in range(n)):
            continue
        for sig in itertools.product(*(itertools.permutations(range(shape[i])) for i in range(n))):
            ok = True
            for s in P:
                t = [None] * n
                for i in range(n):
                    t[pi[i]] = sig[i][s[i]]
                t = tuple(t)
                if any(U[t][pi[i]] != U[s][i] for i in range(n)):
                    ok = False
                    break
            if ok:
                auts.append((pi, sig))
    return auts


def orbits(U, shape):
    n = len(shape)
    auts = automorphisms(U, shape)
    seen, orbs = set(), []
    for s in profiles(shape):
        if s in seen:
            continue
        orb = set()
        for pi, sig in auts:
            t = [None] * n
            for i in range(n):
                t[pi[i]] = sig[i][s[i]]
            orb.add(tuple(t))
        seen |= orb
        orbs.append(sorted(orb))
    return orbs, len(auts)


def invariant_sc_sets(U, shape):
    orbs, _ = orbits(U, shape)
    out = []
    for k in range(1, len(orbs) + 1):
        for comb in itertools.combinations(orbs, k):
            E = sorted(set().union(*map(set, comb)))
            if selfcons(U, E, len(shape)):
                out.append(E)
    return out



def main():
    print('Part 1. PTE limit versus the credible joint set J* (complete knowledge, no opacity) and iterated J*')
    for name, U, shape in [('PD (5,3,1,0)', pd_game(5, 3, 1, 0), (2, 2)), ('chicken', CHICKEN, (2, 2))] + \
            [(f'TD M={M} r={r}', *td_game(M, r)) for M, r in [(6, 2), (12, 2), (12, 5)]]:
        J, L = jstar(U, shape)
        print(f'   {name}: PTE {pte(U, shape)}; J* {J if len(J) <= 6 else str(len(J)) + " profiles"} (leader values {L});'
              f' iterated J* {ij(U, shape)}')
    for shape in [(2, 2), (3, 3), (2, 3), (4, 4)]:
        for ties in [False, True]:
            N = 2000
            same = sub = pte_in_j = nonempty_p = nonempty_ij = 0
            for _ in range(N):
                U = rand_game(shape, ties)
                p, q = pte(U, shape), ij(U, shape)
                J, _ = jstar(U, shape)
                same += (set(p) == set(q))
                nonempty_p += bool(p)
                nonempty_ij += bool(q)
                pte_in_j += set(p) <= set(J)
            print(f'   {shape}, {"payoffs 0..3" if ties else "random rankings"}, {N} games: PTE limit = iterated J* limit in'
                  f' {same}; PTE nonempty {nonempty_p}, iterated J* nonempty {nonempty_ij}; PTE limit inside J* in {pte_in_j}')


    print('\nPart 2. Trilemma: invariant self-consistent sets (0 = A aggressive, 1 = Y yield)')
    for e in [Fr(1, 10), Fr(1, 2), Fr(9, 10)]:
        G = {(0, 0): (0, 0), (0, 1): (3, 1), (1, 0): (1, 3), (1, 1): (e, e)}
        sets = invariant_sc_sets(G, (2, 2))
        orbs, na = orbits(G, (2, 2))
        print(f'   G_e with e = {e}: {na} symmetries, orbits {orbs}; invariant self-consistent sets {sets};'
              f' each contains an undominated profile: {[efficient(G, E, (2, 2)) for E in sets]};'
              f' patchwork predicts {patch(G, (2, 2))}, PTE limit {pte(G, (2, 2))}, truncation {truncation(G, (2, 2))}')
    print(f'   chicken: invariant self-consistent sets {invariant_sc_sets(CHICKEN, (2, 2))}')
    for shape in [(2, 2), (3, 3)]:
        for vals in [range(0, 4), range(0, 10)]:
            N = 3000 if shape == (2, 2) else 1500
            forced = forced_ir = pte_empty = 0
            for _ in range(N):
                P = profiles(shape)
                U1 = {s: random.choice(vals) for s in P}
                U = {(x, y): (U1[(x, y)], U1[(y, x)]) for (x, y) in P}
                sets = invariant_sc_sets(U, shape)
                I = set(ir(U, shape))
                assert sets, 'no invariant self-consistent set'
                if not any(efficient(U, E, shape) for E in sets):
                    forced += 1
                sets_ir = [E for E in sets if set(E) <= I]
                if sets_ir and not any(efficient(U, E, shape) for E in sets_ir):
                    forced_ir += 1
                pte_empty += not pte(U, shape)
            print(f'   random symmetric {shape} games, payoffs in {vals.start}..{vals.stop - 1}, {N} games: PTE limit empty in'
                  f' {pte_empty}; every invariant self-consistent set consists of strictly dominated profiles in {forced};'
                  f' same among individually rational sets in {forced_ir}')


    # ---------- Part 3 ----------
    print('\nPart 3. Candidate rules on random games (existence, self-consistency, U <= phi <= IR, weak efficiency)')
    rules = {'PTE': pte, 'truncation': truncation, 'patchwork': patch, 'PROP': prop}
    stats = {k: dict(exists=0, sc=0, unilateral=0, eff=0) for k in rules}
    N = 0
    for shape in [(2, 2), (3, 3), (2, 3), (2, 2, 2)]:
        for ties in [False, True]:
            for _ in range(400):
                U = rand_game(shape, ties)
                N += 1
                I, Us = set(ir(U, shape)), set(uset(U, shape))
                for k, f in rules.items():
                    E = f(U, shape)
                    st = stats[k]
                    st['exists'] += bool(E)
                    st['sc'] += selfcons(U, E, len(shape))
                    st['unilateral'] += bool(E) and Us <= set(E) <= I
                    st['eff'] += bool(E) and efficient(U, E, shape)
    print(f'   {N} games (2x2, 3x3, 2x3, 2x2x2; random rankings or payoffs 0..3):')
    for k, st in stats.items():
        print(f'   {k}: nonempty {st["exists"]}, self-consistent {st["sc"]}, U <= phi <= IR {st["unilateral"]},'
              f' contains an undominated profile {st["eff"]}')
    for k, f in rules.items():
        tds = [f(*td_game(M, r)) == [(M - 2, M - 2)] for M, r in [(6, 2), (12, 2), (12, 5), (9, Fr(3, 2))]]
        pds = [f(pd_game(T, R, P_, S), (2, 2)) == [(0, 0)] for T, R, P_, S in [(5, 3, 1, 0), (10, 3, 1, 0), (4, 3, 2, 1)]]
        print(f'   {k}: (C,C) alone in 3 PDs {pds}; (M,M) alone in 4 TDs {tds}; chicken {f(CHICKEN, (2, 2))}')

    # ---------- Part 4 ----------
    print('\nPart 4. Pareto efficiency of the PTE limit')
    single_dom = set_dom = singles = sets_ = 0
    for shape in [(2, 2), (3, 3), (2, 3), (2, 2, 2)]:
        for ties in [False, True]:
            for _ in range(500):
                U = rand_game(shape, ties)
                L = pte(U, shape)
                P = profiles(shape)
                if len(L) == 1:
                    singles += 1
                    single_dom += dominated(U, L[0], P, len(shape))
                elif len(L) > 1:
                    sets_ += 1
                    set_dom += any(dominated(U, s, P, len(shape)) for s in L)
    print(f'   singleton limits {singles}, strictly dominated {single_dom}; set-valued limits {sets_}, containing a strictly'
          f' dominated profile {set_dom}')


if __name__ == '__main__':
    main()
