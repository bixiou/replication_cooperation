"""Sequential perfect prediction: Fourny's
verdicts applied one player at a time, in an order of the players.

Definitions (pure profiles). Guarantee m_i(E) = max over t_i with E(t_i) nonempty of min over E(t_i) of u_i;
verdict V_i(E) = {s in E : u_i(s) >= m_i(E)}. The PTE iterates E -> intersection of the V_i(E) (all players at
once). For an order pi of the players, the sequential limit iterates passes E -> V_pi(n)(...V_pi(1)(E)) from S until
a whole pass changes nothing. Variant "exhaustive": each player applies her verdict until it changes nothing before
the next player moves. Claims to test:
 (1) every sequential limit is nonempty (a verdict keeps the section that attains the guarantee);
 (2) it is a fixed point of every verdict, hence self-consistent in Fourny's sense;
 (3) U(G) lies in it, and it is individually rational (Part 4 checks n = 3, 4);
 (4) PD and TD: {(C,C)} and {(M,M)} for every order;
 (5) conjecture: when the PTE limit is nonempty, every order reaches it.
Variant "resolved Jacobi": Fourny's simultaneous rounds, and a one-at-a-time pass in the order only when a
simultaneous round would empty the set; it equals the PTE limit whenever that is nonempty, by construction.
Also recorded: weak efficiency (some profile not strictly Pareto-dominated), the number of distinct limits over the
orders, and whether a limit is completely self-consistent (E = {s in S : u(s) >= m(E)}).
Games: dilemma grids (symmetric and asymmetric PDs, including 2R < T+S; TDs with M up to 40 and nine rewards),
named 2x2 games, and random games (rankings and payoffs in 0..3) of shapes 2x2, 3x3, 4x4, 2x2x2, 3x2x2.
"""
import itertools
import random
from fractions import Fraction as Fr

from coop_standalone import (profiles, m_of, pte, ir, uset, efficient, rand_game, pd_game, td_game, CHICKEN,
                             selfcons)

random.seed(301030)


def verdict(U, E, i, n):
    m = m_of(U, E, n)[i]
    return [s for s in E if U[s][i] >= m]


def sequential(U, shape, order, exhaustive=False):
    n = len(shape)
    E = profiles(shape)
    while True:
        start = E
        for i in order:
            if exhaustive:
                while True:
                    F = verdict(U, E, i, n)
                    if F == E:
                        break
                    E = F
            else:
                E = verdict(U, E, i, n)
        if E == start:
            return E


def jacobi_resolved(U, shape, order):
    """Fourny's simultaneous rounds; when a round would empty the set, apply the verdicts one at a time in the
    order instead (one pass), then go on."""
    n = len(shape)
    E = profiles(shape)
    while True:
        m = m_of(U, E, n)
        F = [s for s in E if all(U[s][i] >= m[i] for i in range(n))]
        if not F:
            F = E
            for i in order:
                F = verdict(U, F, i, n)
        if F == E:
            return E
        E = F


def complete(U, E, shape):
    n = len(shape)
    m = m_of(U, E, n)
    return sorted(E) == sorted(s for s in profiles(shape) if all(U[s][i] >= m[i] for i in range(n)))


def analyse(U, shape):
    n = len(shape)
    orders = list(itertools.permutations(range(n)))
    lims = {o: sequential(U, shape, o) for o in orders}
    exh = {o: sequential(U, shape, o, exhaustive=True) for o in orders}
    jr = {o: jacobi_resolved(U, shape, o) for o in orders}
    I, Us, L = set(ir(U, shape)), set(uset(U, shape)), pte(U, shape)
    rec = {
        'nonempty': all(lims[o] for o in orders),
        'selfcons': all(selfcons(U, lims[o], n) for o in orders),
        'fixed': all(all(verdict(U, lims[o], i, n) == lims[o] for i in range(n)) for o in orders),
        'ir': all(set(lims[o]) <= I for o in orders),
        'u_in': all(Us <= set(lims[o]) for o in orders),
        'eff': all(efficient(U, lims[o], shape) for o in orders),
        'complete': all(complete(U, lims[o], shape) for o in orders),
        'distinct': len(set(tuple(sorted(v)) for v in lims.values())),
        'exh_same': all(sorted(exh[o]) == sorted(lims[o]) for o in orders),
        'pte_nonempty': bool(L),
        'pte_reached': bool(L) and all(sorted(lims[o]) == sorted(L) for o in orders),
        'jr_nonempty': all(jr[o] for o in orders),
        'jr_selfcons': all(selfcons(U, jr[o], n) and complete(U, jr[o], shape) for o in orders),
        'jr_ir': all(set(jr[o]) <= I for o in orders),
        'jr_u_in': all(Us <= set(jr[o]) for o in orders),
        'jr_pte': (not L) or all(sorted(jr[o]) == sorted(L) for o in orders),
        'jr_distinct': len(set(tuple(sorted(v)) for v in jr.values())),
        'jr_same_as_seq': all(sorted(jr[o]) == sorted(lims[o]) for o in orders),
    }
    return rec, lims, L


def show(name, U, shape):
    rec, lims, L = analyse(U, shape)
    out = {o: sorted(v) for o, v in lims.items()}
    print(f'   {name}: PTE limit {sorted(L)}; sequential limits {out}; flags'
          f' {[k for k in ("nonempty", "selfcons", "ir", "u_in", "eff", "complete", "exh_same") if rec[k]]}')


def main():
    print('Part 1. Dilemmas')
    bad, cnt = [], 0
    pds = [(5, 3, 1, 0), (10, 3, 1, 0), (4, 3, 2, 0), (3, 2, 1, 0), (100, 3, 2, 1), (7, 3, 1, 0), (6, 5, 1, 0)]
    for T, R, P, S in pds:
        U = pd_game(T, R, P, S)
        for o in [(0, 1), (1, 0)]:
            for ex in [False, True]:
                cnt += 1
                if sequential(U, (2, 2), o, ex) != [(0, 0)]:
                    bad.append((T, R, P, S, o, ex))
    for _ in range(300):  # asymmetric PDs: each player's own T > R > P > S
        vals = [sorted(random.sample(range(-20, 40), 4), reverse=True) for _ in range(2)]
        (T1, R1, P1, S1), (T2, R2, P2, S2) = vals
        U = {(0, 0): (R1, R2), (0, 1): (S1, T2), (1, 0): (T1, S2), (1, 1): (P1, P2)}
        for o in [(0, 1), (1, 0)]:
            for ex in [False, True]:
                cnt += 1
                if sequential(U, (2, 2), o, ex) != [(0, 0)]:
                    bad.append((vals, o, ex))
    rr = [b for b in bad if b[-1] is False]
    print(f'   PDs: {len(pds)} symmetric ({sum(2 * R < T + S for T, R, P, S in pds)} with 2R < T+S) and 300 asymmetric, both orders, round-robin and'
          f' exhaustive ({cnt} runs): limit other than {{(C,C)}} in {len(bad)} runs, of which round-robin {len(rr)};'
          f' first failures {bad[:2]}')
    U = pd_game(5, 3, 1, 0)
    print(f'   PD (5,3,1,0), exhaustive schedule, order (1,2): limit {sequential(U, (2, 2), (0, 1), True)}'
          f' (player 1 raises her threshold to P, R, then T before player 2 moves)')
    bad, cnt = [], 0
    for M in range(2, 41):
        for r in [Fr(1, 4), Fr(1, 2), 1, Fr(3, 2), 2, Fr(7, 3), Fr(5, 2), 3, 5]:
            U, shape = td_game(M, r)
            top = (M - 2, M - 2)
            for o in [(0, 1), (1, 0)]:
                for ex in [False, True]:
                    cnt += 1
                    if sequential(U, shape, o, ex) != [top]:
                        bad.append((M, r, o, ex, sequential(U, shape, o, ex)))
    jr_bad = []
    for T, R, P, S in pds:
        for o in [(0, 1), (1, 0)]:
            if jacobi_resolved(pd_game(T, R, P, S), (2, 2), o) != [(0, 0)]:
                jr_bad.append((T, R, P, S, o))
    for M in range(2, 41):
        for r in [Fr(1, 4), Fr(1, 2), 1, Fr(3, 2), 2, Fr(7, 3), Fr(5, 2), 3, 5]:
            U, shape = td_game(M, r)
            for o in [(0, 1), (1, 0)]:
                if jacobi_resolved(U, shape, o) != [(M - 2, M - 2)]:
                    jr_bad.append((M, r, o))
    print(f'   resolved Jacobi (Fourny rounds, one-at-a-time pass only when a round would empty the set): failures'
          f' on the symmetric PDs and the TDs {jr_bad}')
    rr = [b for b in bad if b[3] is False]
    print(f'   TDs: M = 2..40, r in {{1/4, 1/2, 1, 3/2, 2, 7/3, 5/2, 3, 5}}, both orders, both schedules ({cnt} runs):'
          f' limit other than {{(M,M)}} in {len(bad)} runs, of which round-robin {len(rr)}; first failures {bad[:2]}')

    print('\nPart 2. Named games (profiles as (row, column), strategy 0 first)')
    show('chicken (A first)', CHICKEN, (2, 2))
    show('G_e, e = 1/2 (A first)', {(0, 0): (0, 0), (0, 1): (3, 1), (1, 0): (1, 3), (1, 1): (Fr(1, 2), Fr(1, 2))}, (2, 2))
    show('Leader, AY (3,2), YA (2,3)', {(0, 0): (0, 0), (0, 1): (3, 2), (1, 0): (2, 3), (1, 1): (1, 1)}, (2, 2))
    show('Battle of the sexes, AY (2,3), YA (3,2)', {(0, 0): (0, 0), (0, 1): (2, 3), (1, 0): (3, 2), (1, 1): (1, 1)}, (2, 2))
    show('matching pennies', {(0, 0): (1, -1), (0, 1): (-1, 1), (1, 0): (-1, 1), (1, 1): (1, -1)}, (2, 2))
    R, P, S = 3, 1, 0
    show('Z_PD (.,3,1,0)', {(0, 0): (R, -R), (0, 1): (R - P + S, -(R - P + S)), (1, 0): (S, -S),
                            (1, 1): (S - 1, 1 - S)}, (2, 2))
    show('stag hunt (4,0,3,3)', {(0, 0): (4, 4), (0, 1): (0, 3), (1, 0): (3, 0), (1, 1): (3, 3)}, (2, 2))

    print('\nPart 3. Random games')
    keys = ['nonempty', 'selfcons', 'fixed', 'ir', 'u_in', 'eff', 'complete', 'exh_same', 'jr_nonempty',
            'jr_selfcons', 'jr_ir', 'jr_u_in', 'jr_pte', 'jr_same_as_seq']
    for shape, N in [((2, 2), 2000), ((3, 3), 2000), ((4, 4), 600), ((2, 2, 2), 1000), ((3, 2, 2), 500)]:
        for ties in [False, True]:
            agg = {k: 0 for k in keys}
            dist, jdist = {}, {}
            pte_ne = pte_ok = pte_e = 0
            ex_fail = None
            for _ in range(N):
                U = rand_game(shape, ties)
                rec, lims, L = analyse(U, shape)
                for k in keys:
                    agg[k] += rec[k]
                dist[rec['distinct']] = dist.get(rec['distinct'], 0) + 1
                jdist[rec['jr_distinct']] = jdist.get(rec['jr_distinct'], 0) + 1
                if rec['pte_nonempty']:
                    pte_ne += 1
                    pte_ok += rec['pte_reached']
                    if not rec['pte_reached'] and ex_fail is None:
                        ex_fail = (U, sorted(L), {o: sorted(v) for o, v in lims.items()})
                else:
                    pte_e += 1
            print(f'   {shape} {"payoffs 0..3" if ties else "rankings"}, {N} games: ' +
                  ', '.join(f'{k} {agg[k]}' for k in keys) +
                  f'; distinct limits over orders {dict(sorted(dist.items()))}; PTE nonempty {pte_ne}, reached by'
                  f' every order in {pte_ok}; PTE empty {pte_e}; resolved-Jacobi distinct limits'
                  f' {dict(sorted(jdist.items()))}')
            if ex_fail is not None:
                print(f'      first game where some order misses a nonempty PTE limit: {ex_fail}')


def part4():
    print('\nPart 4. Individual rationality with three and four players (round-robin, every order)')
    for shape, N in [((2, 2, 2), 20000), ((3, 3, 3), 3000), ((2, 2, 2, 2), 3000), ((3, 3, 2), 5000)]:
        for ties in [False, True]:
            viol, ex = 0, None
            for _ in range(N):
                U = rand_game(shape, ties)
                I = set(ir(U, shape))
                for o in itertools.permutations(range(len(shape))):
                    L = sequential(U, shape, o)
                    if not set(L) <= I:
                        viol += 1
                        ex = ex or (U, o, L)
                        break
            print(f'   {shape} {"payoffs 0..3" if ties else "rankings"}, {N} games: games with an order whose limit'
                  f' is not individually rational: {viol}{"; example " + str(ex) if ex else ""}')


if __name__ == '__main__':
    main()
    part4()
