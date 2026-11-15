"""Perfect prediction under a drawn order of reasoning. Parts 1 to 3 call "device" the
drawn order (vocabulary of the first version); "device necessary" means that no automorphism-invariant set is
self-consistent and weakly efficient.

Part 1. Necessity (Theorem A): in G_e, Leader and Battle of the Sexes, every set invariant under the game's
automorphisms that is self-consistent fails weak efficiency (exhaustive over the nonempty sets of pure profiles), and
sequential perfect prediction gives E under one order and its image under the player swap under the other.
Part 2. Device covariance: for random games and random relabelings of players and strategies, the realization under
the relabeled order in the relabeled game is the image of the realization (DP-general, checked numerically), for
sequential perfect prediction and resolved Jacobi.
Part 3. Where the device is used: share of games in which the realizations differ across orders, for both concepts,
against the share in which a device is necessary (no automorphism-invariant set is self-consistent and weakly
efficient), on all 576 strict ordinal 2x2 games, random symmetric 3x3 games and random 3x3 and 2x2x2 games.
Part 4. Ex ante: when every realization is a single profile, whether the expected payoff vector of the uniform
lottery is strictly Pareto-dominated by a pure profile (the lottery is efficient ex post; the analogue of random serial
dictatorship).
Part 5. Verdict processes (DR-V): from S, apply at each step the verdicts of a nonempty block of players (the
intersection of their verdicts), the block drawn at random, until every verdict leaves the set unchanged or the set is
empty. Checks: every set along the way is an upper contour set {u >= c} with c rising; every nonempty rest point is
completely self-consistent and contains U(G); individual rationality and emptiness counted. In G_e, Leader and BoS:
the only block that commutes with the player swap at a swap-invariant set is the whole set of players, and that path
(Fourny's) ends empty.
Part 6. Reachable rest points (exact): every set reachable from S by applying the verdicts of any nonempty blocks of
players, on the 576 strict ordinal 2x2 games. Counts the games where no nonempty reachable rest point is invariant
under the automorphisms (so a general concept that predicts reachable rest points predicts nothing), and compares with
the games that have a player-swapping automorphism and an empty PTE limit. Named games: chicken, G_e, Leader, BoS.
"""
import itertools
import random
from fractions import Fraction as Fr

from coop_standalone import profiles, m_of, selfcons, automorphisms, rand_game, CHICKEN
from coop_sequential import sequential, jacobi_resolved

random.seed(20261101)


def pareto_optimal(U, s, P, n):
    return not any(all(U[x][i] >= U[s][i] for i in range(n)) and any(U[x][i] > U[s][i] for i in range(n)) for x in P)


def act(a, s):
    pi, sig = a
    t = [None] * len(s)
    for i in range(len(s)):
        t[pi[i]] = sig[i][s[i]]
    return tuple(t)


def relabel(U, shape, a):
    pi, sig = a
    n = len(shape)
    shape2 = [None] * n
    for i in range(n):
        shape2[pi[i]] = shape[i]
    U2 = {}
    for s in profiles(shape):
        t = act(a, s)
        v = [None] * n
        for i in range(n):
            v[pi[i]] = U[s][i]
        U2[t] = tuple(v)
    return U2, tuple(shape2)


def device_necessary(U, shape, auts):
    """True if no nonempty set invariant under every automorphism is self-consistent and contains a Pareto-optimal
    profile. Invariant sets are unions of orbits."""
    n = len(shape)
    P = profiles(shape)
    seen, orbs = set(), []
    for s in P:
        if s in seen:
            continue
        o = {act(a, s) for a in auts}
        seen |= o
        orbs.append(sorted(o))
    po = {s for s in P if pareto_optimal(U, s, P, n)}
    for k in range(1, len(orbs) + 1):
        for comb in itertools.combinations(orbs, k):
            E = sorted(x for o in comb for x in o)
            if set(E) & po and selfcons(U, E, n):
                return False
    return True


def realizations(U, shape, concept):
    n = len(shape)
    return {order: tuple(sorted(concept(U, shape, list(order)))) for order in itertools.permutations(range(n))}


def part1():
    print('Part 1. Necessity: the trilemma witnesses (pure profiles; strategy 0 = A, 1 = Y)')
    games = {
        'G_e, e = 1/2': {(0, 0): (0, 0), (0, 1): (3, 1), (1, 0): (1, 3), (1, 1): (Fr(1, 2), Fr(1, 2))},
        'Leader AY (3,2)': {(0, 0): (0, 0), (0, 1): (3, 2), (1, 0): (2, 3), (1, 1): (1, 1)},
        'BoS AY (2,3)': {(0, 0): (0, 0), (0, 1): (2, 3), (1, 0): (3, 2), (1, 1): (1, 1)},
        'chicken (A = dare)': CHICKEN,
    }
    for name, U in games.items():
        shape = (2, 2)
        P = profiles(shape)
        auts = automorphisms(U, shape)
        inv = []
        for k in range(1, 5):
            for E in itertools.combinations(P, k):
                if all(sorted(act(a, s) for s in E) == list(E) for a in auts):
                    inv.append((list(E), selfcons(U, list(E), 2),
                                any(pareto_optimal(U, s, P, 2) for s in E)))
        spp = realizations(U, shape, sequential)
        rj = realizations(U, shape, jacobi_resolved)
        print(f'   {name}: |Aut| {len(auts)}; invariant sets (set, self-consistent, has a Pareto-optimal profile):'
              f' {inv}; device necessary: {device_necessary(U, shape, auts)}; SPP by order {spp}; RJ by order {rj}')


def part2():
    print('\nPart 2. Device covariance: realization in the relabeled game under the relabeled order = image')
    for shape, N in [((2, 2), 2000), ((3, 3), 1000), ((2, 3), 1000), ((2, 2, 2), 1000), ((3, 2, 2), 500)]:
        n = len(shape)
        bad = {'SPP': 0, 'RJ': 0}
        for _ in range(N):
            U = rand_game(shape, random.random() < 0.5)
            pi = list(range(n))
            random.shuffle(pi)
            sig = [tuple(random.sample(range(shape[i]), shape[i])) for i in range(n)]
            a = (tuple(pi), tuple(sig))
            U2, shape2 = relabel(U, shape, a)
            for name, concept in [('SPP', sequential), ('RJ', jacobi_resolved)]:
                for order in itertools.permutations(range(n)):
                    E = concept(U, shape, list(order))
                    E2 = concept(U2, shape2, [pi[k] for k in order])
                    if sorted(E2) != sorted(act(a, s) for s in E):
                        bad[name] += 1
        print(f'   {shape}: {N} games (half with payoff ties), random relabelings, every order: failures {bad}')


def census(games, shape, label):
    used = {'SPP': 0, 'RJ': 0}
    necessary, nec_used, sym = 0, {'SPP': 0, 'RJ': 0}, 0
    for U in games:
        auts = automorphisms(U, shape)
        sym += len(auts) > 1
        nec = len(auts) > 1 and device_necessary(U, shape, auts)
        necessary += nec
        for name, concept in [('SPP', sequential), ('RJ', jacobi_resolved)]:
            u = len(set(realizations(U, shape, concept).values())) > 1
            used[name] += u
            nec_used[name] += nec and u
    print(f'   {label}: {len(games)} games, {sym} with a nontrivial automorphism; device necessary in {necessary};'
          f' realizations differ across orders: SPP {used["SPP"]}, RJ {used["RJ"]}; among the necessary ones,'
          f' differ: SPP {nec_used["SPP"]}, RJ {nec_used["RJ"]}')


def part3():
    print('\nPart 3. Where the device is used, against where it is necessary')
    P = profiles((2, 2))
    perms = list(itertools.permutations(range(4)))
    census([{s: (p1[k], p2[k]) for k, s in enumerate(P)} for p1 in perms for p2 in perms], (2, 2),
           'strict ordinal 2x2 games (all)')
    P3 = profiles((3, 3))
    sym3 = []
    for _ in range(300):
        a = random.sample(range(9), 9)
        sym3.append({s: (a[3 * s[0] + s[1]], a[3 * s[1] + s[0]]) for s in P3})
    census(sym3, (3, 3), 'random symmetric 3x3 games with rankings')
    sym3t = []
    for _ in range(300):
        a = [random.randint(0, 3) for _ in range(9)]
        sym3t.append({s: (a[3 * s[0] + s[1]], a[3 * s[1] + s[0]]) for s in P3})
    census(sym3t, (3, 3), 'random symmetric 3x3 games with payoffs 0..3')
    census([rand_game((3, 3), False) for _ in range(300)], (3, 3), 'random 3x3 games with rankings')
    census([rand_game((2, 2, 2), False) for _ in range(300)], (2, 2, 2), 'random 2x2x2 games with rankings')


def part4():
    print('\nPart 4. Ex ante payoffs of the uniform lottery when every realization is a single profile')
    aumann = {(0, 0): (0, 0), (0, 1): (7, 2), (1, 0): (2, 7), (1, 1): (6, 6)}
    for name, U in [('chicken (3,1),(1,3),(2,2)', CHICKEN), ("Aumann's chicken (7,2),(2,7),(6,6)", aumann)]:
        R = realizations(U, (2, 2), sequential)
        ex = [sum(Fr(U[E[0]][i]) for E in R.values()) / len(R) for i in range(2)]
        print(f'   {name}: realizations {R}; ex ante payoffs {[str(x) for x in ex]}')
    for shape, N in [((2, 2), 2000), ((3, 3), 1000), ((2, 2, 2), 500)]:
        n = len(shape)
        P = profiles(shape)
        single, dom, differ = 0, 0, 0
        for _ in range(N):
            U = rand_game(shape, False)
            R = realizations(U, shape, sequential)
            if any(len(E) != 1 for E in R.values()):
                continue
            single += 1
            if len(set(R.values())) == 1:
                continue
            differ += 1
            ex = [sum(Fr(U[E[0]][i]) for E in R.values()) / len(R) for i in range(n)]
            dom += any(all(U[s][i] > ex[i] for i in range(n)) for s in P)
        print(f'   {shape} rankings, {N} games: every realization a single profile in {single}; of these, realizations'
              f' differ in {differ}, and the ex ante payoff vector is strictly dominated by a pure profile in {dom}')


def block_process(U, shape, rng, max_steps=10000):
    n = len(shape)
    P = profiles(shape)
    E = list(P)
    upper = True
    for _ in range(max_steps):
        if not E:
            return E, upper
        V = [set(s for s in E if U[s][i] >= m_of(U, E, n)[i]) for i in range(n)]
        if all(len(v) == len(E) for v in V):
            return E, upper
        k = rng.randint(1, n)
        B = rng.sample(range(n), k)
        F = sorted(set.intersection(*(V[i] for i in B)))
        m = m_of(U, F, n) if F else None
        if F and sorted(F) != sorted(s for s in P if all(U[s][i] >= c for i, c in enumerate(
                [min(U[x][i] for x in F) for i in range(n)]))):
            upper = False
        E = F
    raise RuntimeError('no rest point')


def part5():
    print('\nPart 5. Verdict processes with random blocks (DR-V)')
    from coop_standalone import ir, uset
    from coop_sequential import complete
    rng = random.Random(55)
    for shape, N in [((2, 2), 2000), ((3, 3), 1000), ((2, 2, 2), 1000), ((3, 3, 2), 300)]:
        for ties in [False, True]:
            runs, empty, notupper, notcomplete, u_out, not_ir = 0, 0, 0, 0, 0, 0
            for _ in range(N):
                U = rand_game(shape, ties)
                I, Us = set(ir(U, shape)), set(uset(U, shape))
                for _ in range(5):
                    E, upper = block_process(U, shape, rng)
                    runs += 1
                    notupper += not upper
                    if not E:
                        empty += 1
                        continue
                    notcomplete += not complete(U, E, shape)
                    u_out += not Us <= set(E)
                    not_ir += not set(E) <= I
            print(f'   {shape} {"payoffs 0..3" if ties else "rankings"}: {runs} runs (5 random schedules per game):'
                  f' set not an upper contour set at some step {notupper}; empty limit {empty}; nonempty limit not'
                  f' completely self-consistent {notcomplete}, missing U(G) {u_out}, not individually rational {not_ir}')
    for name, U in [('G_e, e = 1/2', {(0, 0): (0, 0), (0, 1): (3, 1), (1, 0): (1, 3), (1, 1): (Fr(1, 2), Fr(1, 2))}),
                    ('Leader', {(0, 0): (0, 0), (0, 1): (3, 2), (1, 0): (2, 3), (1, 1): (1, 1)}),
                    ('BoS', {(0, 0): (0, 0), (0, 1): (2, 3), (1, 0): (3, 2), (1, 1): (1, 1)})]:
        E, path = profiles((2, 2)), []
        while E:
            m = m_of(U, E, 2)
            F = [s for s in E if all(U[s][i] >= m[i] for i in range(2))]
            path.append((F, [str(x) for x in m]))
            if F == E:
                break
            E = F
        print(f'   {name}: symmetric schedule (both verdicts at every step), path (set, guarantees): {path}')


def reachable(U, shape):
    n = len(shape)
    start = tuple(profiles(shape))
    seen, stack = {start}, [start]
    while stack:
        E = stack.pop()
        if not E:
            continue
        m = m_of(U, list(E), n)
        V = [frozenset(s for s in E if U[s][i] >= m[i]) for i in range(n)]
        for k in range(1, n + 1):
            for B in itertools.combinations(range(n), k):
                F = tuple(sorted(frozenset.intersection(*(V[i] for i in B))))
                if F not in seen:
                    seen.add(F)
                    stack.append(F)
    rests = []
    for E in seen:
        if E and selfcons(U, list(E), n):
            rests.append(E)
    return seen, rests


def part6():
    print('\nPart 6. Reachable rest points of the verdicts (exact)')
    from coop_standalone import pte
    from coop_sequential import complete
    P = profiles((2, 2))
    perms = list(itertools.permutations(range(4)))
    games = [{s: (p1[k], p2[k]) for k, s in enumerate(P)} for p1 in perms for p2 in perms]
    no_inv, swap_empty, agree, notcomplete = 0, 0, 0, 0
    for U in games:
        auts = automorphisms(U, (2, 2))
        seen, rests = reachable(U, (2, 2))
        notcomplete += sum(1 for E in rests if not complete(U, list(E), (2, 2)))
        inv = [E for E in rests if all(tuple(sorted(act(a, s) for s in E)) == E for a in auts)]
        a = not inv
        b = any(a_[0] != (0, 1) for a_ in auts) and not pte(U, (2, 2))
        no_inv += a
        swap_empty += b
        agree += a == b
    print(f'   576 strict 2x2 games: no invariant nonempty reachable rest point in {no_inv}; player-swapping'
          f' automorphism and empty PTE limit in {swap_empty}; the two conditions agree in {agree} games;'
          f' reachable nonempty rest points not completely self-consistent: {notcomplete}')
    named = {
        'chicken': CHICKEN,
        'G_e, e = 1/2': {(0, 0): (0, 0), (0, 1): (3, 1), (1, 0): (1, 3), (1, 1): (Fr(1, 2), Fr(1, 2))},
        'Leader': {(0, 0): (0, 0), (0, 1): (3, 2), (1, 0): (2, 3), (1, 1): (1, 1)},
        'BoS': {(0, 0): (0, 0), (0, 1): (2, 3), (1, 0): (3, 2), (1, 1): (1, 1)},
    }
    for name, U in named.items():
        seen, rests = reachable(U, (2, 2))
        print(f'   {name}: reachable sets {sorted(seen, key=len)}; nonempty reachable rest points {sorted(rests)}')


if __name__ == '__main__':
    part1()
    part2()
    part3()
    part4()
    part5()
    part6()
