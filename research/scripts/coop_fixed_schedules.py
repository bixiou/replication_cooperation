"""Census of oblivious schedules of Fourny's verdicts, two players (Online Appendix, Section S4).

Logic. A schedule is oblivious if the block of players applying their verdicts at step k depends on k only (not on
the game nor on the history). Oblivious schedules are eventually periodic sequences of blocks: a prefix and a cycle,
blocks among {1}, {2}, {1,2} (players 0 and 1 in the code). The process starts at S, applies E -> intersection of
V_i(E), i in the block, and stops at a rest point (every verdict leaves E unchanged) or at the empty set. A cycle that
misses a player can stall; such schedules are discarded. Schedules are deduplicated by their first 24 blocks, and a
schedule and its mirror (players exchanged) are kept as one concept (the order is drawn uniformly).
For each schedule we test, as a concept with the order (player 0 = player 1 of the paper first in the round-robin):
  EX  nonempty prediction on all 576 strict ordinal 2x2 games, all 6,561 2x2 games with payoffs in {0,1,2}, and
      400 random 3x3 games (rankings), seed 20261102;
  IR  prediction inside the individually rational profiles on the same games;
  PD  {(C,C)} in every PD with payoffs in 0..5 (each player's own T>R>P>S, all 15x15 combinations);
  TD  {(M,M)} in the traveler's dilemma, M in 2..10, r in {1/2, 1, 3/2, 2, 3, 5};
  PTE equal to the PTE limit whenever that limit is nonempty (same game classes);
and we record whether its outcomes coincide with sequential perfect prediction (SPP, round-robin) for the same order
on every test game (SPP-eq), or with resolved perfect prediction (RPP) for the same order (RPP-eq), and the number
of test games where it differs from SPP.
"""
import itertools
import os
import random
import sys
from fractions import Fraction as Fr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coop_standalone import profiles, m_of, pte, ir, td_game  # noqa: E402
from coop_sequential import sequential, jacobi_resolved  # noqa: E402

random.seed(20261102)
BLOCKS = [(0,), (1,), (0, 1)]
NAME = {(0,): '1', (1,): '2', (0, 1): 'S'}


def step(U, E, block, n):
    m = m_of(U, E, n)
    return [s for s in E if all(U[s][i] >= m[i] for i in block)]


def rest(U, E, n):
    m = m_of(U, E, n)
    return all(U[s][i] >= m[i] for s in E for i in range(n))


def run(U, shape, prefix, cycle, cap=500):
    n = len(shape)
    E = profiles(shape)
    k = 0
    while k < cap:
        if not E:
            return []
        if rest(U, E, n):
            return E
        b = prefix[k] if k < len(prefix) else cycle[(k - len(prefix)) % len(cycle)]
        E = step(U, E, b, n)
        k += 1
    raise RuntimeError('cap')


def games():
    G = []
    P = profiles((2, 2))
    for p1 in itertools.permutations(range(1, 5)):
        for p2 in itertools.permutations(range(1, 5)):
            G.append(({s: (p1[k], p2[k]) for k, s in enumerate(P)}, (2, 2), 'strict2x2'))
    for vals in itertools.product(range(3), repeat=8):
        G.append(({s: (vals[2 * k], vals[2 * k + 1]) for k, s in enumerate(P)}, (2, 2), 'ties2x2'))
    P3 = profiles((3, 3))
    for _ in range(400):
        perms = [random.sample(range(9), 9) for _ in range(2)]
        G.append(({s: (perms[0][k], perms[1][k]) for k, s in enumerate(P3)}, (3, 3), 'rand3x3'))
    return G


def pds():
    out = []
    for a in itertools.combinations(range(6), 4):
        for b in itertools.combinations(range(6), 4):
            S1, P1, R1, T1 = a
            S2, P2, R2, T2 = b
            out.append({(0, 0): (R1, R2), (0, 1): (S1, T2), (1, 0): (T1, S2), (1, 1): (P1, P2)})
    return out


def tds():
    return [(td_game(M, r), M) for M in range(2, 11) for r in [Fr(1, 2), Fr(1), Fr(3, 2), Fr(2), Fr(3), Fr(5)]]


def schedules(maxpre=2, maxcyc=4):
    seen = {}
    for lp in range(maxpre + 1):
        for pre in itertools.product(BLOCKS, repeat=lp):
            for lc in range(1, maxcyc + 1):
                for cyc in itertools.product(BLOCKS, repeat=lc):
                    if set().union(*map(set, cyc)) != {0, 1}:
                        continue
                    seq = [pre[k] if k < lp else cyc[(k - lp) % lc] for k in range(24)]
                    key = ''.join(NAME[b] for b in seq)
                    mirror = key.translate(str.maketrans('12', '21'))
                    if key in seen or mirror in seen:
                        continue
                    seen[key] = (pre, cyc)
    return seen


def main():
    G = games()
    PD = pds()
    TD = tds()
    print(f'test games: {len(G)} (576 strict 2x2, 6561 2x2 with payoffs 0..2, 400 random 3x3); {len(PD)} PDs; {len(TD)} TDs')
    ref_spp = [sequential(U, sh, (0, 1)) for U, sh, _ in G]
    ref_rpp = [jacobi_resolved(U, sh, (0, 1)) for U, sh, _ in G]
    ptes = [pte(U, sh) for U, sh, _ in G]
    irs = [set(ir(U, sh)) for U, sh, _ in G]
    S = schedules()
    print(f'distinct oblivious schedules up to mirror (prefix <= 2, cycle <= 4): {len(S)}')
    rows = []
    for key, (pre, cyc) in S.items():
        res = [run(U, sh, pre, cyc) for U, sh, _ in G]
        ex = all(res)
        exs = sum(1 for (U, sh, c), r in zip(G, res) if c == 'strict2x2' and r)
        irk = all(set(r) <= irs[k] for k, r in enumerate(res))
        pd = all(run(U, (2, 2), pre, cyc) == [(0, 0)] for U in PD)
        td = all(run(U, sh, pre, cyc) == [(M - 2, M - 2)] for (U, sh), M in TD)
        pteq = all(r == p for r, p in zip(res, ptes) if p)
        spp = all(sorted(r) == sorted(q) for r, q in zip(res, ref_spp))
        rpp = all(sorted(r) == sorted(q) for r, q in zip(res, ref_rpp))
        ndiff = sum(1 for r, q in zip(res, ref_spp) if sorted(r) != sorted(q))
        rows.append((key[:12], ex, exs, irk, pd, td, pteq, spp, rpp, ndiff))
    print('\nschedule (first 12 blocks; 1, 2 single players, S both) | EX | strict2x2 nonempty of 576 | IR | PD | TD | PTE-consistent | SPP-eq | RPP-eq | #games differing from SPP(1 first)')
    for r in sorted(rows, key=lambda r: (not (r[1] and r[3] and r[4] and r[5]), r[0])):
        print(' '.join(str(x) for x in r))
    good = [r for r in rows if r[1] and r[3] and r[4] and r[5]]
    print(f'\nschedules with EX, IR, PD and TD: {len(good)}')
    for r in good:
        print('  ', r[0], 'PTE-consistent' if r[6] else '', 'SPP-eq' if r[7] else '', 'RPP-eq' if r[8] else '', f'differs from SPP in {r[9]} games')
    print(f'schedules with EX: {sum(1 for r in rows if r[1])}; with PD and TD: {sum(1 for r in rows if r[4] and r[5])}; '
          f'with EX and PD and TD: {sum(1 for r in rows if r[1] and r[4] and r[5])}')


if __name__ == '__main__':
    main()
