"""coop_uniform_law_rank.py: a game that separates the n! orders, for every n, under RPP and SPP (2026-11-05).

The rank-claiming game: each of n players claims a rank 0, ..., n-1 or stays out (strategy n). Staying out pays 1.
A rank k claimed by nobody else pays v_k = n + 1 - k (rank 0 is the best); a rank claimed by two or more players pays
0 to each of them. Every permutation of the players maps the game to itself. Claim (Online Appendix, the uniform law):
under resolved perfect prediction, the player in position k of the order claims rank k-1; under sequential perfect
prediction, the last player claims rank 0 and the player in position k < n claims rank k. So the n! predictions are
pairwise different for both concepts, and the argument of the uniform law applies for every n. Checked for n = 2..5.
"""
import itertools
from pathlib import Path

OUT = Path(__file__).with_suffix(".out")
_lines = []


def say(x=""):
    print(x)
    _lines.append(str(x))


def profiles(shape):
    return list(itertools.product(*[range(k) for k in shape]))


def guarantee(U, E, i, shape):
    best = None
    for t in range(shape[i]):
        sec = [U[s][i] for s in E if s[i] == t]
        if sec:
            v = min(sec)
            best = v if best is None or v > best else best
    return best


def verdict(U, E, i, shape):
    if not E:
        return frozenset()
    m = guarantee(U, E, i, shape)
    return frozenset(s for s in E if U[s][i] >= m)


def simultaneous(U, E, shape):
    out = frozenset(E)
    for i in range(len(shape)):
        out &= verdict(U, E, i, shape)
    return out


def rpp(U, shape, order):
    E = frozenset(profiles(shape))
    while True:
        F = simultaneous(U, E, shape)
        if not F:
            F = E
            for i in order:
                F = verdict(U, F, i, shape)
        if F == E:
            return E
        E = F


def spp(U, shape, order):
    E = frozenset(profiles(shape))
    while True:
        F = E
        for i in order:
            F = verdict(U, F, i, shape)
        if F == E:
            return E
        E = F


def pte(U, shape):
    E = frozenset(profiles(shape))
    while True:
        F = simultaneous(U, E, shape)
        if F == E:
            return E
        E = F


def rank_game(n):
    shape = (n + 1,) * n
    U = {}
    for s in profiles(shape):
        pay = []
        for i in range(n):
            k = s[i]
            if k == n:
                pay.append(1)
            elif sum(1 for j in range(n) if s[j] == k) == 1:
                pay.append(n + 1 - k)
            else:
                pay.append(0)
        U[s] = tuple(pay)
    return shape, U


for n in range(2, 6):
    shape, U = rank_game(n)
    ok_r = ok_s = True
    pr, ps = set(), set()
    for order in itertools.permutations(range(n)):
        R = rpp(U, shape, order)
        S_ = spp(U, shape, order)
        want_r = [None] * n
        for pos, i in enumerate(order):
            want_r[i] = pos
        want_s = [None] * n
        for pos, i in enumerate(order):
            want_s[i] = 0 if pos == n - 1 else pos + 1
        ok_r &= R == frozenset({tuple(want_r)})
        ok_s &= S_ == frozenset({tuple(want_s)})
        pr.add(R)
        ps.add(S_)
    say(f"n = {n}: PTE limit {sorted(pte(U, shape))}; RPP gives rank k-1 to position k under every order: {ok_r}; "
        f"SPP gives rank 0 to the last and rank k to position k < n: {ok_s}; distinct predictions RPP {len(pr)}, "
        f"SPP {len(ps)}, of {len(list(itertools.permutations(range(n))))} orders")

OUT.write_text("\n".join(_lines) + "\n")
