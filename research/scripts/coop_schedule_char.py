"""coop_schedule_char.py: orders of reasoning by blocks and fixed sequences of verdicts (Online Appendix, Section S4).

Parts, all exact:
1. Weak orders of reasoning. For n = 2, 3, 4 and every ordered partition (B_1, ..., B_k) of the players, repeat the
   round V_{B_k} o ... o V_{B_1} (V_B(E) = intersection of the verdicts of B on E) until a round changes nothing.
   For each partition with a block holding two players i < j, embed G_e (e = 1/2) between i and j, every other player
   a dummy with two strategies and payoff 0, and record the final set (the note says: empty). For the partitions into
   singletons, record the final set in the same embedded games (the note says: nonempty, the paper's Proposition 6)
   and on 300 random games with payoffs in 0..3 of shapes (2,2), (2,2,2), (3,3) and (2,2,2,2) per partition.
2. No lapping. All sequences of length L over n players (n = 2: L <= 12; n = 3: L <= 10; n = 4: L <= 9) in which
   between any two consecutive occurrences of a player every other player occurs at least once, and in which, for
   the check of the converse, every player occurs; compared with the round-robin sequences pi, pi, pi, ... cut at L.
"""
import itertools
import random
from fractions import Fraction as Fr
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


def block(U, E, B, shape):
    out = frozenset(E)
    for i in B:
        out &= verdict(U, E, i, shape)
    return out


def run_round(U, shape, partition):
    E = frozenset(profiles(shape))
    for _ in range(10_000):
        old = E
        for B in partition:
            E = block(U, E, B, shape)
        if E == old:
            return E
    raise RuntimeError("no convergence")


def ordered_partitions(players):
    players = list(players)
    if not players:
        yield []
        return
    for r in range(1, len(players) + 1):
        for first in itertools.combinations(players, r):
            rest = [p for p in players if p not in first]
            for tail in ordered_partitions(rest):
                yield [first] + tail


GE = {(0, 0): (Fr(0), Fr(0)), (0, 1): (Fr(3), Fr(1)), (1, 0): (Fr(1), Fr(3)), (1, 1): (Fr(1, 2), Fr(1, 2))}


def embed_ge(n, i, j):
    shape = (2,) * n
    U = {}
    for s in profiles(shape):
        a, b = GE[(s[i], s[j])]
        U[s] = tuple(a if k == i else b if k == j else Fr(0) for k in range(n))
    return U, shape


def random_game(rng, shape):
    n = len(shape)
    return {s: tuple(rng.randint(0, 3) for _ in range(n)) for s in profiles(shape)}


say(__doc__.strip().splitlines()[0])
say()
say("== 1. Weak orders of reasoning")
rng = random.Random(20261103)
for n in (2, 3, 4):
    parts = list(ordered_partitions(range(n)))
    bad_nonsingleton = 0
    nonsingleton = 0
    singleton_ok = 0
    singleton = 0
    for P in parts:
        pairs = [(i, j) for B in P for i, j in itertools.combinations(sorted(B), 2)]
        if pairs:
            nonsingleton += 1
            i, j = pairs[0]
            U, shape = embed_ge(n, i, j)
            if run_round(U, shape, P):
                bad_nonsingleton += 1
        else:
            singleton += 1
            ok = True
            for i, j in itertools.combinations(range(n), 2):
                U, shape = embed_ge(n, i, j)
                ok &= bool(run_round(U, shape, P))
            singleton_ok += ok
    say(f"n = {n}: {len(parts)} ordered partitions; {nonsingleton} with a block of two or more players, of which "
        f"{nonsingleton - bad_nonsingleton} end empty on the embedded G_e; {singleton} into singletons, of which "
        f"{singleton_ok} are nonempty on every embedded G_e")

shapes = [(2, 2), (3, 3), (2, 2, 2), (2, 2, 2, 2)]
for shape in shapes:
    n = len(shape)
    orders = [list((p,) for p in perm) for perm in itertools.permutations(range(n))]
    empty = 0
    runs = 0
    for _ in range(300):
        U = random_game(rng, shape)
        for P in orders:
            runs += 1
            empty += not run_round(U, shape, P)
    say(f"singleton rounds (sequential perfect prediction) on 300 random games of shape {shape}, payoffs 0..3, all "
        f"orders: {runs} runs, {empty} empty")

say()
say("== 2. No lapping")


def no_lapping(seq, n):
    last = {}
    for pos, p in enumerate(seq):
        if p in last:
            between = set(seq[last[p] + 1:pos])
            if any(q not in between for q in range(n) if q != p):
                return False
        last[p] = pos
    return True


for n, Lmax in ((2, 12), (3, 10), (4, 9)):
    for L in range(n, Lmax + 1):
        rr = {tuple(perm[k % n] for k in range(L)) for perm in itertools.permutations(range(n))}
        ok = [s for s in itertools.product(range(n), repeat=L) if no_lapping(s, n) and set(s) == set(range(n))]
        extra = [s for s in ok if s not in rr]
        missing = [s for s in rr if s not in set(ok)]
        say(f"n = {n}, L = {L}: {len(ok)} sequences with no lapping in which every player occurs; round-robin "
            f"sequences {len(rr)}; not round-robin: {len(extra)}{' e.g. ' + str(extra[:3]) if extra else ''}; "
            f"round-robin failing the condition: {len(missing)}")
say("ABBA (0,1,1,0,0,1,1,0) has no lapping: " + str(no_lapping((0, 1, 1, 0, 0, 1, 1, 0), 2)))

OUT.write_text("\n".join(_lines) + "\n")


# ----------------------------------------------------------------------------------------------------------------------
# 3. A fully symmetric game in which the sequential predictions under the n! orders are pairwise distinct (used for
#    the converse: a game-independent law on orders whose lottery commutes with relabelings must be uniform).
#    Symmetric 3-player games with 3 strategies: u_i(s) = f(s_i, multiset of the others' strategies), f random in 0..9.
# ----------------------------------------------------------------------------------------------------------------------
def spp(U, shape, order):
    return run_round(U, shape, [(p,) for p in order])


say()
say("== 3. A symmetric 3-player game separating the six orders")
rng3 = random.Random(7)
shape = (3, 3, 3)
found = None
tries = 0
for tries in range(1, 200_001):
    keys = [(a, m) for a in range(3) for m in itertools.combinations_with_replacement(range(3), 2)]
    f = {k: rng3.randint(0, 9) for k in keys}
    U = {}
    for s in profiles(shape):
        U[s] = tuple(f[(s[i], tuple(sorted(s[j] for j in range(3) if j != i)))] for i in range(3))
    preds = {perm: spp(U, shape, perm) for perm in itertools.permutations(range(3))}
    if len(set(preds.values())) == 6:
        found = (f, preds)
        break
if found:
    f, preds = found
    say(f"found after {tries} random draws; f(own, others) = {dict(sorted(f.items()))}")
    for perm, E in preds.items():
        say(f"  order {tuple(p + 1 for p in perm)}: {sorted(tuple(x + 1 for x in s) for s in E)}")
else:
    say(f"none found in {tries} draws")

OUT.write_text("\n".join(_lines) + "\n")
