"""coop_orders_and_census.py: computations for the paper: orders of reasoning, symmetric 3x3 games, public goods (2026-11-05).

All parts are exact enumerations except Part 6, which only recomputes intervals from recorded counts.
1. Resolved (RPP) and sequential (SPP) perfect prediction on G_e (e = 1/2) and chicken, both orders.
2. Uniform law for RPP: games with n = 3 and n = 4 players in which the n! RPP predictions are pairwise distinct
   (random search, fixed seeds). Any such game forces a game-independent law on orders whose lottery commutes with
   relabelings of the players to be uniform (the argument of the Online Appendix needs no symmetry of the game).
   Also checks the published SPP witness of coop_schedule_char.out under RPP.
3. Domain of equal treatment: the 362,880 symmetric 3x3 games in which player 1's payoffs are the numbers
   1 to 9. Counts the games where copying pays or fails; where it fails, whether the best symmetric profile is
   Pareto-dominated (then no Pareto-optimal concept can predict it), and whether the PTE limit is that profile.
4. The strict 2x2 games (pairs of strict rankings, 576): the 8 games with no invariant self-consistent set containing a
   Pareto-optimal profile, the 12 with no invariant nonempty reachable rest point, described by their ordinal type.
5. The weighted public goods game of the Online Appendix at a = 3/4: the rounds of the PTE, by hand-checkable steps.
6. Wilson 95% intervals for the sampled rates of the paper (counts and seeds from pte_extensions.out and
   coop_sequential.out).
"""
import itertools
import math
import random
from fractions import Fraction as Fr
from pathlib import Path

OUT = Path(__file__).with_suffix(".out")
_lines = []


def say(x=""):
    print(x)
    _lines.append(str(x))


# ---------------------------------------------------------------------------------------------------------------------
# Core: guarantees, verdicts, PTE, RPP, SPP. A game is (shape, U) with U[s] = tuple of payoffs, s a tuple of ints.
# ---------------------------------------------------------------------------------------------------------------------
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


def simultaneous(U, E, shape, players=None):
    players = range(len(shape)) if players is None else players
    out = frozenset(E)
    for i in players:
        out &= verdict(U, E, i, shape)
    return out


def pte(U, shape):
    E = frozenset(profiles(shape))
    while True:
        F = simultaneous(U, E, shape)
        if F == E:
            return E
        E = F


def rpp(U, shape, order):
    """Simultaneous verdicts while nonempty; when empty, one pass of single verdicts in the order; stop when fixed."""
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


def fmt(E, names=None):
    if names is None:
        return sorted(tuple(x + 1 for x in s) for s in E)
    return sorted("".join(names[x] for x in s) for s in E)


# ---------------------------------------------------------------------------------------------------------------------
say("== 1. G_e (e = 1/2) and chicken: strategy 0 = A (aggressive), 1 = Y (yield)")
Ge = {(0, 0): (0, 0), (0, 1): (3, 1), (1, 0): (1, 3), (1, 1): (Fr(1, 2), Fr(1, 2))}
Ch = {(0, 0): (0, 0), (0, 1): (3, 1), (1, 0): (1, 3), (1, 1): (2, 2)}
for name, U in (("G_e", Ge), ("chicken", Ch)):
    for order in ((0, 1), (1, 0)):
        say(f"{name}, order {tuple(p + 1 for p in order)}: RPP {fmt(rpp(U, (2, 2), order), 'AY')}, "
            f"SPP {fmt(spp(U, (2, 2), order), 'AY')}; PTE {fmt(pte(U, (2, 2)), 'AY')}")

# ---------------------------------------------------------------------------------------------------------------------
say()
say("== 2. Uniform law under RPP: games whose n! RPP predictions are pairwise distinct")
# The SPP witness of coop_schedule_char.out, Part 3 (fully symmetric, u_i(s) = f(s_i, sorted others)).
f_spp = {(0, (0, 0)): 1, (0, (0, 1)): 2, (0, (0, 2)): 4, (0, (1, 1)): 5, (0, (1, 2)): 9, (0, (2, 2)): 9,
         (1, (0, 0)): 5, (1, (0, 1)): 6, (1, (0, 2)): 8, (1, (1, 1)): 2, (1, (1, 2)): 3, (1, (2, 2)): 0,
         (2, (0, 0)): 7, (2, (0, 1)): 5, (2, (0, 2)): 1, (2, (1, 1)): 5, (2, (1, 2)): 7, (2, (2, 2)): 1}


def sym_game(f, n, k):
    shape = (k,) * n
    U = {s: tuple(f[(s[i], tuple(sorted(s[j] for j in range(n) if j != i)))] for i in range(n))
         for s in profiles(shape)}
    return shape, U


shape, U = sym_game(f_spp, 3, 3)
preds = {p: rpp(U, shape, p) for p in itertools.permutations(range(3))}
say(f"SPP witness of coop_schedule_char.out under RPP: {len(set(preds.values()))} distinct predictions over 6 orders; "
    f"PTE limit {fmt(pte(U, shape))}")


def search_symmetric(n, k, seed, tries, vmax):
    rng = random.Random(seed)
    keys = [(a, m) for a in range(k) for m in itertools.combinations_with_replacement(range(k), n - 1)]
    for t in range(1, tries + 1):
        f = {key: rng.randint(0, vmax) for key in keys}
        shape, U = sym_game(f, n, k)
        if pte(U, shape):
            continue
        orders = list(itertools.permutations(range(n)))
        seen = set()
        ok = True
        for p in orders:
            E = rpp(U, shape, p)
            if E in seen:
                ok = False
                break
            seen.add(E)
        if ok:
            return t, f, shape, U
    return None


def search_any(n, k, seed, tries):
    """Games with random strict rankings of the profiles for each player (no ties)."""
    rng = random.Random(seed)
    shape = (k,) * n
    P = profiles(shape)
    for t in range(1, tries + 1):
        ranks = [rng.sample(range(len(P)), len(P)) for _ in range(n)]
        U = {s: tuple(ranks[i][x] for i in range(n)) for x, s in enumerate(P)}
        if pte(U, shape):
            continue
        seen = set()
        ok = True
        for p in itertools.permutations(range(n)):
            E = rpp(U, shape, p)
            if E in seen:
                ok = False
                break
            seen.add(E)
        if ok:
            return t, shape, U
    return None


res = search_symmetric(3, 3, seed=11, tries=200_000, vmax=9)
if res:
    t, f, shape, U = res
    say(f"n = 3, fully symmetric, 3 strategies, payoffs in 0..9, seed 11: found at draw {t}")
    say(f"  f(own strategy, others' strategies sorted) = {dict(sorted(f.items()))}")
    say("  (strategies numbered 1 to 3 below)")
    for p in itertools.permutations(range(3)):
        say(f"  order {tuple(x + 1 for x in p)}: RPP {fmt(rpp(U, shape, p))}, SPP {fmt(spp(U, shape, p))}")
    say(f"  PTE limit: {fmt(pte(U, shape))}")
else:
    say("n = 3, fully symmetric: none found")

res4 = search_symmetric(4, 3, seed=12, tries=20_000, vmax=9)
if res4:
    t, f, shape, U = res4
    say(f"n = 4, fully symmetric, 3 strategies, payoffs in 0..9, seed 12: found at draw {t}")
    say(f"  f(own strategy, others' strategies sorted) = {dict(sorted(f.items()))}")
    for p in itertools.permutations(range(4)):
        say(f"  order {tuple(x + 1 for x in p)}: RPP {fmt(rpp(U, shape, p))}")
else:
    say("n = 4, fully symmetric with 3 strategies: none found in 20,000 draws (seed 12)")
    res4b = search_any(4, 3, seed=13, tries=20_000)
    if res4b:
        t, shape, U = res4b
        say(f"n = 4, strict random rankings, 3 strategies each, seed 13: found at draw {t}; 24 distinct RPP predictions")
        for p in itertools.permutations(range(4)):
            say(f"  order {tuple(x + 1 for x in p)}: RPP {fmt(rpp(U, shape, p))}")
    else:
        say("n = 4, strict random rankings, 3 strategies: none found in 20,000 draws (seed 13)")

# ---------------------------------------------------------------------------------------------------------------------
say()
say("== 3. Domain of equal treatment: symmetric 3x3 games, player 1's payoffs a permutation of 1..9")
n_all = n_cp = n_fail = n_fail_dom = n_fail_po = n_fail_po_pte = n_fail_dom_pte_empty = n_fail_po_pte_empty = 0
n_fail_pte_z = 0
shape = (3, 3)
P = profiles(shape)
for perm in itertools.permutations(range(1, 10)):
    A = {(x, y): perm[3 * x + y] for x in range(3) for y in range(3)}  # u_1(x, y)
    U = {(x, y): (A[(x, y)], A[(y, x)]) for (x, y) in P}
    n_all += 1
    cp = True
    for (x, y) in P:
        if x == y:
            continue
        u1, u2 = U[(x, y)]
        if u1 < u2 and not U[(y, y)][0] > u1:
            cp = False
            break
        if u2 < u1 and not U[(x, x)][1] > u2:
            cp = False
            break
    z = max(range(3), key=lambda x: A[(x, x)])
    c = A[(z, z)]
    dominated = any(U[s][0] > c and U[s][1] > c for s in P)  # no ties: weakly better for both means strictly
    lim = pte(U, shape)
    if cp:
        n_cp += 1
        continue
    n_fail += 1
    if lim == frozenset({(z, z)}):
        n_fail_pte_z += 1
    if dominated:
        n_fail_dom += 1
        if not lim:
            n_fail_dom_pte_empty += 1
    else:
        n_fail_po += 1
        if lim == frozenset({(z, z)}):
            n_fail_po_pte += 1
        elif not lim:
            n_fail_po_pte_empty += 1
say(f"games {n_all}; copying pays {n_cp}; copying fails {n_fail}")
say(f"copying fails: PTE limit = best symmetric profile in {n_fail_pte_z}")
say(f"copying fails, best symmetric profile Pareto-dominated: {n_fail_dom} (PTE limit empty in {n_fail_dom_pte_empty})")
say(f"copying fails, best symmetric profile Pareto-optimal: {n_fail_po} (PTE limit = it in {n_fail_po_pte}, "
    f"empty in {n_fail_po_pte_empty})")

# ---------------------------------------------------------------------------------------------------------------------
say()
say("== 4. Strict 2x2 games: the 8 and the 12")
shape = (2, 2)
P = profiles(shape)
relabelings = []
for swap in (False, True):
    for f1 in ((0, 1), (1, 0)):
        for f2 in ((0, 1), (1, 0)):
            relabelings.append((swap, f1, f2))


def apply(g, s):
    swap, f1, f2 = g
    a, b = f1[s[0]], f2[s[1]]
    return (b, a) if swap else (a, b)


def is_aut(g, U):
    swap = g[0]
    for s in P:
        t = apply(g, s)
        for i in range(2):
            j = 1 - i if swap else i
            if U[t][j] != U[s][i]:
                return False
    return True


def reachable(U):
    start = frozenset(P)
    seen = {start}
    stack = [start]
    while stack:
        E = stack.pop()
        for B in ((0,), (1,), (0, 1)):
            F = simultaneous(U, E, shape, B)
            if F not in seen:
                seen.add(F)
                stack.append(F)
    return seen


def selfcons(U, E):
    return all(U[s][i] >= guarantee(U, E, i, shape) for s in E for i in range(2))


def pareto_opt(U, s):
    return not any(all(U[t][i] >= U[s][i] for i in range(2)) and t != s for t in P)


def canon(U):
    """Canonical form up to relabelings of strategies and players (lexicographically smallest payoff table)."""
    forms = []
    for g in relabelings:
        V = {}
        for s in P:
            t = apply(g, s)
            V[t] = (U[s][1], U[s][0]) if g[0] else U[s]
        forms.append(tuple(V[s] for s in P))
    return min(forms)


eight, twelve = [], []
for r1 in itertools.permutations(range(1, 5)):
    for r2 in itertools.permutations(range(1, 5)):
        U = {s: (r1[x], r2[x]) for x, s in enumerate(P)}
        auts = [g for g in relabelings if is_aut(g, U)]
        subsets = [frozenset(c) for k in range(1, 5) for c in itertools.combinations(P, k)]
        inv = [E for E in subsets if all(frozenset(apply(g, s) for s in E) == E for g in auts)]
        if not any(selfcons(U, E) and any(pareto_opt(U, s) for s in E) for E in inv):
            eight.append(U)
        R = reachable(U)
        rest = [E for E in R if E and all(verdict(U, E, i, shape) == E for i in range(2))]
        if not any(all(frozenset(apply(g, s) for s in E) == E for g in auts) for E in rest):
            twelve.append(U)


def describe(Us, label):
    classes = {}
    for U in Us:
        classes.setdefault(canon(U), []).append(U)
    say(f"{label}: {len(Us)} games in {len(classes)} classes up to relabeling")
    for k, members in sorted(classes.items()):
        U = members[0]
        tab = "  ".join(f"{''.join('ab'[x] for x in s)}:{U[s]}" for s in P)
        pte_lim = fmt(pte(U, shape), "ab")
        swap_aut = any(g[0] and is_aut(g, U) for g in relabelings)
        say(f"  class of {len(members)}: {tab}; PTE {pte_lim}; exchange of players is an automorphism: {swap_aut}")


describe(eight, "no invariant self-consistent set with a Pareto-optimal profile")
describe(twelve, "no invariant nonempty reachable rest point")

# ---------------------------------------------------------------------------------------------------------------------
say()
say("== 5. Weighted public goods game, w = (1, 9/10, 4/5), a = 3/4: PTE rounds")
w = (Fr(1), Fr(9, 10), Fr(4, 5))
a = Fr(3, 4)
shape = (2, 2, 2)
P = profiles(shape)
U = {s: tuple(1 - s[i] * w[i] + a * sum(s[j] * w[j] for j in range(3)) for i in range(3)) for s in P}
for s in P:
    say(f"  profile {s}: payoffs {tuple(str(x) for x in U[s])}")
vals = [U[s][i] for s in P for i in range(3)]
say(f"  ties within a player's payoffs: {any(len(set(U[s][i] for s in P)) < len(P) for i in range(3))}")
E = frozenset(P)
k = 0
while True:
    m = tuple(guarantee(U, E, i, shape) for i in range(3))
    F = simultaneous(U, E, shape)
    say(f"  round {k + 1}: guarantees {tuple(str(x) for x in m)}; kept {sorted(F)}")
    if F == E or not F:
        break
    E = F
    k += 1

# ---------------------------------------------------------------------------------------------------------------------
say()
say("== 6. Wilson 95% intervals for the sampled rates")


def wilson(x, n, z=1.96):
    p = x / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


for label, x, n, src in (
        ("PTE nonempty, 3x3, strict rankings", 2723, 4000, "pte_extensions.py, seed 20261030"),
        ("PTE nonempty, 4x4, strict rankings", 620, 1000, "pte_extensions.py, seed 20261030"),
        ("PTE nonempty, 3x3, payoffs uniform in 0..3", 3709, 4000, "pte_extensions.py, seed 20261030"),
        ("PTE nonempty, 4x4, payoffs uniform in 0..3", 960, 1000, "pte_extensions.py, seed 20261030"),
        ("SPP misses the nonempty PTE limit for some order, 3x3", 97, 1354, "coop_sequential.py, seed 301030"),
        ("SPP misses the nonempty PTE limit for some order, 2x2x2", 193, 670, "coop_sequential.py, seed 301030")):
    lo, hi = wilson(x, n)
    say(f"{label}: {x}/{n} = {x / n:.3f}, 95% interval [{lo:.3f}, {hi:.3f}] ({src})")

OUT.write_text("\n".join(_lines) + "\n")
