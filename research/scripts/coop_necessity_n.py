"""coop_necessity_n.py: the volunteer's dilemma with n players and the drawn order of reasoning.

Parts, all exact (brute force, no sampling):
1. Volunteer's dilemma VD_n(b, c), n = 2..6: every reachable set (breadth-first over every nonempty block of
   players), compared with the family predicted by the note (S and F_D for every D); rest points; automorphism
   group (player permutations x strategy bijections); invariant reachable sets; invariant nonempty reachable rest
   points; PTE limit; sequential (SPP) and resolved (RPP) perfect prediction under every order.
2. The same with player-specific costs c_i in (0, b): who volunteers under SPP and RPP.
3. Schedule form: a deterministic schedule commuting with relabelings, in a game whose automorphism group is
   transitive on the players, must choose the block N at every step whose history is invariant; checked by listing,
   along the PTE path of each witness, the blocks invariant under the stabilizer of the history.
4. The separating 3x3 game (u_2(x,y) = u_1(y,x), rows of u_1 (2,0,8), (6,1,4), (3,5,7)): automorphisms, PTE path,
   SPP path with the guarantees at each verdict, invariant nonempty reachable rest points.
Strategies are 0..k-1; in the volunteer's dilemma 0 = V (volunteer), 1 = N (not).
"""
import itertools
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
    """m_i(E): max over strategies with a nonempty section of the min of u_i over the section."""
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


def step(U, E, B, shape):
    out = frozenset(E)
    for i in B:
        out &= verdict(U, E, i, shape)
    return out


def is_rest(U, E, shape):
    return all(verdict(U, E, i, shape) == E for i in range(len(shape)))


def reachable(U, shape):
    n = len(shape)
    blocks = [B for r in range(1, n + 1) for B in itertools.combinations(range(n), r)]
    S = frozenset(profiles(shape))
    seen, todo = {S}, [S]
    while todo:
        E = todo.pop()
        for B in blocks:
            F = step(U, E, B, shape)
            if F not in seen:
                seen.add(F)
                todo.append(F)
    return seen


def pte_path(U, shape):
    n = len(shape)
    E = frozenset(profiles(shape))
    path = [E]
    while True:
        F = step(U, E, range(n), shape)
        if F == E:
            return path
        path.append(F)
        E = F
        if not E:
            return path


def spp(U, shape, order):
    E = frozenset(profiles(shape))
    while True:
        old = E
        for i in order:
            E = verdict(U, E, i, shape)
        if E == old:
            return E


def rpp(U, shape, order):
    n = len(shape)
    E = frozenset(profiles(shape))
    while True:
        F = step(U, E, range(n), shape)
        if not F:
            F = E
            for i in order:
                F = verdict(U, F, i, shape)
        if F == E:
            return E
        E = F


def relabelings(shape):
    """(player permutation p, strategy bijections): new profile t with t[p[i]] = sigma_i(s[i]); same shape only."""
    n = len(shape)
    for p in itertools.permutations(range(n)):
        if any(shape[p[i]] != shape[i] for i in range(n)):
            continue
        for sig in itertools.product(*[list(itertools.permutations(range(shape[i]))) for i in range(n)]):
            yield p, sig


def apply(p, sig, s):
    t = [None] * len(s)
    for i, x in enumerate(s):
        t[p[i]] = sig[i][x]
    return tuple(t)


def automorphisms(U, shape):
    A = []
    for p, sig in relabelings(shape):
        if all(U[apply(p, sig, s)][p[i]] == U[s][i] for s in U for i in range(len(shape))):
            A.append((p, sig))
    return A


def image(p, sig, E):
    return frozenset(apply(p, sig, s) for s in E)


def vd(n, b, costs):
    shape = (2,) * n
    U = {}
    for s in profiles(shape):
        some = 0 in s
        U[s] = tuple((b - costs[i]) if s[i] == 0 else (b if some else 0) for i in range(n))
    return U, shape


def F(D, n):
    """F_D: every player of D plays N (1) and some player outside D volunteers (0)."""
    return frozenset(s for s in profiles((2,) * n) if all(s[i] == 1 for i in D)
                     and any(s[j] == 0 for j in range(n) if j not in D))


def show(E):
    return sorted("".join("VN"[x] for x in s) for s in E)


# ----------------------------------------------------------------------------------------------------------------------
say(__doc__.strip().splitlines()[0])
say()
say("== 1. Volunteer's dilemma, b = 10, c = 4 (symmetric costs)")
for n in range(2, 7):
    U, shape = vd(n, Fr(10), [Fr(4)] * n)
    R = reachable(U, shape)
    S = frozenset(profiles(shape))
    fam = {S} | {F(D, n) for r in range(n + 1) for D in itertools.combinations(range(n), r)}
    rests = [E for E in R if E and is_rest(U, E, shape)]
    A = automorphisms(U, shape)
    perms = {p for p, _ in A}
    strat_ok = all(all(sg == (0, 1) for sg in sig) for _, sig in A)
    inv = [E for E in R if all(image(p, sig, E) == E for p, sig in A)]
    inv_rest = [E for E in inv if E and is_rest(U, E, shape)]
    pth = pte_path(U, shape)
    say(f"n = {n}: |reachable| = {len(R)}; reachable = {{S}} u {{F_D}}: {R == fam}; |Aut| = {len(A)} "
        f"(player permutations {len(perms)} of {len(list(itertools.permutations(range(n))))}, strategy maps identity: {strat_ok})")
    say(f"  nonempty reachable rest points: {sorted(show(E) for E in rests)}")
    say(f"  invariant reachable sets: sizes {sorted(len(E) for E in inv)}; invariant nonempty rest points: {len(inv_rest)}")
    say(f"  PTE path sizes: {[len(E) for E in pth]}")
    vs, vr = {}, {}
    for order in itertools.permutations(range(n)):
        a, r_ = spp(U, shape, order), rpp(U, shape, order)
        ok_s = a == frozenset({tuple(0 if i == order[0] else 1 for i in range(n))})
        ok_r = r_ == frozenset({tuple(0 if i == order[-1] else 1 for i in range(n))})
        vs[ok_s] = vs.get(ok_s, 0) + 1
        vr[ok_r] = vr.get(ok_r, 0) + 1
    say(f"  SPP = {{only the first reasoner volunteers}} in {vs.get(True, 0)} of {sum(vs.values())} orders; "
        f"RPP = {{only the last reasoner volunteers}} in {vr.get(True, 0)} of {sum(vr.values())} orders")

say()
say("== 2. Player-specific costs, b = 10")
for costs in [(1, 5), (5, 1), (1, 5, 5), (5, 5, 1), (1, 3, 9), (9, 3, 1, 5), (2, 9, 4, 7, 1)]:
    n = len(costs)
    U, shape = vd(n, Fr(10), [Fr(c) for c in costs])
    good_s = good_r = 0
    for order in itertools.permutations(range(n)):
        good_s += spp(U, shape, order) == frozenset({tuple(0 if i == order[0] else 1 for i in range(n))})
        good_r += rpp(U, shape, order) == frozenset({tuple(0 if i == order[-1] else 1 for i in range(n))})
    R = reachable(U, shape)
    rests = [E for E in R if E and is_rest(U, E, shape)]
    say(f"costs {costs}: SPP first reasoner volunteers in {good_s}, RPP last reasoner in {good_r}, of "
        f"{len(list(itertools.permutations(range(n))))} orders; nonempty reachable rest points {sorted(show(E) for E in rests)}")

say()
say("== 3. Schedule form: blocks invariant under the stabilizer of the history, along the PTE path")


def stabilizer_blocks(U, shape, history, A):
    n = len(shape)
    H = [(p, sig) for p, sig in A if all(image(p, sig, E) == E for E in history)]
    blocks = [B for r in range(1, n + 1) for B in itertools.combinations(range(n), r)]
    return [B for B in blocks if all(frozenset(p[i] for i in B) == frozenset(B) for p, _ in H)], len(H)


G_E = {(0, 0): (Fr(0), Fr(0)), (0, 1): (Fr(3), Fr(1)), (1, 0): (Fr(1), Fr(3)), (1, 1): (Fr(1, 2), Fr(1, 2))}
CH = {(0, 0): (0, 0), (0, 1): (3, 1), (1, 0): (1, 3), (1, 1): (2, 2)}
games = [("G_e, e = 1/2", G_E, (2, 2)), ("chicken", CH, (2, 2))]
for n in range(2, 6):
    U, shape = vd(n, Fr(10), [Fr(4)] * n)
    games.append((f"VD_{n}", U, shape))
for name, U, shape in games:
    A = automorphisms(U, shape)
    pth = pte_path(U, shape)
    rows = []
    for k in range(len(pth)):
        bl, h = stabilizer_blocks(U, shape, pth[:k + 1], A)
        rows.append(f"step {k}: |H| = {h}, invariant blocks {bl}")
    say(f"{name}: PTE path sizes {[len(E) for E in pth]}; " + "; ".join(rows))

say()
say("== 4. The separating 3x3 game")
u1 = [[2, 0, 8], [6, 1, 4], [3, 5, 7]]
shape = (3, 3)
U = {(x, y): (u1[x][y], u1[y][x]) for x in range(3) for y in range(3)}
say(f"payoffs: {dict(sorted(U.items()))}")
A = automorphisms(U, shape)
say(f"|Aut| = {len(A)}; automorphisms: {A}; transitive on players: {any(p == (1, 0) for p, _ in A)}")
pth = pte_path(U, shape)
say(f"PTE path: {[sorted(E) for E in pth]}")
E = frozenset(profiles(shape))
log = []
for order in [(0, 1), (1, 0)]:
    E = frozenset(profiles(shape))
    log = []
    while True:
        old = E
        for i in order:
            m = guarantee(U, E, i, shape)
            E = verdict(U, E, i, shape)
            log.append(f"player {i + 1}: guarantee {m}, set {sorted(E)}")
        if E == old:
            break
    say(f"SPP, order {tuple(i + 1 for i in order)}:")
    for row in log:
        say("  " + row)
R = reachable(U, shape)
inv_rest = [E for E in R if E and is_rest(U, E, shape) and all(image(p, sig, E) == E for p, sig in A)]
say(f"reachable sets: {len(R)}; invariant nonempty reachable rest points: {[sorted(E) for E in inv_rest]}")
bl, h = stabilizer_blocks(U, shape, [frozenset(profiles(shape))], A)
say(f"blocks invariant under Aut at the start: {bl}")

OUT.write_text("\n".join(_lines) + "\n")
