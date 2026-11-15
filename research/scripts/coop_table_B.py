"""coop_table_B.py: entries of Table 1 of the paper under the general properties
(Pareto optimality and equal treatment where copying pays, which generalize cooperation in the two dilemmas).

Logic. For each concept of the table that can be computed on finite two-player games, count the games where it fails
  (PO)   Pareto optimality without ties: in a game without ties, every predicted profile is Pareto-optimal (vacuous on
         an empty prediction; the empty count is printed beside it);
  (ETc)  equal treatment where copying pays: in a symmetric game without ties in which copying pays (at every profile
         (x, y) with x != y the player with the lower payoff strictly gains by switching to the other's strategy), the
         prediction is {(z, z)}, with (z, z) the best symmetric profile.
Classes. PO: the 576 strict 2x2 games and 3,000 random 3x3 games in which each player's payoffs rank the nine profiles
(seed 20261104). ETc: the 16 symmetric strict 2x2 games and all 123,648 symmetric 3x3 ranking games in which copying
pays, the 15 symmetric strict PDs with payoffs in {0,...,5} (copying pays in each), and tie-free symmetric
perturbations of the traveler's dilemmas with M = 4..9 and r in {3/2, 2, 5/2, 3} (player 1's payoff at (x, y) plus
eps * k(x, y), with k a random ranking of the profiles, eps = 1/1000, seed 7; player 2's by symmetry).
Concepts. From coop_general_properties.py: SPP and RPP under both orders, the PTE limit, its truncation, NB, MG, MG plus
U(G), the alpha-core, Nash equilibrium, superrationality (symmetric games only). Added here: the union over the two
orders of SPP (UNION_S) and of RPP (UNION_R); the Berge equilibrium (for two players each player's payoff is maximal
over the other's strategies, given her own; with two players it is the Nash equilibrium of the game in which each
player receives the other's payoff); team reasoning (profiles with the largest sum of payoffs); the credible alpha-core
for two players (no profile improves strictly on both payoffs, and no single player gains strictly by a deviation that
the other answers by a best reply, the worst one for the deviator if several); the largest consistent set of Chwe
(1994) with single movers (LCS1) and with any coalition (LCS2): a profile a is indirectly dominated by b if a chain of
moves leads from a to b in which every mover strictly prefers b to the profile she moves from, and the set is obtained
from all profiles by removing, until nothing changes, every profile a from which some move (by a coalition S to a')
cannot be deterred, i.e. no profile b of the current set with b = a' or a' indirectly dominated by b leaves some
member of S not strictly better off than at a.
"""
import itertools
import random
from fractions import Fraction as Fr
from pathlib import Path

import coop_general_properties as g
import coop_td_features as f

OUT = Path(__file__).with_suffix(".out")
_lines = []


def say(x=""):
    print(x, flush=True)
    _lines.append(str(x))


def berge(U, sh):
    out = []
    for s in U:
        x, y = s
        if U[s][0] >= max(U[(x, b)][0] for b in range(sh[1])) and U[s][1] >= max(U[(a, y)][1] for a in range(sh[0])):
            out.append(s)
    return frozenset(out)


def team(U, sh):
    best = max(sum(U[s]) for s in U)
    return frozenset(s for s in U if sum(U[s]) == best)


def credible_core(U, sh):
    out = []
    for s in U:
        if any(U[t][0] > U[s][0] and U[t][1] > U[s][1] for t in U):
            continue
        ok = True
        for t in range(sh[0]):  # player 1 deviates, player 2 best replies (worst best reply for player 1)
            if t == s[0]:
                continue
            b = max(U[(t, y)][1] for y in range(sh[1]))
            if min(U[(t, y)][0] for y in range(sh[1]) if U[(t, y)][1] == b) > U[s][0]:
                ok = False
        for t in range(sh[1]):
            if t == s[1]:
                continue
            b = max(U[(x, t)][0] for x in range(sh[0]))
            if min(U[(x, t)][1] for x in range(sh[0]) if U[(x, t)][0] == b) > U[s][1]:
                ok = False
        if ok:
            out.append(s)
    return frozenset(out)


def moves(s, sh, joint):
    """(movers, target) for every move from s; movers = the players whose strategy changes."""
    out = []
    for t in itertools.product(*[range(k) for k in sh]):
        S = tuple(i for i in range(len(sh)) if t[i] != s[i])
        if S and (joint or len(S) == 1):
            out.append((S, t))
    return out


def lcs(U, sh, joint):
    P = list(U)
    mv = {a: moves(a, sh, joint) for a in P}
    dom = {}  # dom[b] = set of profiles indirectly dominated by b
    for b in P:
        # a -> a' allowed if every mover strictly prefers b to a; a << b iff b reachable in >= 1 step
        reach_to_b = set()
        frontier = [b]
        while frontier:  # backward search: which a have an allowed move to a node already reaching b (or b)
            nxt = []
            for a in P:
                if a in reach_to_b:
                    continue
                for S, t in mv[a]:
                    if (t == b or t in reach_to_b) and all(U[b][i] > U[a][i] for i in S):
                        reach_to_b.add(a)
                        nxt.append(a)
                        break
            frontier = nxt
        dom[b] = reach_to_b
    Z = set(P)
    while True:
        rem = set()
        for a in Z:
            for S, t in mv[a]:
                deter = any((bb == t or t in dom[bb]) and not all(U[bb][i] > U[a][i] for i in S) for bb in Z)
                if not deter:
                    rem.add(a)
                    break
        if not rem:
            return frozenset(Z)
        Z -= rem


CON = dict(g.CONCEPTS)
CON["UNION_S"] = lambda U, sh: g.spp(U, sh, (0, 1)) | g.spp(U, sh, (1, 0))
CON["UNION_R"] = lambda U, sh: g.rpp(U, sh, (0, 1)) | g.rpp(U, sh, (1, 0))
CON["Berge"] = berge
CON["team"] = team
CON["credcore"] = credible_core
CON["LCS1"] = lambda U, sh: lcs(U, sh, False)
CON["LCS2"] = lambda U, sh: lcs(U, sh, True)
CON["superrational"] = g.superrational
ORDER = ["SPP12", "SPP21", "RPP12", "RPP21", "UNION_S", "UNION_R", "PTE", "truncPTE", "NB", "MG", "MGU",
         "alphacore", "credcore", "LCS1", "LCS2", "Berge", "team", "Nash"]


def po_fail(U, E):
    P = g.pareto(U)
    return bool(E) and not E <= P


def cp(U, sh):
    return g.is_symmetric(U, sh) and not g.has_ties(U, sh) and all(f.copy_gain(U, s) for s in U if s[0] != s[1])


def zz(U, sh):
    return max((s for s in U if s[0] == s[1]), key=lambda s: U[s][0])


def tally_po(name, games, concepts):
    games = [G for G in games if not g.has_ties(*G)]
    say(f"-- PO, {name}: {len(games)} games without ties")
    for c in concepts:
        k = e = 0
        first = None
        for U, sh in games:
            E = CON[c](U, sh)
            e += not E
            if po_fail(U, E):
                k += 1
                first = first or (U, sh, E)
        msg = f"   {c}: PO fails in {k}; empty in {e}"
        if first:
            U, sh, E = first
            msg += f"; first failure {sorted(U.items())}, prediction {sorted(E)}"
        say(msg)


def tally_etc(name, games, concepts):
    games = [G for G in games if cp(*G)]
    say(f"-- ETc, {name}: {len(games)} symmetric games without ties in which copying pays")
    for c in concepts:
        k = e = 0
        first = None
        for U, sh in games:
            E = CON[c](U, sh)
            e += not E
            if E != frozenset([zz(U, sh)]):
                k += 1
                first = first or (U, sh, E)
        msg = f"   {c}: ETc fails in {k}; empty in {e}"
        if first:
            U, sh, E = first
            msg += f"; first failure: u_1 rows {[[U[(x, y)][0] for y in range(sh[1])] for x in range(sh[0])]}, prediction {sorted(E)}"
        say(msg)


def sym_pds():
    for S, P, R, T in itertools.combinations(range(6), 4):
        yield {(0, 0): (R, R), (0, 1): (S, T), (1, 0): (T, S), (1, 1): (P, P)}, (2, 2)


def perturbed_tds(rng):
    for M in range(4, 10):
        for r in (Fr(3, 2), Fr(2), Fr(5, 2), Fr(3)):
            U0, sh = g.td_game(M, r)
            P = list(U0)
            k = dict(zip(P, rng.sample(range(len(P)), len(P))))
            u1 = {s: U0[s][0] + Fr(k[s], 1000) for s in P}
            yield g.sym_games(sh[0], u1)


def main():
    rng = random.Random(20261104)
    rank3 = []
    P9 = g.profiles((3, 3))
    for _ in range(3000):
        a = rng.sample(range(1, 10), 9)
        b = rng.sample(range(1, 10), 9)
        rank3.append(({s: (a[i], b[i]) for i, s in enumerate(P9)}, (3, 3)))
    po_concepts = ORDER
    tally_po("strict 2x2", list(g.strict_2x2()), po_concepts)
    tally_po("random 3x3 ranking games (seed 20261104)", rank3, po_concepts)
    tally_po("G_e", [g.GE], po_concepts + ["superrational"])
    etc_concepts = ORDER + ["superrational"]
    tally_etc("symmetric strict 2x2", list(g.sym_strict_2x2()), etc_concepts)
    tally_etc("symmetric strict PDs, payoffs in {0,...,5}", list(sym_pds()), etc_concepts)
    tally_etc("perturbed traveler's dilemmas (seed 7)", list(perturbed_tds(random.Random(7))), etc_concepts)
    tally_etc("all symmetric 3x3 ranking games", list(f.sym_rank(3)), etc_concepts)
    OUT.write_text("\n".join(_lines) + "\n")


if __name__ == "__main__":
    main()
