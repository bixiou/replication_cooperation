"""coop_td_features.py: which feature of the traveler's dilemma makes reasoning select the symmetric optimum?

Logic. Candidate premises on a symmetric two-player game (u_2(x, y) = u_1(y, x)), each with the conclusion of equal
treatment (ET): the prediction is nonempty and every predicted profile is Pareto-optimal with equal payoffs.
  (Cp)    copying pays the worse-off: at every profile (x, y) with x != y and u_1(x,y) != u_2(x,y), the player with
          the lower payoff strictly gains by switching to the other's strategy (player 1: u_1(y, y) > u_1(x, y)).
          The TD (the higher claimant matches the lower claim) and the PD (the cooperator defects) satisfy it;
          chicken, G_e and the battle of the sexes do not (copying the aggressive player leads to the crash).
  (CpPO)  the same at the asymmetric Pareto-optimal profiles only.
  (Cpw)   weak version: the worse-off player does not lose by copying.
  (Cpe)   tie-robust version: at every profile (x, y) with x != y, every player whose payoff is not above the other's
          strictly gains by copying (with unequal payoffs this is (Cp); with equal payoffs both must gain).
Section 2 (2026-11-04, second run): (Cpe) on games with ties: symmetric 2x2 with payoffs in {0,...,4}, symmetric 3x3
with payoffs in {0,1,2} (all 19,683) and the TDs.
Each premise also requires a Pareto-optimal profile with equal payoffs (so that ET has content).
For each class and concept: number of games satisfying the premise, failures of ET. Concepts: SPP and RPP (both
orders), PTE, truncated PTE, NB, MG, alpha-core (from coop_general_properties.py). Classes: symmetric strict 2x2;
symmetric 2x2 with payoffs in {0,...,4}; all 362,880 symmetric 3x3 ranking games; 2,000 random symmetric 4x4 ranking
games (seed 4); the TDs and symmetric PDs of coop_general_properties.py; named games.
"""
import itertools
import random
from pathlib import Path

import coop_general_properties as g

OUT = Path(__file__).with_suffix(".out")
_lines = []


def say(x=""):
    print(x, flush=True)
    _lines.append(str(x))


def has_eq_po(U):
    return any(U[s][0] == U[s][1] for s in g.pareto(U))


def copy_gain(U, s, weak=False, both_if_equal=False):
    x, y = s
    a, b = U[s]
    if a == b:
        if both_if_equal:
            return U[(y, y)][0] > a and U[(x, x)][1] > b
        return True
    if a < b:  # player 1 worse off: copy y
        return U[(y, y)][0] >= a if weak else U[(y, y)][0] > a
    return U[(x, x)][1] >= b if weak else U[(x, x)][1] > b


def premise(U, shape, var):
    if not g.is_symmetric(U, shape) or not has_eq_po(U):
        return False
    if var == "Cp":
        return all(copy_gain(U, s) for s in U if s[0] != s[1])
    if var == "Cpw":
        return all(copy_gain(U, s, weak=True) for s in U if s[0] != s[1])
    if var == "Cpe":
        return all(copy_gain(U, s, both_if_equal=True) for s in U if s[0] != s[1])
    if var == "CpPO":
        return all(copy_gain(U, s) for s in g.pareto(U) if s[0] != s[1])
    raise ValueError(var)


def et(U, E):
    P = g.pareto(U)
    return bool(E) and all(s in P and U[s][0] == U[s][1] for s in E)


CONC = {k: g.CONCEPTS[k] for k in ("SPP12", "SPP21", "RPP12", "RPP21", "PTE", "truncPTE", "NB", "MG", "alphacore")}


def sym_games(m, vals):
    P = g.profiles((m, m))
    for v in itertools.product(vals, repeat=m * m):
        yield g.sym_games(m, dict(zip(P, v)))


def sym_rank(m, rng=None, count=None):
    P = g.profiles((m, m))
    if rng is None:
        for perm in itertools.permutations(range(1, m * m + 1)):
            yield g.sym_games(m, dict(zip(P, perm)))
    else:
        for _ in range(count):
            yield g.sym_games(m, dict(zip(P, rng.sample(range(1, m * m + 1), m * m))))


def run(name, games, variants=("Cp", "CpPO", "Cpw"), concepts=None):
    concepts = concepts or list(CONC)
    games = list(games)
    say(f"-- {name}: {len(games)} games")
    for var in variants:
        rel = [G for G in games if premise(*G, var)]
        row, first = [], {}
        for c in concepts:
            k = 0
            for U, sh in rel:
                E = CONC[c](U, sh)
                if not et(U, E):
                    k += 1
                    first.setdefault(c, (U, sh, E))
            row.append(f"{c} {k}")
        say(f"   ({var}) premise in {len(rel)}; ET failures: " + ", ".join(row))
        for c in ("SPP12", "RPP12", "PTE"):
            if c in first and var in ("Cp", "CpPO"):
                U, sh, E = first[c]
                say(f"      first failure of {c}: u_1 rows {[[U[(x, y)][0] for y in range(sh[1])] for x in range(sh[0])]}, "
                    f"prediction {sorted(E)}")


def named():
    say("-- named games: premise (Cp), (CpPO)")
    games = {
        "PD (5,3,1,0)": {(0, 0): (3, 3), (0, 1): (0, 5), (1, 0): (5, 0), (1, 1): (1, 1)},
        "stag hunt (4,0;3,3)": {(0, 0): (4, 4), (0, 1): (0, 3), (1, 0): (3, 0), (1, 1): (3, 3)},
        "Aumann's chicken": g.CHICKEN[0],
        "G_e, e = 1/2": g.GE[0],
        "battle of the sexes": {(0, 0): (0, 0), (0, 1): (2, 3), (1, 0): (3, 2), (1, 1): (1, 1)},
    }
    for name, U in games.items():
        sh = (2, 2)
        say(f"   {name}: symmetric {g.is_symmetric(U, sh)}, (Cp) {premise(U, sh, 'Cp')}, (CpPO) {premise(U, sh, 'CpPO')}; "
            f"SPP12 {sorted(g.spp(U, sh, (0, 1)))}, RPP12 {sorted(g.rpp(U, sh, (0, 1)))}, PTE {sorted(g.pte(U, sh))}")
    for (M, r), (U, sh) in g.tds():
        if M in (5, 10) and r in (2, 3):
            say(f"   TD M = {M}, r = {r}: (Cp) {premise(U, sh, 'Cp')}")


def _main_features():
    say(__doc__.strip().splitlines()[0])
    named()
    run("symmetric strict 2x2", g.sym_strict_2x2())
    run("symmetric 2x2, payoffs in {0,...,4}", sym_games(2, range(5)))
    run("traveler's dilemmas", [G for _, G in g.tds()])
    run("symmetric PDs", [G for G in g.pds() if g.is_symmetric(*G)])
    run("random symmetric 4x4 ranking games (2,000, seed 4)", sym_rank(4, random.Random(4), 2000),
        concepts=["SPP12", "RPP12", "PTE", "NB", "MG"])
    run("all symmetric 3x3 ranking games", sym_rank(3), concepts=["SPP12", "SPP21", "RPP12", "PTE", "NB", "MG"])
    OUT.write_text("\n".join(_lines) + "\n")


def _main_ties():
    say()
    say("== Section 2: the tie-robust version (Cpe)")
    run("symmetric 2x2, payoffs in {0,...,4}", sym_games(2, range(5)), variants=("Cpe",))
    run("traveler's dilemmas", [G for _, G in g.tds()], variants=("Cpe",))
    run("symmetric 3x3, payoffs in {0,1,2}", sym_games(3, range(3)), variants=("Cpe",),
        concepts=["SPP12", "SPP21", "RPP12", "PTE", "truncPTE", "NB", "MG"])
    run("all symmetric 3x3 ranking games", sym_rank(3), variants=("Cpe",), concepts=["SPP12", "RPP12", "PTE"])


if __name__ == "__main__":
    import sys
    if "--ties" in sys.argv:
        prev = OUT.read_text().splitlines() if OUT.exists() else []
        _main_ties()
        OUT.write_text("\n".join(prev + _lines) + "\n")
    else:
        _main_features()
