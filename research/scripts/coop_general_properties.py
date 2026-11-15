"""coop_general_properties.py: can "cooperation in the PD and the TD" be replaced by general properties?

Logic. For each concept below and each game of the test classes, record whether the prediction is nonempty and
whether it satisfies the candidate general properties:
  Eff    every predicted profile is Pareto-optimal (reported separately on games without ties, EffStrict);
  WE     the prediction contains a Pareto-optimal profile;
  IR     every predicted profile is individually rational (pure security levels);
  ET     equal treatment at the level of profiles: if some Pareto-optimal profile gives all players the same payoff,
         every predicted profile is such a profile;
  Inv    (symmetric games only) the exchange of the players maps the prediction to itself.
Conventions of the printed counts: "Eff" counts an empty prediction as a failure (the note's (PO) holds vacuously on
an empty prediction, so the note reads these columns together with "empty"); "ET" is printed on every class, but the
note's equal treatment is defined on symmetric games only, so the ET column of a non-symmetric class (strict 2x2,
2x2 with ties, PDs) is an equal-payoff test that the note does not use.
Concepts (two players): sequential (SPP) and resolved (RPP) perfect prediction under both orders; the limit of the
perfectly transparent equilibrium (PTE, may be empty); its last nonempty round (truncated PTE); the Nash bargaining
selection NB (individually rational maximizers of the product of gains over security levels); the maximin of the gains
MG; the alpha-core in pure strategies (two players: individually rational and not strictly improved upon by both);
POIR (Pareto-optimal and individually rational profiles); EGAL = U(G) plus the equal-payoff profiles of POIR when
there are some, else POIR; MGU = MG plus U(G) (the set of profiles from which no deviation pays, whatever the others
answer); pure Nash equilibria; superrationality on symmetric games (symmetric profiles with the
highest common payoff).
Test classes: the 576 strict 2x2 games; the 6,561 2x2 games with payoffs in {0,1,2}; the 24 symmetric strict 2x2
games (player 1's payoffs a ranking of the four profiles, u_2(x,y) = u_1(y,x)); the 81 symmetric 2x2 games with
payoffs in {0,1,2}; 3,000 random symmetric 3x3 ranking games (seed 20261103); the 225 prisoner's dilemmas with each
player's T > R > P > S in {0,...,5}; the traveler's dilemmas with M = 2..10 and r in {1/2, 1, 3/2, 2, 3, 5}; chicken
(6,6), (7,2), (2,7), (0,0); G_e (e = 1/2).
Also checked: Pareto optima and individually rational profiles of every PD and TD of the classes (the hand lemmas of
the note); symmetric efficiency SE (section 5); every reachable set of Aumann's chicken (section 6); the symmetric
games with ties in which a nonempty PTE limit breaks equal treatment (section 7).
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


# ---------------------------------------------------------------- perfect prediction
def guarantee(U, E, i, shape):
    return max(min(U[s][i] for s in E if s[i] == t) for t in range(shape[i]) if any(s[i] == t for s in E))


def verdict(U, E, i, shape):
    if not E:
        return frozenset()
    m = guarantee(U, E, i, shape)
    return frozenset(s for s in E if U[s][i] >= m)


def joint(U, E, shape):
    out = frozenset(E)
    for i in range(len(shape)):
        out &= verdict(U, E, i, shape)
    return out


def pte_rounds(U, shape):
    E = frozenset(U)
    path = [E]
    while True:
        F = joint(U, E, shape)
        if F == E:
            return path
        path.append(F)
        if not F:
            return path
        E = F


def pte(U, shape):
    return pte_rounds(U, shape)[-1]


def truncated(U, shape):
    path = pte_rounds(U, shape)
    return [E for E in path if E][-1]


def spp(U, shape, order):
    E = frozenset(U)
    while True:
        old = E
        for i in order:
            E = verdict(U, E, i, shape)
        if E == old:
            return E


def rpp(U, shape, order):
    E = frozenset(U)
    while True:
        F = joint(U, E, shape)
        if not F:
            F = E
            for i in order:
                F = verdict(U, F, i, shape)
        if F == E:
            return E
        E = F


# ---------------------------------------------------------------- payoff-based sets
def security(U, shape):
    n = len(shape)
    return [max(min(U[s][i] for s in U if s[i] == t) for t in range(shape[i])) for i in range(n)]


def ir_set(U, shape):
    v = security(U, shape)
    return frozenset(s for s in U if all(U[s][i] >= v[i] for i in range(len(shape))))


def pareto(U):
    P = []
    for s in U:
        dominated = any(all(U[t][i] >= U[s][i] for i in range(len(U[s]))) and U[t] != U[s] for t in U)
        if not dominated:
            P.append(s)
    return frozenset(P)


def u_set(U, shape):
    n = len(shape)
    out = []
    for s in U:
        ok = True
        for i in range(n):
            for t in range(shape[i]):
                if t != s[i] and max(U[w][i] for w in U if w[i] == t) > U[s][i]:
                    ok = False
        if ok:
            out.append(s)
    return frozenset(out)


def nb(U, shape):
    v = security(U, shape)
    IR = ir_set(U, shape)
    def prod(s):
        p = 1
        for i in range(len(shape)):
            p *= U[s][i] - v[i]
        return p
    best = max(prod(s) for s in IR)
    return frozenset(s for s in IR if prod(s) == best)


def mg(U, shape):
    v = security(U, shape)
    IR = ir_set(U, shape)
    val = lambda s: min(U[s][i] - v[i] for i in range(len(shape)))
    best = max(val(s) for s in IR)
    return frozenset(s for s in IR if val(s) == best)


def alpha_core(U, shape):
    IR = ir_set(U, shape)
    return frozenset(s for s in IR if not any(all(U[t][i] > U[s][i] for i in range(len(shape))) for t in U))


def poir(U, shape):
    return pareto(U) & ir_set(U, shape)


def equal_payoff(U, s):
    return len(set(U[s])) == 1


def egal(U, shape):
    P = poir(U, shape)
    Q = frozenset(s for s in P if equal_payoff(U, s))
    return u_set(U, shape) | (Q if Q else P)


def mgu(U, shape):
    return mg(U, shape) | u_set(U, shape)


def nash(U, shape):
    n = len(shape)
    out = []
    for s in U:
        if all(U[s][i] >= max(U[s[:i] + (t,) + s[i + 1:]][i] for t in range(shape[i])) for i in range(n)):
            out.append(s)
    return frozenset(out)


def superrational(U, shape):
    diag = [s for s in U if len(set(s)) == 1]
    best = max(U[s][0] for s in diag)
    return frozenset(s for s in diag if U[s][0] == best)


# ---------------------------------------------------------------- properties
def has_ties(U, shape):
    return any(len({U[s][i] for s in U}) < len(U) for i in range(len(shape)))


def is_symmetric(U, shape):
    return shape[0] == shape[1] and all(U[(x, y)] == (U[(y, x)][1], U[(y, x)][0]) for x, y in U)


def swap(E):
    return frozenset((y, x) for x, y in E)


def props(U, shape, E, sym):
    P = pareto(U)
    IR = ir_set(U, shape)
    Peq = frozenset(s for s in P if equal_payoff(U, s))
    d = {
        "empty": not E,
        "Eff": bool(E) and E <= P,
        "WE": bool(E & P),
        "IR": E <= IR,
        "ET": (not Peq) or (bool(E) and E <= Peq),
    }
    if sym:
        d["Inv"] = swap(E) == E
    return d


def se_relevant(U, shape):
    return is_symmetric(U, shape) and any(len(set(s)) == 1 for s in pareto(U))


def se_holds(U, shape, E):
    P = pareto(U)
    return bool(E) and all(s in P and len(set(s)) == 1 for s in E)


def reach(U, shape):
    n = len(shape)
    blocks = [B for k in range(1, n + 1) for B in itertools.combinations(range(n), k)]
    S0 = frozenset(U)
    seen, todo = {S0}, [S0]
    while todo:
        E = todo.pop()
        for B in blocks:
            F = frozenset(E)
            for i in B:
                F &= verdict(U, E, i, shape)
            if F not in seen:
                seen.add(F)
                todo.append(F)
    return seen


CONCEPTS = {
    "SPP12": lambda U, sh: spp(U, sh, (0, 1)),
    "SPP21": lambda U, sh: spp(U, sh, (1, 0)),
    "RPP12": lambda U, sh: rpp(U, sh, (0, 1)),
    "RPP21": lambda U, sh: rpp(U, sh, (1, 0)),
    "PTE": pte,
    "truncPTE": truncated,
    "NB": nb,
    "MG": mg,
    "alphacore": alpha_core,
    "POIR": poir,
    "EGAL": egal,
    "MGU": mgu,
    "Nash": nash,
}


def strict_2x2():
    P = profiles((2, 2))
    for p1 in itertools.permutations(range(1, 5)):
        for p2 in itertools.permutations(range(1, 5)):
            yield {s: (p1[k], p2[k]) for k, s in enumerate(P)}, (2, 2)


def ties_2x2():
    P = profiles((2, 2))
    for vals in itertools.product(range(3), repeat=8):
        yield {s: (vals[2 * k], vals[2 * k + 1]) for k, s in enumerate(P)}, (2, 2)


def sym_games(m, values):
    """u_1 on the m x m profiles given by values (a dict), u_2(x, y) = u_1(y, x)."""
    P = profiles((m, m))
    return {s: (values[s], values[(s[1], s[0])]) for s in P}, (m, m)


def sym_strict_2x2():
    P = profiles((2, 2))
    for perm in itertools.permutations(range(1, 5)):
        yield sym_games(2, dict(zip(P, perm)))


def sym_ties_2x2():
    P = profiles((2, 2))
    for vals in itertools.product(range(3), repeat=4):
        yield sym_games(2, dict(zip(P, vals)))


def sym_3x3(rng, count):
    P = profiles((3, 3))
    for _ in range(count):
        perm = rng.sample(range(1, 10), 9)
        yield sym_games(3, dict(zip(P, perm)))


def pds():
    for a in itertools.combinations(range(6), 4):
        for b in itertools.combinations(range(6), 4):
            S1, P1, R1, T1 = a
            S2, P2, R2, T2 = b
            yield {(0, 0): (R1, R2), (0, 1): (S1, T2), (1, 0): (T1, S2), (1, 1): (P1, P2)}, (2, 2)


def td_game(M, r):
    U = {}
    for x in range(2, M + 1):
        for y in range(2, M + 1):
            U[(x - 2, y - 2)] = (x, x) if x == y else ((x + r, x - r) if x < y else (y - r, y + r))
    return U, (M - 1, M - 1)


def tds():
    for M in range(2, 11):
        for r in (Fr(1, 2), Fr(1), Fr(3, 2), Fr(2), Fr(3), Fr(5)):
            yield (M, r), td_game(M, r)


CHICKEN = ({(0, 0): (6, 6), (0, 1): (2, 7), (1, 0): (7, 2), (1, 1): (0, 0)}, (2, 2))  # Y = 0, A = 1
GE = ({(0, 0): (Fr(1, 2), Fr(1, 2)), (0, 1): (1, 3), (1, 0): (3, 1), (1, 1): (0, 0)}, (2, 2))  # Y = 0, A = 1


def tally(name, games, sym=False):
    say(f"-- {name}: {len(games)} games" + ("" if sym else " (not symmetric: the ET column is not the note's (ET))"))
    keys = ["empty", "Eff", "WE", "IR", "ET"] + (["Inv"] if sym else [])
    strict = [not has_ties(U, sh) for U, sh in games]
    say("   concept: " + ", ".join(f"{k} fails" if k != "empty" else "empty" for k in keys) + ", Eff fails without ties")
    for cname, f in CONCEPTS.items():
        cnt = {k: 0 for k in keys}
        eff_strict_fail = 0
        for (U, sh), st in zip(games, strict):
            E = f(U, sh)
            d = props(U, sh, E, sym)
            for k in keys:
                if k == "empty":
                    cnt[k] += d[k]
                else:
                    cnt[k] += not d[k]
            if st and not d["Eff"]:
                eff_strict_fail += 1
        say(f"   {cname:9s}: " + ", ".join(str(cnt[k]) for k in keys) + f", {eff_strict_fail}")
    if sym:
        cnt = {k: 0 for k in keys}
        for U, sh in games:
            d = props(U, sh, superrational(U, sh), sym)
            for k in keys:
                cnt[k] += d[k] if k == "empty" else not d[k]
        say(f"   {'superrat':9s}: " + ", ".join(str(cnt[k]) for k in keys))


def main():
    say(__doc__.strip().splitlines()[0])
    rng = random.Random(20261103)

    say()
    say("== 1. Lemmas on the dilemmas")
    bad = 0
    n = 0
    for U, sh in pds():
        n += 1
        bad += poir(U, sh) != frozenset({(0, 0)})
    say(f"PDs (each player's T > R > P > S in 0..5): {n}; games where Pareto-optimal and individually rational "
        f"profiles differ from {{CC}}: {bad}")
    for (M, r), (U, sh) in tds():
        P = sorted((x + 2, y + 2) for x, y in pareto(U))
        Q = sorted((x + 2, y + 2) for x, y in poir(U, sh))
        Peq = sorted((x + 2, y + 2) for x, y in pareto(U) if equal_payoff(U, (x, y)))
        if M in (2, 3, 5, 6, 10):
            say(f"TD M = {M}, r = {r}: Pareto-optimal {P}; and individually rational {Q}; equal-payoff Pareto-optimal {Peq}")
    viol = []
    for (M, r), (U, sh) in tds():
        P = pareto(U)
        IR = ir_set(U, sh)
        expect_P = {(M - 2, M - 2), (M - 3, M - 2), (M - 2, M - 3)} if (r > 1 and M >= 3) else None
        asym_in_ir = (M - 3, M - 2) in IR if M >= 3 else False
        if expect_P is not None and set(P) != expect_P:
            viol.append(("P", M, r))
        if r > 1 and M >= 3 and asym_in_ir != (M >= r + 3):
            viol.append(("IR", M, r))
    say(f"TD checks (r > 1, M >= 3: Pareto optima are (M,M), (M-1,M), (M,M-1); (M-1,M) individually rational iff "
        f"M >= r + 3): violations {viol}")

    say()
    say("== 2. Which concepts satisfy which properties")
    tally("strict 2x2", list(strict_2x2()))
    tally("2x2 with payoffs in {0,1,2}", list(ties_2x2()))
    tally("symmetric strict 2x2", list(sym_strict_2x2()), sym=True)
    tally("symmetric 2x2 with payoffs in {0,1,2}", list(sym_ties_2x2()), sym=True)
    tally("random symmetric 3x3 ranking games", list(sym_3x3(rng, 3000)), sym=True)
    tally("prisoner's dilemmas", list(pds()))
    tally("traveler's dilemmas", [g for _, g in tds()], sym=True)

    say()
    say("== 3. Named games")
    for name, (U, sh) in (("chicken (Y=0, A=1)", CHICKEN), ("G_e (Y=0, A=1)", GE)):
        say(f"{name}: Pareto-optimal {sorted(pareto(U))}; individually rational {sorted(ir_set(U, sh))}; "
            f"equal-payoff Pareto-optimal {sorted(s for s in pareto(U) if equal_payoff(U, s))}")
        for cname, f in CONCEPTS.items():
            E = f(U, sh)
            say(f"   {cname:9s}: {sorted(E)}  {props(U, sh, E, True)}")

    say()
    say("== 4. Symmetric games in which reasoning in turn breaks equal treatment")
    for name, games in (("symmetric strict 2x2", list(sym_strict_2x2())),
                        ("random symmetric 3x3 ranking games", list(sym_3x3(random.Random(20261103), 3000)))):
        with_eq = 0
        spp_break = 0
        rpp_break = 0
        pte_empty_among_break = 0
        for U, sh in games:
            P = pareto(U)
            Peq = frozenset(s for s in P if equal_payoff(U, s))
            if not Peq:
                continue
            with_eq += 1
            b1 = any(not (spp(U, sh, o) <= Peq) for o in ((0, 1), (1, 0)))
            b2 = any(not (rpp(U, sh, o) <= Peq) for o in ((0, 1), (1, 0)))
            spp_break += b1
            rpp_break += b2
            if b2 and not pte(U, sh):
                pte_empty_among_break += 1
        say(f"{name}: {len(games)} games, {with_eq} with an equal-payoff Pareto-optimal profile; equal treatment "
            f"broken by SPP (some order) in {spp_break}, by RPP in {rpp_break}, of which the PTE limit is empty in "
            f"{pte_empty_among_break}")

    say()
    say("== 5. Symmetric efficiency (SE): in a symmetric game in which some symmetric profile (x, x) is Pareto-optimal,")
    say("   every predicted profile is a Pareto-optimal symmetric profile (no comparison of payoffs across players)")
    classes = (("symmetric strict 2x2", list(sym_strict_2x2())),
               ("symmetric 2x2 with payoffs in {0,1,2}", list(sym_ties_2x2())),
               ("random symmetric 3x3 ranking games", list(sym_3x3(random.Random(20261103), 3000))),
               ("traveler's dilemmas", [g for _, g in tds()]),
               ("symmetric prisoner's dilemmas", [g for g in pds() if is_symmetric(*g)]))
    first_example = {}
    for name, games in classes:
        rel = [(U, sh) for U, sh in games if se_relevant(U, sh)]
        say(f"-- {name}: {len(games)} games, {len(rel)} with a Pareto-optimal symmetric profile; SE fails for:")
        row = []
        for cname, f in list(CONCEPTS.items()) + [("superrat", superrational)]:
            fails = 0
            nonempty_fails = 0
            for U, sh in rel:
                E = f(U, sh)
                if not se_holds(U, sh, E):
                    fails += 1
                    if E:
                        nonempty_fails += 1
                        first_example.setdefault((cname, name), (U, sh, E))
            row.append(f"{cname} {fails} ({nonempty_fails} nonempty)")
        say("   " + "; ".join(row))
    for key in (("PTE", "symmetric 2x2 with payoffs in {0,1,2}"), ("NB", "random symmetric 3x3 ranking games"),
                ("NB", "symmetric 2x2 with payoffs in {0,1,2}"), ("SPP12", "symmetric strict 2x2")):
        if key in first_example:
            U, sh, E = first_example[key]
            say(f"   first example, {key[0]} on {key[1]}: u = {[(s, U[s]) for s in sorted(U)]}; prediction {sorted(E)}; "
                f"Pareto-optimal symmetric profiles {sorted(s for s in pareto(U) if len(set(s)) == 1)}")
    say("   theorem check: in symmetric games without ties, is a nonempty PTE limit always a Pareto-optimal symmetric "
        "profile?")
    for name, games in classes:
        bad = [g for g in games if not has_ties(*g) and pte(*g) and not (len(pte(*g)) == 1 and
               next(iter(pte(*g))) in pareto(g[0]) and len(set(next(iter(pte(*g))))) == 1)]
        say(f"   {name}: counterexamples {len(bad)}")

    say()
    say("== 6. Chicken of Aumann (Y = 0, A = 1): every reachable set (steps E -> intersection of the verdicts of a")
    say("   nonempty set of players)")
    U, sh = CHICKEN
    R = reach(U, sh)
    rest = [E for E in R if all(verdict(U, E, i, sh) == E for i in range(2))]
    say(f"   {len(R)} reachable sets: {[sorted(E) for E in sorted(R, key=lambda E: (len(E), sorted(E)))]}")
    say(f"   reachable rest points: {[sorted(E) for E in sorted(rest, key=lambda E: (len(E), sorted(E)))]}")
    say(f"   is {{(Y,Y)}} reachable: {frozenset({(0, 0)}) in R}; SE requires the prediction to be {{(Y,Y)}}")

    say()
    say("== 7. Equal treatment in symmetric games with ties: nonempty PTE limits that break it (first 6)")
    k = 0
    for U, sh in sym_ties_2x2():
        E = pte(U, sh)
        P = pareto(U)
        Peq = frozenset(s for s in P if equal_payoff(U, s))
        if E and Peq and not E <= Peq:
            k += 1
            if k <= 6:
                say(f"   u = {[U[s] for s in sorted(U)]} (profiles (0,0), (0,1), (1,0), (1,1)); PTE limit {sorted(E)}; "
                    f"Pareto-optimal {sorted(P)}; equal-payoff Pareto-optimal {sorted(Peq)}")
    say(f"   total: {k} of 81")

    OUT.write_text("\n".join(_lines) + "\n")


if __name__ == "__main__":
    main()
