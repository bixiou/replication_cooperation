"""Further concepts of the typology of the paper (Table 1 and Online Appendix, Section S7), 2026-11-03.

Summary of the logic (written before the code).

Games, pure strategies unless stated. Prisoner's dilemma PD(T,R,P,S), strategies C, D; instances (5,3,1,0) (2R > T+S) and
(7,3,1,0) (2R < T+S). Traveler's dilemma TD(M,r): claims 2..M, equal claims paid their value, otherwise both paid the lower
claim x, plus r to the lower claimant and minus r to the higher one; M = 10 with r = 2 and r = 5 (second reward), plus
M = 100, r = 2 for Apt and Schafer's Proposition 8. G0: payoffs (1,1) at (a,b), (2,1) at (a,b'), (0,0) at (a',b) and
(a',b'); (a,b) is in U(G0). Also matching pennies, the paper's chicken and G_e (e = 1/2), and test families: all 65,536
2x2 games with payoffs in {0,1,2,3}, the 576 strict 2x2 games (pairs of strict rankings), random 3x3 games.
U(G): profiles at which no player has another strategy whose best payoff exceeds her current payoff; IR(G): profiles
giving each player at least her pure security level max_t min_y u_i(t,y). A concept is unilateral iff U <= phi <= IR.

1. Similarity-based cooperative equilibrium (Oesterheld et al. 2023): outcomes over all diff functions are the profiles
   giving each player at least her mixed minimax (Theorem 3). In PD and TD the pure maximin equals the pure minimax, so
   the mixed minimax is that common value; we list the pure outcome set and check it contains the pure Nash equilibria
   (Proposition 16: base-game equilibria survive for every diff function).
2. Evidential equilibrium (al-Nowaihi and Dhami 2015): (a) identity social projection function (Definition 6), which needs
   a bijection between the players' strategy sets: player i maximizes u_i(x, phi(x)); consistent = the predicted profile
   is on the diagonal (Definitions 13-14). PD, TD, G0 under its two bijections, and a coordination game in which no
   consistent EE exists (pure, and mixed on a grid). (b) all pure SPFs: the union of consistent EE, by brute force over
   all SPF pairs in PD and TD(M=5), compared with IR(G).
3. Magical thinking (Daley and Sadowski 2017), F uniform on [0,1]: PD cutoff from equation (2) by bisection; TD via the
   Section 5.2 model V(k) = alpha s(k) + (1-alpha) sum_k' P(k') v(k,k'): the population distribution P is a fixed point
   of P -> (measure of types choosing k), computed by damped iteration with the exact upper envelope of the lines in alpha.
4. Selfishness level (Apt and Schafer 2014): for each social optimum s, the least alpha making s a pure Nash equilibrium
   of G(alpha), r_i = u_i + alpha SW; the level is the minimum over optima. Induced selection: the optima attaining it.
5. Nash-2 equilibrium (Sandomirskaia 2019), two players, complete reflection network (Definitions 1-2): no profitable
   deviation after which every profitable reply of the other leaves the deviator at least her initial payoff.
6. Farsighted stability (Chwe 1994; Suzuki and Muto 2000; Nakanishi 2009): indirect domination with individual moves,
   and with coalitional moves; the largest consistent set by Chwe's iteration of f from the set of all profiles;
   farsighted (vNM) stable sets by exhaustive kernel search.
7. Berge equilibrium (Haller 2024): the TD and PD sets, G0, Haller's three-player example (no pure Berge equilibrium).
8. Perfect prediction equilibrium (Fourny, Reiche and Dupuy): the sequential version of a normal-form game in which the
   players move in the order pi, each seeing earlier moves; the PPE by the quick algorithm (their Section 7.2.2) and by
   the iterated first principle plus the second principle (their Principles 1-2); compared with the 'exhaustive'
   schedule (each player applies Fourny's verdict to a fixed point before the next) and with the paper's sequential
   perfect prediction (one verdict per player per pass). Individual rationality of the PPE in the normal form.
Each section prints what it finds; nothing is asserted that the printout does not show.

Usage: python3 research/scripts/coop_other_concepts.py   (writes the .out file beside it)
"""
import itertools
import random
import sys
from fractions import Fraction as Fr
from pathlib import Path

OUT = Path(__file__).with_suffix(".out")
_lines = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    _lines.append(s)


# ----------------------------------------------------------------------------------------------------------------------
# games
# ----------------------------------------------------------------------------------------------------------------------
class Game:
    def __init__(self, strats, pay, name=""):
        self.S = [list(x) for x in strats]
        self.n = len(strats)
        self.P = list(itertools.product(*self.S))
        self.u = {s: tuple(pay(s)) if callable(pay) else tuple(pay[s]) for s in self.P}
        self.name = name


def pd(T, R, P, S):
    t = {("C", "C"): (R, R), ("C", "D"): (S, T), ("D", "C"): (T, S), ("D", "D"): (P, P)}
    return Game([["C", "D"], ["C", "D"]], t, f"PD{(T, R, P, S)}")


def td_pay(x, y, r):
    if x == y:
        return (x, x)
    if x < y:
        return (x + r, x - r)
    return (y - r, y + r)


def td(M, r):
    return Game([list(range(2, M + 1))] * 2, lambda s: td_pay(s[0], s[1], r), f"TD(M={M},r={r})")


G0 = Game([["a", "a'"], ["b", "b'"]], {("a", "b"): (1, 1), ("a", "b'"): (2, 1), ("a'", "b"): (0, 0), ("a'", "b'"): (0, 0)},
          "G0")
MP = Game([["H", "T"], ["H", "T"]], {("H", "H"): (1, -1), ("H", "T"): (-1, 1), ("T", "H"): (-1, 1), ("T", "T"): (1, -1)},
          "matching pennies")
CHICKEN = Game([["A", "Y"], ["A", "Y"]], {("A", "A"): (0, 0), ("A", "Y"): (3, 1), ("Y", "A"): (1, 3), ("Y", "Y"): (2, 2)},
               "chicken")
GE = Game([["A", "Y"], ["A", "Y"]], {("A", "A"): (0, 0), ("A", "Y"): (3, 1), ("Y", "A"): (1, 3), ("Y", "Y"): (Fr(1, 2),) * 2},
          "G_e(e=1/2)")


def section(g, i, t, E=None):
    pool = g.P if E is None else E
    return [s for s in pool if s[i] == t]


def security(g, i):
    return max(min(g.u[s][i] for s in section(g, i, t)) for t in g.S[i])


def pure_minimax2(g, i):  # two players only
    j = 1 - i
    best = None
    for y in g.S[j]:
        m = max(g.u[s][i] for s in g.P if s[j] == y)
        best = m if best is None else min(best, m)
    return best


def IR(g):
    v = [security(g, i) for i in range(g.n)]
    return {s for s in g.P if all(g.u[s][i] >= v[i] for i in range(g.n))}


def U(g):
    out = set()
    for s in g.P:
        ok = True
        for i in range(g.n):
            for t in g.S[i]:
                if t != s[i] and max(g.u[x][i] for x in section(g, i, t)) > g.u[s][i]:
                    ok = False
        if ok:
            out.add(s)
    return out


def dev(s, i, t):
    s = list(s)
    s[i] = t
    return tuple(s)


def pure_ne(g):
    return {s for s in g.P if all(g.u[s][i] >= g.u[dev(s, i, t)][i] for i in range(g.n) for t in g.S[i])}


def unilateral_report(g, phi):
    u, ir = U(g), IR(g)
    return f"U<=phi: {u <= phi} (U={sorted(u)}), phi<=IR: {phi <= ir}"


def show(E, limit=40):
    E = sorted(E)
    return f"{len(E)} profiles: {E if len(E) <= limit else str(E[:limit]) + ' ...'}"


# ----------------------------------------------------------------------------------------------------------------------
# 1. Similarity-based cooperative equilibrium
# ----------------------------------------------------------------------------------------------------------------------
def sbce_union(g):
    v = []
    for i in range(2):
        lo, hi = security(g, i), pure_minimax2(g, i)
        assert lo == hi, "mixed minimax not pinned down by the pure bounds"
        v.append(lo)
    return {s for s in g.P if all(g.u[s][i] >= v[i] for i in range(2))}, v


# ----------------------------------------------------------------------------------------------------------------------
# 2. Evidential equilibrium
# ----------------------------------------------------------------------------------------------------------------------
def ee_identity(g, phi):
    """phi: dict from player 1's strategies onto player 2's (a bijection)."""
    inv = {b: a for a, b in phi.items()}
    v1 = {a: g.u[(a, phi[a])][0] for a in g.S[0]}
    v2 = {b: g.u[(inv[b], b)][1] for b in g.S[1]}
    A = [a for a in g.S[0] if v1[a] == max(v1.values())]
    B = [b for b in g.S[1] if v2[b] == max(v2.values())]
    ee = set(itertools.product(A, B))
    consistent = {s for s in ee if phi[s[0]] == s[1]}
    return ee, consistent


def ee_all_spfs(g):
    """Union over all pure SPF pairs of the consistent evidential equilibria (brute force)."""
    S1, S2 = g.S
    out = set()
    for f1 in itertools.product(S2, repeat=len(S1)):
        P1 = dict(zip(S1, f1))
        v1 = {a: g.u[(a, P1[a])][0] for a in S1}
        A = [a for a in S1 if v1[a] == max(v1.values())]
        for f2 in itertools.product(S1, repeat=len(S2)):
            P2 = dict(zip(S2, f2))
            v2 = {b: g.u[(P2[b], b)][1] for b in S2}
            for a in A:
                for b in S2:
                    if v2[b] == max(v2.values()) and P1[a] == b and P2[b] == a:
                        out.add((a, b))
    return out


# ----------------------------------------------------------------------------------------------------------------------
# 3. Magical thinking (F uniform)
# ----------------------------------------------------------------------------------------------------------------------
def mt_pd_cutoff(T, R, P, S):
    x, y = Fr(T - R, R - P), Fr(P - S, R - P)  # normalized game (1, 0, x, y), Lemma 1
    x, y = float(x), float(y)

    def g(a):  # V(c) - V(d) at the cutoff type, F(a) = a
        F = a
        vc = a * 1 + (1 - a) * (F * (-y) + (1 - F) * 1)
        vd = a * 0 + (1 - a) * (F * 0 + (1 - F) * (1 + x))
        return vc - vd

    lo, hi = 1e-12, 1 - 1e-12
    assert g(lo) < 0 < g(hi)
    for _ in range(200):
        mid = (lo + hi) / 2
        if g(mid) < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def envelope_mass(svals, W):
    """Lebesgue measure of the types alpha in [0,1] for which action k maximizes alpha s(k) + (1-alpha) W(k)."""
    K = len(svals)
    a = [W[k] for k in range(K)]
    b = [svals[k] - W[k] for k in range(K)]
    pts = {0.0, 1.0}
    for j in range(K):
        for k in range(j + 1, K):
            if b[j] != b[k]:
                t = (a[j] - a[k]) / (b[k] - b[j])
                if 0 < t < 1:
                    pts.add(t)
    pts = sorted(pts)
    mass = [0.0] * K
    for lo, hi in zip(pts, pts[1:]):
        m = (lo + hi) / 2
        k = max(range(K), key=lambda k: a[k] + b[k] * m)
        mass[k] += hi - lo
    return mass


def mt_equilibrium(svals, v, lam=0.02, iters=20000):
    K = len(svals)
    P = [1.0 / K] * K
    for _ in range(iters):
        W = [sum(P[kk] * v[k][kk] for kk in range(K)) for k in range(K)]
        Q = envelope_mass(svals, W)
        P = [(1 - lam) * P[k] + lam * Q[k] for k in range(K)]
    W = [sum(P[kk] * v[k][kk] for kk in range(K)) for k in range(K)]
    Q = envelope_mass(svals, W)
    return P, max(abs(P[k] - Q[k]) for k in range(K))


# ----------------------------------------------------------------------------------------------------------------------
# 4. Selfishness level
# ----------------------------------------------------------------------------------------------------------------------
INF = float("inf")


def selfishness(g):
    SW = {s: sum(g.u[s]) for s in g.P}
    best = max(SW.values())
    opt = [s for s in g.P if SW[s] == best]
    need = {}
    for s in opt:
        a = Fr(0)
        for i in range(g.n):
            for t in g.S[i]:
                d = dev(s, i, t)
                gain = g.u[d][i] - g.u[s][i]
                if gain > 0:
                    a = INF if SW[d] == SW[s] else max(a, Fr(gain) / Fr(SW[s] - SW[d]))
        need[s] = a
    lev = min(need.values())
    return lev, {s for s in opt if need[s] == lev}, need


def is_ne_of_altruistic(g, s, alpha):
    r = {x: tuple(g.u[x][i] + alpha * sum(g.u[x]) for i in range(g.n)) for x in g.P}
    return all(r[s][i] >= r[dev(s, i, t)][i] for i in range(g.n) for t in g.S[i])


# ----------------------------------------------------------------------------------------------------------------------
# 5. Nash-2 equilibrium (two players, complete reflection network)
# ----------------------------------------------------------------------------------------------------------------------
def nash2(g):
    out = set()
    for s in g.P:
        bad = False
        for i in range(2):
            j = 1 - i
            for t in g.S[i]:
                d = dev(s, i, t)
                if g.u[d][i] <= g.u[s][i]:
                    continue  # not profitable
                improving = all(g.u[dev(d, j, y)][i] >= g.u[s][i] for y in g.S[j] if g.u[dev(d, j, y)][j] > g.u[d][j])
                if improving:
                    bad = True
        if not bad:
            out.add(s)
    return out


# ----------------------------------------------------------------------------------------------------------------------
# 6. Farsighted stability
# ----------------------------------------------------------------------------------------------------------------------
def moves_of(g, a, coalitional):
    res = []
    players = range(g.n)
    coalitions = [(i,) for i in players]
    if coalitional:
        coalitions = [c for k in range(1, g.n + 1) for c in itertools.combinations(players, k)]
    seen = set()
    for c in coalitions:
        for choice in itertools.product(*[g.S[i] for i in c]):
            d = list(a)
            for i, t in zip(c, choice):
                d[i] = t
            d = tuple(d)
            if d != a and (c, d) not in seen:
                seen.add((c, d))
                res.append((c, d))
    return res


def indirect_dominance(g, coalitional):
    """dom[e] = set of d with e >> d (Chwe 1994; Suzuki and Muto 2000; Nakanishi 2009 for individual moves)."""
    mv = {a: moves_of(g, a, coalitional) for a in g.P}
    dom = {}
    for e in g.P:
        B = set()
        changed = True
        while changed:
            changed = False
            for a in g.P:
                if a in B:
                    continue
                for c, d in mv[a]:
                    if (d == e or d in B) and all(g.u[e][i] > g.u[a][i] for i in c):
                        B.add(a)
                        changed = True
                        break
        dom[e] = B
    return dom, mv


def lcs(g, coalitional):
    dom, mv = indirect_dominance(g, coalitional)
    reach = {d: {d} | {e for e in g.P if d in dom[e]} for d in g.P}
    Z = set(g.P)
    while True:
        Z2 = set()
        for a in g.P:
            ok = True
            for c, d in mv[a]:
                if not any(not all(g.u[e][i] > g.u[a][i] for i in c) for e in reach[d] & Z):
                    ok = False
                    break
            if ok:
                Z2.add(a)
        if Z2 == Z:
            return Z, dom
        Z = Z2


def stable_sets(g, dom, cap=500, budget=2_000_000):
    """All farsighted stable sets K: internally stable (no a, b in K with a >> b) and externally stable."""
    nodes = list(g.P)
    dominators = {b: {a for a in nodes if b in dom[a]} for b in nodes}
    conflict = {a: (dominators[a] | dom[a]) - {a} for a in nodes}
    sols, steps = [], [0]

    def propagate(asg):
        stack = list(asg.items())
        while True:
            changed = False
            for b in nodes:
                if b in asg and asg[b]:
                    for c in conflict[b]:
                        if asg.get(c) is True:
                            return None
                        if c not in asg:
                            asg[c] = False
                            changed = True
            for b in nodes:
                if asg.get(b) is False:
                    doms = dominators[b]
                    if any(asg.get(a) is True for a in doms):
                        continue
                    free = [a for a in doms if a not in asg]
                    if not free:
                        return None
                    if len(free) == 1:
                        asg[free[0]] = True
                        changed = True
                elif b not in asg and not dominators[b]:
                    asg[b] = True
                    changed = True
            if not changed:
                return asg

    def rec(asg):
        steps[0] += 1
        if len(sols) >= cap or steps[0] > budget:
            return
        asg = propagate(dict(asg))
        if asg is None:
            return
        free = [b for b in nodes if b not in asg]
        if not free:
            sols.append(frozenset(b for b in nodes if asg[b]))
            return
        b = free[0]
        for val in (True, False):
            a2 = dict(asg)
            a2[b] = val
            rec(a2)

    rec({})
    exhausted = len(sols) < cap and steps[0] <= budget
    return sols, exhausted


def is_stable_set(g, dom, K):
    K = set(K)
    internal = not any(b in dom[a] for a in K for b in K)
    external = all(any(b in dom[a] for a in K) for b in g.P if b not in K)
    return internal and external


# ----------------------------------------------------------------------------------------------------------------------
# 7. Berge equilibrium
# ----------------------------------------------------------------------------------------------------------------------
def berge(g):
    out = set()
    for s in g.P:
        ok = True
        for i in range(g.n):
            for x in section(g, i, s[i]):
                if g.u[x][i] > g.u[s][i]:
                    ok = False
        if ok:
            out.add(s)
    return out


# ----------------------------------------------------------------------------------------------------------------------
# 8. Perfect prediction: PPE of the sequential game, Fourny's verdicts
# ----------------------------------------------------------------------------------------------------------------------
def verdict(g, E, i):
    if not E:
        return E
    m = max(min(g.u[s][i] for s in sec) for t in g.S[i] if (sec := section(g, i, t, E)))
    return {s for s in E if g.u[s][i] >= m}


def ppe_quick(g, order):
    I = set(g.P)
    for p in order:
        subs = {t: {o for o in I if o[p] == t} for t in g.S[p]}
        subs = {t: v for t, v in subs.items() if v}
        top = max(g.u[o][p] for o in I)
        holders = [t for t, v in subs.items() if any(g.u[o][p] == top for o in v)]
        if len(holders) != 1:
            return None
        t0 = holders[0]
        others = [o for t, v in subs.items() if t != t0 for o in v]
        if others:
            Mv = max(g.u[o][p] for o in others)
            I = {o for o in subs[t0] if g.u[o][p] >= Mv}
        else:
            I = subs[t0]
    return I


def ppe_principles(g, order):
    """Principle 1 iterated at the current node, then Principle 2; None if Principle 2 cannot choose (ties)."""
    I = set(g.P)
    for p in order:
        while True:
            I2 = verdict(g, I, p)
            if I2 == I:
                break
            I = I2
        subs = {t: v for t in g.S[p] if (v := {o for o in I if o[p] == t})}
        if len(subs) == 1:
            I = next(iter(subs.values()))
            continue
        chosen = [t for t, v in subs.items()
                  if min(g.u[o][p] for o in v) > max(g.u[o][p] for tt, w in subs.items() if tt != t for o in w)]
        if len(chosen) != 1:
            return None, I
        I = subs[chosen[0]]
    return I, I


def exhaustive(g, order):
    E = set(g.P)
    while True:
        E0 = set(E)
        for p in order:
            while True:
                E2 = verdict(g, E, p)
                if E2 == E:
                    break
                E = E2
        if E == E0:
            return E


def sequential_pp(g, order):  # the paper's phi^S_pi: one verdict per player per pass
    E = set(g.P)
    while True:
        E0 = set(E)
        for p in order:
            E = verdict(g, E, p)
        if E == E0:
            return E


def pte(g):
    E = set(g.P)
    while True:
        if not E:
            return E
        m = [max(min(g.u[s][i] for s in sec) for t in g.S[i] if (sec := section(g, i, t, E))) for i in range(g.n)]
        E2 = {s for s in E if all(g.u[s][i] >= m[i] for i in range(g.n))}
        if E2 == E:
            return E
        E = E2


def strict_2x2_games():
    perms = list(itertools.permutations(range(4)))
    cells = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for p1 in perms:
        for p2 in perms:
            yield Game([[0, 1], [0, 1]], {c: (p1[k], p2[k]) for k, c in enumerate(cells)})


def all_2x2_games_0123():
    cells = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for vals in itertools.product(range(4), repeat=8):
        yield Game([[0, 1], [0, 1]], {c: (vals[2 * k], vals[2 * k + 1]) for k, c in enumerate(cells)})


def random_game(rng, shape, values=None, strict=False):
    strats = [list(range(k)) for k in shape]
    P = list(itertools.product(*strats))
    if strict:
        cols = [rng.sample(range(len(P)), len(P)) for _ in shape]
        pay = {s: tuple(cols[i][k] for i in range(len(shape))) for k, s in enumerate(P)}
    else:
        pay = {s: tuple(rng.choice(values) for _ in shape) for s in P}
    return Game(strats, pay)


# ----------------------------------------------------------------------------------------------------------------------
def main():
    rng = random.Random(20261103)
    PD1, PD2 = pd(5, 3, 1, 0), pd(7, 3, 1, 0)
    TDS = [td(10, 2), td(10, 5)]
    say("coop_other_concepts.py: further concepts of the typology, pure strategies unless stated.")
    say("G0: U(G0) =", sorted(U(G0)), " IR(G0) =", sorted(IR(G0)))
    for g in [PD1, PD2] + TDS:
        say(f"{g.name}: security levels {[security(g, i) for i in range(2)]}, pure NE {sorted(pure_ne(g))}, "
            f"U = {sorted(U(g))}, |IR| = {len(IR(g))}")

    say("\n== 1. Similarity-based cooperative equilibrium (Oesterheld et al. 2023, Theorem 3, Proposition 16)")
    for g in [PD1] + TDS:
        E, v = sbce_union(g)
        say(f"{g.name}: mixed minimax {v} (pure maximin = pure minimax); union over diff functions, pure part: "
            f"{show(E, 12)}; = IR(G): {E == IR(g)}; contains pure NE: {pure_ne(g) <= E}")

    say("\n== 2. Evidential equilibrium (al-Nowaihi and Dhami 2015)")
    ident = {"C": "C", "D": "D"}
    for g in [PD1, PD2]:
        ee, ce = ee_identity(g, ident)
        say(f"{g.name}, identity SPF: EE {sorted(ee)}, consistent {sorted(ce)}")
    for g in TDS:
        ee, ce = ee_identity(g, {x: x for x in g.S[0]})
        say(f"{g.name}, identity SPF: EE {sorted(ee)}, consistent {sorted(ce)}")
    for lab, phi in [("a<->b, a'<->b'", {"a": "b", "a'": "b'"}), ("a<->b', a'<->b", {"a": "b'", "a'": "b"})]:
        ee, ce = ee_identity(G0, phi)
        say(f"G0 with the identification {lab}: EE {sorted(ee)}, consistent {sorted(ce)}; {unilateral_report(G0, ee)}")
    CO = Game([["a", "b"], ["a", "b"]], {("a", "a"): (1, 0), ("b", "b"): (0, 1), ("a", "b"): (0, 0), ("b", "a"): (0, 0)},
              "coordination (1,0)/(0,1) on the diagonal")
    ee, ce = ee_identity(CO, {"a": "a", "b": "b"})
    say(f"{CO.name}, identity SPF, pure: EE {sorted(ee)}, consistent {sorted(ce)}")
    grid = [Fr(k, 1000) for k in range(1001)]
    p1 = max(grid, key=lambda p: p * p)  # pi_1(p,p): both play a with prob p -> payoff 1 * p * p
    p2 = max(grid, key=lambda q: (1 - q) * (1 - q))  # pi_2(q,q) = (1-q)^2, q = prob of a
    say(f"  mixed, grid of step 1/1000: player 1's optimum p(a) = {p1}, player 2's optimum q(a) = {p2}; "
        f"a consistent EE needs p = q: {p1 == p2}")
    for g in [PD1, td(5, 2)]:
        E = ee_all_spfs(g)
        say(f"{g.name}: union over all pure SPF pairs of consistent EE: {show(E, 20)}; equals IR(G): {E == IR(g)}")

    say("\n== 3. Magical thinking (Daley and Sadowski 2017), F uniform on [0,1]")
    for (T, R, P, S) in [(5, 3, 1, 0), (7, 3, 1, 0)]:
        a = mt_pd_cutoff(T, R, P, S)
        say(f"PD{(T, R, P, S)}: cutoff alpha* = {a:.6f}; Pr(C,C) = {(1 - a) ** 2:.4f}, Pr(D,D) = {a * a:.4f}, "
            f"Pr(one cooperates alone) = {2 * a * (1 - a):.4f}")
        sv = [P, R]  # actions ordered D (0), C (1): s(D) = P, s(C) = R
        v = [[P, T], [S, R]]  # v[k][k']: D vs D = P, D vs C = T; C vs D = S, C vs C = R
        Pd, res = mt_equilibrium([float(x) for x in sv], [[float(x) for x in row] for row in v])
        say(f"  cross-check with the Section 5.2 fixed point: Pr(D) = {Pd[0]:.6f} (residual {res:.1e})")
    for g in TDS:
        claims = g.S[0]
        sv = [float(k) for k in claims]
        v = [[float(g.u[(k, kk)][0]) for kk in claims] for k in claims]
        Pt, res = mt_equilibrium(sv, v)
        Pt2, res2 = mt_equilibrium(sv, v, lam=0.005, iters=60000)
        say(f"{g.name}: equilibrium distribution of claims (residual {res:.1e}; a second run with damping 0.005 differs by "
            f"at most {max(abs(x - y) for x, y in zip(Pt, Pt2)):.1e}, residual {res2:.1e}):")
        say("  " + ", ".join(f"{k}:{p:.4f}" for k, p in zip(claims, Pt)))
        say(f"  Pr(M,M) = {Pt[-1] ** 2:.4f}; Pr(2,2) = {Pt[0] ** 2:.4f}; mean claim = {sum(k * p for k, p in zip(claims, Pt)):.3f}")

    say("\n== 4. Selfishness level (Apt and Schafer 2014)")
    for g in [PD1, PD2] + TDS + [td(100, 2), td(100, 5), MP, G0, CHICKEN]:
        lev, sel, need = selfishness(g)
        extra = ""
        if lev not in (INF,):
            ok = all(is_ne_of_altruistic(g, s, lev) for s in sel)
            extra = f"; selected optima are pure NE of G(level): {ok}"
        say(f"{g.name}: selfishness level {lev}; social optima with their thresholds "
            f"{ {s: str(a) for s, a in need.items()} if len(need) <= 6 else len(need)}; induced selection {sorted(sel) if lev != INF else 'none'}{extra}")
    lev, sel, _ = selfishness(G0)
    say(f"G0: induced selection {sorted(sel)}; {unilateral_report(G0, sel)}")

    say("\n== 5. Nash-2 equilibrium (Sandomirskaia 2019), complete reflection network")
    for g in [PD1, PD2] + TDS + [MP, G0, CHICKEN]:
        E = nash2(g)
        say(f"{g.name}: {show(E, 60)}; {unilateral_report(g, E)}")
    empty, viol, empty_strict = 0, 0, 0
    for g in all_2x2_games_0123():
        E = nash2(g)
        if not E:
            empty += 1
        if not (U(g) <= E <= IR(g)):
            viol += 1
    for g in strict_2x2_games():
        if not nash2(g):
            empty_strict += 1
    say(f"all 65,536 2x2 games with payoffs in 0..3: {empty} without a Nash-2 equilibrium; {viol} violations of U <= NE-2 <= IR. "
        f"Strict 2x2 games: {empty_strict} of 576 without one.")

    say("\n== 6. Farsighted stability: indirect domination (Chwe 1994), largest consistent set, farsighted stable sets")
    for g in [PD1, PD2]:
        for coal in (False, True):
            Z, dom = lcs(g, coal)
            sols, ex = stable_sets(g, dom)
            say(f"{g.name}, {'coalitional' if coal else 'individual'} moves: LCS {sorted(Z)}; stable sets "
                f"{[sorted(K) for K in sols]} (search complete: {ex})")
    for g in TDS:
        M = g.S[0][-1]
        for coal in (False, True):
            Z, dom = lcs(g, coal)
            sols, ex = stable_sets(g, dom, cap=50, budget=400_000)
            say(f"{g.name}, {'coalitional' if coal else 'individual'} moves: LCS {show(Z, 100)}")
            say(f"  (M,M) in LCS: {(M, M) in Z}; (M-1,M) in LCS: {(M - 1, M) in Z}; (2,2) in LCS: {(2, 2) in Z}; "
                f"{unilateral_report(g, Z)}")
            say(f"  {{(M,M)}} is a farsighted stable set: {is_stable_set(g, dom, {(M, M)})}; stable sets found: {len(sols)} "
                f"(search complete: {ex}); first ones: {[sorted(K) for K in sols[:4]]}")
    for coal in (False, True):
        Z, dom = lcs(G0, coal)
        sols, ex = stable_sets(G0, dom)
        say(f"G0, {'coalitional' if coal else 'individual'} moves: LCS {sorted(Z)}; {unilateral_report(G0, Z)}; "
            f"stable sets {[sorted(K) for K in sols]}")
        Z, dom = lcs(MP, coal)
        sols, ex = stable_sets(MP, dom)
        say(f"matching pennies, {'coalitional' if coal else 'individual'} moves: LCS {sorted(Z)}; stable sets "
            f"{[sorted(K) for K in sols]} (search complete: {ex})")
    for coal in (False, True):
        cnt = {"lcs_empty": 0, "U_not_in": 0, "not_IR": 0, "no_stable": 0, "stable_not_unilateral": 0, "games": 0}
        for g in all_2x2_games_0123():
            cnt["games"] += 1
            Z, dom = lcs(g, coal)
            if not Z:
                cnt["lcs_empty"] += 1
            if not U(g) <= Z:
                cnt["U_not_in"] += 1
            if not Z <= IR(g):
                cnt["not_IR"] += 1
        for g in strict_2x2_games():
            Z, dom = lcs(g, coal)
            sols, ex = stable_sets(g, dom)
            if not sols:
                cnt["no_stable"] += 1
        say(f"{'coalitional' if coal else 'individual'} moves, all 65,536 2x2 games with payoffs 0..3: LCS empty in "
            f"{cnt['lcs_empty']}, U not in LCS in {cnt['U_not_in']}, LCS not in IR in {cnt['not_IR']}; "
            f"strict 2x2 games without a farsighted stable set: {cnt['no_stable']} of 576")
        bad = 0
        for _ in range(300):
            g = random_game(rng, (3, 3), values=list(range(4)))
            Z, _d = lcs(g, coal)
            if not (U(g) <= Z <= IR(g)) or not Z:
                bad += 1
        say(f"  random 3x3 games, payoffs uniform in 0..3, 300 draws: {bad} with an empty LCS or outside U <= LCS <= IR")

    G1 = Game([[0, 1], [0, 1]], {(0, 0): (0, 0), (0, 1): (0, 0), (1, 0): (0, 1), (1, 1): (1, 1)}, "G1")
    say("G1 (rows of player 1, columns of player 2): u(0,0) = u(0,1) = (0,0), u(1,0) = (0,1), u(1,1) = (1,1)")
    for coal in (False, True):
        Z, dom = lcs(G1, coal)
        sols, ex = stable_sets(G1, dom)
        say(f"  G1, {'coalitional' if coal else 'individual'} moves: U = {sorted(U(G1))}, LCS {sorted(Z)}; "
            f"(1,1) >> (0,0): {(0, 0) in dom[(1, 1)]}; stable sets {[sorted(K) for K in sols]}; {unilateral_report(G1, Z)}")
    for coal in (False, True):
        nos, nos_ex, first = 0, 0, None
        for g in all_2x2_games_0123():
            Z, dom = lcs(g, coal)
            sols, ex = stable_sets(g, dom)
            if not sols:
                nos += 1
                first = first or g.u
        say(f"{'coalitional' if coal else 'individual'} moves: 2x2 games with payoffs 0..3 without a farsighted stable set: "
            f"{nos} of 65,536; first one: {first}")
        nos, first = 0, None
        for _ in range(300):
            g = random_game(rng, (3, 3), values=list(range(4)))
            Z, dom = lcs(g, coal)
            sols, ex = stable_sets(g, dom)
            if not sols and ex:
                nos += 1
                first = first or g.u
        say(f"  random 3x3 games, payoffs 0..3, 300 draws: {nos} without a farsighted stable set (complete searches); first: {first}")

    say("\n== 7. Berge equilibrium (Haller 2024)")
    for g in [PD1] + TDS + [MP, G0]:
        E = berge(g)
        say(f"{g.name}: Berge {sorted(E)}; {unilateral_report(g, E)}")
    H = Game([[0, 1]] * 3, lambda s: (-(s[1] - s[2]) ** 2, s[0] - s[2], s[0] + s[1]), "Haller p. 857 three-player game")
    say(f"{H.name}: pure Berge equilibria {sorted(berge(H))}")
    HI = Game([["OUT", "IN"], ["OUT", "IN"]], {("OUT", "OUT"): (4, 4), ("OUT", "IN"): (8, 2), ("IN", "OUT"): (2, 8),
                                               ("IN", "IN"): (3, 3)}, "Haller p. 845 game (I), the dual of a PD")
    E = berge(HI)
    say(f"{HI.name}: Berge {sorted(E)} with payoffs {[HI.u[x] for x in sorted(E)]}; security levels "
        f"{[security(HI, i) for i in range(2)]}; {unilateral_report(HI, E)}")
    for g in [PD1, PD2] + TDS:
        coop = Game(g.S, {x: (sum(g.u[x]),) * 2 for x in g.P}, g.name + " with v_i = u_1 + u_2")
        say(f"{g.name}, cooperative orientation v_i = u_i + u_j (Haller p. 849): pure NE {show(pure_ne(coop), 12)}")

    say("\n== 8. Perfect prediction equilibrium (Fourny, Reiche and Dupuy) of the sequential game, order pi")
    for g in [PD1, PD2, CHICKEN, GE]:
        for order in ([0, 1], [1, 0]):
            q = ppe_quick(g, order)
            p, _ = ppe_principles(g, order)
            say(f"{g.name}, order {order}: PPE quick {sorted(q) if q else q}, principles {sorted(p) if p else p}, "
                f"exhaustive {sorted(exhaustive(g, order))}, sequential pp (paper) {sorted(sequential_pp(g, order))}, "
                f"PTE limit {sorted(pte(g))}; IR {sorted(IR(g))}")
    for g in TDS:
        for order in ([0, 1], [1, 0]):
            p, stuck = ppe_principles(g, order)
            say(f"{g.name}, order {order}: PPE by the principles: {sorted(p) if p else 'second principle cannot choose (ties)'}"
                f"{'' if p else '; remaining ' + show(stuck, 12)}; exhaustive {sorted(exhaustive(g, order))}; "
                f"sequential pp {sorted(sequential_pp(g, order))}")
    stats = {"runs": 0, "agree": 0, "ppe_none": 0, "not_IR": 0, "eq_seq": 0, "pte_nonempty": 0, "ppe_eq_pte": 0}
    for g in strict_2x2_games():
        ir = IR(g)
        pt = pte(g)
        for order in ([0, 1], [1, 0]):
            stats["runs"] += 1
            q, (p, _) = ppe_quick(g, order), ppe_principles(g, order)
            x = exhaustive(g, order)
            if q is None or p is None:
                stats["ppe_none"] += 1
                continue
            if q == p == x and len(q) == 1:
                stats["agree"] += 1
            if not q <= ir:
                stats["not_IR"] += 1
            if q == sequential_pp(g, order):
                stats["eq_seq"] += 1
            if pt:
                stats["pte_nonempty"] += 1
                if q == pt:
                    stats["ppe_eq_pte"] += 1
    say(f"576 strict 2x2 games x 2 orders: {stats}")
    for shape, orders, ngames in [((3, 3), [[0, 1], [1, 0]], 500), ((2, 2, 2), [list(o) for o in itertools.permutations(range(3))], 200)]:
        st = {"runs": 0, "agree": 0, "not_IR": 0, "eq_seq": 0}
        for _ in range(ngames):
            g = random_game(rng, shape, strict=True)
            ir = IR(g)
            for order in orders:
                st["runs"] += 1
                q, (p, _) = ppe_quick(g, order), ppe_principles(g, order)
                if q is not None and q == p == exhaustive(g, order) and len(q) == 1:
                    st["agree"] += 1
                if q is not None and not q <= ir:
                    st["not_IR"] += 1
                if q is not None and q == sequential_pp(g, order):
                    st["eq_seq"] += 1
        say(f"random strict games {shape}, {ngames} games, all orders: {st}")

    OUT.write_text("\n".join(_lines) + "\n")


if __name__ == "__main__":
    main()
