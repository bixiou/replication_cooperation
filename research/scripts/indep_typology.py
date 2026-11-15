"""Independent checks of Table 1, the credible alpha-core (Sections 6.1 and S6), and Sections 6.3 and S7 of
the paper. Implemented separately from the paper's own programs; uses only
indep_typology_lib.py. Run: uv run research/scripts/indep_typology.py

Logic summary.
Part A recomputes every worked example the typology rests on (the game in which
seven concepts drop a profile of U(G), Z_PD, the Berge examples, matching pennies,
chicken, the prisoner's dilemma with 2R > T+S and with T+S > 2R, the measure of
mutual advantage examples, the Nash product counterexample, G_e, the farsighted
examples, the perturbed traveler's dilemma, the traveler's dilemma family).
Part B adds constructed tests: a three-player game with an empty pure alpha-core,
a truncated Kantian game with no equilibrium, the similarity-based / program
equilibrium mixed profile of the prisoner's dilemma judged on its support.
Part C runs exhaustive checks on the 576 strict 2x2 games (Pareto optimality,
U(G) <= phi(G) <= IR(G)) and on the 65,536 2x2 games with payoffs in {0,1,2,3}
(the inclusions, which is where ties can break them), including Chwe's largest
consistent set. Part D does the mixed-strategy computations (superrationality,
maximum perfect cooperative equilibrium, measure of mutual advantage, Kantian
equilibrium in the mixed extension of the prisoner's dilemma, magical thinking).
"""
import math
import random
from fractions import Fraction as F
from itertools import product, permutations

from indep_typology_lib import (
    U_set, IR_set, security, has_no_ties, pareto_optimal, profiles, pay,
    is_symmetric, copying_pays, best_symmetric, nash_pure, berge, strong_eq,
    alpha_core, credible_alpha_core, nash2, optimin, mg, nb, ma_security, team,
    superrational, pce_pure, pte_limit, truncation, rpp, spp, union_orders,
    pte_rounds, lcs, indirect_dom, farsighted_stable_sets, strict_2x2_games,
    all_2x2_games_0123, traveler, pd, completely_self_consistent, is_rest_point,
    br1, br2,
)


def show(title, value):
    print(f"{title}: {value}")


def name2(s, rows="aa'", cols=None):
    return s


print("=" * 78)
print("PART A. Worked examples of Section 8 and S7")
print("=" * 78)

# A1. The game of the 'Unilateral' paragraph of Section 8
A = [[1, 2], [0, 0]]
B = [[1, 1], [0, 0]]
print("\nA1. Game (1,1) at (a,b), (2,1) at (a,b'), (0,0) at (a',b), (a',b'); rows a,a' = 0,1; cols b,b' = 0,1")
show("  U(G)", sorted(U_set(A, B)))
show("  IR(G)", sorted(IR_set(A, B)))
for nm, f in [("PCE (pure commitments)", pce_pure), ("Nash product", nb), ("maximin of gains", mg),
              ("mutual advantage (security)", ma_security), ("optimin", optimin), ("sum of payoffs", team)]:
    show(f"  {nm}", sorted(f(A, B)))
# PCE / MPCE with mixed commitments: leader values on a grid of commitments
N = 200
BU1 = max(max(sum((F(k, N) if r == 0 else 1 - F(k, N)) * A[r][c] for r in range(2)) for c in
              [c for c in range(2) if sum((F(k, N) if r == 0 else 1 - F(k, N)) * B[r][c] for r in range(2)) ==
               max(sum((F(k, N) if r == 0 else 1 - F(k, N)) * B[r][cc] for r in range(2)) for cc in range(2))])
          for k in range(N + 1))
BU2 = max(max(sum((F(k, N) if c == 0 else 1 - F(k, N)) * B[r][c] for c in range(2)) for r in
              [r for r in range(2) if sum((F(k, N) if c == 0 else 1 - F(k, N)) * A[r][c] for c in range(2)) ==
               max(sum((F(k, N) if c == 0 else 1 - F(k, N)) * A[rr][c] for c in range(2)) for rr in range(2))])
          for k in range(N + 1))
show("  leader values with mixed commitments (grid 1/200), best answer for the leader", (BU1, BU2))


def mixed_u(M, p, q):
    return p * q * M[0][0] + p * (1 - q) * M[0][1] + (1 - p) * q * M[1][0] + (1 - p) * (1 - q) * M[1][1]


best = None
arg = []
for i in range(N + 1):
    for j in range(N + 1):
        p, q = F(i, N), F(j, N)
        val = min(mixed_u(A, p, q) - BU1, mixed_u(B, p, q) - BU2)
        if best is None or val > best:
            best, arg = val, [(p, q)]
        elif val == best:
            arg.append((p, q))
show("  MPCE: max of min gain over mixed (grid 1/200), value and maximizers (p=Pr a, q=Pr b)", (best, arg[:5], len(arg)))

# A2. Superrationality drops U(G)
A = [[1, 3], [3, 2]]
B = [[A[y][x] for y in range(2)] for x in range(2)]
B = [[A[c][r] for c in range(2)] for r in range(2)]
print("\nA2. Symmetric game: (A,B),(B,A) give both their largest payoff 3, (B,B)=2, (A,A)=1")
show("  symmetric", is_symmetric(A, B))
show("  U(G)", sorted(U_set(A, B)))
show("  superrational", sorted(superrational(A, B)))

# A3. Z_PD
R_, P_, S_ = 3, 1, 0
A = [[R_, R_ - P_ + S_], [S_, S_ - 1]]
B = [[-x for x in row] for row in A]
print("\nA3. Z_PD for (T,R,P,S)=(5,3,1,0); rows a,a'; cols b,b'")
show("  player 1 payoffs", A)
show("  security levels", security(A, B))
show("  Berge (= Nash of the altruistic game with 2 players)", sorted(berge(A, B)))
Aalt, Balt = B, A  # each player gets the other's payoff
show("  Nash of altruistic game", sorted(nash_pure(Aalt, Balt)))
show("  IR(G)", sorted(IR_set(A, B)))
show("  pure Nash", sorted(nash_pure(A, B)))

# A4. Berge fails PO: game that becomes a PD when each receives the other's payoff
T, R, P, S = 5, 3, 1, 0
Ap, Bp = pd(T, R, P, S)
A, B = Bp, Ap  # swapped
print("\nA4. Game G whose swap is the PD (5,3,1,0); strategies C=0, D=1")
show("  payoffs G", [[pay(A, B, (r, c)) for c in range(2)] for r in range(2)])
show("  no ties", has_no_ties(A, B))
show("  security levels", security(A, B))
show("  Berge", sorted(berge(A, B)))
show("  (D,D) Pareto-optimal", pareto_optimal(A, B, (1, 1)))

# A5. Berge fails ET
A = [[1, 2], [4, 3]]
B = [[A[c][r] for c in range(2)] for r in range(2)]
print("\nA5. Symmetric game, player 1: 1 at (a,a), 2 at (a,b), 4 at (b,a), 3 at (b,b)")
show("  no ties / symmetric / copying pays", (has_no_ties(A, B), is_symmetric(A, B), copying_pays(A, B)))
show("  best symmetric", best_symmetric(A))
show("  Berge", sorted(berge(A, B)))

# A6. Matching pennies
A = [[1, -1], [-1, 1]]
B = [[-1, 1], [1, -1]]
print("\nA6. Matching pennies")
show("  Berge", sorted(berge(A, B)))
show("  Nash-2 (weak test, Sandomirskaia Def. 1)", sorted(nash2(A, B, weak=True)))
show("  Nash-2 with a strict test (contrast)", sorted(nash2(A, B, weak=False)))
show("  pure Nash", sorted(nash_pure(A, B)))

# A7. Chicken
A = [[0, 3], [1, 2]]
B = [[0, 1], [3, 2]]
print("\nA7. Chicken (A=0, Y=1)")
show("  credible alpha-core", sorted(credible_alpha_core(A, B)))
show("  alpha-core", sorted(alpha_core(A, B)))
show("  PTE limit", sorted(pte_limit(A, B)))
show("  RPP (1,2), (2,1)", (sorted(rpp(A, B, (0, 1))), sorted(rpp(A, B, (1, 0)))))
show("  SPP (1,2), (2,1)", (sorted(spp(A, B, (0, 1))), sorted(spp(A, B, (1, 0)))))
# mixed equilibrium: q = Pr(other plays A) making the row player indifferent: 3(1-q) = q + 2(1-q)
q = F(1, 2)
show("  mixed Nash: Pr(A) each, crash probability, crash payoff vs security", (q, q * q, (0, security(A, B))))

# A8. Prisoner's dilemma (5,3,1,0)
A, B = pd(5, 3, 1, 0)
print("\nA8. Prisoner's dilemma (5,3,1,0); C=0, D=1")
for nm, f in [("pure Nash", nash_pure), ("strong eq", strong_eq), ("alpha-core", alpha_core),
              ("credible alpha-core", credible_alpha_core), ("Nash-2", nash2), ("optimin", optimin),
              ("PCE pure", pce_pure), ("Berge", berge), ("team", team), ("MG", mg), ("NB", nb),
              ("MA", ma_security), ("PTE", pte_limit), ("superrational", superrational)]:
    show(f"  {nm}", sorted(f(A, B)))
show("  LCS single movers", sorted(lcs(A, B, False)))
show("  LCS joint moves", sorted(lcs(A, B, True)))
show("  farsighted stable sets single", farsighted_stable_sets(A, B, False))
show("  farsighted stable sets joint", farsighted_stable_sets(A, B, True))

# A9. Prisoner's dilemma with T+S > 2R
A, B = pd(7, 3, 1, 0)
print("\nA9. Prisoner's dilemma (7,3,1,0), T+S=7 > 2R=6")
for nm, f in [("team", team), ("MA pure", ma_security), ("MG", mg), ("superrational pure", superrational)]:
    show(f"  {nm}", sorted(f(A, B)))

# A10. Mutual advantage with the rationalizable reference point
A = [[1, 2], [3, 4]]
B = [[A[c][r] for c in range(2)] for r in range(2)]
print("\nA10. Symmetric game, player 1: row a (1,2), row b (3,4)")
show("  no ties / symmetric / copying pays", (has_no_ties(A, B), is_symmetric(A, B), copying_pays(A, B)))
# rationalizable: iterated strict dominance (pure); b strictly dominates a for both
show("  b strictly dominant for player 1", all(A[1][c] > A[0][c] for c in range(2)))
show("  only rationalizable profile (b,b) gives each her largest payoff", (pay(A, B, (1, 1)), max(map(max, A)), max(map(max, B))))

# A11. 1x1 game
A, B = [[5]], [[5]]
print("\nA11. 1x1 game")
show("  MA (security)", ma_security(A, B))
show("  no ties / symmetric / copying pays", (has_no_ties(A, B), is_symmetric(A, B), copying_pays(A, B)))

# A12. Nash product counterexample of Proposition 9
A = [[1, 2, 5], [7, 4, 8], [9, 3, 6]]
B = [[A[c][r] for c in range(3)] for r in range(3)]
print("\nA12. Nash product counterexample (Proposition 9)")
show("  no ties / symmetric / copying pays", (has_no_ties(A, B), is_symmetric(A, B), copying_pays(A, B)))
show("  security", security(A, B))
show("  IR", sorted(IR_set(A, B)))
show("  NB", sorted(nb(A, B)))
show("  MG", sorted(mg(A, B)))
show("  best symmetric", best_symmetric(A))

# A13. G_e with e = 1/2
e = F(1, 2)
A = [[0, 3], [1, e]]
B = [[0, 1], [3, e]]
print("\nA13. G_e, e=1/2")
show("  no ties", has_no_ties(A, B))
show("  superrational", sorted(superrational(A, B)))
show("  (Y,Y) Pareto-optimal", pareto_optimal(A, B, (1, 1)))
show("  truncation", sorted(truncation(A, B)))
show("  RPP (1,2)/(2,1)", (sorted(rpp(A, B, (0, 1))), sorted(rpp(A, B, (1, 0)))))
show("  union", sorted(union_orders(A, B)))

# A14. Farsighted examples
print("\nA14. Farsighted examples (rows a,a'; cols b,b')")
A = [[0, 0], [0, 1]]
B = [[0, 0], [1, 1]]
show("  tie game U(G)", sorted(U_set(A, B)))
dom = indirect_dom(A, B, False)
show("  (a',b) indirectly dominated by (a',b') (single movers)", (1, 0) in dom[(1, 1)])
show("  LCS single / joint", (sorted(lcs(A, B, False)), sorted(lcs(A, B, True))))
show("  farsighted stable sets single / joint", (farsighted_stable_sets(A, B, False), farsighted_stable_sets(A, B, True)))
A = [[0, 1], [2, 3]]
B = [[0, 1], [3, 2]]
show("  no-tie game: no ties", has_no_ties(A, B))
show("  no-tie game U(G)", sorted(U_set(A, B)))
show("  farsighted stable sets single / joint", (farsighted_stable_sets(A, B, False), farsighted_stable_sets(A, B, True)))
show("  LCS single / joint", (sorted(lcs(A, B, False)), sorted(lcs(A, B, True))))

# A15. Perturbed traveler's dilemma: no ties, symmetric, copying pays; alpha-cores keep (M-1,M)
print("\nA15. Perturbed traveler's dilemmas (symmetric, no ties): does the (credible) alpha-core keep (M-1,M)?")
rng = random.Random(20261105)
for (M, r) in [(7, F(3, 2)), (8, F(2)), (9, F(5, 2)), (10, F(3))]:
    A0, B0, claims = traveler(M, r)
    k = len(claims)
    for trial in range(3):
        ranks = list(range(k * k))
        rng.shuffle(ranks)
        A = [[A0[i][j] + F(ranks[i * k + j], 1000 * k * k) for j in range(k)] for i in range(k)]
        B = [[A[j][i] for j in range(k)] for i in range(k)]
        s = (k - 2, k - 1)  # (M-1, M)
        z = best_symmetric(A)
        print(f"  M={M} r={r} draw {trial}: no ties {has_no_ties(A, B)}, symmetric {is_symmetric(A, B)}, "
              f"copying pays {copying_pays(A, B)}, best symmetric claim {claims[z[0]]}, "
              f"(M-1,M) in alpha-core {s in alpha_core(A, B)}, in credible {s in credible_alpha_core(A, B)}, "
              f"strong eq {sorted(strong_eq(A, B))}, PTE {[(claims[a], claims[b]) for a, b in pte_limit(A, B)]}, "
              f"MA {[(claims[a], claims[b]) for a, b in ma_security(A, B)]}")

# A16. Traveler's dilemma family
print("\nA16. Traveler's dilemmas, M=2..10, r in {1/2,1,3/2,2,3,5}: which concepts give (M,M) alone")
bad = {}
cnt = 0
for M in range(2, 11):
    for r in [F(1, 2), F(1), F(3, 2), F(2), F(3), F(5)]:
        A, B, claims = traveler(M, r)
        k = len(claims)
        top = {(k - 1, k - 1)}
        cnt += 1
        res = {"PTE": pte_limit(A, B), "RPP12": rpp(A, B, (0, 1)), "RPP21": rpp(A, B, (1, 0)),
               "SPP12": spp(A, B, (0, 1)), "SPP21": spp(A, B, (1, 0)), "MG": mg(A, B), "NB": nb(A, B),
               "Berge": berge(A, B)}
        if M >= 3:
            res["MA"] = ma_security(A, B)
        for key, val in res.items():
            if set(val) != top:
                bad.setdefault(key, []).append((M, r))
        if r > 1 and M >= r + 3:
            s = (k - 2, k - 1)
            if s not in alpha_core(A, B) or s not in credible_alpha_core(A, B):
                bad.setdefault("alpha-core keeps (M-1,M)", []).append((M, r))
        if M == 2:
            if ma_security(A, B) != set():
                bad.setdefault("MA empty at M=2", []).append((M, r))
show("  traveler's dilemmas checked", cnt)
show("  failures (empty dict = none)", bad)

print("\n" + "=" * 78)
print("PART B. Constructed tests")
print("=" * 78)

# B1. Three-player game with an empty pure alpha-core (majority voting over a Condorcet cycle)
print("\nB1. Three players vote for A, B or C; an alternative with two votes wins, otherwise D")
pref = {0: {"A": 3, "B": 2, "C": 1, "D": 0}, 1: {"B": 3, "C": 2, "A": 1, "D": 0}, 2: {"C": 3, "A": 2, "B": 1, "D": 0}}
alts = ["A", "B", "C"]


def outcome(prof):
    for a in alts:
        if sum(1 for x in prof if x == a) >= 2:
            return a
    return "D"


profs3 = list(product(alts, repeat=3))


def u3(i, prof):
    return pref[i][outcome(prof)]


core3 = []
for s in profs3:
    blocked = False
    for size in (1, 2, 3):
        for S in [c for c in __import__("itertools").combinations(range(3), size)]:
            others = [j for j in range(3) if j not in S]
            for tS in product(alts, repeat=len(S)):
                ok = True
                for tO in product(alts, repeat=len(others)):
                    prof = [None] * 3
                    for idx, j in enumerate(S):
                        prof[j] = tS[idx]
                    for idx, j in enumerate(others):
                        prof[j] = tO[idx]
                    if not all(u3(i, prof) > u3(i, s) for i in S):
                        ok = False
                        break
                if ok:
                    blocked = True
                    break
            if blocked:
                break
        if blocked:
            break
    if not blocked:
        core3.append(s)
show("  pure alpha-core", core3)
# same with a tie-free perturbation: add 1/100 * (index of the profile) / 27 to each payoff
core3b = []
idx = {s: n for n, s in enumerate(profs3)}


def u3b(i, prof):
    return F(pref[i][outcome(prof)]) + F(idx[tuple(prof)] + 27 * i, 10000)


for s in profs3:
    blocked = False
    for size in (1, 2, 3):
        for S in __import__("itertools").combinations(range(3), size):
            others = [j for j in range(3) if j not in S]
            for tS in product(alts, repeat=len(S)):
                ok = True
                for tO in product(alts, repeat=len(others)):
                    prof = [None] * 3
                    for n_, j in enumerate(S):
                        prof[j] = tS[n_]
                    for n_, j in enumerate(others):
                        prof[j] = tO[n_]
                    if not all(u3b(i, prof) > u3b(i, s) for i in S):
                        ok = False
                        break
                if ok:
                    blocked = True
                    break
            if blocked:
                break
        if blocked:
            break
    if not blocked:
        core3b.append(s)
show("  pure alpha-core after a tie-breaking perturbation of size < 0.01", core3b)

# B2. Kantian equilibrium with truncation and no zero strategy
print("\nB2. Multiplicative Kantian equilibrium on [1,2]^2 with truncation, u1=-(x1+x2), u2=x1+x2")
grid = [F(1) + F(k, 20) for k in range(21)]
rs = [F(k, 40) for k in range(1, 161)]  # scaling factors 1/40 .. 4


def trunc(x):
    return min(max(x, F(1)), F(2))


eqs = []
for x1 in grid:
    for x2 in grid:
        base1, base2 = -(x1 + x2), x1 + x2
        ok1 = all(-(trunc(r * x1) + trunc(r * x2)) <= base1 for r in rs)
        ok2 = all((trunc(r * x1) + trunc(r * x2)) <= base2 for r in rs)
        if ok1 and ok2:
            eqs.append((x1, x2))
show("  equilibria on the grid", eqs)

# B3. Similarity-based / program equilibrium mixed profile judged on its support
print("\nB3. PD (5,3,1,0): mixed profile (C, 2/3 C + 1/3 D) [the example after Oesterheld et al. Theorem 3]")
A, B = pd(5, 3, 1, 0)
p, q = F(1), F(2, 3)
u1 = mixed_u(A, p, q)
u2 = mixed_u(B, p, q)
show("  expected payoffs", (u1, u2))
show("  mixed minimax of each player (other defects): P", 1)
show("  pure security levels", security(A, B))
show("  support profiles and payoffs", [((r, c), pay(A, B, (r, c))) for r in range(2) for c in range(2)
                                        if (p if r == 0 else 1 - p) > 0 and (q if c == 0 else 1 - q) > 0])

print("\n" + "=" * 78)
print("PART C. Exhaustive checks")
print("=" * 78)

pure_unilateral_yes = {
    "pure Nash": nash_pure, "Nash-2": nash2, "strong eq": strong_eq, "alpha-core": alpha_core,
    "credible alpha-core": credible_alpha_core, "PTE limit": pte_limit, "truncation": truncation,
    "RPP(1,2)": lambda A, B: rpp(A, B, (0, 1)), "RPP(2,1)": lambda A, B: rpp(A, B, (1, 0)),
    "SPP(1,2)": lambda A, B: spp(A, B, (0, 1)), "SPP(2,1)": lambda A, B: spp(A, B, (1, 0)),
    "union": union_orders,
}
pure_unilateral_no = {"Berge": berge, "optimin": optimin, "MG": mg, "NB": nb, "MA": ma_security,
                      "team": team, "PCE pure": pce_pure}
po_yes = {"strong eq": strong_eq, "alpha-core": alpha_core, "credible alpha-core": credible_alpha_core,
          "MG": mg, "NB": nb, "MA": ma_security, "team": team, "PTE limit": pte_limit,
          "RPP(1,2)": lambda A, B: rpp(A, B, (0, 1)), "RPP(2,1)": lambda A, B: rpp(A, B, (1, 0)),
          "SPP(1,2)": lambda A, B: spp(A, B, (0, 1)), "SPP(2,1)": lambda A, B: spp(A, B, (1, 0)),
          "union": union_orders}
po_no = {"pure Nash": nash_pure, "Nash-2": nash2, "Berge": berge, "optimin": optimin, "truncation": truncation,
         "PCE pure": pce_pure, "LCS single": lambda A, B: lcs(A, B, False), "LCS joint": lambda A, B: lcs(A, B, True)}

print("\nC1. The 576 strict 2x2 games")
games = strict_2x2_games()
show("  number of games", len(games))
po_fail = {k: 0 for k in list(po_yes) + list(po_no)}
incl_fail = {k: [0, 0] for k in list(pure_unilateral_yes) + ["LCS single", "LCS joint"]}
nonempty = {k: 0 for k in ["PTE limit", "credible alpha-core", "strong eq", "Nash-2", "Berge", "pure Nash"]}
order_dep_rpp = order_dep_spp = spp_miss = 0
csc_fail = 0
for (A, B) in games:
    allf = dict(po_yes)
    allf.update(po_no)
    res = {k: set(f(A, B)) for k, f in allf.items()}
    for k, v in res.items():
        if any(not pareto_optimal(A, B, s) for s in v):
            po_fail[k] += 1
    U, IR = U_set(A, B), IR_set(A, B)
    fl = dict(pure_unilateral_yes)
    for k, f in fl.items():
        v = res[k] if k in res else set(f(A, B))
        if not U <= v:
            incl_fail[k][0] += 1
        if not v <= IR:
            incl_fail[k][1] += 1
    for k in ["LCS single", "LCS joint"]:
        v = res[k]
        if not U <= v:
            incl_fail[k][0] += 1
        if not v <= IR:
            incl_fail[k][1] += 1
    for k in nonempty:
        v = res[k] if k in res else set(allf.get(k, pure_unilateral_yes.get(k))(A, B))
        if v:
            nonempty[k] += 1
    r12, r21 = res["RPP(1,2)"], res["RPP(2,1)"]
    s12, s21 = res["SPP(1,2)"], res["SPP(2,1)"]
    if r12 != r21:
        order_dep_rpp += 1
    if s12 != s21:
        order_dep_spp += 1
    lim = res["PTE limit"]
    if lim and (s12 != lim or s21 != lim):
        spp_miss += 1
    for v in (r12, r21, s12, s21):
        if not (completely_self_consistent(A, B, v) and is_rest_point(A, B, v)):
            csc_fail += 1
show("  games where a predicted profile is Pareto-dominated, per concept", po_fail)
show("  [U not in phi, phi not in IR] counts", incl_fail)
show("  games with a nonempty prediction", nonempty)
show("  RPP prediction depends on the order (games)", order_dep_rpp)
show("  SPP prediction depends on the order (games)", order_dep_spp)
show("  PTE nonempty but some SPP order misses it (games)", spp_miss)
show("  RPP/SPP predictions not completely self-consistent rest points (count)", csc_fail)

print("\nC2. The 65,536 2x2 games with payoffs in {0,1,2,3} (ties allowed)")
games = all_2x2_games_0123()
incl_fail = {k: [0, 0] for k in list(pure_unilateral_yes) + list(pure_unilateral_no) + ["LCS single", "LCS joint"]}
pte_nonempty = pte_multi = 0
for (A, B) in games:
    U, IR = U_set(A, B), IR_set(A, B)
    fl = dict(pure_unilateral_yes)
    fl.update(pure_unilateral_no)
    fl["LCS single"] = lambda A, B: lcs(A, B, False)
    fl["LCS joint"] = lambda A, B: lcs(A, B, True)
    for k, f in fl.items():
        v = set(f(A, B))
        if not U <= v:
            incl_fail[k][0] += 1
        if not v <= IR:
            incl_fail[k][1] += 1
    lim = pte_limit(A, B)
    if lim:
        pte_nonempty += 1
        if len(lim) > 1:
            pte_multi += 1
show("  number of games", len(games))
show("  [U not in phi, phi not in IR] counts", incl_fail)
show("  PTE limit nonempty / with more than one profile", (pte_nonempty, pte_multi))

print("\n" + "=" * 78)
print("PART D. Mixed strategies")
print("=" * 78)


def maximize_min(T, R, P, S, N):
    A, B = pd(T, R, P, S)
    best, arg = None, []
    for i in range(N + 1):
        for j in range(N + 1):
            p, q = F(i, N), F(j, N)
            val = min(mixed_u(A, p, q), mixed_u(B, p, q))
            if best is None or val > best:
                best, arg = val, [(p, q)]
            elif val == best:
                arg.append((p, q))
    return best, arg


for (T, R, P, S) in [(7, 3, 1, 0), (5, 3, 1, 0), (6, 3, 1, 0)]:
    x = F(S + T - 2 * P, 2 * (S + T) - 2 * R - 2 * P)
    print(f"\nD1. PD {(T, R, P, S)}, 2R={2 * R}, T+S={T + S}")
    show("  stationary point of the common payoff f(x)=u(x,x)", x)
    best, arg = maximize_min(T, R, P, S, 120)
    show("  max over mixed (p,q) on grid 1/120 of min(u1,u2), maximizers", (best, float(best), arg))
    A, B = pd(T, R, P, S)
    # superrationality / homo moralis kappa=1: maximize f(x) on the grid
    fx = [(mixed_u(A, F(k, 120), F(k, 120)), F(k, 120)) for k in range(121)]
    show("  superrational mixture (grid 1/120): best f and x", max(fx))
    # measure of mutual advantage with security levels (mixed security P), divisor T-P equal for both
    show("  MA objective = (min(u1,u2)-P)/(T-P): same maximizers as min(u1,u2)", True)

# D2. Kantian equilibrium (multiplicative) in the mixed extension of the PD
print("\nD2. Multiplicative Kantian equilibria of the mixed extension of the PD (grid 1/60 on profiles)")


def kantian_eqs(T, R, P, S, N=60, truncate=False):
    A, B = pd(T, R, P, S)
    out = []
    rs = [F(k, 100) for k in range(0, 301)]
    for i in range(N + 1):
        for j in range(N + 1):
            p, q = F(i, N), F(j, N)
            ok = True
            for M in (A, B):
                base = mixed_u(M, p, q)
                for r in rs:
                    pp, qq = r * p, r * q
                    if pp > 1 or qq > 1:
                        if not truncate:
                            continue
                        pp, qq = min(pp, F(1)), min(qq, F(1))
                    if mixed_u(M, pp, qq) > base:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                out.append((p, q))
    return out


for (T, R, P, S) in [(7, 3, 1, 0), (5, 3, 1, 0)]:
    for tr in (False, True):
        eq = kantian_eqs(T, R, P, S, truncate=tr)
        show(f"  PD {(T, R, P, S)} truncation={tr}: equilibria on the grid", eq)

# D3. Magical thinking in the PD (5,3,1,0), types uniform on [0,1]
print("\nD3. Magical thinking, PD (5,3,1,0), types uniform on [0,1]")
T, R, P, S = 5, 3, 1, 0
# type a cooperates iff a*R + (1-a)*(q R + (1-q) S) >= a*P + (1-a)*(q T + (1-q) P), q = Pr(other cooperates) = 1 - a*
# cutoff: a*(R-P) = (1-a*)*(q(T-R) + (1-q)(P-S)) with q = 1-a*
lo, hi = 0.0, 1.0
for _ in range(200):
    a = (lo + hi) / 2
    qq = 1 - a
    lhs = a * (R - P) - (1 - a) * (qq * (T - R) + (1 - qq) * (P - S))
    if lhs > 0:
        hi = a
    else:
        lo = a
a = (lo + hi) / 2
show("  cutoff type, Pr(other cooperates), Pr(mutual cooperation)", (a, 1 - a, (1 - a) ** 2))
show("  closed form (5-sqrt(17))/2 and ((sqrt(17)-3)/2)^2", ((5 - math.sqrt(17)) / 2, ((math.sqrt(17) - 3) / 2) ** 2))
