"""Independent recomputation of the exhaustive counts of the paper.

A. The 362,880 symmetric 3x3 games in which player 1's payoffs are 1..9 (u2(x,y)=u1(y,x)):
   copying pays / fails; PTE limit = best symmetric profile among the failures (main text: 239,232 and 114,192);
   on the 123,648 copying-pays games: Prop 2, Prop 4 (truncation), Prop 6(iii) for both concepts and both
   orders, Remark 1 patchwork, maximin of gains, Nash product only asymmetric (main text: 864),
   team reasoning failing ET (main text: 11,040), measure of mutual advantage;
   on the games with empty PTE limit: count (OA: 125,040) and those with an exchange-invariant nonempty
   reachable rest point (OA: 7,824).
B. The 576 strict 2x2 games: PTE nonempty (516); games with no invariant SC weakly efficient set (8);
   games with no invariant nonempty reachable rest point (12) and the characterization of the 12;
   sequential misses the PTE limit for some order (8); prediction depends on the order (OA: 44 seq, 60 res).
C. The 65,536 2x2 games with payoffs in {0,1,2,3}: PTE nonempty (62,644), more than one profile (28,976);
   and a falsification test of Prop 6 (i), (ii) and the 'Moreover' claim on all of them, both orders.
"""
import sys
import time
from fractions import Fraction as F
from itertools import product, permutations, combinations

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from indep_coop_lib import *  # noqa

t0 = time.time()
print("=== A. symmetric 3x3 games, player 1's payoffs a permutation of 1..9")
n_total = n_cp = n_fail = n_fail_best = 0
bad = {k: 0 for k in ["prop2", "prop2_unique", "trunc", "seq12", "seq21", "res12", "res21", "patch", "mg", "ma", "cpo"]}
n_nb_asym = n_team_fail = 0
n_empty = n_empty_inv_rp = 0
sw = lambda E: frozenset((y, x) for (x, y) in E)
for perm in permutations(range(1, 10)):
    A = [list(perm[0:3]), list(perm[3:6]), list(perm[6:9])]
    G = symmetric(A)
    n_total += 1
    cp = copying_pays(A)
    z = best_symmetric(A)
    E, rounds = pte(G)
    if not E:
        n_empty += 1
        R_ = reachable(G)
        if any(E_ and is_rest(G, E_) and sw(E_) == E_ for E_ in R_):
            n_empty_inv_rp += 1
    if not cp:
        n_fail += 1
        if E == frozenset({z}):
            n_fail_best += 1
        continue
    n_cp += 1
    cstar = A[z[0]][z[0]]
    if E != frozenset({z}):
        bad["prop2"] += 1
    if [s for s in G.S if G.u[s][0] >= cstar and G.u[s][1] >= cstar] != [z]:
        bad["prop2_unique"] += 1
    if z not in pareto_optimal(G):
        bad["cpo"] += 1
    trunc = [R_ for R_ in rounds if R_][-1]
    if trunc != frozenset({z}):
        bad["trunc"] += 1
    if seq(G, [0, 1]) != {z}:
        bad["seq12"] += 1
    if seq(G, [1, 0]) != {z}:
        bad["seq21"] += 1
    if resolved(G, [0, 1]) != {z}:
        bad["res12"] += 1
    if resolved(G, [1, 0]) != {z}:
        bad["res21"] += 1
    v = security(G, 0)
    ir = [s for s in G.S if G.u[s][0] >= v and G.u[s][1] >= v]
    gmin = {s: min(G.u[s][0] - v, G.u[s][1] - v) for s in ir}
    mgset = frozenset(s for s in ir if gmin[s] == max(gmin.values()))
    if mgset != {z}:
        bad["mg"] += 1
    if (U_set(G) | mgset) != {z}:
        bad["patch"] += 1
    gp = {s: (G.u[s][0] - v) * (G.u[s][1] - v) for s in ir}
    nbset = [s for s in ir if gp[s] == max(gp.values())]
    if all(s[0] != s[1] for s in nbset):
        n_nb_asym += 1
    sums = {s: G.u[s][0] + G.u[s][1] for s in G.S}
    team = frozenset(s for s in G.S if sums[s] == max(sums.values()))
    if team != {z}:
        n_team_fail += 1
    # measure of mutual advantage: gains over security divided by the largest possible gain
    d = max(G.u[s][0] for s in G.S) - v
    ma = {s: min(F(G.u[s][0] - v, d), F(G.u[s][1] - v, d)) for s in ir}
    maset = frozenset(s for s in ir if ma[s] == max(ma.values()))
    if maset != {z}:
        bad["ma"] += 1
print("total games", n_total, "| copying pays", n_cp, "| copying fails", n_fail, "| among failures, PTE limit = best symmetric", n_fail_best)
print("paper: fails in 239,232; limit best symmetric in 114,192 of these")
print("violations among copying-pays games (all should be 0):", bad)
print("Nash product predicts only asymmetric profiles in", n_nb_asym, "of", n_cp, "(paper: 864 of 123,648)")
print("team reasoning (sum) fails ET in", n_team_fail, "of", n_cp, "(paper: 11,040)")
print("empty PTE limit in", n_empty, "of", n_total, "(OA: 125,040); with an exchange-invariant nonempty reachable rest point:", n_empty_inv_rp, "(OA: 7,824)")
print("time %.0fs" % (time.time() - t0))

print()
print("=== B. strict 2x2 games (pairs of strict rankings)")
t0 = time.time()
games = []
for p1 in permutations(range(1, 5)):
    for p2 in permutations(range(1, 5)):
        S_ = [(0, 0), (0, 1), (1, 0), (1, 1)]
        u = {s: (p1[k], p2[k]) for k, s in enumerate(S_)}
        games.append(Game((2, 2), u))
print("number of games", len(games))
n_nonempty = 0
n_no_inv_sc_we = 0
n_no_inv_rp = 0
twelve = []
n_seq_miss = 0
n_seq_dep = n_res_dep = 0
n_exch_empty = []
n_res_dep_nonempty = 0
for G in games:
    E, _ = pte(G)
    auts = automorphisms(G)
    po = pareto_optimal(G)
    if E:
        n_nonempty += 1
        if seq(G, [0, 1]) != E or seq(G, [1, 0]) != E:
            n_seq_miss += 1
    inv_sets = [frozenset(c) for k in range(1, 5) for c in combinations(G.S, k) if invariant(G, frozenset(c), auts)]
    if not any(self_consistent(G, X) and (X & po) for X in inv_sets):
        n_no_inv_sc_we += 1
    R_ = reachable(G)
    if not any(X and is_rest(G, X) and invariant(G, X, auts) for X in R_):
        n_no_inv_rp += 1
        twelve.append(G)
    has_exch = any(tau == (1, 0) for tau, f in auts)
    if has_exch and not E:
        n_exch_empty.append(G)
    if seq(G, [0, 1]) != seq(G, [1, 0]):
        n_seq_dep += 1
    if resolved(G, [0, 1]) != resolved(G, [1, 0]):
        n_res_dep += 1
        if E:
            n_res_dep_nonempty += 1
print("PTE nonempty in", n_nonempty, "(paper: 516)")
print("no invariant nonempty SC set containing a PO profile:", n_no_inv_sc_we, "(paper: 8)")
print("no invariant nonempty reachable rest point:", n_no_inv_rp, "(paper: 12)")
print("games with an automorphism exchanging the players and empty PTE:", len(n_exch_empty),
      "| same set as the previous one:", set(G.key() for G in twelve) == set(G.key() for G in n_exch_empty))
print("PTE nonempty and some order of sequential perfect prediction misses it:", n_seq_miss, "(paper: 8)")
print("prediction depends on the order: sequential", n_seq_dep, "(OA: 44); resolved", n_res_dep, "(OA: 60); resolved with nonempty PTE", n_res_dep_nonempty, "(should be 0)")
print("time %.0fs" % (time.time() - t0))

print()
print("=== C. 2x2 games with payoffs in {0,1,2,3}")
t0 = time.time()
n_ne = n_multi = 0
viol = {k: 0 for k in ["nonempty", "rest", "csc", "U", "IR", "res=pte", "reach"]}
for vals in product(range(4), repeat=8):
    S_ = [(0, 0), (0, 1), (1, 0), (1, 1)]
    u = {s: (vals[2 * k], vals[2 * k + 1]) for k, s in enumerate(S_)}
    G = Game((2, 2), u)
    E, _ = pte(G)
    if E:
        n_ne += 1
        if len(E) > 1:
            n_multi += 1
    U = U_set(G)
    IR = IR_set(G)
    R_ = None
    for o in ([0, 1], [1, 0]):
        for X in (seq(G, o), resolved(G, o)):
            if not X:
                viol["nonempty"] += 1
            if not is_rest(G, X):
                viol["rest"] += 1
            if not completely_sc(G, X):
                viol["csc"] += 1
            if not U <= X:
                viol["U"] += 1
            if not X <= IR:
                viol["IR"] += 1
        if E and resolved(G, o) != E:
            viol["res=pte"] += 1
print("PTE nonempty in", n_ne, "of 65536 (paper: 62,644); more than one profile in", n_multi, "(paper: 28,976)")
print("Prop 6 violations over all 65,536 games, both orders, both concepts (all should be 0):", viol)
print("time %.0fs" % (time.time() - t0))
