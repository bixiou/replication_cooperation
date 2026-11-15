"""A prisoner's dilemma with a third pure strategy that cooperates with probability 9/10, checked exactly.

PD (T,R,P,S) = (10,3,1,0), T+S > 2R. Add for both players a pure strategy m that plays
C with probability 9/10 and D otherwise, paid its expected payoffs (a "named coin").
Compute: payoffs of the 3x3 game G', whether it has ties, its pure perfectly transparent
rounds (Fourny: guarantee = max over strategies with nonempty section of the min payoff
in the section), and the main-session candidate Phi (guarantees over mixed commitments,
section = answers y with supp(tau) x {y} inside F), here on pure sets only: the mixed
commitment value is maximized over an exact rational grid of step 1/60 on each support,
so the reported mixed guarantee is a lower bound (exact when the optimum is a grid point;
here round 0 gives the pure security level 1 and round 1 the pure guarantee 3.34, which
the hand computation confirms: on F_1 a mixture of C and m answered by m gets at most 3.34).
"""
from fractions import Fraction as Fr
from itertools import product, combinations

T, R, P, S = Fr(10), Fr(3), Fr(1), Fr(0)
pd = {('C', 'C'): (R, R), ('C', 'D'): (S, T), ('D', 'C'): (T, S), ('D', 'D'): (P, P)}
q = Fr(9, 10)
mix = {'C': {'C': Fr(1)}, 'D': {'D': Fr(1)}, 'm': {'C': q, 'D': 1 - q}}
St = ['C', 'm', 'D']


def u(a, b):
    v1 = v2 = Fr(0)
    for x, px in mix[a].items():
        for y, py in mix[b].items():
            v1 += px * py * pd[(x, y)][0]
            v2 += px * py * pd[(x, y)][1]
    return (v1, v2)


G = {(a, b): u(a, b) for a in St for b in St}
print("Payoffs of G' (row player 1, column player 2):")
for a in St:
    print("  ", a, [f"({float(G[(a, b)][0]):.2f},{float(G[(a, b)][1]):.2f})" for b in St])
p1 = [G[s][0] for s in G]
p2 = [G[s][1] for s in G]
print("ties for player 1:", len(p1) != len(set(p1)), " ties for player 2:", len(p2) != len(set(p2)))


def pay(i, s):
    return G[s][i]


def sec(i, t, F):
    return [s for s in F if s[i] == t]


def pure_guarantee(i, F):
    vals = [min(pay(i, s) for s in sec(i, t, F)) for t in St if sec(i, t, F)]
    return max(vals)


def mixed_guarantee(i, F):
    """max over supports A of S_i with nonempty section (answers y with A x {y} in F)
    of max over tau in Delta(A) (rational grid, step 1/60) of min over y in the section
    of u_i(tau, y): a lower bound on the exact guarantee."""
    best = None
    for k in (1, 2, 3):
        for A in combinations(St, k):
            ys = [y for y in St if all(((a, y) if i == 0 else (y, a)) in F for a in A)]
            if not ys:
                continue
            def val(w):  # w: dict a -> prob
                return min(sum(w[a] * pay(i, (a, y) if i == 0 else (y, a)) for a in A) for y in ys)
            cands = []
            if k == 1:
                cands = [{A[0]: Fr(1)}]
            else:
                N = 60
                for c in product(range(N + 1), repeat=k - 1):
                    if sum(c) <= N:
                        w = [Fr(x, N) for x in c]
                        w.append(1 - sum(w))
                        cands.append(dict(zip(A, w)))
            for w in cands:
                v = val(w)
                if best is None or v > best:
                    best = v
    return best


def rounds(guarantee, label):
    F = set(G)
    k = 0
    while True:
        g = [guarantee(0, F), guarantee(1, F)]
        F2 = {s for s in F if pay(0, s) >= g[0] and pay(1, s) >= g[1]}
        print(f"  {label} round {k}: guarantees ({float(g[0]):.4f}, {float(g[1]):.4f}) -> {sorted(F2)}")
        if not F2 or F2 == F:
            return F2 if F2 else F
        F = F2
        k += 1


print("\nPure perfectly transparent rounds on G':")
rounds(pure_guarantee, "PTE")
print("\nPhi (mixed commitments, grid 1/60 over supports) on G', pure part:")
rounds(mixed_guarantee, "Phi")
print("\nu(m,m) =", G[('m', 'm')], "=", float(G[('m', 'm')][0]), "against R =", R)
