"""demands_paper_examples.py: every number of paper/demands/demands.tex, recomputed exactly.

Run: uv run --no-project --with sympy python research/scripts/demands_paper_examples.py > research/scripts/demands_paper_examples.out

Logic (each part prints the values the paper uses, labelled by the place where it uses them).
- Part A, game K (symmetric 2x2, player 1 gets 7/4, 2, 3, 1 at (A,A), (A,Y), (Y,A), (Y,Y)); p, q = probabilities of A.
  Mixed and pure security levels; the identity min(u1,u2) - 2 = -(3s/4 - 1)^2 + 9e^2/4 - |e| with p = s/2 + e,
  q = s/2 - e; an exact rational grid (step 1/180) locating the maximum of min(u1,u2); the payoffs of
  sigma0 = (2/3,2/3), (A,Y), (Y,A); the common path from the security levels (time to reach (2,2));
  the movers' leximin of K and of K with player 2's payoffs doubled (exact grid plus the argument of the paper);
  the first step of the discrete Kalai-Smorodinsky variant.
- Part B, prisoner's dilemmas (T,R,P,S) = (10,3,1,0) and (5,3,1,0): security level, best symmetric mixed profile,
  grid check of the unique profile giving both f*, the first reasoner's maximum over IR when she raises her
  demand first (exact case analysis with the sign of the derivative factored), the discrete step.
- Part C, chicken (0,3,1,2) and Aumann's chicken (0,7,2,6): security levels, u1 + u2, f*, the discrete step.
- Part D, game G2 (zero-sum between players 1 and 2, player 3 with one strategy): values, the cross IR,
  player 3's payoffs on it.
- Part E, traveler's dilemma: u1 + u2 <= 2M at every pure profile, with equality only at (M,M).
- Part F, game G3: security levels and the gains of (a,b) and (a',b') (the leximin over all players).
- Part G, game K: Raiffa's midpoint step from the security levels leaves the feasible set.
"""
from fractions import Fraction as Fr
import math

import sympy as sp

p, q, s, e = sp.symbols("p q s e", real=True)


def bilinear(a_aa, a_ay, a_ya, a_yy):
    """Player 1's expected payoff, p = P(1 plays A), q = P(2 plays A)."""
    return sp.expand(a_aa * p * q + a_ay * p * (1 - q) + a_ya * (1 - p) * q + a_yy * (1 - p) * (1 - q))


def swap(expr):
    """Symmetric game: u2(p, q) = u1(q, p)."""
    return sp.expand(expr.subs({p: q, q: p}, simultaneous=True))


def maxmin_rows(u1):
    """Mixed security level of player 1 in a 2x2 game: max over p of min over q in {0,1} (exact)."""
    l1, l0 = u1.subs(q, 1), u1.subs(q, 0)
    cands = [sp.Integer(0), sp.Integer(1)]
    sol = sp.solve(sp.Eq(l1, l0), p)
    cands += [x for x in sol if 0 <= x <= 1]
    best = max(cands, key=lambda x: min(l1.subs(p, x), l0.subs(p, x)))
    return sp.nsimplify(min(l1.subs(p, best), l0.subs(p, best))), best


def grid(n):
    return [Fr(k, n) for k in range(n + 1)]


def ev(expr, pv, qv):
    """Exact value of a polynomial in p, q at rational points (Fractions)."""
    poly = sp.Poly(expr, p, q)
    tot = Fr(0)
    for (i, j), cf in poly.terms():
        cf = sp.Rational(cf)
        tot += Fr(int(cf.p), int(cf.q)) * pv ** i * qv ** j
    return tot


print("== Part A: game K ==")
u1 = bilinear(sp.Rational(7, 4), 2, 3, 1)
u2 = swap(u1)
print("u1 =", u1, "; u2 =", u2)
print("pure payoffs of player 1: (A,A)", u1.subs({p: 1, q: 1}), "(A,Y)", u1.subs({p: 1, q: 0}),
      "(Y,A)", u1.subs({p: 0, q: 1}), "(Y,Y)", u1.subs({p: 0, q: 0}))
v, pv = maxmin_rows(u1)
print("[Sec 3, Sec 4 example] mixed security level v =", v, "attained by p =", pv)
print("pure security level =", max(min(u1.subs({p: 1, q: 1}), u1.subs({p: 1, q: 0})),
                                    min(u1.subs({p: 0, q: 1}), u1.subs({p: 0, q: 0}))))
sub = {p: s / 2 + e, q: s / 2 - e}
avg = sp.expand(((u1 + u2) / 2).subs(sub, simultaneous=True))
half_diff = sp.expand(((u1 - u2) / 2).subs(sub, simultaneous=True))
print("(u1 + u2)/2 - 2 =", sp.factor(avg - 2), "; (u1 - u2)/2 =", half_diff)
ident = sp.simplify(avg - 2 - (-(sp.Rational(3, 4) * s - 1) ** 2 + sp.Rational(9, 4) * e ** 2))
print("[Prop K proof] identity (u1+u2)/2 - 2 = -(3s/4 - 1)^2 + 9e^2/4:", ident == 0,
      "; so min(u1,u2) - 2 = -(3s/4 - 1)^2 + 9e^2/4 - |e|")
e_ = sp.symbols("E", positive=True)
bound = sp.expand(-(sp.Rational(3, 4) * (2 - 2 * e_) - 1) ** 2 + sp.Rational(9, 4) * e_ ** 2 - e_)
print("[Prop K proof] value at s = 2 - 2|e| for 4/9 < |e| <= 1/2:", bound)
N = 180
G = grid(N)
U1 = {(a, b): ev(u1, a, b) for a in G for b in G}
best, arg = None, []
for (a, b), x in U1.items():
    m = min(x, U1[(b, a)])
    if best is None or m > best:
        best, arg = m, [(a, b)]
    elif m == best:
        arg.append((a, b))
print("[Prop K, Sec 4 example] exact grid 1/%d: max of min(u1,u2) =" % N, best, "at", [(str(a), str(b)) for a, b in arg])
print("[Prop K] sigma0 = (2/3,2/3) pays", (ev(u1, Fr(2, 3), Fr(2, 3)), ev(u2, Fr(2, 3), Fr(2, 3))),
      "; (A,Y) pays", (ev(u1, Fr(1), Fr(0)), ev(u2, Fr(1), Fr(0))),
      "; (Y,A) pays", (ev(u1, Fr(0), Fr(1)), ev(u2, Fr(0), Fr(1))))
t1 = sp.solve(sp.Eq(3 - (3 - v) * sp.exp(-sp.Symbol("t")), 2), sp.Symbol("t"))
print("[Sec 4 example] common path c(t) = 3 - (3 - v) e^{-t} =", 3 - (3 - v), "... reaches (2,2) at t =", t1)
# movers' leximin in K
vF = Fr(17, 9)
gains = {k: (U1[k] - vF, U1[(k[1], k[0])] - vF) for k in U1}
ir = {k: g for k, g in gains.items() if min(g) >= 0}
first = max(min(g) for g in ir.values())
lex1 = [k for k, g in ir.items() if min(g) == first]
second = max(max(ir[k]) for k in lex1)
lex = [k for k in lex1 if max(ir[k]) == second]
print("[Sec 5 leximin] K, grid 1/%d: max smallest gain =" % N, first, "at", [(str(a), str(b)) for a, b in lex1],
      "; leximin =", [(str(a), str(b)) for a, b in lex])
# K with player 2's payoffs doubled: v = (17/9, 34/9), gains (u1 - 17/9, 2 u2 - 34/9)
gains2 = {k: (U1[k] - vF, 2 * U1[(k[1], k[0])] - 2 * vF) for k in U1}
ir2 = {k: g for k, g in gains2.items() if min(g) >= 0}
first2 = max(min(g) for g in ir2.values())
lex2 = [k for k, g in ir2.items() if min(g) == first2]
print("[Sec 5 leximin] K with player 2's payoffs doubled, grid 1/%d: max smallest gain =" % N, first2, "at",
      [(str(a), str(b)) for a, b in lex2],
      "(argument: smallest gain >= 2/9 needs u1 >= 19/9 > 2 and u2 >= 2, so min(u1,u2) = 2 with u1 > 2: only (Y,A))")
print("[Sec 5 discrete step] K: M(v,v) = (3,3), t* = (2 - v)/(3 - v) =", (2 - v) / (3 - v), "-> (2,2), then t* = 0")

print()
print("== Part B: prisoner's dilemmas ==")
for (T, R, P, S) in [(10, 3, 1, 0), (5, 3, 1, 0)]:
    # p, q = probabilities of cooperating; bilinear with A = C
    w1 = bilinear(R, S, T, P)
    w2 = swap(w1)
    vv, _ = maxmin_rows(w1)
    f = sp.expand(w1.subs(q, p))
    crit = sp.solve(sp.diff(f, p), p)
    cands = [sp.Integer(0), sp.Integer(1)] + [x for x in crit if 0 <= x <= 1]
    pstar = max(cands, key=lambda x: f.subs(p, x))
    fstar = f.subs(p, pstar)
    print("PD", (T, R, P, S), ": u1 =", w1, "; v =", vv, "; f(p) =", f, "; p* =", pstar, "; f* =", fstar,
          "; T+S <= 2R:", T + S <= 2 * R)
    W1 = {(a, b): ev(w1, a, b) for a in grid(120) for b in grid(120)}
    mm = max(min(x, W1[(b, a)]) for (a, b), x in W1.items())
    at = [(str(a), str(b)) for (a, b), x in W1.items() if min(x, W1[(b, a)]) == mm]
    print("  grid 1/120: max of min(u1,u2) =", mm, "at", at)
    theta = sp.Rational(P - S, R - S)
    # first reasoner's maximum of u1 over IR = {u2 >= P}
    q_bind = sp.solve(sp.Eq(w2, P), q)[0]
    g = sp.simplify(w1.subs(q, q_bind))
    dg = sp.factor(sp.together(sp.diff(g, p)))
    p_line = sp.solve(sp.Eq(q_bind, 1), p)
    print("  [Prop S] u2 >= P iff q <=", sp.simplify(q_bind), "; q = 1 feasible iff p >=", p_line,
          "(theta =", theta, "); on q = 1: u1 =", sp.expand(w1.subs(q, 1)),
          "; on the binding curve u1 = g(p), g'(p) =", dg)
    mx = sp.expand(w1.subs(q, 1)).subs(p, theta)
    print("  [Prop S] max of u1 over IR =", mx, "at (theta, 1) =", (theta, 1), "; f* =", fstar,
          "; serial outcome pays (", mx, ",", P, ")")
    M1 = mx
    print("  [Sec 5 discrete step] t* = (f* - v)/(M1 - v) =", sp.nsimplify((fstar - vv) / (M1 - vv)))

print()
print("== Part C: chicken ==")
for name, (a, b, c, d) in [("chicken", (0, 3, 1, 2)), ("Aumann's chicken", (0, 7, 2, 6))]:
    c1 = bilinear(a, b, c, d)
    c2 = swap(c1)
    vv, _ = maxmin_rows(c1)
    f = sp.expand(c1.subs(q, p))
    crit = [x for x in sp.solve(sp.diff(f, p), p) if 0 <= x <= 1]
    cands = [sp.Integer(0), sp.Integer(1)] + crit
    pstar = max(cands, key=lambda x: f.subs(p, x))
    fstar = f.subs(p, pstar)
    print(name, (a, b, c, d), ": u1 =", c1, "; v =", vv, "; u1 + u2 =", sp.factor(c1 + c2), "; f(p) =", sp.factor(f),
          "; f* =", fstar, "at p =", pstar, "; u1 + u2 - 2f* =", sp.factor(c1 + c2 - 2 * fstar))
    Mv = b  # the largest payoff, met at (A,Y) by player 2 receiving c >= v
    print("  [Sec 5 discrete step] M(v,v) = (", Mv, ",", Mv, ") since (A,Y) pays", (c1.subs({p: 1, q: 0}), c2.subs({p: 1, q: 0})),
          "; t* = (f* - v)/(M - v) =", sp.nsimplify((fstar - vv) / (Mv - vv)))

print()
print("== Part D: game G2 ==")
g1 = bilinear(3, -1, -2, 1)  # player 1; player 2 gets -g1; rows a (p), columns b (q)
v1, p1 = maxmin_rows(g1)
# player 2 maximizes -g1 by choosing q: value = max_q min_p (-g1)
l_a, l_ap = (-g1).subs(p, 1), (-g1).subs(p, 0)
qs = sp.solve(sp.Eq(l_a, l_ap), q)[0]
v2 = l_a.subs(q, qs)
print("u1 =", g1, "; v1 =", v1, "at p =", p1, "; v2 =", v2, "at q =", qs)
print("u1 - v1 =", sp.factor(g1 - v1))
d3 = bilinear(0, 1, 2, 3)
print("[Remark movers] player 3 on q = 2/7:", sp.expand(d3.subs(q, sp.Rational(2, 7))),
      "range", (d3.subs({q: sp.Rational(2, 7), p: 1}), d3.subs({q: sp.Rational(2, 7), p: 0})),
      "; on p = 3/7:", sp.expand(d3.subs(p, sp.Rational(3, 7))),
      "range", (d3.subs({p: sp.Rational(3, 7), q: 1}), d3.subs({p: sp.Rational(3, 7), q: 0})))

print()
print("== Part E: traveler's dilemma ==")
for (Mx, r) in [(3, Fr(3, 2)), (10, 2), (100, 2), (100, Fr(1, 2))]:
    ok = True
    for x in range(2, Mx + 1):
        for y in range(2, Mx + 1):
            if x == y:
                a1 = a2 = Fr(x)
            else:
                lo = min(x, y)
                a1 = lo + r if x < y else lo - r
                a2 = lo + r if y < x else lo - r
            tot = a1 + a2
            if tot > 2 * Mx or (tot == 2 * Mx and (x, y) != (Mx, Mx)):
                ok = False
    print("M =", Mx, "r =", r, ": u1 + u2 <= 2M with equality only at (M,M):", ok)
print()
print("== Part F: game G3 (players 1 and 2 move, player 3 has one strategy) ==")
h1 = bilinear(4, 1, 0, 3)  # player 1: (a,b) 4, (a,b') 1, (a',b) 0, (a',b') 3; player 2 symmetric
vh, ph = maxmin_rows(h1)
print("u1 =", h1, "; v1 = v2 =", vh, "at p =", ph, "; v3 = min(0,1,2,3) = 0")
print("[Remark movers] gains at (a,b) =", (4 - vh, 4 - vh, 0), "; at (a',b') =", (3 - vh, 3 - vh, 3),
      "; sorted:", sorted((4 - vh, 4 - vh, 0)), "vs", sorted((3 - vh, 3 - vh, 3)))

print()
print("== Part G: Raiffa's midpoint step in K ==")
mid = v + (3 - v) / 2
print("[Sec 5 bargaining] from (v,v) = (17/9,17/9) the midpoint of the ideal points is (", mid, ",", mid,
      "), above the largest common payoff 2, so outside H")
print("done")
