"""mixed_single_logic.py: perfect prediction with mixed commitments over pure profiles, and variants.

Run: uv run --no-project --with numpy --with scipy --with sympy python research/scripts/mixed_single_logic.py

Logic (summary written before the code).
- A game is (shape, u): shape = strategy counts, u = dict pure profile -> tuple of Fractions.
- Guarantee with mixed commitments over a set F of pure profiles: for each nonempty support T of player i,
  Y(T) = pure answers y of the others with T x {y} inside F; if Y(T) is nonempty, LP(T) = max over tau in Delta(T) of
  min over y in Y(T) of u_i(tau, y) (exact: float LP, rational reconstruction, certified by an exact primal value equal
  to an exact dual value; vertex enumeration as fallback). g_i(F) = max over T of LP(T). We also record whether some
  commitment attains g_i(F) on its own section (an optimal tau whose support T' gives min over Y(T') equal to g_i).
- Phi: F_0 = S; g_k = g(F_k); F_(k+1) = pure profiles of F_k meeting g_k; P_k = mixed profiles with support inside F_k
  meeting g_k in expectation. Prediction P_K, K the last k with P_k nonempty (P_0 = IR^m is nonempty). When F_(k+1) is
  empty and k >= 1, nonemptiness of P_k is decided exactly for two players (boxes with at most two strategies per
  player suffice; on a 2x2 box the feasible set of two bilinear inequalities is nonempty iff one candidate point is
  feasible: corners, roots on edges, degenerate lines, intersections of the two zero curves), numerically for n >= 3.
- RPP^m under an order: simultaneous verdicts on F while the pure result is nonempty; if empty and P(F) is nonempty,
  stop with P(F) (demands compatible only through mixtures); if P(F) is empty, one pass of single verdicts in the
  order, then simultaneous verdicts again. Prediction P(F) at the stop. Union over orders: union of the predictions.
- Checks: g(F_0) = v^m; guarantees nondecreasing; g(F_K) >= v^m; pure bliss profiles inside F_K and meeting g(F_K);
  one-strategy players' guarantee = their minimum over F; prediction nonempty; for RPP^m: rest points completely
  self-consistent on pure profiles; without ties a single pure profile, Pareto-optimal among pure profiles.
- Games: PD grid (T+S > 2R included, asymmetric too), TD (M = 3..8, several real r > 1), matching pennies, chicken,
  battle of the sexes, Leader, G_e, the ties example of the pedagogical paper, the sup-not-attained example, all strict
  2x2 games, random games with ties (2x2, 2x3, 3x3, 1x3, 3x1, 2x2x2, 1x2x2, 2x2x3) and the variant in which F_(k+1)
  is the union of the supports of P_k (PD stall).
- Part 7: random strict 3x3 and 2x2x2 games: Phi against the pure PTE limit, Psi's pure branch, Phi_Q (truncation at
  the last nonempty half-round: its pure members are the last nonempty F_k), Pareto optimality of pure members.
- Part 8: Phi_some, the variant in which an answer is possible after a mixed commitment if SOME realization stays in F.
- Parts 9 and 10 (further variants): Seed 6 (valuation U_i(sigma) =
  min over the others' realizations of u_i(sigma_i, s_-i), Fourny's rounds on mixed profiles with mixed commitments,
  truncation; two players; the rounds are E(c) = {U >= c}, decided support pair by support pair with float LPs);
  Seed 1 (closed classes of the departure graph on pure worlds; a player below her guarantee moves to every (t, y),
  t in the support T of a commitment attaining the guarantee, y in its section Y(T); stationary laws under uniform
  departures); Seed 12's first round (worlds meeting the pure security levels, lotteries meeting the mixed ones).
"""
import itertools
import random
import sys
from fractions import Fraction as Fr

import numpy as np
import sympy as sp
from scipy.optimize import linprog, minimize

ONE, ZERO = Fr(1), Fr(0)


# ------------------------------------------------------------------ games
def profiles(shape):
    return list(itertools.product(*[range(k) for k in shape]))


def others(shape, i):
    return list(itertools.product(*[range(k) for j, k in enumerate(shape) if j != i]))


def ins(y, i, t):
    lst = list(y)
    lst.insert(i, t)
    return tuple(lst)


def game(shape, f):
    return (tuple(shape), {s: tuple(Fr(x) for x in f(s)) for s in profiles(shape)})


def u_mixed(G, sigma, i):
    shape, u = G
    tot = ZERO
    for s in profiles(shape):
        p = ONE
        for j, sj in enumerate(s):
            p *= sigma[j][sj]
            if p == 0:
                break
        if p:
            tot += p * u[s][i]
    return tot


# ------------------------------------------------------------------ exact maxmin
def _solve(A, b):
    n = len(A)
    M = [list(r) + [bb] for r, bb in zip(A, b)]
    for c in range(n):
        piv = next((r for r in range(c, n) if M[r][c] != 0), None)
        if piv is None:
            return None
        M[c], M[piv] = M[piv], M[c]
        for r in range(n):
            if r != c and M[r][c] != 0:
                f = M[r][c] / M[c][c]
                M[r] = [a - f * q for a, q in zip(M[r], M[c])]
    return [M[r][n] / M[r][r] for r in range(n)]


def _maxmin_enum(A):
    n, m = len(A), len(A[0])
    ineq = [([ONE if k == t else ZERO for k in range(n)] + [ZERO], ZERO) for t in range(n)]
    ineq += [([-A[t][j] for t in range(n)] + [ONE], ZERO) for j in range(m)]
    eq = ([ONE] * n + [ZERO], ONE)
    top = None
    for comb in itertools.combinations(range(n + m), n):
        rows = [eq] + [ineq[k] for k in comb]
        x = _solve([r[0] for r in rows], [r[1] for r in rows])
        if x is None:
            continue
        tau, v = x[:n], x[n]
        if any(t < 0 for t in tau) or any(v > sum(tau[t] * A[t][j] for t in range(n)) for j in range(m)):
            continue
        if top is None or v > top[0]:
            top = (v, tuple(tau))
    return top


def _recon(x):
    q = [Fr(float(max(xx, 0.0))).limit_denominator(10**6) for xx in x]
    s = sum(q)
    return [qq / s for qq in q] if s > 0 else None


STATS = {"lp_certified": 0, "lp_enum": 0}


def maxmin(A):
    """max over row mixtures tau of min over columns of tau . A[:, j]; exact (value, tau)."""
    n, m = len(A), len(A[0])
    if n == 1:
        return (min(A[0]), (ONE,))
    if m == 1:
        t = max(range(n), key=lambda r: A[r][0])
        return (A[t][0], tuple(ONE if r == t else ZERO for r in range(n)))
    Af = np.array([[float(a) for a in row] for row in A])
    # primal: max v s.t. v <= tau . A_j
    c = np.zeros(n + 1)
    c[-1] = -1
    res = linprog(c, A_ub=np.c_[-Af.T, np.ones(m)], b_ub=np.zeros(m), A_eq=[np.r_[np.ones(n), 0]], b_eq=[1],
                  bounds=[(0, None)] * n + [(None, None)], method="highs")
    # dual: min w s.t. A_t . rho <= w
    c2 = np.zeros(m + 1)
    c2[-1] = 1
    res2 = linprog(c2, A_ub=np.c_[Af, -np.ones(n)], b_ub=np.zeros(n), A_eq=[np.r_[np.ones(m), 0]], b_eq=[1],
                   bounds=[(0, None)] * m + [(None, None)], method="highs")
    if res.success and res2.success:
        tau = _recon(res.x[:n])
        rho = _recon(res2.x[:m])
        if tau and rho:
            vp = min(sum(tau[t] * A[t][j] for t in range(n)) for j in range(m))
            vd = max(sum(rho[j] * A[t][j] for j in range(m)) for t in range(n))
            if vp == vd:
                STATS["lp_certified"] += 1
                return (vp, tuple(tau))
    STATS["lp_enum"] += 1
    return _maxmin_enum(A)


# ------------------------------------------------------------------ guarantees and sets
def subsets(k):
    for r in range(1, k + 1):
        for T in itertools.combinations(range(k), r):
            yield T


def guarantee(G, F, i, want_attain=False):
    """g_i(F) with mixed commitments; returns (g, attained) where attained says some commitment reaches g."""
    shape, u = G
    best, cands = None, []
    oth = others(shape, i)
    for T in subsets(shape[i]):
        Y = [y for y in oth if all(ins(y, i, t) in F for t in T)]
        if not Y:
            continue
        A = [[u[ins(y, i, t)][i] for y in Y] for t in T]
        v, tau = maxmin(A)
        cands.append((v, T, tau))
        if best is None or v > best:
            best = v
    if not want_attain:
        return best, None
    attained = False
    for v, T, tau in cands:
        if v != best:
            continue
        Tp = tuple(t for t, p in zip(T, tau) if p > 0)
        Yp = [y for y in oth if all(ins(y, i, t) in F for t in Tp)]
        full = [ZERO] * shape[i]
        for t, p in zip(T, tau):
            full[t] = p
        val = min(sum(full[t] * u[ins(y, i, t)][i] for t in Tp) for y in Yp)
        if val == best:
            attained = True
    return best, attained


def guarantees(G, F, want_attain=False):
    return [guarantee(G, F, i, want_attain) for i in range(len(G[0]))]


def meet(G, F, g):
    return frozenset(s for s in F if all(G[1][s][i] >= g[i] for i in range(len(g))))


def pure_guarantee(G, F, i):
    shape, u = G
    best = None
    for t in range(shape[i]):
        sec = [s for s in F if s[i] == t]
        if sec:
            v = min(u[s][i] for s in sec)
            best = v if best is None or v > best else best
    return best


def v_mixed(G):
    S = frozenset(profiles(G[0]))
    return [guarantee(G, S, i)[0] for i in range(len(G[0]))]


def umax(G, i):
    return max(G[1][s][i] for s in G[1])


def pure_bliss(G):
    mv = [i for i, k in enumerate(G[0]) if k >= 2]
    return [s for s in G[1] if all(G[1][s][i] == umax(G, i) for i in mv)]


# ------------------------------------------------------------------ is P(F) nonempty? (mixed profiles over F meeting g)
def boxes(G, F, maxsize):
    shape = G[0]
    choices = [[T for T in subsets(shape[j]) if len(T) <= maxsize] for j in range(len(shape))]
    for A in itertools.product(*choices):
        if all(s in F for s in itertools.product(*A)):
            yield A


def _lin_feasible(fs):
    """fs: list of (a, c): a + c*mu >= 0 for mu in [0,1]."""
    cands = {ZERO, ONE}
    for a, c in fs:
        if c != 0:
            r = -a / c
            if 0 <= r <= 1:
                cands.add(r)
    for mu in cands:
        if all(a + c * mu >= 0 for a, c in fs):
            return mu
    return None


def _bilin_feasible(fs):
    """fs: two tuples (a,b,c,d): f = a + b*l + c*m + d*l*m >= 0 on [0,1]^2. Exact; returns a witness or None."""
    def ok(l, m):
        return all(sp.simplify(a + b * l + c * m + d * l * m) >= 0 for a, b, c, d in fs)

    def okr(l, m):
        return all(a + b * l + c * m + d * l * m >= 0 for a, b, c, d in fs)

    cands = set()
    for l in (ZERO, ONE):
        for m in (ZERO, ONE):
            cands.add((l, m))
    for a, b, c, d in fs:
        for l in (ZERO, ONE):  # edges l fixed: root in m
            den, num = c + d * l, a + b * l
            if den != 0 and 0 <= -num / den <= 1:
                cands.add((l, -num / den))
        for m in (ZERO, ONE):
            den, num = b + d * m, a + c * m
            if den != 0 and 0 <= -num / den <= 1:
                cands.add((-num / den, m))
        # degenerate vertical line l0: c + d l0 = 0 and a + b l0 = 0; horizontal: b + d m0 = 0 and a + c m0 = 0
        if d != 0:
            l0 = -c / d
            if a + b * l0 == 0 and 0 <= l0 <= 1:
                cands |= {(l0, ZERO), (l0, ONE)}
                for a2, b2, c2, d2 in fs:
                    den = c2 + d2 * l0
                    if den != 0 and 0 <= -(a2 + b2 * l0) / den <= 1:
                        cands.add((l0, -(a2 + b2 * l0) / den))
            m0 = -b / d
            if a + c * m0 == 0 and 0 <= m0 <= 1:
                cands |= {(ZERO, m0), (ONE, m0)}
                for a2, b2, c2, d2 in fs:
                    den = b2 + d2 * m0
                    if den != 0 and 0 <= -(a2 + c2 * m0) / den <= 1:
                        cands.add((-(a2 + c2 * m0) / den, m0))
        elif c == 0 and b != 0:
            l0 = -a / b
            if 0 <= l0 <= 1:
                cands |= {(l0, ZERO), (l0, ONE)}
        elif b == 0 and c != 0:
            m0 = -a / c
            if 0 <= m0 <= 1:
                cands |= {(ZERO, m0), (ONE, m0)}
    for (l, m) in cands:
        if okr(l, m):
            return (l, m)
    if len(fs) == 2:
        (a1, b1, c1, d1), (a2, b2, c2, d2) = fs
        # (a2 + b2 l)(c1 + d1 l) - (c2 + d2 l)(a1 + b1 l) = 0
        A2 = b2 * d1 - d2 * b1
        A1 = a2 * d1 + b2 * c1 - c2 * b1 - d2 * a1
        A0 = a2 * c1 - c2 * a1
        L = sp.Symbol("L")
        poly = sp.Rational(A2.numerator, A2.denominator) * L**2 + sp.Rational(A1.numerator, A1.denominator) * L \
            + sp.Rational(A0.numerator, A0.denominator)
        if poly != 0:
            for root in sp.solve(poly, L):
                if not root.is_real:
                    continue
                if not (root >= 0 and root <= 1):
                    continue
                den = sp.Rational(c1.numerator, c1.denominator) + sp.Rational(d1.numerator, d1.denominator) * root
                if sp.simplify(den) == 0:
                    den = sp.Rational(c2.numerator, c2.denominator) + sp.Rational(d2.numerator, d2.denominator) * root
                    num = sp.Rational(a2.numerator, a2.denominator) + sp.Rational(b2.numerator, b2.denominator) * root
                else:
                    num = sp.Rational(a1.numerator, a1.denominator) + sp.Rational(b1.numerator, b1.denominator) * root
                if sp.simplify(den) == 0:
                    continue
                mu = sp.simplify(-num / den)
                if mu >= 0 and mu <= 1 and ok(root, mu):
                    return (root, mu)
    return None


def P_nonempty(G, F, g):
    """Is there a mixed profile with support inside F meeting g? Exact for two players; numeric for n >= 3.
    Returns (bool, witness, exact_flag)."""
    shape, u = G
    n = len(shape)
    if not F:
        return False, None, True
    for s in F:
        if all(u[s][i] >= g[i] for i in range(n)):
            return True, s, True
    if n == 2:
        for A1, A2 in boxes(G, F, 2):
            if len(A1) == 1 and len(A2) == 1:
                continue
            if len(A1) == 1 or len(A2) == 1:
                j = 0 if len(A1) == 2 else 1
                Aj = (A1, A2)[j]
                fs = []
                for i in range(2):
                    def pay(t):
                        s = [A1[0], A2[0]]
                        s[j] = t
                        return u[tuple(s)][i]
                    a = pay(Aj[1]) - g[i]
                    c = pay(Aj[0]) - pay(Aj[1])
                    fs.append((a, c))
                w = _lin_feasible(fs)
                if w is not None:
                    return True, (A1, A2, w), True
                continue
            fs = []
            for i in range(2):
                # l = prob of A1[0], m = prob of A2[0]
                x00, x01 = u[(A1[0], A2[0])][i], u[(A1[0], A2[1])][i]
                x10, x11 = u[(A1[1], A2[0])][i], u[(A1[1], A2[1])][i]
                a = x11 - g[i]
                b = x01 - x11
                c = x10 - x11
                d = x00 - x01 - x10 + x11
                fs.append((a, b, c, d))
            w = _bilin_feasible(fs)
            if w is not None:
                return True, (A1, A2, w), True
        return False, None, True
    # numeric for n >= 3: maximize min_i (u_i - g_i) over boxes with supports up to n
    best = -1e9
    rng = np.random.default_rng(7)
    for A in boxes(G, F, n):
        if all(len(a) == 1 for a in A):
            continue
        sizes = [len(a) for a in A]
        idx = np.cumsum([0] + sizes)

        def h(x):
            sig = []
            for j in range(n):
                z = np.abs(x[idx[j]:idx[j + 1]]) + 1e-12
                sig.append(z / z.sum())
            vals = []
            for i in range(n):
                tot = 0.0
                for comb in itertools.product(*[range(k) for k in sizes]):
                    p = 1.0
                    for j, cj in enumerate(comb):
                        p *= sig[j][cj]
                    s = tuple(A[j][cj] for j, cj in enumerate(comb))
                    tot += p * float(u[s][i])
                vals.append(tot - float(g[i]))
            return -min(vals)

        for _ in range(20):
            x0 = rng.random(idx[-1])
            r = minimize(h, x0, method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 4000})
            best = max(best, -r.fun)
    return best >= -1e-9, best, False


# ------------------------------------------------------------------ concepts
def phi(G, carry="pure"):
    """Returns dict: K, F_K, g_K, end ('fixed' | 'mixed-stop' | 'trunc'), history of (F, g), exact flag."""
    S = frozenset(profiles(G[0]))
    F, hist, k = S, [], 0
    while True:
        g = [x[0] for x in guarantees(G, F)]
        hist.append((F, g))
        Fn = meet(G, F, g)
        if carry == "support" and k <= 6:
            Fn = support_union(G, F, g)
        if Fn == F:
            return dict(K=k, F=F, g=g, end="fixed", hist=hist, exact=True)
        if Fn:
            F, k = Fn, k + 1
            if k > 60:
                return dict(K=k, F=F, g=g, end="cap", hist=hist, exact=True)
            continue
        if k == 0:
            return dict(K=0, F=F, g=g, end="mixed-stop", hist=hist, exact=True)
        ne, w, ex = P_nonempty(G, F, g)
        if ne:
            return dict(K=k, F=F, g=g, end="mixed-stop", hist=hist, exact=ex, witness=w)
        Fp, gp = hist[-2]
        return dict(K=k - 1, F=Fp, g=gp, end="trunc", hist=hist, exact=ex)


def support_union(G, F, g):
    """Variant: next F = union of supports of P(F) (two players; boxes up to 2x2 in F with a feasible mixture)."""
    out = set(meet(G, F, g))
    for A in boxes(G, F, 2):
        if all(len(a) == 1 for a in A):
            continue
        sub = frozenset(itertools.product(*A))
        if P_nonempty_box(G, A, g):
            out |= sub
    return frozenset(out)


def P_nonempty_box(G, A, g):
    """Is there a fully mixed profile on the box A (two players) meeting g? Interior points only, via the closed
    box minus faces: approximated by testing the closed box and requiring a witness with all probabilities in (0,1)
    on a grid of step 1/40 plus the exact witness (used only in the stall illustration for the PD)."""
    shape, u = G
    steps = [Fr(k, 40) for k in range(1, 40)]
    ranges = [[(ONE,)] if len(a) == 1 else [(p, 1 - p) for p in steps] for a in A]
    for probs in itertools.product(*ranges):
        sig = []
        for j, a in enumerate(A):
            v = [ZERO] * shape[j]
            for t, p in zip(a, probs[j]):
                v[t] = p
            sig.append(v)
        if all(u_mixed(G, sig, i) >= g[i] for i in range(len(shape))):
            return True
    return False


def rpp_mixed(G, order):
    S = frozenset(profiles(G[0]))
    F, passes = S, 0
    for _ in range(200):
        g = [x[0] for x in guarantees(G, F)]
        Fn = meet(G, F, g)
        if Fn == F:
            return dict(F=F, g=g, end="rest", passes=passes)
        if Fn:
            F = Fn
            continue
        if F == S:
            return dict(F=F, g=g, end="mixed-stop", passes=passes)
        ne, w, ex = P_nonempty(G, F, g)
        if ne:
            return dict(F=F, g=g, end="mixed-stop", passes=passes, exact=ex)
        passes += 1
        for i in order:
            gi = guarantee(G, F, i)[0]
            F2 = frozenset(s for s in F if G[1][s][i] >= gi)
            assert F2, "single verdict emptied the set"
            F = F2
    raise RuntimeError("no stop")


def pte_pure(G):
    S = frozenset(profiles(G[0]))
    F, last = S, S
    while True:
        g = [pure_guarantee(G, F, i) for i in range(len(G[0]))]
        Fn = meet(G, F, g)
        if Fn == F:
            return F, last
        if not Fn:
            return frozenset(), F
        F = Fn
        last = F


def no_ties(G):
    n = len(G[0])
    return all(len({G[1][s][i] for s in G[1]}) == len(G[1]) for i in range(n))


def pareto_pure(G, s):
    n = len(G[0])
    return not any(all(G[1][t][i] >= G[1][s][i] for i in range(n)) and any(G[1][t][i] > G[1][s][i] for i in range(n))
                   for t in G[1])



def guarantee_some(G, F, i):
    """Variant: an answer y is possible after tau if SOME realization (t, y), t in supp(tau), lies in F; the expected
    payoff then also uses payoffs at profiles outside F. Grouped by support T (the max is attained here)."""
    shape, u = G
    best = None
    oth = others(shape, i)
    for T in subsets(shape[i]):
        W = [y for y in oth if any(ins(y, i, t) in F for t in T)]
        if not W or not all(any(ins(y, i, t) in F for y in oth) for t in T):
            continue
        A = [[u[ins(y, i, t)][i] for y in W] for t in T]
        v, tau = maxmin(A)
        best = v if best is None or v > best else best
    return best


def phi_some(G):
    S = frozenset(profiles(G[0]))
    F, hist = S, []
    while True:
        g = [guarantee_some(G, F, i) for i in range(len(G[0]))]
        hist.append((F, g))
        Fn = meet(G, F, g)
        if Fn == F or not Fn:
            return F, g, hist, Fn == F
        F = Fn


def psi_pure(G, vm):
    """Psi (pure branch): first round = pure profiles of IR^m, then Fourny's pure rounds, truncated. None if the
    first round is empty (fallback branch)."""
    S = frozenset(profiles(G[0]))
    F = meet(G, S, vm)
    if not F:
        return None
    while True:
        g = [pure_guarantee(G, F, i) for i in range(len(G[0]))]
        Fn = meet(G, F, g)
        if Fn == F or not Fn:
            return F
        F = Fn


def phiQ_pure(res):
    """Pure members of Phi_Q's prediction: the last nonempty F_k (equal to Phi's pure members except after a
    truncation, where Phi_Q keeps F_k and Phi falls back to P_(k-1))."""
    if res["end"] == "trunc":
        return res["hist"][-1][0]
    if res["end"] == "mixed-stop":
        return frozenset()
    return res["F"]


# ------------------------------------------------------------------ variants 6, 1, 12 (added 2026-11-07)
TOL = 1e-9



# ------------------------------------------------------------------ Seed 6 (two players, floating point LPs)
def s6_ok(G, i, Ai, Aj, c):
    """Is there sigma_i in relint Delta(Ai) with u_i(sigma_i, s_j) >= c for all s_j in Aj?"""
    shape, u = G
    n = len(Ai)
    # variables tau (n), delta ; maximize delta s.t. tau_t >= delta, u_i(tau, s_j) >= c, sum tau = 1
    cobj = np.zeros(n + 1)
    cobj[-1] = -1
    A, b = [], []
    for k in range(n):
        row = np.zeros(n + 1)
        row[k] = -1
        row[-1] = 1
        A.append(row)
        b.append(0)
    for sj in Aj:
        row = np.zeros(n + 1)
        for k, t in enumerate(Ai):
            s = (t, sj) if i == 0 else (sj, t)
            row[k] = -float(u[s][i])
        A.append(row)
        b.append(-float(c))
    res = linprog(cobj, A_ub=A, b_ub=b, A_eq=[np.r_[np.ones(n), 0]], b_eq=[1], bounds=[(0, 1)] * n + [(None, 1)],
                  method="highs")
    return res.success and -res.fun > TOL


def s6_guarantee(G, i, c, okj):
    """g_i of E(c) for Seed 6. okj[(Ai, Aj)] = the other player's condition (exists sigma_j in relint Delta(Aj)
    with u_j(s_i, sigma_j) >= c_j for all s_i in Ai)."""
    shape, u = G
    j = 1 - i
    best = None
    for Ai in subsets(shape[i]):
        n = len(Ai)
        for Gs in itertools.chain.from_iterable(itertools.combinations(range(shape[j]), r)
                                                for r in range(1, shape[j] + 1)):
            V = set()
            for Aj in subsets(shape[j]):
                if set(Aj) <= set(Gs) and okj[(Ai, Aj)]:
                    V |= set(Aj)
            if not V:
                continue
            # region: u_i(tau, s) >= c_i for s in Gs, < c_i otherwise, tau in relint
            def pay_row(sj):
                return [float(u[(t, sj) if i == 0 else (sj, t)][i]) for t in Ai]
            # nonemptiness with slack
            cobj = np.zeros(n + 1)
            cobj[-1] = -1
            A, b = [], []
            for k in range(n):
                row = np.zeros(n + 1)
                row[k] = -1
                row[-1] = 1
                A.append(row)
                b.append(0)
            for sj in range(shape[j]):
                pr = pay_row(sj)
                row = np.zeros(n + 1)
                if sj in Gs:
                    row[:n] = [-x for x in pr]
                    A.append(row)
                    b.append(-float(c[i]))
                else:
                    row[:n] = pr
                    row[-1] = 1
                    A.append(row)
                    b.append(float(c[i]))
            res = linprog(cobj, A_ub=A, b_ub=b, A_eq=[np.r_[np.ones(n), 0]], b_eq=[1],
                          bounds=[(0, 1)] * n + [(None, 1)], method="highs")
            if not (res.success and -res.fun > TOL):
                continue
            # maximize v over the closure
            cobj = np.zeros(n + 1)
            cobj[-1] = -1
            A, b = [], []
            for sj in range(shape[j]):
                pr = pay_row(sj)
                row = np.zeros(n + 1)
                if sj in Gs:
                    row[:n] = [-x for x in pr]
                    A.append(row.copy())
                    b.append(-float(c[i]))
                else:
                    row[:n] = pr
                    A.append(row.copy())
                    b.append(float(c[i]))
                if sj in V:
                    row = np.zeros(n + 1)
                    row[:n] = [-x for x in pr]
                    row[-1] = 1
                    A.append(row)
                    b.append(0)
            res = linprog(cobj, A_ub=A, b_ub=b, A_eq=[np.r_[np.ones(n), 0]], b_eq=[1],
                          bounds=[(0, 1)] * n + [(None, None)], method="highs")
            if res.success:
                v = -res.fun
                best = v if best is None or v > best else best
    return best


def s6_state(G, c):
    shape = G[0]
    ok = [{}, {}]
    for A0 in subsets(shape[0]):
        for A1 in subsets(shape[1]):
            ok[0][(A0, A1)] = s6_ok(G, 0, A0, A1, c[0])  # player 1's own condition
            ok[1][(A1, A0)] = s6_ok(G, 1, A1, A0, c[1])  # player 2's own condition, keyed (own support, other)
    pairs = [(A0, A1) for A0 in subsets(shape[0]) for A1 in subsets(shape[1]) if ok[0][(A0, A1)] and ok[1][(A1, A0)]]
    return ok, pairs


def seed6(G, cap=40):
    """Rounds E(c): {sigma : U_i(sigma) >= c_i}. Returns (history of thresholds, last nonempty support pairs, end)."""
    shape = G[0]
    lo = min(min(float(x) for x in v) for v in G[1].values()) - 1
    c = [lo, lo]
    hist = []
    ok, pairs = s6_state(G, c)
    last = (c, pairs)
    for k in range(cap):
        # player 1's guarantee needs player 2's condition keyed (A1 commitment of player 1 = Ai, Aj = player 2 support)
        okj0 = {(A0, A1): ok[1][(A1, A0)] for A0 in subsets(shape[0]) for A1 in subsets(shape[1])}
        okj1 = {(A1, A0): ok[0][(A0, A1)] for A0 in subsets(shape[0]) for A1 in subsets(shape[1])}
        g = [s6_guarantee(G, 0, c, okj0), s6_guarantee(G, 1, c, okj1)]
        hist.append(g)
        cn = [max(a, b) for a, b in zip(c, g)]
        if all(abs(a - b) <= 1e-9 for a, b in zip(cn, c)):
            return hist, last, "fixed"
        okn, pn = s6_state(G, cn)
        if not pn:
            return hist, last, "trunc"
        c, ok, pairs = cn, okn, pn
        last = (c, pairs)
    return hist, last, "cap"


# ------------------------------------------------------------------ Seed 1 (exact guarantees, graph on pure worlds)
def _relint_attains(G, i, T, Y, g):
    """Is there tau in relint Delta(T) with u_i(tau, y) >= g for all y in Y? (float LP, slack > 1e-9)"""
    shape, u = G
    n = len(T)
    cobj = np.zeros(n + 1)
    cobj[-1] = -1
    A, b = [], []
    for k in range(n):
        row = np.zeros(n + 1)
        row[k] = -1
        row[-1] = 1
        A.append(row)
        b.append(0)
    for y in Y:
        row = np.zeros(n + 1)
        row[:n] = [-float(u[ins(y, i, t)][i]) for t in T]
        A.append(row)
        b.append(-float(g))
    res = linprog(cobj, A_ub=A, b_ub=b, A_eq=[np.r_[np.ones(n), 0]], b_eq=[1], bounds=[(0, 1)] * n + [(None, 1)],
                  method="highs")
    return res.success and -res.fun > TOL


def best_supports(G, F, i, g):
    """Supports T of the commitments that attain g_i(F) on their own section (some tau in relint Delta(T) does);
    if none attains (supremum not attained), the supports whose LP value is g, flagged."""
    shape, u = G
    att, sup = [], []
    for T in subsets(shape[i]):
        Y = [y for y in others(shape, i) if all(ins(y, i, t) in F for t in T)]
        if not Y:
            continue
        A = [[u[ins(y, i, t)][i] for y in Y] for t in T]
        v, tau = maxmin(A)
        if v == g:
            sup.append((T, Y, tau))
            if _relint_attains(G, i, T, Y, g):
                att.append((T, Y, tau))
    return (att, False) if att else (sup, True)


def sccs(nodes, edges):
    index, low, onst, st, res = {}, {}, set(), [], []
    counter = [0]

    def strong(v):
        index[v] = low[v] = counter[0]
        counter[0] += 1
        st.append(v)
        onst.add(v)
        for w in edges[v]:
            if w not in index:
                strong(w)
                low[v] = min(low[v], low[w])
            elif w in onst:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            comp = []
            while True:
                w = st.pop()
                onst.discard(w)
                comp.append(w)
                if w == v:
                    break
            res.append(frozenset(comp))
    sys.setrecursionlimit(100000)
    for v in nodes:
        if v not in index:
            strong(v)
    return res


def seed1(G, realize="support", cap=100):
    """W_0 = S; at each round, edges from each world w to every (t, y), t in T (or supp tau*), y in Y(T), for each
    player below her guarantee and each support T attaining it; W_(k+1) = union of closed classes."""
    shape, u = G
    n = len(shape)
    W = frozenset(profiles(shape))
    hist = []
    for _ in range(cap):
        g = [x[0] for x in guarantees(G, W)]
        bsf = [best_supports(G, W, i, g[i]) for i in range(n)]
        bs = [x[0] for x in bsf]
        flags = [x[1] for x in bsf]
        edges = {w: set() for w in W}
        for w in W:
            for i in range(n):
                if u[w][i] < g[i]:
                    for T, Y, tau in bs[i]:
                        for t in T:
                            for y in Y:
                                edges[w].add(ins(y, i, t))
        comps = sccs(sorted(W), edges)
        closed = [C for C in comps if all(e in C for v in C for e in edges[v])]
        Wn = frozenset().union(*closed)
        hist.append((W, g, closed, any(flags)))
        if Wn == W:
            return W, g, closed, hist, edges
        W = Wn
    raise RuntimeError("cap")


def stationary(C, edges, G, g, bs_rule="uniform"):
    """Stationary law on a closed class under: from w, choose uniformly a player below her guarantee, then uniformly
    an edge of hers (approximated: uniform over all out-edges of w); rest worlds absorbing."""
    nodes = sorted(C)
    idx = {w: k for k, w in enumerate(nodes)}
    m = len(nodes)
    P = [[ZERO] * m for _ in range(m)]
    for w in nodes:
        out = sorted(edges[w])
        if not out:
            P[idx[w]][idx[w]] = ONE
        else:
            for e in out:
                P[idx[w]][idx[e]] += Fr(1, len(out))
    # solve pi P = pi, sum pi = 1
    A = [[P[r][c] - (ONE if r == c else ZERO) for r in range(m)] for c in range(m)]
    A[-1] = [ONE] * m
    b = [ZERO] * (m - 1) + [ONE]
    pi = _solve(A, b)
    return dict(zip(nodes, pi))

# ------------------------------------------------------------------ checks
def check_phi(G, res, vm):
    n = len(G[0])
    errs = []
    g0 = res["hist"][0][1]
    if g0 != vm:
        errs.append("g(F0) != v^m")
    for (F1, g1), (F2, g2) in zip(res["hist"], res["hist"][1:]):
        if any(a > b for a, b in zip(g1, g2)):
            errs.append("guarantee decreased")
    if any(a < b for a, b in zip(res["g"], vm)):
        errs.append("g(F_K) < v^m")
    for s in pure_bliss(G):
        if s not in res["F"] or any(G[1][s][i] < res["g"][i] for i in range(n)):
            errs.append("bliss profile lost")
    for j in range(n):
        if G[0][j] == 1:
            if res["g"][j] != min(G[1][s][j] for s in res["F"]):
                errs.append("one-strategy guarantee")
    return errs


def describe(G, res):
    pm = sorted(meet(G, res["F"], res["g"]))
    return f"K={res['K']} end={res['end']} g={[str(x) for x in res['g']]} pure members={pm}"


def pd(T, R, P, S, T2=None, R2=None, P2=None, S2=None):
    T2, R2, P2, S2 = (T, R, P, S) if T2 is None else (T2, R2, P2, S2)
    tab = {(0, 0): (R, R2), (0, 1): (S, T2), (1, 0): (T, S2), (1, 1): (P, P2)}
    return game((2, 2), lambda s: tab[s])


def td(M, r):
    r = Fr(r)

    def f(s):
        x, y = s[0] + 2, s[1] + 2
        if x == y:
            return (x, x)
        lo = min(x, y)
        return (lo + r, lo - r) if x < y else (lo - r, lo + r)
    return game((M - 1, M - 1), f)


def two_by_two(tab):
    return game((2, 2), lambda s: tab[s])


def main():
    out = []

    def pr(line):
        out.append(line)
        print(line, flush=True)

    pr("Part 1. Prisoner's dilemmas: Phi, RPP^m (both orders) and the union predict (C,C) alone.")
    vals = [Fr(x, 2) for x in range(-4, 21)]
    cnt = cnt_big = 0
    rng = random.Random(20261107)
    grid = [(T, R, P, S) for T in vals for R in vals for P in vals for S in vals if T > R > P > S]
    rng.shuffle(grid)
    for (T, R, P, S) in grid[:300]:
        G = pd(T, R, P, S)
        vm = v_mixed(G)
        res = phi(G)
        assert not check_phi(G, res, vm)
        assert res["end"] == "fixed" and res["F"] == frozenset({(0, 0)}), (T, R, P, S, describe(G, res))
        for order in ((0, 1), (1, 0)):
            rr = rpp_mixed(G, order)
            assert rr["F"] == frozenset({(0, 0)}) and rr["end"] == "rest"
        cnt += 1
        cnt_big += T + S > 2 * R
    for _ in range(200):
        a = sorted(rng.sample(range(-20, 40), 4), reverse=True)
        b = sorted(rng.sample(range(-20, 40), 4), reverse=True)
        G = pd(*[Fr(x, 3) for x in a], *[Fr(x, 3) for x in b])
        res = phi(G)
        assert res["F"] == frozenset({(0, 0)}) and res["end"] == "fixed"
        cnt += 1
    pr(f"  {cnt} PDs (300 symmetric from a half-integer grid in [-2,10], {cnt_big} of them with T+S>2R; 200 asymmetric"
       " with thirds): Phi ends at the fixed point {(C,C)}, P_K = {(C,C)}; RPP^m under both orders rests at {(C,C)}.")
    G = pd(10, 3, 1, 0)
    res = phi(G)
    pr("  PD (10,3,1,0): " + " | ".join(f"F_{k}={sorted(F)} g={[str(x) for x in g]}" for k, (F, g) in
                                         enumerate(res["hist"])))
    resS = phi(G, carry="support")
    pr("  Variant 'F_(k+1) = union of supports of P_k' on PD (10,3,1,0): F_1 = "
       f"{sorted(resS['hist'][1][0]) if len(resS['hist']) > 1 else sorted(resS['F'])}, end={resS['end']}, "
       f"g={[str(x) for x in resS['g']]}: the variant stalls at the first round (mutual defection kept).")

    pr("")
    pr("Part 2. Traveler's dilemmas, claims 2..M: Phi and RPP^m predict (M,M) alone; mixed commitments never raise a")
    pr("  guarantee above the pure one along the rounds (both computed at every round).")
    for M in range(3, 9):
        for r in (Fr(11, 10), Fr(3, 2), Fr(2), Fr(7, 3), Fr(3), Fr(9, 2), Fr(M + 1)):
            G = td(M, r)
            vm = v_mixed(G)
            res = phi(G)
            assert not check_phi(G, res, vm)
            top = (M - 2, M - 2)
            assert res["end"] == "fixed" and res["F"] == frozenset({top}), (M, r, describe(G, res))
            for F, g in res["hist"]:
                gp = [pure_guarantee(G, F, i) for i in range(2)]
                assert gp == g, (M, r, gp, g)
            for order in ((0, 1), (1, 0)):
                rr = rpp_mixed(G, order)
                assert rr["F"] == frozenset({top})
        pr(f"  M={M}: r in {{11/10, 3/2, 2, 7/3, 3, 9/2, M+1}}: Phi = RPP^m = {{(M,M)}}, rounds = "
           f"{len(phi(td(M, 2))['hist'])} for r=2; mixed guarantee = pure guarantee at every round.")

    pr("")
    pr("Part 3. Named games (pure profile labels: 0 = first strategy).")
    named = {
        "matching pennies": two_by_two({(0, 0): (1, -1), (0, 1): (-1, 1), (1, 0): (-1, 1), (1, 1): (1, -1)}),
        "chicken (A first)": two_by_two({(0, 0): (0, 0), (0, 1): (3, 1), (1, 0): (1, 3), (1, 1): (2, 2)}),
        "battle of the sexes (A first)": two_by_two({(0, 0): (0, 0), (0, 1): (2, 3), (1, 0): (3, 2), (1, 1): (1, 1)}),
        "Leader (A first)": two_by_two({(0, 0): (0, 0), (0, 1): (3, 2), (1, 0): (2, 3), (1, 1): (1, 1)}),
        "G_e, e=1/2": two_by_two({(0, 0): (0, 0), (0, 1): (3, 1), (1, 0): (1, 3), (1, 1): (Fr(1, 2), Fr(1, 2))}),
        "Aumann chicken (6,6)": two_by_two({(0, 0): (0, 0), (0, 1): (7, 2), (1, 0): (2, 7), (1, 1): (6, 6)}),
        "ties example (a,b),(a,b'),(a',b),(a',b')": two_by_two({(0, 0): (1, 0), (0, 1): (0, 0), (1, 0): (1, 1),
                                                                (1, 1): (0, 0)}),
        "BoS with ties (2,1),(0,0),(0,0),(1,2)": two_by_two({(0, 0): (2, 1), (0, 1): (0, 0), (1, 0): (0, 0),
                                                              (1, 1): (1, 2)}),
        "stag hunt (4,4),(0,3),(3,0),(3,3)": two_by_two({(0, 0): (4, 4), (0, 1): (0, 3), (1, 0): (3, 0),
                                                         (1, 1): (3, 3)}),
    }
    for name, G in named.items():
        vm = v_mixed(G)
        res = phi(G)
        errs = check_phi(G, res, vm)
        pte, last = pte_pure(G)
        rr = {o: rpp_mixed(G, o) for o in ((0, 1), (1, 0))}
        pr(f"  {name}: v^m={[str(x) for x in vm]}; Phi: {describe(G, res)}; checks {'ok' if not errs else errs}")
        pr("     rounds: " + " | ".join(f"F_{k}={sorted(F)} g={[str(x) for x in g]}" for k, (F, g) in
                                       enumerate(res["hist"])))
        pr(f"     pure PTE limit={sorted(pte)}, pure truncation={sorted(last)}; RPP^m order (1,2): "
           f"{rr[(0, 1)]['end']} F={sorted(rr[(0, 1)]['F'])}; order (2,1): {rr[(1, 0)]['end']} F={sorted(rr[(1, 0)]['F'])}")

    pr("")
    pr("Part 4. The guarantee is a supremum that a single commitment need not attain.")
    pr("  F = {(a,y1),(b,y1),(a,y2)}, u_1(a,y1)=2, u_1(b,y1)=0, u_1(a,y2)=0 (player 2's payoffs 0).")
    G = game((2, 2), lambda s: ({(0, 0): 2, (1, 0): 0, (0, 1): 0, (1, 1): 0}[s], 0))
    F = frozenset({(0, 0), (1, 0), (0, 1)})
    g1, att = guarantee(G, F, 0, want_attain=True)
    pr(f"  g_1(F) = {g1}; attained by a commitment: {att}. Pure a guarantees 0 (answer y2), pure b 0; a mixture "
       "(1-e)a + e b rules out y2 and guarantees 2(1-e).")

    pr("")
    pr("Part 5. All strict 2x2 games (pairs of strict rankings 1..4): Phi's end types, RPP^m, comparisons.")
    perms = list(itertools.permutations([1, 2, 3, 4]))
    cells = [(0, 0), (0, 1), (1, 0), (1, 1)]
    stats = {}
    for p1 in perms:
        for p2 in perms:
            tab = {c: (p1[k], p2[k]) for k, c in enumerate(cells)}
            G = two_by_two(tab)
            vm = v_mixed(G)
            res = phi(G)
            assert not check_phi(G, res, vm)
            pte, last = pte_pure(G)
            key = res["end"]
            stats[key] = stats.get(key, 0) + 1
            if pte:
                stats["pte nonempty"] = stats.get("pte nonempty", 0) + 1
                if res["end"] == "fixed" and res["F"] == pte:
                    stats["pte nonempty, Phi fixed at the PTE limit"] = stats.get(
                        "pte nonempty, Phi fixed at the PTE limit", 0) + 1
                elif any(G[1][s][i] < vm[i] for s in pte for i in range(2)):
                    stats["pte nonempty, PTE limit outside IR^m"] = stats.get("pte nonempty, PTE limit outside IR^m",
                                                                              0) + 1
                else:
                    stats["pte nonempty, Phi differs otherwise"] = stats.get("pte nonempty, Phi differs otherwise",
                                                                             0) + 1
                    if stats["pte nonempty, Phi differs otherwise"] <= 3:
                        pr(f"   differs: {tab} PTE={sorted(pte)} Phi: {describe(G, res)}")
            for o in ((0, 1), (1, 0)):
                rr = rpp_mixed(G, o)
                k2 = "RPP^m " + rr["end"]
                stats[k2] = stats.get(k2, 0) + 1
                if rr["end"] == "rest":
                    Fs = rr["F"]
                    assert len(Fs) == 1 and pareto_pure(G, next(iter(Fs)))
                    assert Fs == meet(G, frozenset(G[1]), rr["g"])
                if rr["passes"]:
                    stats["RPP^m used the order"] = stats.get("RPP^m used the order", 0) + 1
    for k, v in sorted(stats.items()):
        pr(f"  {k}: {v}")
    pr("  (RPP^m counts are over 576 games x 2 orders; every rest point is a single pure profile, Pareto-optimal among"
       " pure profiles and completely self-consistent.)")

    pr("")
    pr("Part 6. Random games with ties (payoffs 0..3), checks of B inside Phi inside IR^m and nonemptiness.")
    rng = random.Random(1107)
    shapes = [(2, 2)] * 400 + [(2, 3)] * 300 + [(3, 3)] * 300 + [(1, 3)] * 50 + [(3, 1)] * 50 + \
             [(2, 2, 2)] * 150 + [(1, 2, 2)] * 50 + [(2, 2, 3)] * 40
    tally = {}
    bad = 0
    for shape in shapes:
        n = len(shape)
        G = game(shape, lambda s: tuple(rng.randint(0, 3) for _ in range(n)))
        vm = v_mixed(G)
        res = phi(G)
        errs = check_phi(G, res, vm)
        if errs:
            bad += 1
            pr(f"   FAIL {shape} {errs}")
        key = (shape, res["end"] + (" K=0" if res["K"] == 0 else " K>=1"), res["exact"])
        tally[key] = tally.get(key, 0) + 1
        if n == 2:
            for o in ((0, 1), (1, 0)):
                rr = rpp_mixed(G, o)
                if rr["end"] == "rest":
                    assert rr["F"] == meet(G, frozenset(G[1]), rr["g"])
                    assert all(rr["g"][i] >= vm[i] for i in range(2))
                    assert all(s in rr["F"] for s in pure_bliss(G))
                kk = (shape, "RPP^m " + rr["end"])
                tally[kk] = tally.get(kk, 0) + 1
    for k, v in sorted(tally.items(), key=lambda kv: str(kv[0])):
        pr(f"  {k}: {v}")
    pr(f"  games failing a check: {bad} of {len(shapes)}")
    pr(f"  LP values certified by exact primal = dual: {STATS['lp_certified']}, by vertex enumeration: "
       f"{STATS['lp_enum']}")

    pr("")
    pr("Part 7. Random strict games (each player ranks the profiles strictly; 3x3: 300 games, 2x2x2: 100 games):")
    pr("  Phi against the PTE limit, Psi (pure branch) and Phi_Q; Pareto optimality of the pure members.")
    rng = random.Random(5)

    def strict_game(shape):
        N = 1
        for k in shape:
            N *= k
        S = profiles(shape)
        perms = [rng.sample(range(1, N + 1), N) for _ in shape]
        return game(shape, lambda s: tuple(perms[i][S.index(s)] for i in range(len(shape))))

    for shape, count in (((3, 3), 300), ((2, 2, 2), 100)):
        c = {}

        def inc(k):
            c[k] = c.get(k, 0) + 1
        for _ in range(count):
            G = strict_game(shape)
            n = len(shape)
            vm = v_mixed(G)
            res = phi(G)
            assert not check_phi(G, res, vm)
            inc(f"Phi end {res['end']}, K={'0' if res['K'] == 0 else '>=1'}")
            pm = meet(G, res["F"], res["g"])
            inc(f"Phi pure members all Pareto-optimal: {all(pareto_pure(G, s) for s in pm)}")
            q = phiQ_pure(res)
            inc(f"Phi_Q pure members all Pareto-optimal: {all(pareto_pure(G, s) for s in q)}")
            pte, last = pte_pure(G)
            ps = psi_pure(G, vm)
            inc(f"Psi pure branch applies: {ps is not None}")
            if pte:
                inside = all(G[1][s][i] >= vm[i] for s in pte for i in range(n))
                same = res["end"] == "fixed" and res["F"] == pte
                inc(f"PTE nonempty; inside IR^m {inside}; Phi = PTE {same}; Psi = PTE {ps == pte}")
                if inside and not same:
                    raised = any(guarantees(G, F)[i][0] > pure_guarantee(G, F, i) for F, g in res["hist"][1:]
                                 for i in range(n))
                    inc(f"   Phi != PTE inside IR^m: a mixed guarantee above the pure one after round 0: {raised}")
            if n == 2:
                for o in ((0, 1), (1, 0)):
                    rr = rpp_mixed(G, o)
                    inc(f"RPP^m end {rr['end']}")
                    if rr["end"] == "rest":
                        assert len(rr["F"]) == 1 and pareto_pure(G, next(iter(rr["F"])))
        for k, v in sorted(c.items()):
            pr(f"  {shape} {k}: {v}")

    pr("")
    pr("Part 8. The variant Phi_some (an answer is possible if SOME realization stays in F; uses payoffs outside F).")
    G = game((2, 2), lambda s: ({(0, 0): 1, (0, 1): 10, (1, 0): 10, (1, 1): 1}[s], 0))
    F = frozenset({(0, 0), (1, 1)})
    pr(f"  F = {{(a,y1),(b,y2)}}, u_1 = 1 on F and 10 at the two excluded profiles: pure guarantees "
       f"{pure_guarantee(G, F, 0)}, Phi_some guarantee {guarantee_some(G, F, 0)}, Phi guarantee {guarantee(G, F, 0)[0]}.")
    bad = 0
    for M in range(3, 8):
        for r in (Fr(3, 2), Fr(2), Fr(7, 3), Fr(5)):
            Fs, gs, hist, fixed = phi_some(td(M, r))
            bad += not (fixed and Fs == frozenset({(M - 2, M - 2)}))
    for (T, R, P, S) in [(5, 3, 1, 0), (10, 3, 1, 0), (4, 3, 2, 1), (Fr(7, 2), 3, 1, -2)]:
        Fs, gs, hist, fixed = phi_some(pd(T, R, P, S))
        bad += not (fixed and Fs == frozenset({(0, 0)}))
    pr(f"  Phi_some in the TDs M=3..7, r in {{3/2,2,7/3,5}} and four PDs: (M,M) and (C,C) alone; failures: {bad}")
    rng = random.Random(88)
    fails = 0
    for shape in [(2, 2)] * 200 + [(3, 3)] * 200 + [(2, 2, 2)] * 50:
        n = len(shape)
        G = game(shape, lambda s: tuple(rng.randint(0, 3) for _ in range(n)))
        vm = v_mixed(G)
        Fs, gs, hist, fixed = phi_some(G)
        ok = hist[0][1] == vm and all(a >= b for a, b in zip(gs, vm))
        ok = ok and all(s in Fs and all(G[1][s][i] >= gs[i] for i in range(n)) for s in pure_bliss(G))
        ok = ok and all(all(a <= b for a, b in zip(h1[1], h2[1])) for h1, h2 in zip(hist, hist[1:]))
        fails += not ok
    pr(f"  Phi_some on 450 random games with ties (2x2, 3x3, 2x2x2): g(F_0) = v^m, guarantees nondecreasing, bliss "
       f"profiles kept, final guarantees >= v^m; failures: {fails}")

    pr("")
    pr("Part 9. Variants 6 (own coin ex ante, others' realizations worst; rounds on mixed profiles, truncation;")
    pr("  floating-point LPs, tolerance 1e-9: numerical) and 1 (recurrent perfect prediction: closed classes of the")
    pr("  graph of departures on pure worlds; exact guarantees, relative-interior attainment by float LP).")
    for M in range(3, 9):
        for r in (Fr(11, 10), Fr(3, 2), Fr(2), Fr(7, 3), Fr(3), Fr(M + 1)):
            G = td(M, r)
            W, g, closed, hist, edges = seed1(G)
            assert W == frozenset({(M - 2, M - 2)}), ("seed1 TD", M, r)
            if M <= 6:
                h, last, end = seed6(G)
                assert end == "fixed" and last[1] == [((M - 2,), (M - 2,))], ("seed6 TD", M, r)
    pr("  TD, r in {11/10, 3/2, 2, 7/3, 3, M+1}: Seed 1 ends at {(M,M)} for M = 3..8; Seed 6 ends at the fixed point "
       "{(M,M)} for M = 3..6.")
    for (T, R, P, S) in [(5, 3, 1, 0), (10, 3, 1, 0), (Fr(7, 2), 3, 1, -2), (4, 3, 2, 1)]:
        G = pd(T, R, P, S)
        W, g, closed, hist, edges = seed1(G)
        h, last, end = seed6(G)
        assert W == frozenset({(0, 0)}) and end == "fixed" and last[1] == [((0,), (0,))]
    pr("  PD (5,3,1,0), (10,3,1,0), (7/2,3,1,-2), (4,3,2,1): Seeds 1 and 6 end at {(C,C)}.")

    def coin_game(T, R, P, S, q):
        base = {(0, 0): (R, R), (0, 1): (S, T), (1, 0): (T, S), (1, 1): (P, P)}
        mix = [{0: ONE}, {0: q, 1: 1 - q}, {1: ONE}]

        def f(s):
            v = [ZERO, ZERO]
            for x, px in mix[s[0]].items():
                for y, py in mix[s[1]].items():
                    for i in range(2):
                        v[i] += px * py * Fr(base[(x, y)][i])
            return tuple(v)
        return game((3, 3), f)
    G = coin_game(10, 3, 1, 0, Fr(9, 10))
    res = phi(G)
    W, g, closed, hist, edges = seed1(G)
    h, last, end = seed6(G)
    pr(f"  PD (10,3,1,0) with the named coin m = 9/10 C + 1/10 D (strategies C, m, D): Phi {res['end']} "
       f"{sorted(res['F'])}; Seed 1 W* = {sorted(W)}; Seed 6 {end} support pairs {last[1]}, thresholds "
       f"{[round(x, 4) for x in last[0]]}: all three at (m,m) (167/50 each).")
    named9 = {
        "matching pennies": two_by_two({(0, 0): (1, -1), (0, 1): (-1, 1), (1, 0): (-1, 1), (1, 1): (1, -1)}),
        "chicken": two_by_two({(0, 0): (0, 0), (0, 1): (3, 1), (1, 0): (1, 3), (1, 1): (2, 2)}),
        "battle of the sexes": two_by_two({(0, 0): (0, 0), (0, 1): (2, 3), (1, 0): (3, 2), (1, 1): (1, 1)}),
        "G_e, e=1/2": two_by_two({(0, 0): (0, 0), (0, 1): (3, 1), (1, 0): (1, 3), (1, 1): (Fr(1, 2), Fr(1, 2))}),
        "BoS with ties": two_by_two({(0, 0): (2, 1), (0, 1): (0, 0), (1, 0): (0, 0), (1, 1): (1, 2)}),
    }
    for name, G in named9.items():
        W, g, closed, hist, edges = seed1(G)
        st = []
        for C in closed:
            if len(C) > 1:
                pi = stationary(C, edges, G, g)
                st.append(({k: str(v) for k, v in pi.items()},
                           [str(sum(pi[w] * G[1][w][i] for w in C)) for i in range(2)]))
        h, last, end = seed6(G)
        pr(f"  {name}: Seed 1 W* = {sorted(W)}, closed classes {[sorted(C) for C in closed]}, stationary laws "
           f"(uniform departures) {st}; Seed 6 {end}, thresholds {[round(x, 4) for x in last[0]]}, support pairs "
           f"{last[1]}")
    rng = random.Random(3)
    c = {}

    def inc(k):
        c[k] = c.get(k, 0) + 1
    example = None
    for shape in [(2, 2)] * 200 + [(2, 3)] * 100 + [(3, 3)] * 150:
        G = game(shape, lambda s: tuple(rng.randint(0, 3) for _ in range(2)))
        vm = v_mixed(G)
        W, g, closed, hist, edges = seed1(G)
        inc((shape, "Seed 1: pure bliss profiles inside W*", all(s in W for s in pure_bliss(G))))
        inc((shape, "Seed 1: some product profile over W* in IR^m (exact)", P_nonempty(G, W, vm)[0]))
        for C in closed:
            if len(C) > 1:
                pi = stationary(C, edges, G, g)
                okk = all(sum(pi[x] * G[1][x][i] for x in C) >= vm[i] for i in range(2))
                inc((shape, "Seed 1: stationary law of a cycle class gives each at least v^m", okk))
                if not okk and example is None:
                    example = (G, C, pi, vm)
        inc((shape, "Seed 1: guarantee not attained at some round (edges from near-optimal supports)",
             any(hh[3] for hh in hist)))
        if shape != (3, 3) or rng.random() < 0.4:
            h, last, end = seed6(G)
            inc((shape, "Seed 6 end", end))
    for k, v in sorted(c.items(), key=str):
        pr(f"  {k}: {v}")
    if example:
        G, C, pi, vm = example
        pr(f"  Seed 1 example: payoffs {{{', '.join(f'{s}: ({G[1][s][0]},{G[1][s][1]})' for s in sorted(G[1]))}}}, "
           f"v^m = {[str(x) for x in vm]}, cycle class {sorted(C)}, stationary {{{', '.join(f'{k}: {v}' for k, v in pi.items())}}}, "
           f"expected payoffs {[str(sum(pi[x] * G[1][x][i] for x in C)) for i in range(2)]}")

    pr("")
    pr("Part 10. Variant 12, first round: worlds meeting the pure security levels, product lotteries over them")
    pr("  meeting the mixed security levels. A game where that set is empty (exact):")
    G = two_by_two({(0, 0): (3, 0), (0, 1): (1, Fr(19, 10)), (1, 0): (0, 3), (1, 1): (3, 0)})
    vm = v_mixed(G)
    S = frozenset(profiles((2, 2)))
    vp = [pure_guarantee(G, S, i) for i in range(2)]
    W = meet(G, S, vp)
    ne, w, ex = P_nonempty(G, W, vm)
    pr(f"  payoffs (a,c) (3,0), (a,d) (1,19/10), (b,c) (0,3), (b,d) (3,0): pure security {[str(x) for x in vp]}, "
       f"mixed security {[str(x) for x in vm]}, worlds {sorted(W)}, lotteries over them meeting v^m: {ne} "
       f"(exact: {ex}); so the seed's first round is empty and its truncation returns every mixed profile.")
    rng = random.Random(3)
    cnt = 0
    for shape in [(2, 2)] * 200 + [(2, 3)] * 100 + [(3, 3)] * 100:
        G = game(shape, lambda s: tuple(rng.randint(0, 3) for _ in range(2)))
        vm = v_mixed(G)
        S = frozenset(profiles(shape))
        vp = [pure_guarantee(G, S, i) for i in range(2)]
        cnt += P_nonempty(G, meet(G, S, vp), vm)[0]
    pr(f"  random games with payoffs 0..3 (400: 2x2, 2x3, 3x3): first round nonempty in {cnt}.")
    return out


if __name__ == "__main__":
    main()
