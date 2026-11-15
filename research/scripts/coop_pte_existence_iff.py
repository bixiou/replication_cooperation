"""coop_pte_existence_iff.py: when is the limit of the perfectly transparent equilibrium (PTE) nonempty in a
symmetric two-player game without ties?
(2026-11-04.)

Logic.
- A symmetric game is a square matrix u, u[x][y] the payoff of a player choosing x against y; player 1 gets u[x][y]
  and player 2 gets u[y][x] at the profile (x, y). No ties: the n*n entries are distinct (the census uses the
  permutations of 1..n*n).
- Two independent computations of the PTE limit:
  (R) the rounds as defined in the paper, with each player's guarantee computed separately:
      m_i(E) = max over strategies t of i with a nonempty section of min of u_i over the section,
      E' = {s in E : u_i(s) >= m_i(E) for both i}, until E' = E or E' is empty;
  (S) the one-dimensional reduction: with w(s) = min(u_1(s), u_2(s)) and E(c) = {s : w(s) >= c}, the rounds are
      E(c_0), E(c_1), ... with c_0 = player 1's security level and c_{k+1} = g(c_k), g(c) = player 1's guarantee in
      E(c); the limit is nonempty iff c_k <= c* = u(z, z) for all k, z the strategy with the largest diagonal payoff.
  The script checks that (R) and (S) agree on every game, and that every nonempty limit is {(z, z)}.
- Conditions compared with existence (EX):
  CP  copying pays (Proposition prop:copy);
  P   (z, z) is the only profile that gives both players at least c* (equivalently the unique maximizer of w);
  A   g(c) <= c* for every level c <= c* (computed directly on all levels, and in closed form:
      for t != z, beta_t = max{w(t,y) : u(t,y) < c*}, alpha_t = max{w(t,y) : u(t,y) > c*} (-inf if none),
      A fails at t iff beta_t < min(alpha_t, c*));
  Amb A in the "most balanced" form of the note: for every t != z, the profile (t, y) with the largest
      w(t, y) gives the player using t less than c* (checked equal to A on every game);
  Av  A above the security level v: g(c) <= c* for every level c with v <= c <= c*;
  Pairwise versions: A and CP in every symmetric 2x2 restriction {x, y} of the game.
- Classes: all 24 strict symmetric 2x2 games, all 362,880 symmetric 3x3 ranking games (exact); random symmetric
  4x4 and 5x5 games without ties (seeded samples, numerical evidence only).
"""
import itertools
import random
import sys
from pathlib import Path

OUT = Path(__file__).with_suffix(".out")
_lines = []


def say(x=""):
    print(x, flush=True)
    _lines.append(str(x))


NEG = float("-inf")


def pte_rounds(u, n):
    """(R): rounds with both players' guarantees; returns the limit as a frozenset of profiles."""
    E = frozenset((x, y) for x in range(n) for y in range(n))
    while True:
        m1 = max(min(u[x][y] for (a, y) in E if a == x) for x in range(n) if any(a == x for (a, _) in E))
        m2 = max(min(u[y][x] for (x, b) in E if b == y) for y in range(n) if any(b == y for (_, b) in E))
        F = frozenset((x, y) for (x, y) in E if u[x][y] >= m1 and u[y][x] >= m2)
        if F == E or not F:
            return F
        E = F


def w(u, x, y):
    return min(u[x][y], u[y][x])


def g(u, n, c):
    """Player 1's guarantee in E(c); None if E(c) is empty."""
    best = None
    for t in range(n):
        sec = [u[t][y] for y in range(n) if w(u, t, y) >= c]
        if sec:
            m = min(sec)
            best = m if best is None or m > best else best
    return best


def best_sym(u, n):
    return max(range(n), key=lambda x: u[x][x])


def pte_threshold(u, n):
    """(S): returns (nonempty?, trajectory of thresholds)."""
    z = best_sym(u, n)
    cs = u[z][z]
    c = g(u, n, NEG)
    traj = [c]
    while True:
        if c > cs:
            return False, traj
        c2 = g(u, n, c)
        # E(c2) == E(c) iff no profile has w in [c, c2)
        if not any(c <= w(u, x, y) < c2 for x in range(n) for y in range(n)):
            return True, traj
        traj.append(c2)
        c = c2


def cond_cp(u, n):
    return all(not (u[x][y] < u[y][x]) or u[y][y] > u[x][y] for x in range(n) for y in range(n) if x != y)


def cond_P(u, n):
    z = best_sym(u, n)
    cs = u[z][z]
    return all((x, y) == (z, z) or w(u, x, y) < cs for x in range(n) for y in range(n))


def cond_A_direct(u, n):
    z = best_sym(u, n)
    cs = u[z][z]
    levels = sorted({w(u, x, y) for x in range(n) for y in range(n)} | {NEG})
    return all(g(u, n, c) <= cs for c in levels if c <= cs)


def ab(u, n, t, cs):
    beta = max((w(u, t, y) for y in range(n) if u[t][y] < cs), default=NEG)
    alpha = max((w(u, t, y) for y in range(n) if u[t][y] > cs), default=NEG)
    return alpha, beta


def cond_A_closed(u, n):
    z = best_sym(u, n)
    cs = u[z][z]
    for t in range(n):
        if t == z:
            continue
        alpha, beta = ab(u, n, t, cs)
        if beta < min(alpha, cs):
            return False
    return True


def restrict(u, S):
    return [[u[x][y] for y in S] for x in S]


def cond_A_pairwise(u, n):
    return all(cond_A_closed(restrict(u, S), 2) for S in itertools.combinations(range(n), 2))


def cond_cp_pairwise(u, n):
    return all(cond_cp(restrict(u, S), 2) for S in itertools.combinations(range(n), 2))


CONDS = {}  # filled in main after cond_B is defined


def check(u, n, stats, examples, conds):
    ex_r = pte_rounds(u, n)
    ex_s, traj = pte_threshold(u, n)
    z = best_sym(u, n)
    assert bool(ex_r) == ex_s, (u, ex_r, ex_s)
    assert not ex_r or ex_r == frozenset({(z, z)}), (u, ex_r)
    a1, a2, a3 = cond_A_direct(u, n), cond_A_closed(u, n), cond_A_mb(u, n)
    assert a1 == a2 == a3, (u, a1, a2, a3)
    key = (ex_s,) + tuple(conds[k](u, n) for k in conds)
    stats[key] = stats.get(key, 0) + 1
    examples.setdefault(key, (u, traj))


def report(label, stats, examples, conds):
    names = list(conds)
    total = sum(stats.values())
    say(f"-- {label}: {total} games")
    ex_count = sum(v for k, v in stats.items() if k[0])
    say(f"   EX (nonempty limit): {ex_count}; empty: {total - ex_count}")
    for j, nm in enumerate(names, start=1):
        both = sum(v for k, v in stats.items() if k[0] and k[j])
        c_not_ex = sum(v for k, v in stats.items() if (not k[0]) and k[j])
        ex_not_c = sum(v for k, v in stats.items() if k[0] and not k[j])
        say(f"   {nm}: holds in {both + c_not_ex}; EX and {nm} {both}; {nm} without EX {c_not_ex}; EX without {nm} {ex_not_c}")
        for kind, test in ((f"{nm} without EX", lambda k: (not k[0]) and k[j]), (f"EX without {nm}", lambda k: k[0] and not k[j])):
            ks = [k for k in stats if test(k)]
            if ks:
                u, traj = examples[ks[0]]
                say(f"      first {kind}: rows {u}, thresholds {traj}")


def sym_2x2():
    for p in itertools.permutations(range(1, 5)):
        yield [[p[0], p[1]], [p[2], p[3]]]


def sym_3x3():
    for p in itertools.permutations(range(1, 10)):
        yield [list(p[0:3]), list(p[3:6]), list(p[6:9])]


def rand_sym(n, rng):
    p = rng.sample(range(1, n * n + 1), n * n)
    return [p[i * n:(i + 1) * n] for i in range(n)]


def cond_A_mb(u, n):
    z = best_sym(u, n)
    cs = u[z][z]
    for t in range(n):
        if t != z:
            b = max(range(n), key=lambda y: w(u, t, y))
            if u[t][b] > cs:
                return False
    return True


def cond_A_v(u, n):
    z = best_sym(u, n)
    cs = u[z][z]
    v = g(u, n, NEG)
    levels = sorted({w(u, x, y) for x in range(n) for y in range(n)} | {v})
    return all(g(u, n, c) <= cs for c in levels if v <= c <= cs)


def main():
    conds = {"CP": cond_cp, "P": cond_P, "A": cond_A_closed, "Amb": cond_A_mb, "Av": cond_A_v,
             "A2x2": cond_A_pairwise, "CP2x2": cond_cp_pairwise}
    for label, gen in (("strict symmetric 2x2", sym_2x2()), ("symmetric 3x3 ranking games", sym_3x3())):
        stats, examples = {}, {}
        n = 2 if "2x2" in label else 3
        for u in gen:
            check(u, n, stats, examples, conds)
        report(label, stats, examples, conds)
    for n, N, seed in ((4, 100000, 20261104), (5, 20000, 20261105)):
        rng = random.Random(seed)
        stats, examples = {}, {}
        for _ in range(N):
            check(rand_sym(n, rng), n, stats, examples, conds)
        report(f"random symmetric {n}x{n} games without ties (seed {seed})", stats, examples, conds)
    OUT.write_text("\n".join(_lines) + "\n")


if __name__ == "__main__":
    main()
