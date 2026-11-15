"""coop_pte_existence_explore.py: exploratory computations (the census of record
is coop_pte_existence_iff.py).

Logic. On all 362,880 symmetric 3x3 ranking games (and a seeded sample of symmetric 4x4 games without ties):
- the orbit c_0 = security level, c_{k+1} = g(c_k) of coop_pte_existence_iff.py, and the "bad" strategies t != z with
  beta_t < alpha_t < c* (their window (beta_t, alpha_t] is where g exceeds c*: h_t(c) > c* there);
- for the games with a nonempty limit and a bad strategy: which strategy carries the orbit over each window
  (the argmax of the guarantee at the last orbit level <= beta_t): t itself, z, or another one;
- candidate closed forms (each includes P): ROWMIN = every bad t has its row minimum above alpha_t (sufficient);
  GB = g(beta_t) > alpha_t for every bad t (heuristic); MIN = g(c) > alpha_t for every bad t and every level
  c <= beta_t, the level -infinity included, where g = the security level (sufficient: the first orbit level in a
  window would have its predecessor at or below beta_t);
- locality: whether existence is determined by the ordinal types of the three labelled 2x2 restrictions
  {0,1}, {0,2}, {1,2} (copying pays is), the first pair of games with the same types and different existence, and
  the first such pair that differs by the exchange of two adjacent payoff values.
"""
import itertools
import random
from pathlib import Path

import coop_pte_existence_iff as m

OUT = Path(__file__).with_suffix(".out")
_lines = []


def say(x=""):
    print(x, flush=True)
    _lines.append(str(x))


NEG = m.NEG


def h(u, n, t, c):
    sec = [u[t][y] for y in range(n) if m.w(u, t, y) >= c]
    return min(sec) if sec else None


def argmax_g(u, n, c):
    best, arg = None, None
    for t in range(n):
        v = h(u, n, t, c)
        if v is not None and (best is None or v > best):
            best, arg = v, t
    return best, arg


def analyse(u, n):
    z = m.best_sym(u, n)
    cs = u[z][z]
    ex, traj = m.pte_threshold(u, n)
    P = m.cond_P(u, n)
    bad = []
    for t in range(n):
        if t == z:
            continue
        alpha, beta = m.ab(u, n, t, cs)
        if beta < alpha:
            bad.append((t, alpha, beta))
    levels = sorted({m.w(u, x, y) for x in range(n) for y in range(n)} | {NEG})
    self_ok = P and all(min(u[t]) > a for t, a, b in bad)
    gb_ok = P and all(m.g(u, n, b) > a for t, a, b in bad)
    min_ok = P and all(min(m.g(u, n, c) for c in levels if c <= b) > a for t, a, b in bad)
    carriers = []
    if ex:
        orbit = [NEG] + traj
        for t, a, b in bad:
            ck = max(c for c in orbit if c <= b)
            val, arg = argmax_g(u, n, ck)
            carriers.append("self" if arg == t else ("z" if arg == z else "other"))
    return ex, P, bool(bad), self_ok, gb_ok, min_ok, carriers


def restriction_types(u, n):
    key = []
    for S in itertools.combinations(range(n), 2):
        vals = [u[x][y] for x in S for y in S]
        key.append(tuple(sorted(range(4), key=lambda i: vals[i])))
    return tuple(key)


def census(label, games, n, locality):
    cnt = {}
    carry = {}
    groups = {}
    first_pair = None
    for u in games:
        ex, P, has_bad, s, gb, mn, car = analyse(u, n)
        for nm, val in (("ROWMIN", s), ("GB", gb), ("MIN", mn)):
            k = (nm, ex, val)
            cnt[k] = cnt.get(k, 0) + 1
        k = ("bad", ex, has_bad)
        cnt[k] = cnt.get(k, 0) + 1
        for c in car:
            carry[c] = carry.get(c, 0) + 1
        if locality:
            key = restriction_types(u, n)
            if key in groups:
                if groups[key][0] != ex and first_pair is None:
                    first_pair = (groups[key][1], u)
                if groups[key][0] != ex:
                    groups[key] = ("mixed", u)
            else:
                groups[key] = (ex, u)
    say(f"-- {label}")
    for nm in ("bad", "ROWMIN", "GB", "MIN"):
        row = {(ex, val): cnt.get((nm, ex, val), 0) for ex in (True, False) for val in (True, False)}
        say(f"   {nm}: (EX, holds) -> count {row}")
    say(f"   windows jumped, by carrier (EX games with a bad strategy): {carry}")
    if locality:
        mixed = sum(1 for v in groups.values() if v[0] == "mixed")
        say(f"   labelled 2x2 restriction types: {len(groups)} classes, {mixed} with both existence and emptiness")
        if first_pair:
            say(f"   first pair with the same restriction types: rows {first_pair[0]} and rows {first_pair[1]}")


def swap_pair():
    """First 3x3 game (in the census order) with an empty limit that becomes nonempty when two adjacent payoff values
    are exchanged without changing the labelled 2x2 restriction types."""
    for u in m.sym_3x3():
        if m.pte_threshold(u, 3)[0]:
            continue
        flat = [x for r in u for x in r]
        for k in range(1, 9):
            i, j = flat.index(k), flat.index(k + 1)
            f = flat[:]
            f[i], f[j] = f[j], f[i]
            v = [f[0:3], f[3:6], f[6:9]]
            if restriction_types(v, 3) == restriction_types(u, 3) and m.pte_threshold(v, 3)[0]:
                say(f"   adjacent-swap pair: rows {u} (empty limit) and rows {v} (limit {{(z,z)}}), payoffs {k} and {k + 1} exchanged; thresholds {m.pte_threshold(u, 3)[1]} and {m.pte_threshold(v, 3)[1]}")
                return


def main():
    census("symmetric 3x3 ranking games (all 362,880)", m.sym_3x3(), 3, True)
    swap_pair()
    rng = random.Random(20261106)
    census("random symmetric 4x4 games without ties (20,000, seed 20261106)",
           (m.rand_sym(4, rng) for _ in range(20000)), 4, False)
    OUT.write_text("\n".join(_lines) + "\n")


if __name__ == "__main__":
    main()
