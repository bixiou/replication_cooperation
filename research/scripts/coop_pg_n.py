"""coop_pg_n.py: binary public goods games with n players, perfect prediction.

u_i(x) = 1 - x_i + a * sum_j x_j, x_i in {0 (free ride), 1 (contribute)}, return a per unit to each player
(multiplier r = n a, with 1 < r < n as in Fourny 2020, Section 6.2, i.e. 1/n < a < 1).
For n = 2..5 and a on a grid of (1/n, 1) avoiding the tie points a = 1/k, a = k/(k+1) style coincidences only where
they occur (listed), report: the PTE path (sizes) and limit; sequential (SPP) and resolved (RPP) perfect prediction
under every order, summarized as the number of contributors and the positions of the free riders in the order;
whether some nonempty reachable rest point is invariant under all permutations of the players (blocked or not);
and the same after a tie-free perturbation (player i's payoff at profile x plus eps * h(i, x), h a fixed injective
pattern, eps = 1/10000).
"""
import itertools
from fractions import Fraction as Fr
from pathlib import Path

OUT = Path(__file__).with_suffix(".out")
_lines = []


def say(x=""):
    print(x)
    _lines.append(str(x))


def guarantee(U, E, i):
    best = None
    for t in (0, 1):
        sec = [U[s][i] for s in E if s[i] == t]
        if sec:
            v = min(sec)
            best = v if best is None or v > best else best
    return best


def verdict(U, E, i):
    if not E:
        return frozenset()
    m = guarantee(U, E, i)
    return frozenset(s for s in E if U[s][i] >= m)


def step(U, E, B):
    out = frozenset(E)
    for i in B:
        out &= verdict(U, E, i)
    return out


def pte_path(U, n):
    E = frozenset(U)
    path = [E]
    while True:
        F = step(U, E, range(n))
        if F == E or not F:
            if F != E:
                path.append(F)
            return path
        path.append(F)
        E = F


def spp(U, order):
    E = frozenset(U)
    while True:
        old = E
        for i in order:
            E = verdict(U, E, i)
        if E == old:
            return E


def rpp(U, n, order):
    E = frozenset(U)
    while True:
        F = step(U, E, range(n))
        if not F:
            F = E
            for i in order:
                F = verdict(U, F, i)
        if F == E:
            return E
        E = F


def reachable(U, n):
    blocks = [B for r in range(1, n + 1) for B in itertools.combinations(range(n), r)]
    S = frozenset(U)
    seen, todo = {S}, [S]
    while todo:
        E = todo.pop()
        for B in blocks:
            F = step(U, E, B)
            if F not in seen:
                seen.add(F)
                todo.append(F)
    return seen


def is_rest(U, E, n):
    return all(verdict(U, E, i) == E for i in range(n))


def invariant(E, n):
    return all(frozenset(tuple(s[p[i]] for i in range(n)) for s in E) == E
               for p in itertools.permutations(range(n)))


def game(n, a, eps=Fr(0)):
    U = {}
    for x in itertools.product((0, 1), repeat=n):
        tot = sum(x)
        code = int("".join(map(str, x)), 2)
        U[x] = tuple(1 - x[i] + a * tot + eps * (code * n + i + 1) for i in range(n))
    return U


def has_ties(U, n):
    return any(len({U[s][i] for s in U}) < len(U) for i in range(n))


def summarize(pred, order):
    """free riders by position in the order (1 = first reasoner)."""
    out = []
    for s in sorted(pred):
        pos = sorted(order.index(i) + 1 for i in range(len(s)) if s[i] == 0)
        out.append((sum(s), tuple(pos)))
    return tuple(out)


def main():
    say(__doc__.strip().splitlines()[0])
    for n in range(2, 6):
        say()
        say(f"== n = {n}")
        grid = sorted({Fr(k, 20) for k in range(1, 20)} | {Fr(1, n) + Fr(1, 100), Fr(99, 100)})
        grid = [a for a in grid if Fr(1, n) < a < 1]
        for a in grid:
            for eps in (Fr(0), Fr(1, 10000)):
                U = game(n, a, eps)
                pth = pte_path(U, n)
                lim = pth[-1]
                S = {}
                R = {}
                for order in itertools.permutations(range(n)):
                    S.setdefault(summarize(spp(U, order), order), 0)
                    S[summarize(spp(U, order), order)] += 1
                    R.setdefault(summarize(rpp(U, n, order), order), 0)
                    R[summarize(rpp(U, n, order), order)] += 1
                line = (f"a = {a}, eps = {eps}: ties {has_ties(U, n)}; PTE sizes {[len(E) for E in pth]}, limit "
                        f"{sorted(lim) if len(lim) <= 4 else str(len(lim)) + ' profiles'}; "
                        f"SPP (contributors, free riders' positions): {S}; RPP: {R}")
                if n <= 4:
                    Rch = reachable(U, n)
                    inv = [E for E in Rch if E and is_rest(U, E, n) and invariant(E, n)]
                    line += f"; invariant nonempty reachable rest points: {[sorted(E) for E in inv]}"
                say(line)

    OUT.write_text("\n".join(_lines) + "\n")


if __name__ == "__main__":
    main()
