"""Shared functions of the independent recomputations of the paper's claims.

Finite games in pure strategies; payoffs exact (int or Fraction).
Objects:
  Game(sizes, u)  u: dict profile -> tuple of payoffs, profile = tuple of strategy indices.
  guarantee m_i(E) = max over strategies t with nonempty section E[t] of min_{s in E[t]} u_i(s)
  step(G, E, B)  = intersection over i in B of the verdicts V_i(E) = {s in E: u_i(s) >= m_i(E)}
  pte: simultaneous steps from S to the limit; seq / resolved: the paper's two order-based processes;
  reachable: every set reachable from S by steps with any nonempty block B.
"""
from fractions import Fraction as F
from itertools import product, permutations, combinations


class Game:
    def __init__(self, sizes, u):
        self.sizes = tuple(sizes)
        self.n = len(sizes)
        self.S = list(product(*[range(k) for k in sizes]))
        self.u = u

    def key(self):
        return (self.sizes, tuple(self.u[s] for s in self.S))


def bimatrix(A, B):
    """Two-player game from row-player matrix A and column-player matrix B."""
    m, k = len(A), len(A[0])
    return Game((m, k), {(x, y): (A[x][y], B[x][y]) for x in range(m) for y in range(k)})


def symmetric(A):
    """Symmetric two-player game: u1(x,y)=A[x][y], u2(x,y)=A[y][x]."""
    m = len(A)
    return Game((m, m), {(x, y): (A[x][y], A[y][x]) for x in range(m) for y in range(m)})


def guarantee(G, E, i):
    mins = {}
    for s in E:
        v = G.u[s][i]
        t = s[i]
        if t not in mins or v < mins[t]:
            mins[t] = v
    return max(mins.values()) if mins else None


def step(G, E, B):
    if not E:
        return frozenset()
    ms = {i: guarantee(G, E, i) for i in B}
    return frozenset(s for s in E if all(G.u[s][i] >= ms[i] for i in B))


def pte(G, strict=False):
    E = frozenset(G.S)
    rounds = [E]
    while True:
        if strict:
            ms = {i: guarantee(G, E, i) for i in range(G.n)} if E else {}
            E2 = frozenset(s for s in E if all(G.u[s][i] > ms[i] for i in range(G.n)))
        else:
            E2 = step(G, E, range(G.n))
        if E2 == E:
            return E, rounds
        E = E2
        rounds.append(E)


def seq(G, order):
    E = frozenset(G.S)
    while True:
        E0 = E
        for i in order:
            E = step(G, E, [i])
        if E == E0:
            return E


def resolved(G, order):
    E = frozenset(G.S)
    N = list(range(G.n))
    while True:
        E2 = step(G, E, N)
        if E2:
            if E2 == E:
                return E
            E = E2
        else:
            E3 = E
            for i in order:
                E3 = step(G, E3, [i])
            assert E3 != E, "pass changed nothing although the simultaneous step empties"
            E = E3


def blocks_sequence_process(G, seqblocks, max_iter=10000):
    """Repeat the sequence of blocks (each a list of players applied simultaneously) until a full
    repetition changes nothing."""
    E = frozenset(G.S)
    for _ in range(max_iter):
        E0 = E
        for B in seqblocks:
            E = step(G, E, B)
        if E == E0:
            return E
    raise RuntimeError("no convergence")


def all_blocks(n):
    out = []
    for k in range(1, n + 1):
        out += [list(c) for c in combinations(range(n), k)]
    return out


def reachable(G):
    start = frozenset(G.S)
    seen = {start}
    stack = [start]
    Bs = all_blocks(G.n)
    while stack:
        E = stack.pop()
        for B in Bs:
            E2 = step(G, E, B)
            if E2 not in seen:
                seen.add(E2)
                stack.append(E2)
    return seen


def is_rest(G, E):
    return all(step(G, E, [i]) == E for i in range(G.n))


def self_consistent(G, E):
    if not E:
        return True
    return all(G.u[s][i] >= guarantee(G, E, i) for s in E for i in range(G.n))


def completely_sc(G, E):
    if not E:
        return False
    ms = [guarantee(G, E, i) for i in range(G.n)]
    return frozenset(s for s in G.S if all(G.u[s][i] >= ms[i] for i in range(G.n))) == E


def best_pay(G, i, t):
    return max(G.u[s][i] for s in G.S if s[i] == t)


def worst_pay(G, i, t):
    return min(G.u[s][i] for s in G.S if s[i] == t)


def security(G, i):
    return max(worst_pay(G, i, t) for t in range(G.sizes[i]))


def U_set(G):
    bp = {(i, t): best_pay(G, i, t) for i in range(G.n) for t in range(G.sizes[i])}
    return frozenset(s for s in G.S if all(bp[(i, t)] <= G.u[s][i]
                                            for i in range(G.n) for t in range(G.sizes[i]) if t != s[i]))


def IR_set(G):
    v = [security(G, i) for i in range(G.n)]
    return frozenset(s for s in G.S if all(G.u[s][i] >= v[i] for i in range(G.n)))


def pareto_dominates(G, s, t):
    """s Pareto-dominates t."""
    return all(G.u[s][i] >= G.u[t][i] for i in range(G.n)) and any(G.u[s][i] > G.u[t][i] for i in range(G.n))


def pareto_optimal(G):
    return frozenset(s for s in G.S if not any(pareto_dominates(G, t, s) for t in G.S))


def no_ties(G):
    return all(len({G.u[s][i] for s in G.S}) == len(G.S) for i in range(G.n))


def relabel(G, tau, f):
    """tau[i] = new index of old player i; f[i][t] = new label of old strategy t of player i."""
    n = G.n
    sizes = [None] * n
    for i in range(n):
        sizes[tau[i]] = G.sizes[i]
    u = {}
    for s in G.S:
        s2 = [None] * n
        for j in range(n):
            s2[tau[j]] = f[j][s[j]]
        pay = [None] * n
        for i in range(n):
            pay[tau[i]] = G.u[s][i]
        u[tuple(s2)] = tuple(pay)
    return Game(sizes, u)


def map_set(G, E, tau, f):
    out = []
    for s in E:
        s2 = [None] * G.n
        for j in range(G.n):
            s2[tau[j]] = f[j][s[j]]
        out.append(tuple(s2))
    return frozenset(out)


def all_relabelings(G):
    n = G.n
    for tau in permutations(range(n)):
        if any(G.sizes[i] != G.sizes[tau[i]] for i in range(n)):
            # relabeled game has different sizes per index; still a relabeling, but not an automorphism
            pass
        for fs in product(*[list(permutations(range(G.sizes[i]))) for i in range(n)]):
            yield tau, fs


def automorphisms(G):
    out = []
    for tau, fs in all_relabelings(G):
        if any(G.sizes[i] != G.sizes[tau[i]] for i in range(G.n)):
            continue
        H = relabel(G, tau, fs)
        if all(H.u[s] == G.u[s] for s in G.S):
            out.append((tau, fs))
    return out


def invariant(G, E, auts):
    return all(map_set(G, E, tau, f) == E for tau, f in auts)


# ---------- symmetric two-player helpers ----------

def copying_pays(A):
    m = len(A)
    for x in range(m):
        for y in range(m):
            if x == y:
                continue
            u1, u2 = A[x][y], A[y][x]
            if u1 < u2 and not (A[y][y] > u1):
                return False
            if u2 < u1 and not (A[x][x] > u2):
                return False
    return True


def best_symmetric(A):
    m = len(A)
    z = max(range(m), key=lambda x: A[x][x])
    return (z, z)


# ---------- the two dilemmas ----------

def pd(T, R, P, S, T2=None, R2=None, P2=None, S2=None):
    T2 = T if T2 is None else T2
    R2 = R if R2 is None else R2
    P2 = P if P2 is None else P2
    S2 = S if S2 is None else S2
    # strategy 0 = C, 1 = D
    u = {(0, 0): (R, R2), (0, 1): (S, T2), (1, 0): (T, S2), (1, 1): (P, P2)}
    return Game((2, 2), u)


def td(M, r):
    """Traveler's dilemma, claims 2..M encoded as index claim-2."""
    k = M - 1
    u = {}
    for a in range(k):
        for b in range(k):
            x, y = a + 2, b + 2
            if x == y:
                u[(a, b)] = (F(x), F(y))
            elif x < y:
                u[(a, b)] = (x + r, x - r)
            else:
                u[(a, b)] = (y - r, y + r)
    return Game((k, k), u)
