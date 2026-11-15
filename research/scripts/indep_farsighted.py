# Independent check of the farsighted stability claims of the paper (Online Appendix, Section S6),
# implemented directly from the definitions of Chwe (1994):
#   a ->_S d : d != a and d differs from a only in the coordinates of S.
#   a << b (indirect domination): chain a=a0 ->_{S0} a1 ->_{S1} ... -> am=b with every member of S_k
#     strictly preferring b to a_k.
#   Y consistent: a in Y iff for every move a ->_S d there is e in Y with (e == d or d << e) such that
#     NOT every member of S strictly prefers e to a.  LCS = largest fixed point (iterate f from Z).
#   Farsighted stable set: vNM stable set of <<: internal (no a, b in V with a << b, a = b allowed),
#     external (every c outside V has a in V with c << a).
# Coalitions: "single" = {1},{2}; "joint" = {1},{2},{1,2}.
import itertools
from fractions import Fraction

def game_profiles(strats):
    return list(itertools.product(*strats))

def moves_from(a, strats, coalitions):
    out = []
    for S in coalitions:
        choices = [strats[i] if i in S else [a[i]] for i in range(len(a))]
        for d in itertools.product(*choices):
            if d != a:
                out.append((S, d))
    return out

def prefers_all(u, S, x, y):
    # every member of S strictly prefers y to x
    return all(u(i, y) > u(i, x) for i in S)

def indirect_domination(Z, strats, coalitions, u):
    # dom[b] = set of a with a << b
    mv = {a: moves_from(a, strats, coalitions) for a in Z}
    dom = {}
    for b in Z:
        R = set()
        changed = True
        while changed:
            changed = False
            for a in Z:
                if a in R:
                    continue
                for S, d in mv[a]:
                    if (d == b or d in R) and prefers_all(u, S, a, b):
                        R.add(a); changed = True; break
        dom[b] = R
    return dom, mv

def lcs(Z, mv, dom, u):
    # above[d] = set of e with d << e
    above = {d: set() for d in Z}
    for b in Z:
        for a in dom[b]:
            above[a].add(b)
    Y = set(Z)
    while True:
        newY = set()
        for a in Y | set(Z):
            ok = True
            for S, d in mv[a]:
                cands = ({d} | above[d]) & Y
                if not any(not prefers_all(u, S, a, e) for e in cands):
                    ok = False; break
            if ok:
                newY.add(a)
        newY &= Y  # f is isotonic and f(Z) subset Z; iterate downward
        if newY == Y:
            return Y
        Y = newY

def stable_sets(Z, dom, limit=None):
    # all V with internal and external stability w.r.t. <<; backtracking with propagation
    idx = {z: k for k, z in enumerate(Z)}
    n = len(Z)
    above = [set() for _ in range(n)]   # above[a] = {b : a << b}
    below = [set() for _ in range(n)]   # below[b] = {a : a << b}
    for b in Z:
        for a in dom[b]:
            above[idx[a]].add(idx[b]); below[idx[b]].add(idx[a])
    conflict = [above[k] | below[k] for k in range(n)]
    results = []
    # state: 1 in, 0 out, None undecided
    def propagate(state):
        changed = True
        while changed:
            changed = False
            for k in range(n):
                if state[k] == 1:
                    if k in above[k]:
                        return False
                    for m in conflict[k]:
                        if state[m] == 1 and m != k:
                            return False
                        if state[m] is None:
                            state[m] = 0; changed = True
            for k in range(n):
                if state[k] == 1:
                    continue
                # k must be in V or dominated by some member of V: possible dominators = above[k] not out
                poss = [m for m in above[k] if state[m] != 0]
                if any(state[m] == 1 for m in poss):
                    continue
                if not poss:
                    if state[k] == 0:
                        return False
                    if state[k] is None:
                        if k in above[k]:
                            return False
                        state[k] = 1; changed = True
        return True
    def rec(state):
        if limit is not None and len(results) >= limit:
            return
        st = list(state)
        if not propagate(st):
            return
        und = [k for k in range(n) if st[k] is None]
        if not und:
            # verify
            V = [k for k in range(n) if st[k] == 1]
            Vs = set(V)
            assert all(not (above[k] & Vs) for k in V)
            assert all((above[k] & Vs) for k in range(n) if k not in Vs)
            results.append(frozenset(Z[k] for k in V))
            return
        k = und[0]
        for val in (1, 0):
            st2 = list(st); st2[k] = val
            rec(st2)
    rec([None] * n)
    return results

def stable_sets_brute(Z, dom):
    res = []
    for r in range(len(Z) + 1):
        for V in itertools.combinations(Z, r):
            Vs = set(V)
            if any(a in dom[b] for a in V for b in V):
                continue
            if all(any(c in dom[a] for a in V) for c in Z if c not in Vs):
                res.append(frozenset(V))
    return res

def U_set(Z, strats, u):
    out = set()
    for s in Z:
        ok = True
        for i in range(len(s)):
            for t in strats[i]:
                if t == s[i]:
                    continue
                best = max(u(i, z) for z in Z if z[i] == t)
                if best > u(i, s):
                    ok = False
        if ok:
            out.add(s)
    return out

def IR_set(Z, strats, u):
    v = [max(min(u(i, z) for z in Z if z[i] == t) for t in strats[i]) for i in range(len(strats))]
    return {s for s in Z if all(u(i, s) >= v[i] for i in range(len(strats)))}

SINGLE = [(0,), (1,)]
JOINT = [(0,), (1,), (0, 1)]

def fmt(V):
    return sorted(V)

print("=" * 70)
print("(a) Prisoner's dilemma (T,R,P,S) = (5,3,1,0)")
pdpay = {('C', 'C'): (3, 3), ('C', 'D'): (0, 5), ('D', 'C'): (5, 0), ('D', 'D'): (1, 1)}
strats = [['C', 'D'], ['C', 'D']]
Z = game_profiles(strats)
u = lambda i, s: pdpay[s][i]
for name, coal in (("single movers", SINGLE), ("joint moves", JOINT)):
    dom, mv = indirect_domination(Z, strats, coal, u)
    print(f"-- {name}")
    for b in Z:
        print(f"   dominated by {b}: {sorted(dom[b])}")
    L = lcs(Z, mv, dom, u)
    print(f"   LCS = {fmt(L)}")
    ss = stable_sets_brute(Z, dom)
    print(f"   all farsighted stable sets (exhaustive over 16 subsets): {[fmt(V) for V in ss]}")

print("=" * 70)
print("(b) Traveler's dilemma, claims 2..M, M = 10")
M = 10
for r in (2, 5):
    strats = [list(range(2, M + 1)), list(range(2, M + 1))]
    Z = game_profiles(strats)
    def tdu(i, s, r=r):
        x, y = s
        if x == y:
            return x
        lo = min(x, y)
        me = s[i]
        return lo + r if me == lo else lo - r
    for name, coal in (("single movers", SINGLE), ("joint moves", JOINT)):
        dom, mv = indirect_domination(Z, strats, coal, tdu)
        L = lcs(Z, mv, dom, tdu)
        print(f"-- r = {r}, {name}")
        print(f"   LCS ({len(L)} profiles) = {fmt(L)}")
        print(f"   (M-1,M) in LCS: {(M-1, M) in L};  (M,M) in LCS: {(M, M) in L}")
        for cand in ([(M, M)], [(M - 1, M)], [(M, M - 1)]):
            Vs = set(cand)
            internal = not any(a in dom[b] for a in cand for b in cand)
            external = all(any(c in dom[a] for a in cand) for c in Z if c not in Vs)
            print(f"   {cand} stable set: internal {internal}, external {external}")
        ss = stable_sets(Z, dom)
        print(f"   ALL farsighted stable sets (exhaustive backtracking, verified): {len(ss)}")
        for V in ss:
            print(f"     {fmt(V)}")
        U = U_set(Z, strats, tdu); IR = IR_set(Z, strats, tdu)
        print(f"   U(G) = {fmt(U)}; LCS subset IR: {L <= IR}; U subset LCS: {U <= L}")

print("=" * 70)
print("(c) Indifference example: (0,0) at (a,b),(a,b'); (0,1) at (a',b); (1,1) at (a',b')")
pay = {('a', 'b'): (0, 0), ('a', "b'"): (0, 0), ("a'", 'b'): (0, 1), ("a'", "b'"): (1, 1)}
strats = [['a', "a'"], ['b', "b'"]]
Z = game_profiles(strats)
u = lambda i, s: pay[s][i]
U = U_set(Z, strats, u); IR = IR_set(Z, strats, u)
print(f"   U(G) = {fmt(U)}; IR(G) = {fmt(IR)}")
s0 = ("a'", 'b'); target = ("a'", "b'")
# explicit chain check: (a',b) ->1 (a,b) ->2 (a,b') ->1 (a',b'), each mover strictly prefers target
chain = [(("a'", 'b'), 0, ('a', 'b')), (('a', 'b'), 1, ('a', "b'")), (('a', "b'"), 0, ("a'", "b'"))]
ok = all(u(i, target) > u(i, x) and all(x[j] == y[j] for j in range(2) if j != i) for x, i, y in chain)
print(f"   explicit chain (a',b)->(a,b)->(a,b')->(a',b') valid: {ok}")
for name, coal in (("single movers", SINGLE), ("joint moves", JOINT)):
    dom, mv = indirect_domination(Z, strats, coal, u)
    L = lcs(Z, mv, dom, u)
    ss = stable_sets_brute(Z, dom)
    print(f"-- {name}: (a',b) << (a',b'): {s0 in dom[target]}")
    for b in Z:
        print(f"   dominated by {b}: {sorted(dom[b])}")
    print(f"   LCS = {fmt(L)};  (a',b) in LCS: {s0 in L};  U subset LCS: {U <= L}")
    print(f"   stable sets: {[fmt(V) for V in ss]};  (a',b) in some stable set: {any(s0 in V for V in ss)}")

print("=" * 70)
print("(d) Unilaterality sanity checks on 2x2 classes")
def run_class(name, games, coal_list):
    strats = [[0, 1], [0, 1]]
    Z = game_profiles(strats)
    for cname, coal in coal_list:
        cnt = dict(n=0, lcs_empty=0, lcs_notIR=0, U_not_lcs=0, viol=0,
                   ss_none=0, ss_union_viol=0, ss_some_viol=0, ss_every_viol=0, ss_notIR=0, ss_U=0)
        for g in games:
            u = lambda i, s, g=g: g[s][i]
            dom, mv = indirect_domination(Z, strats, coal, u)
            L = lcs(Z, mv, dom, u)
            U = U_set(Z, strats, u); IR = IR_set(Z, strats, u)
            cnt['n'] += 1
            if not L: cnt['lcs_empty'] += 1
            a = not (L <= IR); b = not (U <= L)
            cnt['lcs_notIR'] += a; cnt['U_not_lcs'] += b; cnt['viol'] += (a or b)
            ss = stable_sets_brute(Z, dom)
            if not ss:
                cnt['ss_none'] += 1
            else:
                vio = [not (U <= V and V <= IR) for V in ss]
                if any(vio): cnt['ss_some_viol'] += 1
                if all(vio): cnt['ss_every_viol'] += 1
                un = frozenset().union(*ss)
                if not (U <= un and un <= IR): cnt['ss_union_viol'] += 1
                if any(not (V <= IR) for V in ss): cnt['ss_notIR'] += 1
                if any(not (U <= V) for V in ss): cnt['ss_U'] += 1
        print(f"-- {name}, {cname}: {cnt}")

prof = [(0, 0), (0, 1), (1, 0), (1, 1)]
g012 = []
for vals in itertools.product(range(3), repeat=8):
    g012.append({prof[k]: (vals[2 * k], vals[2 * k + 1]) for k in range(4)})
run_class("2x2 payoffs in {0,1,2} (6,561 games)", g012, [("single movers", SINGLE), ("joint moves", JOINT)])
strict = []
for r1 in itertools.permutations(range(4)):
    for r2 in itertools.permutations(range(4)):
        strict.append({prof[k]: (r1[k], r2[k]) for k in range(4)})
run_class("strict 2x2 (576 games)", strict, [("single movers", SINGLE), ("joint moves", JOINT)])
g0123 = []
for vals in itertools.product(range(4), repeat=8):
    g0123.append({prof[k]: (vals[2 * k], vals[2 * k + 1]) for k in range(4)})
run_class("2x2 payoffs in {0,1,2,3} (65,536 games)", g0123, [("single movers", SINGLE)])
print("done")

print("=" * 70)
print("(e) A strict 2x2 game in which some farsighted stable set (single movers) drops a profile of U(G)")
strats = [[0, 1], [0, 1]]
Z = game_profiles(strats)
for g in strict:
    u = lambda i, s, g=g: g[s][i]
    dom, mv = indirect_domination(Z, strats, SINGLE, u)
    U = U_set(Z, strats, u)
    ss = stable_sets_brute(Z, dom)
    bad = [V for V in ss if not U <= V]
    if bad:
        print(f"   payoffs {[(s, g[s]) for s in Z]}")
        print(f"   U(G) = {fmt(U)}; LCS = {fmt(lcs(Z, mv, dom, u))}; stable sets = {[fmt(V) for V in ss]}")
        break
print("(f) 2x2 payoffs in {0,1,2,3}, joint moves, LCS only")
cnt = dict(n=0, lcs_empty=0, lcs_notIR=0, U_not_lcs=0)
for g in g0123:
    u = lambda i, s, g=g: g[s][i]
    dom, mv = indirect_domination(Z, strats, JOINT, u)
    L = lcs(Z, mv, dom, u)
    U = U_set(Z, strats, u); IR = IR_set(Z, strats, u)
    cnt['n'] += 1; cnt['lcs_empty'] += (not L); cnt['lcs_notIR'] += (not L <= IR); cnt['U_not_lcs'] += (not U <= L)
print(f"   {cnt}")
print("done (e,f)")
