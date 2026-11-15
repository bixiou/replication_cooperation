"""Virtual bargaining equilibrium (VBE) in pure strategies, formal version of
Zeitoun, Melkonyan and Chater (2023, AMR, Appendix B, accepted manuscript pp. 45-46),
which cites Melkonyan, Zeitoun and Chater (2018, Management Science; 2022, JMP):

  worst payoff  w_i(s) = min( u_i(s), u_i(s_i, b_j) ), b_j a best response of j to s_i
                (with several best responses, the best of them for i: JMP 2022 eq. (2),
                "non-spiteful"; TIE=min gives the opposite convention, same outputs here);
  feasible      F = pure Nash equilibria of the game with payoffs w;
  VBE           argmax over F of prod_i (w_i(s) - w_i^m), w_i^m = min over F of w_i.

Also the informal 2014 version (Misyak et al., TiCS, Box 1): feasible if no player's most profitable deviation
lowers the other's payoff; Nash bargaining from a Nash equilibrium status quo.

Games: prisoner's dilemma, traveler's dilemmas, contested leadership G_e of the paper,
chicken, Aumann's chicken, battle of the sexes; then the 576 strict 2x2 games:
how often the formal VBE is empty, a single profile, or contains two profiles
swapped by an exchange of the players (the trilemma configuration), and whether
the VBE set is self-consistent in the sense of the paper (Definition 4).
"""
import itertools

TIE = max  # tie-break among the answerer's best responses, in the deviator's favour (JMP 2022)


def br(u, i, si, S):
    """Best responses of j = 1 - i to i's strategy si; u(s) returns a payoff pair."""
    j = 1 - i
    def prof(sj):
        return (si, sj) if i == 0 else (sj, si)
    best = max(u(prof(sj))[j] for sj in S[j])
    return [prof(sj) for sj in S[j] if u(prof(sj))[j] == best]


def w(u, S, s):
    out = []
    for i in (0, 1):
        resp = br(u, i, s[i], S)
        out.append(min(u(s)[i], TIE(u(t)[i] for t in resp)))
    return tuple(out)


def vbe(u, S):
    W = {s: w(u, S, s) for s in itertools.product(*S)}
    F = []
    for s in W:
        ok = True
        for i in (0, 1):
            for d in S[i]:
                t = (d, s[1]) if i == 0 else (s[0], d)
                if W[t][i] > W[s][i]:
                    ok = False
        if ok:
            F.append(s)
    if not F:
        return W, F, []
    wm = [min(W[s][i] for s in F) for i in (0, 1)]
    prod = {s: (W[s][0] - wm[0]) * (W[s][1] - wm[1]) for s in F}
    m = max(prod.values())
    return W, F, [s for s in F if prod[s] == m]


def guarantee(u, E, i):
    vals = {}
    for s in E:
        vals.setdefault(s[i], []).append(u(s)[i])
    return max(min(v) for v in vals.values())


def self_consistent(u, E):
    return bool(E) and all(u(s)[i] >= guarantee(u, E, i) for s in E for i in (0, 1))


def matrix(d):
    return (lambda s: d[s]), sorted({s[0] for s in d}), sorted({s[1] for s in d})


def td(lo, hi, r):
    def u(s):
        x, y = s
        if x == y:
            return (x, x)
        return (x + r, x - r) if x < y else (y - r, y + r)
    return u, list(range(lo, hi + 1)), list(range(lo, hi + 1))


def report(name, u, S1, S2):
    W, F, V = vbe(u, (S1, S2))
    print(f"{name}: feasible agreements {F}; VBE {V}; payoffs {[u(s) for s in V]}; "
          f"VBE set self-consistent: {self_consistent(u, V)}")


def main():
    report("PD (T,R,P,S)=(5,3,1,0)", *matrix({("C", "C"): (3, 3), ("C", "D"): (0, 5), ("D", "C"): (5, 0), ("D", "D"): (1, 1)}))
    report("TD efforts 1..10, r=2 (AMR p. 10-11)", *td(1, 10, 2))
    report("TD claims 2..100, r=2", *td(2, 100, 2))
    for r in (1.5, 3, 5, 10, 40):
        report(f"TD claims 2..100, r={r}", *td(2, 100, r))
    report("TD claims 80..200, r=80 (Capra et al.)", *td(80, 200, 80))
    report("G_e, e=0.5", *matrix({("A", "A"): (0, 0), ("A", "Y"): (3, 1), ("Y", "A"): (1, 3), ("Y", "Y"): (0.5, 0.5)}))
    report("Chicken (0,0),(3,1),(1,3),(2,2)", *matrix({("A", "A"): (0, 0), ("A", "Y"): (3, 1), ("Y", "A"): (1, 3), ("Y", "Y"): (2, 2)}))
    report("Aumann chicken (0,0),(7,2),(2,7),(6,6)", *matrix({("A", "A"): (0, 0), ("A", "Y"): (7, 2), ("Y", "A"): (2, 7), ("Y", "Y"): (6, 6)}))
    report("Battle of the sexes (2,1),(0,0),(0,0),(1,2)", *matrix({("a", "a"): (2, 1), ("a", "b"): (0, 0), ("b", "a"): (0, 0), ("b", "b"): (1, 2)}))
    report("Nash-product game of Prop. bargaining", *matrix({(x, y): ({"aa": 1, "ab": 2, "ac": 5, "ba": 7, "bb": 4, "bc": 8, "ca": 9, "cb": 3, "cc": 6}[x + y],
                                                                      {"aa": 1, "ab": 2, "ac": 5, "ba": 7, "bb": 4, "bc": 8, "ca": 9, "cb": 3, "cc": 6}[y + x])
                                                                     for x in "abc" for y in "abc"}))

    # census of the 576 strict 2x2 games (pairs of strict rankings of the four profiles)
    profiles = [(0, 0), (0, 1), (1, 0), (1, 1)]
    S = ([0, 1], [0, 1])
    counts = {"empty": 0, "single": 0, "two+": 0, "swapped pair": 0, "VBE self-consistent": 0, "VBE contains a Pareto-optimal profile": 0}
    for r1 in itertools.permutations(range(4)):
        for r2 in itertools.permutations(range(4)):
            d = {p: (r1[k], r2[k]) for k, p in enumerate(profiles)}
            u = lambda s, d=d: d[s]
            _, F, V = vbe(u, S)
            if not V:
                counts["empty"] += 1
                continue
            counts["single" if len(V) == 1 else "two+"] += 1
            if any((s[1], s[0]) in V and s[0] != s[1] for s in V):
                # an exchange of players with the identity on strategy names maps one VBE profile to another
                counts["swapped pair"] += 1
            if self_consistent(u, V):
                counts["VBE self-consistent"] += 1
            po = [s for s in V if not any(all(d[t][i] >= d[s][i] for i in (0, 1)) and d[t] != d[s] for t in profiles)]
            if po:
                counts["VBE contains a Pareto-optimal profile"] += 1
    print("Census of the 576 strict 2x2 games, formal VBE in pure strategies:", counts)


if __name__ == "__main__":
    # both conventions in turn, as in the recorded output; "spiteful" runs the second only
    import sys
    runs = [(min, "Tie-break among best responses against the deviator")]
    if not (len(sys.argv) > 1 and sys.argv[1] == "spiteful"):
        runs.insert(0, (max, "Tie-break among best responses in the deviator's favour (JMP 2022, eq. (2))"))
    for TIE, title in runs:
        print(title)
        main()
