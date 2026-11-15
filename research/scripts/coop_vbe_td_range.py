"""Formal virtual bargaining equilibrium (VBE) in pure strategies, as in
research/scripts/coop_vbe_formal.py (Zeitoun, Melkonyan and Chater 2023,
Appendix B; Melkonyan, Zeitoun and Chater 2022, eq. (2)), on the paper's traveler's
dilemma: claims 2..M, reward r to the lower claimant, minus r to the higher.
Question: for which (M, r) is the VBE the top claims (M, M)?
Both tie-break conventions among best responses are run.
"""
import itertools
from fractions import Fraction as Fr


def td(M, r):
    S = [list(range(2, M + 1))] * 2
    def u(s):
        a, b = s
        if a == b:
            return (a, a)
        lo = min(a, b)
        return (lo + r, lo - r) if a < b else (lo - r, lo + r)
    return S, u


def vbe(S, u, tie):
    def w(s):
        out = []
        for i in (0, 1):
            j = 1 - i
            prof = (lambda sj: (s[0], sj)) if i == 0 else (lambda sj: (sj, s[1]))
            best = max(u(prof(sj))[j] for sj in S[j])
            resp = [prof(sj) for sj in S[j] if u(prof(sj))[j] == best]
            out.append(min(u(s)[i], tie(u(t)[i] for t in resp)))
        return tuple(out)
    prof = list(itertools.product(*S))
    W = {s: w(s) for s in prof}
    F = [s for s in prof
         if all(W[s][0] >= W[(a, s[1])][0] for a in S[0])
         and all(W[s][1] >= W[(s[0], b)][1] for b in S[1])]
    if not F:
        return F, []
    m = [min(W[s][i] for s in F) for i in (0, 1)]
    val = {s: (W[s][0] - m[0]) * (W[s][1] - m[1]) for s in F}
    top = max(val.values())
    return F, [s for s in F if val[s] == top]


for tie in (max, min):
    print(f"tie-break among best responses: {tie.__name__}")
    for r in (Fr(3, 2), 2, 3, 5):
        row = []
        for M in range(3, 16):
            F, V = vbe(*td(M, r), tie)
            row.append(f"M={M}:{'top' if V == [(M, M)] else V}")
        print(f"  r={r}: " + "; ".join(row))
