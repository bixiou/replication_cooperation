# The pure alpha-core and its
# credible version in the traveler's dilemma on both sides of M = r + 3 (Section 7, Proposition 8).
# Two players, claims 2 to M, reward r. A profile is ruled out if a coalition has a joint deviation
# that gives each member strictly more when the others answer with their worst strategy for the
# deviator (alpha-core) or with the worst of their best replies (credible version). The coalition
# of both players has no outsiders.
from fractions import Fraction as Fr


def u(x, y, r):
    if x == y:
        return (Fr(x), Fr(y))
    if x < y:
        return (x + r, x - r)
    return (y - r, y + r)


def cores(M, r):
    C = range(2, M + 1)
    alpha, cred = [], []
    for s in [(x, y) for x in C for y in C]:
        us = u(*s, r)
        grand = any(all(u(a, b, r)[i] > us[i] for i in (0, 1)) for a in C for b in C)
        out_a = out_c = grand
        for i in (0, 1):
            for z in C:
                if z == s[i]:
                    continue
                def pay(y):
                    p = (z, y) if i == 0 else (y, z)
                    return u(*p, r)
                worst = min(pay(y)[i] for y in C)
                j = 1 - i
                top = max(pay(y)[j] for y in C)
                worst_br = min(pay(y)[i] for y in C if pay(y)[j] == top)
                out_a = out_a or worst > us[i]
                out_c = out_c or worst_br > us[i]
        if not out_a:
            alpha.append(s)
        if not out_c:
            cred.append(s)
    return alpha, cred


ok = 0
total = 0
for r in [Fr(3, 2), Fr(2), Fr(5, 2), Fr(3), Fr(5)]:
    for M in range(3, 13):
        a, c = cores(M, r)
        side = "<" if M < r + 3 else ("=" if M == r + 3 else ">")
        expect = [(M, M)] if M < r + 3 else sorted([(M - 1, M), (M, M - 1), (M, M)])
        good = sorted(a) == expect and sorted(c) == expect
        total += 1
        ok += good
        print(f"r={r}, M={M} (M {side} r+3): alpha-core {sorted(a)}; credible {sorted(c)}; as stated: {good}")
print(f"{ok} of {total} games as stated: {{(M,M)}} if M < r+3, otherwise {{(M-1,M),(M,M-1),(M,M)}}")
