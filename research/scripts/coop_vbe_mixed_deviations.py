# Virtual bargaining in the
# traveler's dilemma with mixed deviations (Online Appendix S7).
# Modified payoff of eq. (2) of Melkonyan, Zeitoun and Chater (2022, Appendix), applied to a
# mixed strategy sigma of player 1 against player 2's part y of an agreement:
#   w_1(sigma, y) = min(u_1(sigma, y), max over player 2's pure best replies b to sigma of u_1(sigma, b)).
# (M,M) is a feasible agreement (a Nash equilibrium of the modified game) only if no deviation of
# player 1 has w_1(sigma, M) > M - 1 - r, her modified payoff at (M,M) with pure deviations.
# Part A checks, in exact arithmetic, mixed deviations that make the top claims
# infeasible. Part B checks that pure deviations never do so
# when M >= r + 3 (the pure-strategy statement of S7).
from fractions import Fraction as Fr


def u(x, y, r):
    if x == y:
        return Fr(x), Fr(y)
    if x < y:
        return x + r, x - r
    return y - r, y + r


def w1(sig, y_agree, M, r):
    claims = range(2, M + 1)
    keep = sum(p * u(x, y_agree, r)[0] for x, p in sig.items())
    v2 = {y: sum(p * u(x, y, r)[1] for x, p in sig.items()) for y in claims}
    top = max(v2.values())
    brs = [y for y, v in v2.items() if v == top]
    reply = max(sum(p * u(x, y, r)[0] for x, p in sig.items()) for y in brs)
    return min(keep, reply), brs, keep, reply


print("Part A: mixed deviations of player 1 from (M,M)")
witnesses = [
    (6, Fr(2), {3: Fr(1, 3), 5: Fr(1, 3), 6: Fr(1, 3)}),
    (8, Fr(2), {5: Fr(1, 3), 7: Fr(1, 3), 8: Fr(1, 3)}),
    (10, Fr(2), {7: Fr(14, 43), 9: Fr(29, 86), 10: Fr(29, 86)}),
    (12, Fr(3), {8: Fr(2, 7), 10: Fr(5, 28), 12: Fr(15, 28)}),
    (20, Fr(5, 2), {17: Fr(23, 84), 19: Fr(2, 7), 20: Fr(37, 84)}),
]
for M, r, sig in witnesses:
    assert sum(sig.values()) == 1
    w, brs, keep, reply = w1(sig, M, M, r)
    at = M - 1 - r
    print(f"M={M}, r={r}, M>r+3: {M > r + 3}; deviation {dict((x, str(p)) for x, p in sorted(sig.items()))}: "
          f"payoff if the other keeps M = {keep}, best replies {brs}, best of them for player 1 = {reply}, "
          f"modified payoff = {w} vs M-1-r = {at}; (M,M) infeasible: {w > at}")

print("Part B: pure deviations from (M,M), M from r+3 to 20")
bad = 0
cases = 0
for r in [Fr(5, 4), Fr(3, 2), Fr(2), Fr(5, 2), Fr(3), Fr(5)]:
    for M in range(3, 21):
        if M < r + 3:
            continue
        cases += 1
        best = max(w1({x: Fr(1)}, M, M, r)[0] for x in range(2, M + 1))
        if best > M - 1 - r:
            bad += 1
print(f"{cases} games; pure deviations with modified payoff above M-1-r: {bad}")
