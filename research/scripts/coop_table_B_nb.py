"""coop_table_B_nb.py: a game in which copying pays and the Nash product predicts only asymmetric profiles
(Table 1, row "Nash product").

Logic. Among all symmetric 3x3 games in which player 1's payoffs rank the nine profiles and copying pays (as in
coop_table_B.py), count those in which every profile selected by the Nash product of the gains over the security
levels is asymmetric, and print the first one with its payoffs, security levels, products and the maximin of the gains.
"""
from pathlib import Path

import coop_general_properties as g
import coop_table_B as t
import coop_td_features as f

lines = []
n = tot = 0
first = None
for G in f.sym_rank(3):
    if not t.cp(*G):
        continue
    tot += 1
    U, sh = G
    if all(s[0] != s[1] for s in g.nb(U, sh)):
        n += 1
        first = first or G
lines.append(f"symmetric 3x3 ranking games in which copying pays: {tot}")
lines.append(f"  of which the Nash product predicts only asymmetric profiles: {n}")
U, sh = first
v = g.security(U, sh)
lines.append(f"first such game, u_1 rows: {[[U[(x, y)][0] for y in range(3)] for x in range(3)]}")
lines.append(f"  payoffs: {sorted(U.items())}")
lines.append(f"  security levels {v}; best symmetric profile {t.zz(U, sh)}")
for s in sorted(g.ir_set(U, sh)):
    lines.append(f"  IR profile {s} payoffs {U[s]} product {(U[s][0] - v[0]) * (U[s][1] - v[1])} min gain {min(U[s][0] - v[0], U[s][1] - v[1])}")
lines.append(f"  Nash product: {sorted(g.nb(U, sh))}; maximin of gains: {sorted(g.mg(U, sh))}; copying pays: {t.cp(U, sh)}")
Path(__file__).with_suffix(".out").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
