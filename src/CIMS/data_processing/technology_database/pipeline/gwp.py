"""CO2-equivalents on one basis: the 100-year GWPs of IPCC AR5 that Canada's
National Inventory Report uses (ruling 2026-10-14; conversion-factor.csv
cf-gwp100-ar5-<species>, source ipcc_2013_001).

  * species(): a figure a source gives as CO2-eq that is clearly one gas
    (records/reference/co2e-species.csv: SF6 cover gas, N2O abatement, methane
    abatement) gets a derived row in tonnes of that gas (`tp-<source>-dNNN`), from
    the GWP the source itself used, and the CO2-eq row is restated on AR5.
    Emissions: t gas = t CO2-eq / GWP(source). Abatement cost: cost per t gas =
    cost per t CO2-eq x GWP(source).
  * The fill writes CO2-eq where only the gases are known (id-co2e-from-species).
"""
from __future__ import annotations

import re
from collections import defaultdict

from .common import REFERENCE, fnum, read

SPECIES = ("ch4", "n2o", "sf6", "cf4", "c2f6")


def ar5(factors) -> dict:
    out = {}
    for sp in SPECIES:
        f = factors.rows.get(f"cf-gwp100-ar5-{sp}")
        if f:
            out[sp] = fnum(f["factor"])
    return out


def species(facts, factors) -> dict:
    G = ar5(factors)
    table = {r["record-id"]: r for r in read(REFERENCE / "co2e-species.csv")}
    by_id = {r["record-id"]: r for r in facts}
    unknown = set(table) - set(by_id)
    if unknown:
        raise SystemExit(f"co2e-species.csv names unknown record-ids: {sorted(unknown)}")
    nxt = defaultdict(int)
    for r in facts:
        m = re.match(r"^(tp-.+)-d(\d{3})$", r.get("record-id", ""))
        if m:
            nxt[m.group(1)] = max(nxt[m.group(1)], int(m.group(2)))
    stats = defaultdict(int)
    new = []
    for rid, t in table.items():
        r = by_id[rid]
        if r.get("excluded") or not r.get("unit-norm"):
            stats["listed rows not pooled"] += 1
            continue
        sp, gs = t["species"].lower(), fnum(t["gwp-source"])
        g5 = G.get(sp)
        if not gs or not g5:
            continue
        cost = r["parameter"] == "abatement-cost"
        to_sp = (lambda x: x * gs) if cost else (lambda x: x / gs)
        restate = (lambda x: x * gs / g5) if cost else (lambda x: x * g5 / gs)
        src = r.get("source-id", "")
        nxt[f"tp-{src}"] += 1
        row = dict(r)
        unit = r["unit-norm"]
        if cost:
            unit = re.sub(r"/t-CO2-eq$", f"/t-{sp.upper()}", unit)
        row.update({"record-id": f"tp-{src}-d{nxt[f'tp-{src}']:03d}", "target": sp, "target-norm": sp,
                    "unit-norm": unit, "value-type": "derived", "derived-from": rid,
                    "note": (f"Tonnes of {sp.upper()} from the CO2-eq of {rid}: the source's GWP "
                             f"{gs:g} ({t['note']}). (ruling 2026-10-14)")})
        for col in ("value-norm", "value-min-norm", "value-max-norm"):
            x = fnum(r.get(col))
            row[col] = "" if x is None else f"{to_sp(x):.6g}"
            if x is not None:
                r[col] = f"{restate(x):.6g}"
        r["note"] = (r.get("note") or "") + (f"  CO2-EQ RESTATED on AR5 (GWP {g5:g} for {sp.upper()}, "
                                             f"the source used {gs:g}; ruling 2026-10-14).")
        new.append(row)
        stats["one-gas CO2-eq rows given in their gas"] += 1
    facts.extend(new)
    return dict(stats)
