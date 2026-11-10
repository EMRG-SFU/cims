"""Validation of reference whole plants against the unit chain that replaces them
(general rule 3, Brad 2026-11-05).

Where a whole plant competes with the units inside it, the whole plant becomes a
reference archetype (no CIMS node) and the route reaches the model through the
chain of unit nodes. records/reference/chain-validation.csv lists, for each
reference plant, the chain: component archetypes and the quantity of each
component's product per unit of the reference plant's product, and `supplies`,
the carrier a component makes for another component (hydrogen from an
electrolyser for the shaft furnace). The chain's energy by carrier, capital cost,
fixed operating cost and direct emissions are the quantity-weighted sums of the
components' central values, less the carriers made inside the chain; output/
chain-validation.csv sets them beside the reference plant's own values with the
ratio. A ratio far from 1 points at a component the chain misses or at a basis
difference to look into; it changes no value.
"""
from __future__ import annotations

from collections import defaultdict

from .common import OUTPUT, REFERENCE, fnum, read, write

SEC = "specific-energy-consumption"
FEED = "feedstock-consumption"
METRICS = [("capex", ""), ("opex-fixed", ""), ("emissions-total", "co2e"), ("emissions-total", "co2"),
           ("cost-levelized", "calculated")]
COLS = ["chain-id", "reference", "metric", "target", "unit-basis", "reference-value", "chain-value",
        "ratio", "components", "note"]


def table():
    return read(REFERENCE / "chain-validation.csv")


def check(records, fail) -> None:
    for r in table():
        for k in ("reference", "component"):
            if r.get(k) and r[k] not in records.by_key:
                fail[f"chain-validation names an unknown {k}"].append(f"{r.get('chain-id')}: {r[k]}")
        if fnum(r.get("quantity")) is None:
            fail["chain-validation quantity is not a number"].append(f"{r.get('chain-id')}: {r.get('component')}")


def build(psum, records) -> list:
    T = table()
    if not T:
        write(OUTPUT / "chain-validation.csv", COLS, [])
        return []
    val = defaultdict(dict)
    for r in psum:
        v = fnum(r.get("central"))
        if v is not None:
            val[r["archetype"]][(r["parameter"], r.get("target") or "")] = v
    chains = defaultdict(list)
    for r in T:
        chains[(r["chain-id"], r["reference"])].append(r)
    out = []
    for (cid, ref), comps in chains.items():
        made = {c["supplies"].strip() for c in comps if (c.get("supplies") or "").strip()}
        names = "; ".join(f"{fnum(c['quantity']):g} x {c['component']}" for c in comps)
        note = "; ".join(c["note"] for c in comps if c.get("note"))
        # energy: the total, electricity, steam and bought heat, and the rest as
        # fuel (total less those), less what the chain makes for itself; named fuels
        # are not compared one by one, because a multi-fuel archetype lists
        # alternative fuels whose values do not add up (2026-11-05)
        def energy(a, q=1.0):
            d = val[a]
            tot = d.get((SEC, ""))
            if tot is None:
                return None
            e = {"": tot}
            for k in ("electricity", "steam", "heat"):
                if (SEC, k) in d:
                    e[k] = d[(SEC, k)]
            e["fuel"] = max(tot - sum(e.get(k, 0.0) for k in ("electricity", "steam", "heat")), 0.0)
            for m in made:
                if (SEC, m) in d:
                    e[""] -= d[(SEC, m)]
                    e["fuel"] = max(e["fuel"] - d[(SEC, m)], 0.0)
            fs = sum(v for (p, tg), v in d.items() if p == FEED and tg not in made and not tg.startswith("as-sourced"))
            if fs:
                e["feedstock"] = fs
            return {k: v * q for k, v in e.items()}
        ref_e = energy(ref) or {}
        chain_e, missing_e = {}, []
        for c in comps:
            e = energy(c["component"], fnum(c["quantity"]))
            if e is None:
                missing_e.append(c["component"])
                continue
            for k, v in e.items():
                chain_e[k] = chain_e.get(k, 0.0) + v
        # what a component supplies to another is not bought from outside
        for m in made:
            chain_e.pop(m, None)
        for k in ["", "electricity", "fuel", "steam", "heat", "feedstock"]:
            if k in ref_e or k in chain_e:
                out.append(_row(cid, ref, SEC if k != "feedstock" else FEED, k if k != "feedstock" else "all",
                                ref_e.get(k), chain_e.get(k), names,
                                note + (f"; no energy on {', '.join(missing_e)}" if missing_e else "")))
        for p, tgt in METRICS:
            rv = val[ref].get((p, tgt))
            parts = [(c, val[c["component"]].get((p, tgt))) for c in comps]
            cv = sum(fnum(c["quantity"]) * x for c, x in parts if x is not None) if any(
                x is not None for _, x in parts) else None
            missing = [c["component"] for c, x in parts if x is None]
            out.append(_row(cid, ref, p, tgt, rv, cv, names,
                            note + (f"; no value on {', '.join(missing)}" if missing and cv is not None else "")))
    write(OUTPUT / "chain-validation.csv", COLS, out)
    return out


def _row(cid, ref, p, tgt, rv, cv, names, note):
    ratio = f"{cv / rv:.3g}" if rv not in (None, 0) and cv is not None else ""
    return {"chain-id": cid, "reference": ref, "metric": p, "target": tgt,
            "unit-basis": "per unit of the reference plant's product",
            "reference-value": "" if rv is None else f"{rv:.6g}",
            "chain-value": "" if cv is None else f"{cv:.6g}", "ratio": ratio, "components": names,
            "note": note}
