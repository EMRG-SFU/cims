"""Carbon-balance consistency check on every archetype's CO2 (S184, Brad 2026-11-20),
and the fixes it led to.

For each archetype with a fossil CO2 total (emissions-total CO2, else CO2-eq, else
combustion plus process), the fossil carbon going in (each carrier's energy use and
feedstock x its CO2 factor, conversion-factor.csv `cf-ef-co2-<carrier>`; biogenic
carriers count 0) less the carbon leaving in the product (records/reference/
product-carbon.csv; counted only up to the fossil feedstock it comes from, so a crude,
coking-coal or biomass feed outside the feedstock rows passes through outside the
balance) and the CO2 captured gives the most CO2 the archetype can release from its
fuel and feed. The check compares the total with it:

  * above balance  the total exceeds 1.3 x (the balance plus the process CO2) (carbon
                   from carbonates, electrodes or reductants), plus 2% of the carbon
                   in: more CO2 than the carbon available;
  * below balance  the total is under 0.7 x the balance: carbon leaving in a
                   by-product gas or co-product the table does not count, an
                   uncounted capture, or a total too low;
  * consistent     otherwise;
  * not checked    the product's carbon is not tabulated, a carrier has no CO2 factor,
                   the energy use names no carrier or no fuel, or the total is negative.

`apply_fixes` applies the rulings in records/reference/carbon-balance-fix.csv before
the tables are written (Brad 2026-11-20: obvious errors the check shows are fixed or
replaced with the calculated value); `check` then writes
data/processed_data/technology_database/carbon-balance-check.csv on the result.
"""
from __future__ import annotations

from collections import defaultdict

from .common import OUTPUT, REFERENCE, fnum, read, write
from . import convert, gwp

BIOGENIC = {"ethanol", "biomass", "biomass-solid", "black-liquor", "liquid-biomass", "biogas", "landfill-gas",
            "biomethane", "renewable-natural-gas", "bioliquid", "charcoal"}
NO_CARBON = {"electricity", "steam", "heat", "hot-water", "waste-heat", "solar", "geothermal", "nuclear",
             "hydrogen", "ammonia"}
SKIP_TARGETS = {"", "fuel", "total", "not-stated", "component"}
DERIVED = ("as-sourced", "host-plant", "ccs-add-on", "calculated")
COLS = ["archetype", "family", "product", "product-unit", "result", "co2-total", "co2-total-grade",
        "co2-process", "co2-captured", "fossil-carbon-in", "product-carbon", "balance", "ratio",
        "carbon-in-by-carrier", "fixed", "note"]


def _tables():
    F = convert.Factors().rows
    ef = {fid[len("cf-ef-co2-"):]: f.get("_f") for fid, f in F.items()
          if fid.startswith("cf-ef-co2-") and not fid.endswith("-volume")}
    pc = {(r["product"], r["product-unit"]): fnum(r["co2-per-unit"]) for r in read(REFERENCE / "product-carbon.csv")}
    return ef, pc


def _index(psum):
    by = defaultdict(dict)
    for r in psum:
        by[r["archetype"]][(r["parameter"], r.get("target", "") or "")] = r
    return by


def balance(a, h, ef, pc):
    """(fossil carbon in, balance, process CO2, captured, product carbon, parts, reasons)."""
    why, parts, cin, cfeed, named, fueled = [], [], 0.0, 0.0, False, False
    for (p, t), r in h.items():
        if p not in ("specific-energy-consumption", "feedstock-consumption") or t in SKIP_TARGETS:
            continue
        if t.startswith(DERIVED):
            continue
        v = fnum(r.get("central"))
        if not v:
            continue
        named = True
        if t not in NO_CARBON:
            fueled = True
        if t in BIOGENIC or t in NO_CARBON:
            continue
        f = ef.get(t)
        if f is None:
            why.append(f"no CO2 factor for {t}")
            continue
        cin += v * f
        if p == "feedstock-consumption":
            cfeed += v * f
        parts.append(f"{'feed ' if p == 'feedstock-consumption' else ''}{t} {v:.4g} x {f:.4g}")
    fuel = h.get(("specific-energy-consumption", "fuel"))
    if fuel and fnum(fuel.get("central")) and not fueled:
        why.append("energy use names no fuel")
    tot = h.get(("specific-energy-consumption", ""))
    if not named and tot and fnum(tot.get("central")):
        why.append("energy use names no carrier")
    elif not named and not tot:
        why.append("no energy or feed rows (the fuel is not recorded as a carrier)")
    prod = pc.get((a.get("product", ""), a.get("product-unit", "")))
    if prod is None and cfeed > 0:
        why.append(f"product carbon not tabulated ({a.get('product')} per {a.get('product-unit')})")
    proc = h.get(("emissions-process", "co2"))
    cap = h.get(("emissions-captured", "co2"))
    v_proc = (fnum(proc.get("central")) or 0.0) if proc else 0.0
    v_cap = (fnum(cap.get("central")) or 0.0) if cap else 0.0
    # capture takes the fossil and biogenic mix in proportion (Brad 2026-11-09): with a
    # biogenic stream beside the fossil one, the fossil part of the capture is r x the
    # fossil carbon, r solving captured = r x (fossil + net biogenic / (1 - r))
    bio = h.get(("emissions-biomass", "co2"))
    v_bio = (fnum(bio.get("central")) or 0.0) if bio else 0.0
    fossil_cap = v_cap
    if v_cap > 0 and v_bio > 0 and cin > 0:
        lo, hi_ = 0.0, 0.999
        for _ in range(60):
            r = (lo + hi_) / 2
            if r * (cin + v_bio / (1 - r)) > v_cap:
                hi_ = r
            else:
                lo = r
        fossil_cap = r * cin
    bal = cin - min(prod or 0.0, cfeed) - fossil_cap
    return cin, bal, v_proc, v_cap, prod, parts, why


def _evaluate(a, h, ef, pc):
    if a.get("family") == "co2-capture" or a.get("type") == "component":
        return None
    tot = h.get(("emissions-total", "co2")) or h.get(("emissions-total", "co2e"))
    comb, proc = h.get(("emissions-combustion", "co2")), h.get(("emissions-process", "co2"))
    v_tot = fnum(tot.get("central")) if tot else None
    if v_tot is None and (comb or proc):
        v_tot = (fnum(comb.get("central")) if comb else 0) + (fnum(proc.get("central")) if proc else 0)
    if v_tot is None:
        return None
    cin, bal, v_proc, v_cap, prod, parts, why = balance(a, h, ef, pc)
    src = tot or comb or proc or {}
    row = {"archetype": a["archetype"], "family": a.get("family", ""), "product": a.get("product", ""),
           "product-unit": a.get("product-unit", ""), "co2-total": f"{v_tot:.4g}",
           "co2-total-grade": src.get("grade", ""), "co2-process": f"{v_proc:.4g}", "co2-captured": f"{v_cap:.4g}",
           "fossil-carbon-in": f"{cin:.4g}", "product-carbon": "" if prod is None else f"{prod:.4g}",
           "balance": f"{bal:.4g}", "carbon-in-by-carrier": "; ".join(parts),
           "fixed": "yes" if src.get("method") == "carbon-balance-fix" else ""}
    if v_tot < 0:
        why.append("negative total (CO2 taken up), outside the balance")
    if why:
        row["result"], row["note"] = "not checked", "; ".join(why)
        return row
    hi = 1.3 * (max(bal, 0.0) + max(v_proc, 0.0))
    tol = max(0.02 * cin, 1e-6)          # 2% of the carbon in: rounding and small streams
    row["ratio"] = f"{v_tot / bal:.3g}" if bal > 0 else ""
    if v_tot > hi + tol:
        row["result"] = "above balance"
        row["note"] = (f"total {v_tot:.4g} exceeds 1.3 x (balance {bal:.4g} + process {v_proc:.4g}): "
                       "more CO2 than the fossil carbon going in")
    elif bal > 0 and v_tot < 0.7 * bal - tol:
        row["result"] = "below balance"
        row["note"] = (f"total {v_tot:.4g} is under 0.7 x balance {bal:.4g}: carbon may leave in a "
                       "by-product gas or co-product, or capture is not recorded")
    else:
        row["result"] = "consistent"
    return row


def check(psum, records) -> dict:
    ef, pc = _tables()
    by = _index(psum)
    rows, n = [], defaultdict(int)
    for a in records.archetypes:
        row = _evaluate(a, by.get(a["archetype"], {}), ef, pc)
        if row is None:
            continue
        n[row["result"]] += 1
        rows.append(row)
    write(OUTPUT / "carbon-balance-check.csv", COLS, rows)
    return dict(n)


def _set(r, v, how, rule):
    old = r.get("central", "")
    r.update({"central": f"{v:.6g}", "low": "", "high": "", "p25": "", "p75": "", "grade": "identity",
              "method": "carbon-balance-fix", "rule-id": "carbon-balance-fix",
              "central-from": f"identity: {how} (records/reference/carbon-balance-fix.csv; was {old})",
              "inputs": how, "rationale": rule.get("rationale", "")})


def _new_row(template, archetype, parameter, target, unit):
    r = {k: "" for k in template}
    r.update({"archetype": archetype, "parameter": parameter, "target": target, "unit": unit, "n": 0,
              "n-external": 0, "n-corrected": 0})
    return r


def apply_fixes(psum, records) -> int:
    """Apply records/reference/carbon-balance-fix.csv to the archetype summaries."""
    rules = read(REFERENCE / "carbon-balance-fix.csv")
    if not rules:
        return 0
    ef, pc = _tables()
    by = _index(psum)
    arch = {a["archetype"]: a for a in records.archetypes}
    template = list(psum[0].keys())
    n = 0
    for rule in rules:
        k, p, act = rule["archetype"], rule["parameter"], rule["action"]
        if k not in arch:
            raise SystemExit(f"carbon-balance-fix.csv: unknown archetype {k}")
        h = by[k]
        targets = [rule["target"]] if rule["target"] else [t for t in ("co2", "co2e") if (p, t) in h]
        cin, bal, v_proc, v_cap, prod, parts, why = balance(arch[k], h, ef, pc)
        if act == "balance":
            v = max(bal, 0.0) + max(v_proc, 0.0)
            how = (f"carbon balance: fossil carbon in {cin:.4g} t CO2 ({'; '.join(parts) or 'none'}) less product "
                   f"carbon {min(prod or 0.0, cin):.4g} and captured {v_cap:.4g}, plus process {v_proc:.4g}")
        elif act == "host-scaled":
            host = rule["arg"]
            hh = by.get(host, {})
            ht = hh.get((p, "co2")) or hh.get((p, "co2e"))
            hv = fnum(ht.get("central")) if ht else None
            if hv is None:
                hc, hp = hh.get(("emissions-combustion", "co2")), hh.get(("emissions-process", "co2"))
                hv = (fnum(hc.get("central")) if hc else 0) + (fnum(hp.get("central")) if hp else 0)
            hcin = balance(arch[host], hh, ef, pc)[0]
            if not hcin:
                raise SystemExit(f"carbon-balance-fix.csv: host {host} of {k} has no fossil carbon in")
            v = hv * cin / hcin
            how = f"the host {host}'s total {hv:.4g} x the variant's fossil carbon in over the host's ({cin:.4g} / {hcin:.4g})"
            if not targets:
                targets = ["co2"]
        elif act == "to-biogenic":
            src = next((h[(p, t)] for t in ("co2", "co2e") if (p, t) in h), None)
            bv = fnum(src.get("central")) if src else None
            if bv is None:
                raise SystemExit(f"carbon-balance-fix.csv: {k} has no {p} to move")
            br = h.get(("emissions-biomass", "co2"))
            if br is None:
                br = _new_row(template, k, "emissions-biomass", "co2", src.get("unit", ""))
                psum.append(br)
                h[("emissions-biomass", "co2")] = br
            _set(br, bv, f"the {bv:.4g} the source gives as {p}, biogenic: the archetype burns and converts biomass only", rule)
            v = max(bal, 0.0) + max(v_proc, 0.0)
            how = f"fossil carbon balance {v:.4g}: the archetype's carbon is biogenic, moved to emissions-biomass"
        elif act == "own-sum":
            c, q = h.get(("emissions-combustion", "co2")), h.get(("emissions-process", "co2"))
            v = (fnum((c or {}).get("central")) or 0.0) + (fnum((q or {}).get("central")) or 0.0)
            how = f"the archetype's own combustion plus process CO2 ({v:.4g})"
        elif act == "host-process":
            host = rule["arg"]
            c = h.get(("emissions-combustion", "co2"))
            hq = by.get(host, {}).get(("emissions-process", "co2"))
            vc, vq = fnum((c or {}).get("central")) or 0.0, fnum((hq or {}).get("central")) or 0.0
            v = vc + vq
            how = f"the variant's own combustion CO2 {vc:.4g} plus the host {host}'s process CO2 {vq:.4g} (the coal's carbon is not saved by the measures)"
        elif act == "species-only":
            r = h.get((p, rule["target"] or "co2e"))
            if r is None:
                raise SystemExit(f"carbon-balance-fix.csv: {k} has no {p} {rule['target']}")
            G = gwp.ar5(convert.Factors())      # AR5 GWP100 from conversion-factor.csv
            v = sum((fnum(h[(p, s)].get("central")) or 0.0) * G[s] for s in ("ch4", "n2o") if (p, s) in h)
            how = "CO2-eq of CH4 and N2O only (AR5 GWP100): the fuel is biogenic"
            _set(r, v, how, rule)
            n += 1
            continue
        else:
            raise SystemExit(f"carbon-balance-fix.csv: unknown action {act!r} for {k}")
        if not targets:
            raise SystemExit(f"carbon-balance-fix.csv: {k} has no {p} row to fix (give a target)")
        old_c = fnum((h.get((p, "co2")) or {}).get("central"))
        old_e = fnum((h.get((p, "co2e")) or {}).get("central"))
        for t in targets:
            r = h.get((p, t))
            if r is None:
                r = _new_row(template, k, p, t, (h.get(("emissions-combustion", "co2")) or {}).get("unit", ""))
                psum.append(r)
                h[(p, t)] = r
            # a CO2-eq row keeps its non-CO2 part (CH4, N2O) where both rows exist
            nonco2 = max(old_e - old_c, 0.0) if (t == "co2e" and old_c is not None and old_e is not None) else 0.0
            _set(r, v + nonco2, how, rule)
            n += 1
    return n
