"""Supplied heat is a service request (general rule 1, Brad 2026-11-05).

A unit archetype whose sources include their own boiler or fired heater carries
that plant's fuel in its specific energy consumption. CIMS supplies the same heat
from the sector's steam (or fired-heater, direct-heat) node, so the unit would
count it twice and the low-carbon steam options would never compete for it. The
archetypes listed in records/reference/supplied-heat.csv buy that heat instead:

- the listed fuel carriers become the `to` carrier (steam, or heat bought from
  the direct-heat node) at the generator efficiency: request = fuel x efficiency;
- the example's total loses fuel x (1 - efficiency);
- `convert-share` (default 1) is the share of the listed fuel the generator burns;
  the rest stays as direct fuel (a dry mill's gas fires both the boiler and the
  grain dryers);
- an example that gives only a total has `fuel-share` of it treated as the
  generator's fuel (blank: the example is left as it is and counted);
- `capex-share`, when set, is the share of the source's capital cost kept by the
  unit (the rest is the steam or heater plant, now on the heat node);
- `if-stated` = skip leaves an example that already states the `to` carrier as it
  is (its fuel is direct fuel); the default, add, adds the generator's output to it;
- `emissions` = refill excludes the example's combustion and total CO2 rows, so
  the fill writes them again from the fuel the unit still burns.

The source's own rows stay in example-parameter.csv: a fuel row restated is
excluded with the reason, and a derived row `tp-<source>-hNNN` carries the
request (conversion-class ratio, factor supplied-heat:<archetype>). The
condition on the rule: the sector's heat node must offer the generation options
the unit's sources describe (records/reference/supplied-heat.csv `node`).
"""
from __future__ import annotations

import fnmatch
import re
from collections import defaultdict

from .common import REFERENCE, fnum, read

SEC = "specific-energy-consumption"
COLS = ("value-norm", "value-min-norm", "value-max-norm")
EMIS = ("emissions-combustion", "emissions-total")


def table():
    return read(REFERENCE / "supplied-heat.csv")


def row_for(archetype, rows=None):
    for r in rows if rows is not None else table():
        if fnmatch.fnmatchcase(archetype, r["archetype"]):
            return r
    return None


def _tg(r):
    t = r.get("target-norm") if r.get("target-norm") is not None else r.get("target")
    return (t or "").strip().lower()


def apply(facts, records) -> dict:
    T = table()
    stats = defaultdict(int)
    if not T:
        return dict(stats)
    nxt = defaultdict(int)
    for r in facts:
        m = re.match(r"^(tp-.+)-h(\d{3})$", r.get("record-id", ""))
        if m:
            nxt[m.group(1)] = max(nxt[m.group(1)], int(m.group(2)))
    by_ex = defaultdict(list)
    for r in facts:
        a = r.get("archetype") or ""
        if a and not r.get("excluded") and row_for(a, T):
            by_ex[r["example-id"]].append(r)
    new = []
    for eid, rs in by_ex.items():
        t = row_for(rs[0]["archetype"], T)
        eff = (fnum(t.get("efficiency")) or 0) / 100.0
        if not 0 < eff <= 1.2:
            stats["rows skipped, no efficiency"] += 1
            continue
        share = fnum(t.get("capex-share"))
        if share is not None:
            for r in rs:
                if r["parameter"] == "capex":
                    for c in COLS:
                        x = fnum(r.get(c))
                        if x is not None:
                            r[c] = f"{x * share:.6g}"
                    r["note"] = (r.get("note") or "") + (
                        f"  SUPPLIED HEAT: {share:g} of the capital cost kept; the rest is the steam or heater "
                        "plant, now on the sector's heat node (records/reference/supplied-heat.csv).")
                    r["conversion-class"] = "ratio"
                    r["conversion-factors"] = "; ".join(x for x in [r.get("conversion-factors", ""), f"supplied-heat:{t['archetype']}"] if x)
                    stats["capex rows net of the heat plant"] += 1
        if (t.get("emissions") or "").strip() == "refill":
            for r in rs:
                if r["parameter"] in EMIS and _tg(r) in ("co2", "co2-eq", "co2e", ""):
                    r["excluded"] = ("supplied heat: the generator's combustion CO2 is at the heat node; "
                                     "the fill recomputes it from the fuel the unit still burns")
                    stats["emission rows refilled"] += 1
        fuels = [x.strip() for x in t["carriers"].split("|") if x.strip()]
        to = (t.get("to") or "steam").strip()
        sec = [r for r in rs if r["parameter"] == SEC and any(fnum(r.get(c)) is not None for c in COLS)]
        parent = [r for r in sec if _tg(r) == "fuel" and "fuel" in fuels]
        named = [r for r in sec if _tg(r) in fuels and _tg(r) != "fuel"]
        totals = [r for r in sec if _tg(r) == ""]
        dest = [r for r in sec if _tg(r) == to]
        if dest and (t.get("if-stated") or "add").strip() == "skip":
            # the example states the steam it buys: its fuel is direct fuel, not the
            # generator's (a dryer beside a steam-heated cooker), so it is left as it is
            stats["examples that state the request already, kept"] += 1
            continue
        # the fuel the generator burns, per value column
        F = {}
        for c in COLS:
            src = parent if parent else named
            vals = [fnum(r.get(c)) if fnum(r.get(c)) is not None else fnum(r.get("value-norm")) for r in src]
            vals = [v for v in vals if v is not None]
            if vals:
                F[c] = vals[0] if parent else sum(vals)
        part = fnum(t.get("convert-share"))
        part = 1.0 if part is None else part
        if F and part < 1:
            F = {c: v * part for c, v in F.items()}
        mode = "carriers"
        if not F:
            share = fnum(t.get("fuel-share"))
            if not totals or share is None:
                if totals:
                    stats["examples with a total only and no fuel-share, kept"] += 1
                continue
            for c in COLS:
                v = fnum(totals[0].get(c))
                if v is not None:
                    F[c] = v * share
            mode = "share of total"
        why = (f"SUPPLIED HEAT: the source's own {'boiler' if to == 'steam' else 'fired-heater'} fuel "
               f"is bought as {to} from the sector's heat node ({t.get('node') or 'heat node'}) at "
               f"{eff * 100:g}% generator efficiency (records/reference/supplied-heat.csv; general rule 1, "
               "Brad 2026-11-05).")
        for r in parent + named:
            if part < 1 and mode == "carriers":
                # only part of the fuel raises the steam; the rest is direct fuel
                for c in COLS:
                    x = fnum(r.get(c))
                    if x is not None:
                        r[c] = f"{x * (1 - part):.6g}"
                r["note"] = (r.get("note") or "") + f"  SUPPLIED HEAT: {part:g} of this fuel raised steam, now bought as {to}; the rest is direct fuel."
                stats["fuel rows split between direct fuel and a heat request"] += 1
                continue
            r["excluded"] = "supplied heat: restated as a " + to + " request (records/reference/supplied-heat.csv)"
            stats["fuel rows restated as a heat request"] += 1
        for r in totals:
            for c in COLS:
                x, f = fnum(r.get(c)), F.get(c, F.get("value-norm"))
                if x is not None and f is not None:
                    r[c] = f"{max(x - f * (1 - eff), 0.0):.6g}"
            r["note"] = (r.get("note") or "") + "  " + why
            r["conversion-class"] = "ratio"
            r["conversion-factors"] = "; ".join(x for x in [r.get("conversion-factors", ""), f"supplied-heat:{t['archetype']}"] if x)
            stats["totals net of the generator's losses"] += 1
        if dest:
            d = dest[0]
            for c in COLS:
                x, f = fnum(d.get(c)), F.get(c, F.get("value-norm"))
                if x is not None and f is not None:
                    d[c] = f"{x + f * eff:.6g}"
            d["note"] = (d.get("note") or "") + "  " + why + " Added to the steam the source states."
        else:
            r0 = (parent + named + totals)[0]
            srcid = r0.get("source-id", "")
            nxt[f"tp-{srcid}"] += 1
            row = {k: "" for k in r0}
            row.update({
                "record-id": f"tp-{srcid}-h{nxt[f'tp-{srcid}']:03d}", "example-id": eid,
                "archetype": r0["archetype"], "source-technology": r0.get("source-technology", ""),
                "parameter": SEC, "target": to, "target-norm": to,
                "value": f"{F.get('value-norm', F.get('value-min-norm', 0)) * eff:.6g}",
                "unit": r0.get("unit-norm", ""), "unit-norm": r0.get("unit-norm", ""),
                "conversion-class": "ratio", "conversion-factors": f"supplied-heat:{t['archetype']}",
                "value-type": "derived", "derived-from": "; ".join(r["record-id"] for r in (parent + named or totals)),
                "source-id": srcid, "via-source-id": r0.get("via-source-id", ""),
                "source-file": r0.get("source-file", ""), "source-locator": r0.get("source-locator", ""),
                "region": r0.get("region", ""), "year": r0.get("year", ""), "status": r0.get("status", ""),
                "note": why + f" From the {mode}."})
            for c in COLS:
                if c in F:
                    row[c] = f"{F[c] * eff:.6g}"
            new.append(row)
            stats["heat-request rows written"] += 1
    facts.extend(new)
    return dict(stats)
