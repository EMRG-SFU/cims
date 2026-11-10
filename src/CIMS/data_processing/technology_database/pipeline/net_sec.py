"""Specific energy consumption is GROSS input (ruling 2026-10-13, sanity-check
2026-10-11 section 4.1). Energy a process recovers and exports (BOF gas, top
gas, steam, power-recovery electricity) is an OUTPUT flow, recorded in
example-flow.csv and pooled in archetype-flow.csv, never a credit inside the
energy use.

  * A negative specific energy consumption is a net export: it is excluded from
    the energy use and, where it names a carrier the commodity vocabulary knows
    (steam, electricity, heat), written as a derived output flow of that carrier
    (`tf-<source>-dNNN`) so the export is kept.
  * records/reference/net-energy.csv lists the positive values a source states
    net of a credit: `gross-up` adds the printed credit back (in the row's own
    unit, before conversion); `exclude` rules the row out where the credit is not
    printed.
"""
from __future__ import annotations

from collections import defaultdict

from .common import REFERENCE, fnum, read

SEC = "specific-energy-consumption"
NOTE = ("  NET OF RECOVERED ENERGY: specific energy consumption is gross input (ruling "
        "2026-10-13); ")


def apply(facts, flows, records) -> dict:
    table = {r["record-id"]: r for r in read(REFERENCE / "net-energy.csv")}
    by_id = {r["record-id"]: r for r in facts}
    unknown = set(table) - set(by_id)
    if unknown:
        raise SystemExit(f"net-energy.csv names unknown record-ids: {sorted(unknown)}")
    stats = defaultdict(int)
    nxt = defaultdict(int)
    import re
    for r in flows:
        m = re.match(r"^(tf-.+)-d(\d{3})$", r.get("record-id", ""))
        if m:
            nxt[m.group(1)] = max(nxt[m.group(1)], int(m.group(2)))
    for r in facts:
        if r["parameter"] != SEC or r.get("excluded"):
            continue
        t = table.get(r["record-id"])
        if t and t["action"] == "exclude":
            r["excluded"] = "net of recovered energy, gross input not printed: " + t["reason"]
            stats["net rows excluded"] += 1
            continue
        if t and t["action"] == "gross-up":
            v, add = fnum(r.get("value")), fnum(t.get("add-back"))
            vn = fnum(r.get("value-norm"))
            if v and add is not None and vn is not None:
                k = (v + add) / v
                for col in ("value-norm", "value-min-norm", "value-max-norm"):
                    x = fnum(r.get(col))
                    if x is not None:
                        r[col] = f"{x * k:.6g}"
                r["note"] = (r.get("note") or "") + NOTE + f"grossed up by the printed credit ({t['reason']})."
                stats["net rows grossed up"] += 1
            continue
        vn = fnum(r.get("value-norm"))
        if vn is None:
            # a range-only row: negative when its upper end is (2026-10-13)
            hi_ = fnum(r.get("value-max-norm"))
            lo_ = fnum(r.get("value-min-norm"))
            vn = hi_ if hi_ is not None else lo_
        if vn is None or vn >= 0:
            continue
        a = records.by_key.get(r.get("archetype", ""))
        r["excluded"] = ("net export: a negative energy use is energy recovered and exported, "
                         "recorded as an output flow; specific energy consumption is gross input "
                         "(ruling 2026-10-13)")
        stats["negative rows excluded"] += 1
        carrier = (r.get("target") or "").strip().lower()
        if not a or carrier not in records.commodities:
            continue
        src = r.get("source-id", "")
        nxt[f"tf-{src}"] += 1
        pu = a["product-unit"]
        num = (r.get("unit-norm") or "GJ/").split("/", 1)[0]
        flows.append({
            "record-id": f"tf-{src}-d{nxt[f'tf-{src}']:03d}", "example-id": r["example-id"],
            "archetype": r["archetype"], "source-technology": r.get("source-technology", ""),
            "direction": "output", "commodity": carrier, "value": f"{-vn:.6g}",
            "value-min": "", "value-max": "", "unit": f"{num}/{pu}", "per-unit": pu,
            "co-product": "", "value-type": "derived", "formula": "",
            "region": r.get("region", ""), "year": r.get("year", ""), "source-id": src,
            "source-file": r.get("source-file", ""), "source-locator": r.get("source-locator", ""),
            "via-source-id": r.get("via-source-id", ""), "status": r.get("status", ""),
            "note": (f"Derived from {r['record-id']}: the source prints this export as a negative "
                     f"{carrier} use ({r.get('value')} {r.get('unit')}); an output flow under the "
                     "gross-input ruling of 2026-10-13.")})
        stats["exports written as output flows"] += 1
    return dict(stats)


def recovered_flows(flows, records) -> int:
    """An output flow for what a measure or plant recovers (records/reference/recovered-output.csv):
    methane kept in the system and sold, per t CH4 abated. One derived flow row per
    archetype, example `identity:<archetype>` (ruling 2026-10-14: methane abatement
    costs are gross; the recovered gas is recorded instead of netted)."""
    n = 0
    for t in read(REFERENCE / "recovered-output.csv"):
        a = records.by_key.get(t["archetype"])
        if not a:
            raise SystemExit(f"recovered-output.csv names an unknown archetype: {t['archetype']}")
        pu = a["product-unit"]
        n += 1
        flows.append({
            "record-id": f"tf-identity-d{n:03d}", "example-id": f"identity:{t['archetype']}",
            "archetype": t["archetype"], "source-technology": "", "direction": "output",
            "commodity": t["commodity"], "value": t["value"], "value-min": t.get("value-min", ""),
            "value-max": t.get("value-max", ""),
            "unit": t["unit"], "per-unit": pu, "co-product": "", "value-type": "derived",
            "formula": t.get("factor-id", ""), "region": "", "year": "", "source-id": "",
            "source-file": "records/reference/recovered-output.csv", "source-locator": "",
            "via-source-id": "", "status": "draft", "note": t["note"]})
    return n
