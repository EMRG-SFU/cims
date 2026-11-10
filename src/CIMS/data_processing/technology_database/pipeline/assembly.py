"""Whole-plant costs assembled from component costs the same source gives.

records/reference/whole-plant-assembly.csv lists each assembly: the whole-plant
archetype, the source, the parameter and unit of the sum, and its components.
A component is written `example-id | case | multiplier`:

  * example-id  an example of that source (normally mapped to a `component`
                archetype, such as anaerobic-digester or separator_biogas-upgrading);
  * case        the "[...]" label a row's note starts with, where one example
                holds several cases (size bands, feed mixes); blank if it holds one;
  * multiplier  a factor THE SAME SOURCE gives that puts the component on the
                sum's denominator (its locator goes in `factor-locator`); blank = 1.

    whole-plant cost = sum of (component value x multiplier)

written as a derived row (id tp-<source-id>-aNNN) on the whole-plant archetype,
with derived-from naming every component row. Components from another source,
a component that does not resolve to exactly one row, or a component in another
currency-year stop the build: the rule (src/technology_database/docs/design.md section 2) is that parts
are added only within one source, never across sources.

`whole-example` attaches the row to an existing example of the whole-plant
archetype. Where that example already reports the total (`check-case` names its
row), the assembled row is a consistency check: it is marked a duplicate of the
reported row and not pooled, and the difference is reported. Otherwise a new
example `<source-id>:<archetype>@assembly-<assembly-id>` is created.

A component may carry a fourth field of row filters, `key=value & key=value`
(for example `unit=2018_USD-million/yr & target=retrofit`), where an example
holds several rows of the parameter.

The `kind` column (added 2026-10-02) widens the same machinery:

  * assembly    (default) the sum of a source's component costs, as above;
  * difference  a capture plant's cost by difference: the same source's plant
                WITH capture (multiplier +k) less the same plant WITHOUT it
                (multiplier -k), where k puts the difference per t CO2 captured
                with the source's own captured CO2 (modeller, 2026-10-02);
  * rebase      one cost the source gives per unit of the host's product, put per
                t CO2 captured with the source's own captured CO2 per unit.

An `assumption` column names an assumption the row rests on (for example
`avoided-co2-for-captured`, where the source prints only the emissions with and
without capture); normalise.py then grades the row's conversion class
`assumption`. Both kinds are written as tp-<source-id>-dNNN rows on a capture archetype. The difference
holds the capture plant plus compression, utilities and any host changes the
source counts; pairs are listed by hand, case against case, never by archetype.
"""
from __future__ import annotations

import collections

from .common import REFERENCE, currency_year, fnum, read, split_list

FILE = REFERENCE / "whole-plant-assembly.csv"


def _filters(spec):
    out = {}
    for part in (spec or "").split("&"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _pick(facts, eid, param, case, filt=None):
    rows = [r for r in facts if r["example-id"] == eid and r["parameter"] == param
            and r.get("value-type") != "derived"
            and (not case or (r.get("note") or "").startswith(f"[{case}]"))
            and all((r.get(k) or "") == v for k, v in (filt or {}).items())]
    return rows


def derive(loader) -> dict:
    if not FILE.exists():
        return {}
    errors, stats, seq = [], collections.Counter(), collections.Counter()
    new = []
    for a in read(FILE):
        aid, arch, sid, param, unit = (a["assembly-id"], a["archetype"], a["source-id"],
                                        a["parameter"], a["unit"])
        parts = []
        for spec in split_list(a["components"]):
            eid, case, mult, filt = ([x.strip() for x in spec.split("|")] + ["", "", ""])[:4]
            rows = _pick(loader.facts, eid, param, case, _filters(filt))
            if len(rows) != 1:
                errors.append(f"{aid}: component {eid} [{case}] {param} matches {len(rows)} rows")
                continue
            r = rows[0]
            if r.get("source-id") != sid:
                errors.append(f"{aid}: component {r['record-id']} is from {r.get('source-id')}, not {sid}")
            m = fnum(mult) if mult else 1.0
            if currency_year(r.get("unit", "")) != currency_year(unit):
                errors.append(f"{aid}: {r['record-id']} is in {r.get('unit')}, the sum in {unit}")
            if m == 1.0 and r.get("unit") != unit:
                errors.append(f"{aid}: {r['record-id']} unit {r.get('unit')} differs from {unit} "
                              "and no same-source multiplier is given")
            parts.append((r, m))
        if errors and any(e.startswith(aid + ":") for e in errors):
            continue

        def total(field):
            vals = []
            for r, m in parts:
                v = fnum(r.get(field)) if fnum(r.get(field)) is not None else fnum(r.get("value"))
                if v is None:
                    return ""
                vals.append(v * m)
            return f"{sum(vals):.6g}"
        allv = all(fnum(r.get("value")) is not None for r, _ in parts)
        value = f"{sum(fnum(r['value']) * m for r, m in parts):.6g}" if allv else ""
        lo, hi = total("value-min"), total("value-max")
        if lo == value and hi == value:
            lo = hi = ""
        eid = a.get("whole-example") or ""
        ex = loader.examples.get(eid) if eid else None
        if eid and ex is None:
            errors.append(f"{aid}: whole-example {eid} is not an example")
            continue
        if ex is None:
            ex = loader.example(sid, arch, f"assembly {aid}", {}, parts[0][0].get("extraction-file", ""))
            ex.update({"archetype": arch, "mapping": "mapped", "detail": "assembled from components"})
            eid = ex["example-id"]
        elif ex.get("archetype") != arch:
            errors.append(f"{aid}: whole-example {eid} is mapped to {ex.get('archetype')!r}, not {arch}")
            continue
        kind = (a.get("kind") or "assembly").strip()
        if kind not in ("assembly", "difference", "rebase"):
            errors.append(f"{aid}: unknown kind {kind!r}")
            continue
        tag = "a" if kind == "assembly" else "d"
        seq[(sid, tag)] += 1
        first = parts[0][0]
        formula = " + ".join(f"{r['record-id']}" + ("" if m == 1.0 else f" x {m:g}") for r, m in parts)
        row = dict(first, **{
            "record-id": f"tp-{sid}-{tag}{seq[(sid, tag)]:03d}", "example-id": eid, "archetype": arch,
            "source-technology": ex["source-technology"], "unit": unit, "currency": currency_year(unit),
            "value": value, "value-min": lo, "value-max": hi,
            "uncertainty-type": "reported-range" if (lo or hi) else "",
            "value-type": "derived", "formula": formula, "derived-kind": kind,
            "derived-from": "; ".join(r["record-id"] for r, _ in parts),
            "cost-basis": a.get("cost-basis") or first.get("cost-basis", ""),
            "source-locator": "; ".join(r.get("source-locator", "") for r, _ in parts),
            "whole-plant": "", "set-aside": "",
            "note": ({"assembly": f"ASSEMBLED ({aid}): the sum of this source's own component costs, ",
                      "difference": f"CAPTURE COST BY DIFFERENCE ({aid}): the same source's plant with "
                                    "capture less the same plant without it, per t CO2 captured, ",
                      "rebase": f"CAPTURE COST REBASED ({aid}): the source's cost per unit of the "
                                "host's product, put per t CO2 captured, "}[kind]
                     + f"{formula}. " + (f"Factor: {a['factor-locator']}. " if a.get("factor-locator") else "")
                     + (a.get("note") or ""))})
        row["assume"] = (a.get("assumption") or "").strip()
        if row["assume"]:
            row["note"] = f"ASSUMPTION ({row['assume']}). " + row["note"]
        if kind != "assembly":
            row["target"] = ""      # a with-capture row's case label is not the capture plant's
        if a.get("check-case"):
            rep = _pick(loader.facts, eid, param, a["check-case"])
            if len(rep) != 1:
                errors.append(f"{aid}: check-case matches {len(rep)} reported rows in {eid}")
                continue
            rep = rep[0]
            row["assembly-duplicate"] = rep["record-id"]
            diffs = []
            for fld in ("value", "value-min", "value-max"):
                x, y = fnum(row.get(fld)), fnum(rep.get(fld))
                if x is not None and y is not None and y:
                    diffs.append(abs(x - y) / abs(y))
            d = max(diffs) if diffs else None
            row["note"] = (f"CHECK: the source also prints this total ({rep['record-id']}); "
                           + ("they agree exactly. " if d == 0 else
                              f"they differ by up to {d:.1%}. " if d is not None else "")
                           + row["note"])
            stats["checks"] += 1
            stats["check mismatches"] += int(bool(d))
        else:
            stats["rows"] += 1
        new.append(row)
    if errors:
        raise SystemExit("whole-plant-assembly.csv:\n  " + "\n  ".join(errors))
    loader.facts.extend(new)
    return dict(stats)
