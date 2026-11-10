"""Layer 3: one summarised value set per archetype and parameter.

The layer is deliberately lossy: it summarises what the examples say about an
archetype rather than keeping every source's number side by side. The rule:

  * central  the median of the examples' values when three or more examples
             carry one; otherwise the value from the example whose source ranks
             highest in records/reference/source-priority.csv (a value that came
             through a compilation carries the compilation's rank);
  * low/high the smallest and largest value any example gives, including the
             ranges sources report themselves; written only where the values
             actually spread;
  * n        the number of examples behind the numbers, always shown, so thin
             evidence is visible;
  * n-external  how many of those examples reached the archetype's basis only
             through a factor from outside the source (a heating value, a
             process ratio, a pooled efficiency, an assumption: conversion-class
             `property` or weaker), and `external-factors` which ones. The
             example's own row in example-parameter.csv says exactly how.
  * n-corrected  how many of those examples rest on a value the modeller
             corrected because the source prints it wrongly (a slipped
             decimal, a wrong unit or carrier label): records/reference/
             source-correction.csv, pipeline/correction.py. They are also
             counted in n-external (conversion-class `assumption`).

Within one example, an as-reported value is preferred to one derived from it,
and several rows for the same quantity (sizes, cases) count as one example
whose value is their median. Carrier-specific differences show up as range;
where a difference is large enough that a model would choose between the two,
they are two archetypes, not one wide range.
"""
from __future__ import annotations

from collections import defaultdict
from statistics import median

from .common import fnum
from . import convert

_CLS = None


def external_ids(row) -> list[str]:
    """The factors on a pooled row that come from outside its source."""
    global _CLS
    if _CLS is None:
        _CLS = convert.Factors().cls
    if row.get("conversion-class") not in convert.EXTERNAL:
        return []
    out = []
    for x in (row.get("conversion-factors") or "").split(";"):
        x = x.strip()
        if (_CLS.get(x) in convert.EXTERNAL or x.startswith("assume:")
                or (x.startswith("eff:") and not x.startswith("eff:same-example"))
                or (x.startswith("hpr:") and x != "hpr:same-example")
                or x.startswith("base:")):
            out.append(x.split("=")[0])
    return out


def _example_values(rows):
    """(central, [all values]) for one example's rows of one quantity."""
    rep = [r for r in rows if r.get("value-type") != "derived"] or rows
    cents = [fnum(r["value-norm"]) for r in rep if fnum(r["value-norm"]) is not None]
    allv = [x for r in rows for x in (fnum(r["value-norm"]), fnum(r["value-min-norm"]),
                                      fnum(r["value-max-norm"])) if x is not None]
    return (median(cents) if cents else None), allv


_WEIGHTS = None


def _weights():
    """{archetype: ({example-id: group}, {group: weight})} from records/reference/
    example-group.csv and group-weight.csv (S148, 2026-10-27): an archetype whose
    examples cover several site types is summarised on a stated mix of them."""
    global _WEIGHTS
    if _WEIGHTS is None:
        from .common import REFERENCE, read
        groups = defaultdict(dict)
        for r in read(REFERENCE / "example-group.csv"):
            groups[r["archetype"]][r["example-id"]] = r["group"]
        w = defaultdict(dict)
        for r in read(REFERENCE / "group-weight.csv"):
            w[r["archetype"]][r["group"]] = fnum(r["weight"]) or 0.0
        _WEIGHTS = {a: (groups[a], w[a]) for a in w}
    return _WEIGHTS


def _wquantile(pairs, q):
    """Weighted quantile of [(value, weight)], weights summing to 1."""
    pairs = sorted(pairs)
    acc = 0.0
    for v, w in pairs:
        acc += w
        if acc >= q - 1e-12:
            return v
    return pairs[-1][0]


def summarise(rows, records, keys, examples_by_id):
    """Rows (already normalised, excluded ones skipped) -> one summary per
    (archetype, *keys, unit-norm)."""
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("excluded") or not r.get("unit-norm"):
            continue
        k = (r["archetype"],) + tuple(r.get(x, "") for x in keys) + (r["unit-norm"],)
        groups[k][r["example-id"]].append(r)
    out = []
    for k, by_ex in groups.items():
        per = []
        for eid, rs in by_ex.items():
            c, allv = _example_values(rs)
            srank = min(records.rank(r.get("via-source-id") or r.get("source-id", ""))
                        for r in rs)
            sids = sorted({r.get("source-id", "") for r in rs} |
                          {r.get("via-source-id", "") for r in rs if r.get("via-source-id")},
                          key=lambda s: (records.rank(s), s))
            ext = [x for r in rs for x in external_ids(r)]
            per.append({"example": eid, "central": c, "all": allv, "rank": srank,
                        "sources": sids, "external": ext,
                        "corrected": any(r.get("correction") for r in rs)})
        with_c = [p for p in per if p["central"] is not None]
        wt = _weights().get(k[0])
        wrange = None
        if wt:
            # weighted on the archetype's stated mix: each group's weight is shared
            # by its examples; groups weighted 0 are off the node and left out
            ex_group, gw = wt
            live = [p for p in with_c if gw.get(ex_group.get(p["example"], "unspecified"), 0) > 0]
            cnt = defaultdict(int)
            for p in live:
                cnt[ex_group[p["example"]]] += 1
            tot = sum(gw[g] for g in cnt)
            if live and tot > 0:
                pairs = [(p["central"], gw[ex_group[p["example"]]] / cnt[ex_group[p["example"]]] / tot)
                         for p in live]
                with_c = live
                per = [p for p in per if gw.get(ex_group.get(p["example"], "unspecified"), 0) > 0]
                wrange = (_wquantile(pairs, 0.1), _wquantile(pairs, 0.9))
        if wrange is not None:
            central = _wquantile(pairs, 0.5)
            how = ("weighted median of " + str(len(with_c)) + " on the site mix (records/reference/"
                   "group-weight.csv: " + ", ".join(f"{g} {gw[g]:.0%}" for g in sorted(cnt))
                   + "); low and high are the weighted 10th and 90th percentiles")
        elif len(with_c) >= 3:
            central = median(p["central"] for p in with_c)
            how = f"median of {len(with_c)}"
        elif with_c:
            best = min(with_c, key=lambda p: (p["rank"], p["example"]))
            central = best["central"]
            how = f"top-ranked: {records.short(best['sources'][0]) if best['sources'] else ''}"
        else:
            central, how = None, "range only"
        vals = {round(x, 9) for p in per for x in p["all"]}
        # a single bound and nothing else ("greater than 355 days", "up to 400
        # degC"): kept as the low or the high it is (sanity-check 2026-10-11, S114),
        # rather than a row with a count and no number
        one_lo = one_hi = ""
        if central is None and len(vals) == 1:
            rs_all = [r for rs in by_ex.values() for r in rs]
            if any(fnum(r.get("value-min-norm")) is not None for r in rs_all):
                one_lo = f"{next(iter(vals)):.6g}"
                how = "lower bound only"
            else:
                one_hi = f"{next(iter(vals)):.6g}"
                how = "upper bound only"
        sources = []
        for p in sorted(per, key=lambda p: (p["rank"], p["example"])):
            for s in p["sources"]:
                if s and s not in sources:
                    sources.append(s)
        row = dict(zip(("archetype",) + tuple(keys) + ("unit",), k))
        if wrange is not None:
            vals = {round(x, 9) for p in per for x in p["all"]}
        row.update({"central": "" if central is None else f"{central:.6g}",
                    "low": (f"{wrange[0]:.6g}" if wrange is not None
                            else f"{min(vals):.6g}" if len(vals) > 1 else one_lo),
                    "high": (f"{wrange[1]:.6g}" if wrange is not None
                             else f"{max(vals):.6g}" if len(vals) > 1 else one_hi),
                    "n": len(per), "central-from": how,
                    "n-external": sum(1 for p in per if p["external"]),
                    "n-corrected": sum(1 for p in per if p["corrected"]),
                    "external-factors": "; ".join(dict.fromkeys(x for p in per for x in p["external"])),
                    "sources": "; ".join(sources),
                    "examples": "; ".join(sorted(by_ex))})
        # an energy use's band stops at zero: a negative bound is a source's net-export
        # case (an assay range whose low end exports steam), not a use (Brad
        # 2026-10-28, RX017); a negative central (energy recovered) is left as it is
        if row.get("parameter") == "specific-energy-consumption" and central is not None and central >= 0 \
                and fnum(row["low"]) is not None and fnum(row["low"]) < 0:
            row["low"] = "0"
            row["central-from"] += "; low floored at 0 (a negative bound is net export)"
        out.append(row)
    return out
