"""Whole-plant costs for UKERC efficiency measures.

UKERC gives an efficiency measure's cost as the cost of ADDING it to a plant
(its "Technology costs" equal its "Cost increase"). An archetype carries
absolute costs, so a measure's cost pools only once the host plant's own cost
is added to it. UKERC states that host cost ("Baseline costs") on the
whole-plant technologies of some process steps, never on the measures.

records/reference/ukerc-host-baseline.csv says which archetypes are a host
with measures added, and which UKERC step is that host. For their examples:

    whole-plant cost = UKERC baseline cost of the step + the measure cost

written as a new derived row (id tp-<source-id>-wNNN) that points at the
measure row; the measure row itself is then not pooled. The baseline must be
one value for the step; a step whose sibling rows state different baselines is
ambiguous and is not used.

Every other UKERC measure cost has no host cost in the source, and is set aside
(set-aside = needs-host-cost) until one is established.
"""
from __future__ import annotations

import collections
import re

from .common import RAW, REFERENCE, fnum, read

ROWS = RAW / "other-reports" / "ukerc_2013_usable-energy-database_rows.csv"
MEASURE_COSTS = {"capex": "Capital cost", "opex-fixed": "O&M Fixed", "opex-variable": "O&M cost"}
MEMBERS = re.compile(r"Members: (.*?)\. Energy")
UKERC_ID = re.compile(r"([A-Z]{1,3}-[A-Z0-9]*?\d+)\s*\(")
NEEDS_HOST = ("set aside: needs a host-plant cost. UKERC gives this measure's cost as the cost "
              "of adding it to a plant and states no baseline cost for its host")


def _ukerc():
    by, meta = collections.defaultdict(dict), {}
    for r in read(ROWS):
        k = r["ukerc-id"]
        by[k][(r["column-group"], r["column-name"])] = r["value"]
        meta[k] = (r["table"], r["baseline-applicable"])
    base = collections.defaultdict(lambda: collections.defaultdict(set))
    for k, d in by.items():
        for col in MEASURE_COSTS.values():
            v = fnum(d.get(("Baseline costs", col)))
            if v is not None:
                base[meta[k]][col].add((round(v, 9), k))
    return meta, base


def derive(loader) -> dict:
    hosts = {r["archetype"]: (r["ukerc-table"], r["ukerc-step"])
             for r in read(REFERENCE / "ukerc-host-baseline.csv")}
    meta, base = _ukerc()
    new, seq, stats = [], collections.Counter(), collections.Counter()
    for r in list(loader.facts):
        m = MEMBERS.search(r.get("note") or "")
        if (not m or not r.get("archetype") or r["parameter"] not in MEASURE_COSTS or r.get("value-type") == "derived"
                or "ukerc" not in (r.get("via-source-id") or r.get("source-id") or "")):
            continue
        host = hosts.get(r.get("archetype", ""))
        ids = [i for i in UKERC_ID.findall(m.group(1)) if i in meta]
        steps = {meta[i] for i in ids}
        vals = base.get(host, {}).get(MEASURE_COSTS[r["parameter"]], set()) if host else set()
        if not host or steps != {host} or len({v for v, _ in vals}) != 1:
            r["set-aside"] = "needs-host-cost"
            stats["set aside"] += 1
            continue
        b, = {v for v, _ in vals}
        cites = ", ".join(sorted({k for _, k in vals}))
        sid = r.get("source-id") or "unattributed"
        seq[sid] += 1
        add = lambda x: "" if fnum(x) is None else f"{fnum(x) + b:.6g}"
        row = dict(r, **{
            "record-id": f"tp-{sid}-w{seq[sid]:03d}", "value": add(r.get("value")),
            "value-min": add(r.get("value-min")), "value-max": add(r.get("value-max")),
            "value-type": "derived",
            "formula": f"{b:.6g} + {r.get('value') or ''}".strip(),
            "derived-from": r["record-id"], "set-aside": "", "derived-kind": "whole-plant",
            "note": (f"WHOLE PLANT: UKERC's baseline {MEASURE_COSTS[r['parameter']].lower()} for "
                     f"{host[1]} ({host[0]}, 'Baseline costs' on {cites}) plus the measure cost "
                     f"in {r['record-id']}. ") + (r.get("note") or "")})
        r["whole-plant"] = row["record-id"]
        new.append(row)
        stats["whole-plant rows"] += 1
    loader.facts.extend(new)
    return dict(stats)
