"""Service requests that are not energy carriers (general rules 2 and 3, Brad
2026-11-05).

In CIMS a technology asks its service nodes for what it uses. The database's
energy carriers (specific-energy-consumption by carrier) and feedstocks
(feedstock-consumption by carrier) become those requests one per carrier
(design.md s10). records/reference/service-request.csv holds the rest:

- a bought intermediate made at another CIMS node (reduced iron for an electric
  smelting furnace, anodes for a prebaked cell, oxygen for a smelter), per unit
  of the archetype's product (rule 2);
- a step a route skips or supplies itself, as a zero or negative request on the
  step's node (a thin-slab caster asks for no reheating; a dry-forming paper
  machine gives back the drying its paper node asks for) (rule 3).

Each row: archetype, request (what is asked for), node (hierarchy path, with
<sector>), quantity, low, high, unit (per unit of the archetype's product),
grade (sourced, estimated, identity, judgement), source-id, basis, finding,
decided. `at-node` (optional) limits a row to one of the nodes the archetype
serves, where the quantity differs by node (a smelter's acid per t of copper,
lead or nickel; 2026-11-05). The build checks the archetype and the node, and writes
output/archetype-service-request.csv with the archetype's product unit.
"""
from __future__ import annotations

import fnmatch

from .common import OUTPUT, REFERENCE, fnum, read, write

GRADES = ("sourced", "estimated", "identity", "judgement")
COLS = ["archetype", "request", "node", "at-node", "quantity", "low", "high", "unit", "product-unit",
        "grade", "source-id", "basis", "finding", "decided"]


def table():
    return read(REFERENCE / "service-request.csv")


def check(records, hier_paths, fail) -> None:
    for r in table():
        a = r.get("archetype", "")
        if a not in records.by_key:
            fail["service-request names an unknown archetype"].append(a)
        n = (r.get("node") or "").strip()
        if n and n not in hier_paths and not any(fnmatch.fnmatchcase(h, n) for h in hier_paths):
            fail["service-request names a node not in records/cims-hierarchy.csv"].append(f"{a}: {n}")
        at = (r.get("at-node") or "").strip()
        if at and at not in hier_paths:
            fail["service-request at-node is not in records/cims-hierarchy.csv"].append(f"{a}: {at}")
        if fnum(r.get("quantity")) is None:
            fail["service-request quantity is not a number"].append(f"{a}: {r.get('request')}")
        if (r.get("grade") or "") not in GRADES:
            fail["service-request grade must be sourced, estimated, identity or judgement"].append(
                f"{a}: {r.get('request')}")


def build(records) -> list:
    rows = []
    for r in table():
        a = records.by_key.get(r.get("archetype", ""), {})
        rows.append(dict(r, **{"product-unit": a.get("product-unit", "")}))
    order = {x["archetype"]: i for i, x in enumerate(records.archetypes)}
    rows.sort(key=lambda r: (order.get(r["archetype"], 9999), r.get("node", ""), r.get("request", "")))
    write(OUTPUT / "archetype-service-request.csv", COLS, rows)
    return rows
