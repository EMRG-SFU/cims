"""Layer 2: which archetype each example belongs to.

records/source-example.csv is the one place that link is written. A row per
example, reviewable on its own; changing a decision means editing one cell.

  archetype = <key>    the example is an instance of that archetype
  archetype = (none)   deliberately not an example of anything in scope
  archetype = blank    not yet decided; the example is left out of every
                       summary and listed in the mapping review

An example the extractions produce that the file does not list yet is appended
with a blank archetype, so adding a source is: extract it, build, fill in the
new rows, build again. The build never changes a row that is already there.
"""
from __future__ import annotations

from .common import RECORDS, read, write

COLS = ["example-id", "source-id", "source-technology", "process", "archetype",
        "detail", "seeded-by", "note"]
NONE = "(none)"


def apply(loader, records) -> dict:
    path = RECORDS / "source-example.csv"
    rows = read(path)
    known = {r["example-id"] for r in rows}
    new = []
    for e in loader.examples.values():
        if e["example-id"] not in known:
            new.append({"example-id": e["example-id"], "source-id": e["source-id"],
                        "source-technology": e["source-technology"], "process": e["process"],
                        "archetype": "", "detail": "", "seeded-by": "new", "note": ""})
    if new:
        rows.extend(new)
        write(path, COLS, rows)
    mapping = {r["example-id"]: r for r in rows}
    for e in loader.examples.values():
        m = mapping[e["example-id"]]
        e["archetype"] = m["archetype"] if m["archetype"] != NONE else ""
        e["mapping"] = ("none" if m["archetype"] == NONE else
                        "unmapped" if not m["archetype"] else "mapped")
        e["detail"] = m.get("detail", "")
    for table in (loader.facts, loader.flows, loader.cons):
        for r in table:
            r["archetype"] = loader.examples[r["example-id"]]["archetype"]
    stale = [r["example-id"] for r in rows if r["example-id"] not in loader.examples]
    unknown = [f"{r['example-id']}: {r['archetype']}" for r in rows
               if r["archetype"] not in ("", NONE) and r["archetype"] not in records.by_key]
    return {"new": [r["example-id"] for r in new], "stale": stale, "unknown": unknown}
