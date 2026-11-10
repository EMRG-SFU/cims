"""Capital the unit chain misses, added back as a share (general rule 3, Brad
2026-11-05).

Where a chain of unit archetypes costs well below the reference whole plant it
replaces (output/chain-validation.csv), the gap is capital that sits in no unit
archetype: offsites and utilities, site integration, steps with no archetype.
records/reference/capital-share.csv adds it back on the units that belong only to
that route (not on shared heat supply, hydrogen or haulage), as a multiplier on
each source's capital cost: capex x multiplier. The multiplier is set once from
the chain validation, so the chain then matches its reference plant; the
source's own value stays in the row's note. Variants built on a host take the
host's raised cost through the fill.

A multiplier below 1 does the reverse: it takes off the share of a source's
capital cost for parts the archetype now buys from another node (the air
separation and CO2 purification units of an oxy-fuel kiln).
"""
from __future__ import annotations

from collections import defaultdict

from .common import REFERENCE, fnum, read

COLS = ("value-norm", "value-min-norm", "value-max-norm")


def table():
    return {r["archetype"]: r for r in read(REFERENCE / "capital-share.csv")}


def check(records, fail) -> None:
    for a, r in table().items():
        if a not in records.by_key:
            fail["capital-share names an unknown archetype"].append(a)
        m = fnum(r.get("multiplier"))
        if m is None or not 0.3 <= m <= 4.0:
            fail["capital-share multiplier must be between 0.3 and 4"].append(f"{a}: {r.get('multiplier')}")


def apply(facts) -> dict:
    T = table()
    stats = defaultdict(int)
    for r in facts:
        t = T.get(r.get("archetype") or "")
        if not t or r["parameter"] != "capex" or r.get("excluded"):
            continue
        m = fnum(t["multiplier"])
        src = r.get("value-norm")
        for c in COLS:
            x = fnum(r.get(c))
            if x is not None:
                r[c] = f"{x * m:.6g}"
        r["note"] = (r.get("note") or "") + (
            f"  CAPITAL SHARE: x {m:g} ({t.get('basis', '')}; "
            f"records/reference/capital-share.csv{', chain ' + t['chain-id'] if t.get('chain-id') else ''}); the source's value "
            f"on this basis was {src}.")
        r["conversion-class"] = "ratio"
        r["conversion-factors"] = "; ".join(x for x in [r.get("conversion-factors", ""),
                                                        f"capital-share:{t.get('chain-id') or t['archetype']}"] if x)
        stats["capex rows scaled"] += 1
    return dict(stats)
