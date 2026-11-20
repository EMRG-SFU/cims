"""The check of the calculated levelized cost against the costs sources print (S176,
Brad 2026-11-20).

The levelized or production cost a source prints is not a model value: it rests on
the source's own prices, region, discount rate, scope and year. It is kept off every
wide view (pipeline/placement.py) and compared here, one row per source figure, with
the archetype's calculated levelized cost (non-energy, and with energy at 2025
Canadian prices). `basis` names what the figure includes, read from its note; the
ratio flags a figure outside 0.5-2 of the calculated one.
"""
from __future__ import annotations

import re

from .common import OUTPUT, fnum, write

COLS = ["archetype", "unit", "calculated", "calculated-incl-energy", "sourced-central", "sourced-n",
        "example-id", "source-id", "value", "ratio-to-calculated-incl-energy", "flag", "basis",
        "region", "year", "discount-rate", "cost-basis", "source-locator", "note"]

BASIS = [
    ("includes feedstock or is a selling price", r"feedstock|feed\b|selling price|MFSP|MJSP|MSP|minimum .*price|incl(?:\.|udes|uding)? (?:raw|material|feed)"),
    ("includes a return on capital", r"return|IRR|profit|margin"),
    ("includes CO2 transport and storage", r"transport and storage|T&S|transport|storage cost"),
    ("per t CO2 net removed", r"net removed"),
    ("includes a CO2 price", r"CO2 price|carbon price|ETS|CO2 cost"),
    ("by-products not credited", r"by-?products? (?:revenues? )?not|not deducted|no (?:co|by)-?product credit"),
    ("excludes energy", r"excl(?:\.|udes|uding)? (?:energy|fuel|electricity)|non-energy"),
    ("from market prices", r"market price|spot|plant-gate|Argus|price data"),
    ("a future or projected cost", r"\b20[3-5]0\b|projected|future"),
]


def _basis(note: str) -> str:
    tags = [label for label, pat in BASIS if re.search(pat, note or "", re.I)]
    return "; ".join(tags) if tags else "as printed (no inclusions stated)"


def write_check(psum, facts) -> int:
    calc = {}
    sourced = {}
    for r in psum:
        if r["parameter"] != "cost-levelized":
            continue
        t = r.get("target", "")
        if t in ("calculated", "calculated-incl-energy"):
            calc[(r["archetype"], t)] = r
        elif not t:
            sourced[r["archetype"]] = r
    rows = []
    for r in facts:
        if (r.get("parameter") != "cost-levelized" or r.get("target-norm") or r.get("excluded")
                or r.get("set-aside") or r.get("duplicate-of") or not r.get("archetype")):
            continue
        a = r["archetype"]
        cf, cj = calc.get((a, "calculated"), {}), calc.get((a, "calculated-incl-energy"), {})
        v, j = fnum(r.get("value-norm")), fnum(cj.get("central"))
        ratio = v / j if v and j else None
        rows.append({
            "archetype": a, "unit": r.get("unit-norm", ""), "calculated": cf.get("central", ""),
            "calculated-incl-energy": cj.get("central", ""),
            "sourced-central": sourced.get(a, {}).get("central", ""), "sourced-n": sourced.get(a, {}).get("n", ""),
            "example-id": r.get("example-id", ""), "source-id": r.get("source-id", ""),
            "value": r.get("value-norm", ""),
            "ratio-to-calculated-incl-energy": "" if ratio is None else f"{ratio:.3g}",
            "flag": ("" if ratio is None else "sourced above 2x calculated" if ratio > 2
                     else "sourced below half of calculated" if ratio < 0.5 else ""),
            "basis": _basis(r.get("note", "")),
            "region": r.get("region", ""), "year": r.get("year", ""),
            "discount-rate": r.get("discount-rate", ""), "cost-basis": r.get("cost-basis", ""),
            "source-locator": r.get("source-locator", ""), "note": r.get("note", "")})
    rows.sort(key=lambda x: (x["archetype"], x["source-id"], x["example-id"]))
    write(OUTPUT / "levelized-check.csv", COLS, rows)
    return len(rows)
