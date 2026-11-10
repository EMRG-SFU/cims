"""Clean-up passes on the facts as loaded, before they meet an archetype.

None of these changes what a source said. They evaluate arithmetic a source
wrote as a formula, spell each unit one way, write fractions as percent, link a
derived capex row to the row it came from, flag a quantified row missing the
basis parameter.csv requires, and number records within each source so that a
record id is citable: tp-ukerc_2013_001-0042 is the 42nd row taken from UKERC.
"""
from __future__ import annotations

import ast
import sys
from collections import defaultdict

from .common import REFERENCE, VOCAB, currency_year, read

UNIT_SPELLING = {"C": "degC", "c": "degC", "degrees C": "degC",
                 "kgeCO2/unit": "kg/unit", "kgCO2/unit": "kg/unit",
                 "tCO2/unit": "t/unit", "yrs": "years", "yr": "years",
                 # added 2026-10-02: spellings only, no change of quantity
                 "tCO2/t": "t/t", "short-t/yr": "short-ton/yr", "g/GJ_i": "g/GJ",
                 # added 2026-10-13 (sanity-check S108): spellings of a unit the
                 # table already defines; each was dropping rows as "no conversion"
                 "y": "years", "a": "years", "mtpa": "Mt/yr", "MTPA": "Mt/yr",
                 "Mtpa": "Mt/yr", "tpa": "t/yr", "bpd": "bbl/d", "BPD": "bbl/d",
                 "bar(g)": "bar", "barg": "bar", "tCO2(eq)/GWh": "t/GWh",
                 "tCO2/GWh": "t/GWh"}
# denominators spelled as one word (2026-10-13, S108): a cost per tpa is a cost
# per t-per-yr of capacity
DEN_SPELLING = (("/mtpa", "/Mt-per-yr"), ("/MTPA", "/Mt-per-yr"), ("/tpa", "/t-per-yr"),
                ("/bpd", "/bbl-per-d"), ("/BPD", "/bbl-per-d"))
# trailing species spellings in a money unit (EUR/t-co2e is EUR/t-CO2-eq)
SPECIES_SPELLING = (("/t-co2e", "/t-CO2-eq"), ("/t-co2eq", "/t-CO2-eq"), ("/t-co2", "/t-CO2"))
FRACTION = {"fraction": "%", "fraction/yr": "%/yr"}
BASIS_NOTE = ("  BASIS NOT RECORDED: parameter.csv requires a {col} for {param} "
              "and this extraction does not give one. Marked for review rather "
              "than guessed.")

ASSUMED_INSTALLED = ("  COST BASIS ASSUMED: installed. The source does not state one; unstated "
                     "capex is taken as installed (modeller 2026-10-02).")


def _arith(expr):
    """Value of a formula made only of numbers and + - * / **, else None.
    Formulas that name another parameter are the extraction template's job."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None
    ok = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Add, ast.Sub,
          ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd)
    if not all(isinstance(n, ok) for n in ast.walk(tree)):
        return None
    try:
        return eval(compile(tree, "<formula>", "eval"), {"__builtins__": {}}, {})
    except Exception:
        return None


def clean(facts, flows, cons) -> dict:
    stats = defaultdict(int)
    # "REVIEW:" in an extraction note marks the row for review; it is still pooled
    for r in facts + flows + cons:
        if "REVIEW:" in (r.get("note") or "") and r.get("status") == "draft":
            r["status"] = "review"
            stats["rows marked REVIEW"] += 1
    for r in facts:
        if not r.get("value") and r.get("formula"):
            v = _arith(r["formula"])
            if v is not None:
                r["value"] = f"{v:.6g}"
                stats["formulas evaluated"] += 1
    for r in facts:
        if r.get("unit") in UNIT_SPELLING:
            r["unit"] = UNIT_SPELLING[r["unit"]]
        for a, b in SPECIES_SPELLING:
            if (r.get("unit") or "").endswith(a):
                r["unit"] = r["unit"][: -len(a)] + b
        for a, b in DEN_SPELLING:
            if (r.get("unit") or "").endswith(a):
                r["unit"] = r["unit"][: -len(a)] + b
        if r.get("unit") == "MWe":
            # electrical megawatts: a capacity on the electric-output basis, which
            # normalise puts per unit of heat on a CHP archetype (2026-10-13, S108)
            r["unit"] = "MW"
            if not (r.get("capacity-basis") or "").strip():
                r["capacity-basis"] = "electric-output"
    for r in facts:
        if r.get("unit") in FRACTION:
            for k in ("value", "value-min", "value-max"):
                v = (r.get(k) or "").strip()
                if v:
                    try:
                        r[k] = f"{float(v) * 100:.6g}"
                    except ValueError:
                        pass
            if r.get("formula"):
                r["formula"] = f"({r['formula']}) * 100"
            r["unit"] = FRACTION[r["unit"]]
            stats["fractions written as %"] += 1

    # a derived capex points at the as-reported capex of the same example
    asrep = {}
    for r in facts:
        if r["parameter"] == "capex" and not r.get("formula"):
            asrep.setdefault((r["example-id"], r.get("region", "")), r["record-id"])
    for r in facts:
        if r["parameter"] == "capex" and r.get("formula") and not r.get("derived-from"):
            src = asrep.get((r["example-id"], r.get("region", "")))
            if src:
                r["derived-from"] = src
                stats["derived rows linked"] += 1

    # a capital cost whose source states no cost basis is taken as installed
    # (modeller 2026-10-02): the usual basis of a published capex, and marked so
    for r in facts:
        if (r["parameter"] == "capex" and not (r.get("cost-basis") or "").strip()
                and (r.get("value") or r.get("value-min") or r.get("value-max"))):
            r["cost-basis"] = "installed"
            r["note"] = (r.get("note") or "") + ASSUMED_INSTALLED
            stats["capex cost basis assumed installed"] += 1

    # capacity bases the extraction left out, recorded per row after reading the
    # source (records/reference/capacity-basis.csv; Brad 2026-10-28: recover the
    # dropped source capacities). A `value-new` is the per-unit figure the
    # source's own numbers give (one LNG train = plant nameplate / trains).
    def _k(x):
        try:
            return f"{float(x):.6g}"
        except (TypeError, ValueError):
            return ""
    cb = {(c["example-id"], _k(c["value"]), _k(c["value-min"]), _k(c["value-max"])): c
          for c in read(REFERENCE / "capacity-basis.csv")}
    used = set()
    for r in facts:
        if r["parameter"] != "capacity":
            continue
        key = (r["example-id"], _k(r.get("value")), _k(r.get("value-min")), _k(r.get("value-max")))
        c = cb.get(key)
        if not c:
            continue
        used.add(key)
        r["capacity-basis"] = c["capacity-basis"]
        if c.get("value-new"):
            # a range-only row is restated as its midpoint
            r["value"], r["formula"], r["value-type"] = c["value-new"], c["formula"], "derived"
            r["value-min"] = r["value-max"] = ""
        if c.get("unit-new"):
            r["unit"] = c["unit-new"]
        if c.get("target-new") is not None and c.get("target-new") != "":
            r["target"] = "" if c["target-new"] == "-" else c["target-new"]
        if c.get("assume"):
            # the restatement rests on a proxy: graded as an assumption downstream
            r["assume"] = c["assume"]
        r["note"] = ((r.get("note") or "") + f"  CAPACITY BASIS RECORDED: {c['capacity-basis']} "
                     f"(records/reference/capacity-basis.csv; {c['decided']})"
                     + (f": {c['note']}." if c.get("note") else "."))
        stats["capacity bases recorded"] += 1
    missing = set(cb) - used
    if missing:
        sys.exit(f"capacity-basis.csv: {len(missing)} rows match no capacity fact: {sorted(missing)[:3]}")

    # the basis each parameter requires, read from the dictionary
    basis_req = {r["parameter"]: (r.get("basis-required") or "").strip()
                 for r in read(VOCAB / "parameter.csv")
                 if (r.get("basis-required") or "").strip()}
    for r in facts:
        col = basis_req.get(r["parameter"], "")
        if not col or not (r.get("value") or r.get("value-min") or r.get("value-max")):
            continue
        if (r.get(col) or "").strip():
            continue
        stats["rows missing a required basis"] += 1
        if r.get("status") == "draft":
            r["status"] = "review"
        if "BASIS NOT RECORDED" not in (r.get("note") or ""):
            r["note"] = (r.get("note") or "") + BASIS_NOTE.format(col=col, param=r["parameter"])

    bad = [r["record-id"] for r in facts
           if (r.get("currency") or "") != currency_year(r.get("unit", ""))]
    if bad:
        sys.exit(f"currency is not the currency-year of unit on {len(bad)} rows: {bad[:5]}")

    remap = _renumber(facts, "tp")
    _renumber(flows, "tf")
    _renumber(cons, "tc")
    dangling = []
    for r in facts:
        refs = [x.strip() for x in (r.get("derived-from") or "").split(";") if x.strip()]
        if refs:
            out = []
            for ref in refs:
                (out if ref in remap else dangling).append(remap.get(ref, f"{r['record-id']}->{ref}"))
            r["derived-from"] = "; ".join(out)
    if dangling:
        sys.exit(f"derived-from does not resolve after renumbering: {dangling[:5]}")
    ids = [r["record-id"] for r in facts]
    assert len(ids) == len(set(ids)), "record-id is not unique"
    return dict(stats)


def _renumber(rows, prefix):
    seq, remap = defaultdict(int), {}
    for r in rows:
        sid = (r.get("source-id") or "").strip() or "unattributed"
        seq[sid] += 1
        new = f"{prefix}-{sid}-{seq[sid]:04d}"
        remap[r["record-id"]] = new
        r["record-id"] = new
    return remap
