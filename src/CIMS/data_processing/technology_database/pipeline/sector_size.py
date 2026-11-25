"""Sector sizes for the export to CIMS (Brad 2026-11-20: boilers sized by sector).

The database holds one reference size per archetype (records/reference/
reference-size.csv). Where a CIMS sector uses a different typical size,
records/reference/sector-size.csv names it, and this module writes
archetype-sector-size.csv: the size and the capex restated at that size, for the
archetype and the variants that follow it (their names start with the host's).

Capex per unit of capacity scales by (sector size / reference size)^(exponent - 1),
with the exponent from records/reference/scale-exponent.csv (S173). A variant's
capex scales on the host's part only; a capture add-on keeps its own cost.
"""
from __future__ import annotations

from .common import REFERENCE, fnum, read, write

COLS = ["branch", "archetype", "capacity", "unit", "capacity-default", "exponent",
        "capex-multiplier", "capex-unit", "capex-default", "capex", "capex-low", "capex-high",
        "basis", "decided"]


def rows(psum) -> list[dict]:
    exps = {r["archetype"]: fnum(r["exponent"]) for r in read(REFERENCE / "scale-exponent.csv")}
    ref = {r["archetype"]: r for r in read(REFERENCE / "reference-size.csv")}
    main = {(r["archetype"], r["parameter"]): r for r in psum if not r.get("target")}
    out = []
    for s in read(REFERENCE / "sector-size.csv"):
        a = s["archetype"]
        if a not in ref or a not in exps:
            raise SystemExit(f"sector-size.csv: {a} needs a row in reference-size.csv and scale-exponent.csv")
        if s["unit"] != ref[a]["unit"]:
            raise SystemExit(f"sector-size.csv: {a} unit {s['unit']} differs from reference-size.csv")
        m = (fnum(s["capacity"]) / fnum(ref[a]["capacity"])) ** (exps[a] - 1)
        host = main.get((a, "capex"))
        for k in sorted({x for x, _ in main if x == a or x.startswith(a + "_")}):
            c = main.get((k, "capex"))
            if not c or not host:
                continue
            add = (m - 1) * fnum(host["central"]) if k != a else None

            def at(col):
                v = fnum(c.get(col))
                if v is None:
                    return ""
                return f"{(v * m if add is None else v + add):.6g}"
            out.append({"branch": s["branch"], "archetype": k, "capacity": s["capacity"],
                        "unit": s["unit"], "capacity-default": ref[a]["capacity"],
                        "exponent": f"{exps[a]:g}", "capex-multiplier": f"{m:.4g}",
                        "capex-unit": c.get("unit", ""), "capex-default": c.get("central", ""),
                        "capex": at("central"), "capex-low": at("low"), "capex-high": at("high"),
                        "basis": s["basis"] + ("" if k == a else
                                 f" Variant of {a}: the host's capex part scales; the rest stays."),
                        "decided": s["decided"]})
    return out


def write_sizes(psum, outdir) -> int:
    r = rows(psum)
    write(outdir / "archetype-sector-size.csv", COLS, r)
    return len(r)
