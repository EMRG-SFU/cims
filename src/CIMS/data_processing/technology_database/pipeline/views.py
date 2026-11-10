"""The outputs people read: archetype tables, the hierarchy with its technologies and the wide views.

wide-archetype.csv has ONE ROW PER ARCHETYPE by construction: it is a pivot of
archetype-parameter.csv, whose key is the archetype. Each parameter gets a
central value, and `_low`, `_high` and `_n` beside it (low and high only where
the examples spread). Emission species are separate columns. Every cost and
intensity is per unit of the archetype's own product (`product-unit`), so a
column holds CAD per tonne on a kiln row and CAD per GJ on a boiler row: never
sort, average or chart a cost or intensity column across families.

Every archetype has a row, blank or not, including `component` archetypes
(intermediate parts priced to assemble whole-plant costs) and any archetype that
serves no hierarchy node: its `applies-to` reads NO_CONNECTION (2026-09-30).
`--quantified-only` restricts the wide view to archetypes with pooled numbers.
"""
from __future__ import annotations

import re
from collections import defaultdict

from .common import OUTPUT, read, split_list, write
from .hierarchy import Hierarchy
from .normalise import target_unit
from .placement import placement
from .bands import families
from . import capacity_unit

IDENT = ["archetype", "label", "family", "families", "end-uses", "equipment", "discriminator", "modifiers",
         "type", "carriers", "heat-process-tier", "heat-output-tier", "heat-waste-tier", "applies-to", "trl", "product-unit", "product",
         "secondary-products", "product-density", "product-density-unit", "product-energy-content",
         "product-energy-content-unit", "n-examples", "core-sourced", "sources"]


def unit_label(param: str, target: str) -> str:
    u = target_unit(param, "PU", "")
    if "PU" in u:
        return u.replace("PU-per-yr", "(product-unit per yr)").replace("PU", "product-unit") \
                .replace("kPU/yr", "kt/yr (or MW when product-unit is GJ)")
    return u



def product_properties():
    """{product: {product-density, -unit, product-energy-content, -unit}} from
    records/reference/product-property.csv, the values taken from the factors it
    names in conversion-factor.csv (density in t/m3; energy content, HHV, in GJ/t,
    or GJ/m3 for a product given per volume only) (modeller 2026-10-09)."""
    from . import convert
    from .common import REFERENCE
    rows = convert.Factors().rows
    out = {}
    for r in read(REFERENCE / "product-property.csv"):
        d = {}
        f = rows.get(r.get("density-factor", ""))
        if f and f.get("_f"):
            fu, tu = convert.atom(f["from-unit"]), convert.atom(f["to-unit"])
            k = f["_f"] * tu[1] / fu[1]
            if fu[0] == "volume" and tu[0] == "mass":
                d["product-density"] = f"{k:.4g}"
            elif fu[0] == "mass" and tu[0] == "volume":
                d["product-density"] = f"{1 / k:.4g}"
            if "product-density" in d:
                d["product-density-unit"] = "t/m3"
        f = rows.get(r.get("energy-content-factor", ""))
        if f and f.get("_f"):
            fu, tu = convert.atom(f["from-unit"]), convert.atom(f["to-unit"])
            if tu and fu and tu[0] == "energy" and fu[0] in ("mass", "volume"):
                d["product-energy-content"] = f"{f['_f'] * tu[1] / fu[1]:.4g}"
                d["product-energy-content-unit"] = "GJ/t" if fu[0] == "mass" else "GJ/m3"
        out[r["product"]] = d
    return out


NO_CONNECTION = "no connection"

WIDE_NOTES = {
    "source-corrections": (
        "Parameters whose pooled value includes a figure the source prints wrongly and the "
        "modeller corrected (a slipped decimal or zero, a wrong unit or carrier label), as "
        "examples so corrected / examples pooled. Each correction, the figure as printed, and "
        "the evidence: records/reference/source-correction.csv; the rows carry the "
        "correction-id in example-parameter.csv (correction) and a note starting "
        "ASSUMED CORRECTION. These values are also counted in external-conversions and "
        "shaded as converted in the workbook (see readme)."),
    "filled": (
        "Parameters with no sourced value, filled by the fill step and graded: identity "
        "(calculated from this archetype's own values), estimated (from other archetypes or a "
        "class rule fitted on this database), proxy (a class constant from a named outside "
        "catalogue, records/fill/fill-assumption.csv) or guess (a judgement, "
        "records/fill/fill-judgement.csv); not-applicable marks a parameter that does "
        "not characterise the archetype. Shaded by grade in the workbook (see readme); their _n is 0. Rule, "
        "inputs, and rationale: archetype-parameter.csv (grade, method, rule-id, inputs, "
        "rationale); rules in records/fill/."),
    "external-conversions": (
        "Parameters whose pooled value rests on a conversion factor from outside the source "
        "(a heating value, a density, a process ratio, a pooled efficiency, or an assumption), "
        "as examples so converted / examples pooled. Those values are shaded as converted in the "
        "workbook (see readme). Which factors: archetype-parameter.csv (external-factors); how each example "
        "was converted: example-parameter.csv (conversion, conversion-class, "
        "conversion-factors)."),
}


# short definitions of the identity columns, for the column legend and the header notes
IDENT_NOTES = {
    "archetype": "Identifier: the equipment, then what sets the option apart, then any modifiers, "
                 "joined by underscores.",
    "label": "Plain-language name of the archetype.",
    "family": "Home family: the shared service (heat supply, machine drive, ...) or the sector "
              "whose process equipment the archetype is.",
    "families": "Every family the archetype appears under in the workbook: its home family "
                "first, then the families of the sectors whose end uses it serves.",
    "end-uses": "The end uses (hierarchy service nodes) the archetype serves.",
    "equipment": "Equipment class: the first part of the identifier.",
    "discriminator": "What sets the archetype apart from others on the same equipment (the "
                     "second part of the identifier); blank when nothing does.",
    "modifiers": "Variant tags that end the identifier: an efficiency step "
                 "(efficiency-moderate, efficiency-strong), a capture variant (ccs-concentrated, "
                 "ccs-diffuse) or a fuel or heating change.",
    "type": "conventional (the incumbent in Canadian plants), efficient variant, alternative "
            "(commercial but not common in Canada), prospective (technology readiness level "
            "5-8), component or reference (kept for their numbers; not offered to the model).",
    "carriers": "The energy carriers the archetype can use.",
    "heat-process-tier": "Temperature band of the heat the process needs: low-temp below 150 degC, "
                         "medium-temp 150-400 degC, high-temp above 400 degC; several when the process "
                         "spans bands; not-applicable when it uses no process heat (shaft work, "
                         "transport, cooling, lighting) or only supplies heat. Follows "
                         "heat-process-temp where the sources give one.",
    "heat-output-tier": "Temperature band of the heat an archetype meant to supply heat delivers "
                        "(heat supply, boilers, archetypes with heat or steam among their products); "
                        "not-applicable for every other archetype. Covers every band heat-output-temp "
                        "spans where the sources give one.",
    "heat-waste-tier": "Temperature band of the main stream leaving the archetype that waste-heat "
                       "recovery could use (exhaust, flue gas, cooler air, cooling water, condenser "
                       "heat), after the recovery the equipment itself includes; not-applicable when "
                       "there is none. Follows heat-waste-temp where the sources give one.",
    "applies-to": "The hierarchy nodes the archetype serves, or \"no connection\".",
    "trl": "Technology readiness level, from 1 (basic principles observed) to 9 (proven in "
           "commercial operation).",
    "product-unit": "The unit of output every cost and intensity is per.",
    "product": "The main product or service the product-unit counts.",
    "secondary-products": "Other useful outputs (co-products); costs are not split between them.",
    "product-density": "Density of the product, for converting between mass and volume.",
    "product-density-unit": "Unit of product-density.",
    "product-energy-content": "Energy content of the product (higher heating value), for "
                              "converting between mass or volume and energy.",
    "product-energy-content-unit": "Unit of product-energy-content.",
    "n-examples": "Number of source examples mapped to the archetype.",
    "core-sourced": "How many of the seven core model values (capex, fixed opex, variable opex, "
                    "energy use, lifetime, annual output, CO2 emissions) come from a source, 0 to 7; "
                    "the others are filled (identity, estimated, proxy or guess), not applicable, or "
                    "missing.",
    "sources": "Short names of the sources behind the examples.",
}
SPREAD_NOTES = {
    "_low": "Lowest value any example gives for {b} (for a filled value, the low end of the fill).",
    "_high": "Highest value any example gives for {b} (for a filled value, the high end of the fill).",
    "_n": "Number of examples pooled into {b}; 0 for a filled value.",
}


HEAT_TIERS = ("heat-process-tier", "heat-output-tier", "heat-waste-tier")
HEAT_TEMPS = ("heat-process-temp", "heat-output-temp", "heat-waste-temp")


def heat_sheet(view, heat_cells, records):
    """Order the heat sheet process, output, waste, each tier before its temperature,
    and grade every tier cell: as the temperature it follows from when the recorded
    tier equals the tiers that temperature spans, a guess (a judgement) when no
    temperature decides it, not-applicable when the archetype has no such heat."""
    from .roll_in import tier_from
    cols, rows, groups, units, notes, flagged, graded = view
    ident = [c for c in cols if groups.get(c) == "identity" and c not in HEAT_TIERS]
    new = list(ident)
    for tier, temp in zip(HEAT_TIERS, HEAT_TEMPS):
        new += [tier] + [c for c in (temp, f"{temp}_low", f"{temp}_high", f"{temp}_n") if c in cols]
        groups[tier] = groups.get(temp, "technical") or "technical"
        units[tier] = ""
    for row in rows:
        k = row["archetype"]
        for tier, temp in zip(HEAT_TIERS, HEAT_TEMPS):
            v = row.get(tier, "")
            s = heat_cells.get(k, {}).get(temp)
            order = [t for t in ("low-temp", "medium-temp", "high-temp") if t in v]
            if v == "not-applicable":
                g = "not-applicable"
            elif s and tier_from(s) == "; ".join(order):
                g = ("converted" if (k, temp) in flagged else s.get("grade", "sourced"))
            else:
                g = "guess"
            graded[(k, tier)] = g
    return new, rows, groups, units, notes, flagged, graded


def applies_to(key, service):
    """The nodes an archetype serves, or NO_CONNECTION when it serves none."""
    return "; ".join(Hierarchy.short(s["branch"]) for s in service.get(key, [])) or NO_CONNECTION


def build(records, hier, examples, summaries, flow_summaries, include_empty=True,
          outdir=OUTPUT, wide_only=False):
    """`outdir`: the folder written to; `wide_only`: write the wide views only (the
    unfilled set, output/unfilled/, 2026-10-09). Also returns `flagged`: the (archetype, column) cells of the wide view whose
    pooled value rests on at least one factor from outside its source; and
    `graded`: {(archetype, column): grade} for the cells the fill step wrote."""
    service = defaultdict(list)
    for s in records.service:
        service[s["archetype"]].append(s)
    ex_by_arch = defaultdict(list)
    for e in examples.values():
        if e.get("archetype"):
            ex_by_arch[e["archetype"]].append(e)
    fam_order = {f: i for i, f in enumerate(records.families)}
    param_order = {p: i for i, p in enumerate(records.parameters)}

    # ---- archetype.csv: every archetype, with what the database holds ------
    with_data = ({s["archetype"] for s in summaries
                  if s.get("grade", "sourced") == "sourced" or s.get("method") == "range-midpoint"}
                 | {s["archetype"] for s in flow_summaries})
    with_fill = {s["archetype"] for s in summaries
                 if s.get("grade", "sourced") not in ("sourced", "not-applicable")}
    with_facts = {e["archetype"] for e in examples.values() if e.get("n-facts")}
    from .fill import CIMS_CELLS, coverage
    core = {r["archetype"]: sum(1 for c, _ in CIMS_CELLS if r[c] == "sourced")
            for r in coverage(summaries, records)}
    arows = []
    props = product_properties()
    fams = families(records)   # home family first, then the end uses' families (2026-10-09)
    for a in records.archetypes:
        k = a["archetype"]
        exs = ex_by_arch.get(k, [])
        src = sorted({e["source-id"] for e in exs}, key=lambda s: (records.rank(s), s))
        arows.append(dict(a, **props.get(a.get("product", ""), {}), **{
            "families": fams.get(k, a["family"]),
            "applies-to": applies_to(k, service),
            "end-uses": "; ".join(dict.fromkeys(s["end-use"] for s in service.get(k, []))),
            "n-examples": len(exs),
            # alphabetical by short name (modeller 2026-10-09; was by source rank)
            "sources": "; ".join(sorted((records.short(s) for s in src), key=str.lower)),
            # how many of the seven core model values come from a source (Brad
            # 2026-11-09; replaces the quantified / estimated status flag)
            "core-sourced": core.get(k, 0)}))
    # cross-cutting service families first, in records/vocabulary order (heat supply,
    # power, machine drive, ...), then the sector families alphabetically by the
    # label a reader sees (modeller 2026-09-26, 2026-10-08); the catch-all other
    # industry family last (Brad 2026-10-29)
    fam_rank = {f: ((0, i, "") if r.get("kind") == "service" else
                    (1, 1 if f == "other-industry" else 0, (r.get("label") or f).lower()))
                for i, (f, r) in enumerate(records.families.items())}
    from .bands import type_rank
    arows.sort(key=lambda r: (fam_rank.get(r["family"], (2, 0, r["family"])), type_rank(r), r["archetype"]))
    if not wide_only:
        write(outdir / "archetype.csv",
              IDENT + ["trl-basis", "description", "examples-named", "note"], arows)

    # ---- archetype-application.csv: the long view, every node served ---------
    app = []
    for a in arows:
        for s in service.get(a["archetype"], []):
            for n in hier.expand(s["branch"]):
                app.append({"archetype": a["archetype"], "family": a["family"],
                            "node": n, "annotation": s.get("end-use", ""),
                            "via": s["branch"] if "*" in s["branch"] else "",
                            "note": s.get("note", "")})
    if not wide_only:
        write(outdir / "archetype-application.csv",
              ["archetype", "family", "node", "annotation", "via", "note"], app)

    # ---- cims-hierarchy.csv and cims-hierarchy_technology.csv -----------------
    # cims-hierarchy.csv is the snapshot as the build used it. The technology
    # version keeps its row order, with `technology` as the second column: a node
    # is repeated once per archetype serving it; a node nothing serves appears
    # once with `technology` blank.
    by_node = defaultdict(list)
    for r in app:
        if r["archetype"] not in by_node[r["node"]]:
            by_node[r["node"]].append(r["archetype"])
    hrows = read(hier.path) if not wide_only else []
    hcols = [c for c in (list(hrows[0]) if hrows else ["branch"]) if c != "technology"]
    if not wide_only:
        write(outdir / "cims-hierarchy.csv", hcols, hrows)
    tech = []
    for h in hrows:
        b = (h.get("branch") or "").strip()
        if not b:
            continue
        for t in sorted(by_node.get(b, [])) or [""]:
            tech.append(dict(h, technology=t))
    if not wide_only:
        write(outdir / "cims-hierarchy_technology.csv", hcols[:1] + ["technology"] + hcols[1:], tech)

    # ---- the wide views ---------------------------------------------------------
    # Three views (modeller 2026-10-08/09), placed by pipeline/placement.py:
    #   wide-archetype                   one column set per parameter, the main view
    #   wide-carriers-species            specific energy by carrier and emissions by
    #                                    species, alphabetical (CO2-eq stays main)
    #   wide-secondary-parameter         every other target (CSV only, not in the
    #                                    workbook): values by stream or part
    # The CHP efficiency, the temperature range and the single-stream efficiency
    # are already rolled into the summaries (pipeline/roll_in.py).
    by_arch = defaultdict(dict)
    pt, sunit = {}, defaultdict(set)
    for s in summaries:
        base = s["parameter"] + (f"_{s['target']}" if s.get("target") else "")
        by_arch[s["archetype"]][base] = s
        pt[base] = (s["parameter"], s.get("target", ""))
        sunit[base].add(s.get("unit", ""))
    fill_note = {}
    SHEET = {"main": "wide-archetype", "carriers": "wide-carriers-species",
             "secondary": "wide-secondary-parameter"}
    sheets = {v: defaultdict(dict) for v in SHEET.values()}
    for k, cells in by_arch.items():
        for b, s in cells.items():
            where = SHEET.get(placement(*pt[b]))
            if where:
                sheets[where][k][b] = s
    colmeta = {}

    def unit_of(b):
        p, t = pt.get(b) or colmeta.get(b, (b, ""))
        if t and placement(p, t) == "secondary":
            u = "; ".join(sorted(x for x in sunit.get(b, ()) if x))
            if p in ("capacity", "annual-output") and u:
                return f"{u} {t}" + (" per yr" if p == "annual-output" else "")
            return u or unit_label(p, t)
        if p == "annual-output":
            return "product-unit per yr"
        return unit_label(p, t)

    def note_of(b):
        p, t = pt.get(b) or colmeta.get(b, (b, ""))
        pdef = records.parameters.get(p, {})
        extra = {
            "efficiency": "On CHP archetypes (label says CHP) this column is the total CHP "
                          "efficiency, electricity plus useful heat over fuel in (efficiency-chp); "
                          "the electric and heat efficiencies are beside it.",
            "efficiency_unspecified": "CHP archetypes only: an efficiency the source gives without "
                                      "saying electric, heat or total. The main view's efficiency "
                                      "on these rows is the total CHP efficiency.",
            "efficiency_heat": "Heat out over energy in. On a single-output heat archetype it repeats "
                               "the efficiency column.",
            "efficiency_electricity": "Electricity out over energy in. On a single-output power "
                                      "archetype it repeats the efficiency column.",
            "efficiency-chp": "Non-CHP archetypes: a total heat-plus-power efficiency some sources give "
                              "(a boiler or turbine run in CHP mode).",
            "heat-output-temp": "Low and high also span the minimum and maximum temperatures "
                                "sources give (heat-output-temp-min, heat-output-temp-max).",
            "heat-process-temp": "Low and high also span the minimum and maximum temperatures "
                                 "sources give (heat-process-temp-min, heat-process-temp-max).",
            "heat-waste-temp": "Low and high also span the minimum and maximum temperatures "
                               "sources give (heat-waste-temp-min, heat-waste-temp-max).",
        }.get(b, "")
        return " ".join(x for x in (
            pdef.get("definition", ""), f"Species or stream: {t}." if t else "", extra) if x)

    cap_rules = capacity_unit.load()

    def emit(name, cells, ident, order):
        keys = sorted({b for c in cells.values() for b in c}, key=order)
        spread = {b for b in keys if any(c[b].get("low") for c in cells.values() if b in c)}
        cols = list(ident)
        if name == "wide-archetype":
            cols += ["external-conversions", "source-corrections", "filled"]
        units, groups, notes = {}, {c: "identity" for c in cols}, dict(fill_note)
        for b in keys:
            extra = [b] + ([f"{b}_low", f"{b}_high"] if b in spread else []) + [f"{b}_n"]
            if name == "wide-archetype" and b == "capacity":
                cols.append("capacity-unit")
                groups["capacity-unit"] = records.parameters.get("capacity", {}).get("parameter-class", "")
                units["capacity-unit"] = ""
                notes["capacity-unit"] = ("The unit the capacity columns are in, chosen per archetype as the "
                                          "unit its industry rates plants in (records/reference/capacity-unit.csv). "
                                          "Not the product-unit: annual output and the per-capacity costs stay "
                                          "per product-unit.")
            cols += extra
            p = (pt.get(b) or colmeta.get(b, (b, "")))[0]
            for c in extra:
                units[c] = ("" if c.endswith("_n") else
                            "see capacity-unit" if name == "wide-archetype" and b == "capacity" else unit_of(b))
                groups[c] = records.parameters.get(p, {}).get("parameter-class", "")
            notes[b] = note_of(b)
            for c in extra[1:]:
                notes[c] = SPREAD_NOTES[c[len(b):]].format(b=b)
        index = {c: i for i, c in enumerate(cols)}
        rows, flagged, graded = [], set(), {}
        for a in arows:
            if a["archetype"] not in with_data and not include_empty:
                continue
            row = dict(a)                  # family and equipment stay in the row for banding
            ext, filled, cor = [], [], []
            for b, s in cells.get(a["archetype"], {}).items():
                g = s.get("grade", "sourced")
                if g != "sourced":
                    filled.append((b, f"{b} {g}"))
                    graded.update({(a["archetype"], c): g for c in (b, f"{b}_low", f"{b}_high")})
                row[b], row[f"{b}_n"] = ("not applicable" if g == "not-applicable" else s["central"]), s["n"]
                if b in spread:
                    row[f"{b}_low"], row[f"{b}_high"] = s["low"], s["high"]
                if s.get("n-corrected"):
                    cor.append((b, f"{b} {s['n-corrected']}/{s['n']}"))
                if s.get("n-external"):
                    ext.append((b, f"{b} {s['n-external']}/{s['n']}"))
                    flagged |= {(a["archetype"], c) for c in (b, f"{b}_low", f"{b}_high")}
            if name == "wide-archetype":
                capacity_unit.apply(cap_rules, row)
                row["external-conversions"] = "; ".join(t for _, t in sorted(ext, key=lambda x: index.get(x[0], 0)))
                row["source-corrections"] = "; ".join(t for _, t in sorted(cor, key=lambda x: index.get(x[0], 0)))
                row["filled"] = "; ".join(t for _, t in sorted(filled, key=lambda x: index.get(x[0], 0)))
            rows.append(row)
        return cols, rows, groups, units, notes, flagged, graded

    def by_param(b):
        p, t = pt.get(b) or colmeta.get(b, (b, ""))
        return (param_order.get(p, 99), t)

    fill_note.update(WIDE_NOTES)
    fill_note.update(IDENT_NOTES)
    ident_main = [c for c in IDENT if c not in ("family", "equipment") + HEAT_TIERS]
    ident_side = ["archetype", "label", "product-unit", "product"]
    # the heat columns have their own sheet (Brad 2026-11-09): process, output, waste,
    # each a tier (a value on every archetype) and the temperature that decides it
    heat_cells = defaultdict(dict)
    for k, c in sheets["wide-archetype"].items():
        for b in [b for b in c if b in HEAT_TEMPS]:
            heat_cells[k][b] = c.pop(b)
    out = {}
    for name, cells, ident, order in (("wide-archetype", sheets["wide-archetype"], ident_main, by_param),
                                      ("wide-carriers-species", sheets["wide-carriers-species"], ident_side, lambda b: b),
                                      ("wide-secondary-parameter", sheets["wide-secondary-parameter"], ident_side, by_param),
                                      ("wide-heat", heat_cells, ident_side + list(HEAT_TIERS), by_param)):
        out[name] = emit(name, cells, ident, order)
    out["wide-heat"] = heat_sheet(out["wide-heat"], heat_cells, records)
    # the wide views are built for the workbook only; no wide CSVs (Brad 2026-11-06).
    # The column legend serves the workbook and the review packages.
    if not wide_only:
        legend_sheet = {"wide-archetype": "archetypes", "wide-carriers-species": "carriers-species",
                        "wide-heat": "heat"}
        write(outdir / "column-legend.csv", ["sheet", "column", "group", "unit", "note"],
              [{"sheet": legend_sheet[name], "column": c, "group": o[2].get(c, ""), "unit": o[3].get(c, ""),
                "note": o[4].get(c, "")} for name, o in out.items() if name in legend_sheet for c in o[0]],
              quiet=True)

    # ---- wide-flow.csv: central values per commodity --------------------------
    fby = defaultdict(dict)
    fcols = set()
    for s in flow_summaries:
        c = ("in_" if s["direction"] == "input" else "out_") + s["commodity"]
        fby[s["archetype"]][c] = s["central"] or ""
        fby[s["archetype"]][c + "_n"] = s["n"]
        fcols.add(c)
    fcols = sorted(fcols, key=lambda c: (not c.startswith("in_"), c))
    fident = ["archetype", "label", "family", "equipment", "product-unit", "sources"]
    flow_cols = fident + [x for c in fcols for x in (c, c + "_n")]
    frows = []
    for a in arows:
        if a["archetype"] in fby:
            row = {c: a.get(c, "") for c in fident}
            row.update(fby[a["archetype"]])
            frows.append(row)
    funits = {c: ("" if c in fident or c.endswith("_n") else "per product-unit") for c in flow_cols}
    fgroups = {c: ("identity" if c in fident else "inputs" if c.startswith("in_") else "outputs")
               for c in flow_cols}
    return {"archetype": (IDENT, arows),
            "wide": out,
            "wide-flow": (flow_cols, frows, fgroups, funits),
            "application": app}
