"""Build the harmonized technology database. The one entry point.

    python -m technology_database.build            everything
    python -m technology_database.build --check    stop after the checks (no outputs)
    python -m technology_database.build --record-keys
                                               first add the key of every newly keyed record
                                               to records/reference/record-key.csv
    python -m technology_database.build --quantified-only
                                               the wide view lists only archetypes with
                                               pooled numbers (default: every archetype)

Stages (src/technology_database/docs/pipeline.md):
  1. hierarchy   stamp records/cims-hierarchy.csv, report what changed, and hold
                 every path in the records to it
  2. records     load the archetypes and vocabularies and hold them to the rules
  3. extract     read every extraction as reported (layer 1 -> facts)
  4. clean       evaluate arithmetic, one spelling per unit, per-source record ids
  5. examples    attach each example to its archetype (records/source-example.csv)
  6. checks      the gate: any failure stops the build here
  7. normalise   every fact on its archetype's basis (currency, units, product)
  8. summarise   one value set per archetype and parameter (layer 3)
     fill        graded values for empty cells: identity, estimated, proxy, guess,
                 and not-applicable markers (records/fill/); the summaries before
                 the fill are also written, to output/unfilled/
  9. views       archetype, application, and the wide views (for the workbook only)
 10. reports     source list, citation review, mapping review, data/processed_data/technology_database/vocabulary.md
 11. workbook    the Excel workbook, last, from the outputs

Reads data/raw_data/technology_database/ and data/technology_database/. Writes data/processed_data/technology_database/
and data/processed_data/technology_database/vocabulary.md, and appends new examples to records/source-example.csv.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):
    # run as a script (python src/technology_database/build.py): import the package it sits in
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "technology_database"
    import technology_database  # noqa: F401,E402

from .pipeline import (assembly, net_sec, feedstock, supplied_heat, capital_share, service_request, chain_validation, gwp, capture_energy, capture_share, carriers, checks, correction, ukerc_saving, currency, examples, extract,  # noqa: E402
                      bands, factor_use, facts, fill, hierarchy, normalise, placement, records, reports, roll_in, roll_up, sources,
                      summarise, views, whole_plant, workbook)
from .pipeline.common import OUTPUT, write  # noqa: E402

FACT_COLS = ["record-id", "example-id", "archetype", "source-technology", "parameter", "target",
             "value", "value-min", "value-max", "unit", "value-norm", "value-min-norm",
             "value-max-norm", "unit-norm", "target-norm", "conversion", "heating-value",
             "conversion-class", "conversion-factors", "excluded", "set-aside", "correction", "duplicate-of", "whole-plant", "uncertainty-type",
             "value-type", "formula", "derived-from", "cost-basis", "capacity-basis", "currency",
             "discount-rate", "region", "year", "source-id", "source-file", "source-locator",
             "extraction-file", "via-source-id", "via-locator", "confidence", "status", "note"]
FLOW_COLS = ["record-id", "example-id", "archetype", "source-technology", "direction",
             "commodity", "value", "value-min", "value-max", "unit", "per-unit", "value-norm",
             "value-min-norm", "value-max-norm", "unit-norm", "conversion", "excluded", "co-product",
             "value-type", "formula", "region", "year", "source-id", "source-file",
             "source-locator", "via-source-id", "status", "note"]
CONS_COLS = ["record-id", "example-id", "archetype", "source-technology", "constraint", "region",
             "year", "value", "unit", "constraint-basis", "value-min", "value-max", "value-type",
             "source-id", "source-file", "source-locator", "via-source-id", "status", "note"]
EX_COLS = ["example-id", "archetype", "mapping", "source-id", "source-technology", "process",
           "detail", "sector-reported", "end-use-reported", "step-reported",
           "service-unit-reported", "description", "trl", "trl-as-reported", "trl-scale",
           "state-of-development", "year-available", "extraction-file", "note"]
SUM_COLS = ["central", "low", "high", "n", "n-external", "n-corrected", "central-from", "external-factors",
            "sources", "examples"]
FILL_COLS = ["grade", "method", "rule-id", "inputs", "rationale"]


def main(argv) -> int:
    print("hierarchy")
    hier = hierarchy.Hierarchy()
    print(f"  {len(hier.nodes)} nodes, {len(hier.leaves)} leaves, sha256 {hier.digest}")
    print("records")
    rec = records.Records()
    print(f"  {len(rec.archetypes)} archetypes, {len(rec.service)} service links, "
          f"{len(rec.annotations)} annotations")
    linked = {s["archetype"] for s in rec.service}
    loose = [a["archetype"] for a in rec.archetypes if a["archetype"] not in linked]
    if loose:
        print(f"  {len(loose)} with no connection to the hierarchy: {', '.join(loose)}")
    print("extract")
    L = extract.load()
    for m in L.missing:
        print(f"  warning: {m} not found, skipped")
    stats = facts.clean(L.facts, L.flows, L.cons)
    print(f"  {len(L.examples)} examples, {len(L.facts)} parameter rows, {len(L.flows)} flow "
          f"rows, {len(L.cons)} constraint rows; " + ", ".join(f"{v} {k}" for k, v in stats.items()))
    print("examples")
    ex = examples.apply(L, rec)
    m = Counter(e["mapping"] for e in L.examples.values())
    print(f"  {m['mapped']} mapped, {m['none']} (none), {m['unmapped']} not yet mapped"
          + (f"; {len(ex['new'])} new rows appended to records/source-example.csv" if ex["new"] else ""))
    if ex["stale"]:
        print(f"  {len(ex['stale'])} rows in source-example.csv match no extraction: "
              + ", ".join(ex["stale"][:5]))

    wp = whole_plant.derive(L)
    print(f"  UKERC measures: {wp.get('whole-plant rows', 0)} whole-plant rows derived, "
          f"{wp.get('set aside', 0)} measure costs set aside for a host-plant cost")
    asm = assembly.derive(L)
    if asm:
        print(f"  assembled whole-plant costs: {asm.get('rows', 0)} rows derived, "
              f"{asm.get('checks', 0)} consistency checks against printed totals"
              + (f" ({asm['check mismatches']} differ)" if asm.get("check mismatches") else ""))
    cor = correction.apply(L)
    if cor:
        print(f"  source corrections: {cor.get('added', 0)} values added, {cor.get('replaced', 0)} "
              f"replaced, {cor.get('marked', 0)} already-corrected rows marked, "
              f"{cor.get('rebased', 0)} restated on the archetype's product "
              "(records/reference/source-correction.csv)")
    # a record whose parameter the extraction got wrong (biogenic CO2 entered as
    # combustion or process CO2): records/reference/parameter-remap.csv (2026-10-14)
    from .pipeline.common import REFERENCE, read, split_list
    remap = {r["record-id"]: r for r in read(REFERENCE / "parameter-remap.csv")}
    unknown = set(remap) - {r["record-id"] for r in L.facts}
    if unknown:
        print(f"  parameter-remap.csv names unknown record-ids: {sorted(unknown)}")
        return 1
    for r in L.facts:
        x = remap.get(r["record-id"])
        if x:
            r["note"] = (r.get("note") or "") + (f"  PARAMETER RE-MAPPED from {r['parameter']} to "
                                                 f"{x['parameter']}: {x['reason']} ({x['decided']}).")
            r["parameter"] = x["parameter"]
    if remap:
        print(f"  parameters re-mapped: {len(remap)} rows (records/reference/parameter-remap.csv)")
    from .pipeline import midden_carriers
    mc = midden_carriers.apply(L.facts)
    print("  " + ", ".join(f"{k} {v}" for k, v in mc.items()))
    if "--record-keys" in argv:
        n = checks.write_record_keys(L)
        print(f"  record keys: {n} added to records/reference/record-key.csv")
    print("checks")
    fail = checks.run(rec, hier, L, ex)
    # service requests and chain validations (general rules 2 and 3, 2026-11-05)
    service_request.check(rec, {n for n in hier.nodes}, fail)
    chain_validation.check(rec, fail)
    capital_share.check(rec, fail)
    backlog = fail.pop("_backlog")
    if backlog:
        print(f"  basis backfill outstanding: {sum(backlog.values())} rows marked for review "
              f"({', '.join(f'{s} {n}' for s, n in backlog.most_common())})")
    if fail:
        for k, v in fail.items():
            print(f"\n  {k.upper()} ({len(v)}):")
            for x in v[:12]:
                print(f"      {x}")
            if len(v) > 12:
                print(f"      ... and {len(v) - 12} more")
        return 1
    print("  all checks passed")
    if "--check" in argv:
        return 0

    print("normalise")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    stamp = hierarchy.stamp(hier)
    factors, skipped = currency.factors(L.facts + L.flows)
    if skipped:
        print(f"  no currency factor for {', '.join(skipped)}; those rows are not pooled")
    units = normalise.Units(factors)
    normalise.normalise(L.facts, L.examples, rec, units)
    ns = net_sec.apply(L.facts, L.flows, rec)
    print("  gross energy use: " + (", ".join(f"{k} {v}" for k, v in ns.items()) or "nothing net"))
    print(f"  recovered-product output flows: {net_sec.recovered_flows(L.flows, rec)}")
    gw = gwp.species(L.facts, units.factors)
    print("  CO2-eq on AR5: " + (", ".join(f"{k} {v}" for k, v in gw.items()) or "none"))
    fs = feedstock.apply(L.facts, rec)
    print("  feedstock out of energy use: " + (", ".join(f"{k} {v}" for k, v in fs.items()) or "none"))
    shs = supplied_heat.apply(L.facts, rec)
    print("  supplied heat bought from the heat node: " + (", ".join(f"{k} {v}" for k, v in shs.items()) or "none"))
    cps = capital_share.apply(L.facts)
    print("  capital the unit chain misses (capital-share.csv): " + (", ".join(f"{k} {v}" for k, v in cps.items()) or "none"))
    normalise.normalise_flows(L.flows, rec, units, L.examples)
    xf = normalise.apply_flow_exclusions(L.flows, units.factors)
    print("  flow rows ruled out or rebased (excluded-flow.csv): " + (", ".join(f"{k} {v}" for k, v in xf.items()) or "none"))
    ce = capture_energy.apply(L.facts, L.examples, rec)
    print(f"  capture energy on host plants: {ce.get('set aside', 0)} rows set aside, "
          f"{ce.get('derived per t CO2', 0)} restated per t CO2 on capture archetypes")
    ru = roll_up.apply(L.facts, rec)
    print("  targets rolled into their base parameter: "
          + (", ".join(f"{k} {v}" for k, v in ru.items()) or "none"))
    # example totals before the UKERC savings, which apply to the host's total
    # (sanity-check 2026-10-23, S136)
    et = carriers.example_totals(L.facts, rec)
    print("  energy totals from example carriers: " + (", ".join(f"{k} {v}" for k, v in et.items()) or "none"))
    us = ukerc_saving.derive(L.facts, rec, summarise.summarise)
    correction.propagate(L.facts)
    print("  UKERC buckets: " + ", ".join(f"{k} {v}" for k, v in us.items()))
    pooled = [r for r in L.facts if r.get("value-norm") and not r.get("excluded")]
    cc = Counter(r.get("conversion-class") or "as reported" for r in pooled)
    print("  pooled parameter rows by conversion class: "
          + ", ".join(f"{k} {cc[k]}" for k in ["as reported"] + normalise.convert.CLASSES if cc[k]))
    print("summarise")
    psum = summarise.summarise(L.facts, rec, ("parameter", "target-norm"), L.examples)
    for r in psum:
        r["target"] = r.pop("target-norm")
    cs = carriers.apply(psum, L.facts, rec)
    print("  energy carriers: " + ", ".join(f"{k} {v}" for k, v in cs.items()))
    sh = capture_share.apply(psum, rec, L.facts)
    print("  share-to-capture: " + (", ".join(f"{k} {v}" for k, v in sh.items()) or "none derived"))
    fsum = summarise.summarise(L.flows, rec, ("direction", "commodity"), L.examples)
    # the summaries as the sources alone give them, kept apart so every check can
    # be run with and without the fills (modeller 2026-10-09): output/unfilled/
    unfilled = [dict(r) for r in psum]
    print("fill")
    fst = fill.apply(psum, rec, normalise.target_unit, L.facts, units)
    bt = fst.pop("_backtest")
    above = fst.pop("_above_host", [])
    cc = fst.pop("_comb_conflict", [])
    cco = fst.pop("_comb_conflict_old", [])
    tbp = fst.pop("_total_below_parts", [])
    if tbp:
        print(f"  sourced totals below their parts, replaced by the parts ({len(tbp)}): " + "; ".join(tbp))
    if cco:
        print(f"  combustion CO2 written but above a sourced total with the process CO2 ({len(cco)}; to review): " + "; ".join(cco))
    if cc:
        print(f"  combustion CO2 not written, fuel plus process above a sourced total ({len(cc)}): " + "; ".join(cc))
    if above:
        print(f"  efficiency variants above their host's energy use ({len(above)}): " + "; ".join(above))
    write(OUTPUT / "ccs-variant.csv", ["archetype", "host", "parameter", "unit", "host-value", "add-on",
                                       "tier", "tier1", "tier2", "co2-captured", "all-in", "grade"],
          fst.pop("_ccs"))
    fch = fst.pop("_fuel_choice", {})
    print("  fuel choice: " + ", ".join(f"{k} {v}" for k, v in fch.items()))
    unp = fst.pop("_unpriced")
    if unp:
        print("  carriers with no reference price (records/reference/energy-price.csv): " + ", ".join(unp))
    print("  " + (", ".join(f"{k} {v}" for k, v in fst.items()) or "nothing to fill"))
    ri = roll_in.apply(psum, rec)
    print("  roll-ins: " + ", ".join(f"{k} {v}" for k, v in ri.items()))
    td = roll_in.tier_disagreements(psum, rec)
    if td:
        print(f"  heat tiers against their temperatures: {len(td)} to correct in "
              f"archetype.csv: " + "; ".join(td))
    # a carrier another example names but the archetype's split leaves out (S160,
    # 2026-10-27): listed for the modeller to add by hand (a judgement or an
    # additive row), never added by the pipeline
    have = {(r["archetype"], r.get("target", "")) for r in psum
            if r["parameter"] == "specific-energy-consumption" and r.get("central") not in ("", None)}
    # carriers moved to the alternatives table on purpose (fuel_choice, 2026-11-07)
    from .pipeline import fuel_choice as _fc
    moved = {(r["archetype"], r.get("target", "")) for r in _fc.alternatives}
    gaps = {}
    for r in L.facts:
        a = rec.by_key.get(r.get("archetype", ""))
        t = (r.get("target-norm") or r.get("target") or "").strip()
        if (not a or r["parameter"] != "specific-energy-consumption" or t not in rec.carriers
                or not (r.get("value-norm") or r.get("value-min-norm"))):
            continue
        ex = r.get("excluded") or ""
        if ex and "kept out until" not in ex:
            continue
        if (a["archetype"], t) in have or (a["archetype"], t) in moved:
            continue
        g = gaps.setdefault((a["archetype"], t), {"archetype": a["archetype"], "carrier": t,
                                                   "listed": "yes" if t in split_list(a.get("carriers", "")) else "no",
                                                   "examples": set(), "records": [], "values": []})
        g["examples"].add(r["example-id"])
        g["records"].append(r["record-id"])
        g["values"].append(r.get("value-norm") or r.get("value-min-norm"))
    rows = [dict(g, examples="; ".join(sorted(g["examples"])), records="; ".join(g["records"]),
                 values="; ".join(g["values"])) for g in gaps.values()]
    write(OUTPUT / "carrier-check.csv", ["archetype", "carrier", "listed", "values", "records", "examples"],
          sorted(rows, key=lambda r: (r["archetype"], r["carrier"])))
    print(f"  carriers named by an example but missing from the archetype: {len(rows)} "
          "(output/carrier-check.csv; add by hand)")
    for r in unfilled:
        r.update({"grade": "sourced", "method": "pooled", "rule-id": "", "inputs": "", "rationale": ""})
    roll_in.apply(unfilled, rec)
    cov = fill.coverage(psum, rec)
    write(OUTPUT / "coverage.csv", ["archetype", "family", "product-unit"] + [c for c, _ in fill.CIMS_CELLS] + ["trl"], cov)
    cc = Counter(g for r in cov for c, _ in fill.CIMS_CELLS for g in [r[c]])
    print("  coverage of the CIMS cells (" + ", ".join(c for c, _ in fill.CIMS_CELLS) + "): "
          + ", ".join(f"{k} {cc[k]}" for k in fill.GRADES + [fill.MARKER, "missing"] if cc[k]))
    for rid in dict.fromkeys(b["rule-id"] for b in bt):
        rs = [b for b in bt if b["rule-id"] == rid]
        near = sum(1 for b in rs if b["ratio"] and 0.75 <= float(b["ratio"]) <= 1.25)
        print(f"  back-test {rid}: {sum(b['inside'] == 'yes' for b in rs)}/{len(rs)} sourced "
              f"values inside the estimate's range, {near}/{len(rs)} within 25%")
    write(OUTPUT / "fill-backtest.csv", ["rule-id", "archetype", "parameter", "sourced", "estimate",
                                         "low", "high", "inside", "ratio", "inputs"], bt)
    write(OUTPUT / "example.csv", EX_COLS, sorted(L.examples.values(), key=lambda e: e["example-id"]))
    write(OUTPUT / "example-parameter.csv", FACT_COLS, L.facts)
    write(OUTPUT / "set-aside-host-cost.csv",
          ["archetype", "example-id", "record-id", "parameter", "value", "unit", "source-id",
           "via-source-id", "source-locator", "note"],
          sorted((r for r in L.facts if r.get("set-aside") == "needs-host-cost"),
                 key=lambda r: (r["archetype"], r["example-id"], r["record-id"])))
    # costs sources give on host-with-capture archetypes: capture cost, to move to
    # a co2-capture archetype (reviews/sanity-checks/sanity-check-list.csv)
    write(OUTPUT / "ccs-addon-cost.csv",
          ["archetype", "example-id", "record-id", "parameter", "value", "unit", "cost-basis",
           "source-id", "via-source-id", "source-locator", "note"],
          sorted((r for r in L.facts if r.get("set-aside") == "ccs-add-on"),
                 key=lambda r: (r["archetype"], r["example-id"], r["record-id"])))
    # energy sources give on host plants with capture: the capture plant's, set
    # aside and (where ccs-host-energy.csv maps it) restated per t CO2 captured
    write(OUTPUT / "ccs-addon-energy.csv",
          ["archetype", "example-id", "record-id", "parameter", "target", "value", "unit",
           "value-norm", "unit-norm", "source-id", "via-source-id", "source-locator", "note"],
          sorted((r for r in L.facts if r.get("set-aside") == capture_energy.SET_ASIDE),
                 key=lambda r: (r["archetype"], r["example-id"], r["record-id"])))
    # the worklist for establishing efficiency factors later
    sec = {}
    for r in L.facts:
        if r["parameter"] == "specific-energy-consumption" and r.get("value"):
            sec.setdefault(r["example-id"], f"{r['value']} {r['unit']}")
    aside = [dict(r, **{"fuel-per-unit": sec.get(r["example-id"], "")})
             for r in L.facts if r.get("set-aside") in ("needs-efficiency", "not-bridged")]
    write(OUTPUT / "set-aside-efficiency.csv",
          ["archetype", "example-id", "record-id", "parameter", "value", "unit",
           "fuel-per-unit", "set-aside", "excluded", "duplicate-of", "formula", "source-id",
           "via-source-id", "source-locator", "note"],
          sorted(aside, key=lambda r: (r["archetype"], r["example-id"], r["record-id"])))
    write(OUTPUT / "example-flow.csv", FLOW_COLS, L.flows)
    factor_use.write_outputs(L.facts, units.factors)
    write(OUTPUT / "example-constraint.csv", CONS_COLS, L.cons)
    order = {a["archetype"]: i for i, a in enumerate(rec.archetypes)}
    psum.sort(key=lambda r: (order.get(r["archetype"], 9999), r["parameter"], r["target"]))
    post = checks.summaries(psum)
    if post:
        for k, v in post.items():
            print(f"  FAIL {k}: {len(v)}: " + "; ".join(v[:8]))
        return 1
    fsum.sort(key=lambda r: (order.get(r["archetype"], 9999), r["direction"], r["commodity"]))
    # secondary parameters (values by stream or part) have their own file
    # (modeller 2026-10-09); views still see every summary
    sec = [r for r in psum if placement.placement(r["parameter"], r["target"]) == "secondary"]
    main = [r for r in psum if placement.placement(r["parameter"], r["target"]) != "secondary"]
    write(OUTPUT / "archetype-parameter.csv",
          ["archetype", "parameter", "target", "unit"] + SUM_COLS + FILL_COLS, main)
    # the carriers left out of the wide view: other fuels, purchased steam against
    # on-site firing, other estimates (Brad 2026-11-07); CSV only, not in the workbook
    from .pipeline import fuel_choice
    alt = fuel_choice.alt_rows(psum, rec)
    alt.sort(key=lambda r: (order.get(r["archetype"], 9999), r["rank"] == "", r["rank"] or 0, r["carrier"]))
    write(OUTPUT / "archetype-alternative-carrier.csv", fuel_choice.ALT_COLS, alt)
    unfilled.sort(key=lambda r: (order.get(r["archetype"], 9999), r["parameter"], r["target"]))
    write(OUTPUT / "unfilled" / "archetype-parameter.csv",
          ["archetype", "parameter", "target", "unit"] + SUM_COLS + FILL_COLS,
          [r for r in unfilled if placement.placement(r["parameter"], r["target"]) != "secondary"])
    write(OUTPUT / "unfilled" / "archetype-secondary-parameter.csv",
          ["archetype", "parameter", "target", "unit"] + SUM_COLS + FILL_COLS,
          [r for r in unfilled if placement.placement(r["parameter"], r["target"]) == "secondary"])
    write(OUTPUT / "archetype-secondary-parameter.csv",
          ["archetype", "parameter", "target", "unit"] + SUM_COLS + FILL_COLS, sec)
    write(OUTPUT / "archetype-flow.csv",
          ["archetype", "direction", "commodity", "unit"] + SUM_COLS, fsum)
    nf = Counter(r["example-id"] for r in L.facts + L.flows + L.cons
                 if r.get("value") or r.get("value-min") or r.get("value-max"))
    for e in L.examples.values():
        e["n-facts"] = nf.get(e["example-id"], 0)
    sr = service_request.build(rec)
    cv = chain_validation.build(psum, rec)
    print(f"  service requests: {len(sr)}; chain validations: {len({r['chain-id'] for r in cv})} chains")
    print("views")
    v = views.build(rec, hier, L.examples, psum, fsum, include_empty="--quantified-only" not in argv)
    print("reports")
    srcs = sources.source_list(rec, L.examples, (L.facts, L.flows, L.cons))
    sources.citation_review(srcs)
    reports.mapping_review(rec, hier, L, {"rows": len(hier.nodes), "sha": hier.digest, **stamp},
                           v, ex)
    reports.vocabulary_md(rec, hier, v)
    print("  mapping-review.md, citation-review.md, data/processed_data/technology_database/vocabulary.md")
    print("workbook")
    fam = {f: r["label"] for f, r in rec.families.items()}
    W = v["wide"]
    wcols, wrows = W["wide-archetype"][0], W["wide-archetype"][1]
    fcols, frows, fgroups, funits = v["wide-flow"]
    acols, arows = v["archetype"]
    fam_rank = {f: i for i, f in enumerate(rec.families)}
    src_band = []
    # band the source list in the same family order as the wide view (the main table)
    wide_order = list(dict.fromkeys(r.get("family", "") for r in wrows if r.get("family")))
    fam_order = wide_order + [f for f in rec.families if f not in wide_order]
    for f in fam_order + [""]:
        n0 = len(src_band)
        for r in sorted(srcs, key=lambda x: str(x.get("short-name", "")).lower()):
            fs = [x.strip() for x in str(r.get("families", "")).split(";") if x.strip()]
            if (f in fs) or (not f and not fs):
                src_band.append(dict(r, _band=f))
        if f and len(src_band) == n0:
            # keep the family band visible even when no source informs it yet
            src_band.append({"_band": f, "note": "no source informs this family yet"})
    # general reading notes and the shading key are on the workbook's readme sheet
    # (pipeline/workbook.py README); each banner keeps only what is about its sheet
    # (Brad 2026-10-29)
    # The workbook (Brad 2026-11-04): the archetypes sheet, the carriers-and-species
    # view split into energy, emissions and feedstocks (as in the review packages),
    # and the source list. wide-flow, the archetype list and the long tables are
    # left to the CSVs in output/, which the readme lists.
    wa, wc = W["wide-archetype"], W["wide-carriers-species"]
    wa_rows = bands.expand(wa[1], rec, wa[3])
    wc_rows = bands.expand(wc[1], rec, wc[3], wa[1])
    ident = [c for c in wc[0] if not c.startswith(("specific-energy-consumption", "emissions-",
                                                   "feedstock-consumption"))]
    SPLIT = {"energy": ("specific-energy-consumption",
                        "Energy use by carrier, per product-unit, for the same rows as the archetypes sheet. "
                        "The all-carrier total is on the archetypes sheet. The carriers add up to it, with fuel "
                        "on the most likely fuel (or the fuels a co-fired process needs); fuel is the sum of "
                        "the named fuels beside it, so add the named fuels or fuel, not both. Other fuels "
                        "that could serve the same end use, and purchased steam against on-site firing, are "
                        "in archetype-alternative-carrier.csv, which is not in this workbook."),
             "emissions": ("emissions-",
                           "Emissions by type and species, per product-unit, for the same rows as the "
                           "archetypes sheet. CO2-equivalent totals are on the archetypes sheet."),
             "feedstocks": ("feedstock-consumption",
                            "Feedstock use by carrier (feed whose energy ends up in the product, not counted "
                            "as energy use), per product-unit, for the same rows as the archetypes sheet.")}
    banded = {"archetypes": (wa[0], wa_rows) + wa[2:4] + ("_family", "_sub",
                             "One row per archetype in each end use it serves. Capacity is in the unit named "
                             "in capacity-unit; energy by carrier, emissions by species, and feedstocks are on "
                             "the energy, emissions, and feedstocks sheets; heat tiers and temperatures on the heat sheet.")}
    for name, (pre, banner) in SPLIT.items():
        banded[name] = (ident + [c for c in wc[0] if c.startswith(pre)], wc_rows) + wc[2:4] + (
            "_family", "_sub", banner)
    wh = W["wide-heat"]
    banded["heat"] = (wh[0], bands.expand(wh[1], rec, wh[3], wa[1])) + wh[2:4] + (
        "_family", "_sub",
        "The heat each archetype's process needs, the heat it delivers if it is meant to supply heat, and "
        "the heat it rejects that waste-heat recovery could use, for the same rows as the archetypes sheet. "
        "Each has a tier on every archetype and a temperature where the sources give one; a tier from a "
        "sourced temperature is shaded as sourced, a judgement as a guess.")
    banded["source-list"] = (sources.COLS, src_band, {}, {}, "_band", "", "The sources, by the family they inform.")
    sheet_of = {"archetypes": "wide-archetype", **{n: "wide-carriers-species" for n in SPLIT},
                "heat": "wide-heat"}
    print("  " + workbook.write_book(
        banded, {},
        {**fam, "": "(not tied to one family)"},
        {n: W[k][4] for n, k in sheet_of.items()},
        {n: W[k][5] for n, k in sheet_of.items()},
        {n: W[k][6] for n, k in sheet_of.items()},
        grouped=set(sheet_of),
        band_notes=bands.FAMILY_NOTE))
    q = Counter(a["core-sourced"] for a in arows)
    print(f"\ndone: {len(arows)} archetypes (core values sourced: "
          + ", ".join(f"{n} of 7: {q[n]}" for n in range(7, -1, -1)) + f"); wide view {len(wrows)} rows x {len(wcols)} columns "
          f"(carriers-species {len(W['wide-carriers-species'][0])}, secondary {len(W['wide-secondary-parameter'][0])} columns)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
