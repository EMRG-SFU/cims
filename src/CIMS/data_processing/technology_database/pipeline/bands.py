"""Workbook banding of the wide views: family, then end use (modeller 2026-10-09).

Each family band is split into end-use sub-bands in process-chain order (ORDER).
An archetype that serves several end uses is repeated under each, in the
workbook only (the CSVs stay one row per archetype), and each repeat sits in the
family whose sector owns that end use (haul trucks under mining and under oil and
gas: oil sands mining), so a sub-band gathers every archetype competing for it.
Cross-sector end uses (heat tiers, CO2 capture) stay in the archetype's own
family. Repeats are not styled differently (modeller 2026-10-09). `families()`
gives the derived families column.

  * heat supply: the sub-bands are the heat tiers in the archetype's `heat-output-tier`
    column (the shared steam node carries only the first tier on the service
    record), plus any other end use it serves;
  * machine drive: kept by equipment in its own band, because most drives serve
    many end uses (pumps, fans, compressed air, ...); the family band says so.
    End uses owned by a sector family (gas pipelines, gas processing) are
    repeated in that family like any other archetype's;
  * sub-bands with the same label (oxygen supply for steel and for smelters,
    on-site hydrogen for refineries and upgraders) are one sub-band;
  * an archetype that serves no end use goes in a closing "(no connection)" sub-band;
  * one product per sub-band (modeller 2026-10-21): where the archetypes on an end
    use make more than one product (ammonia and methanol), the end use splits into
    one sub-band per product, labelled "<end use>: <product> (<unit>)". A split that
    only reflects two names for one product shows up here and is fixed by renaming
    the product in records/archetype.csv (reviews/band-product-review_2026-10-22.csv);
  * a product override (modeller 2026-10-22): a records/archetype-service.csv row with
    `product` and `product-unit` shows the archetype under that end use in another
    product. `factor` is the new units in one home unit (18.5 GJ of pellets per t;
    blank = 1, a label change), or `process-heat`: GJ of process heat per home unit = the archetype's SEC
    x its efficiency, or the judged efficiency or host in
    records/reference/process-heat-efficiency.csv (ruling E2, light-industrial heat).
    Per-unit columns are divided by the factor, per-year output and capacity
    multiplied. The CSV wide views stay on the home product;
  * a display unit by product (Brad 2026-10-29): records/reference/product-display-unit.csv
    shows every archetype whose home product and unit match in the unit the product
    is generally discussed in (hydrogen per t, liquid biofuels per m3), wherever no
    service-row override applies, under `product-shown` where the name changes too
    (crude per m3 moved 1000 km). The home unit stays the CIMS basis;
  * `factor` is a number or a product of conversion-factor.csv ids and numbers,
    `a x b per c x d` (= a*b / (c*d)), so no outside factor is typed inline;
  * where an override changes the product (water evaporated -> black liquor solids)
    and is not per GJ, capacity is restated from the canonical value in the new
    product's unit from records/reference/capacity-unit.csv (Brad 2026-10-29).
"""
from __future__ import annotations

import re
from collections import defaultdict

from .common import REFERENCE, RECORDS, read
from . import capacity_unit

ORDER = [
    "heat-low-temp", "heat-medium-temp", "heat-high-temp", "on-site-power-generation",
    "hvac-heating", "hvac-cooling", "refrigeration", "lighting", "mobile-equipment",
    "co2-capture-concentrated", "co2-capture-diffuse", "direct-air-capture", "co2-transport-and-storage",
    "extraction-and-haulage", "ventilation-and-air-heating", "comminution", "separation-and-processing",
    "product-drying", "pelletizing",
    "drilling-and-well-services", "field-engines", "oil-and-gas-production", "gas-processing", "in-situ-bitumen", "oil-sands-mining-and-extraction",
    "bitumen-upgrading", "upgrader-hydrogen", "water-and-tailings-management", "oil-sands-tailings",
    "produced-water-handling", "compressor-drive-engine", "compressor-drive-turbine",
    "lng-liquefaction", "oil-pipelines", "crude-pipeline-pumping", "gas-pipelines", "venting", "flaring",
    "fugitive-emissions", "synthetic-fuels",
    "crude-distillation", "conversion-and-treating", "refinery-hydrogen",
    "centralized-hydrogen-production", "hydrogen-distribution",
    "cokemaking", "agglomeration", "ironmaking", "steel-oxygen-supply", "steelmaking", "casting", "reheating",
    "rolling", "annealing-and-coating", "foundry-melting", "metal-melting-and-heat-treatment",
    "clinker-production", "cement-grinding-and-blending", "lime-calcination",
    "glass-melting", "glass-forming-and-finishing", "drying-and-low-temp-calcining", "ceramic-firing",
    "petrochemical-feedstock", "olefins", "aromatics", "polymers", "chemical-derivatives", "ammonia", "methanol", "urea", "fertilizers", "nitric-and-adipic-acid",
    "electrochemical-products", "hydrogen-peroxide",
    "ethanol-production", "biodiesel-production", "renewable-diesel-and-saf", "renewable-natural-gas",
    "wood-preparation", "chemical-pulping", "mechanical-pulping", "recycled-pulping", "chemical-recovery",
    "lime-kiln", "paper-forming", "paper-drying", "effluent-treatment",
    "alumina-digestion", "alumina-calcination", "anode-production", "aluminum-smelting", "pyrometallurgy", "electric-smelting",
    "hydrometallurgy", "smelter-oxygen-supply", "melting-casting-and-fabrication",
    "thermal-processing", "baking-frying-and-roasting", "drying", "drying-curing-and-forming",
    "wood-drying-and-pressing", "forming-and-pressing", "solid-biofuel", "coal-mine-methane",
    "cement-binder", "chemical-distillation", "n2o-abatement",
]
TIER = {"low-temp": "heat-low-temp", "medium-temp": "heat-medium-temp", "high-temp": "heat-high-temp"}
BY_EQUIPMENT = {"machine-drive"}
FAMILY_NOTE = {"machine-drive": "Shown by equipment, not end use: most machine-drive archetypes "
                                "serve many end uses (pumps, fans, compressed air, materials "
                                "handling and processing, compressor drive). Drives that serve a "
                                "sector's own end use also appear in that sector's band."}
NO_CONNECTION = "(no connection)"


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _overrides(records):
    """(archetype, end use) -> the service row carrying a product override. An end use
    shows an archetype once, so all its rows under that end use must agree."""
    seen = {}
    for s in records.service:
        key = (s["archetype"], s.get("end-use", ""))
        sig = (s.get("product", ""), s.get("product-unit", ""), s.get("factor", ""))
        if seen.setdefault(key, sig) != sig:
            raise SystemExit(f"archetype-service.csv: {key[0]} has rows under end use {key[1]} with "
                             f"different product overrides; give them one override or split the end use")
    return {(s["archetype"], s.get("end-use", "")): s for s in records.service if s.get("product-unit")}


_CF = None


def _cf():
    global _CF
    if _CF is None:
        _CF = {r["factor-id"]: _num(r["factor"]) for r in read(REFERENCE / "conversion-factor.csv")}
    return _CF


def factor_value(expr):
    """An override factor: blank = 1, a number, or `a x b per c x d` of numbers and
    conversion-factor.csv ids."""
    expr = (expr or "").strip()
    if not expr or expr == "process-heat":
        return 1.0 if not expr else None
    if _num(expr) is not None:
        return _num(expr)
    parts = re.split(r"(?:^|\s)per\s", expr, maxsplit=1)
    num, den = (parts + [""])[:2]

    def prod(part):
        out = 1.0
        for t in [x.strip() for x in part.split(" x ") if x.strip()]:
            v = _num(t) if _num(t) is not None else _cf().get(t)
            if not v:
                raise SystemExit(f"override factor {expr!r}: unknown or zero term {t!r}")
            out *= v
        return out
    return prod(num) / prod(den)


def _display_units():
    """(home product, home unit) -> a display override (product-display-unit.csv)."""
    out = {}
    for r in read(REFERENCE / "product-display-unit.csv"):
        factor_value(r["factor"])      # fails the build on an unknown factor id
        out[(r["product"], r["home-unit"])] = {"product": r.get("product-shown") or r["product"], "product-unit": r["product-unit"],
                                               "factor": r["factor"], "note": r.get("note", "")}
    return out


def override_for(ov, disp, k, e, a):
    """The override an archetype takes under end use e: its service row's, else its product's."""
    return ov.get((k, e)) or disp.get((a.get("product", ""), a.get("product-unit", "")))


PER_YR = {"GJ": 31536.0, "t": 1000.0}   # home units per canonical capacity unit a year
_RULES = None


def _restate_capacity(out, r, o, f):
    global _RULES
    _RULES = _RULES or capacity_unit.load()
    hu, ou = r.get("product-unit", ""), o["product-unit"]
    row = {"archetype": "", "end-uses": "", "product": o["product"], "product-unit": ou,
           "product-energy-content": r.get("product-energy-content"), "product-density": r.get("product-density")}
    rule = capacity_unit.rule_for(_RULES, row)
    k = capacity_unit.factor_for(rule, row) if rule else None
    if k is None:
        raise SystemExit(f"capacity-unit.csv: no usable rule for {r.get('archetype')} shown as {o['product']} ({ou})")
    for c, v in r["_capacity-canonical"].items():
        out[c] = "" if v is None else float(f"{v * PER_YR.get(hu, 1.0) * f / PER_YR.get(ou, 1.0) * k:.5g}")
    out["capacity-unit"] = rule["capacity-unit"]


def _factor_fn(rows):
    """archetype -> GJ of process heat per home unit, or None."""
    by = {r.get("archetype"): r for r in rows}
    ph = {r["archetype"]: r for r in read(RECORDS / "reference" / "process-heat-efficiency.csv")}

    def f(a, depth=0):
        ref = ph.get(a, {})
        if ref.get("factor-from") and depth < 3:
            return f(ref["factor-from"], depth + 1)
        r = by.get(a, {})
        sec = _num(r.get("specific-energy-consumption"))
        eff = _num(r.get("efficiency")) or _num(ref.get("efficiency"))
        return sec * eff / 100 if sec and eff else None
    return f


def _apply(r, o, units, factor):
    """The row shown in the override's product: per-unit columns rescaled."""
    k = r.get("archetype", "")
    f = factor(k) if o.get("factor") == "process-heat" else factor_value(o.get("factor"))
    out = dict(r, product=o["product"], **{"product-unit": o["product-unit"]})
    if not f:
        out["product"] = f"{r.get('product', '')} (no process-heat factor: per {r.get('product-unit', '')})"
        out["product-unit"] = r.get("product-unit", "")
        return out
    if f != 1.0:
        for c, u in units.items():
            v = _num(r.get(c))
            if v is None or "product-unit" not in u or c.endswith("_n"):
                continue
            x = v / f if ("/product-unit" in u or "/(product-unit" in u) else v * f
            out[c] = float(f"{x:.6g}")    # 6 significant figures, as the CSVs
        canon = r.get("_capacity-canonical")
        if canon and o["product-unit"] == "GJ" and r.get("product-unit") in ("t", "m3", "GJ"):
            # one capacity unit per end use (Brad 2026-10-28): restated in MW of the
            # override's product (process heat) from the canonical capacity
            per_yr = 1000.0 if r.get("product-unit") == "t" else 31536.0 if r.get("product-unit") == "GJ" else 1.0
            for c, v in canon.items():
                if v is not None:
                    out[c] = float(f"{v * per_yr * f / 31536:.5g}")
            out["capacity-unit"] = "MW"
        note = (f"per {o['product-unit']} {o['product']}: per-unit values divided by {f:.4g} "
                f"{o['product-unit']} per {r.get('product-unit', '')}"
                + (" (SEC x efficiency)" if o.get("factor") == "process-heat" else ""))
        if "filled" in out:
            out["filled"] = "; ".join(x for x in (note, out.get("filled", "")) if x)
    if (o["product"] != r.get("product") and o["product-unit"] != "GJ"
            and r.get("_capacity-canonical") is not None):
        _restate_capacity(out, r, o, f)
    return out


def type_rank(a) -> int:
    """Order within an end use (Brad 2026-10-28): no-abatement first (2026-11-04), conventional, efficient
    variants, capture variants, low-carbon alternatives; anything else after,
    alphabetical within each group."""
    if (a.get("archetype") or "").startswith("no-abatement"):
        # the no-abatement option opens its end use, ahead of the other
        # conventional archetypes (Brad 2026-11-04)
        return -1
    if "_ccs-" in (a.get("archetype") or ""):
        return 2
    return {"conventional": 0, "efficient variant": 1, "alternative": 3,
            "prospective": 3}.get((a.get("type") or "").strip(), 4)


def _product(a) -> str:
    return f"{a.get('product', '')} ({a.get('product-unit', '')})"


def _context(records):
    ann = read(RECORDS / "hierarchy-annotation.csv")
    label = {r["annotation"]: r.get("label") or r["annotation"] for r in ann}
    asector = {r["annotation"]: r.get("sector", "") for r in ann}
    eus = defaultdict(list)
    for s in records.service:
        if s.get("end-use") and s["end-use"] not in eus[s["archetype"]]:
            eus[s["archetype"]].append(s["end-use"])
    return label, asector, eus


def _products_by_end_use(records, eus) -> dict[str, set]:
    """end use -> the products its archetypes make (heat tiers included), after overrides."""
    out = defaultdict(set)
    ov, disp = _overrides(records), _display_units()
    for a in records.archetypes:
        ids = list(eus.get(a["archetype"], []))
        if a.get("family") == "heat-supply":
            ids += [TIER[t.strip()] for t in (a.get("heat-output-tier") or "").split(";") if t.strip() in TIER]
        for e in ids:
            o = override_for(ov, disp, a["archetype"], e, a)
            out[e].add(_product({"product": o["product"], "product-unit": o["product-unit"]} if o else a))
    return out


def sub_label(e, lab, a, prods) -> str:
    """The sub-band label: the end use's, plus the product when the end use has several."""
    return f"{lab}: {_product(a)}" if len(prods.get(e, ())) > 1 else lab


def _home_of(e, f, asector, records):
    """The family an end use's sub-band sits in: the family named by the end use's
    sector, else (cross-sector end uses, or a sector with no family) the archetype's own."""
    s = asector.get(e, "")
    return s if s in records.families and s != "cross-sector" else f


def families(records) -> dict[str, str]:
    """archetype -> its families, home family first, then the families of the sectors
    that own the other end uses it serves (modeller 2026-10-09). Derived from
    records/archetype-service.csv, so it never needs tagging by hand; `family` stays
    the single home family that the pipeline's rules use."""
    label, asector, eus = _context(records)
    out = {}
    for a in records.archetypes:
        f = a["family"]
        fs = [f] + [_home_of(e, f, asector, records) for e in eus.get(a["archetype"], [])]
        out[a["archetype"]] = "; ".join(dict.fromkeys(fs))
    return out


def expand(rows: list[dict], records, units: dict | None = None,
           factor_rows: list[dict] | None = None) -> list[dict]:
    """One workbook row per (archetype, end-use sub-band), placed in the family whose
    sector owns the end use (modeller 2026-10-09), sorted by family (the rows' own
    order, then any family reached only through another family's archetypes),
    sub-band, then type (type_rank) and archetype. Adds `_family` (the band the row sits in)
    and `_sub` (sub-band label). All rows are shown alike (no repeat styling)."""
    label, asector, eus = _context(records)
    prods = _products_by_end_use(records, eus)
    ov, units = _overrides(records), units or {}
    disp = _display_units()
    factor = _factor_fn(factor_rows or rows)
    service_fams = {f for f, r in records.families.items() if r.get("kind") == "service"}

    def own(f, e):
        a = asector.get(e, "")
        return a == f or (f in service_fams and a == "cross-sector")
    rank = {e: i for i, e in enumerate(ORDER)}
    fam_pos = {}
    for r in rows:
        fam_pos.setdefault(r.get("family", ""), len(fam_pos))
    out = []
    for r in rows:
        f = r.get("family", "")
        k = r.get("archetype", "")
        if f in BY_EQUIPMENT:
            # by equipment in its own band; end uses owned by a sector family are
            # repeated there as for any archetype (modeller 2026-10-09)
            placed, seen = [(f, "", r.get("equipment", ""))], set()
            for e in eus.get(k, []):
                h = _home_of(e, f, asector, records)
                lab = sub_label(e, label.get(e, e), override_for(ov, disp, k, e, r) or r, prods)
                if h != f and (h, lab) not in seen:
                    seen.add((h, lab))
                    placed.append((h, e, lab))
        else:
            ids = list(eus.get(k, []))
            if f == "heat-supply":
                tiers = [TIER[t.strip()] for t in (r.get("heat-output-tier") or "").split(";") if t.strip() in TIER]
                ids = tiers + [e for e in ids if e not in TIER.values()] if tiers else ids
            seen, placed = set(), []
            for e in ids:
                h = _home_of(e, f, asector, records)
                lab = sub_label(e, label.get(e, e), override_for(ov, disp, k, e, r) or r, prods)
                if (h, lab) not in seen:
                    seen.add((h, lab))
                    placed.append((h, e, lab))
            placed = placed or [(f, "~", NO_CONNECTION)]
        for h, e, lab in placed:
            fam_pos.setdefault(h, len(fam_pos))
            o = override_for(ov, disp, k, e, r)
            rr = _apply(r, o, units, factor) if o else r
            out.append(dict(rr, _family=h, _sub=lab,
                            _key=(fam_pos[h], (0 if (e == "" or own(h, e)) else 1000 if e != "~" else 5000)
                                  + rank.get(e, 998), lab,
                                  type_rank(r), k)))
    # within a family, a sub-band label sits at the rank of its first end use
    first = {}
    for o in out:
        kk = (o["_key"][0], o["_sub"])
        first[kk] = min(first.get(kk, o["_key"][1]), o["_key"][1])
    out.sort(key=lambda o: (o["_key"][0], first[(o["_key"][0], o["_sub"])], o["_sub"]) + o["_key"][3:])
    for o in out:
        o.pop("_key", None)
    return out
