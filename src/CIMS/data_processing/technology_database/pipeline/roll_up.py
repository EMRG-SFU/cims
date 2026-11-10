"""Targets that name the same quantity as the base parameter are pooled with it
(wide-view review, modeller 2026-10-08).

Runs after normalisation and before summarise, on the pooled parameter rows. A
row whose `target-norm` only restates what the base parameter already measures
is relabelled to the base (target-norm blank), so it pools with the base
examples instead of sitting in a column of its own:

  * capex `greenfield`: a greenfield capex is the capex (brownfield and
    first-of-a-kind stay apart).
  * specific-energy-consumption `total`: a source's all-carrier total is the
    total the base parameter holds (carriers.py reads the blank target as the
    total).
  * capacity, annual-output and cost-levelized where the target is the
    archetype's own product (cost-levelized `hydrogen` on an archetype whose
    product is hydrogen, capacity `ch4` on a methane-abatement archetype priced
    per t CH4, cost-levelized `co2-captured` on a capture unit). A target that is
    not the product (a CHP plant's electric capacity, an electrolyser's
    electricity input, a levelized cost net of a subsidy or including delivery)
    keeps its own column.

  * efficiency `heat` or `electricity` on an archetype with a single useful
    output (not CHP, not boiler_recovery, gasifier_black-liquor or the
    waste-to-energy incinerator): the stream efficiency is the overall
    efficiency (modeller 2026-10-09). pipeline/roll_in.py repeats the pooled value
    in the stream's column afterwards.

A row is relabelled only when its normalised unit is the unit the base already
has on that archetype (or the base has none), so two units never share a cell.
The target as reported stays in `target`; the note says what was done.
"""
from __future__ import annotations

from collections import defaultdict

from .roll_in import dual

ALWAYS = {("capex", "greenfield"), ("specific-energy-consumption", "total")}
PRODUCT_PARAMS = {"capacity", "annual-output", "cost-levelized"}
# targets that name an archetype's product, by the product name (records/archetype.csv)
PRODUCT_TARGETS = {
    "co2": {"co2", "co2-captured"},
    # product names written out (2026-10-23): the targets sources print keep their formulae
    "carbon dioxide": {"co2", "co2-captured"},
    "methane": {"ch4"},
    "chlorine": {"cl2"},
    "potash (k2o basis)": {"k2o"},
    "liquefied natural gas": {"lng"},
    "synthetic natural gas": {"sng"},
    "methylal": {"ome1"},
    "oxymethylene dimethyl ethers": {"ome3-5"},
    "renewable diesel and sustainable aviation fuel": {"renewable-diesel-and-saf"},
    "hvac heating": {"space-heating"},
    "hvac cooling": {"space-cooling"},
    "hydrogen delivered": {"hydrogen"},
    "liquid hydrogen": {"hydrogen"},
}
# per archetype, where the product name is broader than the target but the
# archetype makes only that one product
ARCHETYPE_TARGETS = {
    "reactor_ethanol-to-ethylene": {"ethylene"},   # ethanol dehydration yields ethylene only
}
NOTE = "target '{t}' rolled into the base parameter: {what} (wide-view decisions 2026-10-08/09)"


def _product_targets(a: dict) -> set[str]:
    name = (a.get("product") or "").strip().lower()
    out = {name.replace(" ", "-")} | PRODUCT_TARGETS.get(name, set())
    return out | ARCHETYPE_TARGETS.get(a.get("archetype", ""), set())


def apply(facts: list[dict], records) -> dict:
    arch = {a["archetype"]: a for a in records.archetypes}
    pooled = [r for r in facts if r.get("value-norm") or r.get("value-min-norm") or r.get("value-max-norm")]
    base_units = defaultdict(set)
    for r in pooled:
        if not r.get("excluded") and not r.get("target-norm"):
            base_units[(r["archetype"], r["parameter"])].add(r.get("unit-norm", ""))
    n = defaultdict(int)
    for r in pooled:
        t, p = r.get("target-norm", ""), r["parameter"]
        if not t or r.get("excluded"):
            continue
        if (p, t) in ALWAYS:
            what = "quantity (" + ("greenfield capex is capex" if p == "capex" else
                                   "an all-carrier total is the total") + ")"
        elif p in PRODUCT_PARAMS and t in _product_targets(arch.get(r["archetype"], {})):
            what = "it names the archetype's own product"
        elif p == "efficiency" and t in ("heat", "electricity") and not dual(arch.get(r["archetype"], {})):
            what = "single output stream (its stream efficiency is the overall efficiency)"
        else:
            continue
        have = base_units.get((r["archetype"], p))
        if have and r.get("unit-norm", "") not in have:
            continue
        r["target-norm"] = ""
        r["note"] = ((r.get("note") or "") + " | " if r.get("note") else "") + NOTE.format(t=t, what=what)
        n[f"{p} {t}"] += 1
    return dict(sorted(n.items()))
