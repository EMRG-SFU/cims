"""Where each (parameter, target) belongs (modeller 2026-10-08, 2026-10-09).

  main        archetype-parameter.csv and the wide-archetype sheet
  carriers    archetype-parameter.csv and the wide-carriers-species sheet
              (specific energy by carrier, emissions by species; CO2-eq stays main)
  secondary   archetype-secondary-parameter.csv and wide-archetype-secondary-
              parameter.csv only: values by stream or part (a second product's
              capacity, component and stack lifetimes, efficiency by stream,
              operating ranges, load-range-min, degradation-rate)
  removed     archetype-parameter.csv only; dropped from every wide view
"""
from __future__ import annotations

MAIN_TARGETS = {("efficiency", "electricity"), ("efficiency", "heat"),
                ("cost-levelized", "calculated"), ("cost-levelized", "calculated-incl-energy"),
                ("emissions-combustion", "co2e"), ("emissions-process", "co2e"),
                ("emissions-total", "co2e")}
MAIN_ANY_TARGET = {"capture-rate", "share-to-capture"}
REMOVED = {(p, t) for p in ("heat-output-temp", "heat-process-temp")
           for t in ("exhaust", "flue-gas", "heat", "steam")} | {
    ("capex", "brownfield"), ("capex", "first-of-a-kind")}
REMOVED_ANY_TARGET = {"pressure-output", "abatement-cost"}
EMISSIONS = {"emissions-combustion", "emissions-process", "emissions-total", "emissions-captured",
             "emissions-biomass"}
# secondary with or without a target; efficiency-chp left here is a non-CHP
# archetype's (CHP archetypes carry it as their efficiency, pipeline/roll_in.py)
SECONDARY_PARAMS = {"load-range-min", "degradation-rate", "efficiency-chp"}


def placement(p: str, t: str) -> str:
    if (p, t) in REMOVED or (t and p in REMOVED_ANY_TARGET):
        return "removed"
    if p in SECONDARY_PARAMS:
        return "secondary"
    # a variant's own sourced values and the host-plant / capture add-on parts of a
    # host-with-capture variant (pipeline/fill.py, 2026-10-10)
    if t and (t.startswith("as-sourced") or t in ("host-plant", "ccs-add-on")):
        return "secondary"
    if not t or (p, t) in MAIN_TARGETS or p in MAIN_ANY_TARGET:
        return "main"
    if p in ("specific-energy-consumption", "feedstock-consumption") or p in EMISSIONS:
        return "carriers"
    return "secondary"
