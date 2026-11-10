"""Put every example's facts on its archetype's basis.

Each archetype declares one `product-unit` (t for a kiln, GJ for a boiler).
Every number is converted to a unit fixed by the parameter and that product
unit, or left out with the reason written on its row. Nothing is averaged here;
this only makes numbers comparable. The as-reported value stays beside the
converted one in the output, and in the extraction it came from.

Target units, where pu is the archetype's product-unit:

  capex, opex-fixed                 2025_CAD per (pu/yr of capacity)
  opex-variable, opex-total         2025_CAD per pu
  abatement-cost                    2025_CAD per t CO2-eq (per t CH4 where the target is ch4)
  specific-energy-consumption       GJ per pu
  emissions-* (species in target)   t per pu
  capacity                          MW when pu is GJ, kt/yr when pu is t
  shares, efficiencies, rates       %   (a COP is written as a % efficiency)
  lifetime, construction-time       years
  temperature, pressure             degC, bar

A source's generic `/unit` denominator means the service unit the source
reports for that step (UKERC: t of product, GJ, m3). It is read as that unit,
and as pu only when the source names none; a row per tonne on a GJ archetype is
therefore not pooled. A service unit that is not a physical product (UKERC's
"£k" of output) is never pooled. A row missing the basis parameter.csv requires for it (status review,
BASIS NOT RECORDED) is not pooled: an equipment cost and an installed cost
differ by up to a factor of two. It is converted where possible and marked
excluded until the basis is read back out of its source.
"""
from __future__ import annotations

import re
from collections import defaultdict
from statistics import median

from .common import REFERENCE, fnum, read, split_list
from . import bridge, convert, heating
from .currency import TARGET
from .whole_plant import NEEDS_HOST

PER_CAPACITY = {"capex", "opex-fixed", "decommissioning-cost"}
PER_OUTPUT = {"opex-variable", "opex-total", "cost-levelized"}
# parameters stated per unit of a product (or of product capacity): a basis factor
# (records/reference/example-conversion.csv) applies to these
PER_PRODUCT = {"specific-energy-consumption", "capacity", "annual-output"}
EXTENSIVE = {"capacity", "annual-output"}
PERCENT = {"efficiency", "efficiency-chp", "capacity-factor", "availability", "capture-rate",
           "share-to-capture",
           "emissions-reduction-potential", "load-range-min", "discount-rate",
           "cost-learning-rate", "capital-recovery", "degradation-rate"}
# A share of emissions avoided is one percentage whether the source counted CO2
# alone or CO2-equivalent, so the species is not a column of its own here.
# heat-waste-temp pools every leaving stream (exhaust, flue gas, stack, coolant) as one
# waste-heat temperature; the stream stays in the row (Brad 2026-11-09)
SPECIES_POOLED = {"emissions-reduction-potential", "heat-waste-temp", "heat-waste-temp-min",
                  "heat-waste-temp-max"}
GJ_PER_KW_YR = 8760 * 0.0036
OUTPUT_BASES = {"thermal-output", "electric-output", "product-output"}
# A derived cost written "<cost per t> / <GJ per t>" is per GJ of what the
# technology consumes, not of what it delivers (D13).
import re as _re
FUEL_INPUT_DIVISOR = _re.compile(r"^\s*[\d.eE+-]+\s*/\s*[\d.eE+-]+\s*$")
# An archetype carries absolute costs (design §2). A negative cost is a change
# against a baseline plant -- UKERC's measure costs are the cost of adding the
# measure, and a negative O&M is the saving it brings -- so it is not pooled.
NEGATIVE_COST = ("negative cost: a saving against the baseline plant, not the "
                 "archetype's own absolute cost")
SET_ASIDE = ("set aside: needs an efficiency factor. UKERC gives this cost per tonne of the "
             "host plant's product and only the fuel used per tonne; per GJ of output needs "
             "the efficiency (output = fuel x efficiency), which UKERC does not give")
FUEL_INPUT_NOTE = ("PER GJ OF FUEL INPUT: {parent} per tonne divided by the source's fuel use "
                   "per tonne")
UNIT_ALIAS = {"tonnes of mixed product": "t"}

# a tonne a source names by its product (`GJ/t-glass`) is the archetype's own
# tonne when the product is the archetype's (2026-10-13, sanity-check S108). A
# tonne of ANOTHER product (t-cement on a clinker kiln) is left alone, so it
# meets the product-basis factors or is excluded, never read 1:1.
PRODUCT_TONNE = {
    "glass": ("melt", "glass"), "melt": ("melt",), "clinker": ("clinker",),
    "cement": ("cement",), "brick": ("ceramic product", "brick"),
    "tile": ("ceramic product",), "steel": ("crude steel", "steel"),
    "pulp": ("pulp",), "paper": ("paper",), "lime": ("lime", "quicklime"),
    "methanol": ("methanol",),
    "product": None,                 # the archetype's product by definition
}
TONNE_Q = re.compile(r"^(?P<num>[^/]+)/(?P<k>k|M)?t-(?P<prod>[a-z]+)(?P<yr>-per-yr)?$")


def product_tonne(unit, a, p):
    m = TONNE_Q.match(unit)
    if not m or m.group("prod") not in PRODUCT_TONNE:
        return unit
    names = PRODUCT_TONNE[m.group("prod")]
    prod = (a.get("product") or "").lower().removeprefix("t ")
    if names is not None and not any(prod == n or prod.startswith(n) for n in names):
        return unit
    num = m.group("num")
    if p.startswith("emissions-") and re.fullmatch(r"(t|kg)-CO2(-eq)?", num):
        num = num.split("-", 1)[0]
    return f"{num}/{m.group('k') or ''}t{m.group('yr') or ''}"

# a note saying a cost per MWh is per MWh of HEAT (2026-10-02: the test used to be
# the literal "per MWh of heat", which excluded MWh-th, kWh th and levelized cost
# of heat)
HEAT_DENOMINATED = re.compile(r"per MWh of heat|MWh[- ]?th\b|kWh[- ]?th\b|MWh thermal|"
                              r"cost of (delivered )?heat|LCOH|per MWh of thermal|"
                              r"MWh heat|of thermal energy delivered", re.I)
# A cost with cost-basis `incremental` is the cost of adding something to an
# existing plant (a UK TIMES improvement or add-on, or an example listed in
# records/reference/incremental-cost.csv), so it is not a whole-plant cost.
INCREMENTAL_NOTE = ("set aside: needs a host-plant cost. The source prices this as the cost of "
                    "adding it to an existing plant, not as a whole plant")
# an emission factor per unit of FUEL in an energy unit other than GJ (lb/MMBtu,
# t/MMBtu, kg/MWh of fuel): bridged per GJ of output like a per-GJ factor
# (sanity-check 2026-10-23, S140/HPCH001: EIA and DOE factors had pooled as per GJ
# of electricity or heat, about half the true value on a combined cycle)
JUDGED_CAP = "judged-site-size:"
SITE_DEN = re.compile(r"/(station|plant|well|facility|platform|site)$")
# a gas carrier's heating value per cubic metre, for an energy use stated by volume
CARRIER_VOLUME = {"natural-gas": "cf-natural-gas-hhv-volume", "hydrogen": "cf-h2-hhv-volume"}
ENERGY_DEN = re.compile(r"/(MMBtu|mmbtu|MBtu|Btu|therm|MWh|kWh|GWh|MJ|TJ|PJ|J)$")  # J: Clearstone ng/J (S153)
FUEL_INPUT = re.compile(r"fuel[- ]input|of fuel\b|per (MMBtu|MWh|kWh|GJ|MJ|therm) (of )?(fuel|input)", re.I)
PHYSICAL_SERVICE = {"t", "GJ", "m3", "1000-m3", "m", "m3-km", "MWh"}
# a cost per unit of power rating, as a multiple of "per kW"
PER_POWER = {"W": 1e3, "kW": 1.0, "MW": 1e-3, "kW-yr": 1.0, "kW-per-yr": 1.0}
# a cost per unit of LUMEN rating (lighting, ruling 2026-10-14), as a multiple of
# "per klm"; one klm at nameplate delivers 8.76 Mlm-h a year
PER_LUMEN = {"lm": 1e-3, "klm": 1.0, "Mlm": 1e3, "klm-yr": 1.0, "klm-per-yr": 1.0}
LIGHT_UNIT = "Mlm-h"
MLMH_PER_KLM_YR = 8760 / 1000
# efficacy (lm/W) on a lighting archetype is put as electricity per Mlm-h:
# 1e6 lm-h / (x lm/W) = 1e6/x Wh = 3.6/x GJ
EFFICACY_UNITS = {"lm/W", "lumen/W", "lm-per-W"}
CASE = re.compile(r"^\s*(?:\[([^\]]+)\]|UKERC ([A-Z]+-[A-Z0-9]+):)")
HEAT_PUMP_SPLIT = {"heat-pump_industrial"}
MVR = re.compile(r"\bMVR\b|\bTVR\b|vapou?r[- ]?recompress|recompress(ion|or)|open[- ]cycle|"
                 r"thermo-?compress|steam compression", re.I)
MVR_NOTE = ("  SPECIES: vapour-recompression (open-cycle heat pump); its COP is pooled apart from "
            "the closed-cycle heat pumps (ruling 2026-10-05).")
REFURB = re.compile(r"refurbish|rebuild interval|membrane (life|replacement)|re-?lining interval", re.I)
# a lifetime in operating hours is a stack, lamp or membrane life, never the plant's
# (sanity-check pass 2026-10-04, defect 4): it takes its own species
HOURS_NOTE = ("  SPECIES: {sp} (a life in operating hours, not the plant's service life; "
              "rule 2026-10-05).")


def is_capture(a) -> bool:
    """A capture-plant archetype (the co2-capture family): it holds the add-on's cost."""
    return bool(a) and a.get("family") == "co2-capture"


def abatement_options(records) -> set:
    """Archetypes that compete at a CIMS node with a zero-cost no-abatement option
    (a `no-abatement_*` archetype on the same branch of records/archetype-service.csv).
    Against a baseline that costs nothing, an option's incremental cost IS its whole
    cost, so it needs no host (ruling 2026-10-27, S148)."""
    nab = {s["branch"] for s in records.service if s["archetype"].startswith("no-abatement_")}
    return {s["archetype"] for s in records.service
            if s["branch"] in nab and not s["archetype"].startswith("no-abatement_")}


ABATEMENT_NOTE = ("  COST BASIS: incremental over a zero-cost no-abatement option, so it is this "
                  "option's whole cost (ruling 2026-10-27, S148).")


def is_ccs_host(a) -> bool:
    """A host plant with capture (a _ccs- modifier outside the co2-capture family).
    In CIMS it differs from its host only by a service request to the capture
    archetype, so it holds the host plant's cost and none of the capture plant's."""
    return bool(a) and "_ccs-" in a["archetype"] and a.get("family") != "co2-capture"


# a capture-rate that is not a share of CO2 (landfill gas collection): left as
# it is and listed for review in reviews/sanity-checks/sanity-check-list.csv
CAPTURE_RATE_NOT_CO2 = {"process-improvement_landfill-gas-recovery"}
SHARE_NOTE = ("  PARAMETER: the source's capture rate on this host, recorded as share-to-capture "
              "(the share of the host's CO2 sent to capture; taken equal to the plant-level "
              "rate the source gives, modeller 2026-10-02).")
# Costs that, on a host-with-capture archetype, would carry capture cost
CCS_HOST_COSTS = {"capex", "opex-fixed", "opex-variable", "opex-total", "cost-levelized"}
CCS_HOST_NOTE = ("CCS add-on cost: this archetype holds only the host plant's cost (the same as "
                 "without capture unless a source shows a significant difference); the capture "
                 "plant's cost belongs to the co2-capture archetype it requests (modeller "
                 "2026-10-02). Listed in output/ccs-addon-cost.csv to move to a capture archetype")


def _mixed_carriers(a) -> bool:
    """An archetype whose listed carriers include one with no heating value
    (electricity, heat, steam) or fuels with different HHV/LHV ratios (S144)."""
    listed = [c.strip().lower() for c in split_list(a.get("carriers", "")) if c.strip()]
    ratios = {heating.CARRIER_RATIO.get(c, "?") for c in listed}
    return bool(listed) and ("" in ratios or len(ratios) > 1)


def _gj_per_den(r):
    """(value in GJ, denominator) of an energy-use row as printed, or None: the
    numerator put in GJ so carriers in kWh/t and a total in GJ/t compare (S144).
    A range-only row takes its midpoint."""
    u = (r.get("unit") or "")
    if "/" not in u:
        return None
    num, den = u.split("/", 1)
    at = convert.atom(num)
    v = fnum(r.get("value"))
    if v is None:
        lo, hi = fnum(r.get("value-min")), fnum(r.get("value-max"))
        v = (lo + hi) / 2 if lo is not None and hi is not None else None
    if not at or at[0] != "energy" or v is None:
        return None
    return v * at[1], den


def _fuel_share(r, carriers, a):
    """The fuel share of an all-carrier total: from the example's own carriers in
    the same unit, else 1 on a conversion archetype or one whose first listed
    carrier is a fuel, else 0 (S144)."""
    from . import feedstock
    cs = {t: v for t, v in (carriers or {}).items()
          if t in heating.CARRIER_RATIO and t not in ("total", "component") and v and v > 0}
    if cs:
        fuel = sum(v for t, v in cs.items() if heating.CARRIER_RATIO[t])
        return fuel / sum(cs.values())
    listed = [c.strip().lower() for c in split_list(a.get("carriers", "")) if c.strip()]
    if feedstock.row_for(a["archetype"]) or (listed and heating.CARRIER_RATIO.get(listed[0], "?") not in ("",)):
        return 1.0
    return 0.0


def case_of(r) -> str:
    """The case a row belongs to within its example: a leading [case] in the note,
    or UKERC's row code."""
    m = CASE.match(r.get("note") or "")
    return (m.group(1) or m.group(2)) if m else ""


def service_unit(example: dict | None, pu: str) -> str:
    """What a source's generic `unit` denominator stands for in this example."""
    su = ((example or {}).get("service-unit-reported") or "").strip()
    su = UNIT_ALIAS.get(su, su)
    return su or pu


def target_unit(param: str, pu: str, default: str) -> str:
    if param in PER_CAPACITY:
        return f"{TARGET}/{pu}-per-yr"
    if param in PER_OUTPUT:
        return f"{TARGET}/{pu}"
    if param == "abatement-cost":
        return f"{TARGET}/t-CO2-eq"
    if param in ("specific-energy-consumption", "feedstock-consumption"):
        return f"GJ/{pu}"
    if param.startswith("emissions-") and param != "emissions-reduction-potential":
        return f"t/{pu}"
    if param == "capacity":
        return "MW" if pu == "GJ" else f"k{pu}/yr" if pu == "t" else f"{pu}/yr"
    if param == "annual-output":
        return pu
    if param in PERCENT:
        return "%"
    if param in ("lifetime", "construction-time"):
        return "years"
    if param.startswith(("heat-output-temp", "heat-process-temp", "heat-waste-temp")):
        return "degC"
    if param == "pressure-output":
        return "bar"
    return default


def supplies_heat(a) -> bool:
    """An archetype meant to supply heat: heat supply, a boiler, or heat or steam among
    its products (a recovery boiler raises steam as well as recovering chemicals)."""
    prods = {a.get("product", "")} | set(split_list(a.get("secondary-products", "")))
    return (a.get("family") == "heat-supply" or a["archetype"].split("_")[0] == "boiler"
            or bool(prods & {"heat", "steam"}))


class Units:
    def __init__(self, cur_factors):
        self.table = {}
        for r in read(REFERENCE / "unit-conversion.csv"):
            f = fnum(r["factor"])
            if f is not None:
                self.table[(r["from-unit"], r["to-unit"])] = (f, fnum(r["offset"]) or 0.0)
        self.cur = cur_factors
        self.factors = convert.Factors()
        self.streams = heating.Streams(self.factors)

    def physical(self, unit, target):
        """(factor, offset) from unit to target, or None."""
        if unit == target:
            return (1.0, 0.0)
        if (unit, target) in self.table:
            return self.table[(unit, target)]
        if "/" in unit and "/" in target:
            (n, d), (tn, td) = unit.split("/", 1), target.split("/", 1)
            if d == td and (n, tn) in self.table:
                return self.table[(n, tn)]
        return None


def _num(r):
    return fnum(r.get("value")), fnum(r.get("value-min")), fnum(r.get("value-max"))


CARRIER_ALIAS = {"biomass-solid": "biomass"}
# steam-heated units named outright whose sources give the same duty as heat and steam
# (node alignment PP08, PP14, Brad 2026-11-05)
HEAT_AS_STEAM = {"digester_semi-chemical", "dryer_market-pulp"}


def normalise(facts, examples, records, units) -> None:
    """Adds value-norm, value-min-norm, value-max-norm, unit-norm, target-norm,
    duplicate-of and excluded to each fact row, in place."""
    # one solid-biomass carrier (Brad 2026-11-05): CIMS has one solid biomass fuel,
    # and records/vocabulary/commodity.csv already gives biomass-solid the energy
    # carrier biomass; sources and the MIDDEN map that say "solid biomass" pool with it
    for r in facts:
        t = (r.get("target") or "").strip().lower()
        if t in CARRIER_ALIAS:
            r["target"] = CARRIER_ALIAS[t]
            r["note"] = (r.get("note") or "") + f"  CARRIER: {t} pooled as {CARRIER_ALIAS[t]} (2026-11-05)."
    # per example, what turns a rating or an absolute sum into a rate: the
    # SOURCE'S OWN factors only, and only on an output basis
    scale = defaultdict(dict)
    # a judged size for an example that prices a whole site (records/reference/
    # example-capacity.csv; S148, 2026-10-27): its absolute and per-year costs go
    # per unit of that size, graded as an assumption
    judged_cap = {x["example-id"]: x for x in read(REFERENCE / "example-capacity.csv")}
    for eid, x in judged_cap.items():
        scale[eid]["caps"] = [{"record-id": JUDGED_CAP + eid, "value": x["capacity"],
                               "unit": x["unit"], "capacity-basis": "product-output",
                               "example-id": eid, "note": ""}]
    for r in facts:
        v = fnum(r.get("value"))
        if v is None:
            continue
        if r["parameter"] == "annual-output" and r.get("unit") == "GJ":
            scale[r["example-id"]].setdefault("annual-output", v)
        elif r["parameter"] == "capacity-factor":
            scale[r["example-id"]].setdefault("cf", v / 100 if v > 1.5 else v)
        elif (r["parameter"] == "capacity" and r.get("unit") in ("MW", "kW")
              and r.get("capacity-basis") in OUTPUT_BASES):
            mw = v / 1000 if r["unit"] == "kW" else v
            scale[r["example-id"]].setdefault("mw-out", (mw, r["record-id"]))
        if r["parameter"] == "capacity" and r["example-id"] not in judged_cap:
            scale[r["example-id"]].setdefault("caps", []).append(r)
        # the same source's heat-to-power ratio, or the efficiencies it follows from
        tg = (r.get("target") or "").strip().lower()
        if r["parameter"] == "heat-to-power-ratio":
            scale[r["example-id"]].setdefault("hpr", (v, r["record-id"]))
        elif r["parameter"] == "efficiency" and tg == "electricity":
            scale[r["example-id"]].setdefault("eff-e", (v, r["record-id"]))
        elif r["parameter"] == "efficiency" and tg in ("heat", "steam", "hot-water", "district-heat"):
            scale[r["example-id"]].setdefault("eff-h", (v, r["record-id"]))
        elif r["parameter"] == "efficiency-chp":
            scale[r["example-id"]].setdefault("eff-chp", (v, r["record-id"]))
    # capture split (modeller 2026-10-02): the host records the share of its CO2
    # sent to capture, the capture plant its own capture-rate; their product is
    # the plant-level rate, not recorded. A capture-rate a source gives on any
    # other archetype is plant-level, so it is recorded as the host's share.
    for r in facts:
        a = records.by_key.get(r.get("archetype", ""))
        if (r["parameter"] == "capture-rate" and not (r.get("target") or "").strip()
                and (not a or a["archetype"] not in CAPTURE_RATE_NOT_CO2)):
            r["target"] = "co2"     # a capture rate with no species named is of CO2
        if (r["parameter"] == "capture-rate" and a and not is_capture(a)
                and a["archetype"] not in CAPTURE_RATE_NOT_CO2):
            r["parameter"] = "share-to-capture"
            if SHARE_NOTE.strip() not in (r.get("note") or ""):
                r["note"] = (r.get("note") or "") + SHARE_NOTE
        if r["parameter"] == "share-to-capture" and not (r.get("target") or "").strip():
            r["target"] = "co2"     # a share sent to capture is of CO2 (one row per host)
    # UKERC's 'Specific emissions' column is the step's total direct emissions
    # (fuel plus process), not combustion alone (ruling 2026-10-05; the sibling
    # column 'Direct fuel related emissions' is the combustion value)
    for r in facts:
        if (r["parameter"] == "emissions-combustion"
                and "UKERC column 'Specific emissions'" in (r.get("note") or "")):
            r["parameter"] = "emissions-total"
            r["note"] = (r.get("note") or "") + ("  PARAMETER: UKERC's 'Specific emissions' is the "
                                                 "total, loaded as emissions-total (2026-10-05).")
    # heat pumps: a vapour-recompression (open-cycle, MVR/TVR) example's COP is its
    # own species, so the closed-cycle COP 2.5-4.5 is not pooled with COP 10-30
    # (ruling 2026-10-05, sanity-check pass S09/S84)
    for r in facts:
        if (r.get("archetype") in HEAT_PUMP_SPLIT and r["parameter"] == "efficiency"
                and not (r.get("target") or "").strip()):
            txt = " ".join([r.get("example-id") or "", r.get("source-technology") or "",
                            r.get("note") or ""])
            if MVR.search(txt):
                r["target"] = "vapour-recompression"
                r["note"] = (r.get("note") or "") + MVR_NOTE
    # a lifetime the source gives as a refurbishment, rebuild or membrane interval is
    # a component's life, not the plant's (sanity-check pass R412, 2026-10-05)
    for r in facts:
        if (r["parameter"] == "lifetime" and not (r.get("target") or "").strip()
                and REFURB.search(r.get("note") or "")):
            r["target"] = "component"
            r["note"] = (r.get("note") or "") + ("  SPECIES: component (a refurbishment or "
                                                 "replacement interval, not the plant's life; 2026-10-05).")
    by_id = {r["record-id"]: r for r in facts}
    # an example, or one case of it (`case`: a leading [case] in the note, or
    # UKERC's row code), whose costs are the cost of adding something to a plant
    inc_rows = read(REFERENCE / "incremental-cost.csv")
    incremental = {(r["example-id"], (r.get("case") or "").strip()) for r in inc_rows}
    unknown = {e for e, _ in incremental} - set(examples)
    if unknown:
        raise SystemExit(f"incremental-cost.csv names unknown examples: {unknown}")
    for r in facts:
        if (((r["example-id"], "") in incremental or (r["example-id"], case_of(r)) in incremental)
                and r["parameter"] in PER_CAPACITY | PER_OUTPUT):
            r["cost-basis"] = "incremental"
    # an equipment or installed cost of an add-on is incremental too: on an
    # efficiency variant where the same source prices the measure as an add-on, its
    # equipment-only or installed rows are the same add-on, not a whole plant
    # (sanity-check 2026-10-23, S139/IS007: coke dry quenching had pooled as the
    # whole coke oven)
    inc_src = {(r.get("archetype"), r.get("source-id")) for r in facts
               if r.get("cost-basis") == "incremental" and r["parameter"] in PER_CAPACITY | PER_OUTPUT}
    for r in facts:
        a_ = records.by_key.get(r.get("archetype", "")) or {}
        if not any(m.startswith("efficiency") for m in split_list(a_.get("modifiers", ""))):
            continue        # only on efficiency variants, whose costs are a measure's
        if (r["parameter"] in PER_CAPACITY | PER_OUTPUT
                and (r.get("cost-basis") or "") in ("equipment", "installed")
                and ((r.get("archetype"), r.get("source-id")) in inc_src
                     or "incremental add-on" in (r.get("note") or "")
                     or "of an incremental" in (r.get("note") or ""))):
            r["cost-basis"] = "incremental"
            r["note"] = (r.get("note") or "") + ("  COST BASIS read as incremental: the same source "
                                                 "prices this measure as an add-on (S139, 2026-10-24).")
    bridged = {r["example-id"]: r for r in read(REFERENCE / "efficiency-bridge.csv")}
    # emission rows a source already states per unit of OUTPUT (per GJ of steam, heat or
    # hydrogen): not bridged (records/reference/emission-basis.csv, ruling 2026-10-21; a
    # blank record-id covers every emission row of the source)
    out_basis = read(REFERENCE / "emission-basis.csv")
    out_ids = {x["record-id"] for x in out_basis if x["basis"] == "output" and x["record-id"]}
    out_srcs = {x["source-id"] for x in out_basis if x["basis"] == "output" and not x["record-id"]}
    chp_levelized_total = {r["record-id"] for r in read(REFERENCE / "chp-levelized-total.csv")}

    def heat_per_power(sc):
        """kW of heat per kW of electricity for a CHP example, from the same
        source: its heat-to-power ratio, else its heat and electric efficiencies,
        else its total less its electric efficiency. (ratio, record-ids) or None."""
        if sc.get("hpr") and sc["hpr"][0] > 0:
            return sc["hpr"][0], [sc["hpr"][1]]
        e = sc.get("eff-e")
        if not e or e[0] <= 0:
            return None
        if sc.get("eff-h") and sc["eff-h"][0] > 0:
            return sc["eff-h"][0] / e[0], [sc["eff-h"][1], e[1]]
        if sc.get("eff-chp") and sc["eff-chp"][0] > e[0]:
            return (sc["eff-chp"][0] - e[0]) / e[0], [sc["eff-chp"][1], e[1]]
        return None

    # each heat-supply archetype's pooled heat-to-power ratio: the median of its
    # examples' own, used where an example prices per kW electric without one
    # (class internal, like the pooled efficiency in the bridge)
    _hp = defaultdict(list)
    for eid, sc in scale.items():
        h = heat_per_power(sc)
        arch = (examples.get(eid) or {}).get("archetype", "")
        if h and arch:
            _hp[arch].append(h[0])
    pooled_hp = {k: sorted(v)[len(v) // 2] if len(v) % 2 else
                 (sorted(v)[len(v) // 2 - 1] + sorted(v)[len(v) // 2]) / 2
                 for k, v in _hp.items()}

    def hp_for(sc, a):
        """(ratio, tag) from the example's own figures, else the archetype's pooled one."""
        h = heat_per_power(sc)
        if h:
            return h[0], "hpr:same-example"
        k = (a or {}).get("archetype", "")
        if k in pooled_hp:
            return pooled_hp[k], f"hpr:{k}"
        return None

    def heat_product(a, pu):
        """An archetype whose product is heat (a CHP unit's heat), not electricity."""
        return (pu == "GJ" and (a or {}).get("family") == "heat-supply"
                and "electric" not in (a or {}).get("product", "").lower())

    def own_capacity(r, sc, pu, flist, alias):
        """The same source's capacity for this case as (pu per year, record-id,
        [factor ids], capacity basis), or None. The capacity row must be the only
        one for the case; its basis is its own, else the cost row's."""
        caps = sc.get("caps", [])
        c = case_of(r)
        if c and not any(x["record-id"].startswith(JUDGED_CAP) for x in caps):
            caps = [x for x in caps if case_of(x) == c]
        caps = [x for x in caps if fnum(x.get("value")) is not None]
        if len(caps) != 1:
            return None
        x = caps[0]
        basis = x.get("capacity-basis") or r.get("capacity-basis") or ""
        if basis not in OUTPUT_BASES | heating.INPUT_BASES:
            return None
        tcap = target_unit("capacity", pu, "")
        cu = (x.get("unit") or "").strip()
        route = units.physical(cu, tcap)
        ids = []
        if route is None:
            ext = convert.extensive(cu, tcap, flist, alias)
            if not ext:
                return None
            route, ids = (ext[0], 0.0), ext[1]
        v = fnum(x["value"]) * route[0]
        per_yr = v * GJ_PER_KW_YR * 1000 if tcap == "MW" else v * 1000 if tcap.startswith("k") else v
        # nameplate: the capacity factor is not applied (modeller 2026-10-10)
        return per_yr, x["record-id"], ids, basis, cu

    # each example's own carrier figures by raw unit, for the fuel share of an
    # all-carrier total (S144)
    ex_carriers = defaultdict(dict)
    arch_carriers = defaultdict(dict)
    for r in facts:
        t = (r.get("target") or "").strip().lower()
        g = _gj_per_den(r)
        if r["parameter"] == "specific-energy-consumption" and t and g:
            ex_carriers[(r["example-id"], g[1])][t] = g[0]
            arch_carriers[(r.get("archetype", ""), g[1])].setdefault(t, []).append(g[0])
    for r in facts:
        r.update({"value-norm": "", "value-min-norm": "", "value-max-norm": "",
                  "unit-norm": "", "excluded": "", "conversion": "",
                  "target-norm": r.get("target", ""), "heating-value": "",
                  "conversion-class": "", "conversion-factors": ""})
        cls, bridge_kind, slip = [], "", None
        if r["parameter"] in SPECIES_POOLED:
            r["target-norm"] = ""
        parent = by_id.get(r.get("derived-from", ""))
        r["duplicate-of"] = (parent["record-id"] if parent and r.get("value-type") == "derived"
                             and parent["parameter"] == r["parameter"]
                             and not r.get("derived-kind") else "")
        v, lo, hi = _num(r)
        if v is None and lo is None and hi is None:
            r["excluded"] = "no number"
            continue
        a = records.by_key.get(r.get("archetype", ""))
        if not a:
            r["excluded"] = "example not mapped to an archetype"
            continue
        pu = a["product-unit"]
        p = r["parameter"]
        if p.startswith("heat-output-temp") and not supplies_heat(a):
            # an output temperature counts only on an archetype meant to supply heat
            # (Brad 2026-11-09): not a turbine's exhaust or a heating surface
            r["excluded"] = ("heat-output-temp on an archetype not meant to supply heat "
                             "(not heat supply, not a boiler, no heat or steam product)")
            continue
        if (p == "specific-energy-consumption" and (r.get("target") or "").strip().lower() == "heat"
                and ("steam-heated" in a["archetype"] or a["archetype"] in HEAT_AS_STEAM)
                and "steam" in split_list(a.get("carriers", ""))
                and "heat" not in split_list(a.get("carriers", ""))):
            # heat a source states for a steam-heated unit is its steam (S157/MCF019,
            # 2026-10-27): the same duty, under the carrier the archetype lists
            r["target-norm"] = "steam"
            r["note"] = (r.get("note") or "") + ("  CARRIER: heat on a steam-heated unit, "
                                                 "pooled as its steam (S157, 2026-10-27).")
        pdef = records.parameters.get(p, {})
        tgt = target_unit(p, pu, pdef.get("unit-default", "") or r.get("unit", ""))
        if p == "abatement-cost" and (r.get("target") or "").strip().lower() == "ch4":
            # a cost per unit of methane abated stays per t CH4: no GWP is applied
            # (ruling 2026-10-05; before, it was labelled per t CO2-eq unconverted)
            tgt = f"{TARGET}/t-CH4"
        unit = (r.get("unit") or "").strip()
        efficacy = None
        if pu == LIGHT_UNIT and p == "efficiency" and unit in EFFICACY_UNITS:
            # luminous efficacy: electricity per unit of light delivered (an
            # inverse, so the low and high bounds swap)
            efficacy = unit
            if any(x is not None and x <= 0 for x in (v, lo, hi)):
                r["excluded"] = "efficacy of zero or less"
                continue
            v, lo, hi = (None if v is None else 3.6 / v, None if hi is None else 3.6 / hi,
                         None if lo is None else 3.6 / lo)
            p = r["parameter"] = "specific-energy-consumption"
            r["target"] = r["target-norm"] = "electricity"
            unit = f"GJ/{LIGHT_UNIT}"
            tgt = target_unit(p, pu, unit)
        exrow, flist = units.factors.for_row(a["archetype"], r["example-id"])
        exf = [f for f in flist if exrow and f["factor-id"] in split_list(exrow.get("factor-id", ""))]
        alias = dict(x.split("=", 1) for x in split_list((exrow or {}).get("unit-alias", "")) if "=" in x)
        per_product = (p in PER_PRODUCT or p in PER_CAPACITY or p in PER_OUTPUT
                       or (p.startswith("emissions-") and p != "emissions-reduction-potential"))
        if exrow and exrow.get("factor-id") == "set-aside":
            r["excluded"] = "example set aside by the modeller: " + (exrow.get("note") or "")
            continue
        if exrow and exrow.get("factor-id") == "none" and per_product:
            r["excluded"] = ("reported per a product the archetype does not make, and no factor "
                             "links the two: " + (exrow.get("note") or ""))
            continue
        chp_total = (p == "cost-levelized" and r["record-id"] in chp_levelized_total
                     and heat_product(a, pu))
        if (p == "cost-levelized" and a.get("family") == "heat-supply"
                and re.search(r"/(k|M)Wh\b", unit) and not chp_total
                and not HEAT_DENOMINATED.search(r.get("note") or "")):
            r["excluded"] = ("a cost per unit of electricity (or of unstated output) on an archetype "
                             "whose product is heat: a CHP plant's cost per MWh of electricity is not "
                             "a cost per GJ of heat")
            continue
        bf = convert.basis_factor(exf) if per_product else None
        species_ok = a.get("product", "").lower().removeprefix("t ").startswith(("co2", "carbon dioxide")) or "CO2" in tgt
        used = []
        if r.get("assume"):
            # a derived row resting on a stated assumption (assembly.py `assumption`
            # column, e.g. avoided CO2 taken for captured CO2): graded `assumption`
            used.append("assume:" + r["assume"])
        if (p.startswith("emissions-") and pu == "GJ"
                and (unit.endswith("/GJ") or (ENERGY_DEN.search(unit) and FUEL_INPUT.search(r.get("note") or "")))
                and r["record-id"] not in out_ids and r.get("source-id") not in out_srcs):
            # an emission factor per GJ of FUEL; the archetype's basis is per GJ
            # of OUTPUT, and the two differ by the efficiency (D13): the
            # efficiency bridge (pipeline/bridge.py) divides by it
            bridge_kind = "per-input"
        for k, s in UNIT_ALIAS.items():
            unit = unit.replace(k, s)
        unit = product_tonne(unit, a, p)
        if ((a.get("product") or "").lower() == "methane" and "/" in unit
                and p.startswith(("capex", "opex", "cost-"))):
            # a methane-abatement option is measured per t of methane (abated):
            # a cost per t-CH4 (IEA, NCCP) is the archetype's own tonne, and a
            # cost per yr per (t CH4 per yr) is a cost per t-per-yr (S148, 2026-10-27)
            num_, den_ = unit.split("/", 1)
            den_ = den_.replace("yr-per-t-CH4-per-yr", "t-per-yr").replace("t-CH4", "t")
            unit = f"{num_}/{den_}"
        if r["example-id"] in judged_cap and p.startswith(("capex", "opex")):
            # a cost per site (or per year per site) of an example with a judged site size
            unit = SITE_DEN.sub("", unit)
            unit = re.sub(r"-per-yr$", "/yr", unit) if "/" not in unit else unit
        # steam reported as a MASS (kg or t of steam per unit of product): put in GJ
        # with the steam enthalpy in conversion-factor.csv (cf-steam-enthalpy,
        # 2026-10-02) before any other conversion
        steam_pre = None
        if p == "specific-energy-consumption" and (r.get("target") or "") == "steam" \
                and "/" in unit:
            num_, den_ = unit.split("/", 1)
            at_ = convert.atom(num_)
            sf_ = units.factors.rows.get("cf-steam-enthalpy")
            if at_ and at_[0] == "mass" and sf_:
                steam_pre = at_[1] * fnum(sf_["factor"])
                unit = "GJ/" + den_
        host = bridged.get(r["example-id"], {})
        if "/unit" in unit:
            su = service_unit(examples.get(r["example-id"]), pu)
            if su not in PHYSICAL_SERVICE:
                r["excluded"] = (f"per unit of the source's service ({su}), which is not a "
                                 "physical product")
                continue
            unit = unit.replace("/unit", f"/{su}")
            if (host.get("bridge") == "yes" and pu == "GJ" and su == "t"
                    and p != "specific-energy-consumption"):
                # per tonne of the host plant's product, approved for the bridge:
                # converted per tonne here, per GJ of output in bridge.py
                tgt, bridge_kind = target_unit(p, "t", tgt), "host-tonne"
        if p in ("capital-recovery", "degradation-rate") and unit == "%/yr":
            unit = "%"          # a rate per year by definition (added 2026-10-02)
        if not unit and tgt == "dimensionless":
            unit = tgt              # a ratio a source prints without a unit
        factor, offset, note = None, 0.0, ""
        hv_unit = ""
        m = convert.MONEY.match(unit)
        if m:
            cur = f"{m.group(1)}_{m.group(2)}"
            if cur not in units.cur:
                r["excluded"] = f"no currency factor for {cur}"
                continue
            f = units.cur[cur] * convert.MULT[m.group(3)]
            d = unit.split("/", 1)[1] if "/" in unit else ""
            td = tgt.split("/", 1)[1] if "/" in tgt else ""
            sc = scale.get(r["example-id"], {})
            if d == td:
                factor = f
            elif p in PER_CAPACITY and d == pu:
                factor, note = f, f"per {pu} read as per {pu} of annual capacity"
            elif p in PER_OUTPUT and d == f"{pu}-per-yr":
                factor, note = f, f"annual cost per {pu}/yr is a cost per {pu}"
            elif p in PER_CAPACITY and d in PER_POWER:
                # per unit of NAMEPLATE annual capacity, whatever capacity factor the
                # source runs at (modeller 2026-10-10: the levelized cost divides by
                # the capacity factor once; before, a stated factor was applied here)
                cf = sc.get("cf")
                factor = f * PER_POWER[d] / GJ_PER_KW_YR
                note = (f"per {d} at nameplate (the source's {cf:.0%} capacity factor is not applied)"
                        if cf else f"per {d} at nameplate")
                if r.get("capacity-basis") == "electric-output" and heat_product(a, pu):
                    # a CHP cost per kW of ELECTRICITY on an archetype whose product
                    # is the heat: per kW of heat with the same source's heat-to-power
                    # ratio (modeller 2026-10-01)
                    hp = hp_for(sc, a)
                    if not hp:
                        r["excluded"] = (f"cost per {d} of electric output on a heat-product "
                                         "archetype; no heat-to-power ratio from the source or "
                                         "the archetype's examples")
                        continue
                    factor /= hp[0]
                    note += f"; per {d} electric -> per {d} heat at heat-to-power ratio {hp[0]:.3g}"
                    cls.append("same-source" if hp[1] == "hpr:same-example" else "internal")
                    used.append(hp[1])
                if pu != "GJ" and r.get("capacity-basis") in heating.INPUT_BASES:
                    # per kW of an INPUT (hydrogen fed to an ammonia loop): the
                    # product's heating value does not apply. Only the same
                    # source's input per unit of product could convert it, and
                    # only if it counts the same stream (design s5, 2026-09-30).
                    r["excluded"] = (f"cost per {d} of input capacity cannot be put per {td}: "
                                     "the product's heating value does not apply to an input "
                                     "rating, and the source gives no input of this stream per "
                                     f"{pu} of product")
                    continue
                if pu != "GJ":
                    # a power rating in the product's energy (kW of hydrogen) put
                    # per unit of product with the product's heating value
                    q = convert.qty_ratio(pu, "GJ", flist, species_ok, alias)
                    if not q:
                        r["excluded"] = (f"cost per {d} cannot be put per {td}: no factor gives "
                                         f"the energy in a {pu} of this archetype's product")
                        continue
                    factor *= q[0]
                    used += q[1]
                    note += f"; per GJ/yr -> per {pu}/yr"
            elif p in PER_CAPACITY and d in PER_LUMEN and pu == LIGHT_UNIT:
                # per unit of lumen rating at nameplate (all hours of the year)
                factor = f / (PER_LUMEN[d] * MLMH_PER_KLM_YR)
                note = f"per {d} at nameplate (8.76 Mlm-h a year per klm)"
            elif p in PER_OUTPUT and d == "kWh" and pu == "GJ":
                factor, note = f / 0.0036, "per kWh -> per GJ"
            elif p in PER_CAPACITY and not d and pu == "GJ" and sc.get("annual-output"):
                # the annual output is what the plant makes; over its capacity factor it
                # is the nameplate annual capacity (modeller 2026-10-10)
                cf = sc.get("cf")
                factor = f * (cf or 1.0) / sc["annual-output"]
                note = ("absolute sum over the same source's annual output, put at nameplate "
                        f"with its {cf:.0%} capacity factor" if cf else
                        "absolute sum over the annual output the same source states (no capacity "
                        "factor given: read as nameplate)")
                cls.append("same-source")
            elif p in PER_CAPACITY and not d and pu == "GJ" and sc.get("mw-out"):
                mw, rid = sc["mw-out"]
                cf = sc.get("cf")
                if (by_id.get(rid, {}).get("capacity-basis") == "electric-output"
                        and heat_product(a, pu)):
                    hp = hp_for(sc, a)
                    if not hp:
                        r["excluded"] = ("plant cost over an electric capacity on a heat-product "
                                         "archetype; no heat-to-power ratio")
                        continue
                    mw *= hp[0]
                    cls.append("same-source" if hp[1] == "hpr:same-example" else "internal")
                    used.append(hp[1])
                factor = f / (mw * GJ_PER_KW_YR * 1000)
                note = (f"absolute sum over the source's own output capacity, {mw:.4g} MW "
                        f"({rid}), at nameplate")
                cls.append("same-source")
            elif (p in PER_CAPACITY and d == "yr" and p != "capex"
                  and (oc := own_capacity(r, sc, pu, flist, alias))):
                # an annual sum (fixed O&M in EUR-million/yr) over the same source's
                # annual capacity for the case: a cost per unit of capacity per year
                # (added 2026-10-02)
                per_yr, rid, ids, basis, cu_ = oc
                hv_unit = f"{unit.split('/', 1)[0]}/{cu_.split('/', 1)[0]}"
                factor, used = f / per_yr, used + ids
                note = f"annual sum over the source's own capacity for this case ({rid})"
                if rid.startswith(JUDGED_CAP):
                    note = note.replace("the source's own capacity", "a judged site size")
                    cls.append("assumption")
                    used.append("assume:" + rid)
                else:
                    cls.append("same-source")
                if basis in heating.INPUT_BASES:
                    bridge_kind = "per-input"
            elif p in PER_CAPACITY and not d and (oc := own_capacity(r, sc, pu, flist, alias)):
                per_yr, rid, ids, basis, cu_ = oc
                hv_unit = f"{unit.split('/', 1)[0]}/{cu_.split('/', 1)[0]}"
                factor, used = f / per_yr, used + ids
                note = f"absolute sum over the source's own capacity for this case ({rid})"
                if rid.startswith(JUDGED_CAP):
                    note = note.replace("the source's own capacity", "a judged site size")
                    cls.append("assumption")
                    used.append("assume:" + rid)
                else:
                    cls.append("same-source")
                if basis in heating.INPUT_BASES:
                    bridge_kind = "per-input"
            else:
                ext = convert.intensive(unit, tgt, flist, species_ok, alias) if d else None
                if not ext:
                    r["excluded"] = (f"cost per {d or 'plant'} cannot be put per {td}: the source "
                                     "gives no output factor to convert it with")
                    continue
                factor, used = f * ext[0], used + ext[1]
                note = f"per {d} -> per {td}"
        elif (p.startswith("emissions-") and unit == "%" and (r.get("target") or "").lower() == "ch4"
              and tgt.startswith("t/") and (q := convert.qty_ratio("t", pu, flist, True, alias))):
            # a methane SLIP: the share of the methane handled that is lost. Per unit
            # of methane delivered that is s/(1-s) of it, put per GJ with the heating value
            slip, used = q[0], used + q[1]
            factor, offset = 1.0, 0.0
            note = "methane slip s% of the methane handled: s/(1-s) t lost per t delivered, per GJ"
        else:
            route = units.physical(unit, tgt)
            if route is None and p == "lifetime" and unit in ("hours", "h", "hr"):
                # an operating life in hours (a stack life): years at the example's
                # own capacity factor, or running continuously where it gives none
                cf = scale.get(r["example-id"], {}).get("cf")
                route, note = (1 / (8760 * (cf or 1.0)), 0.0), (
                    f"operating hours at the example's {cf:.0%} capacity factor" if cf
                    else "operating hours at 8,760 h per year (no capacity factor given)")
                cls.append("same-source" if cf else "assumption")
                if not cf:
                    used.append("assume:continuous-operation")
                if not (r.get("target") or "").strip():
                    low = (r.get("note") or "").lower()
                    sp = ("stack" if "stack" in low or a.get("family") in ("hydrogen", "power-generation")
                          else "component")
                    r["target-norm"] = sp
                    r["note"] = (r.get("note") or "") + HOURS_NOTE.format(sp=sp)
            if route is None:
                ext = (convert.extensive(unit, tgt, flist, alias, annual=p == "annual-output")
                       if p in EXTENSIVE
                       else convert.intensive(unit, tgt, flist, species_ok, alias))
                if (not ext and p == "specific-energy-consumption"
                        and (cv := CARRIER_VOLUME.get((r.get("target") or "").strip().lower()))
                        and unit.split("/", 1)[0] in ("m3", "Nm3")
                        and cv in units.factors.rows):
                    # a gas carrier stated by volume (m3 of natural gas or hydrogen per
                    # t): its own heating value per m3 (S150, 2026-10-27); a cubic
                    # metre the source states at STP (0 degC) takes the normal-m3 factor
                    fid = cv
                    if fid == "cf-natural-gas-hhv-volume" and (unit.startswith("Nm3") or re.search(r"\bSTP\b|\bNm3\b|0 ?degC", r.get("note") or "")):
                        fid = "cf-natural-gas-hhv-nm3"
                    ext = convert.intensive(unit, tgt, flist + [units.factors.rows[fid]],
                                            species_ok, alias)
                if not ext:
                    r["excluded"] = f"no conversion from {unit or '(none)'} to {tgt}"
                    continue
                route, used = (ext[0], 0.0), used + ext[1]
            factor, offset = route
        if chp_total and factor is not None:
            # the whole CHP plant's levelized cost per MWh of ELECTRICITY, no heat
            # credit (records/reference/chp-levelized-total.csv): per GJ of heat is
            # the same annual cost over heat = electricity x heat-to-power ratio
            hp = hp_for(scale.get(r["example-id"], {}), a)
            if not hp:
                r["excluded"] = ("CHP levelized cost per unit of electricity; no "
                                 "heat-to-power ratio")
                continue
            factor /= hp[0]
            note = (note + "; " if note else "") + (
                f"whole-plant cost per unit of electricity / heat-to-power ratio {hp[0]:.3g} "
                "(records/reference/chp-levelized-total.csv)")
            cls.append("same-source" if hp[1] == "hpr:same-example" else "internal")
            used.append(hp[1])
        if (p == "capacity" and r.get("capacity-basis") == "electric-output"
                and heat_product(a, pu) and factor is not None):
            # a CHP unit rated in MW electric, on an archetype whose product is heat
            hp = hp_for(scale.get(r["example-id"], {}), a)
            if not hp:
                r["excluded"] = ("electric rating on a heat-product archetype; no "
                                 "heat-to-power ratio")
                continue
            factor *= hp[0]
            note = (note + "; " if note else "") + f"MW electric x heat-to-power ratio {hp[0]:.3g}"
            cls.append("same-source" if hp[1] == "hpr:same-example" else "internal")
            used.append(hp[1])
        if (p in ("opex-variable", "opex-total") and r.get("capacity-basis") == "electric-output"
                and re.search(r"/(k|M)Wh\b", unit) and heat_product(a, pu) and factor is not None):
            # a cost per MWh of ELECTRICITY on an archetype whose product is heat
            # (CHP, waste-to-energy): per GJ of heat with the heat-to-power ratio,
            # as the per-kW costs (fixed 2026-10-09; before, per MWh electric was
            # read as per MWh of heat)
            hp = hp_for(scale.get(r["example-id"], {}), a)
            if not hp:
                r["excluded"] = ("cost per unit of electricity on a heat-product archetype; no "
                                 "heat-to-power ratio from the source or the archetype's examples")
                continue
            factor /= hp[0]
            note = (note + "; " if note else "") + f"per MWh electric / heat-to-power ratio {hp[0]:.3g}"
            cls.append("same-source" if hp[1] == "hpr:same-example" else "internal")
            used.append(hp[1])
        if p in EXTENSIVE and pu == "GJ" and r.get("capacity-basis") in heating.INPUT_BASES:
            bridge_kind = "capacity-in"
        if (p in PER_CAPACITY and pu == "GJ" and not bridge_kind
                and r.get("capacity-basis") in heating.INPUT_BASES):
            bridge_kind = "per-input"
        # an absolute sum put on the same source's capacity: that capacity's unit decides
        # the heating-value basis (2026-10-24, P-H1)
        how = heating.kind(p, hv_unit or unit, pu, r.get("capacity-basis") or "", per_product)
        if how:
            hv = heating.reported(r.get("note") or "")
            # a source that reports on LHV unless it says otherwise (ruling 2026-10-05)
            default, dkind = ("", "") if hv else units.streams.default(r.get("source-id"),
                                                                        r.get("via-source-id"))
            dlabel = "regional default" if dkind == "regional" else "source default"
            r["heating-value"] = hv or (f"LHV ({dlabel})" if default == "LHV" else
                                        f"HHV ({dlabel})" if default == "HHV" else "not stated")
            fs = None
            if (how == "energy-use" and (r.get("target") or "").strip() in ("", "total", "not-stated")
                    and _mixed_carriers(a)):
                # an all-carrier total on an archetype that also uses electricity or
                # heat: the fuel's HHV/LHV ratio goes on the fuel share only (sanity-check
                # 2026-10-23, S144/NFM011: the methane ratio was lifting electricity 11%).
                # The share is the example's own carriers where it gives them, else all
                # fuel on a conversion archetype or one whose first carrier is a fuel,
                # else none
                g = _gj_per_den(r)
                own = ex_carriers.get((r["example-id"], g[1])) if g else None
                if not own and g:
                    # no carriers in the example: the archetype's other examples' median
                    # carriers on the same denominator
                    ac = arch_carriers.get((a["archetype"], g[1]))
                    own = {t: median(v) for t, v in ac.items()} if ac else None
                fs = _fuel_share(r, own, a)
            if (hv == "LHV" or default == "LHV") and fs == 0.0:
                r["heating-value"] = f"{r['heating-value']}; no fuel share, not converted"
            elif hv == "LHV" or default == "LHV":
                h = units.streams.to_hhv(a["archetype"], how, pu, r.get("target") or "", unit,
                                         r.get("example-id") or "")
                if h is not None and fs is not None and fs < 1.0:
                    h = (1 + (h[0] - 1) * fs, h[1])
                    r["heating-value"] = f"{r['heating-value']}; ratio on the fuel share {fs:.0%}"
                if h is None and hv == "LHV":
                    r["excluded"] = ("reported on a lower-heating-value basis; "
                                     "records/reference/heating-value-stream.csv names no fuel or "
                                     "product for this archetype to put it on HHV with")
                    continue
                if h is None:
                    # the source's default basis, but no stream known: pooled as
                    # reported, and said so
                    r["heating-value"] = f"LHV? ({dlabel}, no stream)"
                else:
                    factor, offset = factor * h[0], offset * h[0]
                    if slip:
                        slip /= h[0]
                    used += h[1]
                    note = (note + "; " if note else "") + (
                        "LHV -> HHV" if hv else f"LHV ({dlabel}) -> HHV")
        if bf:
            # the source reports per another product (per t aluminium on an alumina
            # archetype): per-product numbers divide by it, capacities multiply
            factor = factor * bf[0] if p in EXTENSIVE else factor / bf[0]
            used.append(bf[1])
        if steam_pre:
            factor *= steam_pre
            used.append("cf-steam-enthalpy")
            note = (note + "; " if note else "") + "steam mass x enthalpy"
        factor_ids = [u for u in used if u in units.factors.rows]
        if factor_ids:
            note = (note + "; " if note else "") + "factor " + ", ".join(dict.fromkeys(factor_ids)) + \
                " (records/reference/conversion-factor.csv)"
        if m or factor != 1.0 or offset or unit != (r.get("unit") or "").strip():
            cls.append("unit")
        cls += [units.factors.cls.get(u) or ("assumption" if u.startswith("assume:") else "")
                for u in used]
        r["conversion-class"] = convert.weakest(cls)
        r["conversion-factors"] = "; ".join(dict.fromkeys(used))
        if bridge_kind:
            r["_bridge"] = bridge_kind
        if slip:
            conv = lambda x: None if x is None else (x / 100) / (1 - x / 100) / slip * factor
        else:
            conv = lambda x: None if x is None else x * factor + offset
        r["value-norm"], r["value-min-norm"], r["value-max-norm"] = (
            "" if x is None else f"{x:.6g}" for x in (conv(v), conv(lo), conv(hi)))
        if efficacy:
            note = (f"efficacy in {efficacy} -> GJ electricity per {LIGHT_UNIT} (3.6 / efficacy)"
                    + ("; " + note if note else ""))
        r["unit-norm"], r["conversion"] = tgt, note
        if (r.get("status") or "") == "review" and "BASIS NOT RECORDED" in (r.get("note") or ""):
            r["excluded"] = "a required basis (cost-basis or capacity-basis) is not recorded"
        elif p in PER_CAPACITY | PER_OUTPUT and any(x is not None and x < 0 for x in (v, lo, hi)):
            r["excluded"] = NEGATIVE_COST
        if r["duplicate-of"] and FUEL_INPUT_DIVISOR.search(r.get("formula") or "") and pu == "GJ":
            r["conversion"] = FUEL_INPUT_NOTE.format(parent=r["duplicate-of"])
            r["_fuel_input"] = r["duplicate-of"]
            r["excluded"] = "per GJ of fuel input"          # settled in the last pass
    # UKERC emission columns that are not the step's own direct emissions
    # (sanity-check 2026-10-23, S140/RX004, HPCH007, HPCH008): a negative value is
    # a change against UKERC's baseline, not an emission; a 'Specific emissions'
    # above the example's direct fuel plus process emissions (less captured)
    # carries UKERC's grid or steam emissions (scope 2), which the scope-1
    # emissions-total does not hold (2.0 t/GJ on an electric motor)
    ukc = defaultdict(dict)
    for r in facts:
        m = re.search(r"UKERC column '([^']+)'", r.get("note") or "")
        if m and r["parameter"].startswith("emissions-") and fnum(r.get("value")) is not None:
            ukc[r["example-id"]][m.group(1)] = r
    for eid, cols in ukc.items():
        for col, r in cols.items():
            if fnum(r["value"]) < 0 and not r.get("excluded"):
                r["excluded"] = ("UKERC measure delta: a negative emission is a change against "
                                 "UKERC's baseline, not the step's own emissions (S140)")
        sp_ = cols.get("Specific emissions")
        if sp_ is not None and not sp_.get("excluded"):
            direct = sum(fnum(cols[c]["value"]) for c in ("Direct fuel related emissions",
                                                          "Process related emissions") if c in cols)
            cap = fnum(cols["Captured emissions"]["value"]) if "Captured emissions" in cols else 0.0
            if fnum(sp_["value"]) > 1.05 * max(direct - cap, 0.0) + 1e-9:
                sp_["excluded"] = ("UKERC 'Specific emissions' above the example's direct fuel and "
                                   "process emissions: it includes indirect (grid or steam) "
                                   "emissions, not scope 1 (S140)")
    # per GJ of fuel input, per unit of input capacity and (where approved) per
    # tonne of a host plant's product -> per GJ of output, by the efficiency
    bridge.apply(facts, records, examples, units)
    # A fixed O&M the source gives as a percentage of capital per year is put on the
    # example's own capital cost -- the same source's number, so it is only a
    # derivation -- where the example has one pooled capex (or one per case).
    capex = defaultdict(list)
    for r in facts:
        if r["parameter"] == "capex" and r.get("value-norm") and not r.get("excluded"):
            capex[r["example-id"]].append(r)
    for r in facts:
        u = (r.get("unit") or "").strip()
        if r["parameter"] != "opex-fixed" or not u.startswith("%") or r.get("value-norm"):
            continue
        cands = capex.get(r["example-id"], [])
        case = re.match(r"^\[([^\]]+)\]", r.get("note") or "")
        if case and len(cands) > 1:
            cands = [c for c in cands if (c.get("note") or "").startswith(f"[{case.group(1)}]")]
        if len({c["value-norm"] for c in cands}) != 1:
            r["excluded"] = ("percentage of capital cost: the example has "
                             + ("no pooled capital cost" if not cands else "several capital costs")
                             + " to put it on")
            continue
        c = cands[0]
        v, lo, hi = _num(r)
        cv = fnum(c["value-norm"])
        r["value-norm"], r["value-min-norm"], r["value-max-norm"] = (
            "" if x is None else f"{x / 100 * cv:.6g}" for x in (v, lo, hi))
        r["unit-norm"] = c["unit-norm"]
        r["excluded"] = ""
        r["conversion"] = f"percentage of the example's capital cost {c['record-id']} per year"
        r["conversion-class"] = convert.weakest([c.get("conversion-class", ""), "same-source"])
        r["conversion-factors"] = c.get("conversion-factors", "")
    # A parent the build cannot convert (a cost per plant) whose pooled derived row
    # restates it per unit of capacity is a duplicate of that row, not a gap.
    kids = defaultdict(list)
    for r in facts:
        if r.get("value-type") == "derived" and r.get("value-norm") and not r.get("excluded"):
            kids[(r["example-id"], r["parameter"])].append(r)
    for r in facts:
        e = r.get("excluded") or ""
        if not e or e == "no number" or r.get("value-type") == "derived":
            continue
        if not (e.startswith("cost per") or e.startswith("no conversion")):
            continue
        val = (r.get("value") or "").strip()
        for k in kids.get((r["example-id"], r["parameter"]), []):
            f = k.get("formula") or ""
            if k.get("derived-from") == r["record-id"] or (val and re.match(rf"^\s*\(?{re.escape(val)}\b", f)):
                r["duplicate-of"], k["duplicate-of"] = k["record-id"], ""
                r["excluded"] = f"restated in the pooled derived row {k['record-id']} (duplicate-of); not pooled twice"
                break
    # A derived row that restates its parent in another unit adds no information.
    # UKERC's per-GJ costs are the per-tonne cost divided by the fuel used per
    # tonne, i.e. per GJ of fuel INPUT. Where the parent could not be pooled either,
    # the reason stays the parent's.
    # The per-tonne rows those came from are set aside with them: on an
    # energy-service archetype both need the same missing efficiency.
    sec = {r["example-id"] for r in facts if r["parameter"] == "specific-energy-consumption"
           and (r.get("value") or "").strip()}
    for r in facts:
        a = records.by_key.get(r.get("archetype", ""))
        if (a and a["product-unit"] == "GJ" and r["parameter"] in PER_CAPACITY | PER_OUTPUT
                and "ukerc" in (r.get("via-source-id") or r.get("source-id") or "")
                and r["example-id"] in sec and r.get("value-type") != "derived"
                and (r.get("excluded") or "").startswith("cost per t")):
            r["excluded"] = SET_ASIDE
    # examples reviewed for the bridge and not approved: their per-tonne numbers
    # are the host process's, not the equipment's (efficiency-bridge.csv)
    for r in facts:
        b = bridged.get(r["example-id"])
        e = r.get("excluded") or ""
        if b and b.get("bridge") == "no" and (e == SET_ASIDE or e.startswith("cost per t")
                                            or re.match(r"no conversion from \S+/t to \S+/GJ", e)):
            r["excluded"] = "not bridged: " + (b.get("note") or "")
            r["set-aside"] = "not-bridged"
    plant_level = {r["record-id"]: r["note"] for r in read(REFERENCE / "plant-level-capture-rate.csv")}
    # single rows a modeller ruled out, with the reason (records/reference/excluded-row.csv)
    ruled_out = {r["record-id"]: r for r in read(REFERENCE / "excluded-row.csv")}
    unknown = set(ruled_out) - set(by_id)
    if unknown:
        raise SystemExit(f"excluded-row.csv names unknown record-ids: {sorted(unknown)}")
    # a ruling is about a figure, not a record (sanity-check 2026-10-11, S113): the
    # same source's identical figure on the same archetype, parameter and target
    # (a twin card, the same table read for another year or case) is ruled out with it
    def _twin_key(r):
        return (r.get("source-id", ""), r.get("archetype", ""), r["parameter"],
                (r.get("target") or "").strip().lower(), (r.get("unit") or "").strip(),
                (r.get("value") or "").strip(), (r.get("value-min") or "").strip(),
                (r.get("value-max") or "").strip())
    twin_of = {}
    by_twin = defaultdict(list)
    for r in facts:
        by_twin[_twin_key(r)].append(r["record-id"])
    for rid in ruled_out:
        if "(this example only)" in (ruled_out[rid].get("reason") or ""):
            continue        # a reason about the example, not the figure (S152, 2026-10-27)
        for t in by_twin.get(_twin_key(by_id[rid]), []):
            if t != rid and t not in ruled_out:
                twin_of.setdefault(t, rid)
    abatement = abatement_options(records)
    for r in facts:
        a = records.by_key.get(r.get("archetype", ""))
        if r.get("_superseded"):
            # the printed value is taken to be wrong; its corrected copy is pooled
            # instead (pipeline/correction.py, records/reference/source-correction.csv)
            from .correction import superseded_note
            r["excluded"] = superseded_note(r)
            continue
        x = ruled_out.get(r["record-id"])
        if x and r.get("excluded") != "no number":
            r["excluded"] = f"ruled out: {x['reason']} ({x['decided']})"
            continue
        if r["record-id"] in twin_of and r.get("excluded") != "no number":
            t = twin_of[r["record-id"]]
            x = ruled_out[t]
            r["excluded"] = (f"ruled out as the twin of {t} (same source, figure and parameter): "
                             f"{x['reason']} ({x['decided']})")
            continue
        if r["record-id"] in plant_level and r.get("excluded") != "no number":
            # capture-rate on a capture plant is its own rate on the gas it treats
            r["excluded"] = ("plant-level capture rate on a capture plant: "
                             + plant_level[r["record-id"]] + " (modeller 2026-10-02; "
                             "records/reference/plant-level-capture-rate.csv)")
            continue
        if is_ccs_host(a) and r["parameter"] in CCS_HOST_COSTS:
            # the host plant's cost comes from its host archetype (fill rule
            # est-ccs-host-*); a source's with-capture or added cost is capture cost
            if r.get("excluded") != "no number":
                r["set-aside"], r["excluded"] = "ccs-add-on", CCS_HOST_NOTE
            continue
        if is_capture(a) or (a and a["archetype"] in abatement):
            # a capture plant's incremental cost IS its own cost: no host needed;
            # so is an abatement option's against a zero-cost no-abatement option
            if r.get("set-aside") == "needs-host-cost":
                r["set-aside"] = ""
                if r.get("excluded") in (NEEDS_HOST, INCREMENTAL_NOTE):
                    r["excluded"] = ""
                if not is_capture(a) and ABATEMENT_NOTE not in (r.get("note") or ""):
                    r["note"] = (r.get("note") or "") + ABATEMENT_NOTE
        elif (r["parameter"] in PER_CAPACITY | PER_OUTPUT and r.get("cost-basis") == "incremental"
                and r.get("excluded") != "no number"):
            r["set-aside"], r["excluded"] = "needs-host-cost", INCREMENTAL_NOTE
            continue
        if r.get("set-aside") == "needs-host-cost" and r.get("excluded") != "no number":
            r["excluded"] = NEEDS_HOST
        elif r.get("whole-plant") and r.get("excluded") != "no number":
            r["excluded"] = (f"incremental: the whole-plant row {r['whole-plant']} adds the "
                             "host plant's baseline cost")
        elif r.get("excluded") == SET_ASIDE:
            r["set-aside"] = "needs-efficiency"
            r["excluded"] = SET_ASIDE + (". Not yet reviewed for the efficiency bridge: list the "
                                         "example in records/reference/efficiency-bridge.csv")
    for r in facts:
        if r.get("assembly-duplicate"):
            # an assembled sum the source also prints: a check, not a second value
            r["duplicate-of"] = r["assembly-duplicate"]
            r["excluded"] = ("assembly check: the sum of this source's component costs, "
                             "which it also prints as a total (duplicate-of); not pooled")
            continue
        if r.get("_fuel_input"):
            # a per-GJ-of-fuel-input restatement: a duplicate of its parent, which
            # carries the reason if it is not pooled either
            r["duplicate-of"] = r["_fuel_input"]
            par = by_id[r["duplicate-of"]]
            r["set-aside"] = ""
            r["value-norm"] = r["value-min-norm"] = r["value-max-norm"] = ""
            r["excluded"] = ("duplicate of the row in duplicate-of: its cost divided by the source's "
                             "fuel use (per GJ of fuel input)"
                             + ("" if not par.get("excluded") else "; that row is not pooled either"))
            continue
        if r.get("excluded", "").startswith(SET_ASIDE) or r.get("set-aside") == "not-bridged":
            continue
        if r.get("duplicate-of") and r["excluded"] and r["excluded"] != "no number":
            par = by_id[r["duplicate-of"]]
            how = ("its per-tonne cost divided by fuel use per tonne"
                   if FUEL_INPUT_DIVISOR.search(r.get("formula") or "")
                   else "the same number in another unit")
            r["excluded"] = (f"duplicate of the row in duplicate-of: {how}"
                             + ("" if not par.get("excluded") else "; that row is not pooled either"))

    # rows of one example that are parts of one quantity (two electricity uses of
    # one plant): the first keeps the sum, the others point to it (records/reference/
    # additive-row.csv; S150, 2026-10-27). Otherwise the pool would take their median
    groups = defaultdict(list)
    for x in read(REFERENCE / "additive-row.csv"):
        if x["record-id"] not in by_id:
            raise SystemExit(f"additive-row.csv names an unknown record-id: {x['record-id']}")
        groups[x["group"]].append(by_id[x["record-id"]])
    for g, parts in groups.items():
        live = [x for x in parts if not x.get("excluded") and x.get("value-norm")]
        if len(live) < 2:
            continue
        if len({x.get("unit-norm") for x in live}) != 1:
            raise SystemExit(f"additive-row.csv group {g}: parts on different units")
        head = live[0]
        head["value-norm"] = f"{sum(fnum(x['value-norm']) for x in live):.6g}"
        head["value-min-norm"] = head["value-max-norm"] = ""
        head["conversion"] = ((head.get("conversion") + "; ") if head.get("conversion") else "") + (
            "sum of the example's additive rows " + " + ".join(x["record-id"] for x in live)
            + f" (records/reference/additive-row.csv, group {g})")
        for x in live[1:]:
            x["duplicate-of"] = head["record-id"]
            x["excluded"] = f"part of the sum in {head['record-id']} (additive-row.csv, group {g}); not pooled twice"

    # A figure one source prints more than once for the same archetype (the same
    # cost for several technologies, one lifetime for a whole table) is pooled
    # once (modeller, 2026-09-26 and 2026-09-29). The first example in record
    # order keeps it; the others point to it in duplicate-of.
    first = {}
    for r in facts:
        if r.get("excluded") or r.get("archetype") in ("", "(none)", None):
            continue
        norm = (r.get("value-norm"), r.get("value-min-norm"), r.get("value-max-norm"))
        if not any(norm):
            continue
        k = (r["archetype"], r.get("source-id"), r["parameter"], r.get("target-norm", ""),
             r.get("unit-norm"), *norm)
        if k not in first:
            first[k] = r
        elif first[k]["example-id"] != r["example-id"]:
            r["duplicate-of"] = first[k]["record-id"]
            r["excluded"] = (f"repeat: the source prints the same figure for "
                             f"{first[k]['example-id']} (duplicate-of); pooled once")
    # a carrier row split from a MIDDEN total goes with its total, whatever rules
    # the total out: a ruling, a twin, a repeat (pipeline/midden_carriers.py, 2026-10-24)
    for r in facts:
        if r.get("derived-kind") != "carrier-term" or r.get("excluded") == "no number":
            continue
        par = by_id.get(r.get("derived-from", ""))
        if par is None or not par.get("excluded"):
            continue
        pe = par["excluded"]
        if pe.startswith("repeat:"):
            twin = par.get("duplicate-of", "")
            r["duplicate-of"] = f"{twin}-{r.get('target-norm') or r.get('target')}" if twin else ""
            r["excluded"] = pe
        elif pe.startswith(("ruled out", "example not mapped", "example set aside")):
            r["duplicate-of"] = ""
            r["excluded"] = f"goes with its total {par['record-id']}: {pe}"


FLOW_PER_UNIT = {"unit-baseline-output": None, "unit": None}   # None: the archetype's own unit


# t per normal m3 (0 degC, 1 atm)
GAS_DENSITY = {"oxygen": 0.001429, "nitrogen": 0.0012506, "co2": 0.001977, "co2-captured": 0.001977,
               "hydrogen": 0.0000899, "air": 0.001293}
# normal m3 per unit of volume: scf is at 60 degF and 1 atm (0.0283168 m3 x 273.15/288.71)
GAS_VOLUME = {"m3": 1.0, "1000 m3": 1000.0, "scf": 0.0283168 * 273.15 / 288.71, "Mscf": 28.3168 * 273.15 / 288.71}


def apply_flow_exclusions(flows, factors) -> dict:
    """records/reference/excluded-flow.csv: a flow row ruled out (action exclude) or
    put on the archetype's basis by a registered factor (divide or multiply), as
    excluded-row.csv does for parameters (general rule 5, 2026-11-05)."""
    from collections import Counter
    T = {r["record-id"]: r for r in read(REFERENCE / "excluded-flow.csv")}
    stats = Counter()
    for r in flows:
        x = T.get(r.get("record-id", ""))
        if not x:
            continue
        act = (x.get("action") or "exclude").strip()
        if act == "exclude":
            r["excluded"] = "ruled out: " + x.get("reason", "")
            stats["excluded"] += 1
            continue
        f = factors.rows.get(x.get("factor-id", ""), {}).get("_f")
        if not f:
            r["excluded"] = f"ruled out: factor {x.get('factor-id')} not found ({x.get('reason', '')})"
            stats["factor not found"] += 1
            continue
        k = 1.0 / f if act == "divide" else f
        for c in ("value-norm", "value-min-norm", "value-max-norm"):
            v = fnum(r.get(c))
            if v is not None:
                r[c] = f"{v * k:.6g}"
        r["conversion"] = "; ".join(y for y in [r.get("conversion", ""), f"{act} by {x['factor-id']}"] if y)
        stats[act] += 1
    unknown = set(T) - {r.get("record-id", "") for r in flows}
    if unknown:
        stats["unknown record-ids"] = len(unknown)
    return dict(stats)


def _flow_hhv(r, a, pu, per, tnum, units) -> None:
    """Puts an energy flow on the HHV basis the parameters use (Brad 2026-11-06):
    an energy quantity of a fuel commodity is multiplied by the fuel's HHV/LHV
    ratio, and any flow per GJ of an energy product is divided by the product's,
    as heating.py does for parameters. Only rows the source states, or reports by
    default, on LHV change; the conversion column says so."""
    how = ""
    if tnum == "GJ" and r.get("commodity", "") in heating.CARRIER_RATIO:
        how = "energy-use"
    elif pu == "GJ" and (per == pu or per in heating.ENERGY):
        how = "per-product"
    if not how:
        return
    hv = heating.reported(r.get("note") or "")
    default = "" if hv else units.streams.default(r.get("source-id"), r.get("via-source-id"))[0]
    if "LHV" not in (hv, default):
        return
    h = units.streams.to_hhv(a["archetype"], how, pu, r.get("commodity", "") if how == "energy-use" else "",
                             f"{tnum}/{pu}", r.get("example-id") or "")
    label = "LHV" if hv else "LHV (source default)"
    if h is None:
        r["conversion"] = "; ".join(x for x in [r.get("conversion", ""), f"{label}? no stream, not converted"] if x)
        return
    for c in ("value-norm", "value-min-norm", "value-max-norm"):
        x = fnum(r.get(c))
        if x is not None:
            r[c] = f"{x * h[0]:.6g}"
    r["conversion"] = "; ".join(x for x in [r.get("conversion", ""), f"{label} -> HHV x {h[0]:.4g} ({', '.join(h[1])})"] if x)


def normalise_flows(flows, records, units, examples=None) -> None:
    """Flows per unit of the archetype's product: energy in GJ, mass in t,
    gas volumes in m3."""
    from .views import product_properties
    props = product_properties()
    # examples whose flows are per t OF THE PRODUCT: they carry the product itself
    # as an output of 1 per t (a per-t flow can be per t of feed otherwise)
    per_t_product = set()
    for r in flows:
        a = records.by_key.get(r.get("archetype", ""))
        if a and r.get("direction") == "output" and r.get("commodity") == a.get("product") \
                and (r.get("per-unit") or "").strip() == "t" and fnum(r.get("value")) == 1.0:
            per_t_product.add(r.get("example-id"))
    for r in flows:
        r.update({"value-norm": "", "value-min-norm": "", "value-max-norm": "",
                  "unit-norm": "", "excluded": "", "conversion": ""})
        a = records.by_key.get(r.get("archetype", ""))
        v, lo, hi = _num(r)
        if v is None and lo is None and hi is None:
            r["excluded"] = "no number"
            continue
        if not a:
            r["excluded"] = "example not mapped to an archetype"
            continue
        pu = a["product-unit"]
        per = (r.get("per-unit") or "").strip()
        per = UNIT_ALIAS.get(per, per)
        if per in FLOW_PER_UNIT:
            per = service_unit((examples or {}).get(r.get("example-id", "")), pu)
        # per another unit of the same product: an energy unit of an energy product
        # (per MWh on a GJ archetype), or per t / per m3 of the product on a GJ
        # archetype with the product's energy content (product-property.csv)
        # (user 2026-10-23: co-product outputs recorded as flows, gross-input ruling)
        scale = 1.0
        if per != pu:
            pr = props.get(a.get("product", ""), {})
            ec = fnum(pr.get("product-energy-content"))
            ecu = pr.get("product-energy-content-unit", "")
            route = units.physical(per, pu) if per and pu else None
            if route is not None and route[1] == 0.0 and route[0]:
                scale = 1.0 / route[0]          # value per `per` -> per pu
                r["conversion"] = f"per {per} -> per {pu}"
            elif pu == "GJ" and ec and per == "t" and ecu == "GJ/t" and r.get("example-id") in per_t_product:
                scale = 1.0 / ec
                r["conversion"] = f"per {per} of {a.get('product')} -> per GJ ({ec:g} {ecu})"
            else:
                r["excluded"] = f"per {per}, but the archetype's product is measured in {pu}"
                continue
        num = (r.get("unit") or "").split("/", 1)[0].replace("Nm3", "m3")
        cls = records.commodities.get(r["commodity"], {}).get("unit-default", "")
        tnum = "GJ" if num in ("GJ", "MJ", "kJ", "TJ", "kWh", "MWh") else "t" if num in ("kg", "t", "g") else num
        # a gas measured by volume on a commodity measured in t: by its density at
        # normal conditions (0 degC, 1 atm; scf at 60 degF), so the rows pool on one
        # basis (general rule 5, 2026-11-05)
        gas = GAS_DENSITY.get(r.get("commodity", ""))
        if cls == "t" and gas and num in GAS_VOLUME:
            k = gas * GAS_VOLUME[num]
            conv = lambda x, k=k: None if x is None else x * k * scale
            r["value-norm"], r["value-min-norm"], r["value-max-norm"] = (
                "" if x is None else f"{x:.6g}" for x in (conv(v), conv(lo), conv(hi)))
            r["unit-norm"] = f"t/{pu}"
            r["conversion"] = "; ".join(x for x in [r.get("conversion", ""),
                                                    f"{num} of {r['commodity']} -> t at {gas * 1000:g} kg/m3"] if x)
            _flow_hhv(r, a, pu, per, "t", units)
            continue
        if cls and cls in ("GJ", "t") and tnum != cls and tnum in ("GJ", "t"):
            r["excluded"] = f"{r['commodity']} is measured in {cls}, the row in {num}"
            continue
        route = units.physical(num, tnum) if num != tnum else (1.0, 0.0)
        if route is None:
            r["excluded"] = f"no conversion from {num} to {tnum}"
            continue
        f, o = route
        conv = lambda x: None if x is None else (x * f + o) * scale
        r["value-norm"], r["value-min-norm"], r["value-max-norm"] = (
            "" if x is None else f"{x:.6g}" for x in (conv(v), conv(lo), conv(hi)))
        r["unit-norm"] = f"{tnum}/{pu}"
        _flow_hhv(r, a, pu, per, tnum, units)
