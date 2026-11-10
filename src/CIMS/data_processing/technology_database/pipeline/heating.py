"""Heating-value basis: every energy quantity in the database is on a HIGHER
heating value (HHV, gross calorific value) basis (design.md s5, 2026-09-26).

Most sources do not say which basis they use. A number whose basis the source
does not state is read as HHV and marked `not stated` in `heating-value`, unless
its source is one records/reference/source-heating-basis.csv lists as reporting
on LHV by default (DEA catalogues, ETSAP briefs, EU BREFs, MIDDEN; ruling
2026-10-05): then it is converted like a stated LHV and marked `LHV (source
default)`, or, where no stream is known for the archetype, pooled as it is and
marked `LHV? (source default, no stream)`. One the source states as HHV is used
as it is. One the source states as LOWER heating
value (LHV) is converted, using the ratio of the two heating values of the fuel
going in and of the product coming out, as records/reference/
heating-value-stream.csv names them for the archetype (the ratios are factors in
conversion-factor.csv, class `property`). An LHV number on an archetype with no
row there is not pooled.

What changes, with r = HHV/LHV of a stream (1 for electricity, heat or a
stream the table leaves blank):

  efficiency, efficiency-chp        x r_out / r_in      (output over input energy)
  specific energy use (energy/pu)   x r_in, and / r_out when it is per unit of
                                    product ENERGY (pu is GJ, or MWh/MWh on a
                                    per-tonne archetype)
  a cost or emission per GJ of pu   / r_out
  a cost per kW or MWh of product   / r_out, before the product's heating value
    on a per-tonne archetype          puts it per tonne (amended 2026-09-30)
  capacity or output in energy      x r_out (output basis), x r_in (input basis)

r_in for a specific energy use is the ratio of the carrier the row names in
`target` (CARRIER_RATIO): 1 for electricity, heat and steam, which have no
heating value, and the archetype's `fuel-in` only for a carrier the map does not
know or a row with no target. Before 2026-09-30 every target took `fuel-in`, so
an LHV electricity or steam use was raised by the fuel's ratio.

Anything else (a cost per tonne, a lifetime) does not depend on the basis, and
its `heating-value` cell stays blank.
"""
from __future__ import annotations

import re

from .common import REFERENCE, fnum, read

NOT_STATED = re.compile(r"\b(HHV|LHV)\b[^.]{0,25}not stated|not state (LHV|HHV)|"
                        r"basis NOT STATED|basis not stated", re.I)
ALL_CLAIM = re.compile(r"all energy quantities[^.]*?\b(HHV|LHV)\b", re.I)
LHV = re.compile(r"\bLHV\b|lower heating value|net calorific", re.I)
HHV = re.compile(r"\bHHV\b|higher heating value|gross calorific", re.I)
ENERGY = {"GJ", "MJ", "kJ", "TJ", "PJ", "kWh", "MWh", "GWh", "Btu", "kBtu", "MMBtu", "mmBtu", "MBtu"}
ENERGY_RATE = {"MW", "kW", "W", "GW", "MWth", "kWth", "GJ/h", "MJ/h", "MJ/s", "GJ/yr", "TJ/yr", "PJ/yr"}
EFFICIENCY = {"efficiency", "efficiency-chp"}
# a power or energy denominator: a cost per kW or per MWh of the product
ENERGY_DEN = ENERGY | {"kW-yr", "MW-yr", "kW-per-yr", "MWth", "kWth", "kW", "MW", "W", "GW"}
# HHV/LHV ratio factor by the carrier a specific energy use names ("" = no heating value)
NO_HV = {"electricity", "heat", "steam", "cooling", "hot-water", "waste-heat", "solar",
         "geothermal", "nuclear"}
CARRIER_RATIO = {
    **{c: "" for c in NO_HV},
    **{c: "cf-hhv-lhv-methane" for c in ("natural-gas", "biogas", "biomethane", "methane",
                                          "renewable-natural-gas", "landfill-gas", "sng")},
    "hydrogen": "cf-hhv-lhv-hydrogen", "methanol": "cf-hhv-lhv-methanol",
    "ammonia": "cf-hhv-lhv-ammonia", "ethanol": "cf-hhv-lhv-ethanol",
    "black-liquor": "cf-hhv-lhv-black-liquor", "coke-oven-gas": "cf-hhv-lhv-coke-oven-gas",
    "blast-furnace-gas": "cf-hhv-lhv-blast-furnace-gas",
    **{c: "cf-hhv-lhv-solid-liquid" for c in ("coal", "coke", "oil", "diesel", "gasoline", "naphtha",
                                               "petroleum-coke", "biomass", "biomass-solid",
                                               "bioliquid", "liquid-biomass", "fuel-oil",
                                               "heavy-fuel-oil", "waste-fuel")},
}
INPUT_BASES = {"thermal-input", "electric-input"}


def reported(text: str) -> str:
    """The basis a row's note says it is on: 'HHV', 'LHV' or '' (not stated).
    The first explicit mention wins ("HHV. The catalog also gives LHV")."""
    text = text or ""
    m = ALL_CLAIM.search(text)
    if m:
        return m.group(1).upper()
    if NOT_STATED.search(text):
        return ""
    lo, hi = LHV.search(text), HHV.search(text)
    if lo and hi:
        return "LHV" if lo.start() < hi.start() else "HHV"
    return "LHV" if lo else "HHV" if hi else ""


MASS_VOLUME = ("mass", "volume", "amount-h2")


def _dim(tok: str) -> str:
    """The physical dimension of a unit token ('mass', 'volume', 'energy', 'time',
    ...), or '' if unknown; 'kg-H2' reads as 'kg'."""
    if not tok:
        return ""
    from .convert import atom
    a = atom(tok) or atom(tok.split("-")[0])
    return a[0] if a else ""


def kind(param: str, unit: str, pu: str, capacity_basis: str, per_pu: bool) -> str:
    """How a number depends on the heating value, or '' if it does not."""
    num = (unit or "").split("/", 1)[0]
    if param in EFFICIENCY:
        return "efficiency"
    if param == "specific-energy-consumption" and num in ENERGY:
        return "energy-use"
    if param in ("capacity", "annual-output") and (unit in ENERGY_RATE or num in ENERGY):
        return "capacity-in" if capacity_basis in INPUT_BASES else "capacity-out"
    den = (unit or "").split("/", 1)[1] if "/" in (unit or "") else ""
    if pu == "GJ" and per_pu:
        # a number per TONNE (or m3) of the product carries no heating-value basis:
        # the mass or volume factor that puts it per GJ is already HHV (sanity-check
        # run 2026-10-24, HPCH017/RX011: per-t-H2 emission factors had been divided
        # by hydrogen's HHV/LHV ratio a second time, 18% low). The denominator's
        # dimension decides; with no denominator, or a time one (t/yr), the numerator's
        # does. An unknown token (hkWh, ng/J's J) keeps the correction.
        if _dim(den.split("/", 1)[0].split("-per-")[0]) in MASS_VOLUME:
            return ""
        if (not den or _dim(den.split("/", 1)[0].split("-per-")[0]) == "time") \
                and _dim(num) in MASS_VOLUME:
            return ""
        return "per-product"
    if (per_pu and den in ENERGY_DEN and param.startswith(("capex", "opex", "cost-"))
            and capacity_basis not in INPUT_BASES):
        # a cost per kW or MWh of the product's own energy on a per-tonne archetype
        return "per-product"
    return ""


class Streams:
    def __init__(self, factors):
        self.rows = {r["archetype"]: r for r in read(REFERENCE / "heating-value-stream.csv")}
        # an example whose fuel differs from its archetype's usual fuel (moist wood
        # chips on a solid-fuel boiler): its own fuel-in factor (S156, 2026-10-27)
        self.example_fuel = {r["example-id"]: r["fuel-in"].strip()
                             for r in read(REFERENCE / "heating-value-example.csv")}
        self.factors = factors
        self.defaults = [(r["source-pattern"].strip(), (r.get("default-basis") or "").strip().upper())
                         for r in read(REFERENCE / "source-heating-basis.csv")
                         if r.get("source-pattern", "").strip()]
        # the publisher's regional convention, for a source no ruling names
        # (modeller 2026-10-09): HHV for the US, Canada, the UK, Australia and
        # Japan; LHV for continental Europe, international bodies and elsewhere
        self.regional = {r["source-id"]: (r.get("default-basis") or "").strip().upper()
                         for r in read(REFERENCE / "source-region.csv")}

    def default_basis(self, *source_ids) -> str:
        """The basis a source reports on when a row does not say: 'LHV', 'HHV' or ''.
        A ruling in source-heating-basis.csv first (a pattern ending in `_` is a
        prefix, otherwise a whole source id), then the publisher's regional
        convention in source-region.csv."""
        return self.default(*source_ids)[0]

    def default(self, *source_ids):
        """(basis, 'source' | 'regional' | '')."""
        for sid in source_ids:
            sid = (sid or "").strip()
            if not sid:
                continue
            for pat, basis in self.defaults:
                if (sid.startswith(pat) if pat.endswith("_") else sid == pat):
                    return basis, "source"
        for sid in source_ids:
            b = self.regional.get((sid or "").strip(), "")
            if b:
                return b, "regional"
        return "", ""

    def _r(self, fid):
        if not fid:
            return 1.0
        f = self.factors.rows.get(fid)
        return fnum(f["factor"]) if f else None

    def to_hhv(self, archetype: str, how: str, pu: str, target: str = "", unit: str = "",
               example_id: str = ""):
        """(multiplier, [factor ids]) that puts an LHV number on HHV, or None when
        the archetype's streams are not recorded."""
        s = self.rows.get(archetype)
        if not s:
            # an energy use per tonne that names its carrier needs only that
            # carrier's ratio (added 2026-10-02)
            den = unit.split("/", 1)[1] if "/" in unit else ""
            t = (target or "").strip()
            if how == "energy-use" and t in CARRIER_RATIO and pu != "GJ" and den not in ENERGY:
                f = CARRIER_RATIO[t]
                r = self._r(f)
                return (r, [f] if f else []) if r is not None else None
            return None
        fin, fout = s.get("fuel-in", "").strip(), s.get("product-out", "").strip()
        if how == "energy-use" and (target or "").strip() in CARRIER_RATIO:
            fin = CARRIER_RATIO[target.strip()]
        if example_id in self.example_fuel:
            fin = self.example_fuel[example_id]
        rin, rout = self._r(fin), self._r(fout)
        if rin is None or rout is None:
            return None
        den = unit.split("/", 1)[1] if "/" in unit else ""
        # per unit of the product's ENERGY only when the source states it per energy
        # (or gives no denominator on an energy product): a figure per kg or per m3
        # of hydrogen is put per GJ with an HHV property factor, so the product's
        # LHV-to-HHV ratio must not be applied again (fixed 2026-10-13, sanity-check
        # follow-up: electrolyzer kWh/kg read 15% low)
        from .convert import atom
        den_dim = (atom(den.split("-per-")[0]) or ("",))[0] if den else ""
        per_energy = den in ENERGY or (pu == "GJ" and den_dim not in ("mass", "volume", "amount-h2"))
        m = {"efficiency": rout / rin,
             "energy-use": rin / (rout if per_energy else 1.0),
             "per-product": 1 / rout,
             "capacity-out": rout,
             "capacity-in": rin}[how]
        used = {"efficiency": [fout, fin], "energy-use": [fin] + ([fout] if per_energy else []),
                "per-product": [fout], "capacity-out": [fout], "capacity-in": [fin]}[how]
        return m, [u for u in used if u]
