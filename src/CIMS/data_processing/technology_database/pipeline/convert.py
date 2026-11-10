"""Unit and product conversions beyond the plain table in unit-conversion.csv.

Two kinds, both recorded on the row they were used for (column `conversion`):

  * units: a unit written as a numerator over a basis, the basis optionally a rate
    (`Btu/lb`, `2013_USD-thousand/m3-per-h`, `Nm3/h`), converted token by token
    with the definitions below (NIST SP 811 / Handbook 44). These are exact.
  * products: a physical property of a product that links two kinds of quantity
    (hydrogen's heating value links t and GJ; ethanol's density links m3 and t)
    or two products (alumina per tonne of aluminium). Each is a row of
    records/reference/conversion-factor.csv with its source. A factor applies to
    the archetypes named in its `applies-to`, or to the examples that
    records/reference/example-conversion.csv names; an example entry wins over an
    archetype one, and `none` there means the example is reported per a product
    the archetype does not make, so its per-product numbers are not pooled.

Design rule (design.md s5, amended 2026-09-26): a factor from another source is
used only through conversion-factor.csv, never inline. Each factor has a `class`
(CLASSES below) and a `source-id` registered in data/raw_data/technology_database/
source-details_reference.csv (or, for a factor a technology source prints about
its own product, in the main manifest). A conversion may chain at most two
factors (hydrogen in kW -> GJ/yr -> t/yr needs one; a mole of hydrogen -> GJ one).

Energy is on a higher-heating-value basis throughout (design.md s5, 2026-09-26).
"""
from __future__ import annotations

import re
from collections import defaultdict

from .common import REFERENCE, fnum, read, split_list

# token -> (dimension, size in the base unit: GJ, t, m3, yr)
ATOM = {
    "GJ": ("energy", 1.0), "MJ": ("energy", 1e-3), "kJ": ("energy", 1e-6),
    "TJ": ("energy", 1e3), "PJ": ("energy", 1e6), "kWh": ("energy", 0.0036),
    "MWh": ("energy", 3.6), "GWh": ("energy", 3600.0), "Btu": ("energy", 1.055056e-6),
    "kBtu": ("energy", 1.055056e-3), "MMBtu": ("energy", 1.055056), "mmBtu": ("energy", 1.055056),
    # MBtu as LBNL (Martin et al. 2000) and IEA print it means a MILLION Btu, the
    # same as MMBtu (sanity-check 2026-10-11, S108); kBtu is the thousand
    "MBtu": ("energy", 1.055056),
    "t": ("mass", 1.0), "kt": ("mass", 1e3), "Mt": ("mass", 1e6), "kg": ("mass", 1e-3),
    "g": ("mass", 1e-6), "lb": ("mass", 4.5359237e-4), "ton": ("mass", 0.90718474),
    "short-ton": ("mass", 0.90718474),
    # an air-dried tonne (90% dry) is the pulp industry's product tonne; the pulp
    # archetypes' "t pulp" is that tonne (modeller 2026-10-01)
    "ADt": ("mass", 1.0), "ADMT": ("mass", 1.0),
    # an oven-dry tonne of PULP is 1/0.9 air-dry tonnes (air-dry = 90% dry); named
    # per example as a unit alias (ODt=ODt-pulp), since an ODt of black liquor
    # solids or wood is not (2026-10-02)
    "ODt-pulp": ("mass", 1 / 0.9),
    "kcal": ("energy", 4.1868e-6), "Mcal": ("energy", 4.1868e-3), "Gcal": ("energy", 4.1868),
    # metres drilled: the drilling-rig archetypes' product (2026-10-21)
    "m": ("length", 1.0), "km": ("length", 1000.0),
    # a cubic metre moved one kilometre: the oil-pipeline pump stations' product
    # (S151, user 2026-10-28)
    "m3-km": ("volume-distance", 1.0),
    "m3": ("volume", 1.0), "Nm3": ("volume", 1.0), "l": ("volume", 1e-3), "L": ("volume", 1e-3),
    "Ml": ("volume", 1e3), "ML": ("volume", 1e3), "gal": ("volume", 0.003785411784),
    "bbl": ("volume", 0.158987), "barrel": ("volume", 0.158987), "kb": ("volume", 158.987), "kbbl": ("volume", 158.987),
    "bbl-beer": ("volume", 0.117347765),
    "1000-Nm3": ("volume", 1000.0),
    # a thousand cubic metres: the gas archetypes' product unit (user 2026-10-23);
    # e3m3 and e6m3 are the regulators' notation (AER, BCER, Petrinex, CER)
    "1000-m3": ("volume", 1e3), "e3m3": ("volume", 1e3), "e6m3": ("volume", 1e6),
    "Mgal": ("volume", 3785.411784), "Wh": ("energy", 3.6e-6),
    "million-Nm3-CH4": ("volume", 1e6), "thousand-Nm3-biomethane": ("volume", 1e3),
    "mol-H2": ("amount-h2", 1.0),
    # an hour of one operating device (pneumatic controller or pump); linked to the
    # gas a facility handles only through a named ratio factor (2026-10-03)
    "device-h": ("device-time", 1.0),
    # US gas-industry volumes: a standard cubic foot (60 degF, 14.73 psia), taken as
    # 0.0283168 m3 like Nm3 is taken as m3; Mcf = thousand, MMcf = million (2026-10-02)
    "scf": ("volume", 0.0283168), "cf": ("volume", 0.0283168), "Mcf": ("volume", 28.3168),
    "MMcf": ("volume", 28316.8), "Bcf": ("volume", 28316800.0),
    # a million standard cubic feet, as gas processors print it (2026-10-13, OG002)
    "MMscf": ("volume", 28316.8), "MMSCF": ("volume", 28316.8),
    # light delivered: lumen-hours, the lighting archetypes' product (ruling
    # 2026-10-14); Mlm-h is the base
    "Mlm-h": ("light", 1.0), "klm-h": ("light", 1e-3), "lm-h": ("light", 1e-6),
    "Glm-h": ("light", 1e3),
    "yr": ("time", 1.0), "a": ("time", 1.0), "h": ("time", 1 / 8760), "hr": ("time", 1 / 8760),
    "min": ("time", 1 / 525600),
    "d": ("time", 1 / 365), "day": ("time", 1 / 365),
}
# mass of a named species: `t-CO2` is a tonne of CO2. It meets a plain `t` only on
# an archetype whose product is that species.
SPECIES = re.compile(r"^(t|kg|kt|Mt)-(CO2(?:-eq)?|CO2e|CH4)$")
# a money unit may carry a scale: thousand, million, billion, or cent / mill
# (one hundredth and one thousandth of the currency unit; added 2026-10-05, before
# which "2003_USD-cent" was read as whole dollars)
MONEY = re.compile(r"^(\d{4})_([A-Z]{3})(?:-(thousand|million|billion|cent|cents|mill|mills))?")
POWER = {"kW": 0.0036, "MW": 3.6, "GW": 3600.0,      # GJ per hour
         "hp": 0.0026845,                            # 745.7 W (mechanical horsepower)
         "ton-refrigeration": 0.012660}              # 12,000 Btu/h = 3.51685 kW
MULT = {None: 1.0, "thousand": 1e3, "million": 1e6, "billion": 1e9,
        "cent": 1e-2, "cents": 1e-2, "mill": 1e-3, "mills": 1e-3}

# How much a converted number rests on, weakest last. `unit` is a definition
# (Btu, lb, barrel) or the currency method; `same-source` a factor the same source
# gives for the same example; `property` a physical property of a product;
# `ratio` a typical process ratio; `internal` a value pooled in this database
# from other examples; `assumption` a value chosen where no source fixes it.
# Anything from `property` on is EXTERNAL to the source's own numbers.
CLASSES = ["unit", "same-source", "property", "ratio", "internal", "assumption"]
EXTERNAL = set(CLASSES[2:])


def weakest(classes):
    cs = [c for c in classes if c in CLASSES]
    return max(cs, key=CLASSES.index) if cs else ""


def atom(tok, alias=None):
    tok = (alias or {}).get(tok, tok)
    m = SPECIES.match(tok)
    if m:
        return ATOM[m.group(1)][0], ATOM[m.group(1)][1], m.group(2).replace("CO2e", "CO2-eq")
    if tok in ATOM:
        return ATOM[tok][0], ATOM[tok][1], ""
    return None


def split_rate(s):
    """'m3-per-h' -> ('m3', 'h'); 'm3' -> ('m3', None)."""
    if "-per-" in s:
        a, b = s.split("-per-", 1)
        return a, b
    return s, None


class Factors:
    def __init__(self):
        self.rows = {r["factor-id"]: r for r in read(REFERENCE / "conversion-factor.csv")}
        self.cls = {k: (r.get("class") or "") for k, r in self.rows.items()}
        self.by_arch = defaultdict(list)
        for r in self.rows.values():
            r["_f"] = fnum(r["factor"])
            for a in split_list(r.get("applies-to", "")):
                self.by_arch[a].append(r)
        self.by_example = {}
        for r in read(REFERENCE / "example-conversion.csv"):
            self.by_example[r["example-id"]] = r

    def for_row(self, archetype, example_id):
        """(example entry or None, [factor rows]) -- example factors first."""
        e = self.by_example.get(example_id)
        out = []
        if e:
            out += [self.rows[f] for f in split_list(e.get("factor-id", "")) if f in self.rows]
        return e, out + self.by_arch.get(archetype, [])


def _edges(factors):
    """Each factor as an edge between two dimensions: (dim-from, dim-to, k, id),
    where one base unit of dim-from is k base units of dim-to; both directions."""
    out = []
    for f in factors:
        fu, tu = atom(f["from-unit"]), atom(f["to-unit"])
        if not fu or not tu or f.get("_f") in (None, 0) or fu[0] == tu[0]:
            continue
        k = f["_f"] * tu[1] / fu[1]
        out += [(fu[0], tu[0], k, f["factor-id"]), (tu[0], fu[0], 1 / k, f["factor-id"])]
    return out


def qty_ratio(x, tx, factors, species_ok, alias=None):
    """How many `tx` make one `x` (e.g. x='lb', tx='t' -> 4.536e-4), bridging two
    kinds of quantity with at most two product factors. Returns (ratio, [factor ids])."""
    a, b = atom(x, alias), atom(tx, alias)
    if not a or not b:
        return None
    if a[2] and not b[2] and not species_ok:
        return None
    if a[0] == b[0]:
        return a[1] / b[1], []
    edges = _edges(factors)
    paths = [(a[0], a[1], [])]
    for _ in range(2):
        nxt = []
        for dim, val, ids in paths:
            for d0, d1, k, fid in edges:
                if d0 != dim or fid in ids:
                    continue
                if d1 == b[0]:
                    return val * k / b[1], ids + [fid]
                nxt.append((d1, val * k, ids + [fid]))
        paths = nxt
    return None


def basis_factor(example_factors):
    """A same-kind product factor (1 t aluminium = 1.908 t alumina) named for the
    example: it changes which product the numbers are per, not the unit."""
    for f in example_factors:
        fu, tu = atom(f["from-unit"]), atom(f["to-unit"])
        if fu and tu and fu[0] == tu[0] and f["_f"]:
            return f["_f"] * fu[1] / tu[1], f["factor-id"]
    return None


def intensive(unit, tgt, factors, species_ok, alias=None):
    """A quantity per unit of product or capacity: numerator over basis. Money
    numerators are left to the caller (returns the basis part only when num is
    money). Returns (multiplier, [factor ids]) or None."""
    if "/" not in unit or "/" not in tgt:
        return None
    num, den = unit.split("/", 1)
    tnum, tden = tgt.split("/", 1)
    used = []
    if MONEY.match(num):
        nr = 1.0
    else:
        r = qty_ratio(num, tnum, factors, True, alias)
        if not r:
            return None
        nr = r[0]
        used += r[1]
    x, p = split_rate(den)
    tx, tp = split_rate(tden)
    if (p is None) != (tp is None):
        return None
    pr = 1.0
    if p:
        r = qty_ratio(tp, p, [], True)       # how many p in one tp
        if not r:
            return None
        pr = r[0]
    q = qty_ratio(x, tx, factors, species_ok, alias)
    if not q:
        return None
    used += q[1]
    # per (x/p) -> per (tx/tp): one x/p is q tx per p, i.e. q*pr tx per tp
    return nr / (q[0] * pr), used


def extensive(unit, tgt, factors, alias=None, annual=False):
    """A rate (a capacity): `Nm3/h` -> MW, `Ml/yr` -> kt/yr, `kb/d` -> m3/yr.
    `annual`: an annual output given as a bare quantity (`GWh`) is per year."""
    if tgt == "MW" and unit in POWER:
        return POWER[unit] / 3.6, []
    if annual and "/" not in unit:
        unit = unit + "/yr"
    if tgt == "MW":
        tgt, extra = "GJ/h", 1 / 3.6
    else:
        extra = 1.0
        if "/" not in tgt:              # annual output: the archetype's unit per year
            tgt += "/yr"
    if unit in POWER and tgt != "GJ/h":
        # a power rating in the product's energy (MW of methanol) as an annual
        # capacity at nameplate, put in product units with its heating value
        unit, extra = "GJ/h", extra * POWER[unit]
    if "/" not in unit or "/" not in tgt:
        return None
    num, per = unit.split("/", 1)
    tnum, tper = tgt.split("/", 1)
    q = qty_ratio(num, tnum, factors, True, alias)
    t = qty_ratio(tper, per, [], True)          # how many `per` in one `tper`
    if not q or not t:
        return None
    return q[0] * t[0] * extra, q[1]
