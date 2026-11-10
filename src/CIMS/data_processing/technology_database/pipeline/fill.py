"""The fill step: values for empty archetype cells, graded by how they were reached
(design.md s6a, 2026-09-30; widened 2026-10-09).

The summaries (layer 3) hold what the examples say. This step runs after them and
only ever writes a cell that has NO sourced value, so a new source displaces a
fill on the next build without anyone touching the fill. Examples and extractions
are never changed. The build also writes the summaries as they stood BEFORE this
step (output/unfilled/), so every check can be run with and without the fills.

Every archetype-parameter row carries a grade, best first:

  sourced    pooled from examples (reported, or derived within one source)
  identity   calculated from the archetype's own values by a definition or a
             physical relation (100 / efficiency; fuel x its emission factor)
  estimated  borrowed or scaled from other archetypes, or set by a class rule
             fitted on this database's own values
  proxy      a class constant or stand-in value from a named outside catalogue
             or handbook, not extracted as a source (records/fill/fill-assumption.csv)
  guess      a judgement with a stated range and rationale
             (records/fill/fill-judgement.csv)

and one marker that is not a value:

  not-applicable   the parameter does not characterise this archetype (the
                   energy use of a leak-detection programme, the variable O&M of
                   a boiler); written with its reason so the gap reads as closed
                   deliberately. The CIMS exporter writes nothing for it.

A fill takes the worst grade among its inputs and its rule. Rules are records,
not code: records/fill/fill-rule.csv names each one, its grade, scope and
rationale, and switches it on or off. The methods below are what a rule's
`method` can name. Rules run in file order, then the judgements, then the rules
once more, so a rule that needs a cell another rule or a judgement wrote (fixed
O&M from a guessed capex) finds it; it inherits that cell's grade.

A filled row has n = 0 (no example stands behind it), `central-from` saying how
it was reached, `inputs` naming the cells it was computed from with their grades,
and `sources` the sources behind those inputs.
"""
from __future__ import annotations

import re
from statistics import median

from .common import RECORDS, currency_year, fnum, read, split_list
from . import convert

FILL = RECORDS / "fill"
GRADES = ["sourced", "identity", "estimated", "proxy", "guess"]
MARKER = "not-applicable"
STATUSES = ("active", "off")
JUDGEMENT_STATUSES = ("draft", "accepted", "changed", "rejected", "active")
JUDGEMENT_LIVE = ("draft", "accepted", "changed", "active")
JUDGEMENT_BASES = ("analogue-archetype", "held-source-figure", "engineering-relation", "experience",
                   "added-carrier")
FOSSIL = {"natural-gas", "coal", "waste-fuel", "oil", "diesel", "gasoline", "coke", "petroleum-coke", "lpg", "naphtha",
          "fuel-oil", "heavy-fuel-oil", "refinery-gas", "still-gas", "coke-oven-gas", "blast-furnace-gas"}
CHP = re.compile(r"\bCHP\b")
SEC = "specific-energy-consumption"


def worst(grades):
    gs = [g for g in grades if g in GRADES]
    return max(gs, key=GRADES.index) if gs else ""


def _g(x):
    return f"{x:.6g}"


def _scope(rule):
    """{key: value} from a scope cell: `product-unit=GJ; exclude-equipment=a|b`."""
    out = {}
    for part in split_list(rule.get("scope", "")):
        k, _, v = part.partition("=")
        out[k.strip()] = v.strip()
    return out


def is_chp(a):
    return bool(CHP.search(a.get("label", ""))) or a["archetype"].startswith("combined-cycle")


def _in_scope(rule, a, have, ctx=None):
    sc = _scope(rule)
    if "product-unit" in sc and a["product-unit"] not in sc["product-unit"].split("|"):
        return False
    if a["equipment"] in sc.get("exclude-equipment", "").split("|"):
        return False
    if "equipment" in sc and a["equipment"] not in sc["equipment"].split("|"):
        return False
    if "type" in sc and a.get("type") not in sc["type"].split("|"):
        return False
    if "archetype" in sc and a["archetype"] not in sc["archetype"].split("|"):
        return False
    if a["archetype"] in sc.get("exclude-archetype", "").split("|"):
        return False
    if "family" in sc and a.get("family") not in sc["family"].split("|"):
        return False
    if "product" in sc and a.get("product") not in sc["product"].split("|"):
        return False
    if sc.get("exclude-feedstock") == "yes" and ctx is not None and _feedstock_row(a, ctx):
        return False
    if sc.get("exclude-ccs") == "yes" and ("_ccs-" in a["archetype"]
                                         or a.get("family") == "co2-capture"):
        return False
    if sc.get("exclude-chp") == "yes" and (is_chp(a) or any(k[0] == "efficiency-chp" for k in have)):
        return False
    if sc.get("only-ccs-host") == "yes" and not ("_ccs-" in a["archetype"]
                                                 and a.get("family") != "co2-capture"):
        return False
    if "modifier" in sc:
        mods = split_list(a.get("modifiers", ""))
        if not any(m in mods for m in sc["modifier"].split("|")):
            return False
    if "exclude-modifier" in sc and set(split_list(a.get("modifiers", ""))) & set(sc["exclude-modifier"].split("|")):
        return False
    if sc.get("no-modifier") == "yes" and split_list(a.get("modifiers", "")):
        return False
    if ctx is not None:
        if "end-use" in sc and not (set(sc["end-use"].split("|")) & ctx["service"].get(a["archetype"], set())):
            return False
        if "exclude-end-use" in sc and (set(sc["exclude-end-use"].split("|")) & ctx["service"].get(a["archetype"], set())):
            return False
        if "vom-class" in sc and ctx["classes"].get(a["equipment"], {}).get("vom-class") not in sc["vom-class"].split("|"):
            return False
        if "lifetime-class" in sc and ctx["classes"].get(a["equipment"], {}).get("lifetime-class") not in sc["lifetime-class"].split("|"):
            return False
    def _has(spec):
        # `parameter` or `parameter:target`
        p, _, t = spec.partition(":")
        return any(k[0] == p and (not t or k[1] == t) and _live(have[k]) for k in have)
    for spec in split_list(sc.get("require-absent", "").replace(",", ";")):
        if _has(spec):
            return False
    for spec in split_list(sc.get("require-present", "").replace(",", ";")):
        if not _has(spec):
            return False
    return True


def _feedstock_row(a, ctx):
    from . import feedstock
    return bool(feedstock.rows_for(a["archetype"], ctx.get("feedstock")))


def _live(r):
    """A row that holds a value (not a marker)."""
    return bool(r) and r.get("grade") != MARKER and fnum(r.get("central")) is not None


def _apart(r):
    """A carrier row the carriers step pooled apart: not part of the archetype's split,
    so it does not add up with the other carriers (pipeline/carriers.py)."""
    return "pooled apart" in (r.get("central-from") or "")


def _adding_carriers(rows):
    """{carrier: row} -> the carrier rows that add up to the archetype's energy use.
    Rows pooled apart are dropped (they do not sum with the split). One exception:
    where the unnamed `fuel` row has no named fuel beside it and exactly one fuel
    carrier was pooled apart, that carrier names the fuel: the `fuel` quantity is
    carried under its name (a heat-treatment furnace's fuel is natural gas)."""
    keep = {t: r for t, r in rows.items() if not _apart(r)}
    f = keep.get("fuel")
    if f is None:
        return keep
    fv = fnum(f.get("central")) or 0
    cov = sum(fnum(r.get("central")) or 0 for t, r in keep.items()
              if t != "fuel" and t not in NOT_FUEL)
    cand = [t for t, r in rows.items()
            if _apart(r) and t != "fuel" and t not in NON_COMBUSTIBLE and t not in keep]
    if fv > 0 and cov < 0.05 * fv and len(cand) == 1:
        r = dict(f)
        r["central-from"] = (f"the unnamed fuel ({f.get('central-from', '')}), named {cand[0]} "
                             f"by the archetype's pooled-apart {cand[0]} row")
        keep[cand[0]] = r
    return keep


def _vals(row):
    return fnum(row.get("central")), fnum(row.get("low")), fnum(row.get("high"))


def _input(row, label):
    return f"{label} ({row.get('grade') or 'sourced'})"


def _plausible(rule, v):
    sc = _scope(rule)
    if "plausible" not in sc:
        return True
    lo, _, hi = sc["plausible"].partition("|")
    return (fnum(lo) or -1e300) <= v <= (fnum(hi) or 1e300)


# ---- methods: (rule, archetype record, {(param, target): row}, ctx) -> [fill dicts]

COP_EQUIPMENT = {"heat-pump", "chiller", "vapour-recompressor", "refrigeration-unit"}


def cop_capable(a) -> bool:
    """Equipment that moves heat rather than converting energy, so an efficiency
    above 100% is a coefficient of performance (sanity-check 2026-10-11, S105)."""
    return (a.get("equipment") in COP_EQUIPMENT or "heat-pump" in a["archetype"]
            or "heat-pumped" in a["archetype"])


def m_inverse(rule, a, have, ctx):
    """p = 100 / source. With `require-consistent-total=yes` the source row (an
    all-carrier SEC total) must agree with its carrier split (pipeline/carriers.py
    marks a disagreement in central-from)."""
    p, t = rule["parameter"], rule["target"]
    src = split_list(rule["inputs"])[0]
    s = have.get((src, ""))
    if not _live(s) or (p, t) in have:
        return []
    if _scope(rule).get("require-consistent-total") == "yes" and "DISAGREES" in (s.get("central-from") or ""):
        return []
    c, lo, hi = _vals(s)
    if not c or c <= 0:
        return []
    k = 100.0
    if not _plausible(rule, k / c):
        return []
    if p == "efficiency" and k / c > 100 and not cop_capable(a):
        # less energy in than out on equipment that converts energy: the SEC counts
        # utilities, a parasitic load or an index, not the conversion input (S105)
        return []
    return [{"parameter": p, "target": t, "central": k / c,
             "low": k / hi if hi else None, "high": k / lo if lo else None,
             "inputs": [_input(s, src)], "grades": [s.get("grade", "sourced")],
             "sources": s.get("sources", ""), "external": []}]


def m_inverse_net_feed(rule, a, have, ctx):
    """On a conversion archetype (records/reference/feedstock.csv) measured per GJ of
    product: energy use = 100 / efficiency - the energy in the product (the feed
    whose energy ends in the product is feedstock-consumption, the rest is burned)
    (sanity-check 2026-10-23, OG003)."""
    from . import feedstock
    p = rule["parameter"]
    if any(k[0] == p and _live(v) for k, v in have.items()):
        return []
    t0 = feedstock.row_for(a["archetype"], ctx.get("feedstock"))
    s = have.get(("efficiency", ""))
    if not t0 or not _live(s) or a.get("product-unit") != "GJ":
        return []
    e = fnum(t0.get("energy-in-product"))
    c, lo, hi = _vals(s)
    if not e or not c or c <= 0 or c > 100:
        return []
    f = lambda x: None if not x else 100.0 / x - e
    v = f(c)
    if v is None or v <= 0:
        return []
    return [{"parameter": p, "target": "", "central": v, "low": f(hi), "high": f(lo),
             "inputs": [_input(s, "efficiency"), f"energy in product {e:g} GJ/GJ (feedstock.csv)"],
             "grades": [s.get("grade", "sourced")], "sources": s.get("sources", ""), "external": []}]


def m_chp_heat_inverse(rule, a, have, ctx):
    """A CHP archetype's energy use per GJ of heat (its product) = 100 / its heat
    efficiency: the efficiency row with target heat, else the total CHP efficiency
    less the electric efficiency (sanity-check run 2026-10-24, P-C6; biomass CHP had
    no energy use). Heat efficiency outside 15-80% is not used."""
    p = rule["parameter"]
    if not is_chp(a) or a.get("product-unit") != "GJ" or any(k[0] == p and _live(v) for k, v in have.items()):
        return []
    eh = have.get(("efficiency", "heat"))
    if _live(eh):
        h, hl, hh = _vals(eh)
        ins, grades = [_input(eh, "efficiency_heat")], [eh.get("grade", "sourced")]
    else:
        # the total is efficiency-chp: an untargeted efficiency row on a CHP archetype
        # is unspecified until the roll-in (pipeline/roll_in.py) (Fable review)
        et, ee = have.get(("efficiency-chp", "")), have.get(("efficiency", "electricity"))
        if not (_live(et) and _live(ee)):
            return []
        h = fnum(et["central"]) - fnum(ee["central"])
        hl = hh = None
        ins = [_input(et, "efficiency (total CHP)"), _input(ee, "efficiency_electricity")]
        grades = [et.get("grade", "sourced"), ee.get("grade", "sourced")]
    if h is None or not 15 <= h <= 80:
        return []
    return [{"parameter": p, "target": "", "central": 100.0 / h,
             "low": 100.0 / hh if hh else None, "high": 100.0 / hl if hl else None,
             "inputs": ins, "grades": grades, "sources": "", "external": []}]


def m_inverse_net_feed_efficiency(rule, a, have, ctx):
    """The inverse of id-sec-from-efficiency-net-feed: on a conversion archetype per
    GJ of product, efficiency = 100 / (energy use + the energy in the product)
    (sanity-check run 2026-10-24, P-C8). Not where the energy use itself came from
    the efficiency, nor on CHP."""
    from . import feedstock
    p = rule["parameter"]
    if (p, "") in have or a.get("product-unit") != "GJ" or is_chp(a):
        return []
    t0 = feedstock.row_for(a["archetype"], ctx.get("feedstock"))
    s = have.get((SEC, ""))
    hrow = ctx["hosts"].get(a["archetype"], {})
    if hrow.get("status") == "active" and hrow.get("host"):
        return []       # a variant takes its host's efficiency (est-sibling-efficiency)
    if not t0 or not _live(s) or (s.get("grade") or "sourced") != "sourced":
        return []
    e = fnum(t0.get("energy-in-product"))
    c, lo, hi = _vals(s)
    if not e or not c or c <= 0:
        return []
    v = 100.0 / (c + e)
    if not 10 <= v <= 100:
        return []
    return [{"parameter": p, "target": "", "central": v,
             "low": 100.0 / (hi + e) if hi else None, "high": 100.0 / (lo + e) if lo else None,
             "inputs": [_input(s, SEC), f"energy in product {e:g} GJ/GJ (feedstock.csv)"],
             "grades": [s.get("grade", "sourced")], "sources": s.get("sources", ""), "external": []}]


def m_fuel_emission_factor(rule, a, have, ctx):
    p, t = rule["parameter"], rule["target"]
    mid = None
    if (p, t) in have:
        # a range midpoint that implies a CO2 factor above every fuel the archetype
        # burns gives way to the fuel identity, where that sits inside the source's
        # range (Brad 2026-10-28, MCF014)
        mid = have[(p, t)]
        if not _live(mid) or mid.get("method") != "range-midpoint":
            return []
    secs = _adding_carriers({k[1]: v for k, v in have.items() if k[0] == SEC and k[1] and _live(v)})
    if "fuel" in secs:
        # `fuel` is known by fuel only where named fuels stand beside it: the carriers
        # split (pipeline/carriers.py, 2026-10-05), a single-carrier identity, a bucket
        # estimate or a judgement; then the carbon comes from the named fuels alone
        split = {c: r for c, r in secs.items() if c != "fuel" and c not in NON_COMBUSTIBLE}
        if not split:
            return []                       # an unnamed fuel: the carbon cannot be known
        secs = split
    fos = {c: r for c, r in secs.items() if c in FOSSIL}
    if not fos or any((fnum(r.get("central")) or 0) < 0 for r in fos.values()):
        return []   # a negative carrier is fuel saved or recovered, not burned
    c = lo = hi = 0.0
    ins, grades, srcs, ext = [], [], [], []
    for carrier, r in sorted(fos.items()):
        f = ctx["factors"].rows.get(f"cf-ef-co2-{carrier}")
        if not f:
            return []                       # a fossil carrier with no factor: incomplete
        ef, efl, efh = fnum(f["factor"]), fnum(f.get("factor-low")), fnum(f.get("factor-high"))
        v, vl, vh = _vals(r)
        if v is None:
            return []
        c += v * ef
        lo += (vl if vl is not None else v) * (efl or ef)
        hi += (vh if vh is not None else v) * (efh or ef)
        ins.append(_input(r, f"{SEC}_{carrier}"))
        grades.append(r.get("grade", "sourced"))
        srcs += split_list(r.get("sources", ""))
        ext.append(f["factor-id"])
    tot = have.get(("emissions-total", "co2"))
    pr = have.get(("emissions-process", "co2"))
    newly = (_live(pr) or a["equipment"] in ("steam-reformer", "reactor")) and "_ccs-" not in a["archetype"] \
        and a.get("family") != "co2-capture"
    if c <= 0 and newly:
        return []       # zero-valued (guessed) fuel rows say nothing about the carbon
    # the fuel's CO2 with the process CO2 must fit inside a sourced total: where it
    # does not, the energy rows and the total are on different scopes (2026-10-24,
    # sanity-check run P-C1); listed in the build log, not written. Only on the
    # archetypes the rule newly reaches (a process-CO2 row, a reformer or reactor);
    # a capture variant's total is net of capture
    if _live(tot) and (tot.get("grade") or "sourced") == "sourced" and "_ccs-" not in a["archetype"] \
            and a.get("family") != "co2-capture":
        tv = fnum(tot["central"]) or 0.0
        pv = (fnum(pr["central"]) or 0.0) if _live(pr) else 0.0
        if (tv <= 0 or c + pv > 1.1 * tv) and not newly:
            # written as before this rule was widened, but shown (Fable review)
            ctx.setdefault("comb_conflict_old", []).append(
                f"{a['archetype']}: fuel CO2 {c:.4g} + process {pv:.4g} against sourced total {tv:.4g}")
    if newly and _live(tot) and (tot.get("grade") or "sourced") == "sourced":
        tv = fnum(tot["central"]) or 0.0
        pv = (fnum(pr["central"]) or 0.0) if _live(pr) else 0.0
        if tv <= 0 or c + pv > 1.1 * tv:
            ctx.setdefault("comb_conflict", []).append(
                f"{a['archetype']}: fuel CO2 {c:.4g} + process {pv:.4g} against sourced total {tv:.4g}")
            return []
    out = {"parameter": p, "target": t, "central": c, "low": lo, "high": hi,
           "inputs": ins + ext, "grades": grades,
           "sources": "; ".join(dict.fromkeys(srcs)), "external": ext}
    if mid is not None:
        mv, ml, mh = _vals(mid)
        fuel = sum(_vals(r)[0] or 0.0 for r in fos.values())
        efmax = max(fnum(ctx["factors"].rows[f"cf-ef-co2-{k}"]["factor"]) for k in fos)
        if mv is None or ml is None or mh is None or fuel <= 0 or mv / fuel <= efmax or not (ml <= c <= mh):
            return []
        ctx.setdefault("total_below_parts", []).append(
            f"{a['archetype']} {t}: combustion midpoint {mv:.4g} (implies {mv / fuel:.3g} t/GJ, above every listed fuel) "
            f"replaced by the fuel identity {c:.4g}")
        out.update({"replace": mid, "inputs": ins + ext + [f"replaced range midpoint {mv:.4g} ({ml:.4g}-{mh:.4g})"]})
    return [out]


def m_biogenic_emission_factor(rule, a, have, ctx):
    """Biogenic CO2 (emissions-biomass, CIMS) = sum over the biogenic carriers the
    archetype burns of specific-energy-consumption(carrier) x its biogenic CO2
    factor (cf-ef-co2-<carrier>, IPCC 2006 Table 1.4), solid, liquid and gaseous
    alike (Brad 2026-11-09). Only where no source or judgement gives it; the rule's
    scope keeps out archetypes whose biomass is a feedstock (gasifiers to fuels,
    fermenters, reactors, the charcoal retort), whose carbon leaves in the product."""
    p, t = rule["parameter"], rule["target"]
    if (p, t) in have:
        return []
    secs = {k[1]: v for k, v in have.items() if k[0] == SEC and k[1] in BIOGENIC_ALL and _live(v)}
    if not secs or any((fnum(r.get("central")) or 0) < 0 for r in secs.values()):
        return []
    c = lo = hi = 0.0
    ins, grades, srcs, ext = [], [], [], []
    for carrier, r in sorted(secs.items()):
        f = ctx["factors"].rows.get(f"cf-ef-co2-{carrier}")
        if not f:
            return []
        ef, efl, efh = fnum(f["factor"]), fnum(f.get("factor-low")), fnum(f.get("factor-high"))
        v, vl, vh = _vals(r)
        if v is None:
            return []
        c += v * ef
        lo += (vl if vl is not None else v) * (efl or ef)
        hi += (vh if vh is not None else v) * (efh or ef)
        ins.append(_input(r, f"{SEC}_{carrier}"))
        grades.append(r.get("grade", "sourced"))
        srcs += split_list(r.get("sources", ""))
        ext.append(f["factor-id"])
    if c <= 0:
        return []
    return [{"parameter": p, "target": t, "central": c, "low": lo, "high": hi,
             "inputs": ins + ext, "grades": grades,
             "sources": "; ".join(dict.fromkeys(srcs)), "external": ext}]


def m_sec_from_emissions(rule, a, have, ctx):
    """The inverse of the rule above: the fossil energy use = combustion CO2 / the
    CO2 factor of the archetype's fossil carrier, where the archetype names ONE
    fossil carrier. Written as the total and as that carrier."""
    p = rule["parameter"]
    if any(k[0] == p and _live(v) for k, v in have.items()):
        return []
    # CO2 only: a CO2-eq carries CH4 and N2O, which the fuel's CO2 factor does not
    # (sanity-check 2026-10-23, RX002)
    e = have.get(("emissions-combustion", "co2"))
    if not _live(e) or have.get(("emissions-process", "co2")):
        return []
    fos = [c for c in split_list(a.get("carriers", "")) if c in FOSSIL]
    if len(fos) != 1:
        return []
    f = ctx["factors"].rows.get(f"cf-ef-co2-{fos[0]}")
    if not f or not fnum(f["factor"]):
        return []
    ef = fnum(f["factor"])
    c, lo, hi = _vals(e)
    if c is None or c <= 0:
        return []
    out = []
    for t in ("", fos[0], "fuel"):
        out.append({"parameter": p, "target": t, "central": c / ef,
                    "low": lo / ef if lo else None, "high": hi / ef if hi else None,
                    "inputs": [_input(e, "emissions-combustion_co2"), f["factor-id"]],
                    "grades": [e.get("grade", "sourced")],
                    "sources": e.get("sources", ""), "external": [f["factor-id"]],
                    "note": "" if not t else "the archetype's one fossil carrier" if t != "fuel" else "all fuel"})
    return out


def m_sum(rule, a, have, ctx):
    p = rule["parameter"]
    parts = split_list(rule["inputs"])
    species = {k[1] for k in have if k[0] in parts and k[1]}
    out = []
    for t in sorted(species):
        if rule["target"] not in ("*", t):
            continue
        if (p, t) in have:
            r = _below_parts(p, t, parts, a, have, ctx, _scope(rule))
            if r:
                out.append(r)
            continue
        rows = [have.get((q, t)) for q in parts]
        if not all(_live(r) for r in rows):
            continue
        c = sum(fnum(r["central"]) for r in rows)
        lo = sum(fnum(r.get("low")) if fnum(r.get("low")) is not None else fnum(r["central"]) for r in rows)
        hi = sum(fnum(r.get("high")) if fnum(r.get("high")) is not None else fnum(r["central"]) for r in rows)
        out.append({"parameter": p, "target": t, "central": c, "low": lo, "high": hi,
                    "inputs": [_input(r, f"{q}_{t}") for q, r in zip(parts, rows)],
                    "grades": [r.get("grade", "sourced") for r in rows],
                    "sources": "; ".join(dict.fromkeys(s for r in rows for s in split_list(r.get("sources", "")))),
                    "external": []})
    return out


BELOW_PARTS_GRADES = ("sourced", "identity", "estimated")


def _below_parts(p, t, parts, a, have, ctx, sc=None):
    """A sourced total below its own parts (combustion + process, same species) is on
    another scope: the parts replace it where they are sourced, identities or
    estimates, the combustion part is present and the sum is more than 10% above the
    total (Brad 2026-10-28, IS009/MCF011). An absent process row counts as zero; a
    capture variant's total is net of capture and is left alone."""
    sc = sc or {}
    tot = have.get((p, t))
    if not _live(tot) or (tot.get("grade") or "sourced") != "sourced" or tot.get("method") not in (None, "", "pooled"):
        return None
    if "_ccs-" in a["archetype"] or a.get("family") == "co2-capture" or _live(have.get(("emissions-captured", t))):
        return None                       # a total net of capture
    if a["archetype"] in split_list(sc.get("below-parts-exclude", "").replace("|", ";")):
        return None                       # parts on another basis, ruled on (see the rule's rationale)
    rows = [have.get((q, t)) for q in parts]
    present = [(q, r) for q, r in zip(parts, rows) if _live(r)]
    if not present or present[0][0] != parts[0]:
        return None                       # no combustion part
    if any((r.get("grade") or "sourced") not in BELOW_PARTS_GRADES for _, r in present):
        return None
    tv = fnum(tot["central"])
    c = sum(fnum(r["central"]) or 0.0 for _, r in present)
    if tv is None or tv <= 0 or c <= 1.1 * tv:
        return None
    lo = sum(fnum(r.get("low")) if fnum(r.get("low")) is not None else fnum(r["central"]) for _, r in present)
    hi = sum(fnum(r.get("high")) if fnum(r.get("high")) is not None else fnum(r["central"]) for _, r in present)
    ctx.setdefault("total_below_parts", []).append(
        f"{a['archetype']} {t}: sourced total {tv:.4g} replaced by the parts {c:.4g}")
    return {"parameter": p, "target": t, "central": c, "low": lo, "high": hi, "replace": tot,
            "inputs": [_input(r, f"{q}_{t}") for q, r in present] + [f"replaced sourced total {tv:.4g}"],
            "grades": ["identity"] + [r.get("grade", "sourced") for _, r in present],
            "sources": "; ".join(dict.fromkeys(s for _, r in present for s in split_list(r.get("sources", "")))),
            "external": []}


def m_difference(rule, a, have, ctx):
    """p = first input - second input (opex-variable = opex-total - opex-fixed),
    only where the result is positive."""
    p = rule["parameter"]
    if (p, "") in have:
        return []
    x, y = (have.get((q, "")) for q in split_list(rule["inputs"])[:2])
    if not (_live(x) and _live(y)):
        return []
    xc, xl, xh = _vals(x)
    yc, yl, yh = _vals(y)
    c = xc - yc
    if c <= 0:
        return []
    lo = (xl if xl is not None else xc) - (yh if yh is not None else yc)
    hi = (xh if xh is not None else xc) - (yl if yl is not None else yc)
    return [{"parameter": p, "target": "", "central": c, "low": max(lo, 0.0), "high": max(hi, c),
             "inputs": [_input(x, split_list(rule["inputs"])[0]), _input(y, split_list(rule["inputs"])[1])],
             "grades": [x.get("grade", "sourced"), y.get("grade", "sourced")],
             "sources": "; ".join(dict.fromkeys(split_list(x.get("sources", "")) + split_list(y.get("sources", "")))),
             "external": []}]


# capacity (MW | kt/yr | pu/yr) x capacity factor x hours -> annual output in
# product units: MW x 8760 h x 3.6 GJ/MWh; kt/yr x 1000
OUTPUT_K = {"GJ": 8760 * 3.6, "t": 1000.0}


def m_capacity_output(rule, a, have, ctx):
    p = rule["parameter"]
    if (p, "") in have:
        return []
    cap, cf = have.get(("capacity", "")), have.get(("capacity-factor", ""))
    if not (_live(cap) and _live(cf)):
        return []
    k = OUTPUT_K.get(a["product-unit"], 1.0)
    cc, cl, ch = _vals(cap)
    fc, fl, fh = _vals(cf)
    c = cc * fc / 100 * k
    lo = (cl if cl is not None else cc) * (fl if fl is not None else fc) / 100 * k
    hi = (ch if ch is not None else cc) * (fh if fh is not None else fc) / 100 * k
    return [{"parameter": p, "target": "", "central": c, "low": lo, "high": hi,
             "inputs": [_input(cap, "capacity"), _input(cf, "capacity-factor"),
                        f"x {k:g} ({cap.get('unit')} to {a['product-unit']}/yr)"],
             "grades": [cap.get("grade", "sourced"), cf.get("grade", "sourced")],
             "sources": "; ".join(dict.fromkeys(split_list(cap.get("sources", "")) + split_list(cf.get("sources", "")))),
             "external": []}]


def _base_of(a, ctx):
    """The archetype less its last modifier, if that archetype exists."""
    parts = a["archetype"].split("_")
    if len(parts) < 2 or parts[-1] not in ctx["modifiers"]:
        return None
    base = "_".join(parts[:-1])
    return base if base in ctx["arch"] else None


# the highest efficiency a device of this equipment can reach (per cent, the
# database's heating-value basis); a saving never carries a variant past it (S142)
EFF_MAX = {"motor": 97.0, "drive": 97.0, "pump": 92.0, "fan": 92.0, "compressor": 92.0}
EFF_MAX_DEFAULT = 97.0     # a condensing fuel-fired heater on the HHV basis


def m_bucket_saving(rule, a, have, ctx):
    """A variant's energy use = its base archetype's x (1 - s). s is the variant's
    own sourced emissions-reduction-potential (same-fuel measure: the emission
    saving is the energy saving), else the bucket's definition from the assumption
    group the rule names (proxy). Written for the total and every carrier the base
    has, scaled alike, so the carrier split still sums."""
    p = rule["parameter"]
    if any(k[0] == p and _live(v) for k, v in have.items()):
        return []
    base = _base_of(a, ctx)
    if not base:
        return []
    bh = ctx["have"].get(base, {})
    tot = bh.get((p, ""))
    if not _live(tot):
        return []
    erp = have.get(("emissions-reduction-potential", ""))
    if _live(erp) and 0 < fnum(erp["central"]) < 90:
        s, sl, sh = _vals(erp)
        sl = s if sl is None else sl
        sh = s if sh is None else sh
        sl, sh = min(sl, sh, 89.0), min(max(sl, sh), 89.0)
        src = _input(erp, "emissions-reduction-potential")
        grades = [erp.get("grade", "sourced")]
        ext = []
    else:
        row = _assumption(rule, a, ctx)
        if not row:
            return []
        s, sl, sh = fnum(row["value"]), fnum(row.get("low")), fnum(row.get("high"))
        sl = s if sl is None else sl
        sh = s if sh is None else sh
        src = f"assumption {row['assumption-id']} (proxy)"
        grades = ["proxy"]
        ext = [row["assumption-id"]]
    # a device cannot pass its thermodynamic limit: on a GJ-output archetype whose
    # base converts energy at an efficiency of 100% or less, the saving is capped
    # where the variant's efficiency would reach EFF_MAX (sanity-check 2026-10-23,
    # S142/HPCH006, RX005: a UKERC system-level saving gave a solid-fuel heater
    # 127% and a motor 98.9%)
    be = bh.get(("efficiency", ""))
    floor = None
    if a.get("product-unit") == "GJ" and _live(be) and 0 < fnum(be["central"]) <= 100 \
            and not cop_capable(a):
        emax = EFF_MAX.get(a["equipment"], EFF_MAX_DEFAULT)
        smax = max(0.0, (1 - fnum(be["central"]) / emax) * 100)
        if s > smax:
            src += f"; saving {s:.3g}% capped at {smax:.3g}% (efficiency {fnum(be['central']):.3g}% to at most {emax:g}%)"
        s, sl, sh = min(s, smax), min(sl, smax), min(sh, smax)
        floor = 100.0 / emax
    out = []
    for (q, t), r in sorted(bh.items()):
        if q != p or not _live(r) or (t and _apart(r)):
            continue    # a carrier pooled apart is not part of the base's total (2026-10-24)
        c, lo, hi = _vals(r)
        lo = c if lo is None else lo
        hi = c if hi is None else hi
        low = min(lo, hi) * (1 - sh / 100)
        if floor is not None and not t:
            low = max(low, min(floor, c * (1 - s / 100)))
        out.append({"parameter": p, "target": t, "central": c * (1 - s / 100),
                    "low": low, "high": max(lo, hi) * (1 - sl / 100),
                    "inputs": [_input(r, f"{base} {p}" + (f"_{t}" if t else "")), src],
                    "grades": grades + [r.get("grade", "sourced")],
                    "sources": r.get("sources", ""), "external": ext})
    return out


def m_host_increment(rule, a, have, ctx):
    """Host plant's pooled cost + the median increment the set-aside examples give."""
    p = rule["parameter"]
    if (p, "") in have:
        return []
    hm = ctx["hosts"].get(a["archetype"])
    if not hm or hm.get("status") != "active" or not hm.get("host"):
        return []
    host = ctx["have"].get(hm["host"], {}).get((p, ""))
    if not _live(host):
        return []
    by_ex = {}
    for r in ctx["increments"].get((a["archetype"], p), []):
        if r.get("unit-norm") != host.get("unit"):
            continue
        by_ex.setdefault(r["example-id"], []).append(fnum(r["value-norm"]))
    if not by_ex:
        return []
    per = sorted(median(v) for v in by_ex.values())
    inc, ilo, ihi = median(per), per[0], per[-1]
    hc, hl, hh = _vals(host)
    hl = hc if hl is None else hl
    hh = hc if hh is None else hh
    srcs = split_list(host.get("sources", "")) + sorted(
        {r["source-id"] for rs in [ctx["increments"][(a["archetype"], p)]] for r in rs})
    return [{"parameter": p, "target": "", "central": hc + inc, "low": hl + ilo, "high": hh + ihi,
             "inputs": [_input(host, f"{hm['host']} {p}"),
                        f"increment median of {len(per)} example(s): "
                        + ", ".join(sorted(by_ex))],
             "grades": [host.get("grade", "sourced")],
             "sources": "; ".join(dict.fromkeys(srcs)), "external": []}]


def m_host_copy(rule, a, have, ctx):
    """The host plant's cost, unchanged: a host-with-capture archetype differs from
    its host only by a service request to the capture archetype, which carries the
    capture cost (modeller 2026-10-02)."""
    p = rule["parameter"]
    if (p, "") in have:
        return []
    hm = ctx["hosts"].get(a["archetype"])
    if not hm or hm.get("status") != "active" or not hm.get("host"):
        return []
    host = ctx["have"].get(hm["host"], {}).get((p, ""))
    if not _live(host):
        return []
    hc, hl, hh = _vals(host)
    return [{"parameter": p, "target": "", "central": hc,
             "low": hc if hl is None else hl, "high": hc if hh is None else hh,
             "inputs": [_input(host, f"{hm['host']} {p}")],
             "grades": [host.get("grade", "sourced")],
             "sources": host.get("sources", ""), "external": []}]


def m_bucket_increment(rule, a, have, ctx):
    """A variant with a priced host but no increment of its own: host cost x
    (1 + i), i the median increment/host ratio this database's own set-aside
    measure costs give for the same modifier (pool of `min-pool` or more); the
    range from the pool's quartiles."""
    p = rule["parameter"]
    if (p, "") in have:
        return []
    base = _base_of(a, ctx)
    if not base:
        return []
    host = ctx["have"].get(base, {}).get((p, ""))
    if not _live(host):
        return []
    mod = a["archetype"].split("_")[-1]
    pool = ctx["inc_ratios"].get((mod, p), {})
    pool = {k: v for k, v in pool.items() if k != a["archetype"]}
    need = int(_scope(rule).get("min-pool", "3"))
    if len(pool) < need:
        return []
    vs = sorted(pool.values())
    i, ilo, ihi = median(vs), _quantile(vs, 0.25), _quantile(vs, 0.75)
    hc, hl, hh = _vals(host)
    hl = hc if hl is None else hl
    hh = hc if hh is None else hh
    return [{"parameter": p, "target": "", "central": hc * (1 + i), "low": hl * (1 + ilo),
             "high": hh * (1 + ihi),
             "inputs": [_input(host, f"{base} {p}"),
                        f"_{mod} increment/host median {i:.3g} (quartiles {ilo:.3g}-{ihi:.3g}) "
                        f"of {len(pool)} archetypes: " + ", ".join(sorted(pool))],
             "grades": [host.get("grade", "sourced")],
             "sources": host.get("sources", ""), "external": []}]


def m_per_gj_via_sec(rule, a, have, ctx):
    """A capex a source gives per GJ-yr of heat output on a per-tonne archetype
    (excluded at normalise for want of a same-example GJ/t) goes per t-yr with
    the archetype's own pooled energy use (class internal), under the same-GJ
    condition of design s5: same stream, same side, same scope (modeller
    2026-10-09, graded estimated). One value per example, then the median."""
    p = rule["parameter"]
    if (p, "") in have or a["product-unit"] != "t":
        return []
    sec = have.get((SEC, ""))
    if not _live(sec):
        return []
    rows = ctx["per_gj"].get((a["archetype"], p), [])
    if not rows:
        return []
    by_ex = {}
    for r in rows:
        f = ctx["units"].cur.get(currency_year(r["unit"])) if ctx.get("units") else None
        if f is None:
            continue
        v = fnum(r.get("value"))
        if v is None:
            continue
        by_ex.setdefault(r["example-id"], []).append(v * f)
    if not by_ex:
        return []
    per = sorted(median(v) for v in by_ex.values())
    c_gj, lo_gj, hi_gj = median(per), per[0], per[-1]
    sc, sl, sh = _vals(sec)
    sl = sc if sl is None else sl
    sh = sc if sh is None else sh
    return [{"parameter": p, "target": "", "central": c_gj * sc, "low": lo_gj * sl, "high": hi_gj * sh,
             "inputs": [f"capex per GJ-yr of heat output, median of {len(per)} example(s): "
                        + ", ".join(sorted(by_ex)), _input(sec, f"{SEC} (GJ/t, class internal)")],
             "grades": [sec.get("grade", "sourced")],
             "sources": "; ".join(dict.fromkeys(sorted({r["source-id"] for r in rows})
                                                + split_list(sec.get("sources", "")))),
             "external": [f"sec:{a['archetype']}"]}]


NOT_FUEL = {"electricity", "steam", "heat", "waste-heat", "geothermal", "solar"}   # never part of `fuel`
NON_COMBUSTIBLE = {"electricity", "steam", "heat", "geothermal", "hydrogen", "nuclear", "solar", "waste-heat", "ammonia"}
# biogenic fuels: their combustion CO2 is not fossil CO2 (CIMS counts it as emissions-biomass)
BIOGENIC = {"biomass", "biogas", "biomethane", "black-liquor", "landfill-gas", "liquid-biomass", "renewable-natural-gas"}
BIOGENIC_ALL = BIOGENIC | {"biomass-solid"}


def m_zero_if_no_fossil(rule, a, have, ctx):
    """Combustion CO2 = 0 where the archetype burns nothing with carbon: every
    carrier it lists is non-combustible (electricity, steam, hydrogen, ...) and
    no energy-use row names a combustible one."""
    p, t = rule["parameter"], rule["target"]
    if (p, t) in have:
        return []
    carriers = set(split_list(a.get("carriers", "")))
    if not carriers or not carriers <= NON_COMBUSTIBLE | BIOGENIC:
        return []
    if any(k[0] == SEC and k[1] and not is_component_target(k[1]) and k[1] not in NON_COMBUSTIBLE | BIOGENIC
           and _live(v) for k, v in have.items()):
        return []
    # a fossil energy row the run writes later (a judgement, or a variant's rows
    # copied from its host) counts too: look at the judgements and at the host
    # (sanity-check 2026-10-23, S141)
    burns = lambda t: t and t not in NON_COMBUSTIBLE | BIOGENIC and t != "fuel" or t in FOSSIL
    if any(j["archetype"] == a["archetype"] and j["parameter"] == SEC and burns(j.get("target", ""))
           and (j.get("status") or "") != "rejected" for j in judgements()):
        return []
    host = (ctx.get("hosts", {}).get(a["archetype"]) or {}).get("host")
    if host:
        hh = ctx["have"].get(host, {})
        ha = ctx["arch"].get(host, {})
        if any(k[0] == SEC and burns(k[1]) and _live(v) for k, v in hh.items()) or \
                not set(split_list(ha.get("carriers", ""))) <= NON_COMBUSTIBLE | BIOGENIC:
            return []
    bio = carriers & BIOGENIC
    return [{"parameter": p, "target": t, "central": 0.0, "low": None, "high": None,
             "inputs": ["carriers: " + ", ".join(sorted(carriers))], "grades": [],
             "sources": "", "external": [],
             "note": "no fossil carrier" + ("; biogenic CO2 from " + ", ".join(sorted(bio)) + " is not fossil CO2 (CIMS emissions-biomass)" if bio else "")}]


def m_range_midpoint(rule, a, have, ctx):
    """A sourced row that gives a range and no central (the source printed a
    range only) gets the midpoint of that range as its central; the row keeps its
    sources and n, and its grade becomes the rule's. The rule's parameter `*`
    means every parameter."""
    out = []
    for (p, t), r in list(have.items()):
        if rule["parameter"] not in ("*", p) or r.get("grade") != "sourced":
            continue
        c, lo, hi = _vals(r)
        if c is not None or lo is None or hi is None:
            continue
        out.append({"parameter": p, "target": t, "central": (lo + hi) / 2, "low": lo, "high": hi,
                    "inputs": [f"sourced range {lo:g}-{hi:g} ({r.get('central-from')})"],
                    "grades": [], "sources": r.get("sources", ""), "external": [],
                    "replace": r})
    return out


def m_single_carrier_split(rule, a, have, ctx):
    """An archetype that lists ONE carrier and has an all-carrier energy use but
    no carrier row: that carrier carries all of it."""
    p = rule["parameter"]
    carriers = split_list(a.get("carriers", ""))
    if len(carriers) != 1 or (p, carriers[0]) in have:
        return []
    named = [k[1] for k in have if k[0] == p and k[1] and k[1] not in ("fuel", "total")]
    if named:
        return []
    # a variant whose host splits its energy over two or more carriers takes the
    # host's mix (est-variant-host-split), not one listed carrier (sanity-check
    # 2026-10-23, S135/PB004: the strong paper machine had lost its electricity)
    k = a["archetype"]
    hrow = ctx["hosts"].get(k, {})
    host = hrow.get("host") if hrow.get("status") == "active" else None
    if host and "_ccs-" not in k:
        hh = ctx["have"].get(host, {})
        if sum(1 for (hp, ht), r in hh.items() if hp == p and ht and ht not in ("fuel", "total")
               and _live(r) and not _apart(r)) >= 2:
            return []
    # an unnamed `fuel` row with one listed carrier: that carrier is the fuel;
    # else the archetype's total is all that carrier
    src, note = have.get((p, "fuel")), "all of fuel, split from the carriers list"
    if not _live(src):
        src, note = have.get((p, "")), "the only carrier listed"
    if not _live(src):
        return []
    c, lo, hi = _vals(src)
    return [{"parameter": p, "target": carriers[0], "central": c, "low": lo, "high": hi,
             "inputs": [_input(src, p + ("_fuel" if note.startswith("all of fuel") else "")),
                        f"carriers: {carriers[0]}"], "note": note,
             "grades": [src.get("grade", "sourced")], "sources": src.get("sources", ""), "external": []}]


def m_host_split(rule, a, have, ctx):
    """A variant's carrier split = its host's split scaled to the variant's own
    total (sanity-check 2026-10-11, IS009/IS010): an efficiency measure uses less
    of the same mix, it does not change the mix. Only where the variant gives no
    carrier of its own and its host gives a split that adds up."""
    k = a["archetype"]
    hrow = ctx["hosts"].get(k, {})
    host = hrow.get("host") if hrow.get("status") == "active" else None
    if not host or "_ccs-" in k or host not in ctx["arch"]:
        return []
    if rule["parameter"] != SEC:
        # combustion CO2 scales with the energy used, the mix unchanged: the host's
        # combustion CO2 x (variant total / host total). Not process CO2, which is
        # per tonne of product whatever the efficiency.
        p, t = rule["parameter"], rule["target"]
        hh = ctx["have"].get(host, {})
        e, vt, ht = hh.get((p, t)), have.get((SEC, "")), hh.get((SEC, ""))
        if (p, t) in have or not (_live(e) and _live(vt) and _live(ht)) or fnum(ht["central"]) <= 0:
            return []
        ratio = fnum(vt["central"]) / fnum(ht["central"])
        c, lo, hi = _vals(e)
        return [{"parameter": p, "target": t, "central": c * ratio,
                 "low": lo * ratio if lo is not None else None,
                 "high": hi * ratio if hi is not None else None,
                 "inputs": [_input(e, f"{host} {p}_{t}"), _input(vt, f"{k} {SEC}"), _input(ht, f"{host} {SEC}")],
                 "grades": [e.get("grade", "sourced"), vt.get("grade", "sourced")],
                 "sources": e.get("sources", ""), "external": []}]
    tot = have.get((SEC, ""))
    if not _live(tot) or fnum(tot["central"]) <= 0:
        return []
    own = {t: r for (q, t), r in have.items() if q == SEC and t and not is_component_target(t) and _live(r)}
    if own:
        # the variant's own carriers stand; but where its fuel is unnamed (a 'fuel' row
        # and no named fuel), the host's named fuels name it, in the host's shares of
        # its fuel (2026-10-24, sanity-check run batch H: the reverberatory furnace's
        # strong variant had lost its natural gas once its host was named)
        hh = ctx["have"].get(host, {})
        vf = own.get("fuel")
        if vf is None or any(t != "fuel" and t not in NON_COMBUSTIBLE for t in own):
            return []
        hf = hh.get((SEC, "fuel"))
        named = {t: r for (q, t), r in hh.items() if q == SEC and t and t != "fuel"
                 and not is_component_target(t) and t not in NON_COMBUSTIBLE and _live(r)
                 and "pooled apart" not in (r.get("central-from") or "")}
        if not (_live(hf) and named) or fnum(hf["central"]) <= 0:
            return []
        hfc = fnum(hf["central"])
        if abs(sum(fnum(r["central"]) for r in named.values()) / hfc - 1) > 0.05:
            return []
        out = []
        for t, r in sorted(named.items()):
            sh = fnum(r["central"]) / hfc
            c, lo, hi = _vals(vf)
            out.append({"parameter": SEC, "target": t, "central": c * sh,
                        "low": lo * sh if lo is not None else None,
                        "high": hi * sh if hi is not None else None,
                        "inputs": [_input(vf, f"{k} {SEC}_fuel"), _input(r, f"{host} {SEC}_{t}"),
                                   _input(hf, f"{host} {SEC}_fuel")],
                        "grades": [vf.get("grade", "sourced"), r.get("grade", "sourced")],
                        "sources": r.get("sources", ""), "external": [],
                        "note": f"names the variant's fuel by {host}'s share {sh:.1%} of its fuel"})
        return out
    hh = ctx["have"].get(host, {})
    htot = hh.get((SEC, ""))
    if not _live(htot) or fnum(htot["central"]) <= 0:
        return []
    split = _adding_carriers({t: r for (q, t), r in hh.items()
                              if q == SEC and t and not is_component_target(t) and _live(r)})
    if not split or any((fnum(r["central"]) or 0) < 0 for r in split.values()):
        return []
    ratio = fnum(tot["central"]) / fnum(htot["central"])
    out = []
    for t, r in sorted(split.items()):
        c, lo, hi = _vals(r)
        out.append({"parameter": SEC, "target": t, "central": c * ratio,
                    "low": lo * ratio if lo is not None else None,
                    "high": hi * ratio if hi is not None else None,
                    "inputs": [_input(r, f"{host} {SEC}_{t}"), _input(tot, f"{k} {SEC}"),
                               _input(htot, f"{host} {SEC}")],
                    "grades": [r.get("grade", "sourced"), tot.get("grade", "sourced")],
                    "sources": r.get("sources", ""), "external": [],
                    "note": f"{host}'s share {c / fnum(htot['central']):.1%} of its total"})
    return out


def m_feedstock_identity(rule, a, have, ctx):
    """feedstock-consumption = the energy the product holds, on a conversion
    archetype (records/reference/feedstock.csv) whose sources leave the feed out
    (ruling 2026-10-13): the feed is at least what ends in the product."""
    from . import feedstock
    t = feedstock.row_for(a["archetype"], ctx["feedstock"])
    if not t or any(q == feedstock.FEED and _live(r) for (q, _), r in have.items()):
        return []
    e = fnum(t["energy-in-product"])
    carrier = t["feedstock"].split("|")[0]
    return [{"parameter": feedstock.FEED, "target": carrier, "central": e, "low": None, "high": None,
             "inputs": [f"energy in the product {e:g} GJ per {a['product-unit']} (records/reference/feedstock.csv)"],
             "grades": ["identity"], "sources": "", "external": []}]


def m_co2e_from_species(rule, a, have, ctx):
    """CO2-eq = CO2 + the other gases x their AR5 100-year GWP (ruling 2026-10-14),
    for one emissions parameter, where a non-CO2 gas is known and CO2-eq is not.
    On abatement cost: the cost per t CO2-eq = the cost per t of the one gas abated
    / its GWP."""
    from . import gwp
    p = rule["parameter"]
    if (p, "co2e") in have:
        return []
    G = gwp.ar5(ctx["factors"])
    if p == "abatement-cost":
        gs = [(sp, have.get((p, sp))) for sp in G if _live(have.get((p, sp)) or {})]
        if len(gs) != 1:
            return []
        sp, r = gs[0]
        c, lo, hi = _vals(r)
        k = 1 / G[sp]
        return [{"parameter": p, "target": "co2e", "central": c * k,
                 "low": lo * k if lo is not None else None, "high": hi * k if hi is not None else None,
                 "inputs": [_input(r, f"{p}_{sp}"), f"cf-gwp100-ar5-{sp}"], "grades": [r.get("grade", "sourced")],
                 "sources": r.get("sources", ""), "external": [f"cf-gwp100-ar5-{sp}"]}]
    parts = [(sp, have.get((p, sp))) for sp in ("co2",) + tuple(G) if _live(have.get((p, sp)) or {})]
    if not any(sp != "co2" for sp, _ in parts):
        return []
    # CO2-eq without the CO2 is not a CO2-eq: wait for the CO2 (a fill may write it
    # later in the run; the rules run twice) (sanity-check 2026-10-23, RX003)
    if p == "emissions-combustion" and "co2" not in [sp for sp, _ in parts] and (p, "co2") not in have:
        return []
    c = lo = hi = 0.0
    ins, grades, srcs, ext = [], [], [], []
    for sp, r in parts:
        k = 1.0 if sp == "co2" else G[sp]
        v, vl, vh = _vals(r)
        c += v * k
        lo += (vl if vl is not None else v) * k
        hi += (vh if vh is not None else v) * k
        ins.append(_input(r, f"{p}_{sp}"))
        grades.append(r.get("grade", "sourced"))
        srcs += split_list(r.get("sources", ""))
        if sp != "co2":
            ext.append(f"cf-gwp100-ar5-{sp}")
    return [{"parameter": p, "target": "co2e", "central": c, "low": lo, "high": hi,
             "inputs": ins + ext, "grades": grades, "sources": "; ".join(dict.fromkeys(srcs)),
             "external": ext}]


def m_marker(rule, a, have, ctx):
    """Not a value: the parameter does not characterise this archetype. Written
    for the targets the rule names (`*` = the parameter with no target)."""
    p = rule["parameter"]
    if any(k[0] == p for k in have):
        return []
    return [{"parameter": p, "target": "", "marker": True, "central": None, "low": None, "high": None,
             "inputs": [], "grades": [], "sources": "", "external": []}]


def m_constant(rule, a, have, ctx):
    """A value fixed by definition (the rule's formula is the number): written
    only where the archetype has no row of the parameter. Added 2026-10-24
    (sanity-check run P-D4, S91): a methane control's capacity is the methane it
    avoids in a year, so its capacity factor is 100% by definition."""
    p = rule["parameter"]
    if (p, rule.get("target") or "") in have:
        return []
    v = fnum((rule.get("formula") or "").strip())
    if v is None:
        return []
    return [{"parameter": p, "target": rule.get("target") or "", "central": v, "low": v, "high": v,
             "inputs": [f"by definition: {v:g}"], "grades": [], "sources": "", "external": []}]


def m_host_scaled(rule, a, have, ctx):
    """A variant's parameter (fixed O&M) = the host's sourced value x the variant's
    capex / the host's capex, where the variant has none: the host's own ratio, not
    a pooled one (sanity-check run 2026-10-24, P-G3, IS007: a pooled ratio had put
    a variant's fixed O&M at two-thirds of its host's while its capex was higher)."""
    p, k = rule["parameter"], a["archetype"]
    if (p, "") in have:
        return []
    h = ctx["hosts"].get(k)
    if not h or h.get("status") != "active" or not h.get("host"):
        return []
    hh = ctx["have"].get(h["host"], {})
    ho, hc, vc = hh.get((p, "")), hh.get(("capex", "")), have.get(("capex", ""))
    if not (_live(ho) and ho.get("grade") == "sourced" and _live(hc) and _live(vc)):
        return []
    hcv, vcv = fnum(hc["central"]), fnum(vc["central"])
    if not hcv or hcv <= 0 or vcv is None or vcv <= 0:
        return []
    f = vcv / hcv
    c, lo, hi = _lohi(ho)
    return [{"parameter": p, "target": "", "central": c * f, "low": lo * f, "high": hi * f,
             "inputs": [_input(ho, f"{h['host']} {p}"), f"x capex {vcv:.4g} / host capex {hcv:.4g}"],
             "grades": [ho.get("grade", "sourced"), vc.get("grade", "sourced")],
             "sources": ho.get("sources", ""), "external": []}]


# ---- assumptions (records/fill/fill-assumption.csv) -------------------------

def _applies(row, a, ctx):
    """Does an assumption row apply to archetype a? applies-to is a ;-list of
    `all`, `archetype:k`, `equipment:x|y`, `class:<lifetime-class>`,
    `vom-class:<c>`, `family:f`, `end-use:e|f`, `modifier:m`, `product-unit:u`."""
    for term in split_list(row.get("applies-to", "")):
        kind, _, val = term.partition(":")
        vals = val.split("|")
        cls = ctx["classes"].get(a["equipment"], {})
        if (kind == "all"
                or (kind == "archetype" and a["archetype"] in vals)
                or (kind == "equipment" and a["equipment"] in vals)
                or (kind == "class" and cls.get("lifetime-class") in vals)
                or (kind == "vom-class" and cls.get("vom-class") in vals)
                or (kind == "family" and a.get("family") in vals)
                or (kind == "product-unit" and a.get("product-unit") in vals)
                or (kind == "end-use" and set(vals) & ctx["service"].get(a["archetype"], set()))
                or (kind == "modifier" and set(vals) & set(split_list(a.get("modifiers", ""))))):
            return True
    return False


def _assumption(rule, a, ctx):
    """The first row of the rule's assumption group (its `inputs` cell,
    `group=<name>`) that applies to the archetype; rows are tried in file order,
    so a narrower row goes above a wider one."""
    grp = dict(x.partition("=")[::2] for x in split_list(rule.get("inputs", "")) if "=" in x).get("group")
    for row in ctx["assumptions"]:
        if row.get("group") == grp and row.get("status", "active") != "off" and _applies(row, a, ctx):
            return row
    return None


def m_assumption(rule, a, have, ctx):
    """A class constant from fill-assumption.csv. The rule's formula says how it
    is used: `value` writes it as the cell; `share-of:<parameter>` multiplies the
    archetype's value of that parameter (fixed O&M as a share of capex)."""
    p = rule["parameter"]
    if (p, "") in have:
        return []
    row = _assumption(rule, a, ctx)
    if not row:
        return []
    v, lo, hi = fnum(row["value"]), fnum(row.get("low")), fnum(row.get("high"))
    if v is None:
        return []
    lo = v if lo is None else lo
    hi = v if hi is None else hi
    mode = (rule.get("formula") or "value").strip()
    ins = [f"assumption {row['assumption-id']}: {row.get('class') or row.get('applies-to')} "
           f"= {v:g} [{lo:g}-{hi:g}] ({row.get('source-id', '')})"]
    grades, srcs = [], split_list(row.get("source-id", ""))
    if mode.startswith("share-of:"):
        q = mode.split(":", 1)[1]
        b = have.get((q, ""))
        if not _live(b):
            return []
        bc, bl, bh = _vals(b)
        bl = bc if bl is None else bl
        bh = bc if bh is None else bh
        c, lo, hi = bc * v, bl * lo, bh * hi
        ins.insert(0, _input(b, q))
        grades.append(b.get("grade", "sourced"))
        srcs = split_list(b.get("sources", "")) + srcs
        unit_note = f"x {q}"
    else:
        c, unit_note = v, ""
    if not _plausible(rule, c):
        return []
    return [{"parameter": p, "target": "", "central": c, "low": lo, "high": hi,
             "inputs": ins + ([unit_note] if unit_note else []), "grades": grades,
             "sources": "; ".join(dict.fromkeys(srcs)), "external": [row["assumption-id"]]}]


# parameters whose value does not depend on the product unit: siblings need not
# share it (modeller 2026-10-09)
UNIT_FREE = {"lifetime", "construction-time", "capacity-factor"}


def _siblings(a, p, ctx, exclude_chp=False):
    """(tier, [(archetype, row)]) -- the nearest archetypes with a SOURCED value of
    p: tier 1 same equipment and discriminator, tier 2 same equipment; the same
    product unit, except for the unit-free parameters (UNIT_FREE)."""
    A = ctx["arch"]
    for tier, same in ((1, ("equipment", "discriminator")), (2, ("equipment",))):
        rows = []
        for k, h in ctx["have"].items():
            if k == a["archetype"] or k not in A:
                continue
            b = A[k]
            if ((b["product-unit"] != a["product-unit"] and p not in UNIT_FREE)
                    or any(b[f] != a[f] for f in same)
                    or (exclude_chp and is_chp(b))
                    # an efficiency is of one product: a geothermal heat well does
                    # not take a geothermal power plant's (S142/HPCH005)
                    or (p == "efficiency" and (b.get("product") or "") != (a.get("product") or ""))):
                continue
            r = h.get((p, ""))
            if _sourced(r):
                rows.append((k, r))
        if rows:
            return tier, rows
    return None, []


def _sourced(r):
    return bool(r) and r.get("grade") == "sourced" and fnum(r.get("central")) is not None


def _quantile(xs, q):
    xs = sorted(xs)
    if len(xs) == 1:
        return xs[0]
    i = q * (len(xs) - 1)
    lo = int(i)
    return xs[lo] + (xs[min(lo + 1, len(xs) - 1)] - xs[lo]) * (i - lo)


CAP100 = ("capacity-factor", "availability", "efficiency")


# ---- estimators: methods that borrow from OTHER archetypes. Each returns
# {central, low, high, inputs, sources, grade} for archetype `a`; the pool is
# sourced values of other archetypes only, so the same function back-tests a
# cell that has a sourced value (that value is never in its own pool). Where an
# estimator scales a value of the archetype itself (cost ratios), that value may
# be a fill, and the estimate inherits its grade.

def _range(rows, c, widen, p):
    cs = [fnum(r["central"]) for _, r in rows]
    lows = [fnum(r.get("low")) for _, r in rows if fnum(r.get("low")) is not None] + cs
    highs = [fnum(r.get("high")) for _, r in rows if fnum(r.get("high")) is not None] + cs
    if widen >= 1:
        # a multiplicative widen (x3 for widen=2): plant sizes and the like spread
        # by factors, and a low of zero would be meaningless
        lo, hi = min(lows + [c / (1 + widen)]), max(highs + [c * (1 + widen)])
        lo = max(lo, c / (1 + widen) / 3) if lo <= 0 else lo
    else:
        lo, hi = min(lows + [c * (1 - widen)]), max(highs + [c * (1 + widen)])
    if p in CAP100 and p != "efficiency":
        hi = min(hi, 100.0)
    return lo, hi


def e_sibling(rule, a, ctx):
    """The median of the nearest siblings' sourced values; the range spans theirs
    and at least +/- `widen` of the central."""
    p = rule["parameter"]
    sc = _scope(rule)
    if p == "efficiency":
        hv = _host_efficiency(a, ctx)
        if hv:
            return hv
    tier, rows = _siblings(a, p, ctx, exclude_chp=sc.get("sibling-exclude-chp") == "yes")
    if not rows or ("tier" in sc and tier > int(sc["tier"])):
        return None
    widen = fnum(rule.get("widen")) or 0.0
    c = median(fnum(r["central"]) for _, r in rows)
    lo, hi = _range(rows, c, widen, p)
    return {"central": c, "low": lo, "high": hi, "grade": "sourced",
            "inputs": [f"tier {tier}: " + ", ".join(k for k, _ in rows)],
            "sources": [x for _, r in rows for x in split_list(r.get("sources", ""))]}


def _host_efficiency(a, ctx):
    """A capture or efficiency variant's efficiency from its host ahead of any
    sibling (S142/PB009): the host's efficiency with the variant's extra energy
    use added to the input, 100 / (100 / host efficiency + (variant SEC - host
    SEC)); the host's value where either SEC is missing."""
    k = a["archetype"]
    hrow = ctx["hosts"].get(k, {})
    host = hrow.get("host") if hrow.get("status") == "active" else None
    if not host:
        return None
    hh, h = ctx["have"].get(host, {}), ctx["have"].get(k, {})
    he = hh.get(("efficiency", ""))
    if not _live(he) or fnum(he["central"]) <= 0:
        return None
    c, lo, hi = _lohi(he)
    vs, hs = h.get((SEC, "")), hh.get((SEC, ""))
    note = "the host's efficiency"
    if _live(vs) and _live(hs) and a.get("product-unit") == "GJ":
        d = fnum(vs["central"]) - fnum(hs["central"])
        if d > 0:
            f = lambda e: 100.0 / (100.0 / e + d) if e and e > 0 else None
            c, lo, hi = f(c), f(lo), f(hi)
            note = f"the host's efficiency with the variant's extra {d:.3g} GJ/GJ of energy use added to the input"
    return {"central": c, "low": lo, "high": hi, "grade": he.get("grade", "sourced"),
            "inputs": [_input(he, f"{host} efficiency"), note],
            "sources": split_list(he.get("sources", ""))}


def e_class_median(rule, a, ctx):
    """The median sourced value of the archetypes in the same equipment class
    (records/fill/equipment-class.csv), at least `min-pool` of them; a wider
    sibling rule for parameters set by the kind of equipment."""
    p = rule["parameter"]
    cls = ctx["classes"].get(a["equipment"], {}).get("lifetime-class")
    if not cls:
        return None
    A = ctx["arch"]
    rows = []
    for k, h in ctx["have"].items():
        if k == a["archetype"] or k not in A:
            continue
        if ctx["classes"].get(A[k]["equipment"], {}).get("lifetime-class") != cls:
            continue
        if p not in UNIT_FREE and A[k]["product-unit"] != a["product-unit"]:
            continue
        r = h.get((p, ""))
        if _sourced(r):
            rows.append((k, r))
    need = int(_scope(rule).get("min-pool", "3"))
    if len(rows) < need and _scope(rule).get("fallback-all") == "yes":
        rows = [(k, h.get((p, ""))) for k, h in ctx["have"].items()
                if k != a["archetype"] and k in A and _sourced(h.get((p, "")))]
        cls = "all (class pool too small)"
    if len(rows) < need:
        return None
    widen = fnum(rule.get("widen")) or 0.0
    c = median(fnum(r["central"]) for _, r in rows)
    lo, hi = _range(rows, c, widen, p)
    if p == "lifetime":
        # the band of a class lifetime comes from the members' centrals (10th-90th
        # percentile), not from the lowest low any member prints (a stack or
        # measure interval of a year): Brad 2026-10-28, RX019/OT023
        cs = [fnum(r["central"]) for _, r in rows]
        lo, hi = min(_quantile(cs, 0.1), c * (1 - widen)), max(_quantile(cs, 0.9), c * (1 + widen))
    return {"central": c, "low": lo, "high": hi, "grade": "sourced",
            "inputs": [f"class {cls}: median of {len(rows)} archetypes"
                       + (": " + ", ".join(sorted(k for k, _ in rows)) if len(rows) <= 25 else "")],
            "sources": [x for _, r in rows for x in split_list(r.get("sources", ""))]}


def _pu(unit):
    """The product unit a cost is per: `2025_CAD/t-per-yr` and `2025_CAD/t` -> t."""
    return (unit or "").split("/")[-1].replace("-per-yr", "")


def _ratio_unit(pu):
    """A cost ratio (per year of the same product unit on both sides) does not depend
    on the scale of the unit: 10^3 m3 pools with m3 (2026-10-23)."""
    return {"1000-m3": "m3", "e3m3": "m3", "e6m3": "m3"}.get(pu, pu)


def _abates(b):
    return (b.get("product") or "").strip().lower() in ("methane", "carbon dioxide")


def e_cost_ratio(rule, a, ctx):
    """p = base parameter x the median p/base ratio of archetypes with BOTH sourced,
    from the nearest pool of at least `min-pool` (same equipment, then family,
    then all; `min-pool=3|5` gives each level its own minimum). Pools share the
    archetype's product unit. Range: the pool's `widen` and 1-`widen` quantiles.
    The base value may itself be a fill; the result inherits its grade."""
    p, base = rule["parameter"], split_list(rule["inputs"])[0]
    sc = _scope(rule)
    needs = [int(x) for x in sc.get("min-pool", "3").split("|")]
    b = ctx["have"].get(a["archetype"], {}).get((base, ""))
    if not _live(b) or fnum(b["central"]) <= 0:
        return None
    bc = fnum(b["central"])
    A = ctx["arch"]
    ratios = {}
    for k, h in ctx["have"].items():
        if k == a["archetype"] or k not in A:
            continue
        x, y = h.get((base, "")), h.get((p, ""))
        if _sourced(x) and _sourced(y) and fnum(x["central"]) > 0 \
                and _pu(x["unit"]) == _pu(y["unit"]):
            if "vom-class" in sc and ctx["classes"].get(A[k]["equipment"], {}).get("vom-class") not in sc["vom-class"].split("|"):
                continue
            ratios[k] = fnum(y["central"]) / fnum(x["central"])
    # an abatement device (product: t of methane or CO2 abated) prices its capital
    # and labour on another footing from a plant making a product: the two never
    # share a ratio pool (S142/OG001: methane devices had set the LNG trains' O&M)
    # `pool-as-plant`: an abatement archetype that is a process plant in its own right
    # (flare-gas recovery: a compressor station) takes its ratio from the plants, at
    # any product unit, not from the methane devices (Brad 2026-10-28, OG002)
    as_plant = a["archetype"] in split_list(sc.get("pool-as-plant", "").replace("|", ";"))
    ratios = {k: v for k, v in ratios.items() if _abates(A[k]) == (_abates(a) and not as_plant)}
    # same product unit at the first three levels; the last level pools every unit
    # (the ratio is per year of the same product unit on both sides)
    levels = ["equipment", "family", None, "any-unit"]
    pool = {}
    for i, level in enumerate(levels):
        pool = {k: v for k, v in ratios.items()
                if (level == "any-unit" or as_plant or _ratio_unit(A[k]["product-unit"]) == _ratio_unit(a["product-unit"]))
                and (level in (None, "any-unit") or A[k][level] == a[level])}
        need = needs[min(i, len(needs) - 1)]
        if len(pool) >= need:
            break
        pool = {}
    if not pool:
        return None
    q = fnum(rule.get("widen")) or 0.0
    vs = list(pool.values())
    m = median(vs)
    bl, bh = fnum(b.get("low")), fnum(b.get("high"))
    bl = bc if bl is None else bl
    bh = bc if bh is None else bh
    return {"central": bc * m, "low": bl * _quantile(vs, q), "high": bh * _quantile(vs, 1 - q),
            "grade": b.get("grade", "sourced"),
            "inputs": [_input(b, base),
                       f"{p}/{base} median {m:.4g} of {len(pool)} archetypes "
                       f"({'same ' + level if level not in (None, 'any-unit') else 'all' if level is None else 'all, any product unit'}, quartiles {_quantile(vs, q):.3g}-{_quantile(vs, 1 - q):.3g})"],
            "sources": split_list(b.get("sources", ""))}


def e_modifier_ratio(rule, a, ctx):
    """A variant (`base_<modifier>`) = its base archetype's sourced value x the
    median variant/base ratio that single sources give where they price BOTH a
    base and its variant with the same modifier. Needs `min-pool` such pairs.
    Range: the smallest and largest of those ratios, applied to the base's range."""
    p = rule["parameter"]
    parts = a["archetype"].split("_")
    if len(parts) < 2:
        return None
    mod, base = parts[-1], "_".join(parts[:-1])
    b = ctx["have"].get(base, {}).get((p, ""))
    if not _sourced(b):
        return None
    sc = _scope(rule)
    need = int(sc.get("min-pool", "2"))
    A = ctx["arch"]
    # pairs from the same equipment only, and an efficiency modifier never makes the
    # plant cheaper (S142/RX014: ammonia reformers and tunnel kilns had priced a
    # melting furnace's and an HVAC heater's efficient variants below their bases)
    ratios = [(r, sid, v) for (r, sid, v) in ctx["mod_ratios"].get((mod, p), [])
              if v != a["archetype"] and v in A and A[v]["equipment"] == a["equipment"]
              and not (mod.startswith("efficiency") and r < 1)]
    if len(ratios) < need:
        return None
    rs = [r for r, _, _ in ratios]
    m = median(rs)
    bc, bl, bh = _vals(b)
    bl = bc if bl is None else bl
    bh = bc if bh is None else bh
    return {"central": bc * m, "low": bl * min(rs), "high": bh * max(rs), "grade": "sourced",
            "inputs": [_input(b, f"{base} {p}"),
                       f"_{mod}/base median {m:.4g} from {len(rs)} same-source pairs: "
                       + ", ".join(f"{v} ({sid})" for _, sid, v in ratios)],
            "sources": split_list(b.get("sources", ""))}


def e_end_use_median(rule, a, ctx):
    """The median value of the archetypes serving the same end use (annotation
    group) with the same product unit, at least `min-pool` of them. For plant
    scale: the archetypes competing on one end use serve plants of about the same
    size. The pool takes sourced values and identity fills (capacity x capacity
    factor), never estimates."""
    p = rule["parameter"]
    uses = ctx["service"].get(a["archetype"], set())
    if not uses:
        return None
    A = ctx["arch"]
    rows = []
    for k, h in ctx["have"].items():
        if k == a["archetype"] or k not in A or A[k]["product-unit"] != a["product-unit"]:
            continue
        if not (uses & ctx["service"].get(k, set())):
            continue
        r = h.get((p, ""))
        if _live(r) and r.get("grade") in ("sourced", "identity"):
            rows.append((k, r))
    need = int(_scope(rule).get("min-pool", "3"))
    if len(rows) < need:
        return None
    widen = fnum(rule.get("widen")) or 0.0
    c = median(fnum(r["central"]) for _, r in rows)
    lo, hi = _range(rows, c, widen, p)
    return {"central": c, "low": lo, "high": hi,
            "grade": worst([r.get("grade", "sourced") for _, r in rows]),
            "inputs": [f"end use {'/'.join(sorted(uses))}: median of {len(rows)} archetypes: "
                       + ", ".join(sorted(k for k, _ in rows))],
            "sources": [x for _, r in rows for x in split_list(r.get("sources", ""))]}


ESTIMATORS = {"sibling": e_sibling, "cost-ratio": e_cost_ratio, "modifier-ratio": e_modifier_ratio,
              "class-median": e_class_median, "end-use-median": e_end_use_median}


def _estimated(fn):
    def m(rule, a, have, ctx):
        p = rule["parameter"]
        if (p, "") in have:
            return []
        e = fn(rule, a, ctx)
        if not e:
            return []
        return [{"parameter": p, "target": "", "central": e["central"], "low": e["low"],
                 "high": e["high"], "inputs": e["inputs"], "grades": [e.get("grade", "sourced")],
                 "sources": "; ".join(dict.fromkeys(e["sources"])), "external": []}]
    return m


def backtest(records, rules_, have, ctx):
    """Each borrowing rule run on the cells that HAVE a sourced value, as if they
    had none: does the sourced value fall inside the estimate's range, and how far
    is the central from it. Rows for output/fill-backtest.csv."""
    out = []
    for rule in rules_:
        fn = ESTIMATORS.get(rule.get("method"))
        if not fn:
            continue
        p = rule["parameter"]
        for a in records.archetypes:
            r = have.get(a["archetype"], {}).get((p, ""))
            if not _sourced(r) or not _in_scope(rule, a, {}, ctx):
                continue
            e = fn(rule, a, ctx)
            if not e:
                continue
            t, c, lo, hi = fnum(r["central"]), e["central"], e["low"], e["high"]
            out.append({"rule-id": rule["rule-id"], "archetype": a["archetype"], "parameter": p,
                        "sourced": _g(t), "estimate": _g(c), "low": _g(lo), "high": _g(hi),
                        "inside": "yes" if min(lo, hi) <= t <= max(lo, hi) else "no",
                        "ratio": _g(c / t) if t else "", "inputs": "; ".join(e["inputs"])})
    return out


METHODS = {"inverse": m_inverse, "inverse-net-feed": m_inverse_net_feed,
           "chp-heat-inverse": m_chp_heat_inverse, "inverse-net-feed-efficiency": m_inverse_net_feed_efficiency, "fuel-emission-factor": m_fuel_emission_factor, "biogenic-emission-factor": m_biogenic_emission_factor,
           "sec-from-emissions": m_sec_from_emissions, "sum": m_sum, "difference": m_difference,
           "capacity-output": m_capacity_output, "bucket-saving": m_bucket_saving,
           "host-increment": m_host_increment, "host-copy": m_host_copy,
           "bucket-increment": m_bucket_increment, "per-gj-via-sec": m_per_gj_via_sec,
           "assumption": m_assumption, "marker": m_marker, "constant": m_constant, "host-scaled": m_host_scaled, "zero-if-no-fossil": m_zero_if_no_fossil,
           "range-midpoint": m_range_midpoint, "single-carrier-split": m_single_carrier_split,
           "host-split": m_host_split, "feedstock-identity": m_feedstock_identity,
           "co2e-from-species": m_co2e_from_species,
           **{k: _estimated(f) for k, f in ESTIMATORS.items()}}


# ---- variants consistent with their hosts (modeller 2026-10-10) ------------
#
# A variant (an archetype with an active row in host-archetype.csv) is the host
# plant plus something: an efficiency measure or a CO2 capture plant. So it has
# the host's size, and a host-with-capture variant has the host plant's costs,
# energy use and emissions plus the capture add-on's. These rules run after the
# others and OVERRIDE what the variant had: a value the variant's own sources
# gave is kept, moved to target `as-sourced` (a secondary parameter), so it can
# be compared and nothing is lost. The CIMS export writes the host-plant part
# (target `host-plant`) plus the service request to the CCS node; the database's
# main value is the all-in plant.

SIZE = ("annual-output", "capacity", "capacity-factor")
COMPONENT_TARGETS = ("host-plant", "ccs-add-on")
CAPTURE_RATE = 90.0      # % of the CO2 sent to capture that a capture plant captures (design s2)


def is_component_target(t):
    t = t or ""
    return t.startswith("as-sourced") or t in COMPONENT_TARGETS


def _variants(ctx):
    """Active variants, hosts before their variants (a host may itself be a variant)."""
    hs = {k: h["host"] for k, h in ctx["hosts"].items()
          if h.get("status") == "active" and h.get("host") and k in ctx["arch"]}
    out, seen = [], set()

    def visit(k, depth=0):
        if k in seen or depth > 10:
            return
        if hs.get(k) in hs:
            visit(hs[k], depth + 1)
        seen.add(k)
        out.append(k)
    for k in sorted(hs):
        visit(k)
    return [(k, hs[k]) for k in out]


def _as_sourced(t):
    return "as-sourced" + (f"-{t}" if t else "")


def _put(psum, h, row, stats, rid):
    """Write row into the variant's cell, overriding what was there: a sourced
    row moves to target as-sourced; a filled row is dropped."""
    key = (row["parameter"], row["target"])
    old = h.get(key)
    if old is not None:
        if old.get("grade") == "sourced":
            t2 = _as_sourced(old.get("target", ""))
            old["target"] = t2
            old["central-from"] = ("as the variant's own sources give it; the main value is "
                                   "host-based (2026-10-10). " + (old.get("central-from") or ""))
            h[(old["parameter"], t2)] = old
            stats["sourced moved to as-sourced"] = stats.get("sourced moved to as-sourced", 0) + 1
        elif old in psum:
            psum.remove(old)
    psum.append(row)
    h[key] = row
    stats[rid] = stats.get(rid, 0) + 1


def _row(a, p, t, unit, c, lo, hi, grade, rule, inputs, sources, note=""):
    spread = lo is not None and hi is not None and abs(hi - lo) > 1e-12 * max(1, abs(c))
    return {"archetype": a, "parameter": p, "target": t, "unit": unit,
            "central": _g(c), "low": _g(min(lo, hi)) if spread else "",
            "high": _g(max(lo, hi)) if spread else "", "n": 0, "n-external": "",
            "central-from": f"{grade}: {rule['formula']}" + (f" ({note})" if note else ""),
            "external-factors": "", "sources": sources, "examples": "", "grade": grade,
            "method": rule["method"], "rule-id": rule["rule-id"], "inputs": "; ".join(inputs),
            "rationale": rule["rationale"]}


def _lohi(r):
    c, lo, hi = _vals(r)
    return c, (c if lo is None else lo), (c if hi is None else hi)


def pm_host_size(rule, psum, records, target_unit, ctx, stats):
    """Every variant takes its host's value of the rule's parameter."""
    p = rule["parameter"]
    for k, host in _variants(ctx):
        hr = ctx["have"].get(host, {}).get((p, ""))
        if not _live(hr):
            continue
        c, lo, hi = _lohi(hr)
        g = worst([rule["grade"], hr.get("grade", "sourced")])
        _put(psum, ctx["have"].setdefault(k, {}),
             _row(k, p, "", hr["unit"], c, lo, hi, g, rule, [_input(hr, f"{host} {p}")],
                  hr.get("sources", "")), stats, rule["rule-id"])


def _captured(k, host, ctx):
    """(t CO2 captured per unit of product, low, high, grade, inputs, derived) for
    a host-with-capture variant: the capture rate x the variant's share-to-capture
    x the host's gross CO2 (user 2026-10-24, sanity-check S138: a captured tonnage
    from another plant had been netted off this host, giving 61% capture on lime
    and zero net emissions where it exceeded the host's gross). The variant's own
    sourced emissions-captured is used only where no share can be had."""
    from .capture_share import RATES
    h, hh = ctx["have"].get(k, {}), ctx["have"].get(host, {})
    e = h.get(("emissions-captured", "co2"))
    d = _captured_derived(k, host, ctx, RATES.get(k, CAPTURE_RATE))
    if d:
        return d + (True,)
    if _live(e) and fnum(e["central"]) > 0:
        c, lo, hi = _lohi(e)
        return c, lo, hi, e.get("grade", "sourced"), [_input(e, f"{k} emissions-captured_co2")], False
    return None


def _host_gross_co2(host, ctx):
    """(central, low, high, inputs, grades) of the host's gross CO2: its
    emissions-total, else its combustion plus process CO2 (S138: a combined cycle
    gives only combustion)."""
    hh = ctx["have"].get(host, {})
    tot = hh.get(("emissions-total", "co2"))
    parts = [hh.get(("emissions-combustion", "co2")), hh.get(("emissions-process", "co2"))]
    if _live(tot):
        return _lohi(tot), [_input(tot, f"{host} emissions-total_co2")], [tot.get("grade", "sourced")]
    xs = [x for x in parts if _live(x)]
    if xs:
        return (tuple(sum(_lohi(x)[i] for x in xs) for i in range(3)),
                [_input(x, f"{host} {x['parameter']}_co2") for x in xs],
                [x.get("grade", "sourced") for x in xs])
    # last: a CO2-eq total, where the host records no other gas and the CO2-eq was
    # not built from species (sanity-check run 2026-10-24, P-C7: coal-to-methanol)
    te = hh.get(("emissions-total", "co2e"))
    others = any(k[0].startswith("emissions-") and k[1] not in ("", "co2", "co2e") and _live(v)
                 for k, v in hh.items())
    if _live(te) and not others and not (te.get("rule-id") or "").startswith("id-co2e-from-species"):
        return (_lohi(te), [_input(te, f"{host} emissions-total_co2e") + " (CO2-eq read as CO2)"],
                [worst([te.get("grade", "sourced"), "estimated"])])
    return None


def _captured_derived(k, host, ctx, rate):
    h = ctx["have"].get(k, {})
    g = _host_gross_co2(host, ctx)
    if not g:
        return None
    co2, ins, gs = g
    sh = h.get(("share-to-capture", "co2")) or h.get(("share-to-capture", ""))
    if not _live(sh) or co2[0] <= 0:
        return None
    s, sl, shi = _lohi(sh)
    f = rate / 100 / 100
    return (co2[0] * s * f, co2[1] * sl * f, co2[2] * shi * f,
            worst(gs + [sh.get("grade", "sourced")]),
            ins + [_input(sh, f"{k} share-to-capture"), f"capture rate {rate:.4g}%"])


# The reference capture plant per CCS node: where a node has one, the tier-2 add-on
# (cost and energy) is that archetype's value rather than the median of every
# capture archetype on the node. Post-combustion (diffuse) capture is a standard
# amine (MEA) plant, as the cement, lime, steel and power sources describe (user
# 2026-10-23, sanity-check MCF006, IS018, S139).
REFERENCE_CAPTURE = {"diffuse": "capture-unit_amine"}


def _node_archetypes(node, ctx, p=None):
    ks = ctx["ccs_nodes"].get(node, set())
    ref = REFERENCE_CAPTURE.get(node)
    if ref in ks and (p is None or _live(ctx["have"].get(ref, {}).get((p, "")))):
        return {ref}
    return ks


def _node_medians(node, p, ctx, target=""):
    """Centrals of parameter p on the capture archetypes serving the CCS node
    (`diffuse` or `concentrated`); the node's reference plant alone where it has one."""
    vals = {}
    for k in _node_archetypes(node, ctx, p):
        r = ctx["have"].get(k, {}).get((p, target))
        if _live(r) and fnum(r["central"]) >= 0:
            vals[k] = fnum(r["central"])
    return vals


def _rebase_factor(k, host, sid, eid, ctx):
    """(factor, note) putting a same-source add-on on this variant's CO2 captured:
    the variant's captured CO2 per unit of product over the source's own, from the
    same example where it gives one, else the source's median (sanity-check run
    2026-10-24, P-D7, S103). (1.0, "") where the source gives none."""
    a = ctx["arch"][k]
    cu = "t/" + a["product-unit"]
    rows = [r for r in ctx.get("captured_facts", {}).get((k, sid), []) if r.get("unit-norm") == cu]
    if not rows:
        return 1.0, ""
    same = [fnum(r["value-norm"]) for r in rows if r["example-id"] == eid]
    src = median(same) if same else median(fnum(r["value-norm"]) for r in rows)
    cap = _captured(k, host, ctx)
    if not cap or cap[0] <= 0 or src <= 0:
        return 1.0, ""
    f = cap[0] / src
    return f, (f"; rebased per t CO2: x {f:.3g} (this variant captures {cap[0]:.4g} {cu}, "
               f"the source {src:.4g})")


def _tier1(k, host, p, unit, ctx):
    """Same-source add-on: the source's incremental capture cost on this variant,
    else the same source's plant with capture less its plant without. Each
    source's add-on is put on this variant's CO2 captured where the source gives
    its own (2026-10-24, S103). Median of the per-source values."""
    rows = [r for r in ctx["addon_facts"].get((k, p), []) if r.get("unit-norm") == unit]
    vals, ins = [], []
    inc = {}
    for r in rows:
        if (r.get("cost-basis") or "") == "incremental":
            inc.setdefault((r["example-id"], r["source-id"]), []).append(fnum(r["value-norm"]))
    for (eid, sid), vs in inc.items():
        rf, rn = _rebase_factor(k, host, sid, eid, ctx)
        vals.append(median(vs) * rf)
        ins.append(f"incremental {median(vs):.4g} ({eid}){rn}")
    whole, whole_ex = {}, {}
    for r in rows:
        if (r.get("cost-basis") or "") != "incremental":
            whole.setdefault(r["source-id"], []).append(fnum(r["value-norm"]))
            whole_ex.setdefault(r["source-id"], r["example-id"])
    hp = ctx["have"].get(host, {}).get((p, ""))
    hpc = fnum(hp["central"]) if _live(hp) else None
    for sid, vs in whole.items():
        hv = [fnum(r["value-norm"]) for r in ctx["pooled_facts"].get((host, p, sid), [])
              if r.get("unit-norm") == unit]
        if not hv:
            continue
        w, wo = median(vs), median(hv)
        if hpc and hpc > 0 and wo > 0 and not (0.5 <= wo / hpc <= 2.0):
            # the source's plant without capture is on another scale from the pooled
            # host (a cost per MWh of electricity read per GJ of heat): its with/without
            # RATIO goes on the pooled host, not its difference (S139/HPCH011)
            d = hpc * (w / wo - 1)
            how = (f"{sid}: with capture {w:.4g} / without {wo:.4g} = x{w / wo:.3g} on the pooled "
                   f"host {hpc:.4g} (the source's host is {wo / hpc:.2g}x the pooled host)")
        else:
            d = w - wo
            how = f"{sid}: with capture {w:.4g} less without {wo:.4g}"
        if d > 0:
            rf, rn = _rebase_factor(k, host, sid, whole_ex[sid], ctx)
            vals.append(d * rf)
            ins.append(how + rn)
    if not vals:
        return None
    return median(vals), min(vals), max(vals), ins


def _ccs_extra(k, p, t=""):
    """An add-on a capture variant needs beyond the capture node's plant, per t CO2
    captured (records/reference/ccs-addon-extra.csv): the air separation unit of
    an oxy-fired host (S169, Brad 2026-10-28), or a host-specific premium such as
    a lime kiln's capture heat supply. Returns (central, low, high, note, applies)
    or None; applies = "tier-2" means the fill adds it only to a capture-node
    add-on (the CIMS export always adds it)."""
    for r in read(RECORDS / "reference" / "ccs-addon-extra.csv"):
        if r["archetype"] == k and r["parameter"] == p and (r.get("target") or "") == t:
            c = fnum(r["value"])
            lo, hi = fnum(r.get("value-min")), fnum(r.get("value-max"))
            return (c, (lo if lo is not None else c), (hi if hi is not None else c), r.get("note", ""),
                    (r.get("applies") or "").strip())
    return None


def pm_ccs_cost(rule, psum, records, target_unit, ctx, stats):
    """A host-with-capture variant's cost = the host plant's + the capture
    add-on's. The add-on: same-source evidence first (tier 1), else the median
    of the capture archetypes on the matching CCS node per t CO2 x the CO2
    captured per unit of the host's product (tier 2)."""
    p = rule["parameter"]
    for k, host in _variants(ctx):
        a = ctx["arch"][k]
        if "_ccs-" not in k or a.get("family") == "co2-capture":
            continue
        node = "diffuse" if "_ccs-diffuse" in k else "concentrated"
        h, hh = ctx["have"].setdefault(k, {}), ctx["have"].get(host, {})
        unit = target_unit(p, a["product-unit"], records.parameters.get(p, {}).get("unit-default", ""))
        hr = hh.get((p, ""))
        if _live(hr):
            hc, hl, hhi = _lohi(hr)
            hg, hin, hsrc = hr.get("grade", "sourced"), [_input(hr, f"{host} {p}")], hr.get("sources", "")
        elif hr is not None and hr.get("grade") == MARKER:
            hc = hl = hhi = 0.0
            hg, hin, hsrc = "identity", [f"{host} {p} not applicable (0)"], ""
        else:
            continue
        t1 = _tier1(k, host, p, unit, ctx)
        cap = _captured(k, host, ctx)
        nm = _node_medians(node, p, ctx)
        t2 = None
        if cap and nm:
            m = median(nm.values())
            ref = REFERENCE_CAPTURE.get(node)
            if set(nm) == {ref}:
                # the reference plant's own band
                _, rl, rh = _lohi(ctx["have"][ref][(p, "")])
                t2 = (m * cap[0], rl * cap[1], rh * cap[2])
            else:
                t2 = (m * cap[0], min(nm.values()) * cap[1], max(nm.values()) * cap[2])
        if t1 and t2 and t2[0] > 0 and not (0.2 <= t1[0] / t2[0] <= 5.0):
            # a same-source add-on five times off the capture plant's own cost per t
            # CO2 captured is on another scope (an announced project cost, a
            # with-capture plant of another size): tier 2 stands (S139/OG006, HPCH011)
            stats["tier 1 add-ons set aside (5x off tier 2)"] = \
                stats.get("tier 1 add-ons set aside (5x off tier 2)", 0) + 1
            t1_off = t1
            t1 = None
        else:
            t1_off = None
        if t1:
            ac, al, ah = t1[:3]
            ag, ains, tier = "estimated", t1[3], "tier 1 (same source)"
        elif t2:
            ac, al, ah = t2
            ag = worst(["estimated", cap[3]])
            if set(nm) == {REFERENCE_CAPTURE.get(node)}:
                ains = cap[4] + [f"ccs.{node} reference plant {next(iter(nm))} {m:.4g} per t CO2"]
                tier = "tier 2 (reference capture plant x CO2 captured)"
            else:
                ains = cap[4] + [f"ccs.{node} node median {median(nm.values()):.4g} per t CO2 of "
                                 f"{len(nm)}: " + ", ".join(sorted(nm))]
                tier = "tier 2 (capture node x CO2 captured)"
            if t1_off:
                tier += f"; tier 1 {t1_off[0]:.4g} set aside, {t1_off[0] / t2[0]:.2g}x tier 2"
        else:
            stats["ccs add-on not derivable"] = stats.get("ccs add-on not derivable", 0) + 1
            continue
        x = _ccs_extra(k, p)
        if x and x[4] == "tier-2" and t1:
            # a premium the same-source add-on already includes (applies = tier-2):
            # added only on a capture-node add-on, and always in the CIMS export
            # (Brad 2026-11-06, IM18)
            x = None
        if x and cap:
            ac, al, ah = ac + x[0] * cap[0], al + x[1] * cap[1], ah + x[2] * cap[2]
            ag = worst([ag, "guess"])
            ains = ains + [f"plus {x[0]:.4g} per t CO2 captured ({x[3][:60]})"]
            tier += "; plus records/reference/ccs-addon-extra.csv"
        g = worst([rule["grade"], hg, ag])
        ctx["ccs_report"].append({"archetype": k, "host": host, "parameter": p, "unit": unit,
                                  "host-value": _g(hc), "add-on": _g(ac), "tier": tier,
                                  "tier1": _g(t1[0]) if t1 else "", "tier2": _g(t2[0]) if t2 else "",
                                  "co2-captured": _g(cap[0]) if cap else "",
                                  "all-in": _g(hc + ac), "grade": g})
        _put(psum, h, _row(k, p, "", unit, hc + ac, hl + al, hhi + ah, g, rule,
                           hin + [f"add-on {ac:.4g} ({tier})"] + ains, hsrc, tier), stats, rule["rule-id"])
        _put(psum, h, _row(k, p, "host-plant", unit, hc, hl, hhi, worst([rule["grade"], hg]), rule,
                           hin, hsrc, "the host plant's part: what the CIMS export writes"), stats, rule["rule-id"] + ":host-plant")
        _put(psum, h, _row(k, p, "ccs-add-on", unit, ac, al, ah, worst([rule["grade"], ag]), rule,
                           ains, "", tier), stats, rule["rule-id"] + ":ccs-add-on")


def pm_ccs_energy(rule, psum, records, target_unit, ctx, stats):
    """A host-with-capture variant's energy use = the host's (every carrier) +
    the capture plant's per t CO2 (the CCS node's median total, split by the
    node archetypes' mean carrier shares) x the CO2 captured per unit of product."""
    p = SEC
    for k, host in _variants(ctx):
        a = ctx["arch"][k]
        if "_ccs-" not in k or a.get("family") == "co2-capture":
            continue
        node = "diffuse" if "_ccs-diffuse" in k else "concentrated"
        h, hh = ctx["have"].setdefault(k, {}), ctx["have"].get(host, {})
        tot = hh.get((p, ""))
        cap = _captured(k, host, ctx)
        nm = _node_medians(node, p, ctx)
        if not (_live(tot) and cap and nm):
            continue
        per_t = median(nm.values())
        # electricity's share of each node plant's total; the rest is thermal duty
        # (the plant's thermal carriers name the same duty several ways - steam,
        # heat, the fuel that raised it - so they are not added up)
        eshare = []
        for c in _node_archetypes(node, ctx, p):
            ch = ctx["have"].get(c, {})
            t0, te = ch.get((p, "")), ch.get((p, "electricity"))
            if not _live(t0) or fnum(t0["central"]) <= 0:
                continue
            eshare.append(min(1.0, fnum(te["central"]) / fnum(t0["central"])) if _live(te) else 0.0)
        e_ = sum(eshare) / len(eshare) if eshare else 0.0
        sh = {"electricity": e_, "steam": 1.0 - e_}
        # the capture plant needs two things: electricity (fans, compression) and
        # thermal duty (the reboiler). The node archetypes name the thermal duty by
        # whatever raised it (steam, heat, gas, coal); on the host it is steam,
        # the service a capture plant requests. Only these two carriers are
        # written, never the node's fuel names or zero rows (sanity-check
        # 2026-10-11, S107); the host's fuel and named fuels are left as they are,
        # so the two carrier levels still add up.
        # a reboiler takes steam, whatever the host's own heat carrier
        sh = {t: v for t, v in sh.items() if v > 0} or {"electricity": 1.0}
        add = per_t * cap[0]
        g = worst([rule["grade"], tot.get("grade", "sourced"), cap[3]])
        xe = _ccs_extra(k, p, "electricity")
        xadd = xe[0] * cap[0] if xe else 0.0
        if xe:
            g = worst([g, "guess"])
            sh.setdefault("electricity", 0.0)
        ins = cap[4] + [(f"ccs.{node} reference plant {next(iter(nm))} {per_t:.4g} GJ/t CO2" if len(nm) == 1 and next(iter(nm)) == REFERENCE_CAPTURE.get(node) else f"ccs.{node} node median {per_t:.4g} GJ/t CO2 of {len(nm)}")]
        # host rows, every carrier, then the add-on on top
        written = set()
        for (q, t), r in sorted(hh.items()):
            if q != p or is_component_target(t) or not _live(r) or (t and _apart(r)):
                continue    # a carrier pooled apart is not part of the host's total (2026-10-24)
            written.add(t)
            c, lo, hi = _lohi(r)
            extra = (add + xadd) if not t else add * sh.get(t, 0.0) + (xadd if t == "electricity" else 0.0)
            _put(psum, h, _row(k, p, t, r["unit"], c + extra, lo + extra, hi + extra,
                               worst([g, r.get("grade", "sourced")]), rule,
                               [_input(r, f"{host} {p}" + (f"_{t}" if t else "")), f"add-on {extra:.4g}"] + ins,
                               r.get("sources", "")), stats, rule["rule-id"])
        # the fuel aggregate follows the host's named fuels where the host has no
        # `fuel` row of its own, so a sourced `fuel` on the variant (a whole plant
        # with capture) moves to as-sourced and the carriers add up (Brad 2026-11-06)
        if "fuel" not in written:
            from .carriers import NO_HV_CARRIERS, FUEL_AGG
            named = [r for (q, t), r in hh.items() if q == p and t and t not in NO_HV_CARRIERS
                     and t != FUEL_AGG and not is_component_target(t) and not t.startswith("as-sourced")
                     and _live(r) and not _apart(r)]
            if named:
                lo3 = [sum(_lohi(r)[i] for r in named) for i in range(3)]
                _put(psum, h, _row(k, p, FUEL_AGG, tot["unit"], *lo3, worst([g] + [r.get("grade", "sourced") for r in named]), rule,
                                   [f"sum of {host}'s named fuels"] + ins, ""), stats, rule["rule-id"])
                written.add("fuel")
        for t, s in sh.items():
            # a capture carrier the host has no live row for in its total (none, or
            # one pooled apart, as the steam reformer's zero steam) still gets the
            # add-on, so the carriers add up to the total (Brad 2026-11-06, PR06)
            if t not in written:
                v = add * s + (xadd if t == "electricity" else 0.0)
                _put(psum, h, _row(k, p, t, tot["unit"], v, v, v, g, rule,
                                   [f"capture add-on only ({s:.0%} of {add:.4g})"] + ins, ""), stats, rule["rule-id"])
        _put(psum, h, _row(k, p, "host-plant", tot["unit"], *_lohi(tot), worst([g, tot.get("grade", "sourced")]),
                           rule, [_input(tot, f"{host} {p}")], tot.get("sources", "")), stats, rule["rule-id"] + ":host-plant")
        _put(psum, h, _row(k, p, "ccs-add-on", tot["unit"], add + xadd, add + xadd, add + xadd, g, rule,
                           ins + ([f"plus {xe[0]:.4g} GJ electricity per t CO2 ({xe[3][:60]})"] if xe else []), ""),
             stats, rule["rule-id"] + ":ccs-add-on")


def pm_ccs_emissions(rule, psum, records, target_unit, ctx, stats):
    """A host-with-capture variant makes the CO2 its host makes (combustion and
    process, gross); emissions-captured is the CO2 captured per unit of product;
    emissions-total is net of capture, so total + captured = the host's gross
    (ruling 2026-10-13)."""
    for k, host in _variants(ctx):
        a = ctx["arch"][k]
        if "_ccs-" not in k or a.get("family") == "co2-capture":
            continue
        h, hh = ctx["have"].setdefault(k, {}), ctx["have"].get(host, {})
        e = h.get(("emissions-captured", "co2"))
        cap = _captured(k, host, ctx)
        derived_cap = bool(cap and cap[5])
        if cap and (cap[5] or not _live(e)):
            unit = target_unit("emissions-captured", a["product-unit"], "")
            _put(psum, h, _row(k, "emissions-captured", "co2", unit, cap[0], cap[1], cap[2],
                               worst([rule["grade"], cap[3]]), rule, cap[4],
                               "", "captured = capture rate x share-to-capture x the host's gross CO2 (S138)"),
                 stats, rule["rule-id"] + ":captured")
            e = h.get(("emissions-captured", "co2"))
        # captured CO2 split between fossil and biogenic in proportion to the host's
        # gross fossil and biogenic CO2 (ruling 2026-10-14: capture on biogenic CO2
        # lowers emissions-biomass, which can go negative; fossil emissions-total
        # loses only the fossil share)
        bio = hh.get(("emissions-biomass", "co2"))
        fos = hh.get(("emissions-total", "co2"))
        fshare = 1.0
        if _live(bio) and _live(e):
            b0 = fnum(bio["central"]) or 0.0
            # the fossil part of the stream: the host's total, else its combustion plus
            # process CO2 (Brad 2026-11-09: capture takes the complete mix in proportion)
            if _live(fos):
                f0 = fnum(fos["central"]) or 0.0
            else:
                f0 = sum(fnum(hh[(q, "co2")]["central"]) or 0.0 for q in ("emissions-combustion", "emissions-process")
                         if _live(hh.get((q, "co2"))))
            fshare = f0 / (f0 + b0) if (f0 + b0) > 0 else 0.0
            ec, el, eh = _lohi(e)
            bc, bl, bh = _lohi(bio)
            kb = 1.0 - fshare
            _put(psum, h, _row(k, "emissions-biomass", "co2", bio["unit"],
                               bc - kb * ec, bl - kb * eh, bh - kb * el,
                               worst([rule["grade"], bio.get("grade", "sourced"), e.get("grade", "sourced")]),
                               rule, [_input(bio, f"{host} emissions-biomass_co2"),
                                      _input(e, f"{k} emissions-captured_co2"),
                                      f"biogenic share of the captured CO2 {kb:.0%}"],
                               bio.get("sources", ""),
                               "the host's biogenic CO2 less the biogenic part of the CO2 captured"),
                 stats, rule["rule-id"] + ":biogenic")
        # the variant's own sourced combustion or process rows where the host has
        # none: the variant's plant is its host's (M, consistency check 2026-11-07)
        for (q, t), r in list(h.items()):
            if q in ("emissions-combustion", "emissions-process") and t and not t.startswith("as-sourced") \
                    and (q, t) not in hh and _live(r) and (r.get("grade") or "sourced") == "sourced":
                del h[(q, t)]
                r["target"] = _as_sourced(t)
                r["central-from"] = ("as the variant's own sources give it; the host has no such row, and the "
                                     "variant's plant is its host's (2026-11-07). " + (r.get("central-from") or ""))
                h[(q, r["target"])] = r
                stats["variant rows the host lacks set aside"] = stats.get("variant rows the host lacks set aside", 0) + 1
        items = dict(hh)
        if not _live(items.get(("emissions-total", "co2"))):
            g = _host_gross_co2(host, ctx)
            if g:
                (gc, gl, gh), gins, ggs = g
                unit = target_unit("emissions-total", a["product-unit"], "")
                items[("emissions-total", "co2")] = {
                    "parameter": "emissions-total", "target": "co2", "unit": unit,
                    "central": _g(gc), "low": _g(gl), "high": _g(gh), "grade": worst(ggs),
                    "sources": "", "inputs": "; ".join(gins),
                    "central-from": "the host's combustion plus process CO2"}
                stats["net totals from the host's combustion plus process"] = \
                    stats.get("net totals from the host's combustion plus process", 0) + 1
        for (q, t), r in sorted(items.items()):
            if q in ("emissions-combustion", "emissions-process", "emissions-total") and _live(r) \
                    and not is_component_target(t):
                c, lo, hi = _lohi(r)
                ins, note, g = [_input(r, f"{host} {q}_{t}")], "", worst([rule["grade"], r.get("grade", "sourced")])
                if q == "emissions-total" and t in ("co2", "co2e") and _live(e):
                    # net of capture (ruling 2026-10-13, section 4.3): emissions-total
                    # on a capture variant is what leaves the stack; with
                    # emissions-captured it adds back to the host's gross. Combustion
                    # and process stay gross: they say where the CO2 is made.
                    ec, el, eh = (x * fshare for x in _lohi(e))
                    if c - ec < -1e-9:
                        stats["net below zero (captured above the host's gross)"] = \
                            stats.get("net below zero (captured above the host's gross)", 0) + 1
                        note = "CAPTURED EXCEEDS THE HOST'S GROSS: check the scopes; "
                    # a derived captured band follows the gross band (captured = P x
                    # gross), so low pairs with low (S138); a sourced one is independent
                    if derived_cap:
                        c, lo, hi = max(c - ec, 0.0), max(lo - el, 0.0), max(hi - eh, 0.0)
                    else:
                        c, lo, hi = max(c - ec, 0.0), max(lo - eh, 0.0), max(hi - el, 0.0)
                    lo, hi = min(lo, c), max(hi, c)
                    ins.append(_input(e, f"{k} emissions-captured_co2"))
                    g = worst([g, e.get("grade", "sourced")])
                    note += "net of capture: the host's gross less the CO2 captured"
                _put(psum, h, _row(k, q, t, r["unit"], c, lo, hi, g, rule, ins, r.get("sources", ""), note),
                     stats, rule["rule-id"])


# ---- calculated levelized cost (modeller 2026-10-10) -----------------------

def pm_rebuild(rule, psum, records, target_unit, ctx, stats):
    """A periodic rebuild the plant needs within its asset life (a glass furnace's
    cold repair every campaign) is added to its fixed O&M as an annual allowance
    (records/reference/rebuild-allowance.csv: the rebuild cost / the campaign / the
    plant's annual capacity). The allowance is also kept as target `rebuild`."""
    from .common import REFERENCE
    from . import currency
    p = rule["parameter"]
    rows = [r for r in read(REFERENCE / "rebuild-allowance.csv") if r["parameter"] == p]
    fx, _ = currency.factors([{"unit": r["unit"]} for r in rows], write_output=False)
    for r in rows:
        k = r["archetype"]
        if k not in ctx["arch"]:
            continue
        h = ctx["have"].setdefault(k, {})
        base = h.get((p, ""))
        m = fx.get(r["unit"].split("/")[0])
        if not _live(base) or m is None or "/" not in base.get("unit", "") or \
                base["unit"].split("/", 1)[1] != r["unit"].split("/", 1)[1]:
            stats["rebuild allowance not applied"] = stats.get("rebuild allowance not applied", 0) + 1
            continue
        c, lo, hi = (float(r[x]) * m for x in ("central", "low", "high"))
        bc, bl, bh = _lohi(base)
        src = f"{r['source-id']} {r['locator']}"
        comp = _row(k, p, "rebuild", base["unit"], c, lo, hi, rule["grade"], rule,
                    [f"{src}: {r['basis']}"], r["source-id"], "rebuild allowance")
        psum.append(comp)
        h[(p, "rebuild")] = comp
        base.update({"central": _g(bc + c), "low": _g(bl + lo), "high": _g(bh + hi),
                     "grade": worst([base.get("grade") or "sourced", rule["grade"]]),
                     "central-from": (base.get("central-from") or "") +
                     f"; plus a rebuild allowance of {c:.4g} ({src}, rule {rule['rule-id']})",
                     "sources": "; ".join(x for x in [base.get("sources", ""), r["source-id"]] if x),
                     "rule-id": rule["rule-id"], "rationale": rule["rationale"]})
        stats[rule["rule-id"]] = stats.get(rule["rule-id"], 0) + 1


def crf(r, n):
    return r / (1 - (1 + r) ** -n) if r > 0 else 1 / n


def _levelized(k, ctx, rate):
    """(central, low, high, grade, inputs) of the non-energy levelized cost per
    unit of product: (capex x CRF + opex-fixed) / CF + opex-variable."""
    h = ctx["have"].get(k, {})
    cx, of, ov, lt, cf = (h.get((p, "")) for p in ("capex", "opex-fixed", "opex-variable",
                                                     "lifetime", "capacity-factor"))
    if not (_live(cx) and _live(lt) and _live(cf)) or fnum(cf["central"]) <= 0:
        return None
    zero = {"central": "0", "grade": "identity"}
    of = of if _live(of) else (zero if of is not None and of.get("grade") == MARKER else None)
    ov = ov if _live(ov) else (zero if ov is None or ov.get("grade") == MARKER else None)
    if of is None:
        return None
    (c, cl, ch), (o, ol, oh), (v, vl, vh), (n, nl, nh), (f, fl, fh) = (
        _lohi(x) for x in (cx, of, ov, lt, cf))
    f, fl, fh = f / 100, min(fl, 100) / 100, min(fh, 100) / 100
    mid = (c * crf(rate, n) + o) / f + v
    # the band is the cost band at the central's own lifetime and capacity factor
    # (sanity-check 2026-10-11, S119): those two are how the plant is run and
    # written off, not uncertainty in what it costs, and taking their worst
    # corners together with the cost corners gave highs thousands of times the
    # central
    lo = min((cl * crf(rate, n) + ol) / f + vl, mid)
    hi = max((ch * crf(rate, n) + oh) / f + vh, mid)
    g = worst([x.get("grade", "sourced") for x in (cx, of, ov, lt, cf)])
    ins = [f"capex {c:.4g} ({cx.get('grade','sourced')})", f"opex-fixed {o:.4g} ({of.get('grade','sourced')})",
           f"opex-variable {v:.4g} ({ov.get('grade','sourced')})", f"lifetime {n:g} y ({lt.get('grade','sourced')})",
           f"capacity factor {f:.0%} ({cf.get('grade','sourced')})", f"discount rate {rate:.0%}, CRF {crf(rate, n):.4f}"]
    return mid, lo, hi, g, ins


def _rate(ctx):
    for r in ctx["assumptions"]:
        if r.get("group") == "discount-rate" and r.get("status", "active") != "off":
            return fnum(r["value"]) / 100, r["assumption-id"]
    return 0.08, "default"


def pm_levelized(rule, psum, records, target_unit, ctx, stats):
    """cost-levelized, target `calculated`: the non-energy levelized cost of every
    archetype that has capex, fixed O&M, lifetime and a capacity factor."""
    rate, aid = _rate(ctx)
    for a in records.archetypes:
        k = a["archetype"]
        res = _levelized(k, ctx, rate)
        if not res:
            stats["levelized not calculable"] = stats.get("levelized not calculable", 0) + 1
            continue
        c, lo, hi, g, ins = res
        unit = target_unit("cost-levelized", a["product-unit"], "")
        h = ctx["have"].setdefault(k, {})
        _put(psum, h, _row(k, "cost-levelized", rule["target"], unit, c, lo, hi,
                           worst([rule["grade"], g]), rule, ins + [aid], ""), stats, rule["rule-id"])


def _prices(ctx):
    return {r["carrier"]: r for r in ctx["energy_prices"]}


def pm_levelized_energy(rule, psum, records, target_unit, ctx, stats):
    """cost-levelized, target `calculated-incl-energy`: the non-energy levelized
    cost plus each carrier's energy use x its reference price
    (records/reference/energy-price.csv). A carrier priced as `derived` takes the
    named archetype's own calculated-incl-energy cost per GJ (steam from the
    fuel-fired boiler, hydrogen from the methane reformer); where the producing
    archetype is measured per tonne (ammonia, methanol), `derived-per` names the
    conversion factor (GJ per t, HHV) that puts its cost per GJ; `internal` carriers
    (by-product gases, black liquor, waste heat) are not bought and cost 0.
    Written only where every carrier the archetype uses has a price."""
    P = _prices(ctx)
    done = {}
    for _ in range(3):         # derived prices need their source archetype first
        for a in records.archetypes:
            k = a["archetype"]
            if k in done:
                continue
            h = ctx["have"].get(k, {})
            base = h.get(("cost-levelized", "calculated"))
            if not _live(base):
                continue
            secs = _adding_carriers({t: r for (q, t), r in h.items()
                                     if q == SEC and t and not is_component_target(t) and _live(r)})
            # `heat` names the duty the fuels and steam beside it supply: priced only
            # where nothing else supplies it (sanity-check 2026-10-23, S143)
            if "heat" in secs and any(t in FOSSIL or t in BIOGENIC or t in ("steam", "fuel")
                                      for t in secs if t != "heat"):
                secs = {t: r for t, r in secs.items() if t != "heat"}
            tot = h.get((SEC, ""))
            use = {t: r for t, r in secs.items() if t not in ("fuel", "total", "not-stated", "component")}
            # the feedstock the product is made from is bought too (ruling 2026-10-13)
            feeds = {t: r for (q, t), r in h.items() if q == "feedstock-consumption" and t
                     and not is_component_target(t) and _live(r)}
            if "fuel" in secs and not any(t not in NOT_FUEL for t in use):
                use["fuel (unnamed)"] = secs["fuel"]      # unpriceable: no fuel named
            if not use:
                cs = split_list(a.get("carriers", ""))
                if _live(tot) and len(cs) == 1:
                    use = {cs[0]: tot}
                elif tot is not None and tot.get("grade") == MARKER:
                    use = {}                              # energy use not applicable: 0
                else:
                    stats["incl-energy: no energy use"] = stats.get("incl-energy: no energy use", 0) + 1
                    done[k] = None
                    continue
            cost = clo = chi = 0.0
            ins, unpriced, wait = [], [], False
            priced = sorted(use.items()) + sorted((f"{t}", r) for t, r in feeds.items())
            for t, r in priced:
                pr = P.get(t)
                v, vl, vh = _lohi(r)
                if not pr:
                    unpriced.append(t)
                    continue
                kind = pr.get("kind")
                if kind == "internal":
                    ins.append(f"{t} {v:.4g} GJ x 0 (internal)")
                    continue
                if kind == "derived":
                    if pr.get("derived-from") not in done:
                        wait = True
                        break
                    src = done[pr["derived-from"]]
                    if src is None:
                        unpriced.append(f"{t} (from {pr['derived-from']})")
                        continue
                    p0, pl, ph = src
                    per = (pr.get("derived-per") or "").strip()
                    if per:            # a per-tonne product: cost per t / GJ per t
                        f = ctx["factors"].rows.get(per)
                        gj = fnum(f["factor"]) if f else None
                        if not gj:
                            unpriced.append(f"{t} (no factor {per})")
                            continue
                        p0, pl, ph = p0 / gj, pl / gj, ph / gj
                        kind = f"derived {pr['derived-from']} / {gj:g} GJ/t"
                else:
                    p0 = fnum(pr.get("price"))
                    if p0 is None:
                        unpriced.append(t)
                        continue
                    pl, ph = fnum(pr.get("low")) or p0, fnum(pr.get("high")) or p0
                cost += v * p0
                # a negative carrier (energy exported) lowers the cost most when it
                # is largest and its price highest (fixed 2026-10-13, S119)
                clo += vl * pl if v >= 0 else vl * ph
                chi += vh * ph if v >= 0 else vh * pl
                ins.append(f"{t} {v:.4g} GJ x {p0:.4g} ({kind})")
            if wait:
                continue
            if unpriced:
                stats["incl-energy: carrier unpriced"] = stats.get("incl-energy: carrier unpriced", 0) + 1
                ctx["unpriced"].update(unpriced)
                done[k] = None
                continue
            c, lo, hi = _lohi(base)
            clo, chi = min(clo, cost), max(chi, cost)
            done[k] = (c + cost, lo + clo, hi + chi)
            unit = target_unit("cost-levelized", a["product-unit"], "")
            _put(psum, ctx["have"].setdefault(k, {}),
                 _row(k, "cost-levelized", rule["target"], unit, c + cost, lo + clo, hi + chi,
                      worst([rule["grade"], base.get("grade", "sourced")]
                            + [r.get("grade", "sourced") for r in use.values()]),
                      rule, [f"non-energy {c:.4g}"] + ins, ""), stats, rule["rule-id"])


def pm_cf_from_output(rule, psum, records, target_unit, ctx, stats):
    """Capacity factor = annual output / capacity, for archetypes whose output is the
    median throughput of existing facilities and whose capacity is the same
    facilities' licensed capacity (Brad 2026-10-21: Alberta gas plants, AER ST50A).
    Overrides the estimated capacity factor; a pooled one moves to as-sourced.
    Scope: `archetype=` keys in the rule."""
    keys = set()
    for part in (rule.get("scope") or "").split(";"):
        k, _, v = part.strip().partition("=")
        if k == "archetype":
            keys |= set(split_list(v.replace("|", ";")))
    for a in records.archetypes:
        k = a["archetype"]
        if k not in keys:
            continue
        h = ctx["have"].setdefault(k, {})
        out, cap = h.get(("annual-output", "")), h.get(("capacity", ""))
        if not (_live(out) and _live(cap)) or not fnum(cap["central"]):
            continue
        kf = OUTPUT_K.get(a["product-unit"], 1.0)
        oc, ol, oh = _lohi(out)
        cc, cl, ch = _lohi(cap)
        c = oc / (cc * kf) * 100
        # the range pairs like quantiles (small plants' output over small plants'
        # capacity), not the extremes, which would put a large plant's licence
        # under a small plant's throughput
        lo = ol / (cl * kf) * 100 if cl else c
        hi = oh / (ch * kf) * 100 if ch else c
        lo, hi = min(lo, hi, c), min(max(lo, hi, c), 100.0)
        g = worst(["identity", out.get("grade", "sourced"), cap.get("grade", "sourced")])
        _put(psum, h, _row(k, "capacity-factor", "", "%", c, lo, hi, g, rule,
                           [_input(out, "annual-output"), _input(cap, "capacity"), f"x {kf:g}"],
                           "; ".join(dict.fromkeys(split_list(out.get("sources", "")) + split_list(cap.get("sources", ""))))),
             stats, rule["rule-id"])


def pm_capacity_from_output(rule, psum, records, target_unit, ctx, stats):
    """Capacity = annual output / capacity factor, for archetypes whose annual output is
    the median of existing facilities (Brad 2026-10-21: keep capacity and output
    consistent). Overrides a pooled capacity, which moves to target as-sourced (the
    licensed, design or large-plant size the sources give) so it can be compared.
    Scope: `archetype=` keys in the rule. Runs before the variant size rules, so a
    variant takes its host's derived capacity."""
    keys = set()
    for part in (rule.get("scope") or "").split(";"):
        k, _, v = part.strip().partition("=")
        if k == "archetype":
            keys |= set(split_list(v.replace("|", ";")))
    for a in records.archetypes:
        k = a["archetype"]
        if k not in keys:
            continue
        h = ctx["have"].setdefault(k, {})
        out, cf = h.get(("annual-output", "")), h.get(("capacity-factor", ""))
        if not (_live(out) and _live(cf)) or not fnum(cf["central"]):
            continue
        kf = OUTPUT_K.get(a["product-unit"], 1.0)
        oc, ol, oh = _lohi(out)
        fc, fl, fh = _lohi(cf)
        c = oc / (fc / 100) / kf
        lo = ol / (max(fl, fh) / 100) / kf if max(fl, fh) else c
        hi = oh / (min(fl, fh) / 100) / kf if min(fl, fh) else c
        unit = target_unit("capacity", a["product-unit"], records.parameters.get("capacity", {}).get("unit-default", ""))
        g = worst(["identity", out.get("grade", "sourced"), cf.get("grade", "sourced")])
        _put(psum, h, _row(k, "capacity", "", unit, c, lo, hi, g, rule,
                           [_input(out, "annual-output"), _input(cf, "capacity-factor"), f"/ {kf:g}"],
                           "; ".join(dict.fromkeys(split_list(out.get("sources", "")) + split_list(cf.get("sources", ""))))),
             stats, rule["rule-id"])


def _strength(r):
    """How well a size value is supported: its grade first, then its example count."""
    g = r.get("grade", "sourced")
    n = fnum(r.get("n")) or 0
    return (-GRADES.index(g) if g in GRADES else -len(GRADES), n)


def pm_size_anchor(rule, psum, records, target_unit, ctx, stats):
    """Capacity, capacity factor and annual output agree: output = capacity x
    capacity factor x hours (Brad 2026-10-28, extending the 2026-10-21 ruling "keep
    capacity and output consistent" to every archetype).

      * an archetype with an annual output and a capacity factor but no capacity
        gets capacity = output / capacity factor;
      * where all three are held and disagree by more than `tolerance` (default
        10%), one size value anchors and the other is recalculated from it:
          - the annual output anchors when its sources include facility statistics
            (source ids starting with a prefix in `facility-sources`): the median of
            existing plants describes the typical plant (Brad 2026-10-21);
          - otherwise the better-supported value anchors: the better grade, then
            the one pooled from more examples; a tie anchors capacity, the
            nameplate rating sources state;
        a sourced value that is recalculated moves to target as-sourced (kept for
        comparison), as in the other size rules.
    Two rules use it: `mode=align` in the main phase, before annual outputs are
    filled from capacity and before the end-use and sibling medians, so those
    medians read aligned sizes; `mode=fill` in the post phase, after every annual
    output is filled and before the variant size rules, so a variant takes its
    host's capacity."""
    sc = _scope(rule)
    mode = sc.get("mode", "align")      # align: all three held; fill: no capacity held
    tol = fnum(sc.get("tolerance")) or 0.10
    fac = tuple(x for x in sc.get("facility-sources", "").split("|") if x)
    # archetypes a facility-statistics rule sizes (capacity-from-output,
    # cf-from-output) are left to that rule, which knows which value anchors
    own = set()
    for r in rules():
        if r.get("status") == "active" and r.get("method") in ("capacity-from-output", "cf-from-output"):
            own |= set(split_list(_scope(r).get("archetype", "").replace("|", ";")))
    for a in records.archetypes:
        k = a["archetype"]
        if k in own or not _in_scope(rule, a, ctx["have"].get(k, {}), ctx):
            continue
        h = ctx["have"].setdefault(k, {})
        cap, cf, out = h.get(("capacity", "")), h.get(("capacity-factor", "")), h.get(("annual-output", ""))
        if not (_live(cf) and fnum(cf["central"])) or not _live(out):
            continue
        kf = OUTPUT_K.get(a["product-unit"], 1.0)
        unit = target_unit("capacity", a["product-unit"], records.parameters.get("capacity", {}).get("unit-default", ""))
        fc, fl, fh = _lohi(cf)
        oc, ol, oh = _lohi(out)
        anchor = "annual-output"
        held = bool(_live(cap) and fnum(cap["central"]))
        if held != (mode == "align"):
            continue
        if held:
            cc, cl, ch = _lohi(cap)
            implied = cc * fc / 100 * kf
            if implied and abs(oc / implied - 1) <= tol:
                continue
            facility = bool(fac) and any(s.startswith(fac) for s in split_list(out.get("sources", "")))
            if not facility and _strength(cap) >= _strength(out):
                anchor = "capacity"
            note = f"was {oc:.4g} {out.get('unit')} against {implied:.4g} from capacity" if anchor == "capacity" \
                else f"was {cc:.4g} {cap.get('unit')}; output / capacity factor gives {oc / (fc / 100) / kf:.4g}"
        else:
            note = "no capacity held"
        if anchor == "capacity":
            g = worst(["identity", cap.get("grade", "sourced"), cf.get("grade", "sourced")])
            c, lo, hi = cc * fc / 100 * kf, cl * fl / 100 * kf, ch * fh / 100 * kf
            row = _row(k, "annual-output", "", out.get("unit") or a["product-unit"], c, lo, hi, g, rule,
                       [_input(cap, "capacity"), _input(cf, "capacity-factor"), f"x {kf:g}"],
                       "; ".join(dict.fromkeys(split_list(cap.get("sources", "")) + split_list(cf.get("sources", "")))),
                       f"capacity anchors; {note}")
        else:
            g = worst(["identity", out.get("grade", "sourced"), cf.get("grade", "sourced")])
            c = oc / (fc / 100) / kf
            lo = ol / (max(fl, fh) / 100) / kf if max(fl, fh) else c
            hi = oh / (min(fl, fh) / 100) / kf if min(fl, fh) else c
            row = _row(k, "capacity", "", unit, c, lo, hi, g, rule,
                       [_input(out, "annual-output"), _input(cf, "capacity-factor"), f"/ {kf:g}"],
                       "; ".join(dict.fromkeys(split_list(out.get("sources", "")) + split_list(cf.get("sources", "")))),
                       f"annual output anchors; {note}")
        _put(psum, h, row, stats, rule["rule-id"])


def pm_scale_ratio(rule, psum, records, target_unit, ctx, stats):
    """A smaller-scale archetype's cost = the larger archetype's x a same-source
    scale ratio (Brad 2026-10-28: on-site electrolysis from the central plant).
    Pooled costs across sources differ more by cost year than by plant size, so
    the size effect is taken from sources that price both sizes. Scope:
    `archetype=` (the target), `from=` (the archetype scaled), `ratio=c|lo|hi`.
    A sourced value moves to target as-sourced."""
    sc = _scope(rule)
    p, k, src = rule["parameter"], sc.get("archetype", ""), sc.get("from", "")
    rc, rl, rh = (float(x) for x in sc["ratio"].split("|"))
    hr = ctx["have"].get(src, {}).get((p, ""))
    if k not in ctx["arch"] or not _live(hr):
        return
    c, lo, hi = _lohi(hr)
    g = worst([rule["grade"], hr.get("grade", "sourced")])
    _put(psum, ctx["have"].setdefault(k, {}),
         _row(k, p, "", hr["unit"], c * rc, lo * rl, hi * rh, g, rule,
              [_input(hr, f"{src} {p}"), f"x {rc:g} ({rl:g}-{rh:g})"], hr.get("sources", "")),
         stats, rule["rule-id"])


def pm_co2e_consistency(rule, psum, records, target_unit, ctx, stats):
    """CO2-eq = the species x their AR5 GWPs wherever the species are held
    (Brad 2026-10-28: the CO2-eq should sum up the species). The main-phase rule
    fills CO2-eq only where it is missing; a CO2-eq pooled from different examples
    than its species can disagree with them. Where CO2 itself is held and the
    species sum differs from the CO2-eq by more than `tolerance` (default 2%), the
    sum replaces it (a sourced CO2-eq moves to target as-sourced). Skipped when
    the CO2 is the no-fossil zero marker (the CO2-eq then carries a non-fuel CO2
    the species rows do not)."""
    from . import gwp
    sc = _scope(rule)
    tol = fnum(sc.get("tolerance")) or 0.02
    G = gwp.ar5(ctx["factors"])
    params = split_list(sc.get("parameters", "emissions-combustion|emissions-process|emissions-total").replace("|", ";"))
    for a in records.archetypes:
        k = a["archetype"]
        h = ctx["have"].get(k, {})
        for p in params:
            co2, e = h.get((p, "co2")), h.get((p, "co2e"))
            if not (_live(co2) and _live(e)) or co2.get("rule-id") == "id-zero-combustion-no-fossil":
                continue
            parts = [("co2", co2)] + [(sp, h.get((p, sp))) for sp in G if _live(h.get((p, sp)) or {})]
            if p == "emissions-total":
                # a gas the total does not list but a part does (PFCs on a smelter's
                # process emissions) counts at the parts' sum
                held = {sp for sp, _ in parts}
                for sp in G:
                    if sp in held:
                        continue
                    rs = [h.get((q, sp)) for q in ("emissions-combustion", "emissions-process")]
                    rs = [r for r in rs if _live(r or {})]
                    if rs:
                        cs = [_vals(r) for r in rs]
                        parts.append((sp, {"central": sum(c[0] for c in cs),
                                           "low": sum(c[1] if c[1] is not None else c[0] for c in cs),
                                           "high": sum(c[2] if c[2] is not None else c[0] for c in cs),
                                           "grade": worst([r.get("grade", "sourced") for r in rs]),
                                           "sources": "; ".join(r.get("sources", "") for r in rs),
                                           "parameter": "emissions-combustion+process", "target": sp,
                                           "unit": rs[0].get("unit", ""), "method": "sum"}))
            c = lo = hi = 0.0
            ins, grades, srcs = [], [], []
            for sp, r in parts:
                f = 1.0 if sp == "co2" else G[sp]
                v, vl, vh = _vals(r)
                c += v * f
                lo += (vl if vl is not None else v) * f
                hi += (vh if vh is not None else v) * f
                ins.append(_input(r, f"{p}_{sp}") + ("" if sp == "co2" else f" x {f:g}"))
                grades.append(r.get("grade", "sourced"))
                srcs += split_list(r.get("sources", ""))
            ec = fnum(e["central"])
            if ec is not None and abs(c - ec) <= tol * max(abs(ec), abs(c), 1e-12):
                continue
            g = worst(["identity"] + grades)
            _put(psum, ctx["have"].setdefault(k, {}),
                 _row(k, p, "co2e", e.get("unit", ""), c, lo, hi, g, rule, ins,
                      "; ".join(dict.fromkeys(srcs)), f"was {ec:.4g}" if ec is not None else ""),
                 stats, rule["rule-id"])


def pm_process_heat_efficiency(rule, psum, records, target_unit, ctx, stats):
    """Thermal efficiency (heat into the product over energy in) for the
    light-industrial heat archetypes that hold none, from
    records/reference/process-heat-efficiency.csv (Brad 2026-10-28: add
    efficiencies where they matter; the same figures put these archetypes per GJ
    of process heat in the workbook bands). A judged efficiency is graded guess;
    a variant (`factor-from` its host) heats the same product, so its efficiency
    = the host's efficiency x the host's energy use / its own (an identity). Rows
    whose basis says non-thermal are skipped."""
    rows = {r["archetype"]: r for r in read(RECORDS / "reference" / "process-heat-efficiency.csv")}

    def eff_of(k, depth=0):
        h = ctx["have"].get(k, {})
        e = h.get(("efficiency", ""))
        if _live(e):
            return fnum(e["central"]), e.get("grade", "sourced"), _input(e, f"{k} efficiency")
        r = rows.get(k, {})
        if fnum(r.get("efficiency")):
            return fnum(r["efficiency"]), "guess", f"{k} efficiency {r['efficiency']}% (process-heat-efficiency.csv)"
        if r.get("factor-from") and depth < 3:
            host = r["factor-from"]
            he = eff_of(host, depth + 1)
            hs = ctx["have"].get(host, {}).get((SEC, ""))
            ms = h.get((SEC, ""))
            if he and _live(hs) and _live(ms) and fnum(ms["central"]):
                v = he[0] * fnum(hs["central"]) / fnum(ms["central"])
                return v, worst(["identity", he[1], hs.get("grade", "sourced"), ms.get("grade", "sourced")]), \
                    f"{he[2]} x {host} energy use {fnum(hs['central']):.4g} / own {fnum(ms['central']):.4g}"
        return None

    for k, r in rows.items():
        if k not in ctx["arch"] or "non-thermal" in (r.get("basis") or ""):
            continue
        h = ctx["have"].setdefault(k, {})
        if _live(h.get(("efficiency", ""))):
            continue
        got = eff_of(k)
        if not got or not got[0]:
            continue
        v, g, why = got
        v = min(v, 100.0) if "heat-pump" not in k else v
        _put(psum, h, _row(k, "efficiency", "", "%", v, v, v, worst([rule["grade"], g]) if g != "guess" else "guess",
                           rule, [why], "", r.get("basis", "")), stats, rule["rule-id"])


def pm_less_step(rule, psum, records, target_unit, ctx, stats):
    """A process step's value = the whole plant's less a sibling step's
    (Brad 2026-10-28: alumina digestion = the whole Bayer refinery less the gas
    calciner). Scope: `archetype=` (the step filled), `whole=`, `less=`,
    `parameters=` (each with all its targets). Fills only cells the step does
    not hold; a target the sibling step does not hold is taken whole; a result
    below zero is floored at zero."""
    sc = _scope(rule)
    k, wh, ls = sc.get("archetype", ""), sc.get("whole", ""), sc.get("less", "")
    if k not in ctx["arch"]:
        return
    h = ctx["have"].setdefault(k, {})
    W, L = ctx["have"].get(wh, {}), ctx["have"].get(ls, {})
    for p in split_list(sc.get("parameters", "").replace("|", ";")):
        for (pp, t), w in list(W.items()):
            if pp != p or not _live(w) or (p, t) in h or t.startswith("as-sourced"):
                continue
            l = L.get((p, t))
            wc, wl, whi = _lohi(w)
            if _live(l) and l.get("unit") == w.get("unit"):
                lc = fnum(l["central"])
                c, lo, hi = max(wc - lc, 0.0), max(wl - lc, 0.0), max(whi - lc, 0.0)
                ins = [_input(w, f"{wh} {p}{'_' + t if t else ''}"), _input(l, f"less {ls}")]
                g = worst([rule["grade"], w.get("grade", "sourced"), l.get("grade", "sourced")])
            else:
                c, lo, hi = wc, wl, whi
                ins = [_input(w, f"{wh} {p}{'_' + t if t else ''}"), f"{ls} holds none: taken whole"]
                g = worst([rule["grade"], w.get("grade", "sourced")])
            row = _row(k, p, t, w.get("unit", ""), c, lo, hi, g, rule, ins,
                       "; ".join(dict.fromkeys(split_list(w.get("sources", "")) + split_list((l or {}).get("sources", "")))))
            psum.append(row)
            h[(p, t)] = row
            stats[rule["rule-id"]] = stats.get(rule["rule-id"], 0) + 1


def pm_sec_from_efficiency(rule, psum, records, target_unit, ctx, stats):
    """On the energy-service archetypes the rule names, the energy-use total =
    100 / the pooled efficiency, where the total rests on one or two examples and
    the efficiency on many (sanity-check run 2026-10-24, P-G6): the sourced total
    moves to target as-sourced, and the carriers in the total are scaled to it
    (pooled-apart carriers are left as they are)."""
    for a in records.archetypes:
        k = a["archetype"]
        h = ctx["have"].get(k, {})
        if not _in_scope(rule, a, h, ctx):
            continue
        tot, eff = h.get((SEC, "")), h.get(("efficiency", ""))
        if not (_live(tot) and _live(eff)) or tot.get("grade") != "sourced":
            continue
        ec, el, eh = _lohi(eff)
        oc = fnum(tot["central"])
        if not ec or ec <= 0 or not oc or oc <= 0:
            continue
        nc, nlo, nhi = 100 / ec, 100 / eh, 100 / el
        f = nc / oc
        g = worst([rule["grade"], eff.get("grade", "sourced")])
        ins = [_input(eff, f"{k} efficiency (n={eff.get('n', '')})"),
               f"replaces the sourced total {oc:.4g} (n={tot.get('n', '')})"]
        _put(psum, h, _row(k, SEC, "", tot["unit"], nc, nlo, nhi, g, rule, ins,
                           eff.get("sources", "")), stats, rule["rule-id"])
        for (q, t), r in list(h.items()):
            if q != SEC or not t or t.startswith("as-sourced") or is_component_target(t) or not _live(r):
                continue
            if "pooled apart" in (r.get("central-from") or ""):
                continue
            c, lo, hi = _lohi(r)
            _put(psum, h, _row(k, SEC, t, r["unit"], c * f, lo * f, hi * f,
                               worst([g, r.get("grade", "sourced")]), rule,
                               [_input(r, f"{k} {SEC}_{t}"), f"x {f:.4g} (the total on the efficiency)"],
                               r.get("sources", ""), f"scaled x {f:.3g} to the efficiency-based total"),
                 stats, rule["rule-id"] + ":carrier")


POST_METHODS = {"sec-from-efficiency": pm_sec_from_efficiency, "size-anchor": pm_size_anchor, "scale-ratio": pm_scale_ratio, "co2e-consistency": pm_co2e_consistency, "less-step": pm_less_step, "process-heat-efficiency": pm_process_heat_efficiency, "capacity-from-output": pm_capacity_from_output, "cf-from-output": pm_cf_from_output, "host-size": pm_host_size, "ccs-cost": pm_ccs_cost, "ccs-energy": pm_ccs_energy,
                "ccs-emissions": pm_ccs_emissions, "levelized": pm_levelized, "rebuild": pm_rebuild,
                "levelized-energy": pm_levelized_energy}


def rules():
    return read(FILL / "fill-rule.csv")


def judgements():
    return read(FILL / "fill-judgement.csv")


def assumptions():
    return read(FILL / "fill-assumption.csv")


def classes():
    return {r["equipment"]: r for r in read(FILL / "equipment-class.csv")}


def check(records, fail) -> None:
    """The fill records against the vocabularies (part of the build's gate)."""
    seen = set()
    groups = {r.get("group") for r in assumptions()}
    for r in rules():
        rid = r.get("rule-id", "")
        if rid in seen:
            fail["duplicate rule-id in fill-rule.csv"].append(rid)
        seen.add(rid)
        if r.get("status") not in STATUSES:
            fail["fill-rule status must be active or off"].append(rid)
        if r.get("grade") not in GRADES[1:] + [MARKER]:
            fail["fill-rule grade must be identity, estimated, proxy, guess or not-applicable"].append(rid)
        if r.get("method") not in METHODS and r.get("method") not in POST_METHODS:
            fail["fill-rule names an unknown method"].append(f"{rid}: {r.get('method')}")
        if r.get("method") == "marker" and r.get("grade") != MARKER:
            fail["marker rules must have grade not-applicable"].append(rid)
        if r.get("parameter") not in records.parameters and not (r.get("parameter") == "*" and r.get("method") == "range-midpoint"):
            fail["fill-rule names an unknown parameter"].append(f"{rid}: {r.get('parameter')}")
        if not (r.get("rationale") or "").strip():
            fail["fill-rule with no rationale"].append(rid)
        grp = dict(x.partition("=")[::2] for x in split_list(r.get("inputs", "")) if "=" in x).get("group")
        if grp and grp not in groups and r.get("status") == "active":
            fail["fill-rule names an assumption group with no rows"].append(f"{rid}: {grp}")
    hosts = read(FILL / "host-archetype.csv")
    seen_h = set()
    for h in hosts:
        if h.get("archetype") in seen_h:
            fail["duplicate archetype in host-archetype.csv"].append(h["archetype"])
        seen_h.add(h.get("archetype"))
    for h in hosts:
        if h.get("archetype") not in records.by_key:
            fail["host-archetype names an unknown archetype"].append(h.get("archetype", ""))
        if h.get("host") and h["host"] not in records.by_key:
            fail["host-archetype names an unknown host"].append(f"{h['archetype']}: {h['host']}")
        if h.get("status") not in ("active", "unmapped", "off"):
            fail["host-archetype status must be active, unmapped or off"].append(h.get("archetype", ""))
        if h.get("status") == "active" and not h.get("host"):
            fail["host-archetype active with no host"].append(h.get("archetype", ""))
        a_, b_ = records.by_key.get(h.get("archetype")), records.by_key.get(h.get("host"))
        if (h.get("status") == "active" and a_ and b_
                and a_.get("product-unit") != b_.get("product-unit")):
            # a host on another product unit would hand the variant its size and
            # carrier shares in the wrong unit (sanity-check run 2026-10-24, PB008)
            fail["host-archetype active across product units"].append(
                f"{h['archetype']} ({a_.get('product-unit')}) on {h['host']} ({b_.get('product-unit')})")
    cls = classes()
    for eq in records.equipment:
        if eq not in cls:
            fail["equipment with no row in records/fill/equipment-class.csv"].append(eq)
    for eq, r in cls.items():
        if eq not in records.equipment:
            fail["equipment-class names an unknown equipment"].append(eq)
        if r.get("vom-class") not in ("material", "minor", "none"):
            fail["equipment-class vom-class must be material, minor or none"].append(eq)
        if not (r.get("lifetime-class") or "").strip():
            fail["equipment-class with no lifetime-class"].append(eq)
    seen_a = set()
    for r in assumptions():
        aid = r.get("assumption-id", "")
        if aid in seen_a:
            fail["duplicate assumption-id in fill-assumption.csv"].append(aid)
        seen_a.add(aid)
        if r.get("parameter") not in records.parameters:
            fail["fill-assumption names an unknown parameter"].append(f"{aid}: {r.get('parameter')}")
        if fnum(r.get("value")) is None:
            fail["fill-assumption with no numeric value"].append(aid)
        lo, hi, v = fnum(r.get("low")), fnum(r.get("high")), fnum(r.get("value"))
        if v is not None and lo is not None and hi is not None and not lo <= v <= hi:
            fail["fill-assumption value outside its low-high"].append(aid)
        if not (r.get("source-id") or "").strip() or not (r.get("rationale") or "").strip():
            fail["fill-assumption needs a source-id and a rationale"].append(aid)
        for term in split_list(r.get("applies-to", "")):
            kind, _, val = term.partition(":")
            if kind not in ("all", "archetype", "equipment", "class", "vom-class", "family", "end-use",
                            "modifier", "product-unit"):
                fail["fill-assumption applies-to kind unknown"].append(f"{aid}: {term}")
            if kind == "archetype" and any(x not in records.by_key for x in val.split("|")):
                fail["fill-assumption applies to an unknown archetype"].append(f"{aid}: {val}")
            if kind == "equipment" and any(x not in records.equipment for x in val.split("|")):
                fail["fill-assumption applies to an unknown equipment"].append(f"{aid}: {val}")
    seen_j = set()
    for j in judgements():
        k = f"{j.get('archetype')}: {j.get('parameter')}" + (f"_{j['target']}" if j.get("target") else "")
        if k in seen_j:
            fail["duplicate archetype-parameter-target in fill-judgement.csv"].append(k)
        seen_j.add(k)
        if j.get("archetype") not in records.by_key:
            fail["fill-judgement names an unknown archetype"].append(k)
        if j.get("parameter") not in records.parameters:
            fail["fill-judgement names an unknown parameter"].append(k)
        if not (j.get("rationale") or "").strip() or not (j.get("author") or "").strip():
            fail["fill-judgement needs a rationale and an author"].append(k)
        if (j.get("status") or "active") not in JUDGEMENT_STATUSES:
            fail["fill-judgement status must be draft, accepted, changed or rejected"].append(k)
        if j.get("basis") and j["basis"] not in JUDGEMENT_BASES:
            fail["fill-judgement basis must be analogue-archetype, held-source-figure, engineering-relation, experience or added-carrier"].append(k)
        c, lo, hi = fnum(j.get("central")), fnum(j.get("low")), fnum(j.get("high"))
        if c is None or lo is None or hi is None or not lo <= c <= hi:
            fail["fill-judgement needs central within a stated low-high range"].append(k)


def _ctx(psum, records, facts, units):
    have = {}
    for r in psum:
        have.setdefault(r["archetype"], {})[(r["parameter"], r.get("target", ""))] = r
    increments, per_gj = {}, {}
    for r in facts:
        # a set-aside increment that is also ruled out does not count (2026-10-24, HPCH016)
        if (r.get("set-aside") == "needs-host-cost" and r.get("archetype")
                and not (r.get("excluded") or "").startswith("ruled out")
                and fnum(r.get("value-norm")) is not None):
            increments.setdefault((r["archetype"], r["parameter"]), []).append(r)
        if (r.get("archetype") and r["parameter"] in ("capex", "opex-fixed")
                and (r.get("excluded") or "").startswith("cost per GJ-per-yr cannot be put per t-per-yr")
                and (r.get("unit") or "").endswith("/GJ-per-yr") and fnum(r.get("value")) is not None):
            per_gj.setdefault((r["archetype"], r["parameter"]), []).append(r)
    # same-source variant/base ratios: (modifier, parameter) -> [(ratio, source, variant)]
    per = {}
    for r in facts:
        v = fnum(r.get("value-norm"))
        if v is None or r.get("excluded") or r.get("target-norm") or not r.get("archetype"):
            continue
        per.setdefault((r["source-id"], r["archetype"], r["parameter"], r["unit-norm"]), []).append(v)
    mod_ratios = {}
    for (sid, k, p, u), vs in per.items():
        parts = k.split("_")
        if len(parts) < 2:
            continue
        bv = per.get((sid, "_".join(parts[:-1]), p, u))
        if bv and median(bv) > 0:
            mod_ratios.setdefault((parts[-1], p), []).append((median(vs) / median(bv), sid, k))
    hosts = {h["archetype"]: h for h in read(FILL / "host-archetype.csv")}
    # increment/host ratios from the set-aside measure costs, by modifier
    inc_ratios = {}
    for (k, p), rs in increments.items():
        hm = hosts.get(k)
        if not hm or hm.get("status") != "active":
            continue
        host = have.get(hm["host"], {}).get((p, ""))
        if not _sourced(host) or fnum(host["central"]) <= 0:
            continue
        by_ex = {}
        for r in rs:
            if r.get("unit-norm") == host.get("unit"):
                by_ex.setdefault(r["example-id"], []).append(fnum(r["value-norm"]))
        if by_ex:
            inc = median(median(v) for v in by_ex.values())
            inc_ratios.setdefault((k.split("_")[-1], p), {})[k] = inc / fnum(host["central"])
    service = {}
    for s in records.service:
        service.setdefault(s["archetype"], set()).add(s.get("end-use", ""))
    addon_facts, pooled_facts, captured_facts = {}, {}, {}
    for r in facts:
        if (r.get("parameter") == "emissions-captured" and (r.get("target-norm") or r.get("target")) == "co2"
                and "_ccs-" in (r.get("archetype") or "") and not r.get("excluded")
                and fnum(r.get("value-norm")) is not None and fnum(r.get("value-norm")) > 0):
            # a capture variant's own sourced captured CO2, kept per source so a
            # same-source add-on can be rebased per t CO2 (sanity-check run P-D7, S103)
            captured_facts.setdefault((r["archetype"], r["source-id"]), []).append(r)
        if fnum(r.get("value-norm")) is None or not r.get("archetype") or r.get("target-norm"):
            continue
        if r.get("set-aside") == "ccs-add-on":
            addon_facts.setdefault((r["archetype"], r["parameter"]), []).append(r)
        elif not r.get("excluded"):
            pooled_facts.setdefault((r["archetype"], r["parameter"], r["source-id"]), []).append(r)
    ccs_nodes = {}
    for s_ in records.service:
        b = s_.get("branch", "")
        for node in ("diffuse", "concentrated"):
            if b.endswith(f"service.ccs.{node}"):
                ccs_nodes.setdefault(node, set()).add(s_["archetype"])
    from .common import REFERENCE
    return {"addon_facts": addon_facts, "pooled_facts": pooled_facts, "ccs_nodes": ccs_nodes,
            "captured_facts": captured_facts,
            "ccs_report": [], "unpriced": set(),
            "energy_prices": read(REFERENCE / "energy-price.csv"),
            "factors": convert.Factors(), "have": have, "increments": increments, "per_gj": per_gj,
            "arch": records.by_key, "mod_ratios": mod_ratios, "hosts": hosts, "inc_ratios": inc_ratios,
            "service": service, "classes": classes(), "assumptions": assumptions(),
            "modifiers": set(records.modifiers), "units": units,
            "feedstock": read(REFERENCE / "feedstock.csv")}


def _phase(rule):
    return _scope(rule).get("phase", "main")


def _run_rules(psum, records, target_unit, ctx, stats, phase="main"):
    have = ctx["have"]
    for rule in rules():
        if rule.get("status") != "active" or _phase(rule) != phase:
            continue
        if rule["method"] in POST_METHODS:
            POST_METHODS[rule["method"]](rule, psum, records, target_unit, ctx, stats)
            continue
        fn = METHODS[rule["method"]]
        for a in records.archetypes:
            h = have.setdefault(a["archetype"], {})
            if not _in_scope(rule, a, h, ctx):
                continue
            for f in fn(rule, a, h, ctx):
                p, t = f["parameter"], f["target"]
                if f.get("replace") is not None:
                    r0 = f["replace"]
                    grade = worst([rule["grade"]] + f["grades"])
                    if f.get("low") is not None and f.get("high") is not None:
                        lo_, hi_ = min(f["low"], f["high"]), max(f["low"], f["high"])
                        spread_ = abs(hi_ - lo_) > 1e-12 * max(1, abs(f["central"]))
                        r0.update({"low": _g(lo_) if spread_ else "", "high": _g(hi_) if spread_ else "",
                                   "sources": f.get("sources") or r0.get("sources", "")})
                    r0.update({"central": _g(f["central"]), "grade": grade, "method": rule["method"],
                               "rule-id": rule["rule-id"], "inputs": "; ".join(f["inputs"]),
                               "rationale": rule["rationale"],
                               "central-from": f"{grade}: {rule['formula']}; " + (r0.get("central-from") or "")})
                    stats[rule["rule-id"]] = stats.get(rule["rule-id"], 0) + 1
                    continue
                if (p, t) in h:
                    continue
                pdef = records.parameters.get(p, {})
                unit = target_unit(p, a["product-unit"], pdef.get("unit-default", ""))
                if f.get("marker"):
                    row = {"archetype": a["archetype"], "parameter": p, "target": t, "unit": unit,
                           "central": "", "low": "", "high": "", "n": 0, "n-external": "",
                           "central-from": f"{MARKER}: {rule['formula']}", "external-factors": "",
                           "sources": "", "examples": "", "grade": MARKER, "method": "marker",
                           "rule-id": rule["rule-id"], "inputs": "", "rationale": rule["rationale"]}
                else:
                    grade = worst([rule["grade"]] + f["grades"])
                    c, lo, hi = f["central"], f.get("low"), f.get("high")
                    spread = lo is not None and hi is not None and abs(hi - lo) > 1e-12 * max(1, abs(c))
                    row = {"archetype": a["archetype"], "parameter": p, "target": t, "unit": unit,
                           "central": _g(c), "low": _g(min(lo, hi)) if spread else "",
                           "high": _g(max(lo, hi)) if spread else "", "n": 0,
                           "n-external": "", "central-from": f"{grade}: {rule['formula']}"
                           + (f" ({f['note']})" if f.get("note") else ""),
                           "external-factors": "; ".join(f["external"]),
                           "sources": f["sources"], "examples": "",
                           "grade": grade, "method": rule["method"], "rule-id": rule["rule-id"],
                           "inputs": "; ".join(f["inputs"]), "rationale": rule["rationale"]}
                psum.append(row)
                h[(p, t)] = row
                stats[rule["rule-id"]] = stats.get(rule["rule-id"], 0) + 1


def _run_judgements(psum, records, ctx, stats):
    have = ctx["have"]
    for j in judgements():
        if (j.get("status") or "active") not in JUDGEMENT_LIVE:
            continue
        h = have.setdefault(j["archetype"], {})
        key = (j["parameter"], j.get("target", ""))
        dead = None
        if key in h:
            r0 = h[key]
            if not _live(r0) and (fnum(r0.get("low")) is None or fnum(r0.get("high")) is None):
                dead = r0        # a pooled row with no central and no usable range: the guess replaces it
            else:
                # a sourced or calculated value now exists: the guess is stale, not used
                stats["guess displaced"] = stats.get("guess displaced", 0) + 1
                continue
        a = records.by_key[j["archetype"]]
        pdef = records.parameters.get(j["parameter"], {})
        from .normalise import target_unit
        # the target unit of the parameter on this archetype; the judgement's own
        # `unit` is what the author meant and is checked, not copied
        unit = target_unit(j["parameter"], a["product-unit"], pdef.get("unit-default", ""))
        row = {"archetype": j["archetype"], "parameter": j["parameter"], "target": j.get("target", ""),
               "unit": unit, "central": j["central"], "low": j["low"], "high": j["high"],
               "n": 0, "n-external": "", "central-from": "guess: " + j["rationale"],
               "external-factors": "", "sources": "", "examples": "", "grade": "guess",
               "method": "judgement", "rule-id": "",
               "inputs": " ".join(x for x in (j.get("basis", ""), j["author"], j.get("date", ""),
                                              f"[{j.get('status')}]" if j.get("status") else "") if x),
               "rationale": j["rationale"]}
        if dead is not None:
            row["central-from"] = f"guess (replaces a pooled row with no usable value: {dead.get('central-from')}): " + j["rationale"]
            psum.remove(dead)
        psum.append(row)
        h[key] = row
        stats["guess"] = stats.get("guess", 0) + 1


def _sum_carriers(psum, records, ctx, stats, create=True):
    """The energy-use total includes every carrier the archetype holds, sourced and
    estimated (modeller 2026-10-21): where the level-1 carriers (electricity, steam,
    ... plus `fuel`, itself at least the sum of the named fuels) add up to more
    than the total, the total becomes their sum; an archetype with carriers and no
    total gets one. A total above its carriers (the carriers are partial) stays.
    Returns the archetypes whose total changed."""
    from .carriers import NO_HV_CARRIERS, FUEL_AGG
    changed = set()
    have = ctx["have"]
    for a in records.archetypes:
        k = a["archetype"]
        h = have.get(k, {})
        rows = {t: r for (q, t), r in h.items()
                if q == SEC and t and t not in ("total", "not-stated", "component") and not is_component_target(t)
                and not t.startswith("as-sourced") and _live(r)}
        rows = _adding_carriers(rows)
        if not rows:
            continue
        tot = h.get((SEC, ""))
        if tot is not None and (tot.get("method") in ("bucket-saving", "host-split", "host-copy")
                                or (tot.get("rule-id") or "").startswith(("ccs-", "var-", "est-variant"))):
            # a variant whose total and carriers one rule wrote together from its
            # host or base (which is itself summed): left as the rule wrote it
            continue
        named = {t: r for t, r in rows.items() if t not in NO_HV_CARRIERS and t != FUEL_AGG}
        nsum = [sum(_lohi(r)[i] for r in named.values()) for i in range(3)] if named else None
        fuel = rows.get(FUEL_AGG)
        if fuel is not None and nsum and nsum[0] > 1.01 * (fnum(fuel["central"]) or 0):
            ins = [f"{t} {_g(_lohi(r)[0])} ({r.get('grade') or 'sourced'})" for t, r in sorted(named.items())]
            fuel.update({"central": _g(nsum[0]), "low": _g(nsum[1]), "high": _g(nsum[2]),
                         "grade": worst([fuel.get("grade") or "sourced"] + [r.get("grade") or "sourced"
                                                                              for r in named.values()]),
                         "central-from": "sum of the named fuels, sourced and estimated: " + "; ".join(ins)
                         + f" (was {fuel['central']}: {fuel.get('central-from') or ''})"})
            stats["fuel raised to its named fuels"] = stats.get("fuel raised to its named fuels", 0) + 1
        l1 = {t: r for t, r in rows.items() if t in NO_HV_CARRIERS}
        # a generic `heat` row that other carriers split (a judgement giving "the
        # row's heat figure" as steam and natural gas) is not added on top of them
        if "heat" in l1:
            hv = fnum(l1["heat"]["central"]) or 0
            split = [r for t, r in rows.items() if t not in ("heat", "electricity", FUEL_AGG)
                     and "heat figure" in (r.get("central-from") or "")]
            others = [r for t, r in rows.items() if t not in ("heat", "electricity", FUEL_AGG)
                      and t not in NO_HV_CARRIERS or t == "steam"]
            for grp in (split, [r for r in others if r is not l1.get("heat")]):
                gs = sum(fnum(r["central"]) or 0 for r in grp)
                if grp and hv > 0 and abs(gs - hv) <= 0.02 * hv:
                    del l1["heat"]
                    stats["heat rows split by other carriers"] = stats.get("heat rows split by other carriers", 0) + 1
                    break
        if fuel is not None:
            l1[FUEL_AGG] = fuel
        elif named:
            l1.update(named)
        if not l1:
            continue
        lsum = [sum(_lohi(r)[i] for r in l1.values()) for i in range(3)]
        grades = [r.get("grade") or "sourced" for r in l1.values()]
        ins = "; ".join(f"{t} {_g(_lohi(r)[0])} ({r.get('grade') or 'sourced'})" for t, r in sorted(l1.items()))
        tot = h.get((SEC, ""))
        if tot is not None and tot.get("grade") == MARKER:
            continue
        if tot is not None and "DISAGREES" in (tot.get("central-from") or ""):
            # the carriers are on another scope than the total (pipeline/carriers.py):
            # listed for review, not summed
            ctx.setdefault("sum_disagree", set()).add(k)
            continue
        if tot is None and not create:
            continue
        if tot is None:
            row = {"archetype": k, "parameter": SEC, "target": "", "unit": next(iter(l1.values()))["unit"],
                   "central": _g(lsum[0]), "low": _g(lsum[1]) if lsum[1] != lsum[0] else "",
                   "high": _g(lsum[2]) if lsum[2] != lsum[0] else "", "n": 0, "n-external": "",
                   "central-from": "sum of the carriers, sourced and estimated: " + ins,
                   "external-factors": "", "sources": "", "examples": "", "grade": worst(grades),
                   "method": "sum", "rule-id": "id-sec-total-from-carriers", "inputs": ins,
                   "rationale": "The energy-use total includes every carrier, sourced and estimated (modeller 2026-10-21)."}
            psum.append(row)
            h[(SEC, "")] = row
            changed.add(k)
            stats["totals from the carrier sum"] = stats.get("totals from the carrier sum", 0) + 1
            continue
        tv = fnum(tot.get("central"))
        if tv is None or lsum[0] <= 1.01 * tv:
            continue
        # a total no guess fed is not raised by guessed carriers (modeller
        # 2026-10-21): the guessed carriers are scaled down to the room the other
        # carriers leave, so the split still sums to the total
        # (a carrier another example names, added by hand with basis added-carrier
        # (S160, 2026-10-28), adds to the total rather than fitting inside it)
        added = any((r.get("inputs") or "").startswith("added-carrier") for r in rows.values())
        guessed = {t: r for t, r in l1.items() if (r.get("grade") or "sourced") == "guess"
                   and not (r.get("inputs") or "").startswith("added-carrier")
                   and not (added and t == FUEL_AGG)}
        firm = sum(_lohi(r)[0] for t, r in l1.items() if t not in guessed)
        gsum = sum(_lohi(r)[0] for r in guessed.values())
        # (a total that is only the one carrier any example names is not a whole
        # total, so guessed carriers may still raise it)
        whole = not (tot.get("central-from") or "").startswith("the pooled ")
        if whole and (tot.get("grade") or "sourced") != "guess" and guessed and gsum > 0 and firm <= 1.01 * tv:
            room = tv - firm
            f = room / gsum if room > 1e-3 * tv else 0.0
            scale = dict(guessed)
            if FUEL_AGG in guessed:
                scale.update({t: r for t, r in named.items() if (r.get("grade") or "sourced") == "guess"})
            for t, r in scale.items():
                c, lo, hi = _lohi(r)
                r.update({"central": _g(c * f), "low": _g(lo * f) if r.get("low") not in ("", None) else "",
                          "high": _g(hi * f) if r.get("high") not in ("", None) else "",
                          "central-from": (r.get("central-from") or "") + f" (scaled x{f:.3g} to fit the {_g(tv)} total, 2026-10-21)"})
            stats["guessed carriers scaled to the total"] = stats.get("guessed carriers scaled to the total", 0) + 1
            continue
        tot.update({"central": _g(lsum[0]), "low": _g(lsum[1]), "high": _g(lsum[2]),
                    "grade": worst([tot.get("grade") or "sourced"] + grades),
                    "central-from": "sum of the carriers, sourced and estimated: " + ins
                    + f" (was {_g(tv)}: {tot.get('central-from') or ''})",
                    "inputs": ins})
        changed.add(k)
        stats["totals raised to the carrier sum"] = stats.get("totals raised to the carrier sum", 0) + 1
    if changed:
        # an efficiency derived from the old total is stale: drop it so the
        # identity rule derives it again from the new total
        for k in changed:
            h = have.get(k, {})
            for key, r in list(h.items()):
                if key[0] == "efficiency" and r.get("rule-id") == "id-efficiency-from-sec":
                    del h[key]
                    if r in psum:
                        psum.remove(r)
    return changed


def _variant_above_host(ctx):
    """An efficiency variant must not use more energy per unit of product than the
    plant it improves (sanity-check 2026-10-23, MCF005). Flags each one whose
    energy-use total is above its host's or base archetype's in `central-from`;
    returns them for the build log. Values are not changed: the sources or the
    comparison need a look."""
    out = []
    for k, a in ctx["arch"].items():
        if not any(m.startswith("efficiency") for m in split_list(a.get("modifiers", ""))):
            continue
        hrow = ctx["hosts"].get(k, {})
        host = hrow.get("host") if hrow.get("status") == "active" else None
        host = host or _base_of(a, ctx)
        if not host:
            continue
        v, hv = ctx["have"].get(k, {}).get((SEC, "")), ctx["have"].get(host, {}).get((SEC, ""))
        if not (_live(v) and _live(hv)) or v.get("unit") != hv.get("unit"):
            continue
        vc, hc = fnum(v["central"]), fnum(hv["central"])
        if hc and vc > hc * 1.02:
            v["central-from"] = (v.get("central-from") or "") + (
                f"; ABOVE ITS HOST: {vc:.4g} against {host} {hc:.4g} (check the sources' scope)")
            out.append(f"{k} {vc:.4g} > {host} {hc:.4g}")
    return sorted(out)


def apply(psum, records, target_unit, facts=(), units=None) -> dict:
    """Grade the pooled rows, then add the fills: rules, judgements, rules again.
    Returns counts by grade and rule."""
    for r in psum:
        r.update({"grade": "sourced", "method": "pooled", "rule-id": "", "inputs": "",
                  "rationale": ""})
    # one set of carriers that adds up, on the most likely fuel (Brad 2026-11-07)
    from . import fuel_choice
    hosts_ = {h["archetype"]: h for h in read(FILL / "host-archetype.csv") if h.get("status") == "active"}
    _central_choices(psum)
    fc = fuel_choice.apply(psum, records, hosts_, stage="pooled")
    ctx = _ctx(psum, records, facts, units)
    stats = {"_backtest": backtest(records, [r for r in rules() if r.get("status") == "active"],
                                   ctx["have"], ctx)}
    _run_rules(psum, records, target_unit, ctx, stats)
    _run_judgements(psum, records, ctx, stats)
    # the judgements' carriers: chosen fuel only; the sums are the fill's, then closed below
    fc2 = fuel_choice.apply(psum, records, hosts_, have=ctx["have"], stage="judgements", closure=False)
    stats["_fuel_choice"] = {k: fc.get(k, 0) + fc2.get(k, 0) for k in {**fc, **fc2}}
    # raise totals now (identity rules then derive from them); create missing
    # totals only after the rules, so a total from efficiency is not pre-empted
    _sum_carriers(psum, records, ctx, stats, create=False)
    _run_rules(psum, records, target_unit, ctx, stats)
    if _sum_carriers(psum, records, ctx, stats):
        _run_rules(psum, records, target_unit, ctx, stats)
    # what the fill's own carrier sums left unclosed (fuel_choice, 2026-11-07)
    fc3 = fuel_choice.apply(psum, records, hosts_, have=ctx["have"], stage="closure", choose=False)
    for k, v in fc3.items():
        stats["_fuel_choice"][k] = stats["_fuel_choice"].get(k, 0) + v
    _efficiency_fixes(psum, records, ctx, stats)
    _emission_fixes(psum, records, ctx, stats)
    _size_closure(psum, ctx, stats)
    _run_rules(psum, records, target_unit, ctx, stats)
    # variants made consistent with their hosts (modeller 2026-10-10), then the
    # calculated levelized costs, which read every other cell
    _run_rules(psum, records, target_unit, ctx, stats, phase="post")
    _ccs_efficiency(psum, ctx, stats)
    stats["_fuel_choice"]["zero carrier guesses dropped at the end"] = fuel_choice.drop_zero_guesses(psum, ctx["have"])
    _run_rules(psum, records, target_unit, ctx, stats, phase="final")
    stats["_above_host"] = _variant_above_host(ctx)
    stats["_comb_conflict"] = sorted(set(ctx.get("comb_conflict", [])))
    stats["_comb_conflict_old"] = sorted(set(ctx.get("comb_conflict_old", [])))
    stats["_total_below_parts"] = sorted(set(ctx.get("total_below_parts", [])))
    stats["_ccs"] = ctx["ccs_report"]
    stats["_sum_disagree"] = sorted(ctx.get("sum_disagree", ()))
    stats["_unpriced"] = sorted(ctx["unpriced"])
    return stats


# ---- coverage: the CIMS-relevant cells per archetype and how each was reached ----

CIMS_CELLS = [("capex", ("capex",)), ("opex-fixed", ("opex-fixed",)),
              ("opex-variable", ("opex-variable", "opex-total")),
              ("energy", (SEC, "efficiency")), ("lifetime", ("lifetime",)),
              ("annual-output", ("annual-output",)),
              ("emissions", ("emissions-combustion", "emissions-process", "emissions-total"))]


def coverage(psum, records):
    """One row per archetype: for each CIMS cell the grade (or marker) of the
    value the database holds, or `missing`."""
    by = {}
    for r in psum:
        by.setdefault(r["archetype"], {})[(r["parameter"], r.get("target", ""))] = r
    rows = []
    for a in records.archetypes:
        h = by.get(a["archetype"], {})
        row = {"archetype": a["archetype"], "family": a["family"], "product-unit": a["product-unit"],
               "trl": "sourced" if (a.get("trl") or "").strip() else "missing"}
        for name, params in CIMS_CELLS:
            best = ""
            for (p, t), r in h.items():
                if p not in params:
                    continue
                if name == "emissions" and t not in ("co2", "co2e"):
                    continue
                if name == "energy" and t:
                    continue
                g = r.get("grade", "sourced")
                if g == MARKER:
                    best = best or MARKER
                elif _live(r) and (not best or best == MARKER or GRADES.index(g) < GRADES.index(best)):
                    best = g
            row[name] = best or "missing"
        rows.append(row)
    return rows


# ---- emission fixes (consistency check 2026-11-07) ----------------------------

def _central_choices(psum):
    """records/reference/central-choice.csv: a pooled central replaced by the value
    typical for the technology in Canadian industry, from a named source (Brad
    2026-11-09: the precalciner kiln's fuel from GCCA GNR's Canadian precalciner
    kilns, not the European best-practice sources that set the pooled median).
    The pooled row moves to target as-sourced, so it stays visible; the fuel split,
    the totals and the CO2 identities then follow the chosen value."""
    from .common import REFERENCE
    for c in read(REFERENCE / "central-choice.csv"):
        key = (c["archetype"], c["parameter"], c.get("target", ""))
        old = next((r for r in psum if (r["archetype"], r["parameter"], r.get("target", "")) == key), None)
        if old is None:
            raise SystemExit(f"central-choice.csv: no pooled row for {key}")
        new = dict(old)
        new.update({"central": c["central"], "low": c.get("low", ""), "high": c.get("high", ""),
                    "n": 1, "sources": c["source-id"], "examples": "",
                    "central-from": f"CENTRAL CHOICE (records/reference/central-choice.csv, {c['decided']}): "
                                    f"{c['basis']}; the pooled value {old.get('central')} moved to as-sourced",
                    "method": "central-choice"})
        old["target"] = _as_sourced(old.get("target", ""))
        old["central-from"] = "pooled value set aside for a central choice: " + (old.get("central-from") or "")
        psum.append(new)


def _emission_fixes(psum, records, ctx, stats):
    """Make each archetype's emissions agree with its own energy rows and totals,
    per records/reference/emission-fix.csv (the calls are a Fable review's, made
    for Brad under his scope-1 rules). Actions:

      identity              combustion CO2 = the named fossil fuels x their ECCC
                            factors (0 where none is fossil); total = parts
      identity-capture      the same as gross; captured = share-to-capture x
                            gross; total = gross - captured
      identity-remainder    combustion = identity; process = total - combustion
      zero                  combustion CO2 0 (bought heat or electricity only);
                            total = parts
      to-process            the combustion CO2 becomes process CO2
      remainder-process     process = total - combustion
      remainder-combustion  combustion = total - process
      parts-total           total = combustion + process
      reductant (arg fuels) combustion = the other fuels x factors; process =
                            the reductant fuels x factors; total = parts
      net-identity          total = combustion + process - captured
      captured-biogenic     (a capture variant) its captured CO2 is biogenic
                            only; read by the capture-emissions rule

    A value replaced moves to target as-sourced(-co2). An efficient variant with
    no row of its own takes its host's action if that is `identity`, or its
    host's combustion CO2 scaled by their fuel use otherwise."""
    from .common import REFERENCE
    fix = {r["archetype"]: r for r in read(REFERENCE / "emission-fix.csv")}
    ctx["captured_biogenic"] = {k for k, r in fix.items() if r["action"] == "captured-biogenic"}
    hosts = {k: h["host"] for k, h in ctx["hosts"].items() if h.get("status") == "active" and h.get("host")}
    rule = {"rule-id": "emission-fix", "method": "emission-fix", "formula": "emissions made consistent "
            "with the archetype's energy rows and totals (records/reference/emission-fix.csv)",
            "rationale": "Consistency check 2026-11-07: an archetype's combustion, process and total CO2 "
            "must agree with each other and with the fuel it burns."}

    def fuel_co2(h, only=None, skip=()):
        c = lo = hi = 0.0
        ins = []
        for (p, t), r in h.items():
            if p != SEC or not t or t in skip or (only and t not in only) or not _live(r):
                continue
            if t not in FOSSIL:
                continue
            f = ctx["factors"].rows.get(f"cf-ef-co2-{t}")
            v = fnum(r.get("central"))
            if not f or v is None:
                continue
            ef = fnum(f["factor"])
            c += v * ef
            lo += (fnum(r.get("low")) if fnum(r.get("low")) is not None else v) * (fnum(f.get("factor-low")) or ef)
            hi += (fnum(r.get("high")) if fnum(r.get("high")) is not None else v) * (fnum(f.get("factor-high")) or ef)
            ins.append(f"{t} {v:.4g} GJ x {ef:.4g} ({f['factor-id']})")
        return c, lo, hi, ins

    def setv(k, h, p, c, lo, hi, ins, note, grade="identity"):
        a = ctx["arch"][k]
        lo, hi = min(lo, hi, c), max(lo, hi, c)
        unit = _unit_for(h, p, a)
        _put(psum, h, _row(k, p, "co2", unit, c, lo, hi, grade, rule, ins, "", note), stats, "emission-fix")

    def val(h, p):
        r = h.get((p, "co2"))
        return (_lohi(r) if _live(r) else (0.0, 0.0, 0.0)), r

    def parts_total(k, h, note):
        (cc, cl, ch), _ = val(h, "emissions-combustion")
        (pc, pl, ph), pr = val(h, "emissions-process")
        setv(k, h, "emissions-total", cc + pc, cl + pl, ch + ph,
             ["emissions-combustion_co2 + emissions-process_co2"], note)

    done = 0
    order = sorted(ctx["arch"], key=lambda k: (k in hosts, k))
    for k in order:
        h = ctx["have"].setdefault(k, {})
        r = fix.get(k)
        note = (r or {}).get("note", "")
        act = (r or {}).get("action", "")
        if not r and k in hosts and hosts[k] in fix and "_ccs-" not in k \
                and ctx["arch"][k].get("type") == "efficient variant":
            ha = fix[hosts[k]]["action"]
            act = ha if ha in ("identity", "zero") else "host-scaled"
            note = f"as its host {hosts[k]} ({fix[hosts[k]]['action']}): {fix[hosts[k]]['note']}"
        if not act:
            cr = h.get(("emissions-combustion", "co2"))
            if _live(cr) and cr.get("method") == "fuel-emission-factor" and "_ccs-" not in k:
                c, lo, hi, ins = fuel_co2(h)
                if c > 0 and abs(c - (fnum(cr["central"]) or 0)) > 1e-3 * max(c, 1e-9):
                    setv(k, h, "emissions-combustion", c, lo, hi, ins,
                         "recomputed after the energy rows changed (consistency fixes 2026-11-07)")
                    tr = h.get(("emissions-total", "co2"))
                    if _live(tr) and tr.get("method") in ("sum", "emission-fix"):
                        parts_total(k, h, "recomputed with the combustion CO2")
                    done += 1
            continue
        if act == "captured-biogenic":
            continue
        g = "identity"
        if act in ("identity", "identity-capture", "identity-remainder"):
            c, lo, hi, ins = fuel_co2(h)
            setv(k, h, "emissions-combustion", c, lo, hi, ins, note)
            if act == "identity":
                parts_total(k, h, note)
            elif act == "identity-remainder":
                (tc, tl, th), tr = val(h, "emissions-total")
                if _live(tr):
                    setv(k, h, "emissions-process", max(tc - c, 0), max(tl - lo, 0), max(th - hi, 0),
                         ["emissions-total_co2 - emissions-combustion_co2"], note, "estimated")
            else:
                s = h.get(("share-to-capture", "co2"))
                sh = (fnum(s.get("central")) or 100) / 100 if _live(s) else 1.0
                setv(k, h, "emissions-captured", c * sh, lo * sh, hi * sh,
                     [f"share-to-capture {sh:.1%} x gross combustion CO2"], note, "estimated")
                setv(k, h, "emissions-total", c * (1 - sh), lo * (1 - sh), hi * (1 - sh),
                     ["gross combustion CO2 - captured"], note, "estimated")
        elif act == "zero":
            setv(k, h, "emissions-combustion", 0, 0, 0, ["no fuel burned at the unit"], note)
            parts_total(k, h, note)
        elif act == "to-process":
            (cc, cl, ch), cr = val(h, "emissions-combustion")
            setv(k, h, "emissions-process", cc, cl, ch, [_input(cr, f"{k} emissions-combustion_co2")] if cr else [],
                 note, (cr or {}).get("grade", "sourced") if cr and cr.get("grade") != "sourced" else "estimated")
            setv(k, h, "emissions-combustion", 0, 0, 0, ["moved to process CO2"], note)
            parts_total(k, h, note)
        elif act in ("remainder-process", "remainder-combustion"):
            (tc, tl, th), tr = val(h, "emissions-total")
            other = "emissions-combustion" if act == "remainder-process" else "emissions-process"
            target = "emissions-process" if act == "remainder-process" else "emissions-combustion"
            (oc, ol, oh), _ = val(h, other)
            if _live(tr):
                setv(k, h, target, max(tc - oc, 0), max(tl - ol, 0), max(th - oh, 0),
                     [f"emissions-total_co2 - {other}_co2"], note, "estimated")
        elif act == "parts-total":
            parts_total(k, h, note)
        elif act == "reductant":
            red = set(split_list((r.get("arg") or "").replace("|", ";")))
            c, lo, hi, ins = fuel_co2(h, skip=red)
            setv(k, h, "emissions-combustion", c, lo, hi, ins, note)
            c2, lo2, hi2, ins2 = fuel_co2(h, only=red)
            setv(k, h, "emissions-process", c2, lo2, hi2, ins2, note, "estimated")
            parts_total(k, h, note)
        elif act == "share-capture":
            (cc, cl, ch), _ = val(h, "emissions-combustion")
            (pc, pl, ph), _ = val(h, "emissions-process")
            st = h.get(("share-to-capture", "co2"))
            sh = (fnum(st.get("central")) or 100) / 100 if _live(st) else 1.0
            g0, gl, gh = cc + pc, cl + pl, ch + ph
            setv(k, h, "emissions-captured", g0 * sh, gl * sh, gh * sh,
                 [f"share-to-capture {sh:.1%} x gross (combustion + process)"], note, "estimated")
            setv(k, h, "emissions-total", g0 * (1 - sh), gl * (1 - sh), gh * (1 - sh),
                 ["gross - captured"], note, "estimated")
        elif act == "net-identity":
            (cc, cl, ch), _ = val(h, "emissions-combustion")
            (pc, pl, ph), _ = val(h, "emissions-process")
            (xc, xl, xh), _ = val(h, "emissions-captured")
            setv(k, h, "emissions-total", max(cc + pc - xc, 0), max(cl + pl - xh, 0), max(ch + ph - xl, 0),
                 ["combustion + process - captured"], note)
        elif act == "host-scaled":
            hh = ctx["have"].get(hosts[k], {})
            hc = hh.get(("emissions-combustion", "co2"))
            hf, vf = hh.get((SEC, "fuel")), h.get((SEC, "fuel"))
            if _live(hc) and _live(hf) and _live(vf) and (fnum(hf["central"]) or 0) > 0:
                f = fnum(vf["central"]) / fnum(hf["central"])
                c, lo, hi = (x * f for x in _lohi(hc))
                setv(k, h, "emissions-combustion", c, lo, hi,
                     [_input(hc, f"{hosts[k]} emissions-combustion_co2"), f"x fuel ratio {f:.4g}"], note, "estimated")
        done += 1
    stats["emission fixes"] = done


def _unit_for(h, p, a):
    for (q, t), r in h.items():
        if q in ("emissions-combustion", "emissions-process", "emissions-total", "emissions-captured",
                 "emissions-biomass") and (r.get("unit") or "").startswith("t/"):
            return r["unit"]
    return f"t/{a['product-unit']}"


def _efficiency_fixes(psum, records, ctx, stats):
    """Make efficiency, energy use and feed agree, per
    records/reference/efficiency-fix.csv (a Fable review's calls for Brad,
    2026-11-07). Conversion efficiency = product energy / (energy use + feed).

      feed-identity (arg carrier)   feed = 100 / efficiency - energy use
      set-aside-energy (arg)        energy rows to as-sourced; feed = 100 / efficiency
      eff-identity (arg optional)   efficiency = 100 / (energy use + feed); with an
                                    arg and no feed row, a feed of 1 GJ/GJ is written
      sec-identity                  energy use = 100 / efficiency, carriers scaled
      chp-from-overall              electricity = overall / (1 + HPR), heat = overall
                                    x HPR / (1 + HPR); energy = 100 / heat efficiency
    """
    from .common import REFERENCE
    rule = {"rule-id": "efficiency-fix", "method": "efficiency-fix", "formula": "efficiency, energy use and "
            "feed made consistent (records/reference/efficiency-fix.csv)",
            "rationale": "Consistency check 2026-11-07: an archetype's efficiency, energy use and feed must agree."}
    n = 0
    for r in read(REFERENCE / "efficiency-fix.csv"):
        k = r["archetype"]
        if k not in ctx["arch"]:
            continue
        h = ctx["have"].setdefault(k, {})
        act, arg, note = r["action"], (r.get("arg") or "").strip(), r.get("note", "")
        chp = act == "chp-from-overall"
        effk = ("efficiency-chp", "") if chp and _live(h.get(("efficiency-chp", ""))) else ("efficiency", "")
        eff, sec = h.get(effk), h.get((SEC, ""))
        pu = ctx["arch"][k]["product-unit"]

        def feed_sum():
            return sum(fnum(x.get("central")) or 0 for (p, t), x in h.items()
                       if p == "feedstock-consumption" and t and _live(x))

        def put(p, t, c, lo, hi, unit, ins, grade="identity"):
            lo, hi = (min(lo, hi, c), max(lo, hi, c)) if lo is not None and hi is not None else (None, None)
            _put(psum, h, _row(k, p, t, unit, c, lo, hi, grade, rule, ins, "", note), stats, "efficiency-fix")

        def scale_sec(f, why):
            for (p, t), x in list(h.items()):
                if p == SEC and not (t or "").startswith("as-sourced") and _live(x):
                    c, lo, hi = _lohi(x)
                    put(SEC, t, c * f, lo * f, hi * f, x["unit"], [_input(x, f"{k} {SEC}_{t}"), why],
                        worst(["identity", x.get("grade") or "sourced"]))

        sv = fnum(sec.get("central")) if _live(sec) else 0.0
        if act in ("feed-identity", "set-aside-energy") and _live(eff):
            if act == "set-aside-energy":
                for (p, t), x in list(h.items()):
                    if p == SEC and not (t or "").startswith("as-sourced") and _live(x):
                        del h[(p, t)]
                        x["target"] = _as_sourced(t)
                        x["central-from"] = "set aside: " + note + " " + (x.get("central-from") or "")
                        h[(p, x["target"])] = x
                sv = 0.0
            ec, el, eh = _lohi(eff)
            f = 100 / ec - sv
            put("feedstock-consumption", arg, f, 100 / eh - sv, 100 / el - sv, f"GJ/{pu}",
                [_input(eff, f"{k} efficiency"), f"less energy use {sv:.4g}"])
        elif act == "eff-identity" and _live(sec):
            if arg and not any(p == "feedstock-consumption" and _live(x) for (p, t), x in h.items()):
                put("feedstock-consumption", arg, 1.0, 1.0, 1.0, f"GJ/{pu}", ["feed of 1 GJ per GJ of product"])
            d = sv + feed_sum()
            if d > 0:
                lo_, hi_ = _lohi(sec)[1:]
                put("efficiency", "", 100 / d, 100 / (hi_ + feed_sum()) if hi_ else None,
                    100 / (lo_ + feed_sum()) if lo_ else None, "%",
                    [_input(sec, f"{k} {SEC}"), f"feed {feed_sum():.4g}"])
                if _live(h.get(("efficiency", "heat"))):
                    put("efficiency", "heat", 100 / d, None, None, "%", ["as the overall efficiency"])
        elif act == "sec-identity" and _live(eff) and sv > 0:
            target = 100 / fnum(eff["central"])
            scale_sec(target / sv, f"x {target / sv:.4g}: energy use = 100 / efficiency {fnum(eff['central']):.4g}")
        elif chp and _live(eff) and _live(h.get(("heat-to-power-ratio", ""))):
            o = fnum(eff["central"])
            hpr = fnum(h[("heat-to-power-ratio", "")]["central"])
            e_el, e_ht = o / (1 + hpr), o * hpr / (1 + hpr)
            put("efficiency", "electricity", e_el, None, None, "%", [_input(eff, "overall"), f"HPR {hpr:.4g}"])
            put("efficiency", "heat", e_ht, None, None, "%", [_input(eff, "overall"), f"HPR {hpr:.4g}"])
            if sv > 0:
                scale_sec((100 / e_ht) / sv, f"energy per GJ heat = 100 / heat efficiency {e_ht:.4g}")
        else:
            continue
        n += 1
    stats["efficiency fixes"] = n



def _ccs_efficiency(psum, ctx, stats):
    """A capture variant's efficiency is net of capture: 100 / (energy use + feed),
    the capture steam and electricity included; its host keeps its own (Fable
    review E7 for Brad, 2026-11-07; extended by Claude from the energy-service
    variants to every capture variant with an efficiency). CHP variants keep
    their stream efficiencies."""
    rule = {"rule-id": "ccs-efficiency", "method": "ccs-efficiency",
            "formula": "efficiency net of capture = 100 / (energy use + feed)",
            "rationale": "Consistency check 2026-11-07, E7."}
    n = 0
    for k in ctx["arch"]:
        if "_ccs-" not in k or CHP.search(ctx["arch"][k].get("label", "")) or "chp" in k:
            continue
        h = ctx["have"].get(k, {})
        eff, sec = h.get(("efficiency", "")), h.get((SEC, ""))
        if not (_live(eff) and _live(sec)):
            continue
        feed = sum(fnum(x.get("central")) or 0 for (p, t), x in h.items()
                   if p == "feedstock-consumption" and t and _live(x))
        hm = (ctx["hosts"].get(k) or {}).get("host")
        if not feed and hm:
            # a conversion variant with no feed row of its own makes its product from
            # its host's feed
            feed = sum(fnum(x.get("central")) or 0 for (p, t), x in ctx["have"].get(hm, {}).items()
                       if p == "feedstock-consumption" and t and _live(x))
        d = (fnum(sec["central"]) or 0) + feed
        if d <= 0:
            continue
        v = 100 / d
        if abs(v - (fnum(eff["central"]) or 0)) < 0.01:
            continue
        for t in ("", "heat"):
            if t == "" or _live(h.get(("efficiency", t))):
                _put(psum, h, _row(k, "efficiency", t, "%", v, None, None, "identity", rule,
                                   [_input(sec, f"{k} {SEC}"), f"feed {feed:.4g}"], "",
                                   "net of capture; the host keeps its own efficiency"), stats, "ccs-efficiency")
        n += 1
    stats["capture-variant efficiencies net of capture"] = n
    # a capture variant is its host's plant plus the add-on (Brad 2026-10-10): it
    # takes the host's lifetime (Fable review F5, 2026-11-07)
    m = 0
    lrule = {"rule-id": "ccs-lifetime", "method": "host-copy", "formula": "the host's lifetime",
             "rationale": "Consistency check 2026-11-07, F5: a capture variant is its host's plant."}
    for k in ctx["arch"]:
        hm = (ctx["hosts"].get(k) or {}).get("host")
        if "_ccs-" not in k or not hm or (ctx["hosts"].get(k) or {}).get("status") != "active":
            continue
        h, hh = ctx["have"].get(k, {}), ctx["have"].get(hm, {})
        hl, vl = hh.get(("lifetime", "")), h.get(("lifetime", ""))
        if _live(hl) and _live(vl) and abs(fnum(hl["central"]) - fnum(vl["central"])) > 1e-9:
            c, lo, hi = _lohi(hl)
            _put(psum, h, _row(k, "lifetime", "", hl["unit"], c, lo, hi,
                               worst(["estimated", hl.get("grade") or "sourced"]), lrule,
                               [_input(hl, f"{hm} lifetime")], hl.get("sources", ""),
                               f"was {fnum(vl['central']):g} years"), stats, "ccs-lifetime")
            m += 1
    stats["capture-variant lifetimes from the host"] = m



SIZE_FACTOR = {"t": 1000.0, "GJ": 31536.0}


def _size_closure(psum, ctx, stats):
    """Capacity x capacity factor = annual output on every archetype that holds
    all three (Brad 2026-10-28: keep them aligned; facility statistics anchor the
    output). Where they disagree by more than 2%, the capacity factor follows
    (annual output / capacity); where that would pass 100%, the capacity does
    (annual output / capacity factor)."""
    rule = {"rule-id": "size-closure", "method": "size-closure",
            "formula": "capacity x capacity factor = annual output",
            "rationale": "Consistency check 2026-11-07; Brad 2026-10-28 size alignment."}
    n = 0
    for k, a in ctx["arch"].items():
        h = ctx["have"].get(k, {})
        cap, cf, ao = (h.get((p, "")) for p in ("capacity", "capacity-factor", "annual-output"))
        if not (_live(cap) and _live(cf) and _live(ao)):
            continue
        f = SIZE_FACTOR.get(a["product-unit"], 1.0)
        c, x, o = fnum(cap["central"]), fnum(cf["central"]), fnum(ao["central"])
        if not c or not x or not o:
            continue
        exp = c * x / 100 * f
        if abs(o - exp) <= 0.02 * exp:
            continue
        x2 = o / (c * f) * 100
        if x2 <= 100:
            _put(psum, h, _row(k, "capacity-factor", "", "%", x2, None, None,
                               worst(["identity", ao.get("grade") or "sourced", cap.get("grade") or "sourced"]), rule,
                               [_input(ao, "annual-output"), _input(cap, "capacity")], "",
                               f"was {x:.4g}%"), stats, "size-closure")
        else:
            c2 = o / (x / 100 * f)
            _put(psum, h, _row(k, "capacity", "", cap["unit"], c2, None, None,
                               worst(["identity", ao.get("grade") or "sourced", cf.get("grade") or "sourced"]), rule,
                               [_input(ao, "annual-output"), _input(cf, "capacity-factor")], "",
                               f"was {c:.4g}"), stats, "size-closure")
        n += 1
    stats["sizes closed"] = n
