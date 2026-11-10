"""The efficiency bridge: a number stated per unit of energy going IN put on the
archetype's basis, per GJ of what comes OUT (design.md s5, 2026-09-26).

An energy-service archetype is measured per GJ of output (steam, heat, hydrogen).
Some sources state a number per unit of input instead. Dividing by the
efficiency (output / input) moves it across:

  per-input    an emission factor per GJ of fuel, or a cost per kW of input
               capacity (capacity-basis thermal-input or electric-input):
               x 100 / efficiency
  capacity-in  a capacity rated on its input: x efficiency / 100
  host-tonne   a number per tonne of a HOST plant's product, where the source
               gives only the plant's fuel use per tonne (UKERC): per tonne /
               (fuel per tonne x efficiency). Only for the examples marked
               `yes` in records/reference/efficiency-bridge.csv -- the per-tonne
               numbers must be the equipment's own, not the whole host process's.

The efficiency is the example's own, where the same source gives one (class
`same-source`); otherwise the archetype's pooled central value (class
`internal`: it comes from other examples in this database). For a CHP archetype,
whose product is the heat, it is the total less the electric efficiency.
A specific energy use is never bridged: divided by fuel use times efficiency it
would only restate the efficiency.
"""
from __future__ import annotations

from collections import defaultdict
from statistics import median

from .common import fnum
from . import convert, summarise

EFF = ("efficiency", "efficiency-chp")


def _is_chp(key: str, pooled: dict) -> bool:
    return "efficiency-chp" in pooled or "chp" in key or key.startswith("combined-cycle")


def _heat_eff(d: dict, chp: bool):
    """(efficiency %, how) for the useful output, from {parameter: value}."""
    if chp:
        if "efficiency-chp" in d and "efficiency" in d and d["efficiency-chp"] > d["efficiency"]:
            return d["efficiency-chp"] - d["efficiency"], "total less electric efficiency"
        return None
    if "efficiency" in d:
        return d["efficiency"], "efficiency"
    return None


def apply(facts, records, examples, units) -> dict:
    todo = [r for r in facts if r.get("_bridge")]
    stats = defaultdict(int)
    if not todo:
        return stats
    from .normalise import case_of
    own = defaultdict(lambda: defaultdict(list))
    own_how = defaultdict(list)          # the efficiency rows' own conversions
    own_derived = defaultdict(lambda: defaultdict(list))
    own_derived_how = defaultdict(list)
    for r in facts:
        if r["parameter"] in EFF and r.get("value-norm") and not r.get("excluded"):
            if r.get("value-type") != "derived":
                own[r["example-id"]][r["parameter"]].append(fnum(r["value-norm"]))
                own_how[r["example-id"]].append(r)
            else:
                # a derived efficiency (the same source's numbers, e.g. a fraction
                # printed as such) is used only where the example reports none
                own_derived[r["example-id"]][r["parameter"]].append(fnum(r["value-norm"]))
                own_derived_how[r["example-id"]].append(r)
    for eid, d in own_derived.items():
        for p, v in d.items():
            if p not in own[eid]:
                own[eid][p] = v
                own_how[eid] += [x for x in own_derived_how[eid] if x["parameter"] == p]
    pooled = defaultdict(dict)
    for s in summarise.summarise([r for r in facts if r["parameter"] in EFF], records,
                                 ("parameter",), examples):
        c = fnum(s["central"])
        if c:
            pooled[s["archetype"]][s["parameter"]] = (c, s["n"])
    # judged efficiencies (non-CHP), used only where no example on the archetype
    # states one
    from .common import RECORDS, read
    judged = {j["archetype"]: fnum(j["central"])
              for j in read(RECORDS / "fill" / "fill-judgement.csv")
              if j["parameter"] == "efficiency" and not (j.get("target") or "").strip()
              and j.get("status") != "rejected" and fnum(j.get("central"))}
    sec = defaultdict(list)
    for r in facts:
        if r["parameter"] == "specific-energy-consumption" and fnum(r.get("value")) is not None:
            num = (r.get("unit") or "").split("/", 1)[0]
            route = units.physical(num, "GJ") if num != "GJ" else (1.0, 0.0)
            if route:
                sec[(r["example-id"], case_of(r))].append(fnum(r["value"]) * route[0])

    for r in todo:
        kind = r.pop("_bridge")
        a = r.get("archetype", "")
        chp = _is_chp(a, pooled.get(a, {}))
        mine = {k: median(v) for k, v in own.get(r["example-id"], {}).items()}
        # an example is CHP only if its archetype is named so or it reports a total
        # CHP efficiency itself; a heat-only boiler pooled on a boiler-or-CHP
        # archetype bridges by its own efficiency (fixed 2026-10-02: it took the
        # archetype's total less electric, two medians of different plants)
        own_chp = ("chp" in a or a.startswith("combined-cycle") or "efficiency-chp" in mine)
        e = _heat_eff(mine, own_chp)
        extra = []
        if e:
            eff, how, tag = e[0], e[1], "eff:same-example"
            # the example's efficiency may itself have been converted (LHV -> HHV):
            # the bridged number is only as good as that conversion
            src = own_how.get(r["example-id"], [])
            cls = convert.weakest(["same-source"] + [x.get("conversion-class", "") for x in src])
            extra = [i for x in src for i in (x.get("conversion-factors") or "").split("; ") if i]
        else:
            p = {k: v[0] for k, v in pooled.get(a, {}).items()}
            e = _heat_eff(p, chp)
            if not e and a in judged:
                # no example states an efficiency: the modeller's judgement on the
                # archetype (records/fill/fill-judgement.csv; ruling 2026-10-27, S153)
                j = judged[a]
                eff, how, cls = j, f"judgement on {a} (records/fill/fill-judgement.csv)", "assumption"
                tag = f"eff:judgement:{a}"
                e = (eff, how)
            elif not e:
                r["value-norm"] = r["value-min-norm"] = r["value-max-norm"] = ""
                r["excluded"] = ("needs an efficiency to go from an input to an output basis; neither "
                                 "the example nor the archetype has one")
                r["set-aside"] = "needs-efficiency"
                stats["no efficiency"] += 1
                continue
            else:
                n = pooled[a].get("efficiency", pooled[a].get("efficiency-chp"))[1]
                eff, how, cls = e[0], e[1] + f" pooled on {a} (n={n})", "internal"
                tag = f"eff:{a}"
        if eff <= 0:
            r["excluded"] = "efficiency bridge: the efficiency is not positive"
            continue
        if kind == "capacity-in":
            mult, what = eff / 100, "input capacity x"
        elif kind == "per-input":
            mult, what = 100 / eff, "per unit of input / "
        else:                                           # host-tonne
            s = sec.get((r["example-id"], case_of(r))) or sec.get((r["example-id"], ""))
            if not s or len(set(round(x, 6) for x in s)) != 1 or s[0] <= 0:
                r["value-norm"] = r["value-min-norm"] = r["value-max-norm"] = ""
                r["excluded"] = ("efficiency bridge: the example has no single fuel use per tonne "
                                 "for this case")
                continue
            mult = 100 / (s[0] * eff)
            what = f"per t / ({s[0]:.4g} GJ fuel per t x "
            r["unit-norm"] = r["unit-norm"].replace("/t", "/GJ", 1)
        for c in ("value-norm", "value-min-norm", "value-max-norm"):
            v = fnum(r.get(c))
            r[c] = "" if v is None else f"{v * mult:.6g}"
        r["conversion"] = ((r.get("conversion") + "; ") if r.get("conversion") else "") + \
            f"efficiency bridge: {what}{eff:.4g}% ({how})" + (")" if kind == "host-tonne" else "")
        r["conversion-class"] = convert.weakest([r.get("conversion-class", ""), cls])
        r["conversion-factors"] = "; ".join(dict.fromkeys(
            x for x in (r.get("conversion-factors", "").split("; ") + [f"{tag}={eff:.4g}%"] + extra)
            if x))
        stats[f"{kind} ({cls})"] += 1
    return stats
