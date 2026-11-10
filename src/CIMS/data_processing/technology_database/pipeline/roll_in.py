"""Roll-ins on the summarised values (modeller 2026-10-09). They change
archetype-parameter.csv itself, not only the wide view. Runs after the fill step.

  * CHP archetypes (label says CHP): `efficiency` is the total CHP efficiency
    (the efficiency-chp summary); an efficiency the sources give without saying
    electric, heat or total moves to target `unspecified` (a secondary parameter).
  * the -min/-max forms of heat-output-temp, heat-process-temp and heat-waste-temp
    (no target) widen their base:
    its central stays its own; low and high span every value; n counts the
    examples behind any of the three. The min and max rows are then consumed.
  * Single-stream efficiency: on an archetype with one useful output (not CHP
    and not the dual-output list) whose product is heat or electricity, the
    overall efficiency is repeated in that stream's column (efficiency_heat on a
    boiler). The stream-named efficiencies of such archetypes were already pooled
    into `efficiency` before summarise (pipeline/roll_up.py).
"""
from __future__ import annotations

import re

CHP = re.compile(r"\bCHP\b")
DUAL_EXTRA = {"boiler_recovery", "gasifier_black-liquor", "incinerator_waste-to-energy"}
STREAM_OF = {"heat": "heat", "hvac heating": "heat", "electricity": "electricity"}


def dual(a: dict) -> bool:
    return bool(CHP.search(a.get("label", ""))) or a.get("archetype") in DUAL_EXTRA


def _exs(s):
    return {e.strip() for e in (s.get("examples") or "").split(";") if e.strip()}


def apply(psum: list[dict], records) -> dict:
    arch = {a["archetype"]: a for a in records.archetypes}
    by = {}
    for s in psum:
        by.setdefault(s["archetype"], {})[(s["parameter"], s.get("target", ""))] = s
    n = {"chp efficiency swapped": 0, "temperature ranges widened": 0, "stream efficiency repeated": 0}
    drop = set()
    for k, cells in by.items():
        a = arch.get(k, {})
        # CHP swap
        if CHP.search(a.get("label", "")) and ("efficiency-chp", "") in cells:
            base = cells.get(("efficiency", ""))
            if base:
                base["target"] = "unspecified"
                base["central-from"] = "CHP: efficiency not stated as electric, heat or total; " + base.get("central-from", "")
            chp = cells[("efficiency-chp", "")]
            chp["parameter"] = "efficiency"
            chp["central-from"] = "total CHP efficiency (efficiency-chp); " + chp.get("central-from", "")
            n["chp efficiency swapped"] += 1
        # temperature widening, for the output and the process temperature
        for base in ("heat-output-temp", "heat-process-temp", "heat-waste-temp"):
            lo_hi = [cells.get((base + "-min", "")), cells.get((base + "-max", ""))]
            if any(lo_hi):
                main = cells.get((base, ""))
                parts = [x for x in [main] + lo_hi if x]
                vals = [float(v) for x in parts for v in (x.get("central"), x.get("low"), x.get("high"))
                        if v not in (None, "")]
                exs = set().union(*[_exs(x) for x in parts])
                if not main:
                    main = dict(lo_hi[0] or lo_hi[1], parameter=base, target="",
                                central="", grade="sourced")
                    psum.append(main)
                if vals and min(vals) != max(vals):
                    main["low"], main["high"] = f"{min(vals):.6g}", f"{max(vals):.6g}"
                elif vals and not main.get("central"):
                    # one bound only (S114): a minimum is a low, a maximum a high
                    side = "low" if lo_hi[0] else "high"
                    main[side] = f"{vals[0]:.6g}"
                main["n"] = len(exs) or main.get("n")
                main["examples"] = "; ".join(sorted(exs))
                main["central-from"] = ((main.get("central-from") or "") +
                                        f"; low/high include {base}-min and -max").lstrip("; ")
                if not main.get("central") and main.get("low") and main.get("high"):
                    # a temperature given only as a range takes its midpoint, graded
                    # estimated like any range-only value (est-range-midpoint; S114)
                    main["central"] = f"{(float(main['low']) + float(main['high'])) / 2:.6g}"
                    main.update({"grade": "estimated", "method": "range-midpoint",
                                 "rule-id": "est-range-midpoint",
                                 "rationale": "midpoint of the temperature range the sources give "
                                              f"({base}, -min and -max together)"})
                for x in lo_hi:
                    if x:
                        drop.add(id(x))
                n["temperature ranges widened"] += 1
    psum[:] = [s for s in psum if id(s) not in drop]
    # single-stream efficiency repeated in the stream column
    by = {}
    for s in psum:
        by.setdefault(s["archetype"], {})[(s["parameter"], s.get("target", ""))] = s
    for k, cells in by.items():
        a = arch.get(k, {})
        stream = STREAM_OF.get((a.get("product") or "").strip())
        eff = cells.get(("efficiency", ""))
        if not stream or not eff or dual(a) or ("efficiency", stream) in cells:
            continue
        psum.append(dict(eff, target=stream,
                         **{"central-from": "same as efficiency: the archetype's only output stream"}))
        n["stream efficiency repeated"] += 1
    return n


TIERS = ("low-temp", "medium-temp", "high-temp")


def tier_of(temp: float) -> int:
    """low below 150 degC, medium 150-400, high above 400 (design.md, Heat)."""
    return 0 if temp < 150 else 1 if temp <= 400 else 2


# each heat tier and the temperature that decides it where one is sourced
TIER_TEMP = {"heat-process-tier": "heat-process-temp", "heat-output-tier": "heat-output-temp",
             "heat-waste-tier": "heat-waste-temp"}


def tier_from(s) -> str:
    """The tiers a temperature summary spans (central, low and high)."""
    vals = [float(s[x]) for x in ("central", "low", "high") if s.get(x) not in (None, "")]
    if not vals:
        return ""
    return "; ".join(TIERS[tier_of(min(vals)):tier_of(max(vals)) + 1])


def tier_disagreements(psum, rec) -> list[str]:
    """Archetypes whose recorded heat tier differs from the tiers the matching
    temperature spans (for the output tier: does not cover them). Temperature decides (Brad 2026-11-09): the record is corrected
    by hand, never by the pipeline."""
    temp_tier = {v: k for k, v in TIER_TEMP.items()}
    out = []
    for s in psum:
        col = temp_tier.get(s["parameter"])
        if not col or s.get("target"):
            continue
        a = rec.by_key.get(s["archetype"])
        want = tier_from(s)
        if not want or not a:
            continue
        have = "; ".join(t for t in TIERS if t in (a.get(col) or ""))
        if col == "heat-output-tier":
            # sources often print only a supplier's maximum: the recorded output tiers
            # must cover what the sources give, and may reach further
            bad = not set(want.split("; ")) <= set(have.split("; "))
        else:
            bad = want != have
        if bad:
            out.append(f"{s['archetype']} {col}: {a.get(col)} (temperature says {want})")
    return out
