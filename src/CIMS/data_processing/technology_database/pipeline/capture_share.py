"""Share of a host's CO2 sent to capture (ruling 2026-10-05; revised 2026-10-24).

REVISED 2026-10-24 (user, sanity-check S138): `apply` now works from P, the
plant-level capture, and sets both share-to-capture and the capture rate (RATES)
so that captured = rate x share x the host's gross CO2 in the fill. A rate of
85-99% printed on a host is plant-level, not a capture-plant rate; see `apply`.
The text below describes the 2026-10-05 method it replaces.

Sources print the CAPTURE PLANT's rate: 90% of the CO2 in the gas it treats.
That rate stays with the capture archetype. On a host with capture, the share
of the host's CO2 that is sent to the capture plant is

    share-to-capture = CO2 captured / (capture rate x the host's total CO2)

worked out two ways, in this order:

  1. Same example. Where an example on the host-with-capture archetype gives both
     the CO2 captured and the CO2 still emitted (same unit, both >= 0), the host's
     CO2 is their sum, so the share is captured / (rate x (captured + emitted)),
     with the example's own rate where it states one between 85% and 100%, else
     90%. The central is the median over such examples. Both numbers then sit on
     one scope, including the extra fuel the capture plant needs.
  2. Base archetype. Otherwise the host's CO2 is its base archetype's pooled
     total (records/fill/host-archetype.csv, emissions-total co2, same unit):
     captured / (rate x base total), the rate being the pooled share on the host
     if it lies between 85% and 100%, else 90%.

A share up to 110% is read as rounding of the rate and set to 100%. Above 110%
the two inputs are on different scopes (a host total net of exports, a
different product basis), so nothing is derived; the pooled value stays and
`central-from` says why. Example: blast furnace with end-of-pipe capture on the
hot-stove flue gas, 0.376 / (0.9 x 1.10) = 38% sent to capture; 34% of the
furnace's CO2 is captured.

The pooled share-to-capture is replaced by the derivation where one can be made
(the sources' 90% is a capture-plant rate, not a share), and the row says how it
was made in `central-from`. Hosts without the inputs keep the pooled value.
"""
from __future__ import annotations

from collections import defaultdict
from statistics import median

from .common import REFERENCE, fnum, read

DEFAULT_RATE = 0.90
PLANT_LEVEL = 85.0      # a printed rate of 85% to under 99.5% on a host is plant-level (S138)
WHOLE = 99.5            # a printed 100% is the whole flue gas sent to capture: a share
# the capture rate per variant that the fill multiplies by: captured = rate x
# share-to-capture x the host's gross CO2 (user 2026-10-24, sanity-check S138)
RATES: dict = {}


MAX_OVER = 1.10


def _same_example(facts, a):
    """[(plant-level capture, example-id, how)] from examples giving captured and
    emitted CO2: captured / (captured + emitted), the share of the host's CO2 the
    example captures."""
    by = defaultdict(lambda: defaultdict(list))
    for r in facts or ():
        if (r.get("archetype") == a and not r.get("excluded") and r.get("value-norm")
                and r["parameter"] in ("emissions-captured", "emissions-total")
                and (r.get("target-norm") or r.get("target") or "").strip() in ("co2", "")):
            by[r["example-id"]][r["parameter"]].append(r)
    out = []
    for eid, d in by.items():
        if "emissions-captured" not in d or "emissions-total" not in d:
            continue
        c, e = d["emissions-captured"][0], d["emissions-total"][0]
        if c["unit-norm"] != e["unit-norm"]:
            continue
        cv, ev = fnum(c["value-norm"]), fnum(e["value-norm"])
        if cv is None or ev is None or cv <= 0 or ev < 0:
            continue
        out.append((cv / (cv + ev), eid, f"{cv:.3g} / ({cv:.3g} + {ev:.3g})"))
    return out


def apply(summaries, records, facts=None) -> dict:
    """share-to-capture and the capture rate on every host-with-capture variant
    (user 2026-10-24, sanity-check S138). P, the plant-level capture (the share of
    the host's gross CO2 that is captured), comes from, in this order:

      1. a printed share below 85%, or of 100%: a part (the cracker furnaces'
         83%) or the whole of the plant's flue gas sent to capture; P = the
         reference capture rate x the share (no capture plant captures 100%);
      2. examples giving captured and emitted CO2: P = captured / (captured +
         emitted), the median over examples;
      3. the variant's pooled CO2 captured over its base archetype's total CO2
         (records/fill/host-archetype.csv), where it is no more than 110%;
      4. a printed rate of 85% to under 99.5% on the host: plant-level (normalise.py records a
         capture rate printed on a host as share-to-capture); P = that rate.

    Steps 2 and 3 give way to 4 where they come out at 100% (no CO2 left) and a
    printed rate exists. Then the capture rate = the larger of P and the
    reference plant's rate (capture-unit_amine, 90%), and share-to-capture = P /
    that rate, at most 100%. A diffuse host whose whole flue gas goes to capture
    at a printed 90% is a 100% share at a 90% rate; a combined cycle's printed
    95% is a 100% share at a 95% rate. The fill then takes captured = rate x share
    x the host's gross CO2 (pipeline/fill.py, _captured)."""
    hosts = {r["archetype"]: r["host"] for r in read(REFERENCE.parent / "fill" / "host-archetype.csv")
             if (r.get("status") or "active") == "active"}
    idx = {(s["archetype"], s["parameter"], (s.get("target") or "").strip()): s for s in summaries}
    refrow = idx.get(("capture-unit_amine", "capture-rate", "co2"))
    rv = fnum(refrow["central"]) if refrow and refrow.get("central") else None
    ref = rv / 100 if rv and 85 <= rv <= 100 else DEFAULT_RATE
    RATES.clear()
    stats = defaultdict(int)
    for a in records.by_key:
        rec = records.by_key[a]
        if "_ccs-" not in a or rec.get("family") == "co2-capture":
            continue
        sh = idx.get((a, "share-to-capture", "co2"))
        pooled = fnum(sh["central"]) if sh and sh.get("central") else None
        plo = fnum(sh.get("low")) if sh else None
        phi = fnum(sh.get("high")) if sh else None
        base = hosts.get(a)
        cand = {}
        if pooled is not None and (pooled < PLANT_LEVEL or pooled >= WHOLE):
            cand["share"] = (ref * pooled / 100, ref * (plo if plo is not None else pooled) / 100,
                             ref * (phi if phi is not None else pooled) / 100,
                             f"the printed share sent to capture {pooled:.4g}% x the reference "
                             f"capture rate {ref:.0%}")
        same = _same_example(facts, a)
        if same:
            vals = sorted(min(1.0, x[0]) for x in same if x[0] <= MAX_OVER)
            if vals:
                cand["same"] = (median(vals), vals[0], vals[-1],
                                f"{len(vals)} example(s) giving CO2 captured and CO2 emitted: "
                                + "; ".join(f"{x[2]} = {x[0]:.0%}" for x in sorted(same)[:4])
                                + (f"; median {median(vals):.0%}" if len(vals) > 1 else ""))
        cap = idx.get((a, "emissions-captured", "co2"))
        tot = idx.get((base, "emissions-total", "co2")) if base else None
        if cap and tot and cap.get("central") and tot.get("central"):
            if cap["unit"] != tot["unit"]:
                stats["unit mismatch"] += 1
            else:
                c, t = fnum(cap["central"]), fnum(tot["central"])
                if c and t and t > 0:
                    if c / t <= MAX_OVER:
                        cand["base"] = (min(1.0, c / t), None, None,
                                        f"CO2 captured {c:.3g} / {base} total CO2 {t:.3g} = {c / t:.0%}")
                    else:
                        stats["base derivation over 110% (other scope)"] += 1
        if pooled is not None and PLANT_LEVEL <= pooled < WHOLE:
            cand["plant"] = (min(1.0, pooled / 100),
                             min(1.0, (plo if plo is not None else pooled) / 100),
                             min(1.0, (phi if phi is not None else pooled) / 100),
                             f"the printed capture rate {pooled:.4g}% on the host, plant-level")
        pick = None
        for kind in ("share", "same", "base", "plant"):
            if kind in cand:
                if kind in ("same", "base") and cand[kind][0] >= 0.995 and "plant" in cand:
                    stats[f"{kind} at 100% gave way to the printed rate"] += 1
                    continue
                pick = kind
                break
        if pick is None:
            continue
        P, Pl, Ph, how = cand[pick]
        rate = ref if P <= ref + 1e-4 else P
        share = min(1.0, P / rate)
        RATES[a] = rate * 100
        lo = f"{min(1.0, Pl / rate) * 100:.4g}" if Pl is not None else ""
        hi = f"{min(1.0, Ph / rate) * 100:.4g}" if Ph is not None else ""
        if lo == hi:
            lo = hi = ""
        text = (f"plant-level capture {P:.1%} ({how}); capture rate {rate:.0%}"
                + (" (the reference plant's)" if rate <= ref + 1e-4 else " (above the reference plant's, so the printed rate)")
                + f"; share sent to capture = {P:.1%} / {rate:.0%} = {share:.0%} (S138)")
        if sh is None:
            src = cap or {}
            sh = dict(src, **{"archetype": a, "parameter": "share-to-capture", "target": "co2",
                              "unit": "%", "n": 0, "n-external": 0, "sources": (src or {}).get("sources", ""),
                              "examples": ""})
            summaries.append(sh)
            idx[(a, "share-to-capture", "co2")] = sh
            stats["shares added"] += 1
        ext = {"share": "assume:capture-rate", "same": "assume:capture-rate",
               "base": f"base:{base}; assume:capture-rate", "plant": "assume:capture-rate"}[pick]
        sh.update({"central": f"{share * 100:.4g}", "low": lo, "high": hi, "central-from": text,
                   "external-factors": "; ".join(dict.fromkeys(
                       x for x in (sh.get("external-factors"), ext) if x))})
        stats[f"shares from {pick}"] += 1
    return dict(stats)
