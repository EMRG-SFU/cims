"""MIDDEN energy-use totals get their own carriers (sanity-check run 2026-10-24,
S117/S120 and the S120-S132 disagreements).

midden-extract.py derives a card's energy-use total as the sum of every energy
input on the card and prints the terms in the total's formula
(``natural gas 164 TJ + electricity 4.7 TJ + heat 42.6 TJ``; TJ per kt = GJ/t).
The carriers themselves reached the database only as flows, and only for the
labels midden-extract.py maps, so a MIDDEN total was split with another source's
carriers. Here each term becomes a derived energy-use row of its carrier, in the
same example and unit as the total, before normalising, so feedstock take-off,
heating value and the split treat it like any other carrier row.

Every label must map: an unknown label stops the build, because a skipped term
would leave the carriers short of the total.
"""
from __future__ import annotations

import re

SEC = "specific-energy-consumption"
SOURCE = "van-geene_2024_001"

LABEL = {
    "electricity": "electricity",
    "natural gas": "natural-gas", "methane": "natural-gas",
    "green gas": "biomethane", "biogas": "biogas",
    # fuel gas on the chemicals cards is the plant's own off-gas (the cracker's
    # Mynko fuel is process-off-gas too); by-product-gas is the steelworks gas
    "refinery gas": "refinery-gas", "fuel gas": "process-off-gas", "tail gas": "by-product-gas",
    "cog": "coke-oven-gas", "bfg": "blast-furnace-gas", "bofg": "basic-oxygen-furnace-gas",
    "coal": "coal", "coke": "coke", "calcinated petroleum coke": "petroleum-coke",
    "biomass (solid)": "biomass-solid",
    "hydrogen": "hydrogen", "hydrogen (gas)": "hydrogen", "hydrogen gas": "hydrogen",
    "fuel": "fuel", "gasoil": "diesel", "oil": "oil",
    "steam": "steam", "steam (low pressure - 1-9 bar)": "steam",
    "steam (medium pressure - 10-19 bar)": "steam", "steam (medium pressure)": "steam",
    "steam (high pressure - 20-50 bar)": "steam", "steam (high pressure)": "steam",
    "steam (very high pressure - 51-100 bar)": "steam",
    # heat delivered by a thermal-oil or mixed steam/oil system: heat, not a fuel
    "heat": "heat", "hot oil": "heat", "steam and hot oil": "heat",
    "residual heat": "waste-heat", "water (60-100c)": "hot-water",
}
# a card whose fuel gas is the refinery's own (TC0383, a refinery unit)
CARD_LABEL = {("tc0383", "fuel gas"): "refinery-gas"}
TERM = re.compile(r"^(.*?)\s+(-?[\d.]+(?:[eE][+-]?\d+)?)\s+TJ\s*$")


def apply(facts) -> dict:
    stats = {"MIDDEN carrier rows written": 0, "MIDDEN totals split": 0}
    new = []
    ids = {r["record-id"] for r in facts}
    for r in facts:
        if (r.get("source-id") != SOURCE or r["parameter"] != SEC
                or (r.get("target") or "").strip() or " TJ" not in (r.get("formula") or "")):
            continue
        card = (r.get("source-technology") or "").split("_", 1)[0].lower()
        sums = {}
        f_ = re.sub(r"\s*\(in TJ[^)]*\)\s*$", "", r["formula"])
        for t in f_.split(" + "):
            m = TERM.match(t.strip())
            if not m:
                raise SystemExit(f"midden_carriers: cannot parse '{t}' in {r['record-id']}")
            lab = m.group(1).strip().lower()
            c = CARD_LABEL.get((card, lab)) or LABEL.get(lab)
            if c is None:
                raise SystemExit(f"midden_carriers: unknown MIDDEN label '{lab}' in {r['record-id']}; "
                                 "add it to LABEL")
            sums[c] = sums.get(c, 0.0) + float(m.group(2))
        for c, v in sums.items():
            rid = f"{r['record-id']}-{c}"
            if rid in ids:
                continue
            row = dict(r)
            row.update({"record-id": rid, "target": c, "value": f"{v:.6g}", "value-min": "",
                        "value-max": "", "value-type": "derived", "derived-from": r["record-id"],
                        "derived-kind": "carrier-term",
                        "duplicate-of": "", "formula": f"{c} term of the card's total ({r['formula']})",
                        "note": ("carrier of the MIDDEN card's energy-use total, from its formula "
                                 "(pipeline/midden_carriers.py, 2026-10-24). " + (r.get("note") or ""))})
            new.append(row)
            stats["MIDDEN carrier rows written"] += 1
        stats["MIDDEN totals split"] += 1
    facts.extend(new)
    return stats
