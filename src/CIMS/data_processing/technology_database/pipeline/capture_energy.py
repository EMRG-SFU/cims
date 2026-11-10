"""Capture energy a source states on a host plant with capture.

A `_ccs-` host archetype holds the host plant's own energy use and none of the
capture plant's (its description; the same rule as for cost, CCS_HOST_NOTE in
normalise.py). Where a source gives, per unit of the host's product, the steam
or heat of solvent regeneration or the electricity of capture and compression,
that row is set aside here (`set-aside` = ccs-add-on-energy), listed in
output/ccs-addon-energy.csv, and -- where records/reference/ccs-host-energy.csv
names the capture archetype it belongs to and the same example states the CO2
captured per unit of product -- restated per tonne of CO2 captured as a derived
row on a derived example of that capture archetype (ruling 2026-10-05).

    energy per t CO2 = energy per unit of product / CO2 captured per unit of product

Both numbers are the source's own, so the derived row is class `same-source`.
The derived example's id is the host example's id followed by `|capture`; it is
written to output/example.csv like any other and is never added to
records/source-example.csv.
"""
from __future__ import annotations

import re
from collections import defaultdict

from .common import REFERENCE, fnum, read
from .normalise import is_ccs_host

SEC = "specific-energy-consumption"
HOST_ENERGY_TARGETS = {"steam", "heat"}
# words that mark an energy row on a host as the capture plant's
CAPTURE_ENERGY = re.compile(r"captur|reboiler|regenerat|sequestrat|solvent|co2 compress", re.I)
# ... unless the row is the host's own use with the capture plant's folded in
# (kiln coal "plus any capture-process coal", "cement plant plus capture", MARKAL
# "INCLUDES CAPTURE", "production + sequestration"), or says the host's use is
# unchanged: those stay pooled as the host's, with the note saying what is in them
MIXED = re.compile(r"plus (any )?capture|includes capture|includes co2 compress|plant plus capture|"
                   r"production \+|\(carbon sequestration\)|unchanged|per unit of the baseline", re.I)
MIXED_NOTE = ("  MIXED: the host's own energy with the capture plant's folded in, not separable; "
              "kept as the host's (ruling 2026-10-05).")
SET_ASIDE = "ccs-add-on-energy"
NOTE = ("CCS add-on energy: this archetype holds only the host plant's own energy use; "
        "the regeneration steam, heat and the electricity of capture and compression belong "
        "to the co2-capture archetype it requests (ruling 2026-10-05). Listed in "
        "output/ccs-addon-energy.csv")


def apply(facts, examples, records) -> dict:
    table = {r["record-id"]: r for r in read(REFERENCE / "ccs-host-energy.csv")}
    by_id = {r["record-id"]: r for r in facts}
    unknown = set(table) - set(by_id)
    if unknown:
        raise SystemExit(f"ccs-host-energy.csv names unknown record-ids: {sorted(unknown)}")
    bad = {k: r["capture-archetype"] for k, r in table.items()
           if r["capture-archetype"] not in records.by_key}
    if bad:
        raise SystemExit(f"ccs-host-energy.csv names unknown capture archetypes: {bad}")
    captured = {}
    for r in facts:
        if (r["parameter"] == "emissions-captured" and r.get("value-norm")
                and not r.get("excluded") and (r.get("unit-norm") or "").startswith("t/")):
            captured.setdefault(r["example-id"], r)
    stats, seq, new = defaultdict(int), defaultdict(int), []
    for r in facts:
        a = records.by_key.get(r.get("archetype", ""))
        if not (is_ccs_host(a) and r["parameter"] == SEC) or r.get("excluded") == "no number":
            continue
        tg = (r.get("target") or "").strip().lower()
        note = r.get("note") or ""
        if r.get("excluded"):
            continue            # not pooled anyway; its reason stands
        listed = r["record-id"] in table
        if not listed and MIXED.search(note):
            if MIXED_NOTE.strip() not in note:
                r["note"] = note + MIXED_NOTE
            stats["mixed, kept"] += 1
            continue
        if not (tg in HOST_ENERGY_TARGETS or listed or CAPTURE_ENERGY.search(note)):
            continue
        r["set-aside"], r["excluded"] = SET_ASIDE, NOTE
        stats["set aside"] += 1
        t = table.get(r["record-id"])
        c = captured.get(r["example-id"])
        if not t or not c or not r.get("value-norm") and not r.get("value-min-norm"):
            continue
        cap = fnum(c["value-norm"])
        if not cap or cap <= 0:
            continue
        host_ex = examples[r["example-id"]]
        eid = r["example-id"] + "|capture"
        if eid not in examples:
            examples[eid] = dict(host_ex, **{
                "example-id": eid, "archetype": t["capture-archetype"], "mapping": "mapped",
                "source-technology": (host_ex.get("source-technology") or "") + " (capture energy)",
                "detail": "derived: capture energy per t CO2 captured, from the host example "
                          + r["example-id"],
                "note": ((host_ex.get("note") or "") + " DERIVED EXAMPLE (ruling 2026-10-05): "
                         "the capture plant's energy this source states on the host, per t CO2 "
                         "captured; records/reference/ccs-host-energy.csv").strip()})
        sid = r.get("source-id") or "unattributed"
        seq[sid] += 1
        conv = lambda x: "" if fnum(x) is None else f"{fnum(x) / cap:.6g}"
        row = dict(r, **{
            "record-id": f"tp-{sid}-c{seq[sid]:03d}", "example-id": eid,
            "archetype": t["capture-archetype"], "value": "", "value-min": "", "value-max": "",
            "unit": "", "value-norm": conv(r["value-norm"]), "value-min-norm": conv(r["value-min-norm"]),
            "value-max-norm": conv(r["value-max-norm"]), "unit-norm": "GJ/t",
            "target-norm": r.get("target-norm") or tg, "value-type": "derived",
            "derived-from": r["record-id"], "derived-kind": "capture-energy",
            "formula": f"{r['value-norm'] or r['value-max-norm']} / {c['value-norm']}",
            "conversion": (f"host energy per unit of product ({r['record-id']}) over CO2 captured "
                           f"per unit of product ({c['record-id']}), same example"),
            "conversion-class": "same-source", "conversion-factors": "",
            "excluded": "", "set-aside": "", "duplicate-of": "", "whole-plant": "",
            "note": f"CAPTURE ENERGY per t CO2 captured: {t['note']}. From {r['record-id']}: "
                    + note})
        new.append(row)
        stats["derived per t CO2"] += 1
    facts.extend(new)
    return dict(stats)
