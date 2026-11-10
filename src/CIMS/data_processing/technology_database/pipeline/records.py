"""Load the hand-edited records and hold them to their own rules.

An archetype key is `equipment[_discriminator][_modifier...]`:

  * equipment      one term from vocabulary/equipment.csv (closed);
  * discriminator  at most one component, the single distinction that makes this
                   a different option; each term is listed with a definition in
                   vocabulary/discriminator.csv;
  * modifier       zero or more terms from vocabulary/modifier.csv, in the order
                   that file gives them (efficiency, then capture, then retrofit).

The key is the only place those three are written; the build parses them out.
"""
from __future__ import annotations

from collections import Counter, defaultdict

from .common import RECORDS, REFERENCE, VOCAB, read, split_list



# the three heat tiers (Brad 2026-11-09): the heat the process needs, the heat an
# archetype meant to supply heat delivers, and the heat it rejects that could be recovered
HEAT_TIERS = ("heat-process-tier", "heat-output-tier", "heat-waste-tier")

class Records:
    def __init__(self):
        self.archetypes = read(RECORDS / "archetype.csv")
        self.service = read(RECORDS / "archetype-service.csv")
        self.families = {r["family"]: r for r in read(RECORDS / "family.csv")}
        self.annotations = read(RECORDS / "hierarchy-annotation.csv")
        self.equipment = {r["equipment"]: r for r in read(VOCAB / "equipment.csv")}
        self.discriminators = {r["discriminator"]: r for r in read(VOCAB / "discriminator.csv")}
        mods = sorted(read(VOCAB / "modifier.csv"), key=lambda r: int(r["order"]))
        self.modifiers = [r["modifier"] for r in mods]
        self.carriers = {r["carrier"] for r in read(VOCAB / "carrier.csv")}
        self.lists = defaultdict(set)
        for r in read(VOCAB / "lists.csv"):
            self.lists[r["list"]].add(r["term"])
        self.parameters = {r["parameter"]: r for r in read(VOCAB / "parameter.csv")}
        self.commodities = {r["commodity"]: r for r in read(VOCAB / "commodity.csv")}
        self.priority = {r["source-id"]: r for r in read(REFERENCE / "source-priority.csv")}
        self.by_key = {a["archetype"]: a for a in self.archetypes}
        for a in self.archetypes:
            a["equipment"], a["discriminator"], mods = self.parse(a["archetype"])
            a["modifiers"] = "; ".join(mods)

    def parse(self, key: str):
        parts = key.split("_")
        mods = []
        while len(parts) > 1 and parts[-1] in self.modifiers:
            mods.insert(0, parts.pop())
        return parts[0], "_".join(parts[1:]), mods

    def rank(self, source_id: str) -> int:
        return int(self.priority.get(source_id, {}).get("rank") or 99)

    def short(self, source_id: str) -> str:
        return self.priority.get(source_id, {}).get("short-name") or source_id

    def check(self, hier) -> dict[str, list[str]]:
        """Every rule a record can break, as {rule: [offending rows]}."""
        fail = defaultdict(list)
        keys = [a["archetype"] for a in self.archetypes]
        for k, n in Counter(keys).items():
            if n > 1:
                fail["archetype listed twice"].append(k)
        for a in self.archetypes:
            k = a["archetype"]
            eq, disc, mods = self.parse(k)
            if eq not in self.equipment:
                fail["equipment not in vocabulary/equipment.csv"].append(f"{k}: {eq}")
            if "_" in disc:
                fail["more than one discriminator component"].append(k)
            elif disc and disc not in self.discriminators:
                fail["discriminator not in vocabulary/discriminator.csv"].append(f"{k}: {disc}")
            if mods != [m for m in self.modifiers if m in mods]:
                fail["modifiers out of order"].append(k)
            if a["family"] not in self.families:
                fail["family not in family.csv"].append(f"{k}: {a['family']}")
            if a["type"] not in self.lists["type"]:
                fail["type not in lists.csv"].append(f"{k}: {a['type']}")
            for c in split_list(a["carriers"]):
                if c not in self.carriers:
                    fail["carrier not in vocabulary/carrier.csv"].append(f"{k}: {c}")
            for col in HEAT_TIERS:
                tiers = split_list(a.get(col, ""))
                if not tiers:
                    fail[f"archetype with no {col}"].append(k)
                elif any(t not in self.lists["heat-tier"] for t in tiers) or (
                        "not-applicable" in tiers and len(tiers) > 1):
                    fail[f"{col} not in lists.csv (heat-tier)"].append(f"{k}: {a[col]}")
            if not a.get("product-unit"):
                fail["archetype with no product-unit"].append(k)
            serves = any(s["archetype"] == k for s in self.service)
            if a["type"] in ("component", "reference"):
                # a component is a part priced only to assemble whole-plant costs; a
                # reference archetype keeps numbers for a step the hierarchy does not
                # resolve. Neither is offered to the model.
                if serves:
                    fail[f"{a['type']} archetype serves a hierarchy node"].append(k)
            # any other archetype may serve no node (a route the hierarchy leaves out);
            # it stays in the database with applies-to "no connection" (2026-09-30)
        for s in self.service:
            if s["archetype"] not in self.by_key:
                fail["archetype-service row for an unknown archetype"].append(s["archetype"])
            if not hier.expand(s["branch"]):
                fail["archetype-service path not in the hierarchy"].append(
                    f"{s['archetype']}: {s['branch']}")
        ids = {r["annotation"] for r in self.annotations}
        for s in self.service:
            if s.get("end-use") not in ids:
                fail["archetype-service end-use not an annotation id"].append(
                    f"{s['archetype']}: {s.get('end-use', '')!r}")
        for r in self.annotations:
            for b in split_list(r["branches"]):
                if not hier.expand(b):
                    fail["annotation path not in the hierarchy"].append(f"{r['annotation']}: {b}")
        return fail
