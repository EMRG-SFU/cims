"""Feedstock is not energy use (ruling 2026-10-13, sanity-check 2026-10-11 section
4.4): only the feedstock CONSUMED as energy, not the part whose energy ends in the
product, is in specific-energy-consumption.

records/reference/feedstock.csv names the conversion archetypes (a material
feedstock whose energy ends in the product: gas into hydrogen, ammonia or
methanol; biomass into fuels; oil into biodiesel; naphtha into HVC), the feedstock
carrier, and the energy the product holds (GJ HHV per product unit: 1 on a GJ
product). For each example on such an archetype, a total or feedstock-carrier
energy use above `inclusive-above` (default 0.9 x that energy) counts the feed:
the product's energy is taken off it, and written instead as `feedstock-
consumption` (target = the carrier) on a derived row `tp-<source>-dNNN`, so the
feed stays recorded and priced. A value below the threshold already leaves the
feed out and is kept as it is. The fill writes feedstock-consumption = the
product's energy where no example gives one (id-feedstock-in-product).
"""
from __future__ import annotations

import fnmatch
import re
from collections import defaultdict

from .common import REFERENCE, fnum, read

SEC = "specific-energy-consumption"
EXCLUDES_FEED = re.compile(r"\bexclud\w*\s+(the\s+)?(energy\s+content\s+of\s+the\s+)?feed", re.I)
FEED = "feedstock-consumption"


def table():
    return read(REFERENCE / "feedstock.csv")


def rows_for(archetype: str, rows=None):
    return [r for r in (rows if rows is not None else table())
            if fnmatch.fnmatchcase(archetype, r["archetype"])]


def row_for(archetype: str, rows=None):
    """The archetype's product-energy row (the feed whose energy ends in the product)."""
    for r in rows_for(archetype, rows):
        if (r.get("mode") or "product-energy") == "product-energy":
            return r
    return None


def apply(facts, records) -> dict:
    T = table()
    stats = defaultdict(int)
    nxt = defaultdict(int)
    for r in facts:
        m = re.match(r"^(tp-.+)-d(\d{3})$", r.get("record-id", ""))
        if m:
            nxt[m.group(1)] = max(nxt[m.group(1)], int(m.group(2)))
    by_ex = defaultdict(list)
    for r in facts:
        if r["parameter"] == SEC and not r.get("excluded") and r.get("archetype") and any(
                fnum(r.get(c)) is not None for c in ("value-norm", "value-min-norm", "value-max-norm")):
            by_ex[r["example-id"]].append(r)
    new = []
    fallback = {}
    for eid, rs in by_ex.items():
        fallback[eid] = whole(eid, rs, rows_for(rs[0]["archetype"], T), stats)
    for eid, rs in by_ex.items():
        rs = [r for r in rs if r["parameter"] == SEC]
        if not rs:
            continue
        t0 = row_for(rs[0]["archetype"], T)
        ts = ([t0] if t0 else []) + [t for t in fallback[eid] if fnum(t.get("energy-in-product"))]
        for t in ts:
            _product_energy(eid, rs, t, stats, nxt, new)
    facts.extend(new)
    return dict(stats)


def _product_energy(eid, rs, t, stats, nxt, new) -> None:
    """Take the product's energy (or, for a reactant with no row of its own, its
    stoichiometric use) off a feed-inclusive energy use, and write it as a
    feedstock row."""
    e = fnum(t["energy-in-product"])
    thr = fnum(t.get("inclusive-above")) or 0.9 * e
    feeds = t["feedstock"].split("|")
    total_only = (t.get("mode") or "") == "whole-carrier"
    hit = []
    for r in rs:
        tg = (r.get("target-norm") if r.get("target-norm") is not None else r.get("target")) or ""
        tg = tg.strip().lower()
        if tg not in (("",) if total_only else ("",) + tuple(feeds)):
            continue
        if EXCLUDES_FEED.search(r.get("note") or ""):
            # the source says its figure leaves the feed out (IEA 2013 Annex 4):
            # nothing to take off, however high (sanity-check 2026-10-23, CHEM001)
            stats["rows stated net of feedstock, kept"] += 1
            continue
        v = fnum(r.get("value-norm"))
        if v is None:
            # a range-only row counts the feed when its range does (2026-10-13)
            v = fnum(r.get("value-min-norm"))
        if v is None or v < thr:
            continue
        for col in ("value-norm", "value-min-norm", "value-max-norm"):
            x = fnum(r.get(col))
            if x is not None:
                r[col] = f"{max(x - e, 0.0):.6g}"
        r["note"] = (r.get("note") or "") + (
            f"  FEEDSTOCK TAKEN OFF: the figure counts the {feeds[0]} the product is made from "
            f"({e:g} GJ per unit, records/reference/feedstock.csv); that part is "
            "feedstock-consumption, not energy use (rulings 2026-10-13, 2026-10-14).")
        hit.append((r, tg))
        stats["energy rows net of feedstock"] += 1
    if not hit:
        return
    r0 = next((r for r, tg in hit if tg in feeds), hit[0][0])
    tg0 = next((tg for _, tg in hit if tg in feeds and tg != "fuel"), feeds[0])
    src = r0.get("source-id", "")
    nxt[f"tp-{src}"] += 1
    row = {k: "" for k in r0}
    row.update({
        "record-id": f"tp-{src}-d{nxt[f'tp-{src}']:03d}", "example-id": eid,
        "archetype": r0["archetype"], "source-technology": r0.get("source-technology", ""),
        "parameter": FEED, "target": tg0, "target-norm": tg0, "value": f"{e:.6g}",
        "unit": r0.get("unit-norm", ""), "value-norm": f"{e:.6g}", "unit-norm": r0.get("unit-norm", ""),
        "conversion-class": r0.get("conversion-class", ""), "value-type": "derived",
        "derived-from": "; ".join(r["record-id"] for r, _ in hit),
        "source-id": src, "via-source-id": r0.get("via-source-id", ""),
        "source-file": r0.get("source-file", ""), "source-locator": r0.get("source-locator", ""),
        "region": r0.get("region", ""), "year": r0.get("year", ""), "status": r0.get("status", ""),
        "note": (f"The feedstock the product is made from, taken out of the energy use of "
                 f"{', '.join(r['record-id'] for r, _ in hit)} (rulings 2026-10-13, 2026-10-14).")})
    new.append(row)
    stats["feedstock rows written"] += 1


def whole(eid, rs, trows, stats) -> list:
    """mode whole-carrier: a reactant carrier (hydrogen in hydroprocessing, ammonia
    synthesis, e-fuels) is feedstock in full (ruling 2026-10-14). Its energy-use rows
    become feedstock-consumption rows, and the example's total energy use loses the
    same amount where it counted it."""
    idle, done = [], set()
    for t in trows:
        if (t.get("mode") or "") != "whole-carrier":
            continue
        feeds = t["feedstock"].split("|")
        moved = 0.0
        for r in rs:
            tg = ((r.get("target-norm") if r.get("target-norm") is not None else r.get("target")) or "").strip().lower()
            if r["parameter"] != SEC or tg not in feeds:
                continue
            r["parameter"] = FEED
            r["note"] = (r.get("note") or "") + (f"  FEEDSTOCK: the {tg} reacted into the product is "
                                                 "feedstock-consumption, not energy use (ruling 2026-10-14).")
            moved += fnum(r.get("value-norm")) or 0.0
            stats["reactant rows moved to feedstock"] += 1
        if moved <= 0:
            if not (set(feeds) & done):
                idle.append(t)
            continue
        done |= set(feeds)
        for r in rs:
            tg = ((r.get("target-norm") if r.get("target-norm") is not None else r.get("target")) or "").strip()
            v = fnum(r.get("value-norm"))
            if r["parameter"] == SEC and not tg and v is not None and v >= 0.95 * moved:
                for col in ("value-norm", "value-min-norm", "value-max-norm"):
                    x = fnum(r.get(col))
                    if x is not None:
                        r[col] = f"{max(x - moved, 0.0):.6g}"
                r["note"] = (r.get("note") or "") + (f"  FEEDSTOCK TAKEN OFF: {moved:.4g} of {'/'.join(feeds)} "
                                                     "moved to feedstock-consumption (ruling 2026-10-14).")
                stats["totals net of the reactant"] += 1
    return idle
