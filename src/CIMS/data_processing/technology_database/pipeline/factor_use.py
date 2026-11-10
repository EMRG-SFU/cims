"""Where the conversion factors went: one row per factor (and per pooled
efficiency or assumption the bridge used) with the pooled rows that rest on it,
and one row per reference source with the factors it backs.

The reference sources -- physical constants, densities, process ratios -- are
registered in data/raw_data/technology_database/source-details_reference.csv, apart from the
technology sources in source-details_sources.csv: they are not evidence about a
technology, only the means of putting one source's number on another's basis.
"""
from __future__ import annotations

from collections import defaultdict

from .common import OUTPUT, RAW, read, split_list, write

REF_MANIFEST = RAW / "source-details_reference.csv"
USE_COLS = ["factor-id", "class", "quantity", "from-unit", "to-unit", "factor", "factor-low",
            "factor-high", "basis", "source-id", "rows-pooled", "examples", "archetypes"]
REF_COLS = ["source_id", "institution", "title", "year", "url", "date_accessed", "used_for",
            "factors", "rows-pooled"]


def write_outputs(facts, factors) -> None:
    from .currency import DEFLATOR_DEFAULT, DEFLATOR_SOURCE, EXCHANGE_EXTRA, TARGET_CUR, exchange_source
    from .common import currency_year
    uses = defaultdict(list)
    for r in facts:
        if r.get("value-norm") and not r.get("excluded"):
            for x in split_list(r.get("conversion-factors", "")):
                uses[x.split("=")[0]].append(r)
    rows = []
    for fid, f in factors.rows.items():
        rows.append(dict(f, **_use(uses.get(fid, []))))
    for tag in sorted(t for t in uses if t not in factors.rows):
        cls = ("same-source" if tag in ("eff:same-example", "hpr:same-example") else "internal" if tag.startswith(("eff:", "hpr:"))
               else "assumption" if tag.startswith("assume:") else "")
        rows.append({"factor-id": tag, "class": cls,
                     "quantity": ("the example's own efficiency" if tag == "eff:same-example" else
                                  "the example's own heat-to-power ratio" if tag == "hpr:same-example" else
                                  f"pooled heat-to-power ratio of {tag[4:]}" if tag.startswith("hpr:") else
                                  f"pooled efficiency of {tag[4:]}" if tag.startswith("eff:") else
                                  tag[7:].replace("-", " ")),
                     **_use(uses[tag])})
    write(OUTPUT / "conversion-factor-use.csv", USE_COLS, rows)
    backs = defaultdict(list)
    for fid, f in factors.rows.items():
        for s in split_list(f.get("source-id", "")):
            backs[s].append(fid)
    # the currency series: every pooled money value rests on a deflator and,
    # unless it was already in CAD, on the 2025 exchange rate
    for r in facts:
        cy = currency_year(r.get("unit", ""))
        if not cy or not r.get("value-norm") or r.get("excluded"):
            continue
        cur = cy.split("_")[1]
        backs_rows = [DEFLATOR_SOURCE.get(cur, DEFLATOR_DEFAULT)]
        if cur != TARGET_CUR:
            backs_rows.append(exchange_source(2025))
            backs_rows += EXCHANGE_EXTRA.get(cur, [])
        for s in backs_rows:
            uses["currency:" + s].append(r)
    ref = []
    for s in read(REF_MANIFEST):
        if "currency" in (s.get("used_for") or ""):
            ref.append(dict(s, factors="currency-conversion.csv",
                            **{"rows-pooled": len(uses.get("currency:" + s["source_id"], []))}))
            continue
        fids = backs.get(s["source_id"], [])
        ref.append(dict(s, factors="; ".join(fids),
                        **{"rows-pooled": len({id(r) for f in fids for r in uses.get(f, [])})}))
    write(OUTPUT / "reference-source-list.csv", REF_COLS, ref)


def _use(rs):
    return {"rows-pooled": len(rs),
            "examples": len({r["example-id"] for r in rs}),
            "archetypes": "; ".join(sorted({r.get("archetype", "") for r in rs}))}
