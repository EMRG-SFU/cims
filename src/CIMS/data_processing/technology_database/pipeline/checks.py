"""The rules the build holds the records and the facts to.

Everything here reads what is in memory after a build; nothing edits. A failure
stops the build with a non-zero exit, so `python -m technology_database.build` can
gate a commit.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

from .common import RAW, REFERENCE, currency_year, PRIMARY, find_file, read, split_list
from . import convert, fill

# a carrier row split from a MIDDEN total carries the total's id and the carrier
# (tp-van-geene_2024_001-0869-natural-gas; pipeline/midden_carriers.py, 2026-10-24)
RECORD_ID = re.compile(r"^t[pfc]-[a-z0-9_-]+-(\d{4}(-[a-z][a-z-]*)?|[wadx]\d{3})$")


def draws_on_targets(cell: str) -> list[str]:
    """The targets of a draws-on cell: `target (duty); target (duty)`, split at
    parenthesis depth zero so a duty may itself contain a semicolon."""
    parts, depth, cur = [], 0, ""
    for ch in cell or "":
        depth += ch == "("
        depth -= ch == ")"
        if ch == ";" and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return [p.split("(")[0].strip() for p in parts if p.strip()]


def run(records, hier, loader, ex_result) -> dict[str, list[str]]:
    fail = defaultdict(list, records.check(hier))
    fill.check(records, fail)
    for x in ex_result["unknown"]:
        fail["source-example row names an unknown archetype"].append(x)
    ids = {r["annotation"] for r in records.annotations}
    for r in records.annotations:
        for t in draws_on_targets(r.get("draws-on", "")):
            if t not in ids and t not in ("integral", "none"):
                fail["draws-on names an unknown annotation"].append(f"{r['annotation']}: {t}")
    backlog = Counter()
    for r in loader.facts:
        if r["parameter"] not in records.parameters:
            fail["parameter not in vocabulary/parameter.csv"].append(
                f"{r['record-id']}: {r['parameter']}")
        if (r.get("currency") or "") != currency_year(r.get("unit", "")):
            fail["currency is not the currency-year of unit"].append(r["record-id"])
        col = (records.parameters.get(r["parameter"], {}).get("basis-required") or "").strip()
        if col and (r.get("value") or r.get("value-min") or r.get("value-max")) \
                and not (r.get(col) or "").strip():
            if r.get("status") == "review":
                backlog[r.get("source-id", "")] += 1
            else:
                fail["missing a required basis and not marked for review"].append(r["record-id"])
    for name, rows in (("example-parameter", loader.facts), ("example-flow", loader.flows),
                       ("example-constraint", loader.cons)):
        ids_ = [r["record-id"] for r in rows]
        for k, n in Counter(ids_).items():
            if n > 1:
                fail[f"duplicate record-id in {name}"].append(k)
        for i in ids_:
            if not RECORD_ID.match(i):
                fail[f"record-id not <prefix>-<source-id>-NNNN (or -wNNN, -aNNN, -dNNN, -xNNN) in {name}"].append(i)
        known = set(ids_)
        for r in rows:
            for ref in split_list(r.get("derived-from", "")):
                if ref not in known:
                    fail[f"derived-from does not resolve in {name}"].append(
                        f"{r['record-id']} -> {ref}")
    # a fraction left in a per-cent field (sanity-check 2026-10-11, S104): the
    # value is at most 1 while its own band, or the parameter's nature, says per cent
    for r in loader.facts:
        if (r.get("unit") or "") != "%":
            continue
        try:
            v = float(r.get("value") or "nan")
        except ValueError:
            continue
        bounds = []
        for k in ("value-min", "value-max"):
            try:
                bounds.append(float(r.get(k) or "nan"))
            except ValueError:
                pass
        if 0 < v <= 1 and (any(b > 1 for b in bounds if b == b)
                           or r["parameter"] in ("capacity-factor", "availability")):
            fail["per-cent value stored as a fraction"].append(
                f"{r['record-id']}: {r['parameter']} {v} %")
    for r in loader.flows:
        if r["commodity"] not in records.commodities:
            fail["flow commodity not in vocabulary/commodity.csv"].append(
                f"{r['record-id']}: {r['commodity']}")
    conversion_references(fail, records, loader)
    record_keys(fail, loader)
    fail = {k: v for k, v in fail.items() if v}
    fail["_backlog"] = backlog
    return fail


KEY_FILE = REFERENCE / "record-key.csv"


def record_key(r) -> str:
    """What a record id stands for, independent of its position: the example, the
    parameter target or flow commodity, the value as reported and its unit. Record ids
    are numbered by position within a source, so a re-extraction or a change in which
    source a row is credited to can move an id onto another row (2fc1d2b, 2026-11-04:
    26 exclusions then ruled out unrelated rows; consistency check 2026-11-07)."""
    what = r.get("commodity") if "commodity" in r else r.get("target")
    return "|".join(str(r.get(k) or "") for k in ("example-id",)) + "|" + "|".join(
        str(x or "") for x in (what, r.get("value"), r.get("value-min"), r.get("value-max"), r.get("unit")))


def keyed_ids() -> dict[str, list[str]]:
    """{record id: [reference tables naming it]} for every reference table with a
    record-id column."""
    out = defaultdict(list)
    for f in sorted(REFERENCE.glob("*.csv")):
        if f.name == KEY_FILE.name:
            continue
        rows = read(f)
        if rows and "record-id" in rows[0]:
            for r in rows:
                if r.get("record-id"):
                    out[r["record-id"]].append(f.name)
    return out


def record_keys(fail, loader) -> None:
    """Every record id a reference table names must still stand for the row it was
    written for: records/reference/record-key.csv holds that row's key. A missing key
    or a key that no longer matches fails the build; `build --record-keys` adds the
    keys of newly keyed rows (it never changes an existing key)."""
    rows = {r["record-id"]: r for r in [*loader.facts, *loader.flows, *loader.cons]}
    keys = {r["record-id"]: r["key"] for r in read(KEY_FILE)} if KEY_FILE.exists() else {}
    for i, tables in sorted(keyed_ids().items()):
        r = rows.get(i)
        if r is None:
            continue        # an unknown id is each table's own check
        if i not in keys:
            fail["keyed record has no entry in reference/record-key.csv (run build --record-keys)"].append(
                f"{i} ({', '.join(tables)})")
        elif keys[i] != record_key(r):
            fail["keyed record now stands for another row (record ids moved): re-key it"].append(
                f"{i} ({', '.join(tables)}): was {keys[i]}, now {record_key(r)}")


def write_record_keys(loader) -> int:
    """Add the key of every keyed record that has none; existing keys are kept."""
    import csv
    rows = {r["record-id"]: r for r in [*loader.facts, *loader.flows, *loader.cons]}
    keys = {r["record-id"]: r["key"] for r in read(KEY_FILE)} if KEY_FILE.exists() else {}
    n = 0
    for i in keyed_ids():
        if i in rows and i not in keys:
            keys[i] = record_key(rows[i])
            n += 1
    with open(KEY_FILE, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["record-id", "key"])
        for i in sorted(keys):
            w.writerow([i, keys[i]])
    return n


def conversion_references(fail, records, loader) -> None:
    """The conversion factors, the tables that apply them, and the separate
    manifest of the reference sources behind them (design.md s5)."""
    ref = read(RAW / "source-details_reference.csv")
    main = {r["source_id"] for r in read(RAW / "source-details_sources.csv")}
    rid = [r["source_id"] for r in ref]
    for k, n in Counter(rid).items():
        if n > 1:
            fail["duplicate source_id in source-details_reference.csv"].append(k)
    if rid != sorted(rid):
        fail["source-details_reference.csv not in alphabetical order"].append("sort by source_id")
    for k in set(rid) & main:
        fail["source registered in both the reference and the main manifest"].append(k)
    F = convert.Factors()
    backed = set()
    for fid, f in F.rows.items():
        if f.get("class") not in convert.CLASSES:
            fail["conversion factor with no valid class"].append(f"{fid}: {f.get('class')}")
        sids = split_list(f.get("source-id", ""))
        if not sids:
            fail["conversion factor with no source-id"].append(fid)
        for sid in sids:
            backed.add(sid)
            if sid not in rid and sid not in main:
                fail["conversion factor source-id not registered"].append(f"{fid}: {sid}")
        dims = {(convert.atom(u) or ("",))[0] for u in (f["from-unit"], f["to-unit"])}
        if "energy" in dims and "HHV" not in (f.get("basis") or ""):
            fail["energy factor not on a higher-heating-value basis"].append(fid)
        for a in split_list(f.get("applies-to", "")):
            if a not in records.by_key:
                fail["conversion factor applies to an unknown archetype"].append(f"{fid}: {a}")
    # a reference source may also back a fill assumption (records/fill/fill-assumption.csv)
    for a in fill.assumptions():
        for sid in split_list(a.get("source-id", "")):
            backed.add(sid)
            if sid not in rid and sid not in main:
                fail["fill-assumption source-id not registered"].append(f"{a.get('assumption-id')}: {sid}")
    for e in read(REFERENCE / "energy-price.csv"):
        if e.get("kind") not in ("sourced", "derived", "internal", "pending"):
            fail["energy-price kind must be sourced, derived, internal or pending"].append(e.get("carrier", ""))
        if e.get("kind") == "derived" and e.get("derived-from") not in records.by_key:
            fail["energy-price derived from an unknown archetype"].append(e.get("carrier", ""))
        per = (e.get("derived-per") or "").strip()
        if per and per not in convert.Factors().rows:
            fail["energy-price derived-per names an unknown conversion factor"].append(f"{e.get('carrier')}: {per}")
        if e.get("kind") == "sourced" and not (e.get("price") and e.get("source-id")):
            fail["energy-price sourced with no price or source-id"].append(e.get("carrier", ""))
        for sid in split_list(e.get("source-id", "")):
            backed.add(sid)
            if sid not in rid and sid not in main:
                fail["energy-price source-id not registered"].append(f"{e.get('carrier')}: {sid}")
    for r in ref:
        if r["source_id"] not in backed and "currency" not in (r.get("used_for") or ""):
            fail["reference source backs no conversion factor or fill assumption"].append(r["source_id"])
        if (r.get("file_on_disk") or "").strip() and find_file(r["file_on_disk"].strip()) is None and (PRIMARY.is_dir() or r["file_on_disk"].endswith(".csv")):
            fail["reference file recorded but missing"].append(r["file_on_disk"])
    for r in read(REFERENCE / "example-conversion.csv"):
        for x in split_list(r.get("factor-id", "")):
            if x not in F.rows and x not in ("none", "set-aside"):
                fail["example-conversion names an unknown factor"].append(f"{r['example-id']}: {x}")
        if r["example-id"] not in loader.examples:
            fail["example-conversion names an unknown example"].append(r["example-id"])
    hvs = read(REFERENCE / "heating-value-stream.csv")
    for k, n in Counter(r["archetype"] for r in hvs).items():
        if n > 1:
            # a second row silently replaces the first when the table is loaded
            fail["duplicate archetype in heating-value-stream.csv"].append(k)
    for k, n in Counter(r["factor-id"] for r in read(REFERENCE / "conversion-factor.csv")).items():
        if n > 1:
            fail["duplicate factor-id in conversion-factor.csv"].append(k)
    for r in hvs:
        if r["archetype"] not in records.by_key:
            fail["heating-value-stream names an unknown archetype"].append(r["archetype"])
        for x in (r.get("fuel-in", ""), r.get("product-out", "")):
            if x and (x not in F.rows or F.rows[x]["from-unit"] != "GJ-LHV"):
                fail["heating-value-stream names a factor that is not an HHV/LHV ratio"].append(
                    f"{r['archetype']}: {x}")
    for r in read(REFERENCE / "efficiency-bridge.csv"):
        if r["example-id"] not in loader.examples:
            fail["efficiency-bridge names an unknown example"].append(r["example-id"])
        if r.get("bridge") not in ("yes", "no"):
            fail["efficiency-bridge: bridge must be yes or no"].append(r["example-id"])
    fact_ids = {x["record-id"] for x in loader.facts}
    fact_srcs = {x.get("source-id") for x in loader.facts}
    for r in read(REFERENCE / "emission-basis.csv"):
        if r.get("basis") not in ("output", "input"):
            fail["emission-basis: basis must be output or input"].append(r.get("record-id") or r["source-id"])
        if r.get("record-id") and r["record-id"] not in fact_ids:
            fail["emission-basis names an unknown record-id"].append(r["record-id"])
        if r["source-id"] not in fact_srcs:
            fail["emission-basis names a source with no extraction rows"].append(r["source-id"])


def summaries(psum) -> dict[str, list[str]]:
    """Rules the pooled and filled summaries must keep (added 2026-10-13 after the
    2026-10-11 sanity-check pass): one row per archetype, parameter and target;
    no capacity factor or availability below 5% (a fraction left in a per-cent
    field, S104); no row graded sourced that has a count but neither a central nor
    a band."""
    fail = defaultdict(list)
    seen = Counter((r["archetype"], r["parameter"], r.get("target", "")) for r in psum)
    for k, n in seen.items():
        if n > 1:
            fail["more than one summary row for one archetype, parameter and target"].append(
                " / ".join(k))
    for r in psum:
        for col in ("central", "low", "high"):
            try:
                v = float(r.get(col) or "nan")
            except ValueError:
                continue
            if r["parameter"] in ("capacity-factor", "availability") and v == v and v < 5:
                fail["capacity factor or availability below 5%"].append(
                    f"{r['archetype']}: {col} {v}")
    return {k: v for k, v in fail.items() if v}
