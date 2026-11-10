"""Values a source prints wrongly, taken into the database on an assumed correction.

records/reference/source-correction.csv lists each correction the modeller makes
to a figure a source prints wrongly: a slipped decimal or zero, a wrong unit or
carrier label, a misprinted digit, a row shifted against its label. The
extraction keeps the figure as printed (layer 1 is never edited to fit the
database); the correction is made here, in one place, with its evidence.

One row per correction, `correction-id` sc-NNN, of one of three kinds:

  * add      a value the extraction set aside (in its _other-values file, or not
             loaded) because of the error. Written as a new row tp-<source-id>-xNNN
             on an existing example (`example-id`), with the corrected value,
             unit and bases given here; the example's other fields (source,
             region, extraction file) are copied from its first fact row.
  * replace  a loaded row whose printed value is wrong. The corrected copy is
             written as tp-<source-id>-xNNN (derived-from the original), and the
             original is excluded as superseded, so both stay visible.
  * mark     a loaded row the extraction already corrected (its note says how).
             Nothing changes but the marker below, so every value that rests on
             an assumed correction carries the same marker.
  * rebase   not an error in the source but a basis the database had missed: a
             loaded row stated per another product of the same source (per t
             cement on a per-t-clinker kiln) is copied as tp-<source-id>-xNNN,
             multiplied by `multiplier` (the source's own ratio, registered as
             `factor-id` in conversion-factor.csv, class same-source), and the
             original excluded as superseded. A rebased row carries no
             correction marker; its note starts SAME-SOURCE REBASE and its
             conversion factors name the ratio.

Every corrected or marked row carries:

  * `correction` = the correction-id (example-parameter.csv);
  * a note that starts "ASSUMED CORRECTION (sc-NNN): <what was changed>. As
    printed: <the printed figure>. Evidence: <why>.";
  * the conversion factor `assume:source-correction-sc-NNN`, so its
    conversion-class is `assumption`: the summaries count it in n-external and
    in n-corrected, the wide view lists it under source-corrections, and the
    workbook shades the value.

Corrections are the modeller's reading of the source, recorded per row with its
evidence; a correction that cannot be argued from the source itself (its own
totals, its other tables, sibling rows, physical bounds) is not made.
"""
from __future__ import annotations

import collections

from .common import REFERENCE, currency_year, read

FILE = REFERENCE / "source-correction.csv"
KINDS = ("add", "replace", "mark", "rebase")
VALUE_FIELDS = ("parameter", "target", "value", "value-min", "value-max", "unit",
                "cost-basis", "capacity-basis", "source-locator", "formula", "value-type")


def _note(c) -> str:
    return (f"ASSUMED CORRECTION ({c['correction-id']}): {c['correction'].rstrip('.')}. "
            f"As printed: {c['as-printed'].rstrip('.')}. Evidence: {c['evidence'].rstrip('.')}. "
            f"[{c['decided']}] ")


def apply(loader) -> dict:
    if not FILE.exists():
        return {}
    by_id = {r["record-id"]: r for r in loader.facts}
    first = {}
    for r in loader.facts:
        first.setdefault(r["example-id"], r)
    errors, stats, seq, new = [], collections.Counter(), collections.Counter(), []
    seen = set()
    for c in read(FILE):
        cid, kind = c["correction-id"].strip(), c["kind"].strip()
        if cid in seen:
            errors.append(f"{cid}: correction-id used twice")
        seen.add(cid)
        if kind not in KINDS:
            errors.append(f"{cid}: unknown kind {kind!r}")
            continue
        for f in ("correction", "as-printed", "evidence", "decided"):
            if not (c.get(f) or "").strip():
                errors.append(f"{cid}: {f} is blank")
        tag = f"source-correction-{cid}"
        if kind == "mark":
            r = by_id.get(c["record-id"])
            if r is None:
                errors.append(f"{cid}: record-id {c['record-id']} not found")
                continue
            r["correction"], r["assume"] = cid, tag
            r["note"] = _note(c) + (r.get("note") or "")
            stats["marked"] += 1
            continue
        if kind in ("replace", "rebase"):
            orig = by_id.get(c["record-id"])
            if orig is None:
                errors.append(f"{cid}: record-id {c['record-id']} not found")
                continue
            base, eid = orig, orig["example-id"]
        else:
            eid = c["example-id"]
            base = first.get(eid)
            if base is None:
                errors.append(f"{cid}: example {eid} has no fact row to take its context from")
                continue
        sid = base["source-id"]
        seq[sid] += 1
        row = {k: v for k, v in base.items() if not k.startswith("_")}
        row.update({"record-id": f"tp-{sid}-x{seq[sid]:03d}", "example-id": eid,
                    "formula": "", "duplicate-of": "", "whole-plant": "", "set-aside": "",
                    "excluded": "", "status": "draft", "confidence": "",
                    "derived-from": orig["record-id"] if kind in ("replace", "rebase") else "",
                    "value-type": "reported", "assume": tag, "correction": cid})
        if kind == "rebase":
            m = float(c["multiplier"])
            for f in ("value", "value-min", "value-max"):
                x = (orig.get(f) or "").strip()
                row[f] = f"{float(x) * m:.6g}" if x else ""
            row.update({"assume": "", "correction": "", "value-type": "derived", "derived-kind": "rebase",
                        "formula": f"{orig['record-id']} x {m:.6g}",
                        "_rebase": c["factor-id"]})
        for f in VALUE_FIELDS:
            if (c.get(f) or "").strip() or (kind == "add" and f not in ("value-type",)):
                row[f] = (c.get(f) or "").strip()
        lo, hi = row.get("value-min"), row.get("value-max")
        row["uncertainty-type"] = "reported-range" if (lo or hi) else ""
        row["currency"] = currency_year(row.get("unit", ""))
        row["note"] = ((f"SAME-SOURCE REBASE ({cid}): {c['correction'].rstrip('.')} "
                        f"(factor {c['factor-id']}). As printed: {c['as-printed'].rstrip('.')}. "
                        f"Evidence: {c['evidence'].rstrip('.')}. [{c['decided']}] ") if kind == "rebase"
                       else _note(c)) + ((c.get("note") or "").strip() or
                                         (orig.get("note", "") if kind in ("replace", "rebase") else ""))
        if kind in ("replace", "rebase"):
            orig["_superseded"] = row["record-id"]
            stats["rebased" if kind == "rebase" else "replaced"] += 1
        else:
            stats["added"] += 1
        new.append(row)
    if errors:
        raise SystemExit("source-correction.csv:\n  " + "\n  ".join(errors))
    loader.facts.extend(new)
    return dict(stats)


def superseded_note(r) -> str:
    return (f"superseded by {r['_superseded']} (records/reference/source-correction.csv): "
            "the printed value is taken to be wrong, or is restated on the archetype's product")


def propagate(facts) -> int:
    """Rows derived after normalisation from a corrected row (capture energy restated
    per t CO2, pipeline/capture_energy.py) carry its correction-id; give them its
    assumption factor too, so the class and the counts agree."""
    from . import convert
    fclass = {f["factor-id"]: (f.get("class") or "").strip() for f in read(REFERENCE / "conversion-factor.csv")}
    n = 0
    for r in facts:
        fid = r.get("_rebase")
        if fid and (r.get("value-norm") or r.get("value-min-norm") or r.get("value-max-norm")) and not r.get("excluded"):
            used = [x.strip() for x in (r.get("conversion-factors") or "").split(";") if x.strip()]
            if fid not in used:
                r["conversion-factors"] = "; ".join(used + [fid])
                # the rebase factor's own class (same-source for a source's own ratio,
                # assumption for a modelled restatement)
                r["conversion-class"] = convert.weakest([r.get("conversion-class") or "", fclass.get(fid) or "same-source"])
        cid = r.get("correction")
        if not cid or not r.get("value-norm") or r.get("excluded"):
            continue
        tag = f"assume:source-correction-{cid}"
        used = [x.strip() for x in (r.get("conversion-factors") or "").split(";") if x.strip()]
        if tag not in used:
            r["conversion-factors"] = "; ".join([tag] + used)
            r["conversion-class"] = convert.weakest([r.get("conversion-class") or "", "assumption"])
            n += 1
    return n
