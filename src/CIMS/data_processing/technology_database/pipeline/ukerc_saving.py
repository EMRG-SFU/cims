"""UKERC efficiency buckets as savings (ruling 2026-10-05).

A UKERC efficiency bucket groups the measures UKERC lists at one process step.
Its specific energy use was loaded as an absolute: the median of the members'
post-measure SEC, each on UKERC's own UK baseline for the step. That baseline is
not this database's base archetype, so a bucket could land above its base, and a
measure saving 100% or more (an energy export) gave zero or a negative SEC.

Now the bucket's SEC row is not pooled as an absolute (set-aside `ukerc-saving`).
The saving of each member is read from UKERC's rows,

    saving = (baseline SEC - technology SEC) / baseline SEC

and, where the archetype is a variant with a base archetype
(records/fill/host-archetype.csv) and the median saving is between 0% and
90% (a larger one is an energy export or a different service), a derived row
puts the bucket on this database's base:

    SEC = base archetype's pooled total SEC x (1 - median saving)
    low, high = the same with the members' largest and smallest saving

The derived row (id tp-<source>-sNNN) is class `internal` (it rests on a value
pooled in this database) with conversion factor `base:<base archetype>`. A
bucket on an archetype with no base, or with a saving outside that band, is set
aside with the reason.
"""
from __future__ import annotations

import re
from collections import defaultdict
from statistics import median

from .common import REFERENCE, RAW, fnum, read

ROWS = RAW / "other-reports" / "ukerc_2013_usable-energy-database_rows.csv"
SEC = "specific-energy-consumption"
MEMBERS = re.compile(r"Members: (.*?)\. Energy")
UKERC_ID = re.compile(r"([A-Z]{1,3}-[A-Za-z0-9]*?\d+)\s*\(")
SET_ASIDE = "ukerc-saving"
NOTE = ("UKERC efficiency bucket: its SEC is the post-measure level on UKERC's own UK baseline, "
        "not this database's base archetype, so it is not pooled as an absolute; the bucket's "
        "saving is applied to the base archetype instead (ruling 2026-10-05)")


def _savings():
    b, t = {}, {}
    for r in read(ROWS):
        if r["column-name"] != "SEC (final basis)":
            continue
        v = fnum(r["value"])
        if v is None:
            continue
        if r["column-group"] == "Baseline net energy demand and emissions":
            b[r["ukerc-id"]] = v
        elif r["column-group"] == "Technology net energy demand and emissions":
            t[r["ukerc-id"]] = v
    return {k: (b[k] - t[k]) / b[k] for k in b if k in t and b[k] > 0}


def derive(facts, records, summarise_fn) -> dict:
    hosts = {r["archetype"]: r["host"] for r in read(REFERENCE.parent / "fill" / "host-archetype.csv")
             if (r.get("status") or "active") == "active"}
    saving = _savings()
    stats = defaultdict(int)
    queued = []
    for r in facts:
        note = r.get("note") or ""
        if (r["parameter"] != SEC or not note.startswith("BUCKET of") or r.get("excluded")
                or not r.get("value-norm") or r.get("value-type") == "derived"):
            continue
        m = MEMBERS.search(note)
        ids = UKERC_ID.findall(m.group(1)) if m else []
        s = [saving[i] for i in ids if i in saving]
        r["set-aside"] = SET_ASIDE
        stats["buckets set aside"] += 1
        a = r.get("archetype", "")
        base = hosts.get(a)
        if not s:
            r["excluded"] = NOTE + "; UKERC gives no saving for its members"
            continue
        sm = median(s)
        if not base or base not in records.by_key:
            r["excluded"] = NOTE + f"; no base archetype for {a}, so the saving ({sm:.0%}) is not applied"
            stats["no base archetype"] += 1
            continue
        if not 0.0 < sm < 0.9:
            r["excluded"] = NOTE + (f"; the median saving {sm:.0%} is outside 0% to 90% "
                                    "(an energy export or a different service), so it is not applied")
            stats["saving out of band"] += 1
            continue
        r["excluded"] = NOTE + f"; median saving {sm:.1%} applied to {base}"
        queued.append((r, base, sm, min(s), max(s)))
    # a same-source pair (records/reference/pair-saving.csv): a source that prints a
    # base and an improved level for the same product (DOE bandwidth current typical
    # and state of the art) gives the improved archetype a saving on this database's
    # base, the same way (Brad 2026-10-28, S07)
    by_id = {r["record-id"]: r for r in facts}
    for p_ in read(REFERENCE / "pair-saving.csv"):
        r, b = by_id.get(p_["record-id"]), by_id.get(p_["base-record-id"])
        if r is None or b is None:
            raise SystemExit(f"pair-saving.csv: {p_['record-id']} or {p_['base-record-id']} not found")
        v, bv = fnum(r.get("value-norm")), fnum(b.get("value-norm"))
        base = hosts.get(r.get("archetype", ""))
        if v is None or not bv or not base or r.get("unit-norm") != b.get("unit-norm"):
            r["excluded"] = f"pair saving ({p_['base-record-id']}) not applied: no value, base archetype or common unit"
            continue
        sm = (bv - v) / bv
        r["set-aside"] = SET_ASIDE
        if not 0.0 < sm < 0.9:
            r["excluded"] = f"a same-source pair saving {sm:.0%} outside 0% to 90%, not applied"
            continue
        r["excluded"] = (f"a same-source level for the whole product, read as a saving of {sm:.1%} against "
                         f"{p_['base-record-id']} and applied to {base} ({p_.get('note', '')})")
        r["note"] = "BUCKET of 1 same-source pair. " + (r.get("note") or "")
        stats["pair savings"] += 1
        queued.append((r, base, sm, sm, sm))
    # an efficient variant set on same-source savings takes its carriers from its
    # host's split of the derived total, so the carrier levels its own sources print
    # (absolute, on their own basis) are ruled out, repeats included, which the
    # carrier split would otherwise still read (Brad 2026-11-06)
    paired = {r.get("archetype") for r, *_ in queued if r["note"].startswith("BUCKET of 1 same-source pair")
              and (records.by_key.get(r.get("archetype")) or {}).get("type") == "efficient variant"}
    # except a variant whose sources show the step shifting the mix
    # (variant-own-carriers.csv): it keeps the carrier rows of its own examples
    # that report electricity and a fuel both, and the carrier split scales them
    # to the derived total (Brad 2026-11-06)
    own = {r["archetype"] for r in read(REFERENCE / "variant-own-carriers.csv")}
    both = defaultdict(set)
    for x in facts:
        t = (x.get("target-norm") or "").strip()
        if x.get("archetype") in own and x["parameter"] == SEC and t and x.get("value-norm") and not x.get("excluded"):
            both[x.get("example-id")].add("electricity" if t == "electricity" else "steam" if t in ("steam", "heat") else "fuel")
    keep = {e for e, g in both.items() if "electricity" in g and "fuel" in g}
    for x in facts:
        if x.get("archetype") in own and x.get("example-id") in keep:
            continue
        if (x.get("archetype") in paired and x["parameter"] == SEC and (x.get("target-norm") or "").strip()
                and (not x.get("excluded") or x["excluded"].startswith("repeat:"))):
            x["excluded"] = ("a carrier level of an efficient variant set on same-source savings: its "
                             "carriers come from the host's split of the derived total (Brad 2026-11-06)")
            stats["variant carrier rows ruled out"] += 1
    if not queued:
        return dict(stats)
    # the base archetypes' pooled totals, from the rows as they stand
    bases = {b for _, b, *_ in queued}
    rows = [x for x in facts if x.get("archetype") in bases and x["parameter"] == SEC
            and not (x.get("target-norm") or "").strip()]
    tot = {}
    for s_ in summarise_fn(rows, records, ("parameter", "target-norm"), {}):
        if s_["parameter"] == SEC and not (s_.get("target-norm") or "").strip() and s_.get("central"):
            tot[s_["archetype"]] = (fnum(s_["central"]), s_["unit"])
    seq = defaultdict(int)
    new = []
    for r, base, sm, smin, smax in queued:
        if base not in tot:
            r["excluded"] += f"; but {base} has no pooled total SEC to apply it to"
            stats["base has no total"] += 1
            continue
        bt, bu = tot[base]
        if bu != r.get("unit-norm"):
            r["excluded"] += f"; but {base} is per {bu}, the bucket per {r.get('unit-norm')}"
            stats["unit mismatch"] += 1
            continue
        sid = r.get("source-id") or "unattributed"
        seq[sid] += 1
        new.append(dict(r, **{
            "record-id": f"tp-{sid}-s{seq[sid]:03d}", "value-type": "derived",
            "derived-from": r["record-id"], "derived-kind": "ukerc-saving",
            # the saving lands on the host's total, so the derived row is a total
            # even when the pair was read on the source's one carrier (Brad 2026-11-06)
            "target": "", "target-norm": "",
            "value-norm": f"{bt * (1 - sm):.6g}", "value-min-norm": f"{bt * (1 - smax):.6g}",
            "value-max-norm": f"{bt * (1 - smin):.6g}", "unit-norm": bu,
            "formula": f"{bt:.6g} x (1 - {sm:.4g})",
            "conversion": (f"{base}'s pooled total SEC {bt:.4g} x (1 - the bucket's median saving "
                           f"{sm:.1%}; members {smin:.1%} to {smax:.1%})"),
            "conversion-class": "internal", "conversion-factors": f"base:{base}",
            "excluded": "", "set-aside": "", "duplicate-of": "",
            "note": ("SAME-SOURCE PAIR SAVING ON THE BASE ARCHETYPE (S07, 2026-10-28). " if r["note"].startswith("BUCKET of 1 same-source pair") else "UKERC SAVING ON THE BASE ARCHETYPE (ruling 2026-10-05). ") + (r.get("note") or "")}))
        stats["derived on the base archetype"] += 1
    facts.extend(new)
    return dict(stats)
