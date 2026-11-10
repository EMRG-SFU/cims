"""One consistent energy split per archetype (rulings 2026-10-05).

Each carrier of specific-energy-consumption (natural-gas, electricity, steam, ...)
was pooled as its own population, so the carrier medians of an archetype could sum
to 1.5-3x its pooled total, or the total could fall below one carrier. Now the
split has two levels:

  total                       the row with no target: its pooled central, the
                              median of the examples that give one
    electricity, steam, ...   the carriers with no heating value
    fuel                      every combustible carrier together
      natural-gas, coal, ...  the named fuels: a split of `fuel`

  * Level 1. The carriers come from ONE example -- the split example -- as
    SHARES of that example's level-1 sum, each share times the pooled total, so
    they sum to the total. An example's `fuel` is its own `fuel` row, or the sum
    of the named fuels it gives. Low and high are the same shares of the
    total's low and high.
  * Level 2. The named fuels come from one example -- the fuel-split example --
    as shares of that example's named-fuel sum, each share times `fuel`, so they
    sum to `fuel`. A `fuel` row is written wherever the archetype has named
    fuels, even if no source gives one.
  * An archetype with carriers but no total gets a total: the pooled median of
    its one carrier where every example names the same single level-1 carrier
    (modeller 2026-10-21), else the split example's level-1 sum.
  * A carrier or fuel the chosen example does not name keeps its pooled value,
    flagged in `central-from` as pooled apart (it is not part of the split).
  * Where the split example's level-1 sum and the pooled total disagree by more
    than a factor of two, one of them is on another scope (a bucket or route
    total, say): the level-1 carriers are left as the split example states them,
    unscaled, and the total and the carriers say so. The fuel split is still
    applied to whatever `fuel` then is.
  * A row dropped only as a repeat of another example's figure (duplicate-of)
    still counts in its own example's split, so that example's balance is whole.
  * `n` is left as the number of examples that gave the carrier (modeller
    2026-10-05), even where the central comes from the split.

The split example is chosen among the examples that name at least one level-1
carrier: first one covering at least half the pooled total, then one that states
its own total and whose carriers add up to it within 25% (added 2026-10-13), then one naming two
or more level-1 carriers (an energy balance rather than one line), then the one
whose level-1 sum is closest to the pooled total, then the highest-ranked source
(source-priority.csv), then the example id. The fuel-split example is chosen
among the examples that name at least one fuel: first two or more named fuels,
then the one whose named-fuel sum is closest to `fuel`, then rank, then id.

Targets that are not carriers: component, total, not-stated.
"""
from __future__ import annotations

from collections import defaultdict
from math import log
from statistics import median

from .common import REFERENCE, fnum, read, split_list

SEC = "specific-energy-consumption"
NOT_CARRIER = {"", "component", "total", "not-stated"}
FUEL_AGG = "fuel"
STEAMLIKE = {"steam", "heat", "hot-water", "district-heat", "waste-heat"}
NO_HV_CARRIERS = {"electricity", "heat", "steam", "cooling", "hot-water", "waste-heat", "solar",
                  "geothermal", "nuclear", "hydrogen-as-feed"}


def _target(r):
    t = (r.get("target-norm") if r.get("target-norm") is not None else r.get("target")) or ""
    return t.strip().lower()


def _val(r):
    """A row's value: its central, else the midpoint of a range the source printed
    without one (sanity-check 2026-10-23, S135: a range-only electricity row sat
    outside the split, so coke took the whole total and counted it twice)."""
    v = fnum(r.get("value-norm"))
    if v is not None:
        return v
    lo, hi = fnum(r.get("value-min-norm")), fnum(r.get("value-max-norm"))
    return (lo + hi) / 2 if lo is not None and hi is not None else None


def _has(r):
    return _val(r) is not None


def _level1(t):
    """The level-1 carrier a listed carrier belongs to: itself, or `fuel`."""
    t = (t or "").strip().lower()
    return t if t in NO_HV_CARRIERS or t == FUEL_AGG else (FUEL_AGG if t else "")


def _example_carriers(rows, ranges=True):
    """(own total, level-1 carriers, named fuels) for one example's pooled SEC
    rows: the median of its rows per target (as-reported rows preferred to
    derived, as in summarise). Level 1 has `fuel` = the example's own fuel row,
    else the sum of its named fuels."""
    by_t = defaultdict(list)
    for r in rows:
        by_t[_target(r)].append(r)
    out = {}
    for t, rs in by_t.items():
        get = _val if ranges else (lambda r: fnum(r.get("value-norm")))
        rs = [r for r in rs if get(r) is not None]
        rep = [r for r in rs if r.get("value-type") != "derived"] or rs
        vals = [get(r) for r in rep]
        if vals:
            out[t] = median(vals)
    carriers = {t: v for t, v in out.items() if t not in NOT_CARRIER}
    named = {t: v for t, v in carriers.items() if t not in NO_HV_CARRIERS and t != FUEL_AGG}
    top = {t: v for t, v in carriers.items() if t in NO_HV_CARRIERS}
    if FUEL_AGG in carriers:
        top[FUEL_AGG] = carriers[FUEL_AGG]
    elif named:
        top[FUEL_AGG] = sum(named.values())
    return out.get(""), top, named


def _l1_pooled(rows, min_n=0):
    """{level-1 carrier: pooled central} of an archetype's summary rows, before the split."""
    out = {}
    for s in rows:
        t = (s.get("target") or "").strip().lower()
        if t and _level1(t) == t and t not in NOT_CARRIER:
            v = fnum(s.get("central"))
            if v is not None and v > 0 and int(s.get("n") or 0) >= min_n:
                out[t] = v
    return out


def _listed_energy(a):
    """The archetype's listed carriers less its feedstock carriers
    (records/reference/feedstock.csv): hydrogen reacted into ammonia is feed, not a
    carrier the rest of a total can go to (2026-10-24)."""
    from . import feedstock
    feed = {f.strip().lower() for t in feedstock.rows_for(a.get("archetype", ""))
            if (t.get("mode") or "") == "whole-carrier" for f in t["feedstock"].split("|")}
    return [t.strip().lower() for t in split_list(a.get("carriers", "")) if t.strip().lower() not in feed]


def _additive(l1):
    """Carriers that can be added into one total: not fuel beside steam or heat
    (the fuel that raised the steam), nor two names for the same heat."""
    hv = set(l1) & STEAMLIKE
    return not ((FUEL_AGG in l1 and hv) or len(hv) > 1)


def _apart_only_zero(s):
    return (fnum(s.get("central")) or 0) <= 0


def _ok(d):
    return d and all(v is not None and v >= 0 for v in d.values()) and sum(d.values()) > 0


def _short(records, rs):
    return ", ".join(dict.fromkeys(records.short(r.get("source-id", "")) for r in rs))


def _sources(rs):
    return "; ".join(dict.fromkeys(r.get("source-id", "") for r in rs))


def _single_carrier():
    p = REFERENCE / "single-carrier-archetype.csv"
    return {r["archetype"] for r in read(p)} if p.exists() else set()


SINGLE_CARRIER = _single_carrier()


def example_totals(facts, records=None) -> dict:
    """An example that gives two or more level-1 carriers but no total gets a total:
    the sum of its own converted carriers (modeller 2026-10-21). Before this, only
    examples that printed a total entered the pooled total, so a source giving gas
    and electricity as separate lines (AER ST39 mine sites) never moved it. One
    carrier is not enough: a lone line may be one item of a longer table
    (sanity-check S117). The row is derived, sits beside the carriers in the
    example, and is pooled like a printed total."""
    groups = defaultdict(list)
    for r in facts:
        # a row dropped only as a repeat of another example's figure still belongs
        # to this example's balance, as in apply (S135/MCF005: Worrell's container
        # hybrid furnace lost its electricity, and with it its total)
        live = not r.get("excluded") or (r.get("duplicate-of") and r["excluded"].startswith("repeat:"))
        if r["parameter"] == SEC and r.get("value-norm") and live and r.get("archetype"):
            groups[(r["example-id"], r["unit-norm"])].append(r)
    # an example whose own total a modeller ruled out is not given a total again
    # from its carriers (2026-10-24: a ruled-out MIDDEN total came back as a
    # derived one from its carrier rows)
    ruled = {r["example-id"] for r in facts
             if r["parameter"] == SEC and not _target(r) and r.get("excluded")
             and not r["excluded"].startswith("repeat:") and r["excluded"] != "no number"}
    # the printed totals per archetype and unit, to catch a partial table summed
    # into a whole-plant total (sanity-check 2026-10-23, S135/IS002)
    printed = defaultdict(list)
    for (eid, unit), rs in groups.items():
        own = _example_carriers(rs, ranges=False)[0]
        if own is not None and own > 0:
            printed[(rs[0]["archetype"], unit)].append(own)
    # where no example prints a total, the carrier sums of the examples that do
    # name the archetype's own carriers stand in (S135/PB011: sootblowing steam and
    # electricity had become a 0.82 GJ/t recovery-boiler total beside 16 GJ/t)
    main_sums = defaultdict(list)
    for (eid, unit), rs in groups.items():
        own, top, named = _example_carriers(rs, ranges=False)
        if own is not None or not _ok(top) or records is None:
            continue
        a = records.by_key.get(rs[0]["archetype"]) or {}
        listed = {t.strip().lower() for t in split_list(a.get("carriers", ""))}
        if listed & (set(top) | set(named)):
            main_sums[(rs[0]["archetype"], unit)].append(sum(top.values()))
    stats = defaultdict(int)
    new = []
    for (eid, unit), rs in groups.items():
        own, top, named = _example_carriers(rs, ranges=False)
        # one carrier is enough for an archetype whose energy is naturally a single
        # carrier (records/reference/single-carrier-archetype.csv; user 2026-11-04):
        # there a lone line is the whole balance, not one item of a longer table
        single = rs[0]["archetype"] in SINGLE_CARRIER and len(top) == 1
        if own is not None or (len(top) < 2 and not single) or not _ok(top) or eid in ruled:
            continue
        a = (records.by_key.get(rs[0]["archetype"]) if records is not None else None) or {}
        listed = {t.strip().lower() for t in split_list(a.get("carriers", ""))}
        has_fuel_row = FUEL_AGG in {_target(r) for r in rs}
        names_main = (not listed or bool(listed & (set(top) | set(named)))
                      or (has_fuel_row and any(_level1(t) == FUEL_AGG for t in listed)))
        ref = printed.get((rs[0]["archetype"], unit)) or main_sums.get((rs[0]["archetype"], unit))
        if not names_main and ref and sum(top.values()) < 0.3 * median(ref):
            # names none of the archetype's carriers and sums to under 0.3 x the
            # printed totals: one part of a table (the stove fuels of a blast
            # furnace, no coke), not a whole-plant balance
            stats["partial carrier tables not summed"] += 1
            stats.setdefault("_skipped", []).append(f"{eid} {sum(top.values()):.4g} vs {median(ref):.4g}")
            continue
        base = next(r for r in rs if _target(r) in top or _target(r) in named)
        tot = sum(top.values())
        row = dict(base)
        row.update({"record-id": base["record-id"] + "-total", "target": "", "target-norm": "",
                    "value": "", "value-min": "", "value-max": "",
                    "value-norm": f"{tot:.6g}", "value-min-norm": "", "value-max-norm": "",
                    "value-type": "derived", "uncertainty-type": "", "duplicate-of": "", "excluded": "",
                    "formula": "sum of the example's level-1 carriers (" + " + ".join(
                        f"{t} {v:.4g}" for t, v in sorted(top.items())) + ")",
                    "derived-from": "; ".join(r["record-id"] for r in rs if _target(r) in top or _target(r) in named),
                    "note": "total derived by the build from the example's own carriers (carriers.example_totals, 2026-10-21)"})
        new.append(row)
        stats["example totals from carriers"] += 1
        if single:
            stats["of which single-carrier archetypes"] += 1
    facts.extend(new)
    skipped = stats.pop("_skipped", [])
    if skipped:
        print("  partial carrier tables not summed: " + "; ".join(skipped))
    return dict(stats)


def apply(summaries, facts, records) -> dict:
    pooled = defaultdict(lambda: defaultdict(list))
    for r in facts:
        # a row dropped only as a repeat of another example's figure still belongs to
        # this example's energy balance (it is pooled once, but split here in full)
        live = not r.get("excluded") or (r.get("duplicate-of") and r["excluded"].startswith("repeat:"))
        if r["parameter"] == SEC and _has(r) and live and r.get("archetype"):
            pooled[(r["archetype"], r["unit-norm"])][r["example-id"]].append(r)
    by_arch = defaultdict(list)
    for s in summaries:
        if s["parameter"] == SEC:
            by_arch[(s["archetype"], s["unit"])].append(s)
    stats = defaultdict(int)
    for key, rows in by_arch.items():
        total = next((s for s in rows if not (s.get("target") or "").strip()), None)
        tot = fnum(total["central"]) if total else None
        exs = []
        for eid, rs in pooled.get(key, {}).items():
            own, top, named = _example_carriers(rs, ranges=False)
            top_r = _example_carriers(rs)[1]
            rank = min(records.rank(r.get("via-source-id") or r.get("source-id", ""))
                       for r in ([x for x in rs if x.get("value-norm")] or rs))
            exs.append({"eid": eid, "rs": rs, "top": top, "named": named, "rank": rank, "own": own,
                        "top_r": top_r})
        # ---- level 1 -------------------------------------------------------
        # the carrier mix the sources agree on: each level-1 carrier's pooled
        # median as a share of their sum. A split example far from it (a 50%
        # electricity steam reformer) loses to one near it (S135 follow-on,
        # 2026-10-24: which example closed its balance had decided the split)
        cons = _l1_pooled(rows, min_n=3)
        if not _additive(cons):
            cons = {}       # fuel beside steam: two scopes, no mix to agree on
        cons_sum = sum(cons.values())
        cons = {t: v / cons_sum for t, v in cons.items()} if cons_sum > 0 else {}
        cands = []
        for e in exs:
            if not _ok(e["top"]):
                continue
            csum = sum(e["top"].values())
            covers = (csum >= 0.5 * tot) if tot and tot > 0 else True
            closeness = abs(log(csum / tot)) if tot and tot > 0 else 0.0
            # an example that states its own total and whose carriers add up to it
            # is a whole energy balance, not one line of a table: preferred first
            # (sanity-check 2026-10-11, S117: a lone "other fuel" line had become
            # the smelting-reduction vessel's whole fuel)
            balanced = bool(e["own"] and e["own"] > 0 and 0.8 <= csum / e["own"] <= 1.25)
            if cons and len(cons) > 1:
                sh = {t: v / csum for t, v in e["top"].items()}
                dist = sum(abs(sh.get(t, 0.0) - cons.get(t, 0.0)) for t in set(sh) | set(cons))
                # 0.3 bands: within a band, the old order (sum nearest the total) decides
                far = round(dist / 0.3)
            else:
                far = 0
            within = (0.5 <= csum / tot <= 2.0) if tot and tot > 0 else True
            cands.append(((not covers, not within, not balanced, far, len(e["top"]) < 2, closeness, e["rank"], e["eid"]),
                          e, csum))
        if not cands:
            continue
        single = False
        keep_pooled = False
        _, sp, csum = min(cands, key=lambda c: c[0])
        # the chosen example's range-only carriers join its split at their
        # midpoints (S135/NFM004); they do not change which example is chosen
        extra = {t: v for t, v in sp["top_r"].items() if t not in sp["top"] and v is not None and v >= 0}
        if extra:
            sp = dict(sp, top={**sp["top"], **extra})
            csum = sum(sp["top"].values())
            stats["range-only carriers joined to the split"] += len(extra)
        short = _short(records, sp["rs"])
        if tot is None and total is not None:
            # a total the sources give only as a range (sanity-check 2026-10-11,
            # S106): the carrier sum becomes its central where it falls inside the
            # band; otherwise the band stands (the fill takes its midpoint) and the
            # carriers are tested against that midpoint. Never a second total row.
            lo_, hi_ = fnum(total.get("low")), fnum(total.get("high"))
            if lo_ is not None and hi_ is not None and not (lo_ <= csum <= hi_):
                tot = (lo_ + hi_) / 2
                total["central-from"] = (total.get("central-from") or "range only") + (
                    f"; the carrier sum of {short} ({sp['eid']}), {csum:.3g}, falls outside "
                    "the band, so it does not set the central")
                stats["range-only totals kept as ranges"] += 1
            else:
                total["central"] = f"{csum:.6g}"
                total["central-from"] = (f"sum of the carriers of {short} ({sp['eid']}); "
                                         "band from the range-only examples")
                total["sources"] = "; ".join(dict.fromkeys(
                    [x for x in (total.get("sources") or "").split("; ") if x]
                    + _sources(sp["rs"]).split("; ")))
                if sp["eid"] not in (total.get("examples") or ""):
                    total["examples"] = "; ".join(sorted(
                        [x for x in (total.get("examples") or "").split("; ") if x] + [sp["eid"]]))
                    total["n"] = int(total.get("n") or 0) + 1
                tot = csum
                stats["range-only totals given the carrier sum"] += 1
        elif tot is None and len({t for e in exs for t in e["top"]}) == 1 and \
                (one := next(iter({t for e in exs for t in e["top"]}))) in \
                {(s.get("target") or "").strip().lower() for s in rows} and \
                fnum(next(s for s in rows if (s.get("target") or "").strip().lower() == one)
                     .get("central")) not in (None, 0):
            # an archetype whose examples all name the same ONE level-1 carrier
            # (electricity on a lamp, fuel on a dryer): the total is that carrier's
            # pooled median, not one example's line (lighting 2026-10-14; every
            # archetype, modeller 2026-10-21)
            src = next(s for s in rows if (s.get("target") or "").strip().lower() == one)
            total = dict(src, **{"target": "", "central-from":
                                 f"the pooled {one}, the only carrier any example names"})
            summaries.append(total)
            rows.append(total)
            tot = fnum(total["central"])
            single = True
            stats["totals from a single pooled carrier"] += 1
        elif tot is None and len(_l1_pooled(rows)) > len(sp["top"]) \
                and sum(_l1_pooled(rows).values()) > csum and _additive(_l1_pooled(rows)):
            # the examples name different carriers (an electricity line from one
            # source, the fuel firing from another): the total is the sum of the
            # pooled carriers, and each keeps its pooled value (S135/RX007: a
            # refinery's total had been its electricity alone, beside 2.2 GJ/m3 fuel)
            l1p = _l1_pooled(rows)
            total = dict(rows[0], **{"target": "", "central": f"{sum(l1p.values()):.6g}", "low": "",
                                     "high": "", "n": 0, "n-external": 0, "external-factors": "",
                                     "central-from": "sum of the pooled carriers, which different examples "
                                                     "give: " + " + ".join(f"{t} {v:.4g}" for t, v in sorted(l1p.items()))
                                                     + " (S135)",
                                     "sources": "; ".join(dict.fromkeys(x for s_ in rows for x in (s_.get("sources") or "").split("; ") if x)),
                                     "examples": ""})
            summaries.append(total)
            rows.append(total)
            tot = sum(l1p.values())
            csum = tot
            sp = dict(sp, top=dict(l1p))
            keep_pooled = True
            stats["totals summed from carriers of different examples"] += 1
        elif tot is None:
            total = dict(rows[0], **{"target": "", "central": f"{csum:.6g}", "low": "", "high": "",
                                     "n": 1, "n-external": 0, "external-factors": "",
                                     "central-from": f"sum of the carriers of {short} ({sp['eid']})",
                                     "sources": _sources(sp["rs"]), "examples": sp["eid"]})
            summaries.append(total)
            rows.append(total)
            tot = csum
            stats["totals derived from carriers"] += 1
        else:
            stats["totals kept (pooled)"] += 1
            # a total from fewer than three examples against a larger pool of the
            # one carrier every example names: the carrier pool sets the total
            # (sanity-check 2026-10-23, S135/NFM003: one NRCan gold-mill total had
            # replaced ten electricity rows on the SAG-ball mill)
            l1 = {(s.get("target") or "").strip().lower(): s for s in rows
                  if _level1(s.get("target")) == (s.get("target") or "").strip().lower()
                  and (s.get("target") or "").strip() and (fnum(s.get("central")) or 0) > 0}
            tn = int(total.get("n") or 0)
            a = records.by_key.get(key[0]) or {}
            listed = {_level1(t) for t in _listed_energy(a)} - {""}
            if len(l1) == 1 and tn < 3:
                one, src = next(iter(l1.items()))
                # only where the total is impossible -- below the one carrier it
                # must contain -- and that carrier is all the archetype lists (a
                # pool of electricity lines is not the whole of a fuel-fired kiln)
                # (2026-10-24: either way where the pool has 3 or more examples and
                # the carrier is the archetype's only energy carrier once its
                # feedstock is left out, as hydrogen on the Haber-Bosch loop)
                sn = int(src.get("n") or 0)
                if (sn > tn and listed <= {one}
                        and (fnum(src["central"]) > tot or (sn >= 3 and listed == {one}))):
                    old = tot
                    for k in ("central", "low", "high"):
                        total[k] = src.get(k, "")
                    total["central-from"] = (
                        f"the pooled {one} (n={src.get('n')}), the archetype's one energy carrier; "
                        f"it replaces a total from {tn} example(s), {old:.4g} "
                        "(sanity-check 2026-10-23, S135)")
                    total["sources"] = "; ".join(dict.fromkeys(
                        [x for x in ((total.get("sources") or "") + "; " + (src.get("sources") or "")).split("; ") if x]))
                    tot = fnum(total["central"])
                    single = True
                    stats["small totals replaced by the larger carrier pool"] += 1
        tlo, thi = fnum(total.get("low")), fnum(total.get("high"))
        # with one carrier the carrier IS the total: it takes the pooled value
        # even where the split example's own line is far from it
        agree = single or (tot > 0 and 0.5 <= csum / tot <= 2.0)
        if not agree:
            total["central-from"] = (total.get("central-from") or "") + (
                f"; DISAGREES with the split example's carrier sum {csum:.3g} ({short}, {sp['eid']}): "
                "one of the two is on another scope; carriers left unscaled")
            stats["totals disagreeing with carriers"] += 1
        have = {(s.get("target") or "").strip().lower(): s for s in rows
                if (s.get("target") or "").strip()}

        def row_for(t, src_rs, eid):
            s = have.get(t)
            if s is None:
                s = dict(total, **{"target": t, "n": 1, "n-external": 0, "external-factors": "",
                                   "sources": _sources(src_rs), "examples": eid})
                summaries.append(s)
                have[t] = s
            return s

        # a split example that names ONE level-1 carrier is not a 100% share of a
        # multi-carrier total (sanity-check 2026-10-23, S135/CHEM002, CHEM014): the
        # other carriers keep their pooled values and this carrier takes the rest;
        # with no pooled other, the carrier keeps its own pooled value and the
        # rest goes to the one other carrier the archetype lists, if there is one
        in_split = set()
        if keep_pooled:
            for t in sp["top"]:
                s = have.get(t)
                if s is not None:
                    s["central-from"] = (s.get("central-from") or "") + "; part of the total as pooled (S135)"
                    in_split.add(t)
        if agree and not single and not keep_pooled and len(sp["top"]) == 1 and tot and tot > 0:
            c = next(iter(sp["top"]))
            # steam or heat beside fuel is the same energy on another scope (the
            # fuel that raised the steam), not a second carrier to add
            alt = STEAMLIKE if c == FUEL_AGG else ({FUEL_AGG} if c in STEAMLIKE else set())
            others = {t: s for t, s in have.items() if t != c and _level1(t) == t
                      and t not in NOT_CARRIER and t not in alt and not _apart_only_zero(s)}
            osum = sum(fnum(s["central"]) for s in others.values())
            a = records.by_key.get(key[0]) or {}
            listed = [_level1(t) for t in _listed_energy(a)]
            other_listed = sorted({t for t in listed if t and t != c and t not in alt})
            own_c = fnum(have[c]["central"]) if c in have and have[c].get("central") else None
            if others and 0 < osum <= 0.75 * tot:
                rem = tot - osum
                s = row_for(c, sp["rs"], sp["eid"])
                s["central"] = f"{rem:.6g}"
                s["low"] = f"{tlo * rem / tot:.6g}" if tlo is not None else ""
                s["high"] = f"{thi * rem / tot:.6g}" if thi is not None else ""
                s["central-from"] = (f"the total less the pooled {', '.join(sorted(others))} "
                                     f"({osum:.4g}): {short} ({sp['eid']}) names only {c}, so it is "
                                     "not a 100% share (S135)")
                for t, o in others.items():
                    o["central-from"] = (o.get("central-from") or "") + \
                        f"; part of the split: its pooled value, with {c} the rest of the total"
                in_split |= set(others) | {c}
                stats["single-carrier examples given the remainder"] += 1
            elif not others and len(other_listed) == 1 and own_c and 0 < own_c < tot:
                rest = other_listed[0]
                s = have[c]
                s["central-from"] = (s.get("central-from") or "") + (
                    f"; its own pooled value: {short} ({sp['eid']}) names only {c}, so it is not "
                    f"a 100% share; the rest of the total goes to {rest}, the archetype's other "
                    "listed carrier (S135)")
                r2 = row_for(rest, sp["rs"], sp["eid"])
                r2.update({"central": f"{tot - own_c:.6g}", "low": "", "high": "", "n": 0,
                           "central-from": f"the total less {c} ({own_c:.4g}): the archetype's other "
                                           "listed carrier (S135)"})
                in_split |= {c, rest}
                stats["single-carrier examples kept at their own value"] += 1
        for t, v in sp["top"].items():
            if t in in_split:
                continue
            share = v / csum
            s = row_for(t, sp["rs"], sp["eid"])
            if agree:
                s["central"] = f"{share * tot:.6g}"
                s["low"] = f"{share * tlo:.6g}" if tlo is not None else ""
                s["high"] = f"{share * thi:.6g}" if thi is not None else ""
                s["central-from"] = f"share {share:.1%} of the total, split from {short} ({sp['eid']})"
                stats["carriers set as shares"] += 1
            else:
                s["central"] = f"{v:.6g}"
                s["low"], s["high"] = "", ""
                s["central-from"] = (f"as the split example states it ({short}, {sp['eid']}; share "
                                     f"{share:.1%} of its carrier sum {csum:.3g}, which disagrees "
                                     f"with the pooled total {tot:.3g})")
                stats["carriers left unscaled"] += 1
            if t == FUEL_AGG and FUEL_AGG not in {_target(r) for r in sp["rs"]}:
                s["central-from"] += "; fuel = the sum of the named fuels this example gives"
        # ---- level 2: named fuels as a split of `fuel` --------------------------
        fuel_row = have.get(FUEL_AGG)
        f_tot = fnum(fuel_row["central"]) if fuel_row and fuel_row.get("central") else None
        fcands = []
        for e in exs:
            if not _ok(e["named"]):
                continue
            nsum = sum(e["named"].values())
            closeness = abs(log(nsum / f_tot)) if f_tot and f_tot > 0 else 0.0
            fcands.append(((len(e["named"]) < 2, closeness, e["rank"], e["eid"]), e, nsum))
        named_split = set()
        # where `fuel` itself is pooled apart (not in the level-1 split), its named
        # fuels are too: splitting them would put them in totals that leave the
        # fuel out (2026-10-24, reactor_alkylation)
        fuel_in = FUEL_AGG in sp["top"] or FUEL_AGG in in_split
        if fcands and f_tot and f_tot > 0 and fuel_in:
            _, fs, nsum = min(fcands, key=lambda c: c[0])
            fshort = _short(records, fs["rs"])
            flo, fhi = fnum(fuel_row.get("low")), fnum(fuel_row.get("high"))
            for t, v in fs["named"].items():
                share = v / nsum
                s = row_for(t, fs["rs"], fs["eid"])
                s["central"] = f"{share * f_tot:.6g}"
                s["low"] = f"{share * flo:.6g}" if flo is not None else ""
                s["high"] = f"{share * fhi:.6g}" if fhi is not None else ""
                s["central-from"] = f"share {share:.1%} of fuel, split from {fshort} ({fs['eid']})"
                named_split.add(t)
                stats["fuels set as shares of fuel"] += 1
        for t, s in have.items():
            if t in NOT_CARRIER or t in sp["top"] or t in named_split or t in in_split:
                continue
            level = "of fuel" if t not in NO_HV_CARRIERS and t != FUEL_AGG else "of the total"
            s["central-from"] = (s.get("central-from") or "") + \
                f"; pooled apart: not part of the split {level}, so it does not sum with the others"
            stats["pooled apart"] += 1
    return dict(stats)
