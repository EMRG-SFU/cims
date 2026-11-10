"""One set of energy carriers per archetype that adds up, on the most likely fuel
(Brad 2026-11-07), and the alternatives kept beside it.

The carriers step (carriers.py) splits a total by one example's shares and leaves
what that example does not name "pooled apart": other fuels, purchased steam
beside on-site firing, a second name for the same heat. Those rows sat in the
wide view beside the split and could be added up by mistake. This step:

  * moves every row pooled apart out of the archetype's carriers, unless
    records/reference/fuel-choice.csv names it in `add` (a complementary need,
    such as fan electricity beside a fuel-fired dryer, which then joins the
    total);
  * puts `fuel` on the archetype's most likely fuel (mode `single`), or on the
    fuels a co-fired process needs (mode `mix`, scaled to `fuel` with their
    current values as weights), or leaves the split's own named fuels (`as-is`
    and archetypes not in the table); the other named fuels move out;
  * drops zero-valued named-fuel guesses ("could burn this") from the carriers;
  * keeps every moved row in `alternatives`, written by the build to
    archetype-alternative-carrier.csv, which is not part of the wide view or the
    workbook (Brad 2026-11-07: "a separate table that is not included in the wide
    view").

Rules: most likely = the fuel typical for the technology in Canadian industry,
then the fuel most examples name; co-fired mixtures keep every fuel the process
needs (Brad 2026-11-07). The per-archetype calls in fuel-choice.csv were made by
a Fable reviewer under those rules, for Brad's review. A variant with no row of
its own takes its host's (fill/host-archetype.csv).

It runs inside the fill twice: once on the pooled rows before the first rules
(so the fuel's CO2, the energy cost and every variant follow the chosen fuel),
and once after the judgements, which draft carriers of their own.
"""
from __future__ import annotations

from .common import REFERENCE, fnum, read, split_list

SEC = "specific-energy-consumption"
FUEL = "fuel"
NO_HV = {"electricity", "heat", "steam", "cooling", "hot-water", "district-heat", "waste-heat",
         "solar", "geothermal", "nuclear", "hydrogen-as-feed"}
STEAMLIKE = {"steam", "heat", "hot-water", "district-heat", "waste-heat"}
NOT_CARRIER = {"", "component", "total", "not-stated"}
MARK = "pooled apart"

alternatives: list[dict] = []        # rows moved out, for the alternatives table


def table() -> dict:
    p = REFERENCE / "fuel-choice.csv"
    return {r["archetype"]: r for r in read(p)} if p.exists() else {}


def _is_named(t: str) -> bool:
    return bool(t) and t not in NO_HV and t != FUEL and t not in NOT_CARRIER \
        and not t.startswith("as-sourced") and ":" not in t


def _apart(r) -> bool:
    return MARK in (r.get("central-from") or "")


def _g(x):
    return "" if x is None else f"{x:.6g}"


def _scale(r, f):
    for k in ("central", "low", "high"):
        v = fnum(r.get(k))
        if v is not None:
            r[k] = _g(v * f)


def _worse(r, grade="estimated"):
    order = ["sourced", "identity", "estimated", "proxy", "guess"]
    g = r.get("grade") or "sourced"
    if g in order and order.index(g) < order.index(grade):
        r["grade"] = grade


def apply(psum: list[dict], records, hosts: dict, have: dict | None = None, stage: str = "",
          closure: bool = True, choose: bool = True) -> dict:
    """Edit psum in place (and `have`, the fill's index, if given)."""
    choice = table()
    stats = {"moved to alternatives": 0, "zero guesses dropped": 0, "fuels renamed or rescaled": 0,
             "carriers added to the total": 0}
    by = {}
    for r in psum:
        if r["parameter"] == SEC:
            by.setdefault(r["archetype"], []).append(r)
    known = {a["archetype"] for a in records.archetypes}

    def drop(r, relation, why):
        if r in psum:
            psum.remove(r)
        if have is not None:
            h = have.get(r["archetype"], {})
            if h.get((r["parameter"], r.get("target", ""))) is r:
                del h[(r["parameter"], r.get("target", ""))]
        alternatives.append(dict(r, **{"relation": relation, "why": why, "stage": stage}))
        stats["zero guesses dropped" if relation.startswith("listed possible") else "moved to alternatives"] += 1

    for a, rows in by.items():
        if a not in known:
            continue
        c = choice.get(a) or choice.get((hosts.get(a) or {}).get("host", "")) or {}
        mode = c.get("mode", "")
        tag = f"fuel choice ({'own row' if a in choice else 'host ' + (hosts.get(a) or {}).get('host', '')}"\
              f", reference/fuel-choice.csv, Brad 2026-11-07 rules)"
        rows = [r for r in rows if not (r.get("target") or "").startswith("as-sourced")]
        tot = next((r for r in rows if not (r.get("target") or "")), None)
        tgt = {(r.get("target") or ""): r for r in rows}
        adds = set(split_list(c.get("add", "").replace("|", ";")))
        if not choose:
            tgt = {}
        # ---- level 1: rows pooled apart -----------------------------------
        for t, r in list(tgt.items()):
            if not t or not _apart(r) or _is_named(t):
                continue
            if t in adds and tot is not None:
                v = fnum(r.get("central")) or 0.0
                for k in ("central", "low", "high"):
                    tv = fnum(tot.get(k))
                    if tv is not None:
                        tot[k] = _g(tv + (fnum(r.get(k)) if fnum(r.get(k)) is not None else v))
                r["central-from"] = (r["central-from"].replace(MARK, "kept apart by the carriers step, now in the split") +
                                     f"; added to the total as a complementary need ({tag})")
                tot["central-from"] = (tot.get("central-from") or "") + f"; plus {t} {v:.4g} ({tag})"
                _worse(tot)
                stats["carriers added to the total"] += 1
            else:
                rel = ("purchased steam instead of on-site firing" if t == "steam" and FUEL in tgt and not _apart(tgt[FUEL])
                       else "heat of unstated source instead of the fuel" if t in STEAMLIKE and FUEL in tgt and not _apart(tgt[FUEL])
                       else "on-site firing instead of purchased steam or heat" if t == FUEL
                       else "the same heat under another name" if t in STEAMLIKE
                       else "another estimate, not part of the split")
                drop(r, rel, "pooled apart: not part of the split")
                tgt.pop(t, None)
        # ---- level 2: named fuels -------------------------------------------
        frow = tgt.get(FUEL)
        named = {t: r for t, r in tgt.items() if _is_named(t)}
        fuels = [x for x in split_list(c.get("fuels", "").replace("|", ";"))]
        if not choose:
            pass
        elif mode == "single" and fuels:
            keep = fuels[0]
            base = frow
            if base is None:
                # no fuel row: the named fuels in the split are the fuel
                live = [r for t, r in named.items() if not _apart(r)]
                s = sum(fnum(r.get("central")) or 0 for r in live)
                if s <= 0 and keep not in named:
                    pass
                else:
                    base = {"central": _g(s), "low": "", "high": "",
                            "unit": (live or list(named.values()))[0]["unit"]}
            if base is not None:
                cur = named.get(keep)
                cv, bv = fnum(cur.get("central")) if cur else None, fnum(base.get("central"))
                if cur is not None and cv and bv and abs(cv - bv) > 0.05 * abs(bv):
                    if (fnum(cur.get("central")) or 0) > 0 and _apart(cur):
                        alternatives.append(dict(cur, **{"relation": "another estimate of the same fuel",
                                                          "why": "pooled apart from the split", "stage": stage}))
                        stats["moved to alternatives"] += 1
                if cur is None:
                    cur = dict(base, **{"target": keep, "archetype": a, "parameter": SEC, "n": 0,
                                        "examples": base.get("examples", ""), "sources": base.get("sources", "")})
                    psum.append(cur)
                    if have is not None:
                        have.setdefault(a, {})[(SEC, keep)] = cur
                    named[keep] = cur
                    tgt[keep] = cur
                old = cur.get("central")
                for k in ("central", "low", "high"):
                    cur[k] = base.get(k, "")
                if not (cur.get("central-from") or "").startswith("all of fuel, named"):
                    cur["central-from"] = (f"all of fuel, named {keep}: the most likely fuel ({tag}); "
                                           f"was {old or 'none'}; " + (cur.get("central-from") or "").replace(MARK, "kept apart by the carriers step, now in the split"))
                    stats["fuels renamed or rescaled"] += 1
                _worse(cur)
                for t, r in list(named.items()):
                    if t != keep:
                        v = fnum(r.get("central")) or 0
                        drop(r, "alternative fuel" if v > 0 else "listed possible fuel, no quantity",
                             f"not the most likely fuel ({keep})")
        elif mode == "mix" and fuels:
            shares = [fnum(x) for x in split_list(c.get("shares", "").replace("|", ";"))]
            if shares and len(shares) == len(fuels):
                # shares the table gives (a Canadian fuel mix): rows written where missing
                for t, sh in zip(fuels, shares):
                    if t not in named:
                        base = frow or next(iter(named.values()), None)
                        if base is None:
                            continue
                        nr = dict(base, **{"target": t, "n": 0, "examples": "", "central-from": ""})
                        psum.append(nr)
                        if have is not None:
                            have.setdefault(a, {})[(SEC, t)] = nr
                        named[t] = nr
                        tgt[t] = nr
                w = {t: sh for t, sh in zip(fuels, shares) if t in named}
            else:
                w = {t: fnum(named[t].get("central")) or 0 for t in fuels if t in named}
            target = fnum(frow.get("central")) if frow is not None else \
                sum(fnum(r.get("central")) or 0 for t, r in named.items() if not _apart(r))
            s = sum(w.values())
            if s > 0 and target:
                f = target / s
                for t in w:
                    r = named[t]
                    if shares and len(shares) == len(fuels):
                        for kk in ("central", "low", "high"):
                            fv_ = fnum(frow.get(kk)) if frow is not None else None
                            r[kk] = _g(fv_ * w[t] / s) if fv_ is not None else ("" if kk != "central" else _g(target * w[t] / s))
                        r["central-from"] = (f"{w[t] / s:.0%} of fuel, the share in reference/fuel-choice.csv ({tag}); "
                                             + (r.get("central-from") or "").replace(MARK, "kept apart by the carriers step, now in the split"))
                        _worse(r)
                        stats["fuels renamed or rescaled"] += 1
                    elif abs(f - 1) > 1e-6 or _apart(r):
                        _scale(r, f)
                        r["central-from"] = (f"co-fired share of fuel ({', '.join(fuels)}), scaled x{f:.3g} to the fuel "
                                             f"total ({tag}); " + (r.get("central-from") or "").replace(MARK, "kept apart by the carriers step, now in the split"))
                        _worse(r)
                        stats["fuels renamed or rescaled"] += 1
            for t, r in list(named.items()):
                if t not in fuels:
                    v = fnum(r.get("central")) or 0
                    drop(r, "alternative fuel" if v > 0 else "listed possible fuel, no quantity",
                         "not one of the co-fired fuels the process needs")
        else:
            for t, r in list(named.items()):
                v = fnum(r.get("central")) or 0
                if _apart(r):
                    drop(r, "alternative fuel" if v > 0 else "listed possible fuel, no quantity",
                         "pooled apart: not part of the fuel split")
                elif v == 0 and int(r.get("n") or 0) == 0 and c.get("drop-zero", "yes") != "no":
                    drop(r, "listed possible fuel, no quantity", "zero-valued guess")
        if not closure:
            continue
        # ---- closure: the named fuels left in the split add up to `fuel` ----------
        frow = next((r for r in psum if r["archetype"] == a and r["parameter"] == SEC
                     and (r.get("target") or "") == FUEL), None)
        left = [r for r in psum if r["archetype"] == a and r["parameter"] == SEC
                and _is_named(r.get("target") or "") and not _apart(r)]
        fv = fnum(frow.get("central")) if frow is not None else None
        s_ = sum(fnum(r.get("central")) or 0 for r in left)
        derived_fuel = frow is not None and (frow.get("grade") == "identity" or (
            int(frow.get("n") or 0) == 0 and (frow.get("method") or "pooled") in ("pooled", "")))
        if fv and s_ > fv * 1.02 and frow is not None and derived_fuel:
            # named fuels above `fuel`: fuel is raised to them, as the fill's carrier
            # sum does (2026-10-21)
            frow["central-from"] = (f"raised to the sum of the named fuels, was {fv:.4g} (Brad 2026-11-07: "
                                    "the splits add up); " + (frow.get("central-from") or ""))
            frow["central"], frow["low"], frow["high"] = _g(s_), "", ""
            _worse(frow)
            stats["fuel raised to its named fuels"] = stats.get("fuel raised to its named fuels", 0) + 1
        elif fv and s_ > 0 and abs(s_ - fv) > 0.02 * fv:
            f = fv / s_
            for r in left:
                _scale(r, f)
                r["central-from"] = (f"scaled x{f:.3g} so the named fuels add up to fuel ({fv:.4g}; Brad 2026-11-07: "
                                     "the splits add up); " + (r.get("central-from") or ""))
                _worse(r)
            stats["named fuels scaled to fuel"] = stats.get("named fuels scaled to fuel", 0) + 1
        # ---- closure: the level-1 carriers add up to the total ---------------
        # where the carriers are additive (no fuel beside the steam it raised, no
        # two names for one heat). A sourced total pooled from more examples than
        # any carrier keeps its value and the carriers take its scale; otherwise
        # the total becomes the carrier sum (Brad 2026-11-07: the splits add up)
        l1 = [r for r in psum if r["archetype"] == a and r["parameter"] == SEC and not _apart(r)
              and ((r.get("target") or "") in NO_HV or (r.get("target") or "") == FUEL)
              and fnum(r.get("central")) is not None]
        names = {(r.get("target") or "") for r in l1}
        nf = [r for r in psum if r["archetype"] == a and r["parameter"] == SEC
              and _is_named(r.get("target") or "") and not _apart(r) and fnum(r.get("central"))]
        if FUEL not in names and nf:
            # named fuels with no `fuel` row: write it, as the carriers step does
            fr = dict(nf[0], **{"target": FUEL, "central": _g(sum(fnum(r["central"]) for r in nf)),
                                "low": "", "high": "", "central-from": "fuel = the sum of the named fuels "
                                "(fuel_choice, 2026-11-07)", "examples": "", "n": 0})
            psum.append(fr)
            if have is not None:
                have.setdefault(a, {})[(SEC, FUEL)] = fr
            l1.append(fr)
            names.add(FUEL)
        steam_like = names & STEAMLIKE
        if not l1 or (FUEL in names and steam_like) or len(steam_like) > 1:
            continue
        tot = next((r for r in psum if r["archetype"] == a and r["parameter"] == SEC
                    and not (r.get("target") or "")), None)
        cs = sum(fnum(r["central"]) for r in l1)
        tv = fnum(tot.get("central")) if tot is not None else None
        if tot is None or cs <= 0:
            continue
        if tv is not None and tv > 0 and abs(cs - tv) <= 0.02 * tv:
            continue
        tn = int(tot.get("n") or 0)
        cn = max(int(r.get("n") or 0) for r in l1)
        if tv and tv > 0 and cs < tv and (tot.get("grade") or "sourced") == "sourced" and tn > cn:
            f = tv / cs
            for r in l1:
                _scale(r, f)
                r["central-from"] = (f"scaled x{f:.3g} to the total ({tv:.4g}, n={tn}), which rests on more examples "
                                     "than any carrier (Brad 2026-11-07: the splits add up); " + (r.get("central-from") or ""))
                _worse(r)
            stats["carriers scaled to the total"] = stats.get("carriers scaled to the total", 0) + 1
        else:
            old = tot.get("central")
            lo = sum(fnum(r.get("low")) if fnum(r.get("low")) is not None else fnum(r["central"]) for r in l1)
            hi = sum(fnum(r.get("high")) if fnum(r.get("high")) is not None else fnum(r["central"]) for r in l1)
            tot.update({"central": _g(cs), "low": _g(lo) if lo != cs else "", "high": _g(hi) if hi != cs else ""})
            if tot.get("grade") == "not-applicable":
                tot["grade"], tot["method"] = "guess", "carrier-sum"
            tot["central-from"] = (f"the sum of the carriers ({', '.join(sorted(names))}), was {old or 'none'} "
                                   "(Brad 2026-11-07: the splits add up); " + (tot.get("central-from") or ""))
            _worse(tot)
            stats["totals set to the carrier sum"] = stats.get("totals set to the carrier sum", 0) + 1
    return stats


def drop_zero_guesses(psum: list[dict], have: dict | None = None) -> int:
    """Carrier guesses left at zero (a drafted carrier the fill scaled to 0 to fit a
    total): they say only that the archetype could use the carrier, so they leave
    the carriers, to the alternatives table (2026-11-07)."""
    n = 0
    for r in list(psum):
        t = r.get("target") or ""
        if r["parameter"] != SEC or not t or t.startswith("as-sourced") or (fnum(r.get("central")) or 0) != 0:
            continue
        if r.get("grade") not in ("guess", "proxy") or int(r.get("n") or 0) > 0:
            continue
        psum.remove(r)
        if have is not None and have.get(r["archetype"], {}).get((SEC, t)) is r:
            del have[r["archetype"]][(SEC, t)]
        alternatives.append(dict(r, relation="listed possible fuel, no quantity" if _is_named(t) or t == FUEL
                                 else "listed possible carrier, no quantity", why="zero-valued guess", stage="final"))
        n += 1
    return n


def alt_rows(psum: list[dict], records) -> list[dict]:
    """The alternatives table: each moved row with what it stands beside in the
    wide view, a ranking, and for steam the self-generation equivalent."""
    have = {}
    for r in psum:
        have.setdefault(r["archetype"], {})[(r["parameter"], r.get("target", ""))] = r
    boiler = have.get("boiler_fuel-fired", {}).get(("efficiency", ""))
    eta = (fnum(boiler.get("central")) or 82.0) / 100 if boiler else 0.82
    seen = {}
    for r in alternatives:
        k = (r["archetype"], r.get("target", ""))
        if k not in seen or (fnum(r.get("central")) or 0) > 0:
            seen[k] = r
    out = []
    by = {}
    for (a, t), r in seen.items():
        h = have.get(a, {})
        if (SEC, t) in h and r.get("relation") != "another estimate of the same fuel":
            continue        # it came back (a variant rule, a judgement): not an alternative
        by.setdefault(a, []).append(r)
    for a, rs in sorted(by.items()):
        h = have.get(a, {})
        wv = {t: fnum(x.get("central")) for (p, t), x in h.items() if p == SEC and t and fnum(x.get("central"))}
        fuel_v = wv.get(FUEL) or sum(v for t, v in wv.items() if _is_named(t)) or None
        steam_v = sum(v for t, v in wv.items() if t in STEAMLIKE) or None
        named_wv = ", ".join(f"{t} {v:.4g}" for t, v in sorted(wv.items()) if _is_named(t))
        rs = sorted(rs, key=lambda r: (-int(r.get("n") or 0), r.get("target", "")))
        rank = 0
        for r in rs:
            t = r.get("target", "")
            v = fnum(r.get("central"))
            rel = r.get("relation", "")
            note = ""
            if v is None:
                pass
            elif t in STEAMLIKE and fuel_v:
                note = (f"purchased {t} {v:.4g} GJ needs about {v / eta:.4g} GJ of fuel in an on-site boiler at "
                        f"{eta:.0%} (boiler_fuel-fired efficiency); the wide view fires {fuel_v:.4g} GJ of fuel on site")
            elif (t == FUEL or _is_named(t)) and steam_v and not (wv.get(FUEL) or named_wv):
                note = (f"on-site firing of {v:.4g} GJ raises about {v * eta:.4g} GJ of steam at {eta:.0%} "
                        f"(boiler_fuel-fired efficiency); the wide view buys {steam_v:.4g} GJ of steam or heat")
            elif _is_named(t) and rel == "alternative fuel" and fuel_v and (r.get("central-from") or "").startswith("share"):
                # a share of a fleet mix is not what the plant would burn on this fuel
                # alone: the alternative is the same energy on this fuel
                note = (f"in the split example's fuel mix at {v:.4g} GJ ({v / fuel_v:.0%} of fuel); shown at the wide "
                        f"view's fuel total, the same energy on this fuel alone")
                r = dict(r, central=f"{fuel_v:.6g}", low="", high="", grade="estimated")
            elif _is_named(t):
                note = f"its own pooled value, beside the wide view's fuel ({named_wv or (f'fuel {fuel_v:.4g}' if fuel_v else 'none')})"
            if rel.startswith(("alternative fuel", "purchased", "on-site", "heat of")):
                rank += 1
                rk = rank
            else:
                rk = ""
            out.append({"archetype": a, "carrier": t, "relation": rel, "rank": rk,
                        "central": r.get("central", ""), "low": r.get("low", ""), "high": r.get("high", ""),
                        "unit": r.get("unit", ""), "n": r.get("n", ""), "grade": r.get("grade", ""),
                        "wide-view-fuel": named_wv or (f"fuel {fuel_v:.4g}" if fuel_v else ""),
                        "comparison": note, "why-not-in-wide-view": r.get("why", ""),
                        "central-from": (r.get("central-from") or ""), "sources": r.get("sources", ""),
                        "examples": r.get("examples", "")})
    return out


ALT_COLS = ["archetype", "carrier", "relation", "rank", "central", "low", "high", "unit", "n", "grade",
            "wide-view-fuel", "comparison", "why-not-in-wide-view", "central-from", "sources", "examples"]
