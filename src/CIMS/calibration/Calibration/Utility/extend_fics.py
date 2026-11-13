"""
Carry calibrated FICs forward into the forecast period.

Rules (all settings are arguments of `extend_fics`):

1. Start value. Each tech series gets one start value: the median of its
   ACTIVE fitted years in the trailing window (default 2017-2023). A year is
   active when the tech and every structural ancestor are available
   (`available` <= year < own `unavailable`). The placeholder zeros a node holds
   before it exists are never used.
2. Proxies. A series with fewer than `min_fitted_years` active fitted years
   (2001-2023) takes its start value from the proxy named in the proxy map.
   The proxy itself must have at least `min_fitted_years`. A series with no
   usable proxy falls back to its own median if it has at least
   `fallback_min_years` active years, otherwise it is skipped (no rows written,
   so the base FIC or the default applies) and logged.
3. Path. The start value is written for `first_year` and held constant
   through `end_year`, except for techs on the decline list: a POSITIVE start
   value declines linearly to 0 at `decline_zero_year` and stays 0. Negative
   start values on the decline list are held flat.

Output: <calibration_outputs>/<sector>/extended_fics/extended_fics_<region>.csv
(standard model-input columns, Source = `calibration_fic_extension`) plus
<calibration_outputs>/extended_fics_summary.csv with one row per series
showing how its start value was chosen. Re-running replaces both.

Load the output AFTER fitted_fics (add 'extended_fics' after 'fitted_fics' in
Reference.py's calibration_outputs list) so the extension rows win for
`first_year` onward.

Limitations: availability is read from the base model files only; scenario
files that change `available`/`unavailable` are not considered.
"""

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

COLUMNS = ["Branch", "Type", "Region", "Sector", "Service", "Technology", "Parameter",
           "Context", "Sub_Context", "Target", "Source", "Unit", "Year", "Value"]
SOURCE = "calibration_fic_extension"
CALIB_FIRST_YEAR = 2001  # 2000 is the base year and is never fitted


# --------------------------------------------------------------------------- inputs

def load_fitted_fics(outputs_dir):
    frames = []
    for path in sorted(glob.glob(os.path.join(outputs_dir, "*", "fitted_fics", "*.csv"))):
        sector_dir = os.path.basename(os.path.dirname(os.path.dirname(path)))
        frames.append(pd.read_csv(path, encoding="utf-8-sig").assign(sector_dir=sector_dir))
    return pd.concat(frames, ignore_index=True)


def load_availability(model_dir):
    """Return {(branch, tech): {'available': y, 'unavailable': y}} from the base model files."""
    frames = []
    for path in glob.glob(os.path.join(model_dir, "**", "*.csv"), recursive=True):
        try:
            df = pd.read_csv(path, usecols=["Branch", "Technology", "Parameter", "Value"],
                             dtype=str, encoding="utf-8-sig")
        except (ValueError, KeyError):
            continue
        df = df[df["Parameter"].isin(["available", "unavailable"])]
        if len(df):
            frames.append(df)
    av = pd.concat(frames, ignore_index=True)
    av["Value"] = pd.to_numeric(av["Value"], errors="coerce")
    av["Technology"] = av["Technology"].fillna("")
    table = av.pivot_table(index=["Branch", "Technology"], columns="Parameter",
                           values="Value", aggfunc="last")
    return table.to_dict("index")


def _techs_by_branch(avail):
    out = {}
    for (branch, tech), vals in avail.items():
        out.setdefault(branch, []).append((tech, vals))
    return out


def active_window(branch, tech, avail, by_branch, last_year):
    """First and last active calibration year for one series.

    Ancestors limit only the START (a node that does not exist yet has no stock);
    an ancestor's `unavailable` stops NEW stock of that node, not the competition
    among techs inside existing stock, so it is ignored.
    """
    lo, hi = CALIB_FIRST_YEAR, last_year
    comps = branch.split(".")
    for i in range(1, len(comps)):
        parent, child = ".".join(comps[:i]), comps[i]
        for t, vals in by_branch.get(parent, []):
            if t and (child == t or child.startswith(t + " ")):
                a = vals.get("available", np.nan)
                if a == a:
                    lo = max(lo, int(a))
    own = avail.get((branch, tech), {})
    a, u = own.get("available", np.nan), own.get("unavailable", np.nan)
    if a == a:
        lo = max(lo, int(a))
    if u == u:
        hi = min(hi, int(u) - 1)
    return lo, hi


def load_decline_list(path):
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig").fillna("")
    df = df[df["Include"].str.strip().str.lower().isin(["yes", "y", "true", "1"])]
    return [(r.Technology.strip(), re.compile(r.Node_Pattern) if r.Node_Pattern.strip() else None)
            for r in df.itertuples()]


def on_decline_list(branch, tech, rules):
    return any(tech == t and (pat is None or pat.search(branch)) for t, pat in rules)


# --------------------------------------------------------------------------- core

def extend_fics(outputs_dir, model_dir, proxy_map_path, decline_list_path,
                window=(2017, 2023), min_fitted_years=5, fallback_min_years=3,
                first_year=2024, end_year=2100, decline_zero_year=2074,
                write=True):
    calib_last = window[1]
    fitted = load_fitted_fics(outputs_dir)
    fitted = fitted[(fitted["Year"] >= CALIB_FIRST_YEAR) & (fitted["Year"] <= calib_last)]
    avail = load_availability(model_dir)
    by_branch = _techs_by_branch(avail)
    rules = load_decline_list(decline_list_path)

    meta = (fitted.groupby(["Branch", "Technology"])
            [["Region", "Sector", "Service", "sector_dir"]].first().to_dict("index"))
    series = {k: g.set_index("Year")["Value"]
              for k, g in fitted.groupby(["Branch", "Technology"])}

    # Own statistics for every fitted series
    own = {}
    for (b, t), s in series.items():
        lo, hi = active_window(b, t, avail, by_branch, calib_last)
        n_fit = max(0, hi - lo + 1)
        w = s[(s.index >= max(lo, window[0])) & (s.index <= hi)]
        own[(b, t)] = dict(first=lo, last=hi, n_fitted=n_fit, n_window=len(w),
                           median=float(w.median()) if len(w) else np.nan)

    pmap = pd.read_csv(proxy_map_path, dtype=str, encoding="utf-8-sig").fillna("")
    pmap = pmap[pmap["Proxy_Technology"].str.strip() != ""]
    proxies = {(r.Branch, r.Technology): (r.Proxy_Branch, r.Proxy_Technology)
               for r in pmap.itertuples()}

    keys = sorted(set(series) | set(proxies))
    summary, rows = [], []
    years = np.arange(first_year, end_year + 1)
    for b, t in keys:
        o = own.get((b, t), dict(first=None, last=None, n_fitted=0, n_window=0, median=np.nan))
        start, method, proxy = np.nan, "", ""
        if o["n_fitted"] >= min_fitted_years:
            start, method = o["median"], "own"
        else:
            p = proxies.get((b, t))
            po = own.get(p) if p else None
            if po and po["n_fitted"] >= min_fitted_years and po["median"] == po["median"]:
                start, method, proxy = po["median"], "proxy", f"{p[0]} | {p[1]}"
            elif o["n_window"] >= fallback_min_years:
                start, method = o["median"], "own_fallback"
            else:
                method = "skipped"
        declines = on_decline_list(b, t, rules)
        summary.append(dict(Branch=b, Technology=t, Method=method, Proxy=proxy,
                            Fitted_Years=o["n_fitted"], Window_Years=o["n_window"],
                            Start_Value=start, Declines=declines and start > 0,
                            On_Decline_List=declines))
        if method == "skipped":
            continue
        if declines and start > 0:
            frac = np.clip((decline_zero_year - years) / (decline_zero_year - first_year), 0, 1)
            values = start * frac
        else:
            values = np.full(len(years), start)

        m = meta.get((b, t))
        if m is None:  # series only reachable through a proxy (e.g. >2035 vintage)
            pm = meta.get(proxies[(b, t)], {})
            region = re.match(r"^CIMS\.CAN\.([A-Za-z]{2})", b).group(1)
            m = dict(Region=region, Sector=pm.get("Sector", ""),
                     Service=b.split(".")[-1], sector_dir=pm.get("sector_dir", ""))
        for y, v in zip(years, values):
            rows.append(dict(Branch=b, Type="", Region=m["Region"], Sector=m["Sector"],
                             Service=m["Service"], Technology=t, Parameter="fic",
                             Context="", Sub_Context="", Target="", Source=SOURCE,
                             Unit="", Year=int(y), Value=float(v), sector_dir=m["sector_dir"]))

    out = pd.DataFrame(rows)
    summ = pd.DataFrame(summary)
    if write:
        for (sector_dir, region), g in out.groupby(["sector_dir", "Region"]):
            d = os.path.join(outputs_dir, sector_dir, "extended_fics")
            os.makedirs(d, exist_ok=True)
            g[COLUMNS].to_csv(os.path.join(d, f"extended_fics_{region.lower()}.csv"),
                              index=False, lineterminator="\r\n")
        summ.to_csv(os.path.join(outputs_dir, "extended_fics_summary.csv"),
                    index=False, lineterminator="\r\n")
    return out, summ


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--outputs-dir", default="data/model_inputs/calibration_outputs")
    ap.add_argument("--model-dir", default="data/model_inputs/model")
    ap.add_argument("--proxy-map", default="data/model_inputs/calibration_outputs/fic-proxy-map.csv")
    ap.add_argument("--decline-list", default="data/model_inputs/calibration_outputs/fic-decline-techs.csv")
    ap.add_argument("--window", type=int, nargs=2, default=(2017, 2023))
    ap.add_argument("--min-fitted-years", type=int, default=5)
    ap.add_argument("--fallback-min-years", type=int, default=3)
    ap.add_argument("--first-year", type=int, default=2024)
    ap.add_argument("--end-year", type=int, default=2100)
    ap.add_argument("--decline-zero-year", type=int, default=2074)
    a = ap.parse_args()
    out, summ = extend_fics(a.outputs_dir, a.model_dir, a.proxy_map, a.decline_list,
                            tuple(a.window), a.min_fitted_years, a.fallback_min_years,
                            a.first_year, a.end_year, a.decline_zero_year)
    print(summ["Method"].value_counts().to_string())
    print(f"declining series: {int(summ['Declines'].sum())}; rows written: {len(out)}")


if __name__ == "__main__":
    main()
