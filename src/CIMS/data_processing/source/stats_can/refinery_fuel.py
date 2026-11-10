"""
Petroleum refinery fuel use by province and year, 2000-2024 (TJ).

Builds the fuel actually burned at refineries (NAICS 32411) by province, for
the CIMS Petroleum Refining energy demand counterfactual:

    Own (refinery-produced) fuels   still gas, petroleum coke, refinery LPG,
                                    heavy fuel oil
    Purchased fuels                 natural gas, electricity

Inputs
------
raw_data/stats_can/resd/25100029.csv                [S1] StatCan RESD 25-10-0029
processed_data/activity/petroleum_refining.csv     [S2] refinery crude input by
                                                   province (activity/petroleum_refining.py,
                                                   from StatCan 25-10-0014 and 25-10-0063)

Output: processed_data/stats_can/refinery_fuel.csv
    Columns: region, year, fuel, kind (own/purchased), tj, method

Pipeline use:
    from CIMS.data_processing.source.stats_can.refinery_fuel import load_refinery_fuel
    fuel = load_refinery_fuel()   # pandas, CIMS region codes

    # Swap CER refining fuels for this table in a cer_resd_demand-style frame (GJ).
    # Called by cer_resd_demand.main() and mecs_steam_calibration._load_basis().
    from CIMS.data_processing.source.stats_can.refinery_fuel import replace_cer_refinery_fuel
    cer_df = replace_cer_refinery_fuel(cer_df)

Where the amounts go
--------------------
    .Petroleum Refining (sector node): every fuel in this table, in full. CER
        rows for the same fuels at that node are dropped first; other CER
        refining rows (butane, feedstocks, non-energy products) are kept.
    Steam nodes are built by mecs_steam_calibration (see STEAM below).

STEAM (mecs_steam_calibration, same approach as the other sectors)
-----------------------------------------------------------------
S1. Steam fuel target = MECS refining steam share (US) x refining energy
    demand (fuels + electricity; feedstock and non-energy products excluded).
S2. The target is split by each province's mix of the boiler fuels in this
    table: still gas, natural gas, petroleum coke, heavy fuel oil, LPG
    (Canadian mix, as other sectors use the RESD mix). Capped at those fuels'
    total.
S3. Still gas share -> .Petroleum Refining.Still Gas.Steam (the main .Steam
    node has no still gas technology; Still Gas.Steam boilers credit .Steam
    in the model). The rest of the still gas -> .Still Gas.Heater.
S4. Other fuels -> .Petroleum Refining.Steam. Electric boilers = MECS
    electric share x energy demand, as other sectors.
Assumptions: US MECS steam share applies to Canada (Canada's colder climate,
with steam tracing and tank heating, likely means somewhat more steam); each
boiler fuel is used for steam in proportion to its share of refinery fuel.
Model flag: NB and NL have no petroleum coke technology on .Steam.

Results check (2020): Canada refinery fuel 265 PJ vs CER 223 PJ; published
provinces match RESD exactly; the difference is mostly AB still gas (CER
2.5 PJ, this table 38 PJ). 2001 national mix vs NRCan/CIEEDAC benchmark guide
(2002): still gas + coke 70% vs 66%, natural gas 16% vs 16%, HFO 9% vs 8%,
electricity 6% vs 7% (still gas vs coke split differs, likely FCC coke
classification). A per-fuel MECS steam split gave totals within ~10%.

Standalone:
    python refinery_fuel.py <25100029.csv> <petroleum_refining.csv> <output.csv>


WHY
---
RESD reports refinery fuel in two places:
  - 'Refined petroleum products manufacturing': PURCHASED fuels only (mostly
    natural gas and electricity).
  - 'Producer consumption': fuels that energy producers make and burn
    themselves. For still gas, petroleum coke, refinery LPG and heavy fuel oil
    these are refineries' own fuels, EXCEPT in Alberta, where the figure also
    includes oil sands upgraders (which burn the same process gas and coke).
Many provinces are suppressed (BC and NB in every year; NL, SK, QC in many).


METHOD (per fuel and year)
--------------------------
R1. Canada totals: published; the odd suppressed year is interpolated.
R2. Published provinces are kept as they are.
R3. Alberta, own fuels: refineries are separated from upgraders.
        AB refinery fuel = (Canada - AB) / (Canada crude - AB crude) x AB crude
    i.e. AB refineries burn the same own fuel per m3 of crude as refineries in
    the other provinces (which have no upgraders). The rest of AB's published
    value is upgrading and is excluded. Validation: GHGRP refinery emissions
    give AB refinery own fuel within ~10% of this for 2019-2024; AB refineries
    emit ~7% more CO2e per m3 than the rest of Canada (2020). CER's AB
    upgrading demand (~213 PJ own fuel, 2020) is close to the ~200 PJ residual.
    Alberta purchased fuels are refinery-only and are used as published.
R4. Suppressed provinces: residual (Canada - published, using AB's published
    value for own fuels) split by each suppressed province's share of
    refinery crude input [S2]. Negative residuals are set to 0.
R5. Provinces with no refinery crude input get 0 when suppressed. Nova
    Scotia's Dartmouth refinery (closed 2013) has no separate crude input in
    [S2], so any suppressed NS refinery fuel falls into NB/NL.

Not included: producer-consumption natural gas, NGLs and diesel (gas plants,
oil and gas production and vehicles, not refineries).
"""

import sys
from functools import lru_cache

import pandas as pd

YEARS = list(range(2000, 2025))

GEO = {
    "Newfoundland and Labrador": "NL", "Nova Scotia": "NS", "New Brunswick": "NB",
    "Quebec": "QC", "Ontario": "ON", "Manitoba": "MB", "Saskatchewan": "SK",
    "Alberta": "AB", "British Columbia": "BC", "Canada": "CAN",
}
PROVS = ["NL", "NS", "NB", "QC", "ON", "MB", "SK", "AB", "BC"]

PRODUCER = "Producer consumption"
REFINING = "Refined petroleum products manufacturing"

# RESD fuel type -> (CIMS fuel, RESD row, kind)
FUELS = {
    "Still gas":                                  ("Refinery Fuel Gas", PRODUCER, "own"),
    "Petroleum coke":                             ("Petroleum Coke",    PRODUCER, "own"),
    "Refinery liquefied petroleum gases (LPG's)": ("LPG",               PRODUCER, "own"),
    "Heavy fuel oil":                             ("Heavy Fuel Oil",    PRODUCER, "own"),
    "Natural gas":                                ("Natural Gas",       REFINING, "purchased"),
    "Primary electricity, hydro and nuclear":     ("Electricity",       REFINING, "purchased"),
}


def read_resd(path: str) -> pd.DataFrame:
    """RESD 25-10-0029 rows used here: region, year, resd_fuel, tj (NaN = suppressed)."""
    d = pd.read_csv(path, low_memory=False,
                    usecols=["REF_DATE", "GEO", "Fuel type", "Supply and demand characteristics", "VALUE"])
    d = d[d["GEO"].isin(GEO) & d["REF_DATE"].isin(YEARS)]
    keep = pd.Series(False, index=d.index)
    for fuel, (_, row, _) in FUELS.items():
        keep |= (d["Fuel type"] == fuel) & (d["Supply and demand characteristics"] == row)
    d = d[keep]
    return pd.DataFrame({
        "region": d["GEO"].map(GEO), "year": d["REF_DATE"].astype(int),
        "resd_fuel": d["Fuel type"], "tj": pd.to_numeric(d["VALUE"], errors="coerce"),
    })


def read_crude(path: str) -> pd.DataFrame:
    """Refinery crude input, m3, wide: year x region (missing regions = 0)."""
    a = pd.read_csv(path)
    w = a.pivot_table(index="Year", columns="Region", values="Value", aggfunc="sum")
    return w.reindex(index=YEARS, columns=PROVS).fillna(0.0)


def build(resd_path: str, crude_path: str) -> pd.DataFrame:
    resd = read_resd(resd_path)
    crude = read_crude(crude_path)
    rows = []
    for resd_fuel, (fuel, _, kind) in FUELS.items():
        w = (resd[resd["resd_fuel"] == resd_fuel]
             .pivot_table(index="year", columns="region", values="tj", aggfunc="first", dropna=False)
             .reindex(index=YEARS, columns=PROVS + ["CAN"]))
        can = w["CAN"].interpolate(limit_direction="both")                            # R1
        for y in YEARS:
            pub = w.loc[y, PROVS]
            out, how = {}, {}
            for p in PROVS:
                if pd.notna(pub[p]):
                    out[p], how[p] = float(pub[p]), "published"                       # R2
            if kind == "own" and pd.notna(pub["AB"]) and crude.loc[y, "AB"] > 0:      # R3
                ex_ab = can[y] - pub["AB"]
                ex_crude = crude.loc[y].sum() - crude.loc[y, "AB"]
                ab_ref = max(ex_ab, 0.0) / ex_crude * crude.loc[y, "AB"] if ex_crude > 0 else 0.0
                out["AB"] = min(ab_ref, float(pub["AB"]))
                how["AB"] = "AB refineries: Canada-ex-AB per m3 x AB crude (upgraders removed)"
            supp = [p for p in PROVS if pd.isna(pub[p])]                                # R4, R5
            resid = can[y] - pub[[p for p in PROVS if p not in supp]].sum()
            resid = max(float(resid), 0.0)
            wts = crude.loc[y, supp]
            for p in supp:
                share = wts[p] / wts.sum() if wts.sum() > 0 else 0.0
                out[p] = resid * share
                how[p] = "suppressed: residual x crude input share" if share > 0 else "suppressed: no refinery crude, 0"
            for p in PROVS:
                rows.append((p, y, fuel, kind, out[p], how[p]))
    df = pd.DataFrame(rows, columns=["region", "year", "fuel", "kind", "tj", "method"])
    check(df, resd, crude)
    return df.sort_values(["region", "year", "fuel"]).reset_index(drop=True)


def check(df: pd.DataFrame, resd: pd.DataFrame, crude: pd.DataFrame, tol: float = 2.0) -> None:
    """Published values kept; purchased fuels add to Canada; no negatives."""
    if (df["tj"] < -tol).any() or df["tj"].isna().any():
        raise ValueError("negative or missing refinery fuel")
    for resd_fuel, (fuel, _, kind) in FUELS.items():
        r = resd[(resd["resd_fuel"] == resd_fuel)]
        got = df[df["fuel"] == fuel].set_index(["region", "year"])["tj"]
        for _, x in r[r["region"].isin(PROVS) & r["tj"].notna()].iterrows():
            if kind == "own" and x["region"] == "AB":
                continue
            if abs(got[(x["region"], x["year"])] - x["tj"]) > tol:
                raise ValueError(f"{fuel} {x['region']} {x['year']}: published value changed")


# --------------------------------------------------------------------------
SECTOR_NODE = ".Petroleum Refining"
SOURCE_LABEL = "StatCan RESD 25-10-0029 (refinery_fuel)"


def replace_cer_refinery_fuel(cer_df):
    """Replace CER fuel rows at the .Petroleum Refining sector node with this
    table, in a frame shaped like cer_resd_demand output (GJ). Only fuels in
    this table and years it covers are replaced."""
    fuel = load_refinery_fuel()
    fuels = set(fuel["fuel"])
    years = set(fuel["year"])
    drop = (cer_df["Node"] == SECTOR_NODE) & cer_df["Variable"].isin(fuels) & cer_df["Year"].astype(int).isin(years)
    new = pd.DataFrame({
        "Region": fuel["region"], "Node": SECTOR_NODE, "Variable": fuel["fuel"], "Unit": "GJ",
        "Source": SOURCE_LABEL, "Year": fuel["year"].astype(cer_df["Year"].dtype), "Value": fuel["tj"] * 1000.0,
    })
    new = new[new["Value"] > 0]
    out = pd.concat([cer_df[~drop], new], ignore_index=True)
    return out.sort_values(["Region", "Node", "Variable", "Year"]).reset_index(drop=True)


def _paths():
    from CIMS.data_processing.utils.controls_conversions import BASE_PATH
    return (BASE_PATH / "raw_data/stats_can/resd/25100029.csv",
            BASE_PATH / "processed_data/activity/petroleum_refining.csv",
            BASE_PATH / "processed_data/stats_can/refinery_fuel.csv")


@lru_cache(maxsize=1)
def load_refinery_fuel() -> pd.DataFrame:
    *src, dst = _paths()
    out = build(*map(str, src))
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst, index=False)
    return out


if __name__ == "__main__":
    if len(sys.argv) == 4:
        out = build(sys.argv[1], sys.argv[2])
        dst = sys.argv[3]
    else:
        *src, dst = _paths()
        out = build(*map(str, src))
    out.to_csv(dst, index=False)
    print(f"wrote {len(out)} rows -> {dst}")
