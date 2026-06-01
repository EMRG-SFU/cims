"""
Statistics Canada population and GDP by province and territory.

Inputs:
    raw_data/stats_can/population/1710000901.csv  (table 17-10-0009-01)
    raw_data/stats_can/gdp/36100222.csv           (table 36-10-0222-01)
    raw_data/cer/macro-indicators-2026.csv

Output:
    processed_data/stats_can/pop_gdp.csv
    Columns: region, variable, source, unit, year, value

Historical Stats Can data is extended to 2050 using CER Canada-wide
year-over-year growth rates applied uniformly to each province/territory.
From 2051 to 2100 the 2040-2050 CAGR is carried forward with a 0.1
dampener. GDP is converted from chained 2017 CAD to CIMS currency year
CAD millions using convert_currency() with the CER GDP deflator.
"""

import pandas as pd
import polars as pl
from pathlib import Path
import sys

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from utils.controls_conversions import (
    BASE_PATH, DATA_START, PROJECTION_END, LAST_DATA_YEAR,
    load_control_config, load_macro_indicators,
    load_region_mapping, convert_currency,
)
from utils.data_extensions import compute_cagr, extend_cagr_periods
from utils.output_builder import log_output

# --- Paths ---
POP_FILE    = BASE_PATH / "raw_data/stats_can/population/1710000901.csv"
GDP_FILE    = BASE_PATH / "raw_data/stats_can/gdp/36100222.csv"
MACRO_FILE  = BASE_PATH / "raw_data/cer/macro-indicators-2026.csv"
OUTPUT_DIR  = BASE_PATH / "processed_data/stats_can"
OUTPUT_FILE = OUTPUT_DIR / "pop_gdp.csv"

CONTROLS  = load_control_config()
SCENARIO  = CONTROLS["cer_ef_reference_scenario"]
CIMS_YEAR = CONTROLS["currency_year_cims"]        # target currency year (2025)
CER_END   = LAST_DATA_YEAR["cer"]                 # 2050
POP_LAST  = LAST_DATA_YEAR["stat_can_population"]  # 2025
GDP_LAST  = LAST_DATA_YEAR["stat_can_prov_gdp"]    # 2024

CER_GDP_COL = "Real Gross Domestic Product ($2017 millions)"
CER_POP_COL = "Population (thousands)"

# Build Stats Can province name -> CIMS 2-letter code from the shared region map.
# Excludes the Canada national row (CIMS="CAN").
_region_df = load_region_mapping()
REGION_MAP: dict[str, str] = {
    str(sc).strip(): str(cims).strip()
    for sc, cims in zip(_region_df["Stats Can"], _region_df["CIMS"])
    if pd.notna(sc) and pd.notna(cims) and str(cims).strip() != "CAN"
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cer_series(macro_df: pl.DataFrame, col: str) -> pd.Series:
    """Extract one CER macro variable as a year-indexed pandas Series."""
    rows = macro_df.select(["Year", col]).sort("Year")
    return pd.Series(
        rows[col].cast(pl.Float64).to_list(),
        index=[int(y) for y in rows["Year"].to_list()],
        dtype=float,
    )


def _compute_avg_shares(
    hist_df: pd.DataFrame,
    last_hist: int,
    n_years: int = 5,
) -> pd.Series:
    """
    Compute each region's average share of the all-province total over the
    last n_years of historical data. Returns a Series indexed by region code.
    """
    start = last_hist - n_years + 1
    window = hist_df[
        (hist_df["year"] >= start) & (hist_df["year"] <= last_hist)
    ].copy()
    annual_total = window.groupby("year")["value"].sum().rename("total")
    window = window.merge(annual_total, on="year")
    window["share"] = window["value"] / window["total"]
    return window.groupby("region")["share"].mean()


def _extend_to_projection_end(extended: pd.Series, cer_end: int) -> pd.Series:
    """Apply CAGR(2040-2050) * 0.1 dampener from cer_end+1 to PROJECTION_END."""
    raw_cagr = compute_cagr(extended, 2040, 2050)
    tail = extend_cagr_periods(
        base_val=float(extended[cer_end]),
        raw_cagr=raw_cagr,
        periods=[(cer_end + 1, PROJECTION_END, 0.1)],
    )
    return pd.concat([extended, tail]).sort_index()


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_population() -> pd.DataFrame:
    """
    Load quarterly Stats Can population, average to annual, filter to
    DATA_START-POP_LAST. Returns long-format: year, region, value (persons).
    """
    pop = pd.read_csv(POP_FILE, usecols=["REF_DATE", "GEO", "VALUE"])
    pop.columns = ["date", "geo", "value"]
    pop["year"] = pd.to_datetime(pop["date"], format="%Y-%m").dt.year
    pop["value"] = pd.to_numeric(pop["value"], errors="coerce")
    pop = pop[pop["geo"].isin(REGION_MAP) & (pop["year"] <= POP_LAST)]

    annual = pop.groupby(["year", "geo"])["value"].mean().reset_index()
    annual["region"] = annual["geo"].map(REGION_MAP)
    return annual[["year", "region", "value"]].dropna()


def load_gdp() -> pd.DataFrame:
    """
    Load Stats Can provincial GDP at market prices, chained 2017 dollars,
    filtered to DATA_START-GDP_LAST. Returns long-format: year, region, value
    (millions of chained 2017 CAD).
    """
    df = pd.read_csv(
        GDP_FILE,
        usecols=["REF_DATE", "GEO", "Prices", "Estimates", "VALUE"],
    )
    df = df[
        (df["Prices"] == "Chained (2017) dollars") &
        (df["Estimates"] == "Gross domestic product at market prices") &
        (df["GEO"].isin(REGION_MAP))
    ].copy()
    df["year"] = pd.to_numeric(df["REF_DATE"], errors="coerce").astype("Int64")
    df["value"] = pd.to_numeric(df["VALUE"], errors="coerce")
    df = df[(df["year"] >= DATA_START) & (df["year"] <= GDP_LAST)]
    df["region"] = df["GEO"].map(REGION_MAP)
    return df[["year", "region", "value"]].dropna()


# ---------------------------------------------------------------------------
# Core projector
# ---------------------------------------------------------------------------

def process_variable(
    hist_df: pd.DataFrame,
    cer: pd.Series,
    last_hist: int,
    variable: str,
    unit: str,
    source: str,
    macro_df: pl.DataFrame = None,
    currency_from_year: int = None,
    cer_unit_scale: float = 1.0,
) -> pd.DataFrame:
    """
    Build a full DATA_START-PROJECTION_END long-format DataFrame for all regions.

    Projection to CER_END: 5-year average regional shares applied to the CER
    national total (cer_unit_scale converts CER units to match Stats Can units,
    e.g. 1000 for population where CER is in thousands).
    Post-CER: CAGR(2040-2050) * 0.1 dampener per region.

    If macro_df and currency_from_year are supplied, converts from constant
    currency_from_year CAD to CIMS_YEAR CAD via convert_currency().
    """
    shares = _compute_avg_shares(hist_df, last_hist)

    records = []
    for region in sorted(hist_df["region"].unique()):
        reg = (
            hist_df[hist_df["region"] == region]
            .set_index("year")["value"]
            .astype(float)
        )
        reg = reg[reg.index >= DATA_START]

        share = float(shares.get(region, 0.0))
        extended = reg.copy()
        for yr in range(last_hist + 1, CER_END + 1):
            if yr in cer.index:
                extended[yr] = share * float(cer[yr]) * cer_unit_scale
        extended = extended.sort_index()

        full = _extend_to_projection_end(extended, CER_END)

        if macro_df is not None and currency_from_year is not None:
            full = convert_currency(
                full,
                from_year=currency_from_year,
                to_year=CIMS_YEAR,
                from_currency="CAD",
                to_currency="CAD",
                macro_df=macro_df,
                constant_dollars=True,
            )

        for yr, val in full.items():
            if pd.notna(val):
                records.append({
                    "region":   region,
                    "variable": variable,
                    "source":   source,
                    "unit":     unit,
                    "year":     int(yr),
                    "value":    round(float(val), 4),
                })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Entry point — callable directly or imported by other scripts
# ---------------------------------------------------------------------------

def main() -> pd.DataFrame:
    print("Loading CER macro indicators...")
    macro = load_macro_indicators(str(MACRO_FILE), SCENARIO)
    cer_pop = _cer_series(macro, CER_POP_COL)  # thousands — growth rates only
    cer_gdp = _cer_series(macro, CER_GDP_COL)  # $2017M  — growth rates only

    print("Loading Stats Can population (17-10-0009-01)...")
    pop_hist = load_population()
    print(f"  {len(pop_hist):,} region-year rows  ({DATA_START}-{POP_LAST})")

    print("Loading Stats Can GDP (36-10-0222-01)...")
    gdp_hist = load_gdp()
    print(f"  {len(gdp_hist):,} region-year rows  ({DATA_START}-{GDP_LAST})")

    print("Projecting population to 2100...")
    pop_out = process_variable(
        pop_hist, cer_pop, POP_LAST,
        variable="Population",
        unit="persons",
        source="StatsCan/CER",
        cer_unit_scale=1000.0,  # CER population is in thousands; output in persons
    )

    print(f"Projecting GDP to 2100 and converting to {CIMS_YEAR}$ millions...")
    gdp_out = process_variable(
        gdp_hist, cer_gdp, GDP_LAST,
        variable="GDP",
        unit=f"{CIMS_YEAR}$ millions",
        source="Statistics Canada / CER",
        macro_df=macro,
        currency_from_year=2017,  # Stats Can chained 2017 dollars
    )

    result = pd.concat([pop_out, gdp_out], ignore_index=True)
    result = result.sort_values(["variable", "region", "year"]).reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    log_output(
        pl.from_pandas(result), OUTPUT_FILE,
        region_col="region", variable_col="variable", year_col="year",
    )

    return result


if __name__ == "__main__":
    main()
