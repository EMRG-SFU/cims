"""
Steam fuel shares by CIMS sector, region and year from the filled RESD table.

Input
-----
raw_data/stats_can/resd/RESD_filled_2000-2024.xlsx
    StatCan RESD (25-10-0029-01) with suppressed cells filled by resd_fill.py,
    in the CEEDC workbook layout ('Data' and 'Estimated' sheets, GJ).

Method
------
For each CIMS sector, region and year, keeps only the fuels that plausibly go
to boilers/cogeneration and returns each fuel's share of that steam fuel.
Electricity is excluded (it comes from the MECS electric boiler share).
Fuels that never reach MIN_SHARE in any year for a region and sector are
dropped and the rest prorated to 100%. Region-sectors with no activity get
the Canada mix. Biomass is off (INCLUDE_BIOMASS) until there is a provincial
source for it.

Output: processed_data/stats_can/resd_steam_fuel_shares.csv
    Columns: region, cims_sector, year, steam_fuel, share, gj, estimated, source, sector

Pipeline use:
    from CIMS.data_processing.source.stats_can.resd_steam_fuel_shares import load_steam_fuel_shares
    shares = load_steam_fuel_shares()   # pandas, CIMS region codes (AB, BC, ...)

Standalone:
    python resd_steam_fuel_shares.py <RESD_filled_2000-2024.xlsx> <output.csv>
"""

import sys
from functools import lru_cache

import polars as pl

# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
YEARS = (2000, 2024)   # output years, inclusive
MIN_SHARE = 0.01       # drop a fuel if its share never reaches this in any year
                       # for that region and sector; prorate the rest to 100%

# CEEDC sector name -> CIMS sector
CEEDC_TO_CIMS_SECTOR = {
    "Chemicals and fertilizers": "Chemical Products",
    "Iron and steel": "Iron and Steel",
    "Aluminum and non-ferrous metal": "Metal Smelting",
    "Pulp and paper": "Pulp and Paper",
    "All other manufacturing": "Light Industrial",
}

# CEEDC fuel column -> steam fuel name (rename right-hand side to CIMS fuel names)
CEEDC_TO_STEAM_FUEL = {
    "NG": "Natural Gas",
    "HFO": "Heavy Fuel Oil",
    "MD": "Light Fuel Oil",     # middle distillates: LFO + kerosene (diesel is separate)
    "Propane": "LPG",
    "Coal": "Coal",
    "Petcoke": "Petroleum Coke",
    "Wood": "Wood",
    "SPL": "Spent Pulping Liquor",  # black liquor, burned in recovery boilers
}

# CEEDC reports these only for Canada (provincial cells are blank). They are
# split across provinces by allocate_biomass(); replace that step with
# provincial data (e.g. the CER biomass additions) when available.
BIOMASS = ["Wood", "SPL"]

# Fuels that count toward steam in each sector. Everything else is excluded:
# Elec (separate), Steam (purchased; would double count), Diesel/Gasoline
# (vehicles), Coke/COG (reductants), RFG (refinery fuel gas; zero in these
# sectors provincially), Waste (cement kilns only).
_BASE = ["NG", "HFO", "MD", "Wood"]
STEAM_FUELS_BY_SECTOR = {
    "Chemical Products": _BASE + ["Propane", "Coal", "Petcoke"],
    "Iron and Steel":    _BASE + ["Propane"],            # coal/coke/COG = reductant & furnace
    "Metal Smelting":    _BASE + ["Propane"],            # coal/coke/petcoke = reductant & anodes
    "Pulp and Paper":    _BASE + ["Propane", "Coal", "SPL"],  # petcoke = lime kilns
    "Light Industrial":  _BASE,                          # propane = forklifts; coal/petcoke = kilns
}

# Biomass is excluded until there is a provincial source for it. Set to True
# to include wood and spent pulping liquor (provinces then use the
# placeholder split in allocate_biomass()).
INCLUDE_BIOMASS = False
if not INCLUDE_BIOMASS:
    STEAM_FUELS_BY_SECTOR = {
        s: [f for f in fuels if f not in BIOMASS] for s, fuels in STEAM_FUELS_BY_SECTOR.items()
    }

ALL_FUELS = ["Coal", "COG", "Coke", "Diesel", "Elec", "Gasoline", "HFO", "MD",
             "NG", "Petcoke", "Propane", "RFG", "SPL", "Steam", "Waste", "Wood"]
KEYS = ["year", "region", "cims_sector"]


def _read_sheet(path: str, sheet: str) -> pl.DataFrame:
    df = pl.read_excel(path, sheet_name=sheet, has_header=True, infer_schema_length=0)
    df = df.rename({df.columns[0]: "year", df.columns[1]: "region",
                    df.columns[2]: "naics", df.columns[3]: "sector"})
    return df.filter(pl.col("year").is_not_null())  # drop the units row


def load(path: str) -> pl.DataFrame:
    data = _read_sheet(path, "Data")
    est = _read_sheet(path, "Estimated")
    idx = ["year", "region", "sector"]

    vals = (
        data.filter(pl.col("sector").is_in(list(CEEDC_TO_CIMS_SECTOR)))
        .unpivot(index=idx, on=ALL_FUELS + ["Total"], variable_name="fuel", value_name="gj")
        .with_columns(pl.col("year").cast(pl.Int32), pl.col("gj").cast(pl.Float64, strict=False))
    )
    flags = (
        est.filter(pl.col("sector").is_in(list(CEEDC_TO_CIMS_SECTOR)))
        .unpivot(index=idx, on=ALL_FUELS, variable_name="fuel", value_name="flag")
        .with_columns(pl.col("year").cast(pl.Int32), (pl.col("flag") == "x").fill_null(False).alias("estimated"))
        .drop("flag")
    )
    return (
        vals.join(flags, on=idx + ["fuel"], how="left")
        .with_columns(
            pl.col("sector").replace_strict(CEEDC_TO_CIMS_SECTOR).alias("cims_sector"),
            pl.col("estimated").fill_null(False),
        )
    )


def allocate_biomass(long: pl.DataFrame) -> pl.DataFrame:
    """Fill the blank provincial Wood/SPL cells by splitting the Canada value
    for each sector and year across provinces in proportion to each
    province's non-biomass energy use in that sector (CEEDC 'Total' excludes
    biomass for provinces). Allocated cells are flagged as estimated."""
    canada = long.filter((pl.col("region") == "Canada") & pl.col("fuel").is_in(BIOMASS)).select(
        "year", "sector", "fuel", pl.col("gj").fill_null(0).alias("can_gj"))
    weight = (
        long.filter((pl.col("region") != "Canada") & (pl.col("fuel") == "Total"))
        .select("year", "region", "sector", pl.col("gj").fill_null(0).alias("w"))
        .with_columns((pl.col("w") / pl.col("w").sum().over("year", "sector")).fill_nan(0).alias("w"))
    )
    alloc = (
        weight.join(canada, on=["year", "sector"])
        .select("year", "region", "sector", "fuel", (pl.col("w") * pl.col("can_gj")).alias("alloc_gj"))
    )
    return (
        long.join(alloc, on=["year", "region", "sector", "fuel"], how="left")
        .with_columns(
            pl.when(pl.col("alloc_gj").is_not_null()).then(pl.col("alloc_gj")).otherwise(pl.col("gj")).alias("gj"),
            (pl.col("estimated") | pl.col("alloc_gj").is_not_null()).alias("estimated"),
        )
        .drop("alloc_gj")
    )


def steam_fuels(long: pl.DataFrame) -> pl.DataFrame:
    """Keep each sector's steam fuels; build Canada as the sum of provinces."""
    long = allocate_biomass(long)
    keep = pl.concat([
        long.filter((pl.col("cims_sector") == s) & pl.col("fuel").is_in(f))
        for s, f in STEAM_FUELS_BY_SECTOR.items()
    ])
    prov = keep.filter(pl.col("region") != "Canada")
    # CEEDC's own Canada rows don't always match the provinces (some years have
    # refinery fuel gas / petcoke booked under other sectors), so Canada is
    # rebuilt from the provinces.
    canada = prov.group_by(["year", "sector", "cims_sector", "fuel"]).agg(
        pl.col("gj").sum(), pl.col("estimated").any()
    ).with_columns(pl.lit("Canada").alias("region"))
    out = pl.concat([prov, canada.select(prov.columns)])
    return out.with_columns(
        pl.col("fuel").replace_strict(CEEDC_TO_STEAM_FUEL).alias("steam_fuel")
    )


def shares(df: pl.DataFrame) -> pl.DataFrame:
    tot = pl.col("gj").sum().over(KEYS)
    df = df.with_columns((pl.col("gj") / tot).alias("share"), (tot > 0).alias("has_data"))

    # Region-sectors with no steam fuel at all in a year (sector not present):
    # use the Canada mix for that year so CIMS still gets a valid split.
    can = df.filter(pl.col("region") == "Canada").select(
        "year", "cims_sector", "steam_fuel", pl.col("share").alias("can_share"))
    df = df.join(can, on=["year", "cims_sector", "steam_fuel"], how="left").with_columns(
        pl.when(pl.col("has_data")).then(pl.col("share")).otherwise(pl.col("can_share")).alias("share"),
        pl.when(pl.col("has_data")).then(pl.lit("region_data")).otherwise(pl.lit("canada_no_activity")).alias("source"),
    )
    return df.select("region", "cims_sector", "year", "steam_fuel", "share", "gj", "estimated", "source", "sector")


def extend_years(df: pl.DataFrame) -> pl.DataFrame:
    """CEEDC ends in 2023; hold the last year's shares flat to YEARS[1]."""
    last = df["year"].max()
    extra = [
        df.filter(pl.col("year") == last).with_columns(
            pl.lit(y, pl.Int32).alias("year"), pl.lit(None, pl.Float64).alias("gj"),
            pl.lit(f"held_from_{last}").alias("source"))
        for y in range(last + 1, YEARS[1] + 1)
    ]
    return pl.concat([df] + extra) if extra else df


def apply_cutoff(df: pl.DataFrame, cutoff: float = MIN_SHARE) -> pl.DataFrame:
    """Drop fuels whose share never reaches `cutoff` in any year for a region
    and sector, then prorate the remaining shares back to 100%."""
    series = ["region", "cims_sector", "steam_fuel"]
    keep = df.with_columns(pl.col("share").max().over(series).alias("_max")).filter(pl.col("_max") >= cutoff)
    out = keep.with_columns((pl.col("share") / pl.col("share").sum().over(KEYS)).alias("share")).drop("_max")
    if out.filter(pl.col("share").is_nan() | pl.col("share").is_infinite()).height:
        raise ValueError(f"some rows lost all fuels after the {cutoff:.0%} cut-off")
    return out


def build(path: str) -> pl.DataFrame:
    df = shares(steam_fuels(load(path)))
    df = df.filter(pl.col("year").is_between(*YEARS))
    df = extend_years(df)
    df = apply_cutoff(df)
    return df.sort("region", "cims_sector", "year", "steam_fuel")


# --------------------------------------------------------------------------
# Pipeline entry point
# --------------------------------------------------------------------------
REGION_CODES = {
    "British Columbia": "BC", "Alberta": "AB", "Saskatchewan": "SK", "Manitoba": "MB",
    "Ontario": "ON", "Quebec": "QC", "New Brunswick": "NB", "Nova Scotia": "NS",
    "Prince Edward Island": "PE", "Newfoundland and Labrador": "NL", "Yukon": "YT",
    "Northwest Territories": "NT", "Nunavut": "NU", "Canada": "CAN",
}


def _paths():
    from CIMS.data_processing.utils.controls_conversions import BASE_PATH
    return (BASE_PATH / "raw_data/stats_can/resd/RESD_filled_2000-2024.xlsx",
            BASE_PATH / "processed_data/stats_can/resd_steam_fuel_shares.csv")


@lru_cache(maxsize=1)
def load_steam_fuel_shares():
    """Steam fuel shares as pandas, with CIMS region codes and CIMS sector nodes
    ('.Chemical Products', ...). Rebuilt from the RESD file on every new session
    and saved to processed_data for reference."""
    src, dst = _paths()
    out = build(str(src))
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(dst)
    import pandas as pd
    out = out.with_columns(
        pl.col("region").replace_strict(REGION_CODES).alias("region"),
        ("." + pl.col("cims_sector")).alias("node"))
    return pd.DataFrame(out.to_dict(as_series=False))  # no pyarrow needed


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    out = build(src)
    out.write_csv(dst)
    print(f"wrote {out.height} rows -> {dst}")
