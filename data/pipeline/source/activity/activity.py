"""
Gas Production Activity Calculator
====================================
Reads the CER natural gas production CSV and outputs:
  - Total production per region (2000–2050) in 1000 m3/year
  - Percentage splits (as fractions 0–1) for:
      * Conventional  = Non Associated + Solution gas
      * Shale
      * Tight
      * Coalbed Methane

Notes:
  - Source data starts at 2005. Years 2000–2004 are back-extrapolated
    using a linear trend fitted to the 2005–2010 values for Total only.
    Splits for 2000–2004 are held constant at 2005 fractions.
    Splits for 2005 onward are computed from actual sub-type data.
  - Source unit is Million m3/day -> converted to 1000 m3/year:
        Million m3/day x 1,000 x 365 = 1000 m3/year
  - Only the "Current Measures" scenario is used.
  - Regions with no sub-type breakdown (e.g. Ontario, Yukon) default to
    conventional=1, all other splits=0.
"""

import polars as pl
from pathlib import Path

import sys
_current_file = Path(__file__)
_project_root = _current_file.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from utils.controls_conversions import load_control_config
from utils.extensions.data_extensions import trend_backwards

# Configuration
BASE_PATH = Path('C:/cims/data')
MAPPINGS_PATH               = BASE_PATH / 'mappings_conversions'
CONTROL_FILE                = MAPPINGS_PATH / 'control.py'
ENERGY_MAP_FILE             = MAPPINGS_PATH / 'energy_map.csv'
REGION_MAP_FILE             = MAPPINGS_PATH / 'region_map.csv'
SECTOR_MAP_FILE             = MAPPINGS_PATH / 'sector_map.csv'
CONVERSIONS_FILE            = MAPPINGS_PATH / 'energy_conversions.csv'
NATURAL_GAS_PRODUCTION_FILE = BASE_PATH / 'raw_data/cer/natural-gas-production-2023.csv'
CRUDE_OIL_PRODUCTION_FILE   = BASE_PATH / 'raw_data/cer/crude-oil-production-2023.csv'
GHG_FILE                    = BASE_PATH / 'raw_data/eccc/nir/EN_GHG_Econ_Can_Prov_Terr.csv'
OUTPUT_DIR                  = BASE_PATH / 'processed_data/activity'
SCENARIO                    = load_control_config()["cer_ef_reference_scenario"]

# Unit conversion: Million m3/day -> 1000 m3/year
CONVERSION = 1_000 * 365


# -- 1. Load and filter -------------------------------------------------------

raw = pl.read_csv(NATURAL_GAS_PRODUCTION_FILE)

df = (
    raw
    .filter(
        (pl.col("Scenario") == SCENARIO) &
        (pl.col("Unit")     == "Million Cubic Metres per day")
    )
    .select(["Region", "Variable", "Year", "Value"])
    .with_columns(
        (pl.col("Value") * CONVERSION).alias("Value_1000m3")
    )
    .drop("Value")
)

df = df.filter(
    ~pl.col("Region").is_in(["Canada", "Western Canadian Sedimentary Basin"])
)


# -- 2. Back-extrapolate 2000-2004 --------------------------------------------
#
#   Total is trended backwards using linear regression on 2005–2010.
#   Sub-types are backfilled with constant 2005 raw values for 2000–2004
#   (splits are recalculated in step 4, so raw values here don't matter
#   for the final percentages — only total is used for volume).

total_df = df.filter(pl.col("Variable") == "Total")
other_df = df.filter(pl.col("Variable") != "Total")

total_df = trend_backwards(
    total_df,
    group_cols     = ["Region", "Variable"],
    year_col       = "Year",
    value_col      = "Value_1000m3",
    start_year     = 2000,
    fit_start_year = 2005,
    fit_end_year   = 2010,
)

# Backfill sub-types with 2005 values for 2000–2004
splits_2005 = other_df.filter(pl.col("Year") == 2005)
early_splits = pl.concat([
    splits_2005.with_columns(pl.lit(y).alias("Year"))
    for y in range(2000, 2005)
])

total_df    = total_df.with_columns(pl.col("Year").cast(pl.Int64))
other_df    = other_df.with_columns(pl.col("Year").cast(pl.Int64))
early_splits = early_splits.with_columns(pl.col("Year").cast(pl.Int64))

other_df = pl.concat([early_splits, other_df]).sort(["Region", "Variable", "Year"])
df = pl.concat([total_df, other_df]).sort(["Region", "Variable", "Year"])

df = df.filter(
    (pl.col("Year") >= 2000) &
    (pl.col("Year") <= 2050)
)


# -- 3. Pivot: one column per Variable ----------------------------------------

pivot = (
    df
    .pivot(
        values             = "Value_1000m3",
        index              = ["Region", "Year"],
        on                 = "Variable",
        aggregate_function = "first",
    )
    .rename({
        "Total":           "total",
        "Non Associated":  "non_associated",
        "Solution":        "solution",
        "Shale":           "shale",
        "Tight":           "tight",
        "Coalbed Methane": "coalbed_methane",
    })
    .sort(["Region", "Year"])
)

for col in ["non_associated", "solution", "shale", "tight", "coalbed_methane"]:
    if col not in pivot.columns:
        pivot = pivot.with_columns(pl.lit(None).cast(pl.Float64).alias(col))


# -- 4. Compute splits --------------------------------------------------------
#
#   For 2005 onward: compute splits from actual sub-type values.
#   For 2000–2004:  use constant 2005 fractions to avoid drift.
#   Regions with no sub-type breakdown default to conventional=1, others=0.

pivot = pivot.with_columns(
    pl.when(
        pl.col("non_associated").is_not_null() | pl.col("solution").is_not_null()
    )
    .then(pl.col("non_associated").fill_null(0.0) + pl.col("solution").fill_null(0.0))
    .otherwise(None)
    .alias("conventional")
)

# Compute actual splits for all years first
pivot = pivot.with_columns(
    pl.when(pl.col("conventional").is_null() & pl.col("shale").is_null())
        .then(1.0).otherwise(pl.col("conventional") / pl.col("total"))
        .alias("pct_conventional"),
    pl.when(pl.col("shale").is_null() & pl.col("conventional").is_null())
        .then(0.0).otherwise(pl.col("shale") / pl.col("total"))
        .alias("pct_shale"),
    pl.when(pl.col("tight").is_null() & pl.col("conventional").is_null())
        .then(0.0).otherwise(pl.col("tight") / pl.col("total"))
        .alias("pct_tight"),
    pl.when(pl.col("coalbed_methane").is_null() & pl.col("conventional").is_null())
        .then(0.0).otherwise(pl.col("coalbed_methane") / pl.col("total"))
        .alias("pct_coalbed_methane"),
)

# Extract 2005 fractions per region
splits_2005_pct = (
    pivot
    .filter(pl.col("Year") == 2005)
    .select([
        "Region",
        pl.col("pct_conventional").alias("pct_conventional_2005"),
        pl.col("pct_shale").alias("pct_shale_2005"),
        pl.col("pct_tight").alias("pct_tight_2005"),
        pl.col("pct_coalbed_methane").alias("pct_coalbed_methane_2005"),
    ])
)

# Join 2005 fractions and override pre-2005 years
pivot = pivot.join(splits_2005_pct, on="Region", how="left")

pivot = pivot.with_columns(
    pl.when(pl.col("Year") < 2005)
        .then(pl.col("pct_conventional_2005")).otherwise(pl.col("pct_conventional"))
        .alias("pct_conventional"),
    pl.when(pl.col("Year") < 2005)
        .then(pl.col("pct_shale_2005")).otherwise(pl.col("pct_shale"))
        .alias("pct_shale"),
    pl.when(pl.col("Year") < 2005)
        .then(pl.col("pct_tight_2005")).otherwise(pl.col("pct_tight"))
        .alias("pct_tight"),
    pl.when(pl.col("Year") < 2005)
        .then(pl.col("pct_coalbed_methane_2005")).otherwise(pl.col("pct_coalbed_methane"))
        .alias("pct_coalbed_methane"),
)


# -- 5. Build final output ----------------------------------------------------

output = (
    pivot
    .select([
        "Region",
        "Year",
        "total",
        "pct_conventional",
        "pct_shale",
        "pct_tight",
        "pct_coalbed_methane",
    ])
    .unpivot(
        index=["Region", "Year"],
        on=["total", "pct_conventional", "pct_shale", "pct_tight", "pct_coalbed_methane"],
        variable_name="Variable",
        value_name="Value",
    )
    .with_columns(
        pl.col("Variable").replace({
            "total":               "Total",
            "pct_conventional":    "Conventional",
            "pct_shale":           "Shale",
            "pct_tight":           "Tight",
            "pct_coalbed_methane": "Coalbed Methane",
        }).alias("Variable"),
        pl.col("Variable").replace({
            "total":               "1000m3",
            "pct_conventional":    "% of 1000m3",
            "pct_shale":           "% of 1000m3",
            "pct_tight":           "% of 1000m3",
            "pct_coalbed_methane": "% of 1000m3",
        }).alias("Unit"),
    )
    .select(["Region", "Variable", "Unit", "Year", "Value"])
    .sort(["Region", "Variable", "Year"])
)


# -- 6. Save ------------------------------------------------------------------

print(f"\n✅ Gas production complete")
print(f"   Total rows:          {len(output):,}")
print(f"   Regions processed:   {output['Region'].n_unique()}")
print(f"   Years covered:       {output['Year'].min()} – {output['Year'].max()}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
output_path = OUTPUT_DIR / 'gas_production.csv'
output.write_csv(output_path)
print(f"   Saved to:            {output_path}")


# =============================================================================
# OIL PRODUCTION
# =============================================================================
# Reads the CER crude oil production CSV and outputs:
#   - Total oil production per region (2005–2050) in m3/year
#     (Total minus Condensate and C5+)
#   - Level 1 splits (% of m3/year): Bitumen, Light Medium, Heavy
#   - Level 2 splits (% of Bitumen): Upgrading, In-Situ, Mining
#
# Variable mapping:
#   (Upgraded Bitumen) -> Bitumen.Upgrading
#   Conventional Light -> Light Medium
#   Conventional Heavy -> Heavy
#   In Situ Bitumen    -> Bitumen.In-Situ
#   Mined Bitumen      -> Bitumen.Mining
#
# Unit conversion: Thousand m3/day -> m3/year
#   Thousand m3/day x 1,000 x 365 = m3/year
# =============================================================================

OIL_CONVERSION = 1_000 * 365   # Thousand m3/day -> m3/year

# -- O1. Load and filter ------------------------------------------------------

raw_oil = pl.read_csv(CRUDE_OIL_PRODUCTION_FILE)

oil_df = (
    raw_oil
    .filter(
        (pl.col("Scenario") == SCENARIO) &
        (pl.col("Unit")     == "Thousand Cubic Metres per day") &
        (~pl.col("Region").is_in(["Canada"]))
    )
    .select(["Region", "Variable", "Year", "Value"])
    .with_columns(
        (pl.col("Value") * OIL_CONVERSION).alias("Value_m3")
    )
    .drop("Value")
)

# Keep Condensate and C5+ only to subtract from Total; drop all other excludes
oil_df = oil_df.filter(
    ~pl.col("Variable").is_in(["Field Condensate", "C5+", "Total"])
)

# -- O2. Back-extrapolate 2000-2004 ----------------------------------------------
#
#   Total: loaded from CSV, condensate/C5+ subtracted, then trended backwards.
#   Sub-types: backfilled with constant 2005 raw values for 2000-2004.
#   Total is kept separate from sub-types and joined onto pivot in O3
#   so the trended values are not overwritten.

# Load Total and condensate/C5+ from raw CSV
oil_total_raw = (
    raw_oil
    .filter(
        (pl.col("Scenario") == SCENARIO) &
        (pl.col("Unit")     == "Thousand Cubic Metres per day") &
        (pl.col("Variable").is_in(["Total", "Field Condensate", "C5+"])) &
        (~pl.col("Region").is_in(["Canada"]))
    )
    .select(["Region", "Variable", "Year", "Value"])
    .with_columns((pl.col("Value") * OIL_CONVERSION).alias("Value_m3"))
    .drop("Value")
    .pivot(
        values             = "Value_m3",
        index              = ["Region", "Year"],
        on                 = "Variable",
        aggregate_function = "first",
    )
    .with_columns(
        # Total minus condensate and C5+
        (
            pl.col("Total") -
            pl.col("Field Condensate").fill_null(0.0) -
            pl.col("C5+").fill_null(0.0)
        ).alias("Value_m3")
    )
    .select(["Region", "Year", "Value_m3"])
    .with_columns(pl.lit("Total").alias("Variable"))
)

oil_total_df = trend_backwards(
    oil_total_raw,
    group_cols     = ["Region", "Variable"],
    year_col       = "Year",
    value_col      = "Value_m3",
    start_year     = 2000,
    fit_start_year = 2005,
    fit_end_year   = 2010,
)

# Extract as Region+Year->total lookup
oil_total_lookup = (
    oil_total_df
    .with_columns(pl.col("Year").cast(pl.Int64))
    .filter((pl.col("Year") >= 2000) & (pl.col("Year") <= 2050))
    .select(["Region", "Year", pl.col("Value_m3").alias("total")])
)

# Backfill sub-types with constant 2005 raw values for 2000-2004
oil_other_df = oil_df.with_columns(pl.col("Year").cast(pl.Int64))
oil_splits_2005 = oil_other_df.filter(pl.col("Year") == 2005)
oil_early_splits = pl.concat([
    oil_splits_2005.with_columns(pl.lit(y).cast(pl.Int64).alias("Year"))
    for y in range(2000, 2005)
])

oil_other_df = pl.concat([oil_early_splits, oil_other_df]).sort(["Region", "Variable", "Year"])
oil_other_df = oil_other_df.filter(
    (pl.col("Year") >= 2000) &
    (pl.col("Year") <= 2050)
)

# -- O3. Pivot sub-types, then join trended total -----------------------------

oil_pivot = (
    oil_other_df
    .pivot(
        values             = "Value_m3",
        index              = ["Region", "Year"],
        on                 = "Variable",
        aggregate_function = "first",
    )
    .rename({
        "(Upgraded Bitumen)": "upgrading",
        "Conventional Light": "light_medium",
        "Conventional Heavy": "heavy",
        "In Situ Bitumen":    "in_situ",
        "Mined Bitumen":      "mining",
    })
    .sort(["Region", "Year"])
)

for col in ["upgrading", "light_medium", "heavy", "in_situ", "mining"]:
    if col not in oil_pivot.columns:
        oil_pivot = oil_pivot.with_columns(pl.lit(None).cast(pl.Float64).alias(col))

# Join the trended total (condensate/C5+ already subtracted)
oil_pivot = oil_pivot.join(oil_total_lookup, on=["Region", "Year"], how="left")

# -- O4. Compute Level 1 splits: Bitumen, Light Medium, Heavy -----------------
#
#   Bitumen = upgrading + in_situ + mining
#   Splits are fractions out of total (0–1).
#   Regions with no sub-type data default to light_medium=1, others=0.
#   For 2000–2004, constant 2005 fractions are used.

oil_pivot = oil_pivot.with_columns(
    pl.when(
        pl.col("upgrading").is_not_null() |
        pl.col("in_situ").is_not_null()   |
        pl.col("mining").is_not_null()
    )
    .then(
        pl.col("upgrading").fill_null(0.0) +
        pl.col("in_situ").fill_null(0.0)   +
        pl.col("mining").fill_null(0.0)
    )
    .otherwise(None)
    .alias("bitumen")
)

oil_pivot = oil_pivot.with_columns(
    pl.when(pl.col("bitumen").is_null() & pl.col("light_medium").is_null() & pl.col("heavy").is_null())
        .then(0.0).otherwise(pl.col("bitumen").fill_null(0.0) / pl.col("total"))
        .alias("pct_bitumen"),
    pl.when(pl.col("bitumen").is_null() & pl.col("light_medium").is_null() & pl.col("heavy").is_null())
        .then(1.0).otherwise(pl.col("light_medium").fill_null(0.0) / pl.col("total"))
        .alias("pct_light_medium"),
    pl.when(pl.col("bitumen").is_null() & pl.col("light_medium").is_null() & pl.col("heavy").is_null())
        .then(0.0).otherwise(pl.col("heavy").fill_null(0.0) / pl.col("total"))
        .alias("pct_heavy"),
)

# Extract 2005 fractions and hold constant for 2000-2004
oil_splits_2005_pct = (
    oil_pivot
    .filter(pl.col("Year") == 2005)
    .select([
        "Region",
        pl.col("pct_bitumen").alias("pct_bitumen_2005"),
        pl.col("pct_light_medium").alias("pct_light_medium_2005"),
        pl.col("pct_heavy").alias("pct_heavy_2005"),
    ])
)

oil_pivot = oil_pivot.join(oil_splits_2005_pct, on="Region", how="left")

oil_pivot = oil_pivot.with_columns(
    pl.when(pl.col("Year") < 2005)
        .then(pl.col("pct_bitumen_2005")).otherwise(pl.col("pct_bitumen"))
        .alias("pct_bitumen"),
    pl.when(pl.col("Year") < 2005)
        .then(pl.col("pct_light_medium_2005")).otherwise(pl.col("pct_light_medium"))
        .alias("pct_light_medium"),
    pl.when(pl.col("Year") < 2005)
        .then(pl.col("pct_heavy_2005")).otherwise(pl.col("pct_heavy"))
        .alias("pct_heavy"),
)


# -- O5. Compute Level 2 splits: Upgrading, In-Situ, Mining (% of Bitumen) ---

oil_pivot = oil_pivot.with_columns(
    pl.when(pl.col("bitumen").is_null() | (pl.col("bitumen") == 0.0))
        .then(None).otherwise(pl.col("upgrading").fill_null(0.0) / pl.col("bitumen"))
        .alias("pct_upgrading"),
    pl.when(pl.col("bitumen").is_null() | (pl.col("bitumen") == 0.0))
        .then(None).otherwise(pl.col("in_situ").fill_null(0.0) / pl.col("bitumen"))
        .alias("pct_in_situ"),
    pl.when(pl.col("bitumen").is_null() | (pl.col("bitumen") == 0.0))
        .then(None).otherwise(pl.col("mining").fill_null(0.0) / pl.col("bitumen"))
        .alias("pct_mining"),
)

# Extract 2005 level-2 fractions and hold constant for 2000-2004
oil_splits2_2005_pct = (
    oil_pivot
    .filter(pl.col("Year") == 2005)
    .select([
        "Region",
        pl.col("pct_upgrading").alias("pct_upgrading_2005"),
        pl.col("pct_in_situ").alias("pct_in_situ_2005"),
        pl.col("pct_mining").alias("pct_mining_2005"),
    ])
)

oil_pivot = oil_pivot.join(oil_splits2_2005_pct, on="Region", how="left")

oil_pivot = oil_pivot.with_columns(
    pl.when(pl.col("Year") < 2005)
        .then(pl.col("pct_upgrading_2005")).otherwise(pl.col("pct_upgrading"))
        .alias("pct_upgrading"),
    pl.when(pl.col("Year") < 2005)
        .then(pl.col("pct_in_situ_2005")).otherwise(pl.col("pct_in_situ"))
        .alias("pct_in_situ"),
    pl.when(pl.col("Year") < 2005)
        .then(pl.col("pct_mining_2005")).otherwise(pl.col("pct_mining"))
        .alias("pct_mining"),
)


# -- O6. Build oil output -----------------------------------------------------

oil_output = (
    oil_pivot
    .select([
        "Region", "Year",
        "total",
        "pct_bitumen", "pct_light_medium", "pct_heavy",
        "pct_upgrading", "pct_in_situ", "pct_mining",
    ])
    .unpivot(
        index=["Region", "Year"],
        on=[
            "total",
            "pct_bitumen", "pct_light_medium", "pct_heavy",
            "pct_upgrading", "pct_in_situ", "pct_mining",
        ],
        variable_name="Variable",
        value_name="Value",
    )
    .with_columns(
        pl.col("Variable").replace({
            "total":            "Total",
            "pct_bitumen":      "Bitumen",
            "pct_light_medium": "Light Medium",
            "pct_heavy":        "Heavy",
            "pct_upgrading":    "Bitumen.Upgrading",
            "pct_in_situ":      "Bitumen.In-Situ",
            "pct_mining":       "Bitumen.Mining",
        }).alias("Variable"),
        pl.col("Variable").replace({
            "total":            "m3",
            "pct_bitumen":      "% of m3",
            "pct_light_medium": "% of m3",
            "pct_heavy":        "% of m3",
            "pct_upgrading":    "% of Bitumen",
            "pct_in_situ":      "% of Bitumen",
            "pct_mining":       "% of Bitumen",
        }).alias("Unit"),
    )
    .select(["Region", "Variable", "Unit", "Year", "Value"])
    .sort(["Region", "Variable", "Year"])
)

print(f"\n✅ Oil production complete")
print(f"   Total rows:          {len(oil_output):,}")
print(f"   Regions processed:   {oil_output['Region'].n_unique()}")
print(f"   Years covered:       {oil_output['Year'].min()} – {oil_output['Year'].max()}")

output_path = OUTPUT_DIR / 'oil_production.csv'
oil_output.write_csv(output_path)
print(f"   Saved to:            {output_path}")


# =============================================================================
# AGRICULTURE & WASTE EMISSIONS
# =============================================================================
# Reads the Environment Canada GHG by economic sector CSV and outputs:
#
# AGRICULTURE (per individual province/territory, excl. Canada totals):
#   - total_tco2e      : Total Agriculture emissions (kt CO2eq -> tCO2e)
#   - pct_heat         : Agriculture.Heat share
#                        = On Farm Fuel Use / Agriculture Total
#   - pct_process_soils: Agriculture.Process.Soils share
#                        = Crop Production / Agriculture Total
#   - pct_process_enteric_manure: Agriculture.Process.Enteric Fermentation
#                        and Manure Management share
#                        = Animal Production / Agriculture Total
#
# WASTE (per individual province/territory, excl. Canada totals):
#   - total_tco2e      : Total Waste emissions (kt CO2eq -> tCO2e)
#
# Unit conversion:  kt CO2eq  ->  tCO2e  =  value * 1,000
#
# Data availability: 1990–2023. Years 2000–2023 are used directly.
#   Future extension (2024–2050) to be added later via assumptions config.
# =============================================================================

# Regions to exclude (national/territorial aggregates)
_EXCLUDE_REGIONS = {"Canada", "Northwest Territories and Nunavut"}

KT_TO_T = 1_000   # kt CO2eq -> tCO2e

DATA_START = 2000
DATA_END   = 2023

# -- A1. Load GHG file --------------------------------------------------------

raw_ghg = pl.read_csv(
    GHG_FILE,
    columns=["Year", "Region", "Source", "Sector", "CO2eq", "Unit"],
    null_values=["x"],
    schema_overrides={"CO2eq": pl.Float64},
)

# Keep only Agriculture and Waste rows, individual regions, years 2000+
ghg = (
    raw_ghg
    .filter(
        pl.col("Source").is_in(["Agriculture", "Waste"]) &
        (~pl.col("Region").is_in(list(_EXCLUDE_REGIONS))) &
        (pl.col("Year") >= DATA_START)
    )
    .with_columns([
        pl.col("Sector").fill_null("").alias("Sector"),
    ])
    .select(["Year", "Region", "Source", "Sector", "CO2eq"])
)

# Interpolate any suppressed (null) CO2eq values using the surrounding years
# within each Region+Source+Sector group, then forward-fill any trailing nulls,
# then convert kt -> tCO2e.
ghg = (
    ghg
    .sort(["Region", "Source", "Sector", "Year"])
    .with_columns(
        pl.col("CO2eq")
          .interpolate(method="linear")
          .over(["Region", "Source", "Sector"])
          .alias("CO2eq")
    )
    .with_columns(
        pl.col("CO2eq")
          .forward_fill()
          .over(["Region", "Source", "Sector"])
          .alias("CO2eq")
    )
    .with_columns(
        (pl.col("CO2eq") * KT_TO_T).alias("tco2e")
    )
    .select(["Year", "Region", "Source", "Sector", "tco2e"])
)


# -- A2. Build Agriculture totals and sector sub-totals -----------------------
#
#   Sector == ""  ->  Agriculture total for that region/year
#   Sector == "On Farm Fuel Use"  ->  Heat sub-total
#   Sector == "Crop Production"   ->  Process.Soils sub-total
#   Sector == "Animal Production" ->  Process.Enteric Fermentation & Manure

agri_raw = ghg.filter(pl.col("Source") == "Agriculture")

# Total Agriculture per region/year
agri_total = (
    agri_raw
    .filter(pl.col("Sector") == "")
    .select(["Region", "Year", pl.col("tco2e").alias("agri_total")])
)

# On Farm Fuel Use -> Heat
agri_heat = (
    agri_raw
    .filter(pl.col("Sector") == "On Farm Fuel Use")
    .select(["Region", "Year", pl.col("tco2e").alias("agri_heat")])
)

# Crop Production -> Process.Soils
agri_soils = (
    agri_raw
    .filter(pl.col("Sector") == "Crop Production")
    .select(["Region", "Year", pl.col("tco2e").alias("agri_soils")])
)

# Animal Production -> Process.Enteric + Manure
agri_animal = (
    agri_raw
    .filter(pl.col("Sector") == "Animal Production")
    .select(["Region", "Year", pl.col("tco2e").alias("agri_animal")])
)

# Join all Agriculture sub-tables
agri_pivot = (
    agri_total
    .join(agri_heat,   on=["Region", "Year"], how="left")
    .join(agri_soils,  on=["Region", "Year"], how="left")
    .join(agri_animal, on=["Region", "Year"], how="left")
    .sort(["Region", "Year"])
)

# Compute splits (fraction 0-1); guard against zero total
agri_pivot = agri_pivot.with_columns([
    pl.when(pl.col("agri_total") != 0)
        .then(pl.col("agri_heat").fill_null(0.0)   / pl.col("agri_total"))
        .otherwise(0.0).alias("pct_heat"),
    pl.when(pl.col("agri_total") != 0)
        .then(pl.col("agri_soils").fill_null(0.0)  / pl.col("agri_total"))
        .otherwise(0.0).alias("pct_soils"),
    pl.when(pl.col("agri_total") != 0)
        .then(pl.col("agri_animal").fill_null(0.0) / pl.col("agri_total"))
        .otherwise(0.0).alias("pct_animal"),
])


# -- A3. Build Waste totals ---------------------------------------------------

waste_total = (
    ghg
    .filter((pl.col("Source") == "Waste") & (pl.col("Sector") == ""))
    .select(["Region", "Year", pl.col("tco2e").alias("waste_total")])
    .sort(["Region", "Year"])
)


# -- A4. Use actual data only (2000-2023) -------------------------------------

agri_all  = agri_pivot.select(["Region", "Year", "agri_total", "pct_heat", "pct_soils", "pct_animal"])
waste_all = waste_total


# -- A5. Build Agriculture output ---------------------------------------------

agri_output = (
    agri_all
    .unpivot(
        index=["Region", "Year"],
        on=["agri_total", "pct_heat", "pct_soils", "pct_animal"],
        variable_name="Variable",
        value_name="Value",
    )
    .with_columns(
        pl.col("Variable").replace({
            "agri_total":  "Total",
            "pct_heat":    "Agriculture.Heat",
            "pct_soils":   "Agriculture.Process.Soils",
            "pct_animal":  "Agriculture.Process.Enteric Fermentation and Manure Management",
        }).alias("Variable"),
        pl.col("Variable").replace({
            "agri_total":  "tCO2e",
            "pct_heat":    "% of tCO2e",
            "pct_soils":   "% of tCO2e",
            "pct_animal":  "% of tCO2e",
        }).alias("Unit"),
    )
    .select(["Region", "Variable", "Unit", "Year", "Value"])
    .sort(["Region", "Variable", "Year"])
)


# -- A6. Build Waste output ---------------------------------------------------

waste_output = (
    waste_all
    .unpivot(
        index=["Region", "Year"],
        on=["waste_total"],
        variable_name="Variable",
        value_name="Value",
    )
    .with_columns(
        pl.lit("Total").alias("Variable"),
        pl.lit("tCO2e").alias("Unit"),
    )
    .select(["Region", "Variable", "Unit", "Year", "Value"])
    .sort(["Region", "Variable", "Year"])
)


# -- A7. Build Construction and Forestry outputs ------------------------------
#
#   Both sit under Source == "Light Manufacturing, Construction and Forest Resources".
#   Construction  -> Sector == "Construction"
#   Forestry      -> Sector == "Forest Resources"
#
#   Note: NWT and Nunavut Construction are suppressed (x) for 2005–2023.
#   Interpolation fills these from the 2000–2004 values forward.

lmcf_raw = (
    pl.read_csv(
        GHG_FILE,
        columns=["Year", "Region", "Source", "Sector", "CO2eq", "Unit"],
        null_values=["x"],
        schema_overrides={"CO2eq": pl.Float64},
    )
    .filter(
        (pl.col("Source") == "Light Manufacturing, Construction and Forest Resources") &
        (pl.col("Sector").is_in(["Construction", "Forest Resources"])) &
        (~pl.col("Region").is_in(list(_EXCLUDE_REGIONS))) &
        (pl.col("Year") >= DATA_START)
    )
    .sort(["Region", "Sector", "Year"])
    .with_columns(
        pl.col("CO2eq")
          .interpolate(method="linear")
          .over(["Region", "Sector"])
          .alias("CO2eq")
    )
    .with_columns(
        pl.col("CO2eq")
          .forward_fill()
          .over(["Region", "Sector"])
          .alias("CO2eq")
    )
    .with_columns(
        (pl.col("CO2eq") * KT_TO_T).alias("tco2e")
    )
    .select(["Year", "Region", "Sector", "tco2e"])
)

def _build_simple_output(df: pl.DataFrame, sector: str) -> pl.DataFrame:
    """Extract a single sector as a Total tCO2e output table."""
    return (
        df
        .filter(pl.col("Sector") == sector)
        .select(["Region", "Year", pl.col("tco2e").alias("total")])
        .unpivot(
            index=["Region", "Year"],
            on=["total"],
            variable_name="Variable",
            value_name="Value",
        )
        .with_columns(
            pl.lit("Total").alias("Variable"),
            pl.lit("tCO2e").alias("Unit"),
        )
        .select(["Region", "Variable", "Unit", "Year", "Value"])
        .sort(["Region", "Variable", "Year"])
    )

construction_output = _build_simple_output(lmcf_raw, "Construction")
forestry_output     = _build_simple_output(lmcf_raw, "Forest Resources")


# -- A8. Save -----------------------------------------------------------------

print(f"\n✅ Agriculture emissions complete")
print(f"   Total rows:          {len(agri_output):,}")
print(f"   Regions processed:   {agri_output['Region'].n_unique()}")
print(f"   Years covered:       {agri_output['Year'].min()} – {agri_output['Year'].max()}")

agri_path = OUTPUT_DIR / 'agriculture_emissions.csv'
agri_output.write_csv(agri_path)
print(f"   Saved to:            {agri_path}")

print(f"\n✅ Waste emissions complete")
print(f"   Total rows:          {len(waste_output):,}")
print(f"   Regions processed:   {waste_output['Region'].n_unique()}")
print(f"   Years covered:       {waste_output['Year'].min()} – {waste_output['Year'].max()}")

waste_path = OUTPUT_DIR / 'waste_emissions.csv'
waste_output.write_csv(waste_path)
print(f"   Saved to:            {waste_path}")

print(f"\n✅ Construction emissions complete")
print(f"   Total rows:          {len(construction_output):,}")
print(f"   Regions processed:   {construction_output['Region'].n_unique()}")
print(f"   Years covered:       {construction_output['Year'].min()} – {construction_output['Year'].max()}")

construction_path = OUTPUT_DIR / 'construction_emissions.csv'
construction_output.write_csv(construction_path)
print(f"   Saved to:            {construction_path}")

print(f"\n✅ Forestry emissions complete")
print(f"   Total rows:          {len(forestry_output):,}")
print(f"   Regions processed:   {forestry_output['Region'].n_unique()}")
print(f"   Years covered:       {forestry_output['Year'].min()} – {forestry_output['Year'].max()}")

forestry_path = OUTPUT_DIR / 'forestry_emissions.csv'
forestry_output.write_csv(forestry_path)
print(f"   Saved to:            {forestry_path}")
