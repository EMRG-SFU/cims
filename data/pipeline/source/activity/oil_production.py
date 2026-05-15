"""
=============================================================================
OIL PRODUCTION
=============================================================================
Reads the CER crude oil production CSV and outputs:
  - Total oil production per region (2005–2050) in m3/year
    (Total minus Condensate and C5+)
  - Level 1 splits (% of m3/year): Bitumen, Light Medium, Heavy
  - Level 2 splits (% of Bitumen): Upgrading, In-Situ, Mining

Variable mapping:
  (Upgraded Bitumen) -> Bitumen.Upgrading
  Conventional Light -> Light Medium
  Conventional Heavy -> Heavy
  In Situ Bitumen    -> Bitumen.In-Situ
  Mined Bitumen      -> Bitumen.Mining

Unit conversion: Thousand m3/day -> m3/year
  Thousand m3/day x 1,000 x 365 = m3/year
=============================================================================
"""

import polars as pl
from pathlib import Path

import sys
_current_file = Path(__file__)
_project_root = _current_file.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from utils.controls_conversions import load_control_config
from utils.data_fill import trend_backwards
from utils.data_extensions import extend_cagr_periods, compute_cagr, load_cagr_assumptions
from utils.extractors.cer import find_cer_file
from utils.controls_conversions import DATA_START, PROJECTION_END

# Configuration
BASE_PATH = Path('C:/cims/data')
CER_DIR   = BASE_PATH / 'raw_data/cer'
CRUDE_OIL_PRODUCTION_FILE   = find_cer_file(CER_DIR, 'crude-oil-production')
REGION_MAP_FILE             = BASE_PATH / 'mappings_conversions/region_map.csv'
OUTPUT_DIR                  = BASE_PATH / 'processed_data/activity'
_config                     = load_control_config()
SCENARIO                    = _config["cer_ef_reference_scenario"]
LAST_DATA_YEAR              = _config["last_data_year"]

OIL_CONVERSION = 1_000 * 365   # Thousand m3/day -> m3/year

ASSUMPTIONS_FILE = Path('C:/cims/data/raw_data/assumptions/activity_cagr_projections.csv')
CAGR_START, CAGR_END, CAGR_PERIODS = load_cagr_assumptions('Oil', ASSUMPTIONS_FILE)

# Per-region overrides — one explicit annual rate per period.
CAGR_OVERRIDES: dict[str, tuple[float, ...]] = {
}

# -- O1. Load and filter ------------------------------------------------------

raw_oil = pl.read_csv(CRUDE_OIL_PRODUCTION_FILE)

_region_map_df = pl.read_csv(REGION_MAP_FILE, encoding="utf8-lossy")
_OIL_REGION_MAP = dict(zip(
    _region_map_df["CER"].to_list(),
    _region_map_df["CIMS"].to_list(),
))

raw_oil = raw_oil.with_columns(
    pl.col("Region")
    .replace_strict(_OIL_REGION_MAP, default=None)
    .fill_null(pl.col("Region"))
    .alias("Region")
)

oil_df = (
    raw_oil
    .filter(
        (pl.col("Scenario") == SCENARIO) &
        (pl.col("Unit")     == "Thousand Cubic Metres per day") &
        (~pl.col("Region").is_in(["Canada", "CAN"]))
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
        (~pl.col("Region").is_in(["Canada", "CAN"]))
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
    .pipe(lambda df: df.with_columns(
        pl.lit(0.0).alias("Field Condensate")
        if "Field Condensate" not in df.columns else pl.col("Field Condensate")
    ))
    .pipe(lambda df: df.with_columns(
        pl.lit(0.0).alias("C5+")
        if "C5+" not in df.columns else pl.col("C5+")
    ))
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
    .filter((pl.col("Year") >= DATA_START) & (pl.col("Year") <= LAST_DATA_YEAR["cer"]))
    .select(["Region", "Year", pl.col("Value_m3").alias("total")])
)

# Backfill sub-types with constant 2005 raw values for 2000-2004
oil_other_df = oil_df.with_columns(pl.col("Year").cast(pl.Int64))
oil_splits_2005 = oil_other_df.filter(pl.col("Year") == 2005)
oil_early_splits = pl.concat([
    oil_splits_2005.with_columns(pl.lit(y).cast(pl.Int64).alias("Year"))
    for y in range(DATA_START, 2005)
])

oil_other_df = pl.concat([oil_early_splits, oil_other_df]).sort(["Region", "Variable", "Year"])
oil_other_df = oil_other_df.filter(
    (pl.col("Year") >= DATA_START) &
    (pl.col("Year") <= LAST_DATA_YEAR["cer"])
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
    # Bitumen = In Situ + Mined only. Upgraded Bitumen is a derived product
    # that sits outside the raw production sum and must not be included here.
    pl.when(
        pl.col("in_situ").is_not_null() |
        pl.col("mining").is_not_null()
    )
    .then(
        pl.col("in_situ").fill_null(0.0) +
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

oil_pivot = (
    oil_pivot
    .with_columns(
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
    .drop(["pct_bitumen_2005", "pct_light_medium_2005", "pct_heavy_2005"])
    .with_columns(
        pl.col("pct_bitumen").clip(0.0, 1.0),
        pl.col("pct_light_medium").clip(0.0, 1.0),
        pl.col("pct_heavy").clip(0.0, 1.0),
    )
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
).drop(["pct_upgrading_2005", "pct_in_situ_2005", "pct_mining_2005"])


# -- O5b. Onshore / Offshore splits (% of Light Medium) ----------------------
#
#   NL is entirely offshore; all other regions are entirely onshore.
#   These are static fractions — no source data variation to compute.

oil_pivot = oil_pivot.with_columns(
    pl.when(pl.col("Region") == "NL")
        .then(0.0).otherwise(1.0)
        .alias("pct_onshore"),
    pl.when(pl.col("Region") == "NL")
        .then(1.0).otherwise(0.0)
        .alias("pct_offshore"),
)


# -- O5c. Extend to 2100 via CAGR ---------------------------------------------
#
#   Total: CAGR computed from 2040-2050 per region,
#   floored at 0, compounding from the 2050 base. Mirrors gas production.
#   All ratio columns: held flat at 2050 values to 2100.

_oil_anchor = (
    oil_pivot
    .filter(pl.col("Year").is_in([CAGR_START, CAGR_END]))
    .select(["Region", "Year", "total"])
    .pivot(values="total", index="Region", on="Year", aggregate_function="first")
    .rename({str(CAGR_START): "total_2040", str(CAGR_END): "total_2050"})
)

# CAGR = ((total_2050 / total_2040) ^ (1/(CAGR_END - CAGR_START)) - 1)
_oil_anchor = _oil_anchor.with_columns(
    pl.when((pl.col("total_2040").is_null()) | (pl.col("total_2040") == 0))
        .then(0.0)
        .otherwise(
            ((pl.col("total_2050") / pl.col("total_2040")).pow(1.0 / (CAGR_END - CAGR_START)) - 1.0)
        )
        .alias("cagr")
)

# All ratio columns pinned at 2050 values
_oil_vals_2050 = (
    oil_pivot
    .filter(pl.col("Year") == 2050)
    .select([
        "Region", "total",
        "pct_bitumen", "pct_light_medium", "pct_heavy",
        "pct_upgrading", "pct_in_situ", "pct_mining",
        "pct_onshore", "pct_offshore",
    ])
    .rename({"total": "total_2050_val"})
)

_oil_anchor_map    = {r["Region"]: r for r in _oil_anchor.to_dicts()}
_oil_vals_2050_map = {r["Region"]: r for r in _oil_vals_2050.to_dicts()}

_oil_extension_rows = []
for _region in _oil_anchor_map:
    _anc = _oil_anchor_map[_region]
    _v   = _oil_vals_2050_map[_region]
    ext = extend_cagr_periods(
        base_val = _anc["total_2050"],
        raw_cagr = _anc["cagr"],
        periods  = CAGR_PERIODS,
        override = CAGR_OVERRIDES.get(_region),
    )
    for _yr, _total in ext.items():
        _oil_extension_rows.append({
            "Region":           _region,
            "Year":             _yr,
            "total":            _total,
            "pct_bitumen":      _v["pct_bitumen"],
            "pct_light_medium": _v["pct_light_medium"],
            "pct_heavy":        _v["pct_heavy"],
            "pct_upgrading":    _v["pct_upgrading"],
            "pct_in_situ":      _v["pct_in_situ"],
            "pct_mining":       _v["pct_mining"],
            "pct_onshore":      _v["pct_onshore"],
            "pct_offshore":     _v["pct_offshore"],
        })

_oil_extension_pl = pl.DataFrame(_oil_extension_rows).with_columns(
    pl.col("Year").cast(pl.Int64)
)

oil_pivot = pl.concat([oil_pivot, _oil_extension_pl], how="diagonal").sort(["Region", "Year"])


# -- O6. Build oil output -----------------------------------------------------
#   oil_pivot now covers 2000-2100: CER data for 2000-2050, CAGR extension for 2051-2100.

oil_output = (
    oil_pivot
    .with_columns(
        # Fill null ratios with 0: regions with no bitumen get 0 for Level 2
        # splits; regions where total hits 0 get 0 for all Level 1 splits.
        pl.col("pct_bitumen").fill_null(0.0).fill_nan(0.0),
        pl.col("pct_light_medium").fill_null(0.0).fill_nan(0.0),
        pl.col("pct_heavy").fill_null(0.0).fill_nan(0.0),
        pl.col("pct_upgrading").fill_null(0.0).fill_nan(0.0),
        pl.col("pct_in_situ").fill_null(0.0).fill_nan(0.0),
        pl.col("pct_mining").fill_null(0.0).fill_nan(0.0),
    )
    .select([
        "Region", "Year",
        "total",
        "pct_bitumen", "pct_light_medium", "pct_heavy",
        "pct_upgrading", "pct_in_situ", "pct_mining",
        "pct_onshore", "pct_offshore",
    ])
    .unpivot(
        index=["Region", "Year"],
        on=[
            "total",
            "pct_bitumen", "pct_light_medium", "pct_heavy",
            "pct_upgrading", "pct_in_situ", "pct_mining",
            "pct_onshore", "pct_offshore",
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
            "pct_onshore":      "Light Medium.Onshore",
            "pct_offshore":     "Light Medium.Offshore",
        }).alias("Variable"),
        pl.col("Variable").replace({
            "total":            "m3",
            "pct_bitumen":      "% of m3",
            "pct_light_medium": "% of m3",
            "pct_heavy":        "% of m3",
            "pct_upgrading":    "% of Bitumen",
            "pct_in_situ":      "% of Bitumen",
            "pct_mining":       "% of Bitumen",
            "pct_onshore":      "% of Light Medium",
            "pct_offshore":     "% of Light Medium",
        }).alias("Unit"),
    )
    .filter(~pl.col("Region").is_in(["Canada", "CAN"]))
    .select(["Region", "Variable", "Unit", "Year", "Value"])
    .sort(["Region", "Variable", "Year"])
)

print(f"\n✅ Oil production complete")
print(f"   Total rows:          {len(oil_output):,}")
print(f"   Regions processed:   {oil_output['Region'].n_unique()}")
print(f"   Years covered:       {oil_output['Year'].min()} – {oil_output['Year'].max()}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
output_path = OUTPUT_DIR / 'oil_production.csv'
oil_output.write_csv(output_path)
print(f"   Saved to:            {output_path}")
