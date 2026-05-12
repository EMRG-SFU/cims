"""
====================================
Gas Production Activity Calculator
====================================
Reads the CER natural gas production CSV and outputs:
  - Marketable Production per region (2000–2050) in 1000 m3/year
      (formerly "Total" — this is the CER marketable gas volume)
  - Total (Gross) Production = Marketable Production / Processed ratio
  - Processed ratio per region (2000–2100) as a fraction 0–1
      Derived from StatsCan 25100029 RESD:
        Processed = (Production − Producer consumption) / Production
      Regions: AB, BC, SK, NB, NS, NT, ON, YT
      NB missing values are set equal to NS.
      2024 value is extended flat to 2100.
  - Percentage splits (as fractions 0–1) for:
      * Conventional  = Non Associated + Solution gas
      * Shale
      * Tight
      * Coalbed Methane
  - LNG Compression % of gross production for BC (2000–2100)

Notes:
  - Source data starts at 2005. Years 2000–2004 are back-extrapolated
    using a linear trend fitted to the 2005–2010 values for Total only.
    Splits for 2000–2004 are held constant at 2005 fractions.
    Splits for 2005 onward are computed from actual sub-type data.
  - Source unit is Million m3/day -> converted to 1000 m3/year:
        Million m3/day x 1,000 x 365 = 1000 m3/year
  - Only the scenario defined in the control config is used.
  - Regions with no sub-type breakdown (e.g. Ontario, Yukon) default to
    conventional=1, all other splits=0.
"""

import pandas as pd
import polars as pl
from pathlib import Path

import sys
_current_file = Path(__file__)
_project_root = _current_file.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from utils.controls_conversions import load_control_config
from utils.extensions.data_extensions import trend_backwards, backfill_constant, extend_constant, interpolate_gaps, extend_cagr_periods, compute_cagr, load_cagr_assumptions

# Configuration
BASE_PATH = Path('C:/cims/data')
NATURAL_GAS_PRODUCTION_FILE = BASE_PATH / 'raw_data/cer/natural-gas-production-2026.csv'
RESD_FILE                   = BASE_PATH / 'raw_data/stats_can/resd/25100029.csv'
LNG_EXPORT_FILE             = BASE_PATH / 'raw_data/cer/lng-export-assumptions-2026.csv'
OUTPUT_DIR                  = BASE_PATH / 'processed_data/activity'
SCENARIO                    = load_control_config()["cer_ef_reference_scenario"]
ASSUMPTIONS_FILE = Path('C:/cims/data/raw_data/assumptions/activity_cagr_projections.csv')
CAGR_START, CAGR_END, CAGR_PERIODS = load_cagr_assumptions('Natural Gas', ASSUMPTIONS_FILE)

# Per-region overrides — one explicit annual rate per period.
# Bypasses the dampener for that region. Use when historical CAGR is unrealistic.
CAGR_OVERRIDES: dict[str, tuple[float, ...]] = {
}


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

total_df     = total_df.with_columns(pl.col("Year").cast(pl.Int64))
other_df     = other_df.with_columns(pl.col("Year").cast(pl.Int64))
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
).drop(["pct_conventional_2005", "pct_shale_2005", "pct_tight_2005", "pct_coalbed_methane_2005"])


# -- 5. Build Processed ratio from StatsCan RESD (25100029) ------------------
#
#   Processed = (Production − Producer consumption) / Production
#   Provinces: AB, BC, SK, NB, NS, NT, ON, YT
#   NB missing values are set equal to NS.
#   Gaps where production = 0 are forward-filled from the last valid value.
#   2024 value is extended flat to 2100.

_RESD_GEO_MAP = {
    "Alberta":                  "AB",
    "British Columbia":         "BC",
    "Saskatchewan":             "SK",
    "New Brunswick":            "NB",
    "Nova Scotia":              "NS",
    "Northwest Territories":    "NT",
    "Ontario":                  "ON",
    "Yukon":                    "YT",
}

_resd_raw = pl.read_csv(RESD_FILE, infer_schema_length=0)

_resd_pivot = (
    _resd_raw
    .filter(
        (pl.col("Fuel type") == "Natural gas") &
        pl.col("Supply and demand characteristics").is_in(
            ["Production", "Producer consumption"]
        ) &
        pl.col("GEO").is_in(list(_RESD_GEO_MAP.keys())) &
        (pl.col("REF_DATE").cast(pl.Int32) >= 2000) &
        (pl.col("REF_DATE").cast(pl.Int32) <= 2024)
    )
    .with_columns([
        pl.col("GEO").replace(_RESD_GEO_MAP).alias("Province"),
        pl.col("REF_DATE").cast(pl.Int32).alias("Year"),
        pl.col("VALUE").cast(pl.Float64, strict=False),
    ])
    .pivot(
        index=["Year", "Province"],
        on="Supply and demand characteristics",
        values="VALUE",
        aggregate_function="first",
    )
)

# Compute processed ratio only where both Production and Producer consumption
# are known. If producer consumption is null (not reported, not zero), we treat
# the ratio as unknown so backfill_constant can fill it from the nearest real
# observation rather than defaulting to 1.0.
_resd_pivot = _resd_pivot.with_columns(
    pl.when(
        (pl.col("Production").fill_null(0.0) > 0) &
        pl.col("Producer consumption").is_not_null()
    )
    .then(
        (pl.col("Production") - pl.col("Producer consumption"))
        / pl.col("Production")
    )
    .alias("Processed")
)

# Build a complete grid: all 8 provinces × years 2000–2100
_all_years = list(range(2000, 2101))
_all_provs = list(_RESD_GEO_MAP.values())
_processed = (
    pl.DataFrame({"Province": [p for p in _all_provs for _ in _all_years],
                  "Year":     [y for _ in _all_provs for y in _all_years]})
    .join(_resd_pivot.select(["Province", "Year", "Processed"]),
          on=["Province", "Year"], how="left")
)

# NB matches NS: replace NB Processed with NS Processed
_ns_vals = (
    _processed
    .filter(pl.col("Province") == "NS")
    .select(["Year", pl.col("Processed").alias("NS_Processed")])
)
_processed = (
    _processed
    .join(_ns_vals, on="Year", how="left")
    .with_columns(
        pl.when(pl.col("Province") == "NB")
        .then(pl.col("NS_Processed"))
        .otherwise(pl.col("Processed"))
        .alias("Processed")
    )
    .drop("NS_Processed")
    .sort(["Province", "Year"])
)

# Extend each province's series using backfill_constant (fills gaps before the
# first valid value with that first value) then extend_constant (holds the last
# value flat to 2100).
def _extend_processed(grp: pl.DataFrame) -> pl.DataFrame:
    province = grp["Province"][0]
    s = pd.Series(grp["Processed"].to_list(), index=grp["Year"].to_list(), dtype=float)
    s = backfill_constant(s, start_year=2000)
    s = extend_constant(s, end_year=2100)
    return pl.DataFrame({
        "Province": [province] * len(s),
        "Year":     [int(y) for y in s.index],
        "Processed": list(s.values),
    })

_processed_pl = (
    _processed
    .group_by("Province", maintain_order=True)
    .map_groups(_extend_processed)
)
_processed_pl = interpolate_gaps(
    _processed_pl,
    group_cols=["Province"],
    year_col="Year",
    value_col="Processed",
)

# Validate: clip to [0, 1] to guard against data anomalies
_processed_pl = _processed_pl.with_columns(
    pl.col("Processed").clip(0.0, 1.0)
)

# Build processed output table (2000–2100)
_processed_output = (
    _processed_pl
    .rename({"Province": "Region"})
    .with_columns(
        pl.lit("Processing").alias("Variable"),
        pl.lit("% of 1000m3").alias("Unit"),
        pl.col("Processed").alias("Value"),
    )
    .select(["Region", "Variable", "Unit", "Year", "Value"])
)


# -- 5b. Map CER region names to province abbreviations -----------------------
#
#   The CER file uses full region names; we need province abbreviations
#   to join with the Processed table.

_CER_REGION_MAP = {
    "Alberta":               "AB",
    "British Columbia":      "BC",
    "Saskatchewan":          "SK",
    "New Brunswick":         "NB",
    "Nova Scotia":           "NS",
    "Northwest Territories": "NT",
    "Ontario":               "ON",
    "Yukon":                 "YT",
    # Other CER regions not in scope — kept as-is for completeness
}

pivot = pivot.with_columns(
    pl.col("Region")
    .replace_strict(_CER_REGION_MAP, default=None)
    .fill_null(pl.col("Region"))
    .alias("Region")
)


# -- 5c. Join Processed and compute Gross (Total) Production ------------------
#
#   Marketable Production = CER "Total" (already in pivot as "total")
#   Total (Gross) Production = Marketable Production / Processed
#   Processed is already extended to 2100; pivot will be extended in 5e below.

_processed_pl_regions = _processed_pl.rename({"Province": "Region"})

pivot = (
    pivot
    .join(_processed_pl_regions, on=["Region", "Year"], how="left")
    .with_columns(
        (pl.col("total") / pl.col("Processed")).alias("gross_total")
    )
)


# -- 5e. Extend pivot to 2100 -------------------------------------------------
#
#   Marketable Production (total): extended per region using the CAGR
#   calculated from 2035–2050, floored at 0.
#   Gross Production (gross_total): recalculated as total / Processed,
#   where Processed is the 2050 value held flat (already done in _processed_pl).
#   Splits (pct_*): held constant at 2050 values per region.

# Anchor values for CAGR: marketable production at CAGR_START and CAGR_END per region.
# The raw CAGR is computed from the historical window defined in the assumptions CSV
# (Natural Gas: 2040–2050). Dampening is handled by extend_cagr_periods via CAGR_PERIODS.
_anchor = (
    pivot
    .filter(pl.col("Year").is_in([CAGR_START, CAGR_END]))
    .select(["Region", "Year", "total"])
    .pivot(values="total", index="Region", on="Year", aggregate_function="first")
    .rename({str(CAGR_START): "total_cagr_start", str(CAGR_END): "total_cagr_end"})
)

# CAGR = (total_cagr_end / total_cagr_start) ^ (1 / (CAGR_END - CAGR_START)) - 1
# If total_cagr_start is 0 or null, CAGR is set to 0 (production stays flat at 0).
# Dampening (20% for 2051-2100) is applied inside extend_cagr_periods via CAGR_PERIODS.
_anchor = _anchor.with_columns(
    pl.when((pl.col("total_cagr_start").is_null()) | (pl.col("total_cagr_start") == 0))
        .then(0.0)
        .otherwise(
            (pl.col("total_cagr_end") / pl.col("total_cagr_start")).pow(1.0 / (CAGR_END - CAGR_START)) - 1.0
        )
        .alias("cagr")
)

# 2050 values per region for splits and Processed (held flat beyond 2050)
_vals_2050 = (
    pivot
    .filter(pl.col("Year") == 2050)
    .select([
        "Region",
        "total",
        "Processed",
        "pct_conventional",
        "pct_shale",
        "pct_tight",
        "pct_coalbed_methane",
    ])
    .rename({"total": "total_2050_val", "Processed": "processed_2050"})
)

# Build extension rows for 2051–2100
# Convert to dicts for fast per-region lookup without pandas roundtrip
_anchor_map    = {r["Region"]: r for r in _anchor.to_dicts()}
_vals_2050_map = {r["Region"]: r for r in _vals_2050.to_dicts()}

_extension_rows = []
for _region, _anc in _anchor_map.items():
    _base        = _vals_2050_map[_region]["total_2050_val"]  # compounds from CAGR_END (2050)
    _processed50 = _vals_2050_map[_region]["processed_2050"]
    _pct_conv    = _vals_2050_map[_region]["pct_conventional"]
    _pct_shale   = _vals_2050_map[_region]["pct_shale"]
    _pct_tight   = _vals_2050_map[_region]["pct_tight"]
    _pct_cbm     = _vals_2050_map[_region]["pct_coalbed_methane"]

    ext = extend_cagr_periods(
        base_val = _base,
        raw_cagr = _anc["cagr"],
        periods  = CAGR_PERIODS,
        override = CAGR_OVERRIDES.get(_region),
    )
    for _yr, _total in ext.items():
        _gross = (_total / _processed50) if (_processed50 and _processed50 > 0) else 0.0
        _extension_rows.append({
            "Region":              _region,
            "Year":                _yr,
            "total":               _total,
            "gross_total":         _gross,
            "Processed":           _processed50,
            "pct_conventional":    _pct_conv,
            "pct_shale":           _pct_shale,
            "pct_tight":           _pct_tight,
            "pct_coalbed_methane": _pct_cbm,
        })

_extension_pl = pl.DataFrame(_extension_rows).with_columns(pl.col("Year").cast(pl.Int64))

# Drop helper columns added during the 2050 join before concatenating
pivot = pivot.drop([c for c in ["Processed"] if c in pivot.columns])
_extension_pl = _extension_pl.drop("Processed")

pivot = pl.concat([pivot, _extension_pl], how="diagonal").sort(["Region", "Year"])


# -- 5d. LNG exports for BC ---------------------------------------------------
#
#   Source: CER LNG export assumptions (Bcf/day per scenario).
#   Conversion: Bcf/day × 1e9 cf/Bcf × 0.0283168 m3/cf × 365 days/yr ÷ 1000
#             = Bcf/day × 10,335,632  →  1000 m3/year
#   LNG Export % = LNG volume (1000 m3) / BC gross production (1000 m3)
#   For 2000–2050: actual CER LNG volumes are used.
#   For 2051–2100: 2050 LNG volume is held flat; fraction recalculates naturally
#   against the extended gross_total.

_BCF_TO_1000M3 = 1e9 * 0.0283168 * 365 / 1000   # ≈ 10,335,632

# Use the reference scenario from control config — column name is "<SCENARIO>"
# so if SCENARIO changes in control.py this picks up the right column automatically.
_lng_col = f"{SCENARIO}"
_lng_raw = (
    pl.read_csv(LNG_EXPORT_FILE)
    .select(["Year", _lng_col])
    .rename({_lng_col: "lng_vol"})
    .with_columns(
        (pl.col("lng_vol").fill_null(0.0) * _BCF_TO_1000M3).alias("lng_vol")
    )
)

# Hold the 2050 volume flat for 2051–2100
_lng_vol_2050 = (
    _lng_raw.filter(pl.col("Year") == 2050)
    .select("lng_vol")
    .item()
)
_lng_extension = pl.DataFrame({
    "Year":    list(range(2051, 2101)),
    "lng_vol": [_lng_vol_2050] * 50,
})
_lng_all = pl.concat([_lng_raw, _lng_extension]).filter(
    (pl.col("Year") >= 2000) & (pl.col("Year") <= 2100)
)

# Join BC gross production and compute LNG fraction.
# LNG draws from the full gas stream before processing losses, so gross_total
# is the correct denominator. The share is capped at 1.0 — LNG can never
# exceed total production.
_bc_gross = (
    pivot
    .filter(pl.col("Region") == "BC")
    .select(["Year", "gross_total"])
)

_lng_output = (
    _lng_all
    .join(_bc_gross, on="Year", how="left")
    .with_columns(
        pl.when((pl.col("gross_total").is_not_null()) & (pl.col("gross_total") > 0))
            .then((pl.col("lng_vol") / pl.col("gross_total")).clip(0.0, 1.0))
            .otherwise(0.0)
            .alias("Value")
    )
    .with_columns(
        pl.lit("BC").alias("Region"),
        pl.lit("LNG Compression").alias("Variable"),
        pl.lit("% of 1000m3").alias("Unit"),
    )
    .select(["Region", "Variable", "Unit", "Year", "Value"])
    .sort("Year")
)


# -- 6. Build final output ----------------------------------------------------
#   pivot now covers 2000–2100: CER data for 2000–2050, CAGR extension for 2051–2100.

output_gas = (
    pivot
    .select([
        "Region",
        "Year",
        "total",        # Marketable Production
        "gross_total",  # Total (Gross) Production
        "pct_conventional",
        "pct_shale",
        "pct_tight",
        "pct_coalbed_methane",
    ])
    .unpivot(
        index=["Region", "Year"],
        on=["total", "gross_total", "pct_conventional", "pct_shale",
            "pct_tight", "pct_coalbed_methane"],
        variable_name="Variable",
        value_name="Value",
    )
    .with_columns(
        pl.col("Variable").replace({
            "total":               "Marketable Production",
            "gross_total":         "Extraction",
            "pct_conventional":    "Extraction.Conventional",
            "pct_shale":           "Extraction.Shale",
            "pct_tight":           "Extraction.Tight",
            "pct_coalbed_methane": "Extraction.Coalbed Methane",
        }).alias("Variable"),
        pl.col("Variable").replace({
            "total":               "1000m3",
            "gross_total":         "1000m3",
            "pct_conventional":    "% of 1000m3",
            "pct_shale":           "% of 1000m3",
            "pct_tight":           "% of 1000m3",
            "pct_coalbed_methane": "% of 1000m3",
        }).alias("Unit"),
    )
    .select(["Region", "Variable", "Unit", "Year", "Value"])
    .sort(["Region", "Variable", "Year"])
)

# Combine gas production output with the Processed ratio and LNG export series
output = pl.concat([output_gas, _processed_output, _lng_output]).sort(["Region", "Variable", "Year"])


# -- 7. Save ------------------------------------------------------------------

print(f"\n✅ Gas production complete")
print(f"   Total rows:          {len(output):,}")
print(f"   Regions processed:   {output['Region'].n_unique()}")
print(f"   Years covered:       {output['Year'].min()} – {output['Year'].max()}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
output_path = OUTPUT_DIR / 'gas_production.csv'
output.write_csv(output_path)
print(f"   Saved to:            {output_path}")
