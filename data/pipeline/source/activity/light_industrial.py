"""
CIMS Light Industrial Manufacturing GDP activity data processing
"""

import polars as pl
import pandas as pd
from pathlib import Path

import sys
_current_file = Path(__file__)
_project_root = _current_file.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from utils.data_extensions import extend_cagr_periods, compute_cagr, load_cagr_assumptions

# Configuration
BASE_PATH           = Path('C:/cims/data')
MAPPINGS_PATH       = BASE_PATH / 'mappings_conversions'
REGION_MAP_FILE     = MAPPINGS_PATH / 'region_map.csv'
LIGHT_INDUSTRY_FILE = BASE_PATH / 'raw_data/stats_can/activity/36100711.csv'
OUTPUT_DIR          = BASE_PATH / 'processed_data/activity'

DATA_START = 2000

ASSUMPTIONS_FILE = Path('C:/cims/data/raw_data/assumptions/activity_cagr_projections.csv')
CAGR_START, CAGR_END, CAGR_PERIODS = load_cagr_assumptions('Light Industrial', ASSUMPTIONS_FILE)

# Per-region overrides — one explicit annual rate per period.
CAGR_OVERRIDES: dict[str, tuple[float, ...]] = {
    "NS": (0.01, 0.01, 0.005),
    "PE": (0.01, 0.01, 0.005),
    "YT": (0.01, 0.01, 0.005),
}

# =============================================================================
# LIGHT INDUSTRIAL GDP ACTIVITY
# =============================================================================
# Reads Statistics Canada Table 36-10-0711-01 (GDP at basic prices by industry,
# provinces and territories, annual) and outputs activity levels for the seven
# CIMS Light Industrial Manufacturing nodes in units of $M GDP (chained 2017
# dollars), by province/territory, for 2000–2024.
#
# NAICS → CIMS node mapping:
#   Food Tobacco and Beverage  = Food manufacturing [311]
#                              + Beverage and tobacco product manufacturing [312]
#   Rubber and Plastics        = Plastics and rubber products manufacturing [326]
#   Leather and Textiles       = Textile, clothing and leather product mfg [31X]
#   Wood Products              = Wood product manufacturing [321]
#   Furniture Printing and
#     Machinery                = Printing and related support activities [323]
#                              + Machinery manufacturing [333]
#                              + Furniture and related product manufacturing [337]
#   Transportation Equipment   = Transportation equipment manufacturing [336]
#   Electronics and Other      = Computer and electronic product mfg [334]
#                              + Electrical equipment, appliance and component
#                                manufacturing [335]
#                              + Miscellaneous manufacturing [339]
#
# Source unit: millions of chained (2017) dollars  →  output unit: $M GDP
# (no unit conversion needed; values are already in $M)
#
# Data availability: 1997–2024. Years 2000–2024 are used.
# All 13 provinces/territories are present with no missing values.
# =============================================================================


NAICS_COL = 'North American Industry Classification System (NAICS)'

# Each tuple: (CIMS node name, list of NAICS labels to sum)
LIGHT_INDUSTRY_NODES = [
    (
        'Food Tobacco and Beverage',
        [
            'Food manufacturing [311]',
            'Beverage and tobacco product manufacturing [312]',
        ],
    ),
    (
        'Rubber and Plastics',
        [
            'Plastics and rubber products manufacturing [326]',
        ],
    ),
    (
        'Leather and Textiles',
        [
            'Textile, clothing and leather product manufacturing [31X]',
        ],
    ),
    (
        'Wood Products',
        [
            'Wood product manufacturing [321]',
        ],
    ),
    (
        'Furniture Printing and Machinery',
        [
            'Printing and related support activities [323]',
            'Machinery manufacturing [333]',
            'Furniture and related product manufacturing [337]',
        ],
    ),
    (
        'Transportation Equipment',
        [
            'Transportation equipment manufacturing [336]',
        ],
    ),
    (
        'Electronics and Other',
        [
            'Computer and electronic product manufacturing [334]',
            'Electrical equipment, appliance and component manufacturing [335]',
            'Miscellaneous manufacturing [339]',
        ],
    ),
]

# -- L1. Load and filter ------------------------------------------------------

raw_li = pl.read_csv(LIGHT_INDUSTRY_FILE)

# Collect all NAICS labels we need across all nodes
all_naics_needed = [label for _, labels in LIGHT_INDUSTRY_NODES for label in labels]

li_df = (
    raw_li
    .filter(
        (pl.col('Prices')   == 'Chained (2017) dollars') &
        (pl.col(NAICS_COL).is_in(all_naics_needed)) &
        (pl.col('REF_DATE') >= 2000) &
        (pl.col('REF_DATE') <= 2024)
    )
    .select(['REF_DATE', 'GEO', NAICS_COL, 'VALUE'])
    .rename({
        'REF_DATE': 'Year',
        'GEO':      'Region',
        NAICS_COL:  'NAICS',
        'VALUE':    'gdp_millions',
    })
    # Stats Can stores VALUE as strings (empty string = suppressed/confidential).
    # Cast to Float64 (strict=False turns empty strings into null), then fill
    # any suppressed cells with 0 so downstream sums are not disrupted.
    .with_columns(
        pl.col('gdp_millions').cast(pl.Float64, strict=False).fill_null(0.0)
    )
)


# -- L2. Aggregate NAICS codes into CIMS nodes --------------------------------
#
#   For nodes that map to a single NAICS code this is a passthrough.
#   For nodes that map to multiple codes (e.g. Food + Beverage), values are
#   summed across NAICS codes within each Region × Year group.

node_frames = []
for node_name, naics_labels in LIGHT_INDUSTRY_NODES:
    node_frames.append(
        li_df
        .filter(pl.col('NAICS').is_in(naics_labels))
        .group_by(['Region', 'Year'])
        .agg(pl.col('gdp_millions').sum().alias('Value'))
        .with_columns(pl.lit(node_name).alias('Variable'))
    )

sub_vars = pl.concat(node_frames)


# -- L3. Map NIR region names to CIMS codes -----------------------------------

region_map = pl.read_csv(REGION_MAP_FILE, columns=["CIMS", "NIR"])

sub_vars = (
    sub_vars
    .with_columns(
        pl.col("Region").str.replace(r'\s*\[.*?\]$', '').str.strip_chars()
    )
    .join(region_map, left_on="Region", right_on="NIR", how="left")
    .with_columns(pl.col("CIMS").fill_null(pl.col("Region")).alias("Region"))
    .drop("CIMS")
)


# -- L4. Build total and percentage-split variables ---------------------------
#
#   Light Industrial                          = sum of all sub-variables ($M GDP)
#   Light Industrial.Manufacturing.<node>     = share of total (fraction 0-1)

li_total = (
    sub_vars
    .group_by(["Region", "Year"])
    .agg(pl.col("Value").sum().alias("Value"))
    .with_columns([
        pl.lit("Light Industrial").alias("Variable"),
        pl.lit("$M 2017 GDP").alias("Unit"),
    ])
    .select(["Region", "Variable", "Unit", "Year", "Value"])
)

li_splits = (
    sub_vars
    .join(
        li_total.select(["Region", "Year", pl.col("Value").alias("total")]),
        on=["Region", "Year"], how="left",
    )
    .with_columns(
        pl.when(pl.col("total") > 0)
        .then(pl.col("Value") / pl.col("total"))
        .otherwise(0.0)
        .alias("Value"),
        ("Light Industrial.Manufacturing." + pl.col("Variable")).alias("Variable"),
        pl.lit("% of $M 2017 GDP").alias("Unit"),
    )
    .select(["Region", "Variable", "Unit", "Year", "Value"])
)

combined = pl.concat([li_total, li_splits]).sort(["Region", "Variable", "Year"])


# -- L5. Extend to 2100 -------------------------------------------------------
#
#   Light Industrial total: raw CAGR computed from direct point values at
#   CAGR_START and CAGR_END (2014–2024), then dampened per CAGR_PERIODS:
#     (2025-2035, 60%), (2036-2050, 40%), (2051-2100, 5%).
#   Percentage splits: held flat at last historical value (cagr = 0).

last_year = combined["Year"].max()

# Convert the Polars combined frame to a pandas pivot for easy per-series access
combined_pd = combined.to_pandas()

future_rows_list = []

for (region, variable, unit), grp in combined_pd.groupby(["Region", "Variable", "Unit"]):
    series = grp.set_index("Year")["Value"].sort_index()

    if unit == "% of $M 2017 GDP":
        # Percentage splits: hold flat at last historical value
        last_val = series.loc[last_year] if last_year in series.index else series.iloc[-1]
        for yr in range(last_year + 1, CAGR_PERIODS[-1][1] + 1):
            future_rows_list.append({
                "Region": region,
                "Variable": variable,
                "Unit": unit,
                "Year": yr,
                "Value": last_val,
            })
    else:
        # Total Light Industrial: CAGR from direct point values at CAGR_START / CAGR_END
        base_val = series.loc[CAGR_END] if CAGR_END in series.index else series.iloc[-1]
        raw_cagr = compute_cagr(series, CAGR_START, CAGR_END)

        ext = extend_cagr_periods(
            base_val=base_val,
            raw_cagr=raw_cagr,
            periods=CAGR_PERIODS,
            override=CAGR_OVERRIDES.get(region),
        )
        for yr, val in ext.items():
            future_rows_list.append({
                "Region": region,
                "Variable": variable,
                "Unit": unit,
                "Year": yr,
                "Value": val,
            })

future_rows = pl.DataFrame(future_rows_list).with_columns([
    pl.col("Year").cast(pl.Int64),
    pl.col("Value").cast(pl.Float64),
])

combined = pl.concat([combined, future_rows]).sort(["Region", "Variable", "Year"])


# -- L6. Save -----------------------------------------------------------------

print(f'\n✅ Light industrial complete')
print(f'   Total rows:          {len(combined):,}')
print(f'   Regions processed:   {combined["Region"].n_unique()}')
print(f'   Variables:           {sorted(combined["Variable"].unique().to_list())}')
print(f'   Years covered:       {combined["Year"].min()} – {combined["Year"].max()}')

li_path = OUTPUT_DIR / 'light_industrial.csv'
combined.write_csv(li_path)
print(f'   Saved to:            {li_path}')