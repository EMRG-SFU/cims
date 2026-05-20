"""
=============================================================================
AGRICULTURE, WASTE, CONSTRUCTION, FORESTRY EMISSIONS
=============================================================================
Reads the Environment Canada GHG by economic sector CSV and outputs a single
combined CSV with columns: Region, Variable, Unit, Year, Value.

AGRICULTURE (per individual province/territory, excl. Canada totals):
  - Agriculture         : Total Agriculture emissions (kt CO2eq -> tCO2e)
  - Agriculture.Heat    : On Farm Fuel Use / Agriculture Total
  - Agriculture.Process.Soils
                        : Crop Production / Agriculture Total
  - Agriculture.Process.Enteric Fermentation and Manure Management
                        : Animal Production / Agriculture Total

WASTE (per individual province/territory, excl. Canada totals):
  - Waste               : Total Waste emissions (kt CO2eq -> tCO2e)

CONSTRUCTION (per individual province/territory, excl. Canada totals):
  - Construction        : Total Construction emissions (kt CO2eq -> tCO2e)

FORESTRY (per individual province/territory, excl. Canada totals):
  - Forestry            : Total Forestry emissions (kt CO2eq -> tCO2e)

Unit conversion:  kt CO2eq  ->  tCO2e  =  value * 1,000

Data availability: 1990-2024.  Years 2000+ are used directly.
Region names are mapped from NIR to CIMS codes via region_map.csv.
Extension to 2100 via CAGR over the full historical period (DATA_START–last
year), giving a smoother structural trend. Percentage variables held flat.

SUPPRESSION HANDLING:
    The ECCC GHG CSV uses 'x' for suppressed or confidential cell values,
    primarily for NWT and Nunavut Construction totals in certain years.
    These are declared as null at load time via pl.read_csv(null_values=["x"]).
    Mid-series null gaps are then filled by Polars linear interpolation;
    trailing nulls are forward-filled from the last valid value. Percentage
    variables (Agriculture.Heat, etc.) are held flat at their last valid
    observation rather than interpolated, as interpolating emission-intensity
    ratios would misrepresent the underlying activity split.
=============================================================================
"""

import polars as pl
from pathlib import Path

import sys
_current_file = Path(__file__)
_project_root = _current_file.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from utils.data_extensions import extend_cagr_periods, compute_cagr, load_cagr_assumptions
from utils.controls_conversions import load_control_config, BASE_PATH


# Configuration
MAPPINGS_PATH    = BASE_PATH / 'mappings_conversions'
REGION_MAP_FILE  = MAPPINGS_PATH / 'region_map.csv'
GHG_FILE         = BASE_PATH / 'raw_data/eccc/nir/GHG_Econ_Can_Prov_Terr.csv'
OUTPUT_DIR       = BASE_PATH / 'processed_data/activity'

# Regions to exclude (national/territorial aggregates)
_EXCLUDE_REGIONS = {"Canada", "Northwest Territories and Nunavut"}

KT_TO_T = 1_000   # kt CO2eq -> tCO2e

DATA_START = load_control_config()["pipeline"]["data_start"]

ASSUMPTIONS_FILE = BASE_PATH / 'raw_data/assumptions/activity_cagr_projections.csv'
CAGR_START, CAGR_END, CAGR_PERIODS = load_cagr_assumptions('Emissions Drivers', ASSUMPTIONS_FILE)

# Per-region overrides — one explicit annual rate per period.
CAGR_OVERRIDES: dict[str, tuple[float, ...]] = {
    ("Alberta", "Waste"): (0.02, 0.01, 0.005),
}


# -- A1. Load GHG file --------------------------------------------------------

raw_ghg = pl.read_csv(
    GHG_FILE,
    columns=["Year", "Region", "Source", "Sector", "Total (kt CO2e)"],
    null_values=["x"],
    schema_overrides={"Total (kt CO2e)": pl.Float64},
)

# Keep only Agriculture and Waste rows, individual regions, years 2000+.
# Sector is null on total rows (no sub-sector breakdown); fill with "" so
# filters below can use a plain string equality check.
ghg = (
    raw_ghg
    .filter(
        pl.col("Source").is_in(["Agriculture", "Waste"]) &
        (~pl.col("Region").is_in(list(_EXCLUDE_REGIONS))) &
        (pl.col("Year") >= DATA_START)
    )
    .with_columns(
        pl.col("Sector").fill_null("").alias("Sector"),
    )
    .select(["Year", "Region", "Source", "Sector", pl.col("Total (kt CO2e)").alias("CO2eq")])
)

# Interpolate any null CO2eq values using surrounding years within each
# Region+Source+Sector group, then forward-fill any trailing nulls,
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

agri_total = (
    agri_raw
    .filter(pl.col("Sector") == "")
    .select(["Region", "Year", pl.col("tco2e").alias("agri_total")])
)
agri_heat = (
    agri_raw
    .filter(pl.col("Sector") == "On Farm Fuel Use")
    .select(["Region", "Year", pl.col("tco2e").alias("agri_heat")])
)
agri_soils = (
    agri_raw
    .filter(pl.col("Sector") == "Crop Production")
    .select(["Region", "Year", pl.col("tco2e").alias("agri_soils")])
)
agri_animal = (
    agri_raw
    .filter(pl.col("Sector") == "Animal Production")
    .select(["Region", "Year", pl.col("tco2e").alias("agri_animal")])
)

# Join all Agriculture sub-tables and compute splits (fraction 0-1)
agri_pivot = (
    agri_total
    .join(agri_heat,   on=["Region", "Year"], how="left")
    .join(agri_soils,  on=["Region", "Year"], how="left")
    .join(agri_animal, on=["Region", "Year"], how="left")
    .sort(["Region", "Year"])
    .with_columns([
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
)

agri_output = (
    agri_pivot
    .select(["Region", "Year", "agri_total", "pct_heat", "pct_soils", "pct_animal"])
    .unpivot(
        index=["Region", "Year"],
        on=["agri_total", "pct_heat", "pct_soils", "pct_animal"],
        variable_name="Variable",
        value_name="Value",
    )
    .with_columns([
        pl.col("Variable").replace({
            "agri_total":  "Agriculture",
            "pct_heat":    "Agriculture.Heat",
            "pct_soils":   "Agriculture.Process.Soils",
            "pct_animal":  "Agriculture.Process.Enteric Fermentation and Manure Management",
        }),
        pl.col("Variable").replace({
            "agri_total":  "tCO2e",
            "pct_heat":    "% of tCO2e",
            "pct_soils":   "% of tCO2e",
            "pct_animal":  "% of tCO2e",
        }).alias("Unit"),
    ])
    .select(["Region", "Variable", "Unit", "Year", "Value"])
    .sort(["Region", "Variable", "Year"])
)


# -- A3. Build Waste output ---------------------------------------------------

waste_output = (
    ghg
    .filter((pl.col("Source") == "Waste") & (pl.col("Sector") == ""))
    .select(["Region", "Year", pl.col("tco2e").alias("Value")])
    .with_columns([
        pl.lit("Waste").alias("Variable"),
        pl.lit("tCO2e").alias("Unit"),
    ])
    .select(["Region", "Variable", "Unit", "Year", "Value"])
    .sort(["Region", "Variable", "Year"])
)


# -- A4. Build Construction and Forestry outputs ------------------------------
#
#   Both sit under Source == "Light Manufacturing, Construction and Forest Resources".
#   Construction  -> Sector == "Construction"
#   Forestry      -> Sector == "Forest Resources"
#
#   Note: NWT and Nunavut Construction are suppressed (x) for some years.
#   Interpolation fills gaps; forward-fill handles any trailing nulls.

lmcf_raw = (
    pl.read_csv(
        GHG_FILE,
        columns=["Year", "Region", "Source", "Sector", "Total (kt CO2e)"],
        null_values=["x"],
        schema_overrides={"Total (kt CO2e)": pl.Float64},
    )
    .rename({"Total (kt CO2e)": "CO2eq"})
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

def _build_simple_output(df: pl.DataFrame, sector: str, variable: str) -> pl.DataFrame:
    """Extract a single sector as a Total tCO2e output table."""
    return (
        df
        .filter(pl.col("Sector") == sector)
        .select(["Region", "Year", pl.col("tco2e").alias("Value")])
        .with_columns([
            pl.lit(variable).alias("Variable"),
            pl.lit("tCO2e").alias("Unit"),
        ])
        .select(["Region", "Variable", "Unit", "Year", "Value"])
        .sort(["Region", "Variable", "Year"])
    )

construction_output = _build_simple_output(lmcf_raw, "Construction",    "Construction")
forestry_output     = _build_simple_output(lmcf_raw, "Forest Resources", "Forestry")


# -- A5. Combine ---------------------------------------------------------------

combined = pl.concat([
    agri_output,
    waste_output,
    construction_output,
    forestry_output,
]).sort(["Region", "Variable", "Year"])


# -- A6. Map NIR region names to CIMS codes ------------------------------------

region_map = pl.read_csv(REGION_MAP_FILE, columns=["CIMS", "NIR"])

combined = (
    combined
    .join(region_map, left_on="Region", right_on="NIR", how="left")
    .with_columns(
        pl.col("CIMS").fill_null(pl.col("Region")).alias("Region")
    )
    .drop("CIMS")
    .sort(["Region", "Variable", "Year"])
)


# -- A7. Extend to 2100 using CAGR with per-period dampeners ------------------
#
#   Raw CAGR is computed from direct point values at CAGR_START and CAGR_END
#   (loaded from the assumptions CSV).  Percentage variables are held flat at
#   the last historical value.  Three projection periods with dampeners are
#   applied via extend_cagr_periods().

last_year = combined["Year"].max()

future_rows_list = []

for (region, variable, unit), grp_pl in combined.group_by(["Region", "Variable", "Unit"]):
    series = (
        grp_pl
        .sort("Year")
        .select(["Year", "Value"])
        .to_pandas()
        .set_index("Year")["Value"]
        .astype(float)
    )

    if unit == "% of tCO2e":
        # Hold percentage variables flat at the last historical value.
        base_val = series.loc[last_year] if last_year in series.index else series.iloc[-1]
        for yr in range(last_year + 1, CAGR_PERIODS[-1][1] + 1):
            future_rows_list.append({
                "Region": region, "Variable": variable, "Unit": unit,
                "Year": yr, "Value": base_val,
            })
    else:
        raw_cagr = compute_cagr(series, CAGR_START, CAGR_END)
        base_val = series.loc[CAGR_END] if CAGR_END in series.index else series.iloc[-1]

        ext = extend_cagr_periods(
            base_val=base_val,
            raw_cagr=raw_cagr,
            periods=CAGR_PERIODS,
            override=CAGR_OVERRIDES.get(region),
        )
        for yr, val in ext.items():
            future_rows_list.append({
                "Region": region, "Variable": variable, "Unit": unit,
                "Year": yr, "Value": val,
            })

if future_rows_list:
    future_rows = pl.DataFrame(future_rows_list).cast({"Year": pl.Int64, "Value": pl.Float64})
    combined = pl.concat([combined, future_rows]).sort(["Region", "Variable", "Year"])


# -- A8. Save ------------------------------------------------------------------

print(f"\n✅ Emissions complete")
print(f"   Total rows:          {len(combined):,}")
print(f"   Regions processed:   {combined['Region'].n_unique()}")
print(f"   Variables:           {sorted(combined['Variable'].unique().to_list())}")
print(f"   Years covered:       {combined['Year'].min()} - {combined['Year'].max()}")

output_path = OUTPUT_DIR / 'emissions_drivers.csv'
output_path.parent.mkdir(parents=True, exist_ok=True)
combined.write_csv(output_path)
print(f"   Saved to:            {output_path}")
