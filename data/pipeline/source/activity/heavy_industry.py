"""
heavy_industry.py
=================
Full pipeline from raw StatsCan and CEEDC inputs to provincial physical output
estimates by CIMS sector, extended to 2100 via CAGR projections.

PIPELINE OVERVIEW:
  Step 1  Load nominal Gross Output (GO) and price indexes (IPPI, RMPI)
  Step 2  Deflate nominal GO to real GO (2020 base year)
  Step 3  Compute each province's share of national real GO by sector and year
  Step 4  Load CEEDC annual physical output
  Step 5  Extend GO shares to cover years beyond the GO data end (hold last year constant)
  Step 6  Build weighted combined GO shares for multi-sector CIMS products
  Step 7  Build national physical output table (summing components where needed)
  Step 8  Disaggregate national totals to provinces using GO shares
  Step 9  Project to 2100 using CAGR dampeners; add Source column; save

HOW TO RUN:
    python heavy_industry.py

INPUTS:
    raw_data/stats_can/activity/36100488.csv         <- StatsCan GO (nominal, incl. mining)
    raw_data/stats_can/activity/18100267.csv         <- StatsCan IPPI (incl. primary metals)
    raw_data/stats_can/activity/18100268.csv         <- StatsCan RMPI (raw materials)
    raw_data/ceedc/Production.xlsx                   <- CEEDC annual physical output
    raw_data/assumptions/activity_cagr_projections.csv  <- CAGR projection assumptions

OUTPUTS:
    processed_data/activity/heavy_industry.csv
        Columns: Region | Variable | Unit | Source | Year | Value
        Variables: parent sectors in tonnes; child sub-products as fraction of parent (% of tonnes)
        Source: "CEEDC/Stats Can" up to last_data_year["ceedc"], "Assumptions" thereafter
        UTF-8-BOM encoded (opens correctly in Excel)

LIBRARY NOTES:
    Polars is used for all DataFrame work (fast, memory-efficient).
    pandas is kept only for pd.read_excel() (Polars xlsx support requires extra deps)
    and for the final UTF-8-BOM CSV write (Polars write_csv does not support BOM).

SUPPRESSION HANDLING:
    Statistics Canada suppresses GO, IPPI, and RMPI cells where disclosure
    would identify individual respondents, marking them with codes 'x', '..', or 'F'.
    Suppressed rows are identified (VALUE fails cast to Float64) and removed entirely —
    not filled with 0 or interpolated — because suppressed GO values would distort the
    provincial share weights. The _go_suppressed table tracks which sector/province/year
    combinations were excluded so mid-series gaps can be interpolated downstream.
"""

import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path

import sys
_current_file = Path(__file__)
_project_root = _current_file.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from utils.data_fill import interpolate_gaps
from utils.data_extensions import extend_cagr_periods
from utils.extractors.stats_can import read_statscan_csv
from utils.controls_conversions import load_control_config, DATA_START, BASE_PATH

# =============================================================================
# Configuration
# =============================================================================
GO_FILE               = BASE_PATH / 'raw_data/stats_can/activity/36100488.csv'
IPPI_FILE             = BASE_PATH / 'raw_data/stats_can/activity/18100267.csv'
RMPI_FILE             = BASE_PATH / 'raw_data/stats_can/activity/18100268.csv'
PRODUCTION_FILE       = BASE_PATH / 'raw_data/ceedc/Production.xlsx'
CAGR_ASSUMPTIONS_FILE = BASE_PATH / 'raw_data/assumptions/activity_cagr_projections.csv'
REGION_MAP_FILE       = BASE_PATH / 'mappings_conversions/region_map.csv'

BASE_YEAR = 2020   # real GO base year (index = 100); matches IPPI/RMPI base period.

# =============================================================================
# DEFLATOR MAPPING
# =============================================================================
DEFLATOR_MAP = {
    # --- Mining ---
    "21220":  {"source": "RMPI", "label": "Metal ores, concentrates and scrap [M61]"},
    "21230":  {"source": "RMPI", "label": "Non-metallic minerals [M31]"},
    "212396": {"source": "RMPI", "label": "Potash [161]"},
    # --- Pulp / Paper ---
    "32210":  {"source": "IPPI", "label": "Pulp, paper and paperboard mills [3221]"},
    # --- Chemicals ---
    "32510":  {"source": "IPPI", "label": "Basic chemical manufacturing [3251]"},
    "325C0":  {"source": "IPPI", "label": "Chemical manufacturing [325]"},
    "32530":  {"source": "IPPI", "label": "Pesticide, fertilizer and other agricultural chemical manufacturing [3253]"},
    "32540":  {"source": "IPPI", "label": "Pharmaceutical and medicine manufacturing [3254]"},
    # --- Non-metallic minerals ---
    "327A0":  {"source": "IPPI", "label": "Lime and gypsum product manufacturing [3274]"},
    "32730":  {"source": "IPPI", "label": "Cement and concrete product manufacturing [3273]"},
    # --- Primary metals ---
    "33100":  {"source": "IPPI", "label": "Primary metal manufacturing [331]"},
    "331100": {"source": "IPPI", "label": "Iron and steel mills and ferro-alloy manufacturing [3311]"},
    "331300": {"source": "IPPI", "label": "Alumina and aluminum production and processing [3313]"},
    "331400": {"source": "IPPI", "label": "Non-ferrous metal (except aluminum) production and processing [3314]"},
}

# Maps GO file codes (BS-prefix) to clean NAICS sector labels used in outputs
NAICS_LABEL_MAP = {
    "21220":  "2122",
    "21230":  "2123",
    "212396": "212396",
    "32210":  "3221",
    "32510":  "3251",
    "325C0":  "325",
    "32530":  "3253",
    "32540":  "3254",
    "327A0":  "327A0",
    "32730":  "3273",
    "33100":  "3310",
    "331100": "3311",
    "331300": "3313",
    "331400": "3314",
}

# =============================================================================
# SECTOR DEFINITIONS
# =============================================================================
SECTORS = [
    # ── Mining ──────────────────────────────────────────────────────────────
    {
        "type": "combined",
        "cims_sector": "Mining",
        "note": "Sum of 2122 (metal ore) + 212396 (potash); "
                "provincial share weighted by real GO from sectors 2122 and 212396",
        "components": [
            {"naics_code": "2122",   "measure": "Production / Shipments", "go_sector": "2122"},
            {"naics_code": "212396", "measure": "Potash produced",        "go_sector": "212396"},
        ],
    },
    {
        "type": "single",
        "naics_code": "21221", "measure": "Production / Shipments",
        "cims_sector": "Iron Product",
        "go_sector": "2122",
        "note": "Iron ore mining (21221); GO sector 2122 used as parent proxy",
    },
    {
        "type": "derived",
        "cims_sector": "Non-Iron Product",
        "note": "Derived: 2122 (total metal ore) minus 21221 (iron ore); "
                "provincial share weighted by real GO from sector 2122",
        "minuend":    {"naics_code": "2122",  "measure": "Production / Shipments", "go_sector": "2122"},
        "subtrahend": {"naics_code": "21221", "measure": "Production / Shipments"},
    },
    {
        "type": "single",
        "naics_code": "212396", "measure": "Potash produced",
        "cims_sector": "Potash",
        "go_sector": "212396",
        "note": "Direct GO match",
    },
    # ── Pulp & Paper ────────────────────────────────────────────────────────
    {
        "type": "single",
        "naics_code": "32211",  "measure": "Pulp, total",
        "cims_sector": "Pulp and Paper",
        "go_sector": "3221",
        "note": "NAICS 32211; parent GO sector 3221",
    },
    {
        "type": "single",
        "naics_code": "32211",  "measure": "Pulp, total",
        "cims_sector": "Pulp",
        "go_sector": "3221",
        "note": "NAICS 32211; parent GO sector 3221; separate CIMS product",
    },
    {
        "type": "single",
        "naics_code": "322121", "measure": "Total paper (except newsprint)",
        "cims_sector": "Uncoated, coated, tissue",
        "go_sector": "3221",
        "note": "Parent GO sector 3221",
    },
    {
        "type": "single",
        "naics_code": "322122", "measure": "Newsprint",
        "cims_sector": "Newsprint",
        "go_sector": "3221",
        "note": "Parent GO sector 3221",
    },
    {
        "type": "single",
        "naics_code": "32213",  "measure": "Paperboard",
        "cims_sector": "Linerboard",
        "go_sector": "3221",
        "note": "Parent GO sector 3221",
    },
    # ── Chemicals ───────────────────────────────────────────────────────────
    {
        "type": "single",
        "naics_code": "325", "measure": "Total chemicals",
        "cims_sector": "Chemical Product",
        "go_sector": "325",
        "note": "Direct GO match",
    },
    {
        "type": "combined",
        "cims_sector": "Other Petrochemicals",
        "note": "Sum of 32511 (petrochemicals) + 3252 (resin, rubber, fibres); "
                "provincial share weighted by real GO from sectors 3251 and 325",
        "components": [
            {"naics_code": "32511", "measure": "Total petrochemicals",           "go_sector": "3251"},
            {"naics_code": "3252",  "measure": "Total resin, rubber, and fibres", "go_sector": "325"},
        ],
    },
    {
        "type": "single",
        "naics_code": "32518",  "measure": "Total inorganic chemicals",
        "cims_sector": "Chlor Alkali, Hydrogen Peroxide, Sodium Chlorate",
        "go_sector": "3251",
        "note": "Parent GO sector 3251",
    },
    {
        "type": "single",
        "naics_code": "32519",  "measure": "Formaldehyde",
        "cims_sector": "Adipic Acid",
        "go_sector": "3251",
        "note": "Parent GO sector 3251 — basic organic chemicals bucket",
    },
    {
        "type": "single",
        "naics_code": "3253", "measure": "Ammonia",
        "cims_sector": "Ammonia Methanol",
        "go_sector": "3253",
        "note": "NAICS 3253 (fertilizers & agricultural chemicals); parent GO sector 3253",
    },
    # ── Non-metallic minerals ────────────────────────────────────────────────
    {
        "type": "single",
        "naics_code": "32731", "measure": "Cement",
        "cims_sector": "Cement",
        "go_sector": "3273",
        "note": "Direct GO match",
    },
    {
        "type": "single",
        "naics_code": "32741", "measure": "Lime",
        "cims_sector": "Lime",
        "go_sector": "327A0",
        "note": "Lime inside GO aggregate 327A0 (non-metallic excl. cement)",
    },
    {
        "type": "direct_sum",
        "cims_sector": "Industrial Minerals",
        "naics_code": "Cement and Lime",
        "note": "Direct sum of Cement (32731) + Lime (32741) provincial values; "
                "no GO share applied — provincial values are literal sums of components",
        "components": [
            {"cims_sector": "Cement"},
            {"cims_sector": "Lime"},
        ],
    },
    # ── Iron & Steel ─────────────────────────────────────────────────────────
    {
        "type": "single",
        "naics_code": "3311",   "measure": "Steel, primary forms",
        "cims_sector": "Iron and Steel",
        "go_sector": "3311",
        "note": "Direct GO match",
    },
    # ── Primary & Non-ferrous metals ─────────────────────────────────────────
    {
        "type": "single",
        "naics_code": "331313", "measure": "Molten aluminium",
        "cims_sector": "Aluminum",
        "go_sector": "3313",
        "note": "Parent GO sector 3313 (aluminum)",
    },
    {
        "type": "direct_sum",
        "cims_sector": "Metal Smelting",
        "naics_code": "Aluminum+Copper+Lead+Magnesium+Nickel+Zinc",
        "note": "Direct sum of six individual metal provincial values; "
                "each metal disaggregated via its own GO sector share",
        "components": [
            {"cims_sector": "Aluminum"},
            {"cims_sector": "Copper"},
            {"cims_sector": "Lead"},
            {"cims_sector": "Magnesium"},
            {"cims_sector": "Nickel"},
            {"cims_sector": "Zinc"},
        ],
    },
    # ── 33141 sub-products (individual metals) ───────────────────────────────
    {
        "type": "single",
        "naics_code": "33141",  "measure": "Primary production - Copper",
        "cims_sector": "Copper",
        "go_sector": "3314",
        "note": "Parent GO sector 3314; no finer GO series available for individual metals",
    },
    {
        "type": "single",
        "naics_code": "33141",  "measure": "Secondary refined production - Lead",
        "cims_sector": "Lead",
        "go_sector": "3314",
        "note": "Parent GO sector 3314; no finer GO series available for individual metals",
    },
    {
        "type": "single",
        "naics_code": "33141",  "measure": "Refined production - Magnesium",
        "cims_sector": "Magnesium",
        "go_sector": "3314",
        "note": "Parent GO sector 3314; no finer GO series available for individual metals",
    },
    {
        "type": "single",
        "naics_code": "33141",  "measure": "Primary production - Nickel",
        "cims_sector": "Nickel",
        "go_sector": "3314",
        "note": "Parent GO sector 3314; no finer GO series available for individual metals",
    },
    {
        "type": "single",
        "naics_code": "33141",  "measure": "Primary production - Zinc",
        "cims_sector": "Zinc",
        "go_sector": "3314",
        "note": "Parent GO sector 3314; no finer GO series available for individual metals",
    },
]

# =============================================================================
# HIERARCHY
# =============================================================================
HIERARCHY = {
    "Mining":              ["Iron Product", "Non-Iron Product", "Potash"],
    "Pulp and Paper":      ["Pulp", "Uncoated, coated, tissue", "Newsprint", "Linerboard"],
    "Chemical Product":    ["Other Petrochemicals",
                            "Chlor Alkali, Hydrogen Peroxide, Sodium Chlorate",
                            "Adipic Acid", "Ammonia Methanol"],
    "Industrial Minerals": ["Cement", "Lime"],
    "Iron and Steel":      [],
    "Metal Smelting":      ["Aluminum", "Copper", "Lead", "Magnesium", "Nickel", "Zinc"],
}

# =============================================================================
# INORGANIC SUB-CHEMICAL SPLITS
# =============================================================================

# Province's share of national capacity for each sub-chemical (sums to 1.0)
INORGANIC_CAPACITY_SHARES: dict[str, dict[str, float]] = {
    "Chlor Alkali": {
        "Ontario":            0.45,
        "Quebec":             0.45,
        "British Columbia":   0.10,
    },
    "Hydrogen Peroxide": {
        "Quebec":             0.55,
        "Alberta":            0.45,
    },
    "Sodium Chlorate": {
        "British Columbia":   0.60,
        "Quebec":             0.30,
        "New Brunswick":      0.10,
    },
}

# Each chemical's share of total NAICS 32518 output nationally (sums to 1.0)
INORGANIC_NATIONAL_FRACTIONS: dict[str, float] = {
    "Chlor Alkali":      0.60,
    "Hydrogen Peroxide": 0.15,
    "Sodium Chlorate":   0.25,
}

# Provinces to exclude from disaggregation for specific CIMS sectors.
PROVINCE_EXCLUSIONS: dict[str, set[str]] = {
    "Iron and Steel":     {"New Brunswick", "Nova Scotia"},
    "Industrial Minerals": {"Yukon", "Prince Edward Island", "Manitoba", "New Brunswick", "Northwest Territories"},
    "Chemical Product":   {"Yukon", "Northwest Territories", "Nunavut"},
    "Metal Smelting":     {"Nova Scotia", "Saskatchewan"},
}


def make_annual_index(df: pl.DataFrame, label_col: str, label_val: str,
                      fill_start_year: int | None = None) -> pl.DataFrame:
    """
    Filter a monthly price index to one series, average to annual,
    and rebase to BASE_YEAR = 100.

    Returns a Polars DataFrame with columns [YEAR, PRICE_INDEX_REBASED].

    fill_start_year: back-extend to this year by holding the earliest
    available index value constant (needed for RMPI Metal ores / Potash
    which only start in 2010).
    """
    series = df.filter(pl.col(label_col) == label_val)
    if series.is_empty():
        raise ValueError(f"Label not found in index file: '{label_val}'")

    # REF_DATE is a monthly string like "2010-01"; extract the year part.
    annual = (
        series
        .with_columns(pl.col("REF_DATE").str.slice(0, 4).cast(pl.Int32).alias("YEAR"))
        .group_by("YEAR")
        .agg(pl.col("VALUE").mean().alias("PRICE_INDEX"))
        .sort("YEAR")
    )

    if fill_start_year is not None:
        first_year = annual["YEAR"].min()
        if first_year > fill_start_year:
            first_val = annual.filter(pl.col("YEAR") == first_year)["PRICE_INDEX"][0]
            gap = pl.DataFrame({
                "YEAR":        list(range(fill_start_year, first_year)),
                "PRICE_INDEX": [first_val] * (first_year - fill_start_year),
            }).with_columns(pl.col("YEAR").cast(pl.Int32))
            annual = pl.concat([gap, annual]).sort("YEAR")

    base_rows = annual.filter(pl.col("YEAR") == BASE_YEAR)
    if base_rows.is_empty():
        raise ValueError(f"Base year {BASE_YEAR} not found in series for '{label_val}'")
    base_val = base_rows["PRICE_INDEX"][0]

    annual = annual.with_columns(
        (pl.col("PRICE_INDEX") / base_val * 100).alias("PRICE_INDEX_REBASED")
    )
    return annual.select(["YEAR", "PRICE_INDEX_REBASED"])


def _parse_period_str(s: str) -> tuple[int, int]:
    a, b = s.strip().split('-')
    return int(a), int(b)


def _parse_pct_str(s: str) -> float:
    return float(str(s).strip().replace('%', '')) / 100.0


def main() -> pl.DataFrame:
    """Build heavy industry DataFrame and write to CSV."""

    # =============================================================================
    # STEP 1 — LOAD RAW DATA
    # =============================================================================
    go_raw   = read_statscan_csv(GO_FILE,   log_suppressed=True)
    ippi_raw = read_statscan_csv(IPPI_FILE, log_suppressed=True)
    rmpi_raw = read_statscan_csv(RMPI_FILE, log_suppressed=True)

    prod_pd  = pd.read_excel(PRODUCTION_FILE, dtype=str)
    prod_raw = pl.from_pandas(prod_pd)

    # ── GO: flag and remove suppressed values ('x', '..', 'F', etc.) ─────────────
    go_value_numeric = pl.Series("VALUE_num", go_raw["VALUE"]).cast(pl.Float64, strict=False)
    suppressed_mask  = go_value_numeric.is_null() & go_raw["VALUE"].is_not_null()

    n_suppressed = suppressed_mask.sum()
    _go_suppressed = pl.DataFrame({
        "GEO":          pl.Series([], dtype=pl.Utf8),
        "NAICS_SECTOR": pl.Series([], dtype=pl.Utf8),
        "YEAR":         pl.Series([], dtype=pl.Int32),
    })
    if n_suppressed > 0:
        suppressed_df = go_raw.filter(suppressed_mask)

        _go_suppressed = (
            suppressed_df
            .with_columns([
                pl.col("REF_DATE").cast(pl.Int32).alias("YEAR"),
                pl.col("Industry").str.extract(r"\[BS(\w+)\]", 1).alias("GO_CODE"),
            ])
            .filter(pl.col("GEO") != "Canada")
            .with_columns(
                pl.col("GO_CODE").map_elements(
                    lambda c: NAICS_LABEL_MAP.get(c, c), return_dtype=pl.Utf8
                ).alias("NAICS_SECTOR")
            )
            .select(["GEO", "NAICS_SECTOR", "YEAR"])
        )

    go_raw = (
        go_raw
        .with_columns(go_value_numeric.alias("VALUE"))
        .filter(pl.col("VALUE").is_not_null())
    )

    # ── IPPI / RMPI: same suppression handling ────────────────────────────────────
    for name, df_ref in [("IPPI", ippi_raw), ("RMPI", rmpi_raw)]:
        num = pl.Series("VALUE_num", df_ref["VALUE"]).cast(pl.Float64, strict=False)
        n   = (num.is_null() & df_ref["VALUE"].is_not_null()).sum()
        if name == "IPPI":
            ippi_raw = df_ref.with_columns(num.alias("VALUE")).filter(pl.col("VALUE").is_not_null())
        else:
            rmpi_raw = df_ref.with_columns(num.alias("VALUE")).filter(pl.col("VALUE").is_not_null())

    # ── Extract NAICS / GO codes from StatsCan label strings ─────────────────────
    go_raw = go_raw.with_columns(
        pl.col("Industry").str.extract(r"\[BS(\w+)\]", 1).alias("GO_CODE")
    )

    ippi_col = "North American Industry Classification System (NAICS)"
    ippi_raw = ippi_raw.with_columns(
        pl.col(ippi_col).str.extract(r"\[(\w+)\]", 1).alias("NAICS_CODE")
    )

    rmpi_col = "North American Product Classification System (NAPCS)"
    rmpi_raw = rmpi_raw.with_columns(
        pl.col(rmpi_col).str.extract(r"\[(\w+)\]", 1).alias("NAPCS_CODE")
    )

    # ── Production file cleanup ───────────────────────────────────────────────────
    prod_raw = prod_raw.with_columns([
        pl.col("naics_code").cast(pl.Utf8),
        pl.col("year").cast(pl.Int32, strict=False),
        pl.col("value").cast(pl.Float64, strict=False),
    ])

    # Build one annual index per unique series (cached to avoid redundant work)
    ippi_cache: dict[str, pl.DataFrame] = {}
    rmpi_cache: dict[str, pl.DataFrame] = {}

    for go_code, mapping in DEFLATOR_MAP.items():
        lbl = mapping["label"]
        if mapping["source"] == "IPPI" and lbl not in ippi_cache:
            ippi_cache[lbl] = make_annual_index(ippi_raw, ippi_col, lbl)
        elif mapping["source"] == "RMPI" and lbl not in rmpi_cache:
            rmpi_cache[lbl] = make_annual_index(rmpi_raw, rmpi_col, lbl, fill_start_year=2000)

    deflated_frames: list[pl.DataFrame] = []

    for go_code, mapping in DEFLATOR_MAP.items():
        lbl    = mapping["label"]
        source = mapping["source"]
        idx_df = ippi_cache[lbl] if source == "IPPI" else rmpi_cache[lbl]

        subset = (
            go_raw
            .filter(pl.col("GO_CODE") == go_code)
            .with_columns(pl.col("REF_DATE").cast(pl.Int32).alias("YEAR"))
        )
        merged = subset.join(idx_df, on="YEAR", how="left")

        merged = merged.with_columns([
            (pl.col("VALUE") / (pl.col("PRICE_INDEX_REBASED") / 100)).alias("REAL_GO"),
            pl.lit(NAICS_LABEL_MAP.get(go_code, go_code)).alias("NAICS_SECTOR"),
            pl.lit(source).alias("DEFLATOR_SOURCE"),
            pl.lit(lbl).alias("DEFLATOR_SERIES"),
        ])
        deflated_frames.append(merged)

    real_go = pl.concat(deflated_frames)

    real_go_clean = (
        real_go
        .select([
            "REF_DATE", "GEO", "Industry", "NAICS_SECTOR",
            "VALUE", "REAL_GO", "PRICE_INDEX_REBASED", "DEFLATOR_SOURCE", "DEFLATOR_SERIES",
        ])
        .rename({
            "REF_DATE": "YEAR",
            "VALUE":    "NOMINAL_GO_MILLIONS",
            "REAL_GO":  "REAL_GO_MILLIONS_2020",
        })
        .with_columns(pl.col("YEAR").cast(pl.Int32))
        .sort(["YEAR", "NAICS_SECTOR", "GEO"])
    )

    national_go = (
        real_go_clean
        .filter(pl.col("GEO") == "Canada")
        .select(["YEAR", "NAICS_SECTOR", "REAL_GO_MILLIONS_2020"])
        .rename({"REAL_GO_MILLIONS_2020": "NATIONAL_REAL_GO"})
    )

    provinces_go = real_go_clean.filter(pl.col("GEO") != "Canada")

    shares = (
        provinces_go
        .join(national_go, on=["YEAR", "NAICS_SECTOR"], how="left")
        .with_columns(
            (pl.col("REAL_GO_MILLIONS_2020") / pl.col("NATIONAL_REAL_GO")).alias("PROVINCIAL_SHARE")
        )
        .select([
            "YEAR", "GEO", "NAICS_SECTOR", "Industry",
            "NOMINAL_GO_MILLIONS", "REAL_GO_MILLIONS_2020",
            "NATIONAL_REAL_GO", "PROVINCIAL_SHARE",
            "DEFLATOR_SOURCE", "DEFLATOR_SERIES",
        ])
        .sort(["YEAR", "NAICS_SECTOR", "GEO"])
    )

    # Sanity check: shares sum to ~1.0 per sector/year
    share_check = (
        shares
        .group_by(["YEAR", "NAICS_SECTOR"])
        .agg(pl.col("PROVINCIAL_SHARE").sum().alias("SUM_OF_SHARES"))
    )
    bad_shares = share_check.filter((pl.col("SUM_OF_SHARES") - 1.0).abs() > 0.01)

    shares_2022 = shares.filter(pl.col("YEAR") == 2022)
    extended    = pl.concat([
        shares,
        shares_2022.with_columns(pl.lit(2023).cast(pl.Int32).alias("YEAR")),
        shares_2022.with_columns(pl.lit(2024).cast(pl.Int32).alias("YEAR")),
    ])
    shares = extended

    _hvy_config    = load_control_config()
    LAST_DATA_YEAR = _hvy_config["last_data_year"]
    ANNUAL_YEARS   = list(range(DATA_START, LAST_DATA_YEAR["ceedc"] + 1))

    def build_combined_share(go_sectors: list[str]) -> pl.DataFrame:
        """
        Combined provincial share for a list of GO sectors, weighted by real GO.
        Returns a Polars DataFrame with columns [YEAR, GEO, PROVINCIAL_SHARE].
        """
        by_prov = (
            shares
            .filter(pl.col("NAICS_SECTOR").is_in(go_sectors))
            .group_by(["YEAR", "GEO"])
            .agg(pl.col("REAL_GO_MILLIONS_2020").sum().alias("prov_go"))
        )
        nat_tot = (
            by_prov
            .group_by("YEAR")
            .agg(pl.col("prov_go").sum().alias("nat_go"))
        )
        return (
            by_prov
            .join(nat_tot, on="YEAR")
            .with_columns((pl.col("prov_go") / pl.col("nat_go")).alias("PROVINCIAL_SHARE"))
            .select(["YEAR", "GEO", "PROVINCIAL_SHARE"])
        )

    combined_share_cache: dict[tuple, pl.DataFrame] = {}
    for sector in SECTORS:
        if sector["type"] == "combined":
            go_sectors = tuple(sorted(set(c["go_sector"] for c in sector["components"])))
            if go_sectors not in combined_share_cache:
                combined_share_cache[go_sectors] = build_combined_share(list(go_sectors))

    def get_series(naics_code: str, measure: str) -> dict[int, float]:
        """Return year->value dict for one NAICS code + measure from Production.xlsx."""
        mask = pl.col("naics_code") == naics_code
        if measure:
            mask = mask & (pl.col("measure") == measure)
        sub = prod_raw.filter(mask).filter(pl.col("value").is_not_null()).select(["year", "value"])
        return dict(zip(sub["year"].to_list(), sub["value"].to_list()))

    national_rows: list[dict] = []

    for sector in SECTORS:
        if sector["type"] == "direct_sum":
            continue

        if sector["type"] == "single":
            series = get_series(sector["naics_code"], sector["measure"])
            if not series:
                print(f"  WARNING: No data for {sector['naics_code']} / {sector['measure']}")
                continue
            naics_label   = sector["naics_code"]
            measure_label = sector["measure"]
            go_label      = sector["go_sector"]

        elif sector["type"] == "derived":
            minuend_s    = get_series(sector["minuend"]["naics_code"],    sector["minuend"]["measure"])
            subtrahend_s = get_series(sector["subtrahend"]["naics_code"], sector["subtrahend"]["measure"])
            series = {yr: minuend_s[yr] - subtrahend_s.get(yr, 0) for yr in minuend_s}
            naics_label   = f"{sector['minuend']['naics_code']} - {sector['subtrahend']['naics_code']}"
            measure_label = (f"{sector['minuend']['measure']} minus "
                             f"{sector['subtrahend']['measure']}")
            go_label      = sector["minuend"]["go_sector"]

        else:  # combined
            combined: dict[int, float] = {}
            naics_parts, measure_parts, go_parts = [], [], []
            for comp in sector["components"]:
                s = get_series(comp["naics_code"], comp["measure"])
                for yr, val in s.items():
                    combined[yr] = combined.get(yr, 0) + val
                naics_parts.append(comp["naics_code"])
                measure_parts.append(comp["measure"])
                go_parts.append(comp["go_sector"])
            series        = combined
            naics_label   = sector.get("naics_desc", " + ".join(naics_parts))
            measure_label = " + ".join(measure_parts)
            go_label      = " + ".join(sorted(set(go_parts))) + " (weighted)"

        for yr in ANNUAL_YEARS:
            national_rows.append({
                "YEAR":            yr,
                "NAICS_CODE":      naics_label,
                "MEASURE":         measure_label,
                "CIMS_SECTOR":     sector["cims_sector"],
                "UNIT":            "tonnes",
                "PHYSICAL_OUTPUT": series.get(yr, None),
                "GO_SECTOR_USED":  go_label,
                "NOTE":            sector["note"],
                "SOURCE":          "CEEDC Production.xlsx (annual)",
            })

    national_df = (
        pl.DataFrame(national_rows)
        .sort(["CIMS_SECTOR", "YEAR"])
    )

    # ── direct_sum sectors: national totals ──────────────────────────────────────
    direct_sum_prov_rows: list[dict] = []

    for sector in SECTORS:
        if sector["type"] != "direct_sum":
            continue
        comp_sectors = [c["cims_sector"] for c in sector["components"]]

        for yr in ANNUAL_YEARS:
            comp_vals = (
                national_df
                .filter(
                    pl.col("CIMS_SECTOR").is_in(comp_sectors) &
                    (pl.col("YEAR") == yr)
                )["PHYSICAL_OUTPUT"]
                .drop_nulls()
            )
            nat_val = comp_vals.sum() if not comp_vals.is_empty() else None

            national_rows.append({
                "YEAR":            yr,
                "NAICS_CODE":      sector["naics_code"],
                "MEASURE":         " + ".join(comp_sectors),
                "CIMS_SECTOR":     sector["cims_sector"],
                "UNIT":            "tonnes",
                "PHYSICAL_OUTPUT": nat_val,
                "GO_SECTOR_USED":  "direct sum of components",
                "NOTE":            sector["note"],
                "SOURCE":          "CEEDC Production.xlsx (annual)",
            })

    # Rebuild national_df to include direct_sum rows
    national_df = pl.DataFrame(national_rows).sort(["CIMS_SECTOR", "YEAR"])

    def get_prov_shares(go_sector_used: str, year: int,
                        cims_sector: str = "") -> dict[str, float]:
        """
        Return GEO -> share dict, with suppressed provinces filled by
        interpolating from their nearest available GO share years.
        Any remaining gap after interpolation is redistributed proportionally.
        Provinces in PROVINCE_EXCLUSIONS[cims_sector] are dropped before rescaling.
        """
        if "(weighted)" in go_sector_used:
            codes    = tuple(sorted(go_sector_used.replace(" (weighted)", "").split(" + ")))
            all_rows = combined_share_cache[codes]
        else:
            all_rows = (
                shares
                .filter(
                    (pl.col("NAICS_SECTOR") == go_sector_used) &
                    (pl.col("GEO") != "Canada")
                )
                .select(["YEAR", "GEO", "PROVINCIAL_SHARE"])
            )

        all_geos = all_rows["GEO"].unique().to_list()
        yr_rows = all_rows.filter(pl.col("YEAR") == year)
        raw     = dict(zip(yr_rows["GEO"].to_list(), yr_rows["PROVINCIAL_SHARE"].to_list()))

        for geo in all_geos:
            if geo in raw:
                continue

            geo_series = (
                all_rows
                .filter(pl.col("GEO") == geo)
                .sort("YEAR")
            )
            years  = geo_series["YEAR"].to_list()
            values = geo_series["PROVINCIAL_SHARE"].to_list()

            before = [(y, v) for y, v in zip(years, values) if y < year]
            after  = [(y, v) for y, v in zip(years, values) if y > year]

            if before and after:
                y0, v0 = before[-1]
                y1, v1 = after[0]
                weight = (year - y0) / (y1 - y0)
                estimated = v0 + weight * (v1 - v0)
            elif before:
                estimated = before[-1][1]
            elif after:
                estimated = after[0][1]
            else:
                continue

            raw[geo] = estimated

        for excl in PROVINCE_EXCLUSIONS.get(cims_sector, set()):
            raw.pop(excl, None)

        total = sum(raw.values())
        if total > 0:
            raw = {geo: share / total for geo, share in raw.items()}

        return raw

    provincial_rows: list[dict] = []

    for nat_row in national_df.to_dicts():
        yr      = nat_row["YEAR"]
        nat_val = nat_row["PHYSICAL_OUTPUT"]

        if nat_val is None:
            continue
        if nat_row["GO_SECTOR_USED"] == "direct sum of components":
            continue

        prov_shares = get_prov_shares(nat_row["GO_SECTOR_USED"], yr, nat_row["CIMS_SECTOR"])

        if not prov_shares:
            print(f"  WARNING: No GO shares for {nat_row['CIMS_SECTOR']} / "
                  f"{nat_row['GO_SECTOR_USED']} / {yr}")
            continue

        canada_row = dict(nat_row)
        canada_row["PHYSICAL_OUTPUT"] = round(nat_val * 1000, 4)
        provincial_rows.append({**canada_row, "GEO": "Canada", "PROVINCIAL_SHARE": 1.0})

        for geo, share in prov_shares.items():
            row = dict(nat_row)
            row["GEO"]              = geo
            row["PHYSICAL_OUTPUT"]  = round(nat_val * share * 1000, 4)
            row["PROVINCIAL_SHARE"] = round(share, 6)
            provincial_rows.append(row)

    # ── direct_sum sectors: provincial rows ──────────────────────────────────────
    for sector in SECTORS:
        if sector["type"] != "direct_sum":
            continue
        comp_sectors = [c["cims_sector"] for c in sector["components"]]

        for yr in ANNUAL_YEARS:
            nat_val = next(
                (r["PHYSICAL_OUTPUT"] for r in national_rows
                 if r["CIMS_SECTOR"] == sector["cims_sector"] and r["YEAR"] == yr),
                None,
            )
            canada_val = round(nat_val * 1000, 4) if nat_val is not None else None

            comp_prov: dict[str, float] = {}
            for cs in comp_sectors:
                for r in provincial_rows:
                    if r.get("CIMS_SECTOR") == cs and r.get("YEAR") == yr and r.get("GEO") != "Canada":
                        geo = r["GEO"]
                        comp_prov[geo] = comp_prov.get(geo, 0.0) + r["PHYSICAL_OUTPUT"]

            for excl in PROVINCE_EXCLUSIONS.get(sector["cims_sector"], set()):
                comp_prov.pop(excl, None)

            if canada_val:
                prov_sum = sum(comp_prov.values())
                if prov_sum:
                    scale = canada_val / prov_sum
                    comp_prov = {geo: val * scale for geo, val in comp_prov.items()}

            direct_sum_prov_rows.append({
                "YEAR": yr, "NAICS_CODE": sector["naics_code"],
                "MEASURE": " + ".join(comp_sectors),
                "CIMS_SECTOR": sector["cims_sector"], "UNIT": "tonnes",
                "PHYSICAL_OUTPUT": canada_val, "GO_SECTOR_USED": "direct sum of components",
                "NOTE": sector["note"], "SOURCE": "CEEDC Production.xlsx (annual)",
                "GEO": "Canada", "PROVINCIAL_SHARE": 1.0,
            })
            for geo, val in comp_prov.items():
                direct_sum_prov_rows.append({
                    "YEAR": yr, "NAICS_CODE": sector["naics_code"],
                    "MEASURE": " + ".join(comp_sectors),
                    "CIMS_SECTOR": sector["cims_sector"], "UNIT": "tonnes",
                    "PHYSICAL_OUTPUT": round(val, 4),
                    "GO_SECTOR_USED": "direct sum of components",
                    "NOTE": sector["note"], "SOURCE": "CEEDC Production.xlsx (annual)",
                    "GEO": geo,
                    "PROVINCIAL_SHARE": round(val / canada_val, 6) if canada_val else None,
                })

    provincial_df = (
        pl.concat([
            pl.DataFrame(provincial_rows),
            pl.DataFrame(direct_sum_prov_rows),
        ])
        .with_columns([
            pl.col("PHYSICAL_OUTPUT").cast(pl.Float64),
            pl.col("PROVINCIAL_SHARE").cast(pl.Float64),
        ])
        .sort(["CIMS_SECTOR", "YEAR", "GEO"])
    )

    # ── Interpolate mid-series gaps in PHYSICAL_OUTPUT ────────────────────────
    _geo_sector_grid = provincial_df.select(["GEO", "CIMS_SECTOR"]).unique()
    _year_grid       = pl.DataFrame({"YEAR": ANNUAL_YEARS})

    provincial_slim = (
        _geo_sector_grid
        .join(_year_grid, how="cross")
        .join(
            provincial_df.select(["GEO", "CIMS_SECTOR", "YEAR", "PHYSICAL_OUTPUT"]),
            on=["GEO", "CIMS_SECTOR", "YEAR"],
            how="left",
        )
    )

    provincial_slim = interpolate_gaps(
        provincial_slim,
        group_cols=["GEO", "CIMS_SECTOR"],
        year_col="YEAR",
        value_col="PHYSICAL_OUTPUT",
    )

    prov_sum = (
        provincial_df
        .filter(
            (pl.col("GEO") != "Canada") &
            pl.col("PHYSICAL_OUTPUT").is_not_null()
        )
        .group_by(["CIMS_SECTOR", "YEAR"])
        .agg(pl.col("PHYSICAL_OUTPUT").sum().alias("PROV_SUM"))
    )
    canada_vals = (
        provincial_df
        .filter(
            (pl.col("GEO") == "Canada") &
            pl.col("PHYSICAL_OUTPUT").is_not_null()
        )
        .select(["CIMS_SECTOR", "YEAR", "PHYSICAL_OUTPUT"])
        .rename({"PHYSICAL_OUTPUT": "CANADA_TOTAL"})
    )
    check = (
        prov_sum
        .join(canada_vals, on=["CIMS_SECTOR", "YEAR"], how="inner")
        .with_columns(
            ((pl.col("PROV_SUM") - pl.col("CANADA_TOTAL")).abs()
             / pl.col("CANADA_TOTAL") * 100).alias("DIFF_PCT")
        )
    )
    bad = check.filter(pl.col("DIFF_PCT") > 0.1)

    # ── Build output: Region | Variable | Unit | Year | Value ─────────────────
    _child_to_parent = {
        child: parent
        for parent, children in HIERARCHY.items()
        for child in children
    }
    _all_parents  = set(HIERARCHY.keys())
    _all_children = set(_child_to_parent.keys())

    _proj_assumptions = pd.read_csv(CAGR_ASSUMPTIONS_FILE)
    _all_row = _proj_assumptions[_proj_assumptions['Sector'] == 'Heavy Industry'].iloc[0]

    CAGR_START, CAGR_END = _parse_period_str(_all_row['CAGR Period'])

    CAGR_PERIODS = [
        (*_parse_period_str(_all_row['1st period']), _parse_pct_str(_all_row['1st period dampener'])),
        (*_parse_period_str(_all_row['2nd period']), _parse_pct_str(_all_row['2nd period dampener'])),
        (*_parse_period_str(_all_row['3d period']),  _parse_pct_str(_all_row['3rd period dampener'])),
    ]
    EXTEND_TO = max(end for _, end, _ in CAGR_PERIODS)

    CAGR_OVERRIDES: dict[tuple[str, str], tuple[float, float, float]] = {
        ("SK", "Mining"):           (0.003, 0.002, 0.001),
        ("NU", "Mining"):           (0.003, 0.002, 0.001),
        ("NL", "Chemical Product"): (0.02, 0.005, 0.001),
        ("NS", "Chemical Product"): (0.02, 0.005, 0.001),
        ("PE", "Chemical Product"): (0.02, 0.005, 0.001),
        ("SK", "Chemical Product"): (0.02, 0.005, 0.001),
        ("AB", "Mining"):           (0.04, 0.02, 0.005),
        ("QC", "Mining"):           (0.02, 0.005, 0.001),
    }

    # ── Historical parent tonnes (2000-2024), provinces only ──────────────────
    _region_map = pl.read_csv(REGION_MAP_FILE, columns=["CIMS", "NIR"])

    _parent_hist = (
        provincial_slim
        .filter(
            pl.col("CIMS_SECTOR").is_in(list(_all_parents)) &
            (pl.col("GEO") != "Canada")
        )
        .rename({
            "GEO":             "Region",
            "CIMS_SECTOR":     "Variable",
            "YEAR":            "Year",
            "PHYSICAL_OUTPUT": "Value",
        })
        .with_columns([
            pl.lit("tonnes").alias("Unit"),
            pl.col("Year").cast(pl.Int64),
        ])
        .select(["Region", "Variable", "Unit", "Year", "Value"])
        .join(_region_map, left_on="Region", right_on="NIR", how="left")
        .with_columns(pl.col("CIMS").fill_null(pl.col("Region")).alias("Region"))
        .drop("CIMS")
    )

    # ── CAGR per region×variable anchored on 2014→2024 ────────────────────────
    _cagr_anchor = (
        _parent_hist
        .filter(pl.col("Year").is_in([CAGR_START, CAGR_END]))
        .pivot(
            values             = "Value",
            index              = ["Region", "Variable", "Unit"],
            on                 = "Year",
            aggregate_function = "first",
        )
        .rename({str(CAGR_START): "val_start", str(CAGR_END): "val_end"})
        .with_columns(
            pl.when(
                pl.col("val_start").is_null() |
                (pl.col("val_start") == 0.0) |
                pl.col("val_end").is_null()
            )
            .then(0.0)
            .otherwise(
                (
                    (pl.col("val_end") / pl.col("val_start"))
                    .pow(1.0 / (CAGR_END - CAGR_START))
                    - 1.0
                )
            )
            .alias("cagr")
        )
        .select([
            "Region", "Variable", "Unit",
            pl.col("val_end").fill_null(0.0).alias("base_val"),
            "cagr",
        ])
    )

    _anchor_map = {(r["Region"], r["Variable"]): r for r in _cagr_anchor.to_dicts()}

    _parent_ext_rows = []
    for (region, variable), anc in _anchor_map.items():
        ext = extend_cagr_periods(
            base_val = anc["base_val"],
            raw_cagr = anc["cagr"],
            periods  = CAGR_PERIODS,
            override = CAGR_OVERRIDES.get((region, variable)),
        )
        for yr, val in ext.items():
            _parent_ext_rows.append({
                "Region": region, "Variable": variable,
                "Unit": "tonnes", "Year": yr, "Value": val,
            })

    _parent_ext = pl.DataFrame(_parent_ext_rows).with_columns(pl.col("Year").cast(pl.Int64))
    _parent_out = pl.concat([_parent_hist, _parent_ext]).sort(["Region", "Variable", "Year"])

    # ── Subnode ratios: GO-weighted, normalised per parent×province×year ──────

    def _suppressed_nulls(naics_sectors: list[str]) -> pl.DataFrame:
        """Return (GEO, YEAR, PROVINCIAL_SHARE=null, CIMS_SECTOR) rows for suppressed
        province-years, so they enter the interpolation grid rather than being absent."""
        return (
            _go_suppressed
            .filter(pl.col("NAICS_SECTOR").is_in(naics_sectors))
            .select(["GEO", "YEAR"])
            .unique()
            .with_columns(pl.lit(None).cast(pl.Float64).alias("PROVINCIAL_SHARE"))
        )

    _child_go_parts: list[pl.DataFrame] = []
    for _sec in SECTORS:
        _cims = _sec["cims_sector"]
        if _cims not in _all_children:
            continue
        if _sec["type"] == "single":
            _go_sector = _sec["go_sector"]
            _sdf = (
                shares
                .filter(pl.col("NAICS_SECTOR") == _go_sector)
                .select(["GEO", "YEAR", "PROVINCIAL_SHARE"])
            )
            _supp = _suppressed_nulls([_go_sector])
        elif _sec["type"] == "derived":
            _go_sector = _sec["minuend"]["go_sector"]
            _sdf = (
                shares
                .filter(pl.col("NAICS_SECTOR") == _go_sector)
                .select(["GEO", "YEAR", "PROVINCIAL_SHARE"])
            )
            _supp = _suppressed_nulls([_go_sector])
        else:  # combined
            _go_secs = tuple(sorted(set(c["go_sector"] for c in _sec["components"])))
            _sdf = (
                combined_share_cache[_go_secs]
                .select(["GEO", "YEAR", "PROVINCIAL_SHARE"])
            )
            _supp = _suppressed_nulls(list(_go_secs))
        _child_go_parts.append(
            pl.concat([_sdf, _supp])
            .with_columns(pl.lit(_cims).alias("CIMS_SECTOR"))
        )

    _child_go_long = (
        pl.concat(_child_go_parts)
        .filter(pl.col("YEAR").is_in(ANNUAL_YEARS))
        .group_by(["CIMS_SECTOR", "GEO", "YEAR"])
        .agg(pl.col("PROVINCIAL_SHARE").max())
        .select(["CIMS_SECTOR", "GEO", "YEAR", "PROVINCIAL_SHARE"])
    )

    _year_grid_go = pl.DataFrame({"YEAR": ANNUAL_YEARS}).with_columns(pl.col("YEAR").cast(pl.Int32))

    _child_go_full = (
        _child_go_long
        .select(["CIMS_SECTOR", "GEO"])
        .unique()
        .join(_year_grid_go, how="cross")
        .join(
            _child_go_long.select(["CIMS_SECTOR", "GEO", "YEAR", "PROVINCIAL_SHARE"]),
            on=["CIMS_SECTOR", "GEO", "YEAR"],
            how="left",
        )
        .sort(["CIMS_SECTOR", "GEO", "YEAR"])
    )

    _nonzero_bounds = (
        _child_go_full
        .filter(pl.col("PROVINCIAL_SHARE").is_not_null() & (pl.col("PROVINCIAL_SHARE") > 0))
        .group_by(["CIMS_SECTOR", "GEO"])
        .agg(
            pl.col("YEAR").min().alias("first_nonzero"),
            pl.col("YEAR").max().alias("last_nonzero"),
        )
    )
    _child_go_full = (
        _child_go_full
        .join(_nonzero_bounds, on=["CIMS_SECTOR", "GEO"], how="left")
        .with_columns(
            pl.when(
                pl.col("PROVINCIAL_SHARE").is_not_null() &
                (pl.col("PROVINCIAL_SHARE") == 0.0) &
                pl.col("first_nonzero").is_not_null() &
                pl.col("YEAR").is_between(pl.col("first_nonzero"), pl.col("last_nonzero"))
            )
            .then(pl.lit(None).cast(pl.Float64))
            .otherwise(pl.col("PROVINCIAL_SHARE"))
            .alias("PROVINCIAL_SHARE")
        )
        .drop(["first_nonzero", "last_nonzero"])
    )

    _child_go_interp = interpolate_gaps(
        _child_go_full,
        group_cols=["CIMS_SECTOR", "GEO"],
        year_col="YEAR",
        value_col="PROVINCIAL_SHARE",
    )

    _child_go_interp = (
        _child_go_interp
        .sort(["CIMS_SECTOR", "GEO", "YEAR"])
        .with_columns(
            pl.col("PROVINCIAL_SHARE").forward_fill().over(["CIMS_SECTOR", "GEO"])
        )
        .with_columns(
            pl.col("PROVINCIAL_SHARE").backward_fill().over(["CIMS_SECTOR", "GEO"])
        )
    )

    _still_null = (
        _child_go_interp
        .filter(pl.col("PROVINCIAL_SHARE").is_null())
        .select(["CIMS_SECTOR", "GEO"])
        .unique()
    )
    if len(_still_null) > 0:
        print("  WARNING: the following province×child-sector combinations are entirely")
        print("  suppressed in the GO data (no valid year to interpolate from) and will")
        print("  be assigned a zero subproduct ratio:")
        for row in _still_null.iter_rows(named=True):
            print(f"    {row['GEO']:30s}  {row['CIMS_SECTOR']}")

    _nat_child_kt = (
        national_df
        .filter(pl.col("CIMS_SECTOR").is_in(list(_all_children)))
        .select(["CIMS_SECTOR", "YEAR", "PHYSICAL_OUTPUT"])
        .rename({"PHYSICAL_OUTPUT": "nat_kt"})
    )

    _child_weighted = (
        _nat_child_kt
        .join(_child_go_interp, on=["CIMS_SECTOR", "YEAR"], how="left")
        .with_columns(
            (pl.col("nat_kt").fill_null(0.0) * pl.col("PROVINCIAL_SHARE").fill_null(0.0)).alias("weighted")
        )
        .with_columns(
            pl.col("CIMS_SECTOR").replace_strict(_child_to_parent, default=None).alias("parent_sector")
        )
        .with_columns(
            (pl.col("parent_sector") + "." + pl.col("CIMS_SECTOR")).alias("Variable")
        )
    )

    _parent_totals = (
        _child_weighted
        .group_by(["parent_sector", "GEO", "YEAR"])
        .agg(
            pl.col("weighted").sum().alias("total_weighted"),
            pl.col("PROVINCIAL_SHARE").sum().alias("total_prov_share"),
        )
    )

    _child_hist = (
        _child_weighted
        .join(_parent_totals, on=["parent_sector", "GEO", "YEAR"], how="left")
        .with_columns(
            pl.when(
                pl.col("total_weighted").is_not_null() & (pl.col("total_weighted") > 0.0)
            )
            .then((pl.col("weighted") / pl.col("total_weighted")).clip(0.0, 1.0))
            .when(
                pl.col("total_prov_share").is_not_null() & (pl.col("total_prov_share") > 0.0)
            )
            .then((pl.col("PROVINCIAL_SHARE") / pl.col("total_prov_share")).clip(0.0, 1.0))
            .otherwise(0.0)
            .alias("ratio")
        )
        .rename({"GEO": "Region", "YEAR": "Year"})
        .with_columns([
            pl.lit("% of tonnes").alias("Unit"),
            pl.col("Year").cast(pl.Int64),
        ])
        .select(["Region", "Variable", "Unit", "Year", pl.col("ratio").alias("Value")])
    )

    # ── Pulp and Paper override: ratio = child / parent, not sibling-normalised ──
    _pp_parent_nat = (
        national_df
        .filter(pl.col("CIMS_SECTOR") == "Pulp and Paper")
        .select(["YEAR", pl.col("PHYSICAL_OUTPUT").alias("nat_parent")])
        .with_columns(pl.col("YEAR").cast(pl.Int64))
    )

    _pp_nat_ratios = (
        national_df
        .filter(pl.col("CIMS_SECTOR").is_in(list(HIERARCHY["Pulp and Paper"])))
        .with_columns(pl.col("YEAR").cast(pl.Int64))
        .join(_pp_parent_nat, on="YEAR", how="left")
        .with_columns(
            pl.when(
                pl.col("nat_parent").is_not_null() & (pl.col("nat_parent") != 0.0)
            )
            .then((pl.col("PHYSICAL_OUTPUT") / pl.col("nat_parent")).clip(0.0, 1.0))
            .otherwise(0.0)
            .alias("ratio")
        )
        .select(["CIMS_SECTOR", "YEAR", "ratio"])
    )

    _pp_provinces = (
        shares
        .filter(
            (pl.col("NAICS_SECTOR") == "3221") &
            pl.col("YEAR").is_in(ANNUAL_YEARS)
        )
        .select([pl.col("GEO"), pl.col("YEAR").cast(pl.Int64)])
        .unique()
    )

    _pp_child_override = (
        _pp_provinces
        .join(_pp_nat_ratios, on="YEAR", how="left")
        .with_columns(
            (pl.lit("Pulp and Paper.") + pl.col("CIMS_SECTOR")).alias("Variable")
        )
        .rename({"GEO": "Region", "YEAR": "Year"})
        .with_columns(pl.lit("% of tonnes").alias("Unit"))
        .select(["Region", "Variable", "Unit", "Year", pl.col("ratio").alias("Value")])
    )

    _pp_variables = ["Pulp and Paper." + c for c in HIERARCHY["Pulp and Paper"]]
    _child_hist = pl.concat([
        _child_hist.filter(~pl.col("Variable").is_in(_pp_variables)),
        _pp_child_override,
    ])

    # ── Pulp and Paper sub-product split ──────────────────────────────────────
    PAPER_SPLITS = {"Uncoated": 0.40, "Coated": 0.47, "Tissue": 0.13}

    _paper_combined_var = "Pulp and Paper.Uncoated, coated, tissue"
    _paper_split_df = pl.DataFrame([
        {"sub_paper": name, "split_factor": frac}
        for name, frac in PAPER_SPLITS.items()
    ])

    _paper_override = (
        _child_hist
        .filter(pl.col("Variable") == _paper_combined_var)
        .select(["Region", "Unit", "Year", pl.col("Value").alias("combined_ratio")])
        .join(_paper_split_df, how="cross")
        .with_columns(
            (pl.col("combined_ratio") * pl.col("split_factor")).alias("Value"),
            (pl.lit("Pulp and Paper.") + pl.col("sub_paper")).alias("Variable"),
        )
        .select(["Region", "Variable", "Unit", "Year", "Value"])
    )

    _paper_new_vars = ["Pulp and Paper." + s for s in PAPER_SPLITS]
    _child_hist = pl.concat([
        _child_hist.filter(~pl.col("Variable").is_in([_paper_combined_var] + _paper_new_vars)),
        _paper_override,
    ])

    # ── Inorganic sub-chemicals override ──────────────────────────────────────
    _inorganic_combined_var = "Chemical Product.Chlor Alkali, Hydrogen Peroxide, Sodium Chlorate"

    _all_inorg_regions = (
        _child_hist
        .filter(pl.col("Variable") == _inorganic_combined_var)
        ["Region"].unique().to_list()
    )

    _inorg_split_rows = []
    for _geo in _all_inorg_regions:
        _denom = sum(
            INORGANIC_NATIONAL_FRACTIONS[c] * INORGANIC_CAPACITY_SHARES[c].get(_geo, 0.0)
            for c in INORGANIC_CAPACITY_SHARES
        )
        for _chem in INORGANIC_CAPACITY_SHARES:
            if _denom > 0.0:
                _num = INORGANIC_NATIONAL_FRACTIONS[_chem] * INORGANIC_CAPACITY_SHARES[_chem].get(_geo, 0.0)
                _split = _num / _denom
            else:
                _split = INORGANIC_NATIONAL_FRACTIONS[_chem]
            _inorg_split_rows.append({"Region": _geo, "sub_chem": _chem, "split_factor": _split})

    _inorg_split_df = pl.DataFrame(_inorg_split_rows)

    _inorganic_override = (
        _child_hist
        .filter(pl.col("Variable") == _inorganic_combined_var)
        .select(["Region", "Unit", "Year", pl.col("Value").alias("combined_ratio")])
        .join(_inorg_split_df, on="Region", how="inner")
        .with_columns(
            (pl.col("combined_ratio") * pl.col("split_factor")).alias("Value"),
            (pl.lit("Chemical Product.") + pl.col("sub_chem")).alias("Variable"),
        )
        .select(["Region", "Variable", "Unit", "Year", "Value"])
    )

    _inorganic_new = ["Chemical Product." + c for c in INORGANIC_CAPACITY_SHARES]
    _child_hist = pl.concat([
        _child_hist.filter(~pl.col("Variable").is_in([_inorganic_combined_var] + _inorganic_new)),
        _inorganic_override,
    ])

    # ── Map child regions to CIMS names ──────────────────────────────────────
    _child_hist = (
        _child_hist
        .join(_region_map, left_on="Region", right_on="NIR", how="left")
        .with_columns(pl.col("CIMS").fill_null(pl.col("Region")).alias("Region"))
        .drop("CIMS")
    )

    # ── Extend ratios flat at 2024 value out to 2100 ──────────────────────────
    _ratios_2024 = (
        _child_hist
        .filter(pl.col("Year") == CAGR_END)
        .select(["Region", "Variable", "Unit", "Value"])
    )

    _child_ext_rows = []
    for r in _ratios_2024.to_dicts():
        if r["Value"] is None:
            continue
        for yr in range(CAGR_END + 1, EXTEND_TO + 1):
            _child_ext_rows.append({
                "Region": r["Region"], "Variable": r["Variable"],
                "Unit": r["Unit"], "Year": yr, "Value": r["Value"],
            })

    _child_ext = pl.DataFrame(_child_ext_rows).with_columns(pl.col("Year").cast(pl.Int64))
    _child_out = pl.concat([_child_hist, _child_ext]).sort(["Region", "Variable", "Year"])

    _child_out = (
        _child_out
        .with_columns(
            pl.when(pl.col("Value").is_null() | (pl.col("Value") == 0.0))
            .then(pl.lit(None).cast(pl.Float64))
            .otherwise(pl.col("Value"))
            .alias("_nonzero")
        )
        .with_columns(
            pl.col("_nonzero").forward_fill().over(["Region", "Variable"]).alias("_ffilled")
        )
        .with_columns(
            pl.col("_ffilled").backward_fill().over(["Region", "Variable"]).alias("Value")
        )
        .drop(["_nonzero", "_ffilled"])
    )

    _active_parents = (
        _parent_out
        .filter(pl.col("Value").is_not_null() & (pl.col("Value") > 0))
        .select(["Region", "Variable"])
        .unique()
    )

    _parent_out = _parent_out.join(_active_parents, on=["Region", "Variable"], how="semi")

    _child_out = (
        _child_out
        .with_columns(
            pl.col("Variable").str.split(".").list.get(0).alias("_parent_var")
        )
        .join(
            _active_parents.rename({"Variable": "_parent_var"}),
            on=["Region", "_parent_var"],
            how="semi",
        )
        .drop("_parent_var")
    )

    output_df = (
        pl.concat([_parent_out, _child_out])
        .filter(pl.col("Variable").is_not_null() & (pl.col("Variable") != ""))
        .sort(["Region", "Variable", "Year"])
    )

    output_df = (
        output_df
        .with_columns(
            pl.when(pl.col("Year") <= LAST_DATA_YEAR["ceedc"])
            .then(pl.lit("CEEDC/Stats Can"))
            .otherwise(pl.lit("Assumptions"))
            .alias("Source")
        )
        .select(["Region", "Variable", "Unit", "Source", "Year", "Value"])
    )

    print(f"\n✅ Heavy industry complete")
    print(f"   Total rows:          {len(output_df):,}")
    print(f"   Regions processed:   {output_df['Region'].n_unique()}")
    print(f"   Variables:           {sorted(output_df['Variable'].unique().to_list())}")
    print(f"   Years covered:       {output_df['Year'].min()} – {output_df['Year'].max()}")
    _heavy_industry_path = BASE_PATH / 'processed_data/activity/heavy_industry.csv'
    _heavy_industry_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_pandas().to_csv(
        _heavy_industry_path,
        index=False,
        encoding="utf-8-sig",
        na_rep="",
    )
    print(f"   Saved to:            {_heavy_industry_path}")

    return output_df


if __name__ == '__main__':
    main()
