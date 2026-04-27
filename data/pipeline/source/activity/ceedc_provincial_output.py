"""
ceedc_provincial_output.py
===========================
Single script that runs the full pipeline from raw StatsCan and CEEDC inputs
to annual provincial physical output estimates by CIMS sector.

PIPELINE OVERVIEW:
  Step 1  Load nominal Gross Output (GO) and price indexes (IPPI, RMPI)
  Step 2  Deflate nominal GO to real GO (2020 dollars)
  Step 3  Compute each province's share of national real GO by sector and year
  Step 4  Load CEEDC annual physical output
  Step 5  Extend GO shares to 2023-2024 (hold 2022 constant; GO ends at 2022)
  Step 6  Build weighted combined GO shares for multi-sector CIMS products
  Step 7  Build national physical output table (summing components where needed)
  Step 8  Disaggregate national totals to provinces using GO shares
  Step 9  Sanity checks and save outputs

HOW TO RUN:
    python ceedc_provincial_output.py

INPUTS (place in same folder, or update the FILE PATHS section below):
    3610048801_databaseLoadingData__2_.csv  <- StatsCan GO (nominal, incl. mining)
    1810026701_databaseLoadingData__1_.csv  <- StatsCan IPPI (incl. primary metals)
    1810026801_databaseLoadingData.csv      <- StatsCan RMPI (raw materials)
    Production.xlsx                         <- CEEDC annual physical output

OUTPUTS:
    provincial_physical_annual.csv  <- Annual provincial physical output by CIMS sector (2000-2024)
                                       Values in tonnes. UTF-8-BOM encoded (opens correctly in Excel).
"""

import pandas as pd
import numpy as np
from pathlib import Path

import sys
_current_file = Path(__file__)
_project_root = _current_file.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
#
# Configuration
BASE_PATH = Path('C:/cims/data')
GO_FILE         = BASE_PATH / 'raw_data/stats_can/activity_levels/36100488.csv'
IPPI_FILE       = BASE_PATH / 'raw_data/stats_can/activity_levels/18100267.csv'
RMPI_FILE       = BASE_PATH / 'raw_data/stats_can/activity_levels/18100268.csv'
PRODUCTION_FILE      = BASE_PATH / 'raw_data/ceedc/Production.xlsx'

BASE_YEAR = 2020   # real GO base year (index = 100); matches IPPI/RMPI base period. Doesn't matter what base year is used. 

# =============================================================================
# DEFLATOR MAPPING
#    Maps each GO industry code (BS-prefix in the GO file) to the best
#    available price index in IPPI or RMPI.
#
#    Notes:
#      - Mining (21220, 21230, 212396): RMPI — correct for extractive industries.
#      - 327A0: StatsCan residual aggregate for non-metallic minerals excl.
#        cement. Lime (32741) is inside this aggregate but inseparable from
#        glass, clay, etc. IPPI Lime and gypsum [3274] used as deflator.
#      - 33100: IPPI Primary metal manufacturing [331] — direct match.
#      - 325C0: broad IPPI Chemical manufacturing [325] — no sub-industry series.
#      - RMPI Metal ores and Potash start in 2010; back-extended to 2000 by
#        holding the 2010 index value constant (conservative assumption).
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
#    Defines every CIMS output product and how it maps to CEEDC data.
#
#    Two entry types:
#      "single"   — one NAICS code/measure -> one CIMS product
#      "combined" — two or more NAICS codes summed -> one CIMS product
#                   Provincial share is weighted by each component's real GO.
#
#    NAICS 327 (parent aggregate): all dashes in CEEDC — excluded.
#    Sub-codes 32731 and 32741 are combined into Industrial Minerals.
# =============================================================================
SECTORS = [
    # ── Mining ──────────────────────────────────────────────────────────────
    {
        "type": "combined",
        "cims_sector": "Mining",
        "note": "Sum of 2122 (metal ore) + 2123 (non-metallic mineral); "
                "provincial share weighted by real GO from sectors 2122 and 2123",
        "components": [
            {"naics_code": "2122",   "measure": "Production / Shipments", "go_sector": "2122"},
            {"naics_code": "2123",   "measure": "Minerals produced",       "go_sector": "2123"},
        ],
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
        "naics_code": "325",    "measure": "Total chemicals",
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
        "naics_code": "325313", "measure": "Ammonia",
        "cims_sector": "Ammonia Methanol",
        "go_sector": "3253",
        "note": "Parent GO sector 3253 (fertilizers)",
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
        "type": "combined",
        "cims_sector": "Industrial Minerals",
        "naics_desc": "Cement and Lime",
        "note": "Sum of 32731 (cement) + 32741 (lime); "
                "provincial share weighted by real GO from sectors 3273 and 327A0",
        "components": [
            {"naics_code": "32731", "measure": "Cement", "go_sector": "3273"},
            {"naics_code": "32741", "measure": "Lime",   "go_sector": "327A0"},
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
        "type": "combined",
        "cims_sector": "Metal Smelting",
        "note": "Sum of 33141 (total non-ferrous metal) + 331313 (molten aluminium); "
                "provincial share weighted by real GO from sectors 3314 and 3313",
        "components": [
            {"naics_code": "33141",  "measure": "Total non-ferrous metal", "go_sector": "3314"},
            {"naics_code": "331313", "measure": "Molten aluminium",        "go_sector": "3313"},
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

# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD RAW DATA
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1  Loading raw data")
print("=" * 60)

go_raw       = pd.read_csv(GO_FILE)
ippi_raw     = pd.read_csv(IPPI_FILE)
rmpi_raw     = pd.read_csv(RMPI_FILE)
prod_raw     = pd.read_excel(PRODUCTION_FILE)

# Extract codes from StatsCan label strings
go_raw["GO_CODE"]        = go_raw["Industry"].str.extract(r"\[BS(\w+)\]")
ippi_col                 = "North American Industry Classification System (NAICS)"
ippi_raw["NAICS_CODE"]   = ippi_raw[ippi_col].str.extract(r"\[(\w+)\]")
rmpi_col                 = "North American Product Classification System (NAPCS)"
rmpi_raw["NAPCS_CODE"]   = rmpi_raw[rmpi_col].str.extract(r"\[(\w+)\]")

prod_raw["naics_code"]   = prod_raw["naics_code"].astype(str)
prod_raw["value"]        = pd.to_numeric(prod_raw["value"], errors="coerce")

print(f"  GO rows:         {len(go_raw):,}")
print(f"  IPPI rows:       {len(ippi_raw):,}")
print(f"  RMPI rows:       {len(rmpi_raw):,}")
print(f"  Production rows: {len(prod_raw):,}")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — DEFLATE NOMINAL GO TO REAL GO
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2  Deflating nominal GO to real GO")
print("=" * 60)

def make_annual_index(df, label_col, label_val, code_col, fill_start_year=None):
    """
    Filter a monthly price index to one series, average to annual,
    and rebase to BASE_YEAR = 100.

    fill_start_year: back-extend to this year by holding the earliest
    available index value constant (needed for RMPI Metal ores / Potash
    which only start in 2010).
    """
    series = df[df[label_col] == label_val].copy()
    if series.empty:
        raise ValueError(f"Label not found in index file: '{label_val}'")

    series["YEAR"] = pd.to_datetime(series["REF_DATE"]).dt.year
    annual = (
        series.groupby("YEAR")["VALUE"]
        .mean().reset_index()
        .rename(columns={"VALUE": "PRICE_INDEX"})
    )

    if fill_start_year is not None:
        first_year = annual["YEAR"].min()
        if first_year > fill_start_year:
            first_val = annual.loc[annual["YEAR"] == first_year, "PRICE_INDEX"].values[0]
            gap_df = pd.DataFrame({
                "YEAR":        list(range(fill_start_year, first_year)),
                "PRICE_INDEX": first_val,
            })
            annual = pd.concat([gap_df, annual], ignore_index=True).sort_values("YEAR")
            print(f"  NOTE: Back-extended '{label_val}' from {first_year} to "
                  f"{fill_start_year} (constant index — document this assumption)")

    base_val = annual.loc[annual["YEAR"] == BASE_YEAR, "PRICE_INDEX"].values
    if len(base_val) == 0:
        raise ValueError(f"Base year {BASE_YEAR} not found in series for '{label_val}'")

    annual["PRICE_INDEX_REBASED"] = annual["PRICE_INDEX"] / base_val[0] * 100
    return annual[["YEAR", "PRICE_INDEX_REBASED"]]


# Build one annual index per unique series (cached to avoid redundant work)
ippi_cache = {}
rmpi_cache = {}

for go_code, mapping in DEFLATOR_MAP.items():
    lbl = mapping["label"]
    if mapping["source"] == "IPPI" and lbl not in ippi_cache:
        ippi_cache[lbl] = make_annual_index(ippi_raw, ippi_col, lbl, "NAICS_CODE")
    elif mapping["source"] == "RMPI" and lbl not in rmpi_cache:
        rmpi_cache[lbl] = make_annual_index(
            rmpi_raw, rmpi_col, lbl, "NAPCS_CODE", fill_start_year=2000
        )

deflated_rows = []

for go_code, mapping in DEFLATOR_MAP.items():
    lbl    = mapping["label"]
    source = mapping["source"]
    idx_df = ippi_cache[lbl] if source == "IPPI" else rmpi_cache[lbl]

    subset = go_raw[go_raw["GO_CODE"] == go_code].copy()
    subset["YEAR"] = subset["REF_DATE"].astype(int)
    merged = subset.merge(idx_df, on="YEAR", how="left")

    missing = merged["PRICE_INDEX_REBASED"].isna().sum()
    if missing > 0:
        print(f"  WARNING: {missing} rows missing price index for {go_code} ({lbl})")

    merged["REAL_GO"]        = merged["VALUE"] / (merged["PRICE_INDEX_REBASED"] / 100)
    merged["NAICS_SECTOR"]   = NAICS_LABEL_MAP.get(go_code, go_code)
    merged["DEFLATOR_SOURCE"] = source
    merged["DEFLATOR_SERIES"] = lbl
    deflated_rows.append(merged)

real_go = pd.concat(deflated_rows, ignore_index=True)

real_go_clean = real_go[[
    "REF_DATE", "GEO", "Industry", "NAICS_SECTOR",
    "VALUE", "REAL_GO", "PRICE_INDEX_REBASED", "DEFLATOR_SOURCE", "DEFLATOR_SERIES",
]].rename(columns={
    "REF_DATE":           "YEAR",
    "VALUE":              "NOMINAL_GO_MILLIONS",
    "REAL_GO":            "REAL_GO_MILLIONS_2020",
})

real_go_clean["YEAR"] = real_go_clean["YEAR"].astype(int)
real_go_clean = real_go_clean.sort_values(["YEAR", "NAICS_SECTOR", "GEO"]).reset_index(drop=True)
print(f"  Real GO rows: {len(real_go_clean):,}")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 — COMPUTE PROVINCIAL GO SHARES (2000-2022)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3  Computing provincial GO shares")
print("=" * 60)

national_go = (
    real_go_clean[real_go_clean["GEO"] == "Canada"]
    [["YEAR", "NAICS_SECTOR", "REAL_GO_MILLIONS_2020"]]
    .rename(columns={"REAL_GO_MILLIONS_2020": "NATIONAL_REAL_GO"})
)

provinces_go = real_go_clean[real_go_clean["GEO"] != "Canada"].copy()
shares_raw_df = provinces_go.merge(national_go, on=["YEAR", "NAICS_SECTOR"], how="left")
shares_raw_df["PROVINCIAL_SHARE"] = (
    shares_raw_df["REAL_GO_MILLIONS_2020"] / shares_raw_df["NATIONAL_REAL_GO"]
)

shares = shares_raw_df[[
    "YEAR", "GEO", "NAICS_SECTOR", "Industry",
    "NOMINAL_GO_MILLIONS", "REAL_GO_MILLIONS_2020",
    "NATIONAL_REAL_GO", "PROVINCIAL_SHARE",
    "DEFLATOR_SOURCE", "DEFLATOR_SERIES",
]].sort_values(["YEAR", "NAICS_SECTOR", "GEO"]).reset_index(drop=True)

# Sanity check: shares sum to ~1.0 per sector/year
share_check = (
    shares.groupby(["YEAR", "NAICS_SECTOR"])["PROVINCIAL_SHARE"]
    .sum().reset_index()
    .rename(columns={"PROVINCIAL_SHARE": "SUM_OF_SHARES"})
)
bad_shares = share_check[abs(share_check["SUM_OF_SHARES"] - 1.0) > 0.01]
if not bad_shares.empty:
    print(f"  WARNING: {len(bad_shares)} sector-years where shares don't sum to 1.0")
    print(bad_shares.to_string())
else:
    print("  Share check PASSED: all sector-years sum to ~1.0")

# (Intermediate GO files are not saved — only the final provincial output is written)

# ═════════════════════════════════════════════════════════════════════════════
# STEP 5 — EXTEND GO SHARES TO 2023-2024 (hold 2022 constant)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5  Extending GO shares to 2023-2024")
print("=" * 60)

shares_2022 = shares[shares["YEAR"] == 2022].copy()
for ext_year in [2023, 2024]:
    ext = shares_2022.copy()
    ext["YEAR"] = ext_year
    shares = pd.concat([shares, ext], ignore_index=True)

print("  GO shares now cover: 2000-2024 (2023-2024 held at 2022 values)")

ANNUAL_YEARS = list(range(2000, 2025))

# ═════════════════════════════════════════════════════════════════════════════
# STEP 6 — BUILD WEIGHTED COMBINED GO SHARES
#    For combined CIMS products, the provincial share is weighted by each
#    component sector's real GO contribution across the combined total.
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 6  Building weighted combined GO shares")
print("=" * 60)

def build_combined_share(go_sectors):
    """
    Combined provincial share for a list of GO sectors, weighted by real GO.
    Returns a DataFrame with columns [YEAR, GEO, PROVINCIAL_SHARE].
    """
    go_sub   = shares[shares["NAICS_SECTOR"].isin(go_sectors)].copy()
    by_prov  = (
        go_sub.groupby(["YEAR", "GEO"])["REAL_GO_MILLIONS_2020"]
        .sum().reset_index()
        .rename(columns={"REAL_GO_MILLIONS_2020": "prov_go"})
    )
    nat_tot  = (
        by_prov.groupby("YEAR")["prov_go"]
        .sum().reset_index()
        .rename(columns={"prov_go": "nat_go"})
    )
    merged = by_prov.merge(nat_tot, on="YEAR")
    merged["PROVINCIAL_SHARE"] = merged["prov_go"] / merged["nat_go"]
    return merged[["YEAR", "GEO", "PROVINCIAL_SHARE"]]

combined_share_cache = {}
for sector in SECTORS:
    if sector["type"] == "combined":
        go_sectors = tuple(sorted(set(c["go_sector"] for c in sector["components"])))
        if go_sectors not in combined_share_cache:
            combined_share_cache[go_sectors] = build_combined_share(list(go_sectors))
            print(f"  Built combined share for GO sectors: {' + '.join(go_sectors)}")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 7 — BUILD NATIONAL PHYSICAL OUTPUT TABLE
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 7  Building national physical output table")
print("=" * 60)

def get_series(naics_code, measure):
    """Return year->value dict for one NAICS code + measure from Production.xlsx."""
    mask = prod_raw["naics_code"] == naics_code
    if measure:
        mask &= prod_raw["measure"] == measure
    sub = prod_raw[mask][["year", "value"]].dropna(subset=["value"])
    return dict(zip(sub["year"], sub["value"]))

national_rows = []

for sector in SECTORS:
    if sector["type"] == "single":
        series = get_series(sector["naics_code"], sector["measure"])
        if not series:
            print(f"  WARNING: No data for {sector['naics_code']} / {sector['measure']}")
            continue
        naics_label   = sector["naics_code"]
        measure_label = sector["measure"]
        go_label      = sector["go_sector"]

    else:  # combined
        combined      = {}
        naics_parts   = []
        measure_parts = []
        go_parts      = []
        for comp in sector["components"]:
            s = get_series(comp["naics_code"], comp["measure"])
            for yr, val in s.items():
                combined[yr] = combined.get(yr, 0) + val
            naics_parts.append(comp["naics_code"])
            measure_parts.append(comp["measure"])
            go_parts.append(comp["go_sector"])
        series        = combined
        naics_label   = " + ".join(naics_parts)
        measure_label = " + ".join(measure_parts)
        go_label      = " + ".join(sorted(set(go_parts))) + " (weighted)"

    for yr in ANNUAL_YEARS:
        national_rows.append({
            "YEAR":            yr,
            "NAICS_CODE":      naics_label,
            "MEASURE":         measure_label,
            "CIMS_SECTOR":     sector["cims_sector"],
            "UNIT":            "tonnes",
            "PHYSICAL_OUTPUT": series.get(yr, np.nan),
            "GO_SECTOR_USED":  go_label,
            "NOTE":            sector["note"],
            "SOURCE":          "CEEDC Production.xlsx (annual)",
        })

national_df = (
    pd.DataFrame(national_rows)
    .sort_values(["CIMS_SECTOR", "YEAR"])
    .reset_index(drop=True)
)

print(f"  {national_df['CIMS_SECTOR'].nunique()} CIMS sectors x {national_df['YEAR'].nunique()} years "
      f"= {len(national_df):,} rows")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 8 — DISAGGREGATE NATIONAL TO PROVINCES
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 8  Disaggregating to provinces")
print("=" * 60)

def get_prov_shares(go_sector_used, year):
    """Return GEO->share dict. Handles both single and combined GO sectors."""
    if "(weighted)" in go_sector_used:
        codes     = tuple(sorted(go_sector_used.replace(" (weighted)", "").split(" + ")))
        share_df  = combined_share_cache[codes]
        yr_shares = share_df[share_df["YEAR"] == year]
    else:
        yr_shares = shares[
            (shares["YEAR"] == year) &
            (shares["NAICS_SECTOR"] == go_sector_used) &
            (shares["GEO"] != "Canada")
        ][["GEO", "PROVINCIAL_SHARE"]]
    return dict(zip(yr_shares["GEO"], yr_shares["PROVINCIAL_SHARE"]))

provincial_rows = []

for _, nat_row in national_df.iterrows():
    yr      = nat_row["YEAR"]
    nat_val = nat_row["PHYSICAL_OUTPUT"]

    if pd.isna(nat_val):
        continue

    prov_shares = get_prov_shares(nat_row["GO_SECTOR_USED"], yr)

    if not prov_shares:
        print(f"  WARNING: No GO shares for {nat_row['CIMS_SECTOR']} / "
              f"{nat_row['GO_SECTOR_USED']} / {yr}")
        continue

    # Canada total row (convert kt -> tonnes)
    canada_row = nat_row.to_dict()
    canada_row["PHYSICAL_OUTPUT"] = round(nat_val * 1000, 4) if not pd.isna(nat_val) else np.nan
    provincial_rows.append({**canada_row, "GEO": "Canada", "PROVINCIAL_SHARE": 1.0})

    # Province / territory rows
    for geo, share in prov_shares.items():
        row = nat_row.to_dict()
        row["GEO"]              = geo
        row["PHYSICAL_OUTPUT"]  = round(nat_val * share * 1000, 4)
        row["PROVINCIAL_SHARE"] = round(share, 6)
        provincial_rows.append(row)

provincial_df = (
    pd.DataFrame(provincial_rows)
    .sort_values(["CIMS_SECTOR", "YEAR", "GEO"])
    .reset_index(drop=True)
)

print(f"  {len(provincial_df):,} rows")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 9 — SANITY CHECKS AND SAVE
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 9  Sanity checks and saving outputs")
print("=" * 60)

prov_sum = (
    provincial_df[provincial_df["GEO"] != "Canada"]
    .groupby(["CIMS_SECTOR", "YEAR"])["PHYSICAL_OUTPUT"]
    .sum().reset_index()
    .rename(columns={"PHYSICAL_OUTPUT": "PROV_SUM"})
)
canada_vals = (
    provincial_df[provincial_df["GEO"] == "Canada"]
    [["CIMS_SECTOR", "YEAR", "PHYSICAL_OUTPUT"]]
    .rename(columns={"PHYSICAL_OUTPUT": "CANADA_TOTAL"})
)
check = prov_sum.merge(canada_vals, on=["CIMS_SECTOR", "YEAR"])
check["DIFF_PCT"] = (
    abs(check["PROV_SUM"] - check["CANADA_TOTAL"]) / check["CANADA_TOTAL"] * 100
)
bad = check[check["DIFF_PCT"] > 0.1]
if not bad.empty:
    print(f"  WARNING: {len(bad)} sector-years where provincial sum differs >0.1% from Canada")
    print(bad.head(10).to_string())
else:
    print("  Province sum check PASSED")

# Save provincial output only, UTF-8-BOM so Excel opens without encoding issues
# NaN -> empty string so no garbled symbols appear in downloaded CSV

provincial_df.to_csv(
    r"C:\cims\data\processed_data\activity\mfg_ind.csv",
    index=False,
    encoding="utf-8-sig",
    na_rep="",
)

print(f"\n{'=' * 60}")
print("ALL DONE")
print(f"{'=' * 60}")
print(f"  provincial_physical_annual.csv  -> {len(provincial_df):,} rows "
      f"({national_df['CIMS_SECTOR'].nunique()} sectors x 25 years x 14 geographies)")
print(f"  Encoding: UTF-8-BOM (opens correctly in Excel)")
print(f"  Units: tonnes (converted from kt source data)")
print(f"\nBase year for deflation: {BASE_YEAR}")
print("\nCombined CIMS sectors:")
for sector in SECTORS:
    if sector["type"] == "combined":
        codes = " + ".join(c["naics_code"] for c in sector["components"])
        print(f"  {sector['cims_sector']:25s} = {codes}")
print("\nNotes:")
print("  - GO shares 2023-2024: held constant at 2022 values (GO data ends 2022)")
print("  - RMPI Metal ores and Potash back-extended 2000-2009 (constant index)")
print("  - NAICS 327 excluded (parent aggregate; sub-codes in Industrial Minerals)")
