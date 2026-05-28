"""
CER Demand by CIMS Node

Maps CER energy demand data to CIMS nodes using cer_to_cims_map.csv and
aggregates fuel demand (PJ) by region, cims_node, fuel, and year.

Input files (raw_data/cer/cer_demand_data_update/):
    vDmd-CIMS.csv      — general energy demand        (Sector + Enduse)
    vDmd_OG-CIMS.csv   — oil & gas energy demand      (Sector + Enduse)
    vTrDmd-CIMS.csv    — transport energy demand      (Sector + Technology)
    vFsDmd-CIMS.csv    — feedstock demand             (Sector only)
    vTrFsDmd-CIMS.csv  — transport feedstock demand   (Sector + Technology)

Output: processed_data/cer/cer_resd_demand.csv
    Columns: Region | Node | Variable | Unit | Source | Year | Value
"""

import pandas as pd
from pathlib import Path
import sys

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from utils.controls_conversions import BASE_PATH, MAPPINGS_PATH

CER_DIR         = BASE_PATH / "raw_data/cer/cer_demand_data_update"
MAP_FILE        = MAPPINGS_PATH / "cer_to_cims_map.csv"
ENERGY_MAP_FILE = MAPPINGS_PATH / "energy_map.csv"
REGION_MAP_FILE = MAPPINGS_PATH / "region_map.csv"
OUTPUT_DIR      = BASE_PATH / "processed_data/cer"
OUTPUT_FILE     = OUTPUT_DIR / "cer_resd_demand.csv"

# CER Area values that don't match the region_map CER column exactly
_AREA_NORMALIZE = {
    "newfoundland":        "Newfoundland and Labrador",
    "northwest territory": "Northwest Territories",
    "yukon territory":     "Yukon",
}

# CER fuel names that differ from their energy_map CER column only by spelling
_FUEL_NORMALIZE = {
    "electric":                  "electricity",
    "petrochemical feedstocks":  "petrochemical feedstock",
    "coke oven gas":             "coke",
}


def build_fuel_lookup(energy_map: pd.DataFrame) -> dict:
    """Return {cer_fuel_lower: cims_fuel} for unambiguous 1-to-1 mappings.

    Entries where the same CER fuel maps to multiple CIMS fuels (e.g. Biomass,
    Coal, Natural Gas) are excluded and handled by special-case logic instead.
    """
    mapping: dict = {}
    ambiguous: set = set()

    for _, row in energy_map.iterrows():
        cer_val = row.get("CER", "")
        if pd.isna(cer_val) or str(cer_val).strip() == "":
            continue
        cims = row["CIMS"]
        for cer_raw in str(cer_val).split(","):
            key = cer_raw.strip().lower()
            if key in mapping and mapping[key] != cims:
                ambiguous.add(key)
            else:
                mapping[key] = cims

    for k in ambiguous:
        mapping.pop(k, None)

    return mapping


def map_fuel_names(df: pd.DataFrame, energy_map: pd.DataFrame) -> pd.DataFrame:
    """Rename CER fuel names to CIMS fuel names using energy_map.

    Special cases resolved by cims_node:
      Biomass → Black Liquor  (Pulp and Paper nodes only)
              → Solid Biomass (all other nodes)
      Coal    → Metallurgical Coal (Iron and Steel nodes only)
              → Thermal Coal       (all other nodes)
      Natural Gas → Natural Gas Feedstock (feedstock rows: cer_enduse == "")
                  → Natural Gas           (all other rows)
    """
    lookup = build_fuel_lookup(energy_map)
    df = df.copy()

    df["_fuel_norm"] = df["Fuel"].str.strip().str.lower()
    for old, new in _FUEL_NORMALIZE.items():
        df["_fuel_norm"] = df["_fuel_norm"].where(df["_fuel_norm"] != old, new)

    # Unambiguous renames
    df["Fuel"] = df["_fuel_norm"].map(lookup).fillna(df["Fuel"])

    # Biomass
    mask = df["_fuel_norm"] == "biomass"
    if mask.any():
        is_pp = df["cims_node"].str.startswith(".Pulp and Paper")
        df.loc[mask & is_pp,  "Fuel"] = "Black Liquor"
        df.loc[mask & ~is_pp, "Fuel"] = "Solid Biomass"

    # Coal
    mask = df["_fuel_norm"] == "coal"
    if mask.any():
        is_is = df["cims_node"].str.startswith(".Iron and Steel")
        df.loc[mask & is_is,  "Fuel"] = "Metallurgical Coal"
        df.loc[mask & ~is_is, "Fuel"] = "Thermal Coal"

    # Natural Gas
    mask = df["_fuel_norm"] == "natural gas"
    if mask.any():
        is_feedstock = df["cer_enduse"] == ""
        df.loc[mask & is_feedstock,  "Fuel"] = "Natural Gas Feedstock"
        df.loc[mask & ~is_feedstock, "Fuel"] = "Natural Gas"

    df.drop(columns="_fuel_norm", inplace=True)
    return df


def load_cer_data() -> pd.DataFrame:
    frames = []

    # Files with an Enduse column
    for fname in ("vDmd-CIMS.csv", "vDmd_OG-CIMS.csv"):
        df = pd.read_csv(
            CER_DIR / fname, sep=";",
            usecols=["Area", "Sector", "Enduse", "Fuel", "Year", "Data"],
        )
        df.rename(columns={"Sector": "cer_sector", "Enduse": "cer_enduse"}, inplace=True)
        frames.append(df)

    # Files with a Technology column (treated as enduse)
    for fname in ("vTrDmd-CIMS.csv", "vTrFsDmd-CIMS.csv"):
        df = pd.read_csv(
            CER_DIR / fname, sep=";",
            usecols=["Area", "Sector", "Technology", "Fuel", "Year", "Data"],
        )
        df.rename(columns={"Sector": "cer_sector", "Technology": "cer_enduse"}, inplace=True)
        frames.append(df)

    # Feedstock file — no enduse column
    df = pd.read_csv(
        CER_DIR / "vFsDmd-CIMS.csv", sep=";",
        usecols=["Area", "Sector", "Fuel", "Year", "Data"],
    )
    df.rename(columns={"Sector": "cer_sector"}, inplace=True)
    df["cer_enduse"] = ""
    frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined["cer_sector"] = combined["cer_sector"].str.strip().str.lower()
    combined["cer_enduse"] = combined["cer_enduse"].fillna("").str.strip().str.lower()
    return combined


def main() -> pd.DataFrame:
    print("Loading CER demand data...")
    cer = load_cer_data()
    print(f"  {len(cer):,} rows across all CER files")

    print("Loading mappings...")
    mapping = pd.read_csv(MAP_FILE)
    mapping["cer_sector"] = mapping["cer_sector"].str.strip().str.lower()
    mapping["cer_enduse"] = mapping["cer_enduse"].fillna("").str.strip().str.lower()
    energy_map = pd.read_csv(ENERGY_MAP_FILE)
    region_map = pd.read_csv(REGION_MAP_FILE)
    area_to_cims = dict(zip(region_map["CER"].str.strip(), region_map["CIMS"].str.strip()))

    print("Joining to CIMS nodes...")
    specific_map = mapping[mapping["cer_enduse"] != ""]
    sector_map   = mapping[mapping["cer_enduse"] == ""][["cer_sector", "cims_node"]]

    # Exact match on sector + enduse
    merged_specific = cer.merge(specific_map, on=["cer_sector", "cer_enduse"], how="inner")

    # Sector-only rows: sum all enduses for that sector
    merged_sector = cer.merge(sector_map, on="cer_sector", how="inner")

    merged = pd.concat([merged_specific, merged_sector], ignore_index=True)

    # Report sectors that have no mapping at all (neither specific nor sector-level)
    mapped_sectors = set(specific_map["cer_sector"]) | set(sector_map["cer_sector"])
    unmatched = (
        cer[["cer_sector", "cer_enduse"]]
        .drop_duplicates()
        .loc[lambda d: ~d["cer_sector"].isin(mapped_sectors)]
        .sort_values(["cer_sector", "cer_enduse"])
    )
    if not unmatched.empty:
        print(f"  Warning: {len(unmatched['cer_sector'].unique())} sectors have no mapping:")
        print(unmatched.to_string(index=False))

    print("Mapping fuel names to CIMS names...")
    merged = map_fuel_names(merged, energy_map)

    ignore_fuels = {"air thermal", "geothermal", "steam"}
    merged = merged[~merged["Fuel"].str.lower().isin(ignore_fuels)]

    print("Mapping regions to CIMS names...")
    merged["_area_norm"] = merged["Area"].str.strip().str.lower()
    for cer_raw, cer_std in _AREA_NORMALIZE.items():
        merged["_area_norm"] = merged["_area_norm"].where(
            merged["_area_norm"] != cer_raw, cer_std.lower()
        )
    _area_lower = {k.lower(): v for k, v in area_to_cims.items()}
    merged["Region"] = merged["_area_norm"].map(_area_lower).fillna(merged["Area"])
    merged.drop(columns="_area_norm", inplace=True)

    unmatched_areas = merged.loc[merged["Region"] == merged["Area"], "Area"].unique()
    if len(unmatched_areas):
        print(f"  Warning: unmapped areas: {sorted(unmatched_areas)}")

    print("Aggregating by region, cims_node, fuel, year...")
    agg = (
        merged
        .groupby(["Region", "cims_node", "Fuel", "Year"], as_index=False)["Data"]
        .sum()
    )
    agg["Data"] = agg["Data"] * 1000

    result = pd.DataFrame({
        "Region":   agg["Region"],
        "Node":     agg["cims_node"],
        "Variable": agg["Fuel"],
        "Unit":     "GJ",
        "Source":   "CER/RESD",
        "Year":     agg["Year"],
        "Value":    agg["Data"],
    })
    result = result.sort_values(["Region", "Node", "Variable", "Year"]).reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig", na_rep="")

    print(f"\nDone.")
    print(f"  Rows written:  {len(result):,}")
    print(f"  Regions:       {result['Region'].nunique()}")
    print(f"  Nodes:         {result['Node'].nunique()}")
    print(f"  Variables:     {result['Variable'].nunique()}")
    print(f"  Years:         {result['Year'].min()} – {result['Year'].max()}")
    print(f"  Output:        {OUTPUT_FILE}")

    return result


if __name__ == "__main__":
    main()
