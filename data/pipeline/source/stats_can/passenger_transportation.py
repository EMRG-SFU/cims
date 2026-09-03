from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

DEFAULT_STATCAN_FILE = Path(
    r"C:\cims\data\raw_data\stats_can\market_shares\2010002501-eng.csv"
)
DEFAULT_EPA_FILE = Path(r"C:\cims\data\raw_data\epa\table_export.csv")
DEFAULT_OUTPUT_FILE = Path("passenger_transportation_market_shares.csv")

HISTORICAL_START_YEAR = 2000
FIRST_OBSERVED_YEAR = 2017
LAST_OBSERVED_YEAR = 2025
TOTAL_FUEL = "All fuel types"

PROXY_REGIONS: Dict[str, str] = {
    "Alberta": "Saskatchewan",
    "Newfoundland and Labrador": "Nova Scotia",
    "Nunavut": "Northwest Territories",
}
EXCLUDE_REGIONS = {"Canada"}

TECHNOLOGIES = [
    "Gasoline_Low Efficiency",
    "Gasoline_Medium Efficiency",
    "Gasoline_High Efficiency",
    "Diesel_Low Efficiency",
    "Diesel_Medium Efficiency",
    "Diesel_High Efficiency",
    "Hybrid",
    "Plug-in Hybrid",
    "BEV 500",
    "BEV 800",
]

LOW_PACKAGES = {
    "Carb, Fixed Valve Timing, Two-Valve",
    "Carb, Fixed Valve Timing, Multi-Valve",
    "TBI, Fixed Valve Timing, Two-Valve",
    "TBI, Fixed Valve Timing, Multi-Valve",
    "Port, Fixed Valve Timing, Two-Valve",
    "Port, Fixed Valve Timing, Multi-Valve",
}

MEDIUM_PACKAGES = {
    "GDPI, Fixed Valve Timing, Multi-Valve",
    "GDPI, Variable Valve Timing, Multi-Valve",
    "GDI, Fixed Valve Timing, Multi-Valve",
    "Port, Variable Valve Timing, Multi-Valve",
    "Port, Variable Valve Timing, Two-Valve",
}

HIGH_PACKAGES = {
    "GDI, Variable Valve Timing, Multi-Valve",
    "GDI, Variable Valve Timing, Two-Valve",
}

FUEL_NAME_MAP = {
    "Other fuel types 4": "Other fuel types",
    "Newfoundland and Labrador 2": "Newfoundland and Labrador",
    "Alberta 2": "Alberta",
}


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.exists():
        return p

    basename = str(path).replace("\\", "/").split("/")[-1]
    for loc in [Path.cwd() / basename, Path("/mnt/data") / basename]:
        if loc.exists():
            return loc
    return p


def clean_name(v) -> str:
    s = "" if v is None else str(v).strip().lstrip("\ufeff")
    s = FUEL_NAME_MAP.get(s, s)
    s = re.sub(r"\s+\d+$", "", s).strip()
    return FUEL_NAME_MAP.get(s, s)


def to_number(v) -> float:
    if v is None:
        return np.nan
    if isinstance(v, (int, float)):
        return float(v)

    s = str(v).strip().replace(",", "")
    if s in {"", "..", "-", "nan", "NaN"}:
        return np.nan

    try:
        return float(s)
    except ValueError:
        return np.nan


def parse_year(v) -> Optional[int]:
    match = re.search(r"(19|20)\d{2}", str(v)) if v is not None else None
    return int(match.group(0)) if match else None


def read_statscan_vehicle_sales(
    csv_path,
    first_observed_year=FIRST_OBSERVED_YEAR,
    last_observed_year=LAST_OBSERVED_YEAR,
) -> pd.DataFrame:
    with open(resolve_path(csv_path), "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))

    geography_row_idx = None
    quarter_row_idx = None
    units_row_idx = None

    for i, row in enumerate(rows):
        if row and clean_name(row[0]) == "Geography":
            geography_row_idx = i
        elif row and clean_name(row[0]) == "Fuel type":
            quarter_row_idx = i
        elif row and any(clean_name(c) == "Units" for c in row[:3]):
            units_row_idx = i
            break

    if geography_row_idx is None or quarter_row_idx is None or units_row_idx is None:
        raise ValueError(
            "Could not find Geography, Fuel type, and Units rows in StatCan CSV."
        )

    geography_row = rows[geography_row_idx]
    quarter_row = rows[quarter_row_idx]
    geographies = []
    current_geo = None

    for col in range(max(len(geography_row), len(quarter_row))):
        if col == 0:
            geographies.append(None)
            continue

        geo = clean_name(geography_row[col]) if col < len(geography_row) else ""
        if geo:
            current_geo = geo
        geographies.append(current_geo)

    records = []
    for row in rows[units_row_idx + 1 :]:
        if not row:
            continue

        fuel = clean_name(row[0])
        if not fuel or fuel in {"Symbol legend:", "Footnotes:", "How to cite:"}:
            break

        for col in range(1, min(len(row), len(quarter_row), len(geographies))):
            region = geographies[col]
            year = parse_year(quarter_row[col])
            if (
                region is None
                or year is None
                or year < first_observed_year
                or year > last_observed_year
            ):
                continue

            records.append(
                {
                    "region_original": clean_name(region),
                    "year": year,
                    "fuel_type": fuel,
                    "registrations": to_number(row[col]),
                }
            )

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("No vehicle registration data were read from StatCan CSV.")

    return (
        df.groupby(["region_original", "year", "fuel_type"], as_index=False)[
            "registrations"
        ]
        .sum(min_count=1)
        .rename(columns={"registrations": "annual_sales"})
    )


def apply_region_proxies(annual, proxy_regions):
    annual = annual.copy()
    annual["region"] = annual["region_original"]
    annual["source_region"] = annual["region_original"]
    annual["is_proxy"] = False

    non_proxy = annual[
        ~annual["region_original"].isin(proxy_regions.keys())
    ].copy()
    frames = []

    for target, source in proxy_regions.items():
        proxy = annual[annual["region_original"] == source].copy()
        if proxy.empty:
            raise ValueError(f"Proxy source region {source!r} has no data for {target!r}.")

        proxy["region"] = target
        proxy["source_region"] = source
        proxy["is_proxy"] = True
        frames.append(proxy)

    return (
        pd.concat([non_proxy] + frames, ignore_index=True)
        .query("region not in @EXCLUDE_REGIONS")
        .copy()
    )


def cagr(start, end, periods):
    if pd.isna(start) or pd.isna(end) or periods <= 0:
        return np.nan
    if start == 0 and end == 0:
        return 0.0
    if start == 0 and end != 0:
        return 0.0
    if start > 0 and end == 0:
        return -1.0
    if start < 0 or end < 0:
        return np.nan
    return (end / start) ** (1 / periods) - 1


def backcast_group(g):
    key = g[["region", "source_region", "is_proxy", "fuel_type"]].iloc[0].to_dict()
    series = g.set_index("year")["annual_sales"].to_dict()

    start = series.get(FIRST_OBSERVED_YEAR, np.nan)
    end = series.get(LAST_OBSERVED_YEAR, np.nan)
    growth_rate = cagr(
        start,
        end,
        LAST_OBSERVED_YEAR - FIRST_OBSERVED_YEAR,
    )

    output = []
    for year in range(HISTORICAL_START_YEAR, LAST_OBSERVED_YEAR + 1):
        if year in series and not pd.isna(series[year]):
            value = series[year]
        elif year < FIRST_OBSERVED_YEAR:
            if pd.isna(start) or pd.isna(growth_rate):
                value = np.nan
            elif start == 0 or growth_rate <= -1:
                value = 0.0
            else:
                value = start / ((1 + growth_rate) ** (FIRST_OBSERVED_YEAR - year))
        else:
            value = np.nan

        output.append({**key, "year": year, "annual_sales": value})

    return pd.DataFrame(output)


def add_backcast_history(df):
    return pd.concat(
        [
            backcast_group(group)
            for _, group in df.groupby(
                ["region", "source_region", "is_proxy", "fuel_type"],
                dropna=False,
            )
        ],
        ignore_index=True,
    )


def classify_engine_package(pkg):
    pkg = clean_name(pkg)

    if pkg in HIGH_PACKAGES:
        return "Gasoline_High Efficiency"
    if pkg in MEDIUM_PACKAGES:
        return "Gasoline_Medium Efficiency"
    if pkg in LOW_PACKAGES:
        return "Gasoline_Low Efficiency"
    return None


def load_gasoline_technology_shares(epa_path):
    df = pd.read_csv(resolve_path(epa_path), dtype=str)

    for col in ["Engine Package", "Model Year", "Production (000)"]:
        if col not in df.columns:
            raise ValueError(f"EPA file missing required column: {col}")

    df["year"] = df["Model Year"].map(parse_year)
    df["class"] = df["Engine Package"].map(classify_engine_package)
    df["production"] = df["Production (000)"].map(to_number)

    df = df[
        df["year"].notna()
        & df["class"].notna()
        & df["production"].notna()
        & (df["production"] > 0)
    ].copy()
    df["year"] = df["year"].astype(int)

    wide = (
        df.groupby(["year", "class"], as_index=False)["production"]
        .sum()
        .pivot(index="year", columns="class", values="production")
        .fillna(0)
        .reset_index()
    )

    gasoline_technologies = [
        "Gasoline_Low Efficiency",
        "Gasoline_Medium Efficiency",
        "Gasoline_High Efficiency",
    ]
    for col in gasoline_technologies:
        if col not in wide:
            wide[col] = 0.0

    wide["total"] = wide[gasoline_technologies].sum(axis=1)
    wide = wide[wide["total"] > 0].copy()

    wide["low_share"] = wide["Gasoline_Low Efficiency"] / wide["total"]
    wide["medium_share"] = wide["Gasoline_Medium Efficiency"] / wide["total"]
    wide["high_share"] = wide["Gasoline_High Efficiency"] / wide["total"]

    return (
        wide[["year", "low_share", "medium_share", "high_share"]]
        .sort_values("year")
        .reset_index(drop=True)
    )


def lookup_share(shares, year):
    if year in shares:
        return shares[year]

    years = sorted(shares)
    prior_years = [available_year for available_year in years if available_year <= year]
    return shares[prior_years[-1] if prior_years else years[0]]


def expand_vehicle_technologies(fuel_df, shares_df):
    shares = (
        shares_df.set_index("year")[["low_share", "medium_share", "high_share"]]
        .to_dict("index")
    )
    rows = []

    for _, row in fuel_df.iterrows():
        fuel = clean_name(row["fuel_type"])
        year = int(row["year"])
        sales = row["annual_sales"]
        base = {
            "region": row["region"],
            "source_region": row.get("source_region", row["region"]),
            "is_proxy": row.get("is_proxy", False),
            "year": year,
        }

        if fuel in {TOTAL_FUEL, "All zero-emission vehicles"}:
            continue

        if fuel == "Gasoline":
            year_shares = lookup_share(shares, year)
            splits = [
                ("Gasoline_Low Efficiency", year_shares["low_share"]),
                ("Gasoline_Medium Efficiency", year_shares["medium_share"]),
                ("Gasoline_High Efficiency", year_shares["high_share"]),
            ]
        elif fuel == "Diesel":
            year_shares = lookup_share(shares, year)
            splits = [
                ("Diesel_Low Efficiency", year_shares["low_share"]),
                ("Diesel_Medium Efficiency", year_shares["medium_share"]),
                ("Diesel_High Efficiency", year_shares["high_share"]),
            ]
        elif fuel == "Battery electric":
            splits = [("BEV 500", 1.0), ("BEV 800", 0.0)]
        elif fuel == "Plug-in hybrid electric":
            splits = [("Plug-in Hybrid", 1.0)]
        elif fuel == "Hybrid electric":
            splits = [("Hybrid", 1.0)]
        elif fuel == "Other fuel types":
            continue

        for technology, share in splits:
            rows.append(
                {
                    **base,
                    "fuel_type": technology,
                    "annual_sales": sales * share,
                }
            )

    tech_df = pd.DataFrame(rows)
    region_years = tech_df[
        ["region", "source_region", "is_proxy", "year"]
    ].drop_duplicates()
    bev_800_existing = tech_df[tech_df["fuel_type"] == "BEV 800"][
        ["region", "year"]
    ].drop_duplicates()
    additions = []

    for _, row in region_years.iterrows():
        exists = (
            (bev_800_existing["region"] == row["region"])
            & (bev_800_existing["year"] == row["year"])
        ).any()
        if not exists:
            additions.append(
                {
                    **row.to_dict(),
                    "fuel_type": "BEV 800",
                    "annual_sales": 0.0,
                }
            )

    if additions:
        return pd.concat([tech_df, pd.DataFrame(additions)], ignore_index=True)
    return tech_df


def calculate_market_shares(tech_df):
    totals = (
        tech_df.groupby(["region", "year"], as_index=False)["annual_sales"]
        .sum(min_count=1)
        .rename(columns={"annual_sales": "total"})
    )

    output = tech_df.merge(totals, on=["region", "year"], how="left")
    output["market_share"] = np.where(
        output["total"].notna() & (output["total"] != 0),
        output["annual_sales"] / output["total"],
        np.nan,
    )

    return output.groupby(
        ["region", "year", "fuel_type"], as_index=False
    )["market_share"].sum(min_count=1)


def build_market_shares(
    statcan_file=DEFAULT_STATCAN_FILE,
    epa_file=DEFAULT_EPA_FILE,
    output_file=DEFAULT_OUTPUT_FILE,
):
    annual = read_statscan_vehicle_sales(statcan_file)
    annual = apply_region_proxies(annual, PROXY_REGIONS)
    annual = add_backcast_history(annual)

    shares = load_gasoline_technology_shares(epa_file)
    tech_sales = expand_vehicle_technologies(annual, shares)
    result = calculate_market_shares(tech_sales)

    order = {technology: i for i, technology in enumerate(TECHNOLOGIES)}
    result = result[result["fuel_type"].isin(TECHNOLOGIES)].copy()
    result["_order"] = result["fuel_type"].map(order)
    result = result.sort_values(["region", "year", "_order"]).drop(
        columns="_order"
    )

    output_path = Path(output_file)
    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    result.to_csv(output_path, index=False)
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build passenger transportation market shares for CIMS."
    )
    parser.add_argument("--statcan-file", default=str(DEFAULT_STATCAN_FILE))
    parser.add_argument("--epa-file", default=str(DEFAULT_EPA_FILE))
    parser.add_argument("--output-file", default=str(DEFAULT_OUTPUT_FILE))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    df = build_market_shares(
        args.statcan_file,
        args.epa_file,
        args.output_file,
    )
    print(f"Wrote {len(df):,} rows to {args.output_file}")
