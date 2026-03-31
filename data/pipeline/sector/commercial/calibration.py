"""
Extract commercial calibration data and save to CIMS-formatted CSV files.

This script:
1. Extracts data for ALL commercial regions
2. ALWAYS applies projections from commercial_assumptions.csv
3. Exports to CIMS-formatted CSV files

Output columns: Branch, Type, Region, Sector, Service, Technology, Parameter,
                Context, Sub_Context, Target, Source, Unit, Year, Value

The following data is extracted:
- Total floorspace by region (m2)
- Building shell shares by region (Wholesale, Retail, Offices, etc.)
- Hot water technologies by region
- HVAC technologies by region — Cold climate (all regions),
  Marine climate (BC only)
  NOTE: BC exports BOTH Marine AND Cold climate HVAC data

All percentage values are on 0-1 scale (fractions) as produced by the pipeline.
"""

import sys
import argparse
from pathlib import Path

import polars as pl
import pandas as pd

_current_file = Path(__file__)
_project_root = _current_file.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from source.nrcan.ceud.commercial.commercial import extract_all_data, REGIONS


# ==============================================================================
# HELPERS
# ==============================================================================

def _get_series(df: pl.DataFrame, variable: str, category: str = '') -> dict:
    """
    Extract {year: value} from a long-format Polars DataFrame for one
    variable/category combination.  Uses .to_list() on numeric-only columns
    so pyarrow is never required.

    Parameters
    ----------
    df : pl.DataFrame
        Region-level long-format DataFrame from extract_all_data().
    variable : str
    category : str
        Empty string for scalar variables (e.g. total_floorspace).

    Returns
    -------
    dict  {int year: float value}  — empty dict if nothing matches.
    """
    mask = pl.col('variable') == variable
    if category:
        mask = mask & (pl.col('category') == category)
    subset = df.filter(mask)
    if len(subset) == 0:
        return {}
    years  = subset.get_column('year').cast(pl.Int64).to_list()
    values = subset.get_column('value').cast(pl.Float64).to_list()
    return {int(y): float(v) for y, v in zip(years, values) if v is not None}


def _get_categories(df: pl.DataFrame, variable: str) -> list[str]:
    """Return sorted list of unique category values for a given variable."""
    subset = df.filter(pl.col('variable') == variable)
    return sorted(subset.get_column('category').unique().to_list())


# ==============================================================================
# CIMS FORMATTER
# ==============================================================================

def format_to_cims(df: pl.DataFrame, output_file: str | Path,
                   region_code: str) -> str:
    """
    Convert a region's long-format Polars DataFrame to a CIMS-formatted CSV.

    Parameters
    ----------
    df : pl.DataFrame
        Output of extract_all_data() for one region.
    output_file : str or Path
    region_code : str

    Returns
    -------
    str  Path to the saved CSV file.
    """
    region = region_code.upper()
    is_bc  = region == 'BC'
    rows: list[dict] = []

    def make_row(meta: dict, year_dict: dict, scale: float = 1.0) -> list[dict]:
        """Build one CIMS output row per year from a {year: value} dict."""
        result = []
        for year, value in year_dict.items():
            if value is not None:
                result.append({
                    'Branch':      meta.get('Branch', ''),
                    'Type':        meta.get('Type', ''),
                    'Region':      region,
                    'Sector':      'Commercial',
                    'Service':     meta.get('Service', ''),
                    'Technology':  meta.get('Technology', ''),
                    'Parameter':   meta.get('Parameter', ''),
                    'Context':     meta.get('Context', ''),
                    'Sub_Context': meta.get('Sub_Context', ''),
                    'Target':      meta.get('Target', ''),
                    'Source':      'CEUD',
                    'Unit':        meta.get('Unit', ''),
                    'Year':        int(year),
                    'Value':       float(value) * scale,
                })
        return result

    # ------------------------------------------------------------------
    # 1. TOTAL FLOORSPACE
    # ------------------------------------------------------------------
    rows.extend(make_row(
        {'Branch':    f'CIMS.CAN.{region}',
         'Type':      'Region',
         'Parameter': 'service_request',
         'Target':    f'CIMS.CAN.{region}.Commercial',
         'Unit':      'm2'},
        _get_series(df, 'total_floorspace'),
    ))

    # ------------------------------------------------------------------
    # 2. BUILDING SHELL SHARES
    # ------------------------------------------------------------------
    # For BC: split share equally between Marine (80%) and Cold (20%).
    # For all others: Cold only (weight = 1.0).
    climate_weights = (
        {'(Marine)': 0.80, '(Cold)': 0.20} if is_bc else {'(Cold)': 1.0}
    )

    for activity in _get_categories(df, 'building_shell_shares'):
        raw_share = _get_series(df, 'building_shell_shares', activity)
        if not raw_share:
            continue
        for climate_label, weight in climate_weights.items():
            weighted = {yr: val * weight for yr, val in raw_share.items()}
            rows.extend(make_row(
                {'Branch':    f'CIMS.CAN.{region}.Commercial.Buildings.Shell',
                 'Type':      'Service',
                 'Service':   'Shell',
                 'Parameter': 'service_request',
                 'Target':    f'CIMS.CAN.{region}.Commercial.Buildings.Shell.{activity} {climate_label}',
                 'Unit':      '% of m2'},
                weighted,
            ))

    # ------------------------------------------------------------------
    # 3. HOT WATER TECHNOLOGIES
    # ------------------------------------------------------------------
    for tech in _get_categories(df, 'hot_water_tech'):
        hw = _get_series(df, 'hot_water_tech', tech)
        if hw:
            rows.extend(make_row(
                {'Branch':    f'CIMS.CAN.{region}.Commercial.Hot Water',
                 'Type':      'Service',
                 'Service':   'Hot Water',
                 'Technology': tech,
                 'Parameter': 'market_share_total',
                 'Unit':      '% of hot water'},
                hw,
            ))

    # ------------------------------------------------------------------
    # 4. HVAC TECHNOLOGIES
    # ------------------------------------------------------------------
    def export_hvac(climate_var: str, climate_label: str) -> None:
        """Export weighted HVAC shares for one climate zone."""
        for tech in _get_categories(df, climate_var):
            hvac = _get_series(df, climate_var, tech)
            if hvac:
                rows.extend(make_row(
                    {'Branch':    f'CIMS.CAN.{region}.Commercial.HVAC {climate_label}',
                     'Type':      'Service',
                     'Service':   'HVAC',
                     'Technology': tech,
                     'Parameter': 'market_share_total',
                     'Unit':      '% of GJ of heat'},
                    hvac,
                ))

    if is_bc:
        export_hvac('hvac_marine', '(Marine)')
        export_hvac('hvac_cold',   '(Cold)')
    else:
        export_hvac('hvac_cold', '(Cold)')

    # ------------------------------------------------------------------
    # Assemble output DataFrame
    # ------------------------------------------------------------------
    out = pd.DataFrame(rows)
    column_order = ['Branch', 'Type', 'Region', 'Sector', 'Service', 'Technology',
                    'Parameter', 'Context', 'Sub_Context', 'Target', 'Source',
                    'Unit', 'Year', 'Value']
    out = out[column_order]
    if not out.empty:
        out = out.sort_values(['Branch', 'Technology', 'Year'])

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"  ✅ Saved {len(out):,} rows to {output_path}")
    return str(output_path)


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract commercial data from CEUD and export to CIMS-formatted CSV files"
    )
    parser.add_argument(
        "--output-dir",
        default=r"C:\cims\data\calibration\commercial",
        help=r"Output directory for CSV files (default: C:\cims\data\calibration\commercial)",
    )
    args = parser.parse_args()

    region_codes = list(REGIONS.keys())
    output_dir   = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict = {}
    failed:  list = []

    print("=" * 80)
    print("COMMERCIAL DATA EXTRACTION - ALL REGIONS")
    print("=" * 80)
    print(f"Regions:          {', '.join(region_codes)}")
    print(f"Projections:      ENABLED")
    print(f"Output format:    CIMS")
    print(f"Output directory: {output_dir}")
    print("=" * 80)

    for region in region_codes:
        try:
            print(f"\n{region} — {REGIONS[region.upper()]}:")
            df = extract_all_data(region, apply_projections=True)
            results[region] = df
            print(f"  ✅ Extraction complete")
            format_to_cims(df, output_dir / f"commercial_{region.upper()}.csv", region)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"  ❌ Failed: {exc}")
            failed.append((region, str(exc)))

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✅ Successful: {len(results)}/{len(region_codes)} regions")
    if failed:
        print(f"❌ Failed: {len(failed)} regions")
        for region, err in failed:
            print(f"  • {region}: {err}")
    print("=" * 80)
    print(f"\n✅ Complete! CSV files saved to: {output_dir}")
