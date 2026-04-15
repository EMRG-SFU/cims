"""
Extract commercial calibration data and save to CIMS-formatted CSV files.

This script:
1. Extracts data for ALL commercial regions including disaggregated
   provinces/territories: AT → NL, PE, NS, NB and BC → BC + YT, NT, NU
2. ALWAYS applies projections from commercial_assumptions.csv
3. Exports one CIMS-formatted CSV file per province/territory

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

from source.nrcan.ceud.commercial.commercial import main as commercial_main


# ==============================================================================
# HELPERS
# ==============================================================================

def _get_series(df: pl.DataFrame, variable: str, category: str = '') -> dict:
    """
    Extract {year: value} from a long-format Polars DataFrame for one
    variable/category combination.

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
        Output of commercial main() for one region/province/territory.
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
                {'Branch':    f'CIMS.CAN.{region}.Commercial.Buildings.Hot Water',
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
        """Export HVAC shares for one climate zone."""
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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("COMMERCIAL CALIBRATION DATA EXTRACTION - ALL PROVINCES & TERRITORIES")
    print("=" * 80)
    print(f"Projections:      ENABLED")
    print(f"Output format:    CIMS")
    print(f"Output directory: {output_dir}")
    print("=" * 80)

    # Run the full pipeline including AT → NL/PE/NS/NB and BC → BC/YT/NT/NU
    # Returns {region_code: pl.DataFrame} for all disaggregated regions
    results = commercial_main(apply_projections=True, export_csv=False)

    failed = []
    for region_code, df in results.items():
        try:
            format_to_cims(df, output_dir / f"commercial_{region_code.upper()}.csv", region_code)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"  ❌ Failed formatting {region_code}: {exc}")
            failed.append((region_code, str(exc)))

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✅ Successful: {len(results) - len(failed)}/{len(results)} regions")
    if failed:
        print(f"❌ Failed: {len(failed)}")
        for region, err in failed:
            print(f"  • {region}: {err}")
    print("=" * 80)
    print(f"\n✅ Complete! CSV files saved to: {output_dir}")
