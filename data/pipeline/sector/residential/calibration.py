"""
Extract residential calibration data and save to CIMS-formatted CSV files.

This script:
1. Extracts data for ALL provinces and territories (including disaggregated
   YT, NT, NU from TR)
2. ALWAYS applies projections from residential_assumptions.csv
3. Exports one CIMS-formatted CSV file per province/territory

Output columns: Branch, Type, Region, Sector, Service, Technology, Parameter,
                Context, Sub_Context, Target, Source, Unit, Year, Value

The following data is extracted:
- Housing stock (households)
- Building shares by type (Single Detached, Single Attached, Apartments, Mobile Homes)
- Floorspace per building
- Appliances per household
- Vintage bins (building age distributions) — LowMed and High density
- Heating technologies — Cold climate (all provinces), Marine climate (BC only)
  NOTE: BC exports BOTH Marine AND Cold climate heating data
- Cooling technologies (Room and Central)
- Water heating service request and technologies

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

from source.nrcan.ceud.residential.residential import main as residential_main


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
                   province_code: str) -> str:
    """
    Convert a province's long-format Polars DataFrame to a CIMS-formatted CSV.

    Parameters
    ----------
    df : pl.DataFrame
        Output of residential main() for one province/territory.
    output_file : str or Path
    province_code : str

    Returns
    -------
    str  Path to the saved CSV file.
    """
    prov  = province_code.upper()
    is_bc = prov == 'BC'
    rows: list[dict] = []

    def make_row(meta: dict, year_dict: dict, scale: float = 1.0) -> list[dict]:
        """Build one CIMS output row per year from a {year: value} dict."""
        result = []
        for year, value in year_dict.items():
            if value is not None:
                result.append({
                    'Branch':      meta.get('Branch', ''),
                    'Type':        meta.get('Type', ''),
                    'Region':      prov,
                    'Sector':      'Residential',
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
    # 1. HOUSING STOCK
    # ------------------------------------------------------------------
    rows.extend(make_row(
        {'Branch':    f'CIMS.CAN.{prov}',
         'Type':      'Region',
         'Parameter': 'service_request',
         'Target':    f'CIMS.CAN.{prov}.Residential',
         'Unit':      'household'},
        _get_series(df, 'housing_thousand'),
    ))

    # ------------------------------------------------------------------
    # 2. APPLIANCES
    # ------------------------------------------------------------------
    for appl_name in _get_categories(df, 'appliances_per_household'):
        rows.extend(make_row(
            {'Branch':    f'CIMS.CAN.{prov}.Residential.Dwellings',
             'Type':      'Service',
             'Service':   'Dwellings',
             'Parameter': 'service_request',
             'Target':    f'CIMS.CAN.{prov}.Residential.Dwellings.{appl_name}',
             'Unit':      'unit/building'},
            _get_series(df, 'appliances_per_household', appl_name),
        ))

    # ------------------------------------------------------------------
    # 3. BUILDING TYPES — shares and floorspace per building
    # ------------------------------------------------------------------
    building_map = {
        'Apartments':      ('Apartment', 'High Density'),
        'Single Detached': ('Detached',  'LowMed Density'),
        'Single Attached': ('Attached',  'LowMed Density'),
        'Mobile Homes':    ('Mobile',    'LowMed Density'),
    }
    base_branch = f'CIMS.CAN.{prov}.Residential.Dwellings.Building Type'

    for source_key, (tech_name, density) in building_map.items():
        share = _get_series(df, 'building_shares', source_key)
        if share:
            rows.extend(make_row(
                {'Branch':    base_branch,
                 'Type':      'Service',
                 'Service':   'Building Type',
                 'Technology': tech_name,
                 'Parameter': 'market_share_total',
                 'Unit':      '%'},
                share,
            ))

        fs = _get_series(df, 'floorspace_per_building', source_key)
        if fs:
            rows.extend(make_row(
                {'Branch':    base_branch,
                 'Type':      'Service',
                 'Service':   'Building Type',
                 'Technology': tech_name,
                 'Parameter': 'service_request',
                 'Target':    f'{base_branch}.{density}',
                 'Unit':      'm2'},
                fs,
            ))

    # ------------------------------------------------------------------
    # 4 & 5. COOLING
    # ------------------------------------------------------------------
    for ac_tech in _get_categories(df, 'cooling_share_data'):
        cooling = _get_series(df, 'cooling_share_data', ac_tech)
        if not cooling:
            continue
        for density_label in ['LowMed Density', 'High Density']:
            rows.extend(make_row(
                {'Branch':    f'CIMS.CAN.{prov}.Residential.Dwellings.Building Type.{density_label}.Cooling',
                 'Type':      'Service',
                 'Service':   'Cooling',
                 'Parameter': 'service_request',
                 'Target':    f'CIMS.CAN.{prov}.Residential.Dwellings.Building Type.{density_label}.Cooling.{ac_tech}',
                 'Unit':      'GJ cooling/GJ cooling'},
                cooling,
            ))

    # ------------------------------------------------------------------
    # 6 & 7. WATER HEATING SERVICE REQUEST
    # ------------------------------------------------------------------
    for var, density_label in [('wh_lowmed', 'LowMed Density'), ('wh_high', 'High Density')]:
        wh = _get_series(df, var)
        if wh:
            rows.extend(make_row(
                {'Branch':    f'CIMS.CAN.{prov}.Residential.Water Heating.{density_label}',
                 'Type':      'Service',
                 'Service':   'Water Heating',
                 'Parameter': 'service_request',
                 'Target':    f'CIMS.CAN.{prov}.Residential.Water Heating.{density_label}',
                 'Unit':      'GJ water heating/GJ water heating'},
                wh,
            ))

    # ------------------------------------------------------------------
    # 8 & 9. WATER HEATING TECHNOLOGIES
    # ------------------------------------------------------------------
    for var, density_label in [('wh_tech_lowmed', 'LowMed Density'),
                                ('wh_tech_high',   'High Density')]:
        for tech in _get_categories(df, var):
            wh_tech = _get_series(df, var, tech)
            if wh_tech:
                rows.extend(make_row(
                    {'Branch':    f'CIMS.CAN.{prov}.Residential.Water Heating.{density_label}',
                     'Type':      'Service',
                     'Service':   density_label,
                     'Technology': tech,
                     'Parameter': 'market_share_total',
                     'Unit':      '% of GJ of water heating'},
                    wh_tech,
                ))

    # ------------------------------------------------------------------
    # 10 & 11. VINTAGE BINS
    # ------------------------------------------------------------------
    for var, density_label in [('vintage_bins_high',   'High Density'),
                                ('vintage_bins_lowmed', 'LowMed Density')]:
        for vint in _get_categories(df, var):
            vint_data = _get_series(df, var, vint)
            if vint_data:
                rows.extend(make_row(
                    {'Branch':    f'CIMS.CAN.{prov}.Residential.Dwellings.Building Type.{density_label}.Vintage',
                     'Type':      'Service',
                     'Service':   'Vintage',
                     'Technology': vint,
                     'Parameter': 'market_share_total',
                     'Unit':      '% of m2'},
                    vint_data,
                ))

    # ------------------------------------------------------------------
    # 12–15. HEATING TECHNOLOGIES
    # ------------------------------------------------------------------
    lowmed_vintages = _get_categories(df, 'vintage_bins_lowmed')
    high_vintages   = _get_categories(df, 'vintage_bins_high')

    def export_heating(lowmed_var: str, high_var: str, climate_label: str) -> None:
        """Export heating market shares for one climate (Cold or Marine)."""
        lowmed_techs = _get_categories(df, lowmed_var)
        high_techs   = _get_categories(df, high_var)

        for vint in lowmed_vintages:
            for tech in lowmed_techs:
                tech_data = _get_series(df, lowmed_var, tech)
                if tech_data:
                    rows.extend(make_row(
                        {'Branch':    f'CIMS.CAN.{prov}.Residential.Dwellings.Building Type'
                                      f'.LowMed Density.Vintage.{vint} Bldg Code.Heating ({climate_label})',
                         'Type':      'Service',
                         'Service':   'Heating',
                         'Technology': tech,
                         'Parameter': 'market_share_total',
                         'Unit':      '% of GJ of heat'},
                        tech_data,
                    ))

        for vint in high_vintages:
            for tech in high_techs:
                tech_data = _get_series(df, high_var, tech)
                if tech_data:
                    rows.extend(make_row(
                        {'Branch':    f'CIMS.CAN.{prov}.Residential.Dwellings.Building Type'
                                      f'.High Density.Vintage.{vint} Bldg Code.Heating ({climate_label})',
                         'Type':      'Service',
                         'Service':   'Heating',
                         'Technology': tech,
                         'Parameter': 'market_share_total',
                         'Unit':      '% of GJ of heat'},
                        tech_data,
                    ))

    if is_bc:
        export_heating('heating_lowmed_marine', 'heating_high_marine', 'Marine')
        export_heating('heating_lowmed_cold',   'heating_high_cold',   'Cold')
    else:
        export_heating('heating_lowmed_cold', 'heating_high_cold', 'Cold')

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
        description="Extract residential data from CEUD and export to CIMS-formatted CSV files"
    )
    parser.add_argument(
        "--output-dir",
        default=r"C:\cims\data\calibration\residential",
        help=r"Output directory for CSV files (default: C:\cims\data\calibration\residential)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("RESIDENTIAL CALIBRATION DATA EXTRACTION - ALL PROVINCES & TERRITORIES")
    print("=" * 80)
    print(f"Projections:      ENABLED")
    print(f"Output format:    CIMS")
    print(f"Output directory: {output_dir}")
    print("=" * 80)

    # Run the full pipeline including TR → YT/NT/NU disaggregation
    # Returns {province_code: pl.DataFrame} for all 13 provinces/territories
    results = residential_main(apply_projections=True, export_csv=False)

    failed = []
    for prov_code, df in results.items():
        try:
            format_to_cims(df, output_dir / f"residential_{prov_code.upper()}.csv", prov_code)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"  ❌ Failed formatting {prov_code}: {exc}")
            failed.append((prov_code, str(exc)))

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✅ Successful: {len(results) - len(failed)}/{len(results)} provinces/territories")
    if failed:
        print(f"❌ Failed: {len(failed)}")
        for prov, err in failed:
            print(f"  • {prov}: {err}")
    print("=" * 80)
    print(f"\n✅ Complete! CSV files saved to: {output_dir}")
