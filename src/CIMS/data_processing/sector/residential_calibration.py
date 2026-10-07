"""
Extract residential calibration data and save to CIMS-formatted CSV files.

Sources
-------
Emissions  (calibration_emissions_total from crosswalk; calibration_emissions_by_type from nir_to_cims)
    nir_crosswalk_tables_cims.py  → total tCO2e per residential CIMS branch
                                    5-year intervals (2000–2020);
                                    abbreviation regions (AB, BC, …)
    nir_to_cims.py                → per-gas kt per residential CIMS branch,
                                    annual resolution (2000–latest NIR year);
                                    summed to tCO2e using AR5 GWP100 factors;
                                    full province names mapped to abbreviations

Energy demand  (calibration_quantity_requested)
    cer_resd_demand.py            → energy demand in PJ by fuel and CIMS node;
                                    abbreviation regions

Heating and water heating technologies  (calibration_market_share_total)
    residential.py                → CEUD-derived market shares with projections;
                                    one DataFrame per province/territory

Output columns
--------------
Branch, Type, Region, Sector, Service, Technology, Parameter,
Context, Sub_Context, Target, Source, Unit, Year, Value
"""

from pathlib import Path

import polars as pl

# ── path setup ────────────────────────────────────────────────────────────────


import CIMS.data_processing.source.eccc.nir.nir_crosswalk_tables_cims as _crosswalk_mod
import CIMS.data_processing.source.eccc.nir.nir_to_cims as _nir_mod
import CIMS.data_processing.source.cer.cer_resd_demand as _cer_mod
from CIMS.data_processing.source.nrcan.ceud.residential.residential import main as _ceud_main
from CIMS.data_processing.utils.controls_conversions import BASE_PATH, load_sector_regions, filter_excluded_branches
from CIMS.data_processing.utils.calibration_helpers import (
    OUTPUT_COLS,
    build_cer_energy,
    build_crosswalk_emissions,
    build_nir_emissions,
    empty_df,
)
from CIMS.data_processing.utils.output_builder import write_per_region_csvs

# ── configuration ─────────────────────────────────────────────────────────────
OUTPUT_DIR = BASE_PATH / 'calibration/residential'

SECTOR_NAME = 'Residential'

# ── helpers ───────────────────────────────────────────────────────────────────

def _get_series(df: pl.DataFrame, variable: str, category: str = '') -> dict:
    """Extract {year: value} from a long-format Polars DataFrame."""
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
    """Return sorted unique category values for a given variable."""
    subset = df.filter(pl.col('variable') == variable)
    return sorted(subset.get_column('category').unique().to_list())


# ── technology builders ───────────────────────────────────────────────────────

def _build_heating_techs(ceud_results: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Extract heating technology market shares from CEUD data.

    BC exports both Marine and Cold climate; all other provinces export Cold only.
    """
    rows: list[dict] = []

    for prov_code, df in ceud_results.items():
        prov   = prov_code.upper()
        is_bc  = prov == 'BC'

        lowmed_vintages = _get_categories(df, 'vintage_bins_lowmed')
        high_vintages   = _get_categories(df, 'vintage_bins_high')

        climates = (
            [('heating_lowmed_marine', 'heating_high_marine', 'Marine'),
             ('heating_lowmed_cold',   'heating_high_cold',   'Cold')]
            if is_bc else
            [('heating_lowmed_cold', 'heating_high_cold', 'Cold')]
        )

        for lowmed_var, high_var, climate_label in climates:
            for vint in lowmed_vintages:
                for tech in _get_categories(df, lowmed_var):
                    for year, value in _get_series(df, lowmed_var, tech).items():
                        rows.append({
                            'Branch':      (f'CIMS.CAN.{prov}.Residential.Dwellings.Building Type'
                                            f'.LowMed Density.Vintage.{vint} Bldg Code'
                                            f'.Heating ({climate_label})'),
                            'Type':        'Service',
                            'Region':      prov,
                            'Sector':      'Residential',
                            'Service':     'Heating',
                            'Technology':  tech,
                            'Parameter':   'calibration_market_share_total',
                            'Context':     '',
                            'Sub_Context': '',
                            'Target':      '',
                            'Source':      'CEUD',
                            'Unit':        '%',
                            'Year':        str(year),
                            'Value':       str(value),
                        })

            for vint in high_vintages:
                for tech in _get_categories(df, high_var):
                    for year, value in _get_series(df, high_var, tech).items():
                        rows.append({
                            'Branch':      (f'CIMS.CAN.{prov}.Residential.Dwellings.Building Type'
                                            f'.High Density.Vintage.{vint} Bldg Code'
                                            f'.Heating ({climate_label})'),
                            'Type':        'Service',
                            'Region':      prov,
                            'Sector':      'Residential',
                            'Service':     'Heating',
                            'Technology':  tech,
                            'Parameter':   'calibration_market_share_total',
                            'Context':     '',
                            'Sub_Context': '',
                            'Target':      '',
                            'Source':      'CEUD',
                            'Unit':        '%',
                            'Year':        str(year),
                            'Value':       str(value),
                        })

    if not rows:
        return empty_df()
    return pl.DataFrame(rows, schema={c: pl.Utf8 for c in OUTPUT_COLS})


def _build_vintage_composition(ceud_results: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """
    Extract vintage-bin housing-stock composition market shares from CEUD
    data (vintage_bins_lowmed/vintage_bins_high, sourced from CEUD Tables
    19-20 via extract_vintages in residential.py).

    The "Vintage" node (Building Type.{Density}.Vintage) is a genuine
    tech-compete node whose "technologies" are the vintage bins themselves
    (<1960, 1961-1980, ...) -- which bin a unit of new construction lands in
    is a real competition outcome with FIC/lifetime levers, structurally the
    same as Heating (Cold)'s fuel competition. Previously this CEUD series
    was only used to enumerate vintage categories for `_build_heating_techs`
    and, downstream, only its year-2000 slice was read as a one-off anchor
    (`_build_vintage_bin_rows`/`bin_share` in residential_model_inputs.py) --
    its real annual values for every other year went unused. Exporting the
    full series here lets `optimize_ms_via_fics_and_lifetimes` (generic to
    any tech-compete node with a calibration_market_share_total target, no
    optimizer changes needed) calibrate the Vintage node the same way it
    already calibrates Heating/Water Heating fuel choice.

    CEUD has no data for '>2035' (nothing can have been built after 2035
    yet), so that bin gets no calibration target here -- same gap
    `vintage_bins_lowmed`/`vintage_bins_high` themselves already have.
    """
    rows: list[dict] = []

    for prov_code, df in ceud_results.items():
        prov = prov_code.upper()

        for variable, density in [('vintage_bins_lowmed', 'LowMed Density'),
                                  ('vintage_bins_high', 'High Density')]:
            branch = f'CIMS.CAN.{prov}.Residential.Dwellings.Building Type.{density}.Vintage'
            for vint in _get_categories(df, variable):
                for year, value in _get_series(df, variable, vint).items():
                    rows.append({
                        'Branch':      branch,
                        'Type':        'Service',
                        'Region':      prov,
                        'Sector':      'Residential',
                        'Service':     'Vintage',
                        'Technology':  vint,
                        'Parameter':   'calibration_market_share_total',
                        'Context':     '',
                        'Sub_Context': '',
                        'Target':      '',
                        'Source':      'CEUD',
                        'Unit':        '%',
                        'Year':        str(year),
                        'Value':       str(value),
                    })

    if not rows:
        return empty_df()
    return pl.DataFrame(rows, schema={c: pl.Utf8 for c in OUTPUT_COLS})


def _build_wh_techs(ceud_results: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Extract water heating technology market shares from CEUD data."""
    rows: list[dict] = []

    for prov_code, df in ceud_results.items():
        prov = prov_code.upper()

        for var, density_label in [('wh_tech_lowmed', 'LowMed Density'),
                                    ('wh_tech_high',   'High Density')]:
            for tech in _get_categories(df, var):
                for year, value in _get_series(df, var, tech).items():
                    rows.append({
                        'Branch':      f'CIMS.CAN.{prov}.Residential.Water Heating.{density_label}',
                        'Type':        'Service',
                        'Region':      prov,
                        'Sector':      'Residential',
                        'Service':     density_label,
                        'Technology':  tech,
                        'Parameter':   'calibration_market_share_total',
                        'Context':     '',
                        'Sub_Context': '',
                        'Target':      '',
                        'Source':      'CEUD',
                        'Unit':        '%',
                        'Year':        str(year),
                        'Value':       str(value),
                    })

    if not rows:
        return empty_df()
    return pl.DataFrame(rows, schema={c: pl.Utf8 for c in OUTPUT_COLS})


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> pl.DataFrame:
    """Assemble residential calibration data and write one CSV per region."""
    print('=' * 60)
    print('RESIDENTIAL CALIBRATION')
    print('=' * 60)

    print('\nRunning NIR crosswalk (nir_crosswalk_tables_cims)...')
    crosswalk_df = pl.from_pandas(_crosswalk_mod.main())

    print('\nRunning NIR to CIMS (nir_to_cims)...')
    nir_df = _nir_mod.main()

    print('\nRunning CER demand (cer_resd_demand)...')
    cer_df = _cer_mod.main()

    print('\nRunning CEUD residential pipeline (with projections)...')
    ceud_results = _ceud_main(apply_projections=True, export_csv=False)

    print('\nBuilding CER energy demand rows...')
    cer_rows = build_cer_energy(cer_df, SECTOR_NAME)
    print(f'  Rows: {len(cer_rows):,}')

    print('Building crosswalk emission rows...')
    crosswalk_rows = build_crosswalk_emissions(crosswalk_df, SECTOR_NAME)
    print(f'  Rows: {len(crosswalk_rows):,}')

    print('Building NIR annual emission rows (tCO2e via AR5 GWP100)...')
    nir_rows = build_nir_emissions(nir_df, SECTOR_NAME)
    print(f'  Rows: {len(nir_rows):,}')

    print('Building heating technology rows...')
    heating_rows = _build_heating_techs(ceud_results)
    print(f'  Rows: {len(heating_rows):,}')

    print('Building water heating technology rows...')
    wh_rows = _build_wh_techs(ceud_results)
    print(f'  Rows: {len(wh_rows):,}')

    print('Building vintage-bin composition rows...')
    vintage_rows = _build_vintage_composition(ceud_results)
    print(f'  Rows: {len(vintage_rows):,}')

    print('Combining...')
    output = (
        pl.concat([cer_rows, crosswalk_rows, nir_rows, heating_rows, wh_rows, vintage_rows],
                  how='diagonal_relaxed')
        .select(OUTPUT_COLS)
    )

    print('Filtering to regions with Residential (see sector_region_map.csv)...')
    allowed_regions = load_sector_regions().get(SECTOR_NAME)
    if allowed_regions:
        before_count = len(output)
        output = output.filter(pl.col('Region').is_in(list(allowed_regions)))
        dropped_count = before_count - len(output)
        if dropped_count:
            print(f'  Dropped {dropped_count:,} rows for regions without Residential')

    output = filter_excluded_branches(output)

    regions = write_per_region_csvs(output, OUTPUT_DIR, 'residential', skip_if_all_zero=True)

    print(f'\n✅ Residential calibration complete')
    print(f'   Total rows:  {len(output):,}')
    print(f'   Files:       {len(regions)} (one per region)')

    return output


if __name__ == '__main__':
    main()
