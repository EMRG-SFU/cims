"""
Energy Price Multipliers Processing System
Calculates price multipliers by dividing end-use prices by production costs.

Output: Multipliers for all CIMS energies, sectors, and regions for years 2000-2100.
Uses Polars for CSV reading, pandas for Excel and Series operations.
"""

import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Robust path setup using __file__
# File lives at:  C:/cims/data/pipeline/source/energy_prices
# parent.parent.parent resolves to: C:/cims/data/pipeline/
# parent (the energy_prices dir itself) is also added so energy_prices.py is importable.
import sys
_current_file = Path(__file__)
_project_root = _current_file.parent.parent.parent
_energy_prices_dir = _current_file.parent
for _p in (_project_root, _energy_prices_dir):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from utils.controls_conversions import (
    load_macro_indicators,
    convert_currency,
    load_control_config,
)
from utils.data_fill import backfill_constant, interpolate_5year_to_annual
from utils.data_extensions import extend_constant
from utils.extractors.cer import find_cer_file
from utils.controls_conversions import DATA_START, PROJECTION_END, BASE_PATH, LAST_DATA_YEAR, ARCHIVED_DATA
from energy_prices import main as get_production_costs


# Configuration
MAPPINGS_PATH    = BASE_PATH / 'mappings_conversions'
CONTROL_FILE     = MAPPINGS_PATH / 'control.py'
ENERGY_MAP_FILE  = MAPPINGS_PATH / 'energy_map.csv'
REGION_MAP_FILE  = MAPPINGS_PATH / 'region_map.csv'
SECTOR_MAP_FILE  = MAPPINGS_PATH / 'sector_map.csv'
SECTOR_REGION_MAP_FILE = MAPPINGS_PATH / 'sector_region_map.csv'
CER_DIR             = BASE_PATH / 'raw_data/cer'
MACRO_FILE          = find_cer_file(CER_DIR, 'macro-indicators')
END_USE_PRICES_FILE = find_cer_file(CER_DIR, 'end-use-prices')
CIMS_PRICES_FILE = BASE_PATH / 'raw_data/energy_prices/CIMS Prices and Calcs.xlsx'
OUTPUT_DIR = BASE_PATH / 'processed_data/energy_prices'






def load_control_data() -> Dict:
    """
    Load control and mapping data from CSV files.

    Returns
    -------
    dict
        Dictionary containing sector, region, and energy mapping DataFrames,
        as well as derived lists and lookup dictionaries.
    """

    sectors_df = pd.read_csv(SECTOR_MAP_FILE)
    regions_df = pd.read_csv(REGION_MAP_FILE)
    energy_df  = pd.read_csv(ENERGY_MAP_FILE)
    sector_region_df = pd.read_csv(SECTOR_REGION_MAP_FILE)

    cims_sectors = sectors_df['CIMS'].unique().tolist()
    cims_regions = [r for r in regions_df['CIMS'].unique().tolist() if r != 'CAN']
    cims_energies = energy_df['CIMS'].dropna().unique().tolist()

    cer_to_cims_region: Dict[str, str] = {}
    jcims_to_cims_region: Dict[str, str] = {}
    for _, row in regions_df.iterrows():
        if pd.notna(row['CER']) and pd.notna(row['CIMS']):
            cer_to_cims_region[row['CER']] = row['CIMS']
        if pd.notna(row.get('JCIMS')) and pd.notna(row['CIMS']):
            jcims_to_cims_region[row['JCIMS']] = row['CIMS']

    # Sector -> set of CIMS regions that sector actually has fixed structural
    # data in. Used to drop sector/region combinations (e.g. Coal Mining in
    # PE/NL) that would otherwise leak into every sector's output because the
    # multiplier calculators build a full sector x region cross product.
    sector_regions: Dict[str, set] = {
        row['Sector']: set(row['Regions'].split(';'))
        for _, row in sector_region_df.iterrows()
        if pd.notna(row.get('Regions'))
    }

    return {
        'sectors_df': sectors_df,
        'regions_df': regions_df,
        'energy_df': energy_df,
        'cims_sectors': cims_sectors,
        'cims_regions': cims_regions,
        'cims_energies': cims_energies,
        'cer_to_cims_region': cer_to_cims_region,
        'jcims_to_cims_region': jcims_to_cims_region,
        'sector_regions': sector_regions,
    }


def load_price_data(control_data: Dict) -> Dict:
    """
    Load all price data sources.

    Parameters
    ----------
    control_data : dict
        Control data dictionary from load_control_data().

    Returns
    -------
    dict
        Dictionary containing end-use prices, electricity multipliers,
        JCIMS prices, macro indicators, and the active scenario name.
    """

    end_use_prices = pl.read_csv(END_USE_PRICES_FILE)
    elec_mult_df = pd.read_excel(CIMS_PRICES_FILE, sheet_name='CIMS Electricity')
    jcims_prices_df = pd.read_excel(CIMS_PRICES_FILE, sheet_name='JCIMS Prices')

    config = load_control_config()
    scenario = config.get('cer_ef_reference_scenario', 'Current Measures')
    macro_df = load_macro_indicators(MACRO_FILE, scenario)

    return {
        'end_use_prices': end_use_prices,
        'elec_mult_df': elec_mult_df,
        'jcims_prices_df': jcims_prices_df,
        'macro_df': macro_df,
        'scenario': scenario,
    }


def calculate_electricity_multipliers(
    elec_mult_df: pd.DataFrame,
    sectors: List[str],
    sectors_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate electricity multipliers from the CIMS Electricity tab.

    Transportation sectors use Commercial multipliers. Each CIMS sector is
    mapped to its CER grouping (Residential, Commercial, Industrial) to select
    the appropriate multiplier.

    Parameters
    ----------
    elec_mult_df : pd.DataFrame
        CIMS electricity multiplier data (from Excel).
    sectors : list of str
        List of CIMS sector names.
    sectors_df : pd.DataFrame
        Sector mapping DataFrame.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns: region, sector, energy, year, multiplier, source.
    """

    sector_to_group: Dict[str, str] = {}
    for _, row in sectors_df.iterrows():
        cims_sector = row.get('CIMS')
        cer_group = row.get('CER (Grouped)')
        if pd.notna(cims_sector) and pd.notna(cer_group):
            sector_to_group[cims_sector] = cer_group

    rows = []
    for _, row in elec_mult_df.iterrows():
        region = row['Region']
        multipliers_by_group = {
            'Residential': row['Residential'],
            'Commercial': row['Commercial'],
            'Industrial': row['Industrial'],
            'Transportation': row['Commercial'],  # Transportation uses Commercial
        }

        for sector in sectors:
            cer_group = sector_to_group.get(sector, 'Industrial')
            multiplier = multipliers_by_group.get(cer_group, multipliers_by_group['Industrial'])

            for year in range(DATA_START, PROJECTION_END + 1):
                rows.append({
                    'region': region,
                    'sector': sector,
                    'energy': 'Electricity',
                    'year': year,
                    'multiplier': multiplier,
                    'source': 'CER',
                })

    return pd.DataFrame(rows)


def calculate_hydrogen_multipliers(
    regions: List[str],
    sectors: List[str],
    sectors_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate hydrogen multipliers.

    Residential and Commercial sectors use a multiplier of 1.1;
    Transportation and Industrial sectors use 1.0.

    Parameters
    ----------
    regions : list of str
        List of CIMS region codes.
    sectors : list of str
        List of CIMS sector names.
    sectors_df : pd.DataFrame
        Sector mapping DataFrame.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns: region, sector, energy, year, multiplier, source.
    """

    sector_groups: Dict[str, str] = {}
    for _, row in sectors_df.iterrows():
        cims_sector = row['CIMS']
        cer_group = row['CER (Grouped)']
        if pd.notna(cims_sector) and pd.notna(cer_group):
            sector_groups[cims_sector] = cer_group

    rows = []
    for region in regions:
        for sector in sectors:
            cer_group = sector_groups.get(sector, 'Industrial')
            multiplier = 1.1 if cer_group in ['Residential', 'Commercial'] else 1.0

            for year in range(DATA_START, PROJECTION_END + 1):
                rows.append({
                    'region': region,
                    'sector': sector,
                    'energy': 'Hydrogen',
                    'year': year,
                    'multiplier': multiplier,
                    'source': 'Assumptions',
                })

    return pd.DataFrame(rows)


def calculate_end_use_energy_multipliers(
    end_use_prices: pl.DataFrame,
    production_costs: pd.DataFrame,
    regions: List[str],
    sectors: List[str],
    sectors_df: pd.DataFrame,
    cer_to_cims_region: Dict[str, str],
    scenario: str,
) -> pd.DataFrame:
    """
    Calculate multipliers for Diesel, Gasoline, Heavy/Light Fuel Oil, and Natural Gas.

    Divides end-use prices by production costs, then applies cross-sector rules
    (e.g. Transportation inherits Commercial Natural Gas multipliers).

    Parameters
    ----------
    end_use_prices : pl.DataFrame
        CER end-use prices data.
    production_costs : pd.DataFrame
        Production costs from process_energy_prices.main().
    regions : list of str
        List of CIMS region codes.
    sectors : list of str
        List of CIMS sector names.
    sectors_df : pd.DataFrame
        Sector mapping DataFrame.
    cer_to_cims_region : dict
        Mapping from CER full region names to CIMS codes.
    scenario : str
        CER scenario name.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns: region, sector, energy, year, multiplier, source.
    """

    energy_mapping = {
        'Diesel': 'Diesel',
        'Gasoline': 'Gasoline',
        'Oil': {
            'Residential': 'Light Fuel Oil',
            'Commercial': 'Light Fuel Oil',
            'Industrial': 'Heavy Fuel Oil',
        },
        'Natural Gas': 'Natural Gas',
    }

    # Filter to scenario in Polars, then extract rows as plain Python dicts
    # using .to_struct().to_list() — no pyarrow needed.
    end_use_rows = (
        end_use_prices
        .filter(pl.col('Scenario') == scenario)
        .select(['Year', 'Area', 'Sector', 'Fuel', 'Sum of Price'])
        .to_struct(name='r')
        .to_list()
    )

    _max_cer_year = LAST_DATA_YEAR["cer"]

    sector_groups: Dict[str, str] = {}
    for _, row in sectors_df.iterrows():
        cims_sector = row['CIMS']
        cer_group = row['CER (Grouped)']
        if pd.notna(cims_sector) and pd.notna(cer_group):
            sector_groups[cims_sector] = cer_group

    base_multipliers: Dict[Tuple, float] = {}

    for fuel_name in energy_mapping.keys():
        for price_row in end_use_rows:
            if price_row['Fuel'] != fuel_name:
                continue

            year           = price_row['Year']
            cer_region     = price_row['Area']
            cer_sector     = price_row['Sector']
            end_use_price  = price_row['Sum of Price']

            if cer_region == 'Canada':
                continue

            cims_region = cer_to_cims_region.get(cer_region, cer_region)

            if isinstance(energy_mapping[fuel_name], dict):
                cims_energy = energy_mapping[fuel_name].get(cer_sector, 'Heavy Fuel Oil')
            else:
                cims_energy = energy_mapping[fuel_name]

            prod_cost_row = production_costs[
                (production_costs['Energy'] == cims_energy) &
                (production_costs['Region'] == 'generic') &
                (production_costs['Year'] == year)
            ]

            if len(prod_cost_row) == 0:
                continue

            prod_cost = prod_cost_row['Price'].values[0]
            multiplier = end_use_price / prod_cost if prod_cost > 0 else 1.0
            base_multipliers[(cims_region, cer_sector, cims_energy, year)] = multiplier

    # Interpolate and extend base multipliers to 2000-2100
    grouped: Dict[Tuple, Dict] = {}
    for (region, sector, energy, year), mult in base_multipliers.items():
        key = (region, sector, energy)
        if key not in grouped:
            grouped[key] = {}
        grouped[key][year] = mult

    extended_base_multipliers: Dict[Tuple, float] = {}
    for (region, sector, energy), year_mults in grouped.items():
        mult_series = pd.Series(year_mults)
        extended_series = backfill_constant(mult_series, DATA_START)
        extended_series = extend_constant(extended_series, PROJECTION_END)
        for year, mult in extended_series.items():
            if pd.notna(mult):
                extended_base_multipliers[(region, sector, energy, int(year))] = mult

    base_multipliers = extended_base_multipliers

    rows = []
    for cims_region in regions:
        if cims_region == 'CAN':
            continue

        for year in range(DATA_START, PROJECTION_END + 1):
            # Heavy Fuel Oil: Industrial base, 1.1× for Residential/Commercial
            hfo_ind_key = (cims_region, 'Industrial', 'Heavy Fuel Oil', year)
            if hfo_ind_key in base_multipliers:
                hfo_ind_mult = base_multipliers[hfo_ind_key]
                for sector in sectors:
                    cer_group = sector_groups.get(sector, 'Industrial')
                    mult = hfo_ind_mult * 1.1 if cer_group in ['Residential', 'Commercial'] else hfo_ind_mult
                    rows.append({'region': cims_region, 'sector': sector,
                                 'energy': 'Heavy Fuel Oil', 'year': year, 'multiplier': mult,
                                 'source': 'CER' if year <= _max_cer_year else 'Assumptions'})

            # Light Fuel Oil: Commercial/Residential base, 10% lower for Industrial/Transport
            lfo_base_mult = (
                base_multipliers.get((cims_region, 'Commercial', 'Light Fuel Oil', year)) or
                base_multipliers.get((cims_region, 'Residential', 'Light Fuel Oil', year))
            )
            if lfo_base_mult is not None:
                for sector in sectors:
                    cer_group = sector_groups.get(sector, 'Industrial')
                    mult = lfo_base_mult if cer_group in ['Residential', 'Commercial'] else lfo_base_mult * 0.9
                    rows.append({'region': cims_region, 'sector': sector,
                                 'energy': 'Light Fuel Oil', 'year': year, 'multiplier': mult,
                                 'source': 'CER' if year <= _max_cer_year else 'Assumptions'})

            # Diesel: Transportation base, 1.1× for Residential/Commercial
            diesel_trans_key = (cims_region, 'Transportation', 'Diesel', year)
            if diesel_trans_key in base_multipliers:
                diesel_trans_mult = base_multipliers[diesel_trans_key]
                for sector in sectors:
                    cer_group = sector_groups.get(sector, 'Industrial')
                    mult = diesel_trans_mult * 1.1 if cer_group in ['Residential', 'Commercial'] else diesel_trans_mult
                    rows.append({'region': cims_region, 'sector': sector,
                                 'energy': 'Diesel', 'year': year, 'multiplier': mult,
                                 'source': 'CER' if year <= _max_cer_year else 'Assumptions'})

            # Gasoline: Transportation base, 1.1× for Residential/Commercial
            gas_trans_key = (cims_region, 'Transportation', 'Gasoline', year)
            if gas_trans_key in base_multipliers:
                gas_trans_mult = base_multipliers[gas_trans_key]
                for sector in sectors:
                    cer_group = sector_groups.get(sector, 'Industrial')
                    mult = gas_trans_mult * 1.1 if cer_group in ['Residential', 'Commercial'] else gas_trans_mult
                    rows.append({'region': cims_region, 'sector': sector,
                                 'energy': 'Gasoline', 'year': year, 'multiplier': mult,
                                 'source': 'CER' if year <= _max_cer_year else 'Assumptions'})

            # Natural Gas: Transportation uses Commercial multipliers
            for cer_sector in ['Residential', 'Commercial', 'Industrial', 'Transportation']:
                lookup_sector = 'Commercial' if cer_sector == 'Transportation' else cer_sector
                ng_key = (cims_region, lookup_sector, 'Natural Gas', year)
                if ng_key in base_multipliers:
                    ng_mult = base_multipliers[ng_key]
                    for sector in sectors:
                        if sector_groups.get(sector) == cer_sector:
                            rows.append({'region': cims_region, 'sector': sector,
                                         'energy': 'Natural Gas', 'year': year, 'multiplier': ng_mult,
                                         'source': 'CER' if year <= _max_cer_year else 'Assumptions'})

    return pd.DataFrame(rows)


def calculate_simple_fuel_multipliers(
    regions: List[str],
    sectors: List[str],
    sectors_df: pd.DataFrame,
    max_cer_year: int,
) -> pd.DataFrame:
    """
    Calculate multipliers for fuels with fixed values.

    Applies 1.25 for Industrial sectors and 1.5 for all other sectors
    (Residential, Commercial, Transportation) for: Asphalt, Lubricants,
    Petrochemical Feedstock, Naphtha Specialties, Other Non-Energy Products.

    Parameters
    ----------
    regions : list of str
        List of CIMS region codes.
    sectors : list of str
        List of CIMS sector names.
    sectors_df : pd.DataFrame
        Sector mapping DataFrame.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns: region, sector, energy, year, multiplier, source.
    """

    simple_fuels = [
        'Asphalt',
        'Lubricants',
        'Petrochemical Feedstock',
        'Naphtha Specialties',
        'Other Non-Energy Products',
    ]

    sector_groups: Dict[str, str] = {}
    for _, row in sectors_df.iterrows():
        cims_sector = row['CIMS']
        cer_group = row['CER (Grouped)']
        if pd.notna(cims_sector) and pd.notna(cer_group):
            sector_groups[cims_sector] = cer_group

    rows = []
    for energy in simple_fuels:
        for region in regions:
            for sector in sectors:
                cer_group = sector_groups.get(sector, 'Industrial')
                multiplier = 1.25 if cer_group == 'Industrial' else 1.5

                for year in range(DATA_START, PROJECTION_END + 1):
                    rows.append({
                        'region': region,
                        'sector': sector,
                        'energy': energy,
                        'year': year,
                        'multiplier': multiplier,
                        'source': 'CER' if year <= max_cer_year else 'Assumptions',
                    })

    return pd.DataFrame(rows)


def calculate_jcims_multipliers(
    jcims_prices_df: pd.DataFrame,
    production_costs: pd.DataFrame,
    macro_df: pl.DataFrame,
    regions: List[str],
    sectors: List[str],
    jcims_to_cims_region: Dict[str, str],
    energy_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate multipliers for JCIMS fuels using the JCIMS Prices tab.

    Divides JCIMS prices by production costs for all region/sector/fuel
    combinations, with fallback logic for missing data.

    Parameters
    ----------
    jcims_prices_df : pd.DataFrame
        JCIMS prices data (from Excel).
    production_costs : pd.DataFrame
        Production costs from process_energy_prices.main().
    macro_df : pl.DataFrame
        Macro indicators for currency conversion.
    regions : list of str
        List of CIMS region codes.
    sectors : list of str
        List of CIMS sector names.
    jcims_to_cims_region : dict
        Mapping from JCIMS region codes to CIMS codes.
    energy_df : pd.DataFrame
        Energy mapping DataFrame.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns: region, sector, energy, year, multiplier.
    """

    # Build one-to-many mapping: one JCIMS fuel can map to multiple CIMS fuels
    # (e.g. JCIMS 'Natural Gas' -> CIMS 'Natural Gas' AND 'Natural Gas Feedstock')
    jcims_to_cims_fuels: Dict[str, List[str]] = {}
    for _, row in energy_df.iterrows():
        if pd.notna(row.get('JCIMS')) and pd.notna(row.get('CIMS')):
            jcims_to_cims_fuels.setdefault(row['JCIMS'], []).append(row['CIMS'])


    # Translate JCIMS sector names to CIMS sector names so base_multipliers keys
    # are consistent with the CIMS sector names used in the output loop.
    jcims_to_cims_sector: Dict[str, str] = {
        'Natural Gas Extraction':  'Natural Gas',
        'Other Manufacturing':     'Light Industrial',
    }

    use_light_ind = ['Construction', 'Forestry', 'Hydrogen']
    base_multipliers: Dict[Tuple, float] = {}
    year_cols = [col for col in jcims_prices_df.columns if isinstance(col, (int, str)) and str(col).isdigit()]
    _max_jcims_year = ARCHIVED_DATA["jcims_prices"]

    for _, row in jcims_prices_df.iterrows():
        jcims_region = row['Region']
        jcims_sector = row['Sector']
        jcims_fuel   = row['Fuel']

        # Translate sector name to CIMS equivalent where they differ
        cims_sector = jcims_to_cims_sector.get(jcims_sector, jcims_sector)

        cims_fuels = jcims_to_cims_fuels.get(jcims_fuel)
        if not cims_fuels:
            continue

        for year_col in year_cols:
            year = int(year_col)
            jcims_price = row[year_col]

            if pd.isna(jcims_price) or jcims_price <= 0:
                continue

            jcims_price_series = pd.Series({year: jcims_price})
            jcims_price_2025 = convert_currency(
                jcims_price_series,
                from_year=2005,
                to_year=2025,
                from_currency='CAD',
                to_currency='CAD',
                macro_df=macro_df,
                constant_dollars=True,
            )
            jcims_price_converted = jcims_price_2025.values[0]

            # Store a multiplier for every CIMS fuel this JCIMS fuel maps to
            for cims_fuel in cims_fuels:
                prod_cost_row = production_costs[
                    (production_costs['Energy'] == cims_fuel) &
                    (production_costs['Region'] == 'generic') &
                    (production_costs['Year'] == year)
                ]

                if len(prod_cost_row) == 0:
                    continue

                prod_cost = prod_cost_row['Price'].values[0]
                multiplier = jcims_price_converted / prod_cost if prod_cost > 0 else 1.0
                base_multipliers[(jcims_region, cims_sector, cims_fuel, year)] = multiplier

    def get_multiplier_with_fallback(
        jcims_region: str,
        sector: str,
        fuel: str,
        year: int,
    ) -> Optional[float]:
        """
        Get a multiplier with cascading fallback logic for missing data.

        Parameters
        ----------
        jcims_region : str
            JCIMS region code.
        sector : str
            JCIMS sector name.
        fuel : str
            CIMS fuel name.
        year : int
            Year.

        Returns
        -------
        float or None
            Multiplier value, or None if no fallback succeeds.
        """
        key = (jcims_region, sector, fuel, year)
        if key in base_multipliers:
            return base_multipliers[key]

        # Fallback regions must be JCIMS codes (AT covers NB/NS/PE/NL)
        for other_region in ['AB', 'BC', 'ON', 'SK', 'MB', 'QC', 'AT']:
            key = (other_region, sector, fuel, year)
            if key in base_multipliers:
                return base_multipliers[key]

        key = (jcims_region, 'Light Industrial', fuel, year)
        if key in base_multipliers:
            return base_multipliers[key]

        for other_region in ['AB', 'BC', 'ON', 'SK', 'MB', 'QC', 'AT']:
            key = (other_region, 'Light Industrial', fuel, year)
            if key in base_multipliers:
                return base_multipliers[key]

        if fuel in ['Propane', 'Petroleum Coke', 'Uranium', 'Waste', 'Coke']:
            return 1.0

        return None

    # Build a direct CIMS->JCIMS region lookup from the region_map passed in.
    # jcims_to_cims_region has AT->NL (last-write-wins) so we can't reverse it safely.
    # Instead rebuild from the regions_df stored on the passed dict's inverse entries.
    # We derive it here by inverting all unique pairs we know about:
    #   BC->BC, AB->AB, SK->SK, MB->MB, ON->ON, QC->QC, NB->AT, NS->AT, PE->AT, NL->AT
    cims_to_jcims_region: Dict[str, str] = {}
    for j_reg, c_reg in jcims_to_cims_region.items():
        # jcims_to_cims_region may only have AT->NL; we need ALL CIMS->JCIMS pairs.
        # Rebuild by iterating the original mapping in reverse for all known entries.
        cims_to_jcims_region[c_reg] = j_reg
    # The above still only captures one CIMS per JCIMS code for shared codes like AT.
    # Hard-code the Atlantic provinces that share AT:
    for atlantic in ['NB', 'NS', 'PE', 'NL']:
        cims_to_jcims_region[atlantic] = 'AT'

    # All unique CIMS fuel names produced by the JCIMS path
    jcims_fuels = list({cims_f for fuels in jcims_to_cims_fuels.values() for cims_f in fuels})
    rows = []

    for cims_region in regions:
        if cims_region == 'CAN':
            continue

        jcims_region = cims_to_jcims_region.get(cims_region)

        if cims_region in ['YT', 'NT', 'NU']:
            jcims_region = 'BC'

        if jcims_region is None:
            continue

        for cims_sector in sectors:
            source_sector = 'Light Industrial' if cims_sector in use_light_ind else cims_sector

            for fuel in jcims_fuels:
                for year in range(DATA_START, PROJECTION_END + 1):
                    mult = get_multiplier_with_fallback(jcims_region, source_sector, fuel, year)
                    if mult is not None:
                        rows.append({
                            'region': cims_region,
                            'sector': cims_sector,
                            'energy': fuel,
                            'year': year,
                            'multiplier': mult,
                        })

    df = pd.DataFrame(rows)

    if len(df) == 0:
        return df

    interpolated_rows = []
    for (region, sector, energy), group in df.groupby(['region', 'sector', 'energy']):
        multipliers_series = group.set_index('year')['multiplier'].sort_index()
        annual_mult = interpolate_5year_to_annual(multipliers_series)
        extended_mult = backfill_constant(annual_mult, DATA_START)
        extended_mult = extend_constant(extended_mult, PROJECTION_END)

        for year, mult in extended_mult.items():
            if pd.notna(mult):
                interpolated_rows.append({
                    'region': region,
                    'sector': sector,
                    'energy': energy,
                    'year': int(year),
                    'multiplier': mult,
                    'source': 'JCIMS' if int(year) <= _max_jcims_year else 'Assumptions',
                })

    return pd.DataFrame(interpolated_rows)


def apply_derivative_multipliers(
    base_multipliers: pd.DataFrame,
    regions: List[str],
    sectors: List[str],
) -> pd.DataFrame:
    """
    Apply derivative fuel multipliers that copy from base fuels.

    Direct copies: Natural Gas Feedstock and RNG ← Natural Gas; Kerosene ← Diesel;
    Biodiesel and Renewable Diesel ← Diesel; Ethanol and Renewable Gasoline ← Gasoline;
    SAF ← Jet Fuel.

    Parameters
    ----------
    base_multipliers : pd.DataFrame
        Combined base multipliers DataFrame.
    regions : list of str
        List of CIMS region codes.
    sectors : list of str
        List of CIMS sector names.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame of derivative multipliers.
    """

    direct_mappings = {
        'Natural Gas Feedstock': 'Natural Gas',
        'Renewable Natural Gas': 'Natural Gas',
        'Kerosene': 'Diesel',
        'Biodiesel': 'Diesel',
        'Renewable Diesel': 'Diesel',
        'Ethanol': 'Gasoline',
        'Renewable Gasoline': 'Gasoline',
        'SAF': 'Jet Fuel',
        'Ethane': 'Propane',
        'Butane': 'Propane'
    }

    _AFDC_FUELS = {'Biodiesel', 'Renewable Diesel', 'Ethanol', 'Renewable Gasoline'}

    rows = []
    for new_fuel, base_fuel in direct_mappings.items():
        base_data = base_multipliers[base_multipliers['energy'] == base_fuel].copy()
        base_data['energy'] = new_fuel
        if new_fuel in _AFDC_FUELS:
            base_data['source'] = base_data['year'].apply(
                lambda y: 'AFDC' if y <= LAST_DATA_YEAR["afdc"] else 'Assumptions'
            )
        else:
            base_data['source'] = 'Assumptions'
        rows.append(base_data)

    # TODO: Add logic for sector-specific derivatives (Light/Heavy Fuel Oil, etc.)

    if rows:
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame()


def main() -> pd.DataFrame:
    """
    Main execution function.

    Returns
    -------
    pd.DataFrame
        Final output DataFrame of price multipliers.
    """

    control_data = load_control_data()
    price_data = load_price_data(control_data)

    production_costs = get_production_costs()

    all_multipliers = []
    failed = []

    steps = [
        ("Electricity multipliers", lambda: calculate_electricity_multipliers(
            price_data['elec_mult_df'],
            control_data['cims_sectors'],
            control_data['sectors_df'],
        )),
        ("Hydrogen multipliers", lambda: calculate_hydrogen_multipliers(
            control_data['cims_regions'],
            control_data['cims_sectors'],
            control_data['sectors_df'],
        )),
        ("End-use energy multipliers", lambda: calculate_end_use_energy_multipliers(
            price_data['end_use_prices'],
            production_costs,
            control_data['cims_regions'],
            control_data['cims_sectors'],
            control_data['sectors_df'],
            control_data['cer_to_cims_region'],
            price_data['scenario'],
        )),
        ("Simple fuel multipliers", lambda: calculate_simple_fuel_multipliers(
            control_data['cims_regions'],
            control_data['cims_sectors'],
            control_data['sectors_df'],
            int(price_data['end_use_prices']
                .filter(pl.col('Scenario') == price_data['scenario'])['Year'].max()),
        )),
        ("JCIMS multipliers", lambda: calculate_jcims_multipliers(
            price_data['jcims_prices_df'],
            production_costs,
            price_data['macro_df'],
            control_data['cims_regions'],
            control_data['cims_sectors'],
            control_data['jcims_to_cims_region'],
            control_data['energy_df'],
        )),
    ]

    for label, step_fn in steps:
        try:
            all_multipliers.append(step_fn())
        except Exception as e:
            failed.append((label, str(e)))

    base_mult_df = pd.concat(all_multipliers, ignore_index=True)

    try:
        derivative_mult = apply_derivative_multipliers(
            base_mult_df,
            control_data['cims_regions'],
            control_data['cims_sectors'],
        )
        if len(derivative_mult) > 0:
            all_multipliers.append(derivative_mult)
    except Exception as e:
        failed.append(("Derivative multipliers", str(e)))

    output_df = pd.concat(all_multipliers, ignore_index=True)
    output_df = (
        output_df
        .rename(columns={
            'region':     'Region',
            'energy':     'Energy',
            'sector':     'Sector',
            'source':     'Source',
            'year':       'Year',
            'multiplier': 'Multiplier',
        })
        .assign(Unit='ratio')
        [['Region', 'Energy', 'Sector', 'Unit', 'Source', 'Year', 'Multiplier']]
        .sort_values(['Energy', 'Region', 'Sector', 'Year'])
        .reset_index(drop=True)
    )

    # Drop sector/region combinations that sector has no fixed structural data
    # for (see sector_regions in load_control_data / sector_region_map.csv).
    # Sectors not listed there are left unrestricted (all cims_regions apply).
    sector_regions = control_data['sector_regions']
    if sector_regions:
        drop_mask = pd.Series(False, index=output_df.index)
        for sector, allowed in sector_regions.items():
            drop_mask |= (output_df['Sector'] == sector) & (~output_df['Region'].isin(allowed))
        dropped = int(drop_mask.sum())
        if dropped:
            output_df = output_df[~drop_mask].reset_index(drop=True)
            print(f"   Dropped {dropped:,} rows for sector/region combinations "
                  f"with no fixed data (see sector_region_map.csv)")

    print(f"\n✅ Multipliers complete")
    print(f"   Total rows:  {len(output_df):,}")
    print(f"   Energies:    {output_df['Energy'].nunique()}")
    print(f"   Regions:     {output_df['Region'].nunique()}")
    print(f"   Sectors:     {output_df['Sector'].nunique()}")
    print(f"   Years:       {output_df['Year'].min()} – {output_df['Year'].max()}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / 'energy_price_multipliers.csv'
    output_df.to_csv(output_path, index=False)
    print(f"   Saved to:    {output_path}")

    if failed:
        print(f"\n⚠️  {len(failed)} step(s) failed:")
        for label, error in failed:
            print(f"   • {label}: {error}")

    return output_df


if __name__ == "__main__":
    output_df = main()
