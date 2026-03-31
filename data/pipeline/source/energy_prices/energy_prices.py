"""
Energy Price Processing System
Generates production costs for various energy types from 2000-2100 in 2025 C$/GJ.

Combined processing script with all energy processors integrated.
Hybrid approach: Uses Polars for DataFrames, pandas for Excel and Series operations.
"""

import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Robust path setup using __file__
# File lives at:  C:\CIMS Data Processing\intermediate\energy_prices\
# parent.parent.parent resolves to: C:\CIMS Data Processing\
# src\ lives directly under there, so imports use the src. prefix.
import sys
_current_file = Path(__file__)
_project_root = _current_file.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from utils.controls_conversions import (
    load_macro_indicators,
    convert_currency,
    load_energy_conversions,
    load_control_config,
    load_energy_mapping,
)
from utils.extensions.dataframe_data_extensions import (
    backfill_constant,
    extend_constant,
    interpolate_5year_to_annual,
)


# Configuration
BASE_PATH = Path('C:/cims/data')
MACRO_FILE = BASE_PATH / 'raw_data/cer/macro-indicators-2026.csv'
BENCHMARK_FILE = BASE_PATH / 'raw_data/energy_prices/cer/benchmark-prices-2023.csv'
END_USE_PRICES_FILE = BASE_PATH / 'raw_data/energy_prices/cer/end-use-prices-2026.csv'
END_USE_DEMAND_FILE = BASE_PATH / 'raw_data/energy_prices/cer/end-use-demand-2023.csv'
CIMS_PRICES_FILE = BASE_PATH / 'raw_data/energy_prices/CIMS Prices and Calcs.xlsx'
RETAIL_FUEL_FILE = BASE_PATH / 'raw_data/energy_prices/alternative_fuels_data_center/10326_retail_fuel_prices_1-23-26.xlsx'
RENEWABLE_DIESEL_FILE = BASE_PATH / 'raw_data/energy_prices/alternative_fuels_data_center/10969_renewable_diesel_prices_2-3-26.xlsx'
OUTPUT_DIR = Path('C:/cims/data/processed_data/energy_prices')


# Regions for regional energies
REGIONS = ['BC', 'AB', 'SK', 'MB', 'ON', 'QC', 'NB', 'NS', 'PE', 'NL', 'YT', 'NT', 'NU']


# ============================================================================
# ENERGY PROCESSORS
# ============================================================================

def process_cer_benchmark_energy(
    benchmark_df: pl.DataFrame,
    macro_df: pl.DataFrame,
    scenario: str,
    variable_name: str,
    unit_from: str,
    gj_conversion: float,
    from_year: int,
    from_currency: str,
) -> pd.Series:
    """
    Process CER benchmark energy prices (Natural Gas, WTI-based energies).

    Parameters
    ----------
    benchmark_df : pl.DataFrame
        CER benchmark prices data.
    macro_df : pl.DataFrame
        Macro indicators for currency conversion.
    scenario : str
        CER scenario name.
    variable_name : str
        Variable name to filter in benchmark data.
    unit_from : str
        Source unit (e.g., 'MMBtu', 'bbl').
    gj_conversion : float
        Conversion factor to GJ.
    from_year : int
        Source currency year.
    from_currency : str
        Source currency country code.

    Returns
    -------
    pd.Series
        Price series in 2025 C$/GJ indexed by year (2000–2100).
    """
    data = benchmark_df.filter(
        (pl.col('Scenario') == scenario) &
        (pl.col('Variable') == variable_name)
    )
    # Extract year and value as plain lists — avoids pyarrow dependency of to_pandas()
    years  = data.get_column('Year').cast(pl.Int64).to_list()
    values = data.get_column('Value').cast(pl.Float64).to_list()
    prices = pd.Series(values, index=years, dtype=float)

    prices_cad_gj = convert_currency(
        prices,
        from_year=from_year,
        to_year=2025,
        from_currency=from_currency,
        to_currency='CAD',
        macro_df=macro_df,
        constant_dollars=True,
    )
    prices_cad_gj = prices_cad_gj * gj_conversion
    prices_full = backfill_constant(prices_cad_gj, 2000)
    prices_full = extend_constant(prices_full, 2100)

    return prices_full


def process_end_use_weighted_avg(
    prices_df: pl.DataFrame,
    demand_df: pl.DataFrame,
    scenario: str,
    sector: str,
    energy_price: str,
    energy_demand: Optional[str] = None,
) -> pd.Series:
    """
    Process end-use energy with weighted average across provinces.

    Parameters
    ----------
    prices_df : pl.DataFrame
        End-use prices data.
    demand_df : pl.DataFrame
        End-use demand data.
    scenario : str
        CER scenario name.
    sector : str
        Sector name (e.g., 'Transportation').
    energy_price : str
        Energy name in the prices file.
    energy_demand : str, optional
        Energy name in the demand file. Defaults to energy_price.

    Returns
    -------
    pd.Series
        Price series in 2025 C$/GJ indexed by year (2000–2100).
    """
    prices = weighted_average_by_province(
        prices_df, demand_df, scenario, sector, energy_price, energy_demand
    )
    prices_full = backfill_constant(prices, 2000)
    prices_full = extend_constant(prices_full, 2100)

    return prices_full


def process_jcims_energy(
    jcims_df: pd.DataFrame,
    macro_df: pl.DataFrame,
    jcims_energy_name: str,
) -> pd.Series:
    """
    Process energy from JCIMS production cost tab.

    Parameters
    ----------
    jcims_df : pd.DataFrame
        JCIMS production cost data (loaded from Excel).
    macro_df : pl.DataFrame
        Macro indicators for currency conversion.
    jcims_energy_name : str
        Fuel name to look up in the JCIMS data.

    Returns
    -------
    pd.Series
        Price series in 2025 C$/GJ indexed by year (2000–2100).
        Returns an empty Series if the energy name is not found.
    """
    energy_data = jcims_df[
        (jcims_df['Province'] == 'Generic') &
        (jcims_df['Fuel'] == jcims_energy_name)
    ].copy()

    if len(energy_data) == 0:
        print(f"  ⚠️  {jcims_energy_name} not found in JCIMS data")
        return pd.Series(dtype=float)

    year_cols = [col for col in jcims_df.columns if isinstance(col, (int, np.integer))]

    prices_dict = {}
    for year in year_cols:
        value = energy_data[year].values[0]
        if pd.notna(value) and value > 0:
            prices_dict[year] = value

    prices = pd.Series(prices_dict)

    prices_2025 = convert_currency(
        prices,
        from_year=2005,
        to_year=2025,
        from_currency='CAD',
        to_currency='CAD',
        macro_df=macro_df,
        constant_dollars=True,
    )
    prices_annual = interpolate_5year_to_annual(prices_2025)
    prices_full = backfill_constant(prices_annual, 2000)
    prices_full = extend_constant(prices_full, 2100)

    return prices_full


def process_saf_prices(jet_energy_prices: pd.Series) -> pd.Series:
    """
    Process SAF (Sustainable Aviation Fuel) prices.

    Uses 160 C$/GJ until 2030, then linearly declines to 2× jet fuel by 2050,
    holding at 2× jet fuel thereafter.

    Parameters
    ----------
    jet_energy_prices : pd.Series
        Jet fuel price series in 2025 C$/GJ indexed by year.

    Returns
    -------
    pd.Series
        SAF price series in 2025 C$/GJ indexed by year (2000–2100).
    """
    years = range(2000, 2101)
    saf_prices = pd.Series(index=years, dtype=float)

    saf_prices.loc[2000:2030] = 160.0

    jet_2050 = jet_energy_prices.loc[2050]
    target_2050 = 2 * jet_2050

    for year in range(2031, 2051):
        fraction = (year - 2030) / (2050 - 2030)
        saf_prices.loc[year] = 160.0 + (target_2050 - 160.0) * fraction

    saf_prices.loc[2050:2100] = 2 * jet_energy_prices.loc[2050:2100]

    return saf_prices


def process_electricity_regional(
    elec_df: pd.DataFrame,
    region: str,
) -> pd.Series:
    """
    Process electricity prices for a specific region.

    Parameters
    ----------
    elec_df : pd.DataFrame
        CIMS electricity price data (loaded from Excel).
    region : str
        2-letter region code.

    Returns
    -------
    pd.Series
        Price series in 2025 C$/GJ indexed by year (2000–2100).
        Returns an empty Series if the region is not found.
    """
    region_data = elec_df[elec_df['Region'] == region].copy()

    if len(region_data) == 0:
        return pd.Series(dtype=float)

    year_cols = [col for col in region_data.columns if isinstance(col, (int, np.integer))]

    prices_dict = {}
    for year in year_cols:
        value = region_data[year].values[0]
        if pd.notna(value):
            prices_dict[year] = value

    prices = pd.Series(prices_dict)
    prices_full = backfill_constant(prices, 2000)
    prices_full = extend_constant(prices_full, 2100)

    return prices_full


def process_hydrogen_regional(
    h2_df: pd.DataFrame,
    macro_df: pl.DataFrame,
    region: str,
) -> pd.Series:
    """
    Process hydrogen prices for a specific region.

    Parameters
    ----------
    h2_df : pd.DataFrame
        CIMS hydrogen price data (loaded from Excel).
    macro_df : pl.DataFrame
        Macro indicators for currency conversion.
    region : str
        2-letter region code.

    Returns
    -------
    pd.Series
        Price series in 2025 C$/GJ indexed by year (2000–2100).
        Returns an empty Series if the region is not found.
    """
    region_data = h2_df[h2_df['Region'] == region].copy()

    if len(region_data) == 0:
        return pd.Series(dtype=float)

    year_cols = [col for col in region_data.columns if isinstance(col, (int, np.integer))]

    prices_dict = {}
    for year in year_cols:
        value = region_data[year].values[0]
        if pd.notna(value):
            prices_dict[year] = value

    prices = pd.Series(prices_dict)

    prices_2025 = convert_currency(
        prices,
        from_year=2005,
        to_year=2025,
        from_currency='CAD',
        to_currency='CAD',
        macro_df=macro_df,
        constant_dollars=True,
    )
    prices_annual = interpolate_5year_to_annual(prices_2025)
    prices_full = backfill_constant(prices_annual, 2000)
    prices_full = extend_constant(prices_full, 2100)
    return prices_full


def process_afdc_energy(
    afdc_df: pd.DataFrame,
    macro_df: pl.DataFrame,
    energy_column: str,
    conversions: Dict,
    energy_type: str,
    backfill_to_2000: bool = False,
) -> pd.Series:
    """
    Process energy from Alternative Fuels Data Center (AFDC) files.

    Prices are nominal $/gallon and are converted to 2025 C$/GJ.

    Parameters
    ----------
    afdc_df : pd.DataFrame
        AFDC retail fuel price data (loaded from Excel).
    macro_df : pl.DataFrame
        Macro indicators for currency conversion.
    energy_column : str
        Column name for the energy price (e.g., 'E85', 'B20').
    conversions : dict
        Energy conversion factors dictionary.
    energy_type : str
        Key for GJ/m³ conversion factor in conversions dict.
    backfill_to_2000 : bool
        Whether to backfill to year 2000 (default False).

    Returns
    -------
    pd.Series
        Price series in 2025 C$/GJ indexed by year.
        Returns an empty Series if a date column cannot be found.
    """
    afdc_df = afdc_df.copy()

    date_col = None
    for col in afdc_df.columns:
        if 'date' in col.lower():
            date_col = col
            break

    if date_col is None:
        print(f"  ⚠️  Could not find date column for {energy_type}")
        return pd.Series(dtype=float)

    afdc_df = afdc_df[afdc_df[date_col].notna()]

    if afdc_df[date_col].dtype == 'object':
        afdc_df = afdc_df[afdc_df[date_col].astype(str).str.contains(r'\d+', regex=True, na=False)]

    if len(afdc_df) == 0:
        return pd.Series(dtype=float)

    afdc_df[date_col] = pd.to_datetime(afdc_df[date_col], format='mixed', dayfirst=False, errors='coerce')
    afdc_df = afdc_df[afdc_df[date_col].notna()]
    afdc_df['Year'] = afdc_df[date_col].dt.year
    afdc_df = afdc_df[afdc_df[energy_column].notna()]

    annual_avg = afdc_df.groupby('Year')[energy_column].mean()

    if len(annual_avg) == 0:
        return pd.Series(dtype=float)

    prices_2025_cad_gal = convert_currency(
        annual_avg,
        from_year=annual_avg.index[0],
        to_year=2025,
        from_currency='USD',
        to_currency='CAD',
        macro_df=macro_df,
        constant_dollars=False,
    )

    gal_to_m3 = 0.00378541
    gj_per_m3 = conversions[f'{energy_type}_gj_per_m3']
    gj_per_gal = gal_to_m3 * gj_per_m3
    prices_gj = prices_2025_cad_gal / gj_per_gal

    start_year = 2000 if backfill_to_2000 else int(prices_gj.index.min())
    prices_full = backfill_constant(prices_gj, start_year)
    prices_full = extend_constant(prices_full, 2100)

    return prices_full


def weighted_average_by_province(
    prices_df: pl.DataFrame,
    demand_df: pl.DataFrame,
    scenario: str,
    sector: str,
    energy_price: str,
    energy_demand: Optional[str] = None,
) -> pd.Series:
    """
    Calculate demand-weighted average Canadian price across provinces.

    Parameters
    ----------
    prices_df : pl.DataFrame
        End-use prices data.
    demand_df : pl.DataFrame
        End-use demand data.
    scenario : str
        CER scenario name.
    sector : str
        Sector name (e.g., 'Transportation', 'Industrial').
    energy_price : str
        Energy name in the prices file (e.g., 'Gasoline', 'Oil').
    energy_demand : str, optional
        Energy name in the demand file. Defaults to energy_price.

    Returns
    -------
    pd.Series
        Weighted average prices by year (pandas Series for downstream compatibility).
    """
    if energy_demand is None:
        energy_demand = energy_price

    price_data = prices_df.filter(
        (pl.col('Scenario') == scenario) &
        (pl.col('Sector') == sector) &
        (pl.col('Fuel') == energy_price)
    )

    demand_data = demand_df.filter(
        (pl.col('Scenario') == scenario) &
        (pl.col('Sector') == sector) &
        (pl.col('Variable') == energy_demand)
    ).rename({'Region': 'Area', 'Value': 'Sum of Demand'})

    merged = price_data.join(
        demand_data.select(['Year', 'Area', 'Sum of Demand']),
        on=['Year', 'Area'],
        how='inner',
    )

    if len(merged) == 0:
        return pd.Series(dtype=float)

    merged = merged.with_columns([
        (pl.col('Sum of Price') * pl.col('Sum of Demand')).alias('weighted_price')
    ])

    weighted_avg = merged.group_by('Year').agg([
        pl.col('weighted_price').sum(),
        pl.col('Sum of Demand').sum(),
    ])

    weighted_avg = weighted_avg.with_columns([
        (pl.col('weighted_price') / pl.col('Sum of Demand')).alias('price')
    ]).select(['Year', 'price']).sort('Year')

    # Extract as plain lists — avoids pyarrow dependency of to_pandas()
    years  = weighted_avg.get_column('Year').cast(pl.Int64).to_list()
    prices = weighted_avg.get_column('price').cast(pl.Float64).to_list()
    return pd.Series(prices, index=years, dtype=float)


# ============================================================================
# MAIN PROCESSING FUNCTIONS
# ============================================================================

def load_all_data() -> Dict:
    """
    Load all required data files.

    Returns
    -------
    dict
        Dictionary containing all loaded DataFrames and configuration values.
    """
    print("Loading configuration and conversion factors...")
    config = load_control_config()
    conversions = load_energy_conversions()
    energy_map = load_energy_mapping()

    print("Loading macro indicators...")
    scenario = config.get('cer_ef_reference_scenario', 'Current Measures')
    macro_df = load_macro_indicators(MACRO_FILE, scenario)

    print("Loading benchmark prices...")
    benchmark_df = pl.read_csv(BENCHMARK_FILE)

    print("Loading end-use prices and demand...")
    end_use_prices = pl.read_csv(END_USE_PRICES_FILE)
    end_use_demand = pl.read_csv(END_USE_DEMAND_FILE)

    print("Loading CIMS prices...")
    jcims_prod = pd.read_excel(CIMS_PRICES_FILE, sheet_name='JCIMS Production Cost')
    elec_prices = pd.read_excel(CIMS_PRICES_FILE, sheet_name='CIMS Electricity')
    h2_prices = pd.read_excel(CIMS_PRICES_FILE, sheet_name='CIMS Hydrogen')

    print("Loading alternative fuel prices...")
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')
        retail_fuel = pd.read_excel(
            RETAIL_FUEL_FILE,
            sheet_name='Average Retail Fuel Prices',
            skiprows=2,
        )
        renewable_diesel = pd.read_excel(RENEWABLE_DIESEL_FILE, skiprows=2)

    return {
        'config': config,
        'conversions': conversions,
        'energy_map': energy_map,
        'macro_df': macro_df,
        'benchmark_df': benchmark_df,
        'end_use_prices': end_use_prices,
        'end_use_demand': end_use_demand,
        'jcims_prod': jcims_prod,
        'elec_prices': elec_prices,
        'h2_prices': h2_prices,
        'retail_fuel': retail_fuel,
        'renewable_diesel': renewable_diesel,
        'scenario': scenario,
    }


def process_generic_energies(data: Dict) -> List[Tuple[str, str, pd.Series]]:
    """
    Process energies with generic region production costs.

    Parameters
    ----------
    data : dict
        Loaded data dictionary from load_all_data().

    Returns
    -------
    list of (str, str, pd.Series)
        List of (energy_name, region, price_series) tuples.
    """
    results = []
    config = data['config']
    conversions = data['conversions']
    scenario = data['scenario']
    macro_df = data['macro_df']
    benchmark_df = data['benchmark_df']
    end_use_prices = data['end_use_prices']
    end_use_demand = data['end_use_demand']
    jcims_prod = data['jcims_prod']

    from_year = int(config['currency_year_cer_benchmark'])
    from_currency = config['currency_country_cer_benchmark']

    print("\nProcessing Natural Gas...")
    ng_prices = process_cer_benchmark_energy(
        benchmark_df, macro_df, scenario,
        'Nova Inventory Transfer (NIT) - 2022 US$/MMBtu',
        'MMBtu', conversions['mmbtu_to_gj'],
        from_year, from_currency,
    )
    results.append(('Natural Gas', 'generic', ng_prices))

    print("Processing Natural Gas Feedstock...")
    results.append(('Natural Gas Feedstock', 'generic', ng_prices))

    print("Processing Petrochemical Feedstock...")
    petro_prices = process_cer_benchmark_energy(
        benchmark_df, macro_df, scenario,
        'West Texas Intermediate (WTI) - 2022 US$/bbl',
        'bbl',
        1/(conversions['bbl_to_m3'] * conversions['petrochemical_feedstock_gj_per_m3']),
        from_year, from_currency,
    )
    results.append(('Petrochemical Feedstock', 'generic', petro_prices))

    print("Processing Naphtha Specialties...")
    naphtha_prices = process_cer_benchmark_energy(
        benchmark_df, macro_df, scenario,
        'West Texas Intermediate (WTI) - 2022 US$/bbl',
        'bbl',
        1/(conversions['bbl_to_m3'] * conversions['naphtha_specialties_gj_per_m3']),
        from_year, from_currency,
    )
    results.append(('Naphtha Specialties', 'generic', naphtha_prices))

    print("Processing Asphalt...")
    asphalt_prices = process_cer_benchmark_energy(
        benchmark_df, macro_df, scenario,
        'West Texas Intermediate (WTI) - 2022 US$/bbl',
        'bbl',
        1/(conversions['bbl_to_m3'] * conversions['asphalt_gj_per_m3']),
        from_year, from_currency,
    )
    results.append(('Asphalt', 'generic', asphalt_prices))

    print("Processing Lubricants...")
    lube_prices = process_cer_benchmark_energy(
        benchmark_df, macro_df, scenario,
        'West Texas Intermediate (WTI) - 2022 US$/bbl',
        'bbl',
        1/(conversions['bbl_to_m3'] * conversions['lubricants_gj_per_m3']),
        from_year, from_currency,
    )
    results.append(('Lubricants', 'generic', lube_prices))

    print("Processing Other Non-Energy Products...")
    other_prices = process_cer_benchmark_energy(
        benchmark_df, macro_df, scenario,
        'West Texas Intermediate (WTI) - 2022 US$/bbl',
        'bbl',
        1/(conversions['bbl_to_m3'] * conversions['other_non_energy_products_gj_per_m3']),
        from_year, from_currency,
    )
    results.append(('Other Non-Energy Products', 'generic', other_prices))

    print("Processing Heavy Fuel Oil...")
    hfo_prices = process_end_use_weighted_avg(
        end_use_prices, end_use_demand, scenario,
        'Industrial', 'Oil', 'RPP',
    )
    results.append(('Heavy Fuel Oil', 'generic', hfo_prices))

    print("Processing Light Fuel Oil...")
    lfo_prices = process_end_use_weighted_avg(
        end_use_prices, end_use_demand, scenario,
        'Commercial', 'Oil', 'RPP',
    )
    results.append(('Light Fuel Oil', 'generic', lfo_prices))

    print("Processing Gasoline...")
    gasoline_prices = process_end_use_weighted_avg(
        end_use_prices, end_use_demand, scenario,
        'Transportation', 'Gasoline', 'Motor Gasoline',
    )
    results.append(('Gasoline', 'generic', gasoline_prices))

    print("Processing Diesel...")
    diesel_prices = process_end_use_weighted_avg(
        end_use_prices, end_use_demand, scenario,
        'Transportation', 'Diesel', 'Diesel',
    )
    results.append(('Diesel', 'generic', diesel_prices))

    print("Processing Kerosene...")
    results.append(('Kerosene', 'generic', diesel_prices))

    print("Processing Renewable Natural Gas...")
    rng_prices = ng_prices * 1.5
    results.append(('Renewable Natural Gas', 'generic', rng_prices))

    jcims_energies = {
        'Black Liquor': 'Black Liquor',
        'Thermal Coal': 'Coal',
        'Metallurgical Coal': 'Coal',
        'Coke': 'Coke',
        'Jet Fuel': 'Jet Fuel',
        'LPG': 'LPG',
        'Propane': 'Propane',
        'Refinery Fuel Gas': 'Refinery Fuel Gas',
        'Solid Biomass': 'Solid Biomass',
        'Uranium': 'Uranium',
        'Waste': 'Waste Fuel',
        'Petroleum Coke': 'Petroleum Coke',
    }

    for cims_energy, jcims_energy in jcims_energies.items():
        print(f"Processing {cims_energy}...")
        prices = process_jcims_energy(jcims_prod, macro_df, jcims_energy)
        results.append((cims_energy, 'generic', prices))
    
    # --- Add Ethane and Butane using Propane prices ---
    propane_prices = process_jcims_energy(jcims_prod, macro_df, 'Propane')

    for fuel in ['Ethane', 'Butane']:
        print(f"Processing {fuel} (using Propane prices)...")
        results.append((fuel, 'generic', propane_prices))

    print("Processing SAF...")
    jet_prices = [r[2] for r in results if r[0] == 'Jet Fuel'][0]
    saf_prices = process_saf_prices(jet_prices)
    results.append(('SAF', 'generic', saf_prices))

    return results


def process_regional_energies(data: Dict) -> List[Tuple[str, str, pd.Series]]:
    """
    Process energies with regional production costs.

    Parameters
    ----------
    data : dict
        Loaded data dictionary from load_all_data().

    Returns
    -------
    list of (str, str, pd.Series)
        List of (energy_name, region, price_series) tuples.
    """
    results = []
    elec_prices = data['elec_prices']
    h2_prices = data['h2_prices']
    macro_df = data['macro_df']
    retail_fuel = data['retail_fuel']
    renewable_diesel_df = data['renewable_diesel']
    conversions = data['conversions']

    print("\nProcessing Electricity (regional)...")
    for region in REGIONS:
        prices = process_electricity_regional(elec_prices, region)
        if len(prices) > 0:
            results.append(('Electricity', region, prices))

    print("Processing Hydrogen (regional)...")
    for region in REGIONS:
        prices = process_hydrogen_regional(h2_prices, macro_df, region)
        if len(prices) > 0:
            results.append(('Hydrogen', region, prices))

    print("Processing Ethanol (regional)...")
    ethanol_prices = process_afdc_energy(
        retail_fuel, macro_df, 'E85', conversions, 'ethanol', backfill_to_2000=True
    )
    for region in REGIONS:
        if len(ethanol_prices) > 0:
            results.append(('Ethanol', region, ethanol_prices.copy()))

    print("Processing Biodiesel (regional)...")
    biodiesel_prices = process_afdc_energy(
        retail_fuel, macro_df, 'B20', conversions, 'biodiesel', backfill_to_2000=True
    )
    for region in REGIONS:
        if len(biodiesel_prices) > 0:
            results.append(('Biodiesel', region, biodiesel_prices.copy()))

    print("Processing Renewable Diesel (regional)...")
    ren_diesel_prices = process_afdc_energy(
        renewable_diesel_df, macro_df, 'Renewable Diesel', conversions,
        'renewable_diesel', backfill_to_2000=True,
    )
    for region in REGIONS:
        if len(ren_diesel_prices) > 0:
            results.append(('Renewable Diesel', region, ren_diesel_prices.copy()))

    print("Processing Renewable Gasoline (regional)...")
    for region in REGIONS:
        if len(ren_diesel_prices) > 0:
            ren_gas_prices = ren_diesel_prices * 1.1
            results.append(('Renewable Gasoline', region, ren_gas_prices.copy()))

    return results


def create_output_dataframe(
    generic_results: List[Tuple[str, str, pd.Series]],
    regional_results: List[Tuple[str, str, pd.Series]],
) -> pd.DataFrame:
    """
    Create the final output DataFrame from generic and regional results.

    Parameters
    ----------
    generic_results : list of (str, str, pd.Series)
        Generic energy results from process_generic_energies().
    regional_results : list of (str, str, pd.Series)
        Regional energy results from process_regional_energies().

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns: region, energy, year, unit, price.
    """
    rows = []

    for energy, region, prices in generic_results:
        for year, price in prices.items():
            if pd.notna(price):
                rows.append({
                    'region': region,
                    'energy': energy,
                    'year': int(year),
                    'unit': '2025 C$/GJ',
                    'price': price,
                })

    for energy, region, prices in regional_results:
        if len(prices) > 0:
            for year, price in prices.items():
                if pd.notna(price):
                    rows.append({
                        'region': region,
                        'energy': energy,
                        'year': int(year),
                        'unit': '2025 C$/GJ',
                        'price': price,
                    })

    df = pd.DataFrame(rows)
    df = df.sort_values(['energy', 'region', 'year']).reset_index(drop=True)

    return df


def main() -> pd.DataFrame:
    """
    Main execution function.

    Returns
    -------
    pd.DataFrame
        Final output DataFrame of production costs.
    """
    print("=" * 80)
    print("ENERGY PRICE PROCESSING")
    print("=" * 80)

    data = load_all_data()

    generic_results = process_generic_energies(data)
    regional_results = process_regional_energies(data)

    print("\n" + "=" * 80)
    print("Creating output dataframe...")
    output_df = create_output_dataframe(generic_results, regional_results)

    print(f"\n✅ Production costs complete")
    print(f"   Total rows:          {len(output_df):,}")
    print(f"   Energies processed:  {output_df['energy'].nunique()}")
    print(f"   Years covered:       {output_df['year'].min()} – {output_df['year'].max()}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / 'energy_prices.csv'
    output_df.to_csv(output_path, index=False)
    print(f"   Saved to:            {output_path}")

    return output_df


if __name__ == "__main__":
    output_df = main()
