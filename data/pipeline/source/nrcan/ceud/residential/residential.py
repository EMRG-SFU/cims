"""
Residential Pipeline

This module provides functions to extract and process residential building data
from NRCan CEUD (Comprehensive Energy Use Database) for all Canadian provinces
and territories.

Key differences from BC version:
- Works with all provinces/territories using their 2-letter codes
- BC has Marine climate heating technologies (heating only, not water heating)
- All provinces use Cold climate for water heating technologies
- MB (Manitoba) has a different last vintage bin: "2022_after" instead of "2021_after"
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl

# Robust path setup using __file__
import sys
_current_file = Path(__file__)
_project_root = _current_file.parent.parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from utils.extractors.nrcan_ceud import get_row_series
from utils.dict_ops import sum_dicts, weighted_average_dicts, divide_dicts, multiply_dicts
from utils.extensions.dict_data_extensions import (
    extend_data_constant,
    extend_data_linear,
    extend_trend_decline,
)

# Configuration
BASE_PATH = Path('C:/cims/data/raw_data/nrcan/ceud/residential')
YEARS = list(range(2000, 2101))
ASSUMPTIONS_CSV = Path('C:/cims/data/raw_data/assumptions/residential_assumptions.csv')
OUTPUT_DIR = Path('C:/cims/data/processed_data/nrcan/ceud')

# Province/Territory codes
PROVINCES = {
    'AB': 'Alberta',
    'BC': 'British Columbia',
    'MB': 'Manitoba',
    'NB': 'New Brunswick',
    'NL': 'Newfoundland and Labrador',
    'NS': 'Nova Scotia',
    'ON': 'Ontario',
    'PE': 'Prince Edward Island',
    'QC': 'Quebec',
    'SK': 'Saskatchewan',
    'TR': 'Territories',
}


# ==============================================================================
# PARAMETER LOADING AND EXTENSION APPLICATION
# ==============================================================================

def load_projection_params(assumptions_csv: Path = ASSUMPTIONS_CSV) -> Dict:
    """
    Load projection parameters from the residential assumptions CSV file.

    The CSV uses a sparse/wide layout where parameters are stored at specific
    row positions. This function parses those known positions to reconstruct
    the projection parameter dictionary used by apply_extensions().

    Parameters
    ----------
    assumptions_csv : Path
        Path to the residential_assumptions.csv file.

    Returns
    -------
    dict
        Projection parameters structured identically to the former JSON format,
        or empty dict if the file is not found or cannot be parsed.
    """
    # Province name → code mapping (matches column labels in the CSV)
    PROV_NAME_TO_CODE = {
        'British Columbia': 'BC',
        'Alberta': 'AB',
        'Saskatchewan': 'SK',
        'Manitoba': 'MB',
        'Ontario': 'ON',
        'Quebec': 'QC',
        'New Brunswick': 'NB',
        'Nova Scotia': 'NS',
        'Prince Edward Island': 'PE',
        'Newfoundland and Labrador': 'NL',
        'Atlantic': None,       # aggregate — skipped
        'Territories': 'TR',
        'Yukon': None,
        'Northwest Territories': None,
        'Nunavut': None,
    }

    try:
        df = pd.read_csv(assumptions_csv, header=None, dtype=str)
    except FileNotFoundError:
        print(f"⚠️  Assumptions CSV {assumptions_csv} not found. Extensions will not be applied.")
        return {}
    except Exception as e:
        print(f"❌ Error reading assumptions CSV: {e}")
        return {}

    def cell(row, col):
        """Return stripped string value at (row, col), or None if missing/empty."""
        try:
            v = df.iloc[row, col]
            if pd.isna(v) or str(v).strip() == '':
                return None
            return str(v).strip()
        except (IndexError, KeyError):
            return None

    def pct(row, col):
        """Parse a percentage string like '1.4%' → 0.014, or a plain float."""
        v = cell(row, col)
        if v is None:
            return None
        v = v.replace('%', '').strip()
        try:
            return float(v) / 100.0
        except ValueError:
            return None

    params = {}

    # ------------------------------------------------------------------
    # 1. HOUSING STOCK — Linear growth (rows 9–22, 0-indexed rows 9-22)
    #    CSV layout (0-indexed):
    #      col 5  = province name
    #      col 6  = historical CAGR (2010-2020)  [used as reference only]
    #      col 9  = period 1 CAGR (2021-2050)
    #      col 10 = period 2 CAGR (2051-2100)
    #    Row indices (0-based): BC=9, AB=10, SK=11, MB=12, ON=13, QC=14,
    #                            NB=15, NS=16, PE=17, NL=18, (Yukon=19, NWT=20, Nunavut=21)
    #                            Atlantic=22, Territories=23
    # ------------------------------------------------------------------
    params['housing_stock'] = {'method': 'linear'}

    for row_idx in range(9, 24):
        prov_name = cell(row_idx, 5)
        if prov_name is None:
            continue
        code = PROV_NAME_TO_CODE.get(prov_name)
        if code is None:
            continue
        rate1 = pct(row_idx, 9)
        rate2 = pct(row_idx, 10)
        if rate1 is None or rate2 is None:
            continue
        params['housing_stock'][code] = {
            'periods': [[2023, 2051, rate1], [2051, 2101, rate2]]
        }

    # ------------------------------------------------------------------
    # 2. BUILDING SHARES — Trend decline (rows 29–32, 0-indexed)
    #    The CSV stores global acceleration percentages (col 9 = period 1,
    #    col 10 = period 2) on row 31 ("Trend 2000-2020").
    #    These values are the same for all provinces in the reference scenario.
    # ------------------------------------------------------------------
    params['building_shares'] = {
        'method': 'trend_decline',
        'trend_start': 2000,
        'trend_end': 2022,
        'trend_period': [2023, 2031],
    }

    bs_dec1 = pct(31, 9)   # e.g. -5% → -0.05
    bs_dec2 = pct(31, 10)  # e.g. -10% → -0.10
    if bs_dec1 is None:
        bs_dec1 = -0.05
    if bs_dec2 is None:
        bs_dec2 = -0.10

    for code in ['AB', 'BC', 'MB', 'NB', 'NL', 'NS', 'ON', 'PE', 'QC', 'SK', 'TR']:
        params['building_shares'][code] = {
            'decrease_periods': [[2031, 2051, abs(bs_dec1)], [2051, 2101, abs(bs_dec2)]]
        }

    # ------------------------------------------------------------------
    # 3. FLOORSPACE PER BUILDING — Trend decline (rows 36–40, 0-indexed)
    #    Same layout as building shares; decline values on row 39.
    # ------------------------------------------------------------------
    params['floorspace_per_building'] = {
        'method': 'trend_decline',
        'trend_start': 2000,
        'trend_end': 2022,
        'trend_period': [2023, 2031],
    }

    fs_dec1 = pct(38, 9)
    fs_dec2 = pct(38, 10)
    if fs_dec1 is None:
        fs_dec1 = -0.05
    if fs_dec2 is None:
        fs_dec2 = -0.10

    for code in ['AB', 'BC', 'MB', 'NB', 'NL', 'NS', 'ON', 'PE', 'QC', 'SK', 'TR']:
        params['floorspace_per_building'][code] = {
            'decrease_periods': [[2031, 2051, abs(fs_dec1)], [2051, 2101, abs(fs_dec2)]]
        }

    print(f"✅ Loaded projection parameters from {assumptions_csv}")
    return params


def apply_extensions(data: Dict, province_code: str, projection_params: Dict) -> Dict:
    """
    Apply projection extensions to extracted data based on parameters.

    Parameters
    ----------
    data : dict
        Extracted data for a province.
    province_code : str
        2-letter province code.
    projection_params : dict
        Projection parameters from JSON file.

    Returns
    -------
    dict
        Data with extensions applied.
    """
    if not projection_params:
        print(f"  ⚠️  No projection parameters — skipping extensions for {province_code}")
        return data

    province_code_upper = province_code.upper()
    print(f"  📈 Applying extensions for {province_code_upper}...")

    # 1. HOUSING STOCK — Linear growth
    if 'housing_stock' in projection_params and province_code_upper in projection_params['housing_stock']:
        params = projection_params['housing_stock'][province_code_upper]
        if 'periods' in params:
            data['housing_thousand'] = extend_data_linear(
                data['housing_thousand'],
                base_year=2022,
                periods=params['periods'],
            )
            print(f"    ✓ Housing stock extended")

    # 2. BUILDING SHARES — Trend decline
    if 'building_shares' in projection_params and province_code_upper in projection_params['building_shares']:
        global_params = projection_params['building_shares']
        prov_params = global_params.get(province_code_upper, {})

        for building_type in ['Single Detached', 'Single Attached', 'Mobile Homes']:
            if building_type in data['building_shares']:
                data['building_shares'][building_type] = extend_trend_decline(
                    data['building_shares'][building_type],
                    base_year=2022,
                    trend_start=global_params.get('trend_start', 2000),
                    trend_end=global_params.get('trend_end', 2022),
                    trend_period=tuple(global_params.get('trend_period', [2023, 2031])),
                    decrease_periods=prov_params.get('decrease_periods', [[2031, 2051, 0.05], [2051, 2101, 0.1]]),
                )

        # Apartments = 100% - sum(others)
        if 'Apartments' in data['building_shares']:
            for year in range(2023, 2101):
                other_sum = sum(
                    data['building_shares'][bt].get(year, 0)
                    for bt in ['Single Detached', 'Single Attached', 'Mobile Homes']
                )
                data['building_shares']['Apartments'][year] = 1 - other_sum

        print(f"    ✓ Building shares extended")

    # 3. FLOORSPACE PER BUILDING — Trend decline
    if 'floorspace_per_building' in projection_params and province_code_upper in projection_params['floorspace_per_building']:
        global_params = projection_params['floorspace_per_building']
        prov_params = global_params.get(province_code_upper, {})

        for building_type in ['Single Detached', 'Single Attached', 'Apartments', 'Mobile Homes']:
            if building_type in data['floorspace_per_building']:
                data['floorspace_per_building'][building_type] = extend_trend_decline(
                    data['floorspace_per_building'][building_type],
                    base_year=2022,
                    trend_start=global_params.get('trend_start', 2000),
                    trend_end=global_params.get('trend_end', 2022),
                    trend_period=tuple(global_params.get('trend_period', [2023, 2031])),
                    decrease_periods=prov_params.get('decrease_periods', [[2031, 2051, 0.05], [2051, 2101, 0.1]]),
                )

        print(f"    ✓ Floorspace per building extended")

    # 4. WATER HEATING — Constant (hold 2022 values)
    if 'wh_lowmed' in data:
        data['wh_lowmed'] = extend_data_constant(data['wh_lowmed'], base_year=2022)
    if 'wh_high' in data:
        data['wh_high'] = extend_data_constant(data['wh_high'], base_year=2022)
    print(f"    ✓ Water heating extended (constant)")

    # 5. COOLING — Constant (hold 2022 values)
    if 'cooling_share_data' in data:
        for cooling_type in ['Room', 'Central']:
            if cooling_type in data['cooling_share_data']:
                data['cooling_share_data'][cooling_type] = extend_data_constant(
                    data['cooling_share_data'][cooling_type],
                    base_year=2022,
                )
        print(f"    ✓ Cooling data extended (constant)")

    return data


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_tables(province_code: str) -> Dict:
    """
    Load all required tables from the Excel file for a given province.

    Parameters
    ----------
    province_code : str
        2-letter province/territory code (e.g., 'bc', 'on', 'ab').

    Returns
    -------
    dict
        Dictionary of table names to Polars DataFrames.

    Raises
    ------
    FileNotFoundError
        If the data file for the province does not exist.
    """
    province_code_lower = province_code.lower()
    file_path = BASE_PATH / f"res_{province_code_lower}_e.xls"

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    table_names = [
        "Table 4", "Table 10", "Table 11", "Table 15", "Table 18",
        "Table 19", "Table 20", "Table 22", "Table 23", "Table 24",
        "Table 25", "Table 31",
    ]

    return {name: pl.read_excel(str(file_path), sheet_name=name, has_header=False)
            for name in table_names}


# ==============================================================================
# HOUSING STOCK
# ==============================================================================

def extract_housing_stock(tables: Dict) -> Tuple[Dict, Dict]:
    """
    Extract total housing stock and housing stock by building type.

    Parameters
    ----------
    tables : dict
        Dictionary of loaded tables.

    Returns
    -------
    tuple of (dict, dict)
        (housing_thousand, housing_stock_by_type)
    """
    t15 = tables["Table 15"]

    housing_thousand_raw = get_row_series(t15, "Total Housing Stock (thousands)")
    housing_thousand = {year: val * 1000 if val is not None else None
                        for year, val in housing_thousand_raw.items()}

    housing_stock_by_type = {
        "Single Detached": get_row_series(t15, "Single Detached", 0),
        "Single Attached": get_row_series(t15, "Single Attached", 0),
        "Apartments": get_row_series(t15, "Apartments", 0),
        "Mobile Homes": get_row_series(t15, "Mobile Homes", 0),
    }

    return housing_thousand, housing_stock_by_type


# ==============================================================================
# BUILDING SHARES
# ==============================================================================

def extract_building_shares(tables: Dict) -> Dict:
    """
    Extract building share percentages by type.

    Parameters
    ----------
    tables : dict
        Dictionary of loaded tables.

    Returns
    -------
    dict
        Building shares by type as fractions (0–1 scale).
    """
    t15 = tables["Table 15"]

    def convert_to_fraction(data_dict: Dict) -> Dict:
        """Convert percentage values to fractions (0–1 scale)."""
        result = {}
        for year, val in data_dict.items():
            if val is None:
                result[year] = None
            elif isinstance(val, (int, float)):
                result[year] = val / 100
            else:
                try:
                    result[year] = float(val) / 100
                except (ValueError, TypeError):
                    result[year] = None
        return result

    return {
        "Single Detached": convert_to_fraction(get_row_series(t15, "Single Detached", 1)),
        "Single Attached": convert_to_fraction(get_row_series(t15, "Single Attached", 1)),
        "Apartments": convert_to_fraction(get_row_series(t15, "Apartments", 1)),
        "Mobile Homes": convert_to_fraction(get_row_series(t15, "Mobile Homes", 1)),
    }


# ==============================================================================
# FLOOR SPACE
# ==============================================================================

def extract_floorspace_per_building(tables: Dict, housing_stock_by_type: Dict) -> Dict:
    """
    Extract and calculate floor space per building by type.

    Parameters
    ----------
    tables : dict
        Dictionary of loaded tables.
    housing_stock_by_type : dict
        Housing stock by building type (from extract_housing_stock).

    Returns
    -------
    dict
        Floor space per building (m²/building) by type.
    """
    t18 = tables["Table 18"]

    fs_data_raw = {
        "Single Detached": get_row_series(t18, "Single Detached", 0),
        "Single Attached": get_row_series(t18, "Single Attached", 0),
        "Apartments": get_row_series(t18, "Apartments", 0),
        "Mobile Homes": get_row_series(t18, "Mobile Homes", 0),
    }

    floorspace_per_building = {}
    for name in ["Single Detached", "Single Attached", "Apartments", "Mobile Homes"]:
        numerator = {k: v * 1e6 if v is not None else None for k, v in fs_data_raw[name].items()}
        denominator = {k: v * 1000 if v is not None else None for k, v in housing_stock_by_type[name].items()}
        floorspace_per_building[name] = divide_dicts(numerator, denominator)

    return floorspace_per_building


# ==============================================================================
# APPLIANCES
# ==============================================================================

def extract_appliances(tables: Dict) -> Tuple[Dict, List[str]]:
    """
    Extract appliances per household by type.

    Parameters
    ----------
    tables : dict
        Dictionary of loaded tables.

    Returns
    -------
    tuple of (dict, list of str)
        (appliances_per_household, appliance_types)
    """
    t31 = tables["Table 31"]

    appliance_mapping = {
        "Refrigerator": "Refrigerators",
        "Freezer": "Freezers",
        "Range": "Ranges",
        "Dishwasher": "Dishwashing",
        "Clothes Washer": "Clothes Washing",
        "Other Appliances1": "Minor Appliances",
    }

    appliances_per_household = {}
    for excel_name, cims_name in appliance_mapping.items():
        try:
            appliances_per_household[cims_name] = get_row_series(t31, excel_name, match_n=1)
        except KeyError:
            appliances_per_household[cims_name] = {}

    return appliances_per_household, list(appliance_mapping.values())


# ==============================================================================
# VINTAGES
# ==============================================================================

def extract_vintages(tables: Dict, building_shares: Dict, province_code: str = '') -> Tuple[Dict, Dict]:
    """
    Extract and aggregate vintage (age) data into bins by density type.

    Parameters
    ----------
    tables : dict
        Dictionary of loaded tables.
    building_shares : dict
        Building shares by type (from extract_building_shares).
    province_code : str, optional
        Province code. MB uses "2022_after" as the last vintage bin; all
        others use "2021_after".

    Returns
    -------
    tuple of (dict, dict)
        (vintage_bins_lowmed, vintage_bins_high)
    """
    t19 = tables["Table 19"]
    t20 = tables["Table 20"]

    last_vintage_label = "2022_after" if province_code.upper() == 'MB' else "2021_after"

    vintage_labels = [
        "Before 1946", "1946–1960", "1961–1977", "1978–1983", "1984–1995",
        "1996–2000", "2001–2005", "2006–2010", "2011–2015", "2016_2020", last_vintage_label,
    ]

    def convert_to_fraction(data_dict: Dict) -> Dict:
        """Convert percentage values to fractions (0–1 scale)."""
        result = {}
        for year, val in data_dict.items():
            if val is None:
                result[year] = None
            elif isinstance(val, (int, float)):
                result[year] = val / 100
            else:
                try:
                    result[year] = float(val) / 100
                except (ValueError, TypeError):
                    result[year] = None
        return result

    vintage_data_detached = {label: convert_to_fraction(get_row_series(t19, label, 1)) for label in vintage_labels}
    vintage_data_attached = {label: convert_to_fraction(get_row_series(t19, label, 3)) for label in vintage_labels}
    vintage_data_apartments = {label: convert_to_fraction(get_row_series(t20, label, 1)) for label in vintage_labels}
    vintage_data_mobile = {label: convert_to_fraction(get_row_series(t20, label, 3)) for label in vintage_labels}

    vintage_data_lowmed = {
        label: weighted_average_dicts(
            [vintage_data_detached[label], vintage_data_attached[label], vintage_data_mobile[label]],
            [building_shares["Single Detached"], building_shares["Single Attached"], building_shares["Mobile Homes"]],
        )
        for label in vintage_labels
    }

    vintage_data_high = vintage_data_apartments

    vintage_bins_lowmed = {
        "<1960": sum_dicts(vintage_data_lowmed["Before 1946"], vintage_data_lowmed["1946–1960"]),
        "1961-1980": sum_dicts(vintage_data_lowmed["1961–1977"], vintage_data_lowmed["1978–1983"]),
        "1981-2000": sum_dicts(vintage_data_lowmed["1984–1995"], vintage_data_lowmed["1996–2000"]),
        "2001-2020": sum_dicts(vintage_data_lowmed["2001–2005"], vintage_data_lowmed["2006–2010"],
                               vintage_data_lowmed["2011–2015"], vintage_data_lowmed["2016_2020"]),
        "2021-2035": vintage_data_lowmed[last_vintage_label],
    }

    vintage_bins_high = {
        "<1960": sum_dicts(vintage_data_high["Before 1946"], vintage_data_high["1946–1960"]),
        "1961-1980": sum_dicts(vintage_data_high["1961–1977"], vintage_data_high["1978–1983"]),
        "1981-2000": sum_dicts(vintage_data_high["1984–1995"], vintage_data_high["1996–2000"]),
        "2001-2020": sum_dicts(vintage_data_high["2001–2005"], vintage_data_high["2006–2010"],
                               vintage_data_high["2011–2015"], vintage_data_high["2016_2020"]),
        "2021-2035": vintage_data_high[last_vintage_label],
    }

    return vintage_bins_lowmed, vintage_bins_high


# ==============================================================================
# HEATING TECHNOLOGIES
# ==============================================================================

def extract_heating_technologies(tables: Dict, building_shares: Dict, province_code: str) -> Dict:
    """
    Extract heating technology shares by density type.

    For BC, returns both Marine and Cold climate data.
    For all other provinces, returns only Cold climate data.

    Parameters
    ----------
    tables : dict
        Dictionary of loaded tables.
    building_shares : dict
        Building shares by type.
    province_code : str
        2-letter province/territory code.

    Returns
    -------
    dict
        Heating data for the province. BC includes marine climate data;
        all others include cold climate data only.
    """
    t22, t23, t24, t25 = tables["Table 22"], tables["Table 23"], tables["Table 24"], tables["Table 25"]
    is_bc = province_code.upper() == 'BC'

    heating_tech_cold = {
        "NG - Low Efficiency": ["Natural Gas – Normal Efficiency", "Other1", "Natural Gas/Electric"],
        "NG - Medium Efficiency": ["Natural Gas – Medium Efficiency"],
        "NG - High Efficiency": ["Natural Gas – High Efficiency"],
        "Electric - Resistance": ["Electric"],
        "Heating Oil - Low Efficiency": ["Heating Oil – Normal Efficiency", "Heating Oil/Electric"],
        "Heating Oil - Medium Efficiency": ["Heating Oil – Medium Efficiency", "Heating Oil – High Efficiency"],
        "Wood": ["Wood", "Wood/Electric", "Wood/Heating Oil"],
        "NG - ASHP / NG - backup": [],
        "Electric - ASHP / NG - backup": ["Heat Pump"],
        "Electric - ASHP / Electric - backup": [],
    }

    def convert_to_fraction(data_dict: Dict) -> Dict:
        """Convert percentage values to fractions (0–1 scale)."""
        result = {}
        for year, val in data_dict.items():
            if val is None:
                result[year] = None
            elif isinstance(val, (int, float)):
                result[year] = val / 100
            else:
                try:
                    result[year] = float(val) / 100
                except (ValueError, TypeError):
                    result[year] = None
        return result

    def extract_heating_bucket(table, bucket_technologies: List[str]) -> Dict:
        """Extract and sum multiple technologies into a single bucket."""
        if not bucket_technologies:
            return {}
        tech_data = []
        for tech_name in bucket_technologies:
            try:
                data = get_row_series(table, tech_name, 1)
                if data:
                    tech_data.append(convert_to_fraction(data))
            except KeyError:
                pass
        return sum_dicts(*tech_data) if tech_data else {}

    heating_detached_cold = {tech: extract_heating_bucket(t22, techs) for tech, techs in heating_tech_cold.items()}
    heating_attached_cold = {tech: extract_heating_bucket(t23, techs) for tech, techs in heating_tech_cold.items()}
    heating_apartments_cold = {tech: extract_heating_bucket(t24, techs) for tech, techs in heating_tech_cold.items()}
    heating_mobile_cold = {tech: extract_heating_bucket(t25, techs) for tech, techs in heating_tech_cold.items()}

    heating_lowmed_cold = {
        heat_tech: weighted_average_dicts(
            [heating_detached_cold[heat_tech], heating_attached_cold[heat_tech], heating_mobile_cold[heat_tech]],
            [building_shares["Single Detached"], building_shares["Single Attached"], building_shares["Mobile Homes"]],
        )
        for heat_tech in heating_tech_cold.keys()
    }

    result = {
        'heating_lowmed_cold': heating_lowmed_cold,
        'heating_high_cold': heating_apartments_cold,
        'heating_tech_cold': list(heating_tech_cold.keys()),
    }

    if is_bc:
        heating_tech_marine = {
            "NG - Low Efficiency": ["Natural Gas – Normal Efficiency", "Other1", "Natural Gas/Electric"],
            "NG - Medium Efficiency": ["Natural Gas – Medium Efficiency"],
            "NG - High Efficiency": ["Natural Gas – High Efficiency"],
            "Electric - Resistance": ["Electric"],
            "Heating Oil - Low Efficiency": ["Heating Oil – Normal Efficiency", "Heating Oil/Electric"],
            "Heating Oil - Medium Efficiency": ["Heating Oil – Medium Efficiency", "Heating Oil – High Efficiency"],
            "Wood": ["Wood", "Wood/Electric", "Wood/Heating Oil"],
            "NG - ASHP": [],
            "Electric - ASHP": ["Heat Pump"],
        }

        heating_detached_marine = {tech: extract_heating_bucket(t22, techs) for tech, techs in heating_tech_marine.items()}
        heating_attached_marine = {tech: extract_heating_bucket(t23, techs) for tech, techs in heating_tech_marine.items()}
        heating_apartments_marine = {tech: extract_heating_bucket(t24, techs) for tech, techs in heating_tech_marine.items()}
        heating_mobile_marine = {tech: extract_heating_bucket(t25, techs) for tech, techs in heating_tech_marine.items()}

        heating_lowmed_marine = {
            heat_tech: weighted_average_dicts(
                [heating_detached_marine[heat_tech], heating_attached_marine[heat_tech], heating_mobile_marine[heat_tech]],
                [building_shares["Single Detached"], building_shares["Single Attached"], building_shares["Mobile Homes"]],
            )
            for heat_tech in heating_tech_marine.keys()
        }

        result.update({
            'heating_lowmed_marine': heating_lowmed_marine,
            'heating_high_marine': heating_apartments_marine,
            'heating_tech_marine': list(heating_tech_marine.keys()),
        })

    return result


# ==============================================================================
# COOLING TECHNOLOGIES
# ==============================================================================

def extract_cooling_technologies(tables: Dict, province_code: Optional[str] = None) -> Dict:
    """
    Extract cooling technology shares.

    For territories (TR), missing data for 2002 and 2014 is filled with
    2001 and 2013 data respectively.

    Parameters
    ----------
    tables : dict
        Dictionary of loaded tables.
    province_code : str, optional
        2-letter province/territory code (e.g., 'TR' for Territories).

    Returns
    -------
    dict
        Cooling shares by technology.
    """
    t4 = tables["Table 4"]

    def convert_to_fraction(data_dict: Dict) -> Dict:
        """Convert percentage values to fractions (0–1 scale)."""
        result = {}
        for year, val in data_dict.items():
            if val is None:
                result[year] = None
            elif isinstance(val, (int, float)):
                result[year] = val / 100
            else:
                try:
                    result[year] = float(val) / 100
                except (ValueError, TypeError):
                    result[year] = None
        return result

    cooling_share_data = {
        "Room": convert_to_fraction(get_row_series(t4, "Room", 1)),
        "Central": convert_to_fraction(get_row_series(t4, "Central", 1)),
    }

    if province_code and province_code.upper() == 'TR':
        for cooling_type in ["Room", "Central"]:
            if 2001 in cooling_share_data[cooling_type] and cooling_share_data[cooling_type][2001] is not None:
                if (2002 not in cooling_share_data[cooling_type] or
                        cooling_share_data[cooling_type][2002] is None or
                        cooling_share_data[cooling_type][2002] == 0):
                    cooling_share_data[cooling_type][2002] = cooling_share_data[cooling_type][2001]
                    print(f"  → Filled {cooling_type} 2002 with 2001 value: {cooling_share_data[cooling_type][2001]}")

            if 2013 in cooling_share_data[cooling_type] and cooling_share_data[cooling_type][2013] is not None:
                if (2014 not in cooling_share_data[cooling_type] or
                        cooling_share_data[cooling_type][2014] is None or
                        cooling_share_data[cooling_type][2014] == 0):
                    cooling_share_data[cooling_type][2014] = cooling_share_data[cooling_type][2013]
                    print(f"  → Filled {cooling_type} 2014 with 2013 value: {cooling_share_data[cooling_type][2013]}")

    return cooling_share_data


# ==============================================================================
# WATER HEATING
# ==============================================================================

def extract_water_heating(tables: Dict, building_shares: Dict, heating_data: Dict, province_code: str) -> Dict:
    """
    Extract and calculate water heating technology shares.

    As CIMS distinguishes multiple efficiencies for water heating equipment but
    the CEUD does not, the efficiency split from heating technologies is applied
    to water heating.

    Parameters
    ----------
    tables : dict
        Dictionary of loaded tables.
    building_shares : dict
        Building shares by type.
    heating_data : dict
        Heating technology data from extract_heating_technologies.
    province_code : str
        2-letter province/territory code.

    Returns
    -------
    dict
        Dictionary containing water heating data (Cold climate only for all provinces).
    """
    t10, t11 = tables["Table 10"], tables["Table 11"]

    wh_detached = get_row_series(t11, "Single Detached", match_n=1)
    wh_attached = get_row_series(t11, "Single Attached", match_n=1)
    wh_apartments = get_row_series(t11, "Apartments", match_n=1)
    wh_mobile = get_row_series(t11, "Mobile Homes", match_n=1)

    def to_fraction(d: Dict) -> Dict:
        return {year: val / 100 if val is not None else None for year, val in d.items()}

    wh_lowmed = sum_dicts(to_fraction(wh_detached), to_fraction(wh_attached), to_fraction(wh_mobile))
    wh_high = to_fraction(wh_apartments)

    wh_techs_to_extract = ["Natural Gas", "Other2", "Electricity", "Wood", "Heating Oil"]
    wh_tech_shares: Dict[str, Dict] = {}
    for wh_tech in wh_techs_to_extract:
        try:
            wh_data = get_row_series(t10, wh_tech, match_n=1)
            converted = {}
            for year, val in wh_data.items():
                if val is None:
                    converted[year] = None
                elif isinstance(val, (int, float)):
                    converted[year] = float(val) / 100
                else:
                    try:
                        converted[year] = float(val) / 100
                    except (ValueError, TypeError):
                        converted[year] = None
            wh_tech_shares[wh_tech] = converted
        except KeyError:
            wh_tech_shares[wh_tech] = {}

    ng_wh_share = sum_dicts(wh_tech_shares.get("Natural Gas", {}), wh_tech_shares.get("Other2", {}))
    oil_wh_share = wh_tech_shares.get("Heating Oil", {})
    elec_wh_share = sum_dicts(wh_tech_shares.get("Electricity", {}), wh_tech_shares.get("Wood", {}))

    heating_lowmed_cold = heating_data['heating_lowmed_cold']
    heating_high_cold = heating_data['heating_high_cold']

    # Cold climate — LowMed density
    ng_total_lowmed_cold = sum_dicts(
        heating_lowmed_cold.get("NG - Low Efficiency", {}),
        heating_lowmed_cold.get("NG - Medium Efficiency", {}),
        heating_lowmed_cold.get("NG - High Efficiency", {}),
    )
    ng_existing_share_lowmed_cold = divide_dicts(heating_lowmed_cold.get("NG - Low Efficiency", {}), ng_total_lowmed_cold)
    ng_standard_share_lowmed_cold = divide_dicts(heating_lowmed_cold.get("NG - Medium Efficiency", {}), ng_total_lowmed_cold)
    ng_efficient_share_lowmed_cold = divide_dicts(heating_lowmed_cold.get("NG - High Efficiency", {}), ng_total_lowmed_cold)

    oil_total_lowmed_cold = sum_dicts(
        heating_lowmed_cold.get("Heating Oil - Low Efficiency", {}),
        heating_lowmed_cold.get("Heating Oil - Medium Efficiency", {}),
    )
    oil_existing_share_lowmed_cold = divide_dicts(heating_lowmed_cold.get("Heating Oil - Low Efficiency", {}), oil_total_lowmed_cold)
    oil_efficient_share_lowmed_cold = divide_dicts(heating_lowmed_cold.get("Heating Oil - Medium Efficiency", {}), oil_total_lowmed_cold)

    # Cold climate — High density
    ng_total_high_cold = sum_dicts(
        heating_high_cold.get("NG - Low Efficiency", {}),
        heating_high_cold.get("NG - Medium Efficiency", {}),
        heating_high_cold.get("NG - High Efficiency", {}),
    )
    ng_existing_share_high_cold = divide_dicts(heating_high_cold.get("NG - Low Efficiency", {}), ng_total_high_cold)
    ng_standard_share_high_cold = divide_dicts(heating_high_cold.get("NG - Medium Efficiency", {}), ng_total_high_cold)
    ng_efficient_share_high_cold = divide_dicts(heating_high_cold.get("NG - High Efficiency", {}), ng_total_high_cold)

    oil_total_high_cold = sum_dicts(
        heating_high_cold.get("Heating Oil - Low Efficiency", {}),
        heating_high_cold.get("Heating Oil - Medium Efficiency", {}),
    )
    oil_existing_share_high_cold = divide_dicts(heating_high_cold.get("Heating Oil - Low Efficiency", {}), oil_total_high_cold)
    oil_efficient_share_high_cold = divide_dicts(heating_high_cold.get("Heating Oil - Medium Efficiency", {}), oil_total_high_cold)

    wh_tech_lowmed = {
        "NG - Storage - Low Efficiency": multiply_dicts(ng_wh_share, ng_existing_share_lowmed_cold),
        "NG - Storage - Medium Efficiency": multiply_dicts(ng_wh_share, ng_standard_share_lowmed_cold),
        "NG - Storage - High Efficiency": multiply_dicts(ng_wh_share, ng_efficient_share_lowmed_cold),
        "Oil - Storage - Low Efficiency": multiply_dicts(oil_wh_share, oil_existing_share_lowmed_cold),
        "Oil - Storage - Medium Efficiency": multiply_dicts(oil_wh_share, oil_efficient_share_lowmed_cold),
        "Electric - Storage - Low Efficiency": elec_wh_share,
    }
    wh_tech_high = {
        "NG - Storage - Low Efficiency": multiply_dicts(ng_wh_share, ng_existing_share_high_cold),
        "NG - Storage - Medium Efficiency": multiply_dicts(ng_wh_share, ng_standard_share_high_cold),
        "NG - Storage - High Efficiency": multiply_dicts(ng_wh_share, ng_efficient_share_high_cold),
        "Oil - Storage - Low Efficiency": multiply_dicts(oil_wh_share, oil_existing_share_high_cold),
        "Oil - Storage - Medium Efficiency": multiply_dicts(oil_wh_share, oil_efficient_share_high_cold),
        "Electric - Storage - Low Efficiency": elec_wh_share,
    }

    return {
        'wh_lowmed': wh_lowmed,
        'wh_high': wh_high,
        'wh_tech_lowmed': wh_tech_lowmed,
        'wh_tech_high': wh_tech_high,
    }


# ==============================================================================
# MAIN EXTRACTION FUNCTION
# ==============================================================================

def extract_all_data(
    province_code: str,
    apply_projections: bool = True,
    projection_params: Optional[Dict] = None,
) -> Dict:
    """
    Extract all residential data for a given province/territory.

    Parameters
    ----------
    province_code : str
        2-letter province/territory code (e.g., 'bc', 'on', 'ab').
    apply_projections : bool
        Whether to apply projection extensions (default True).
    projection_params : dict, optional
        Projection parameters. If None, will be loaded from file.

    Returns
    -------
    dict
        Dictionary containing all extracted data. BC includes Marine climate
        heating data; all provinces include Cold climate data.

    Raises
    ------
    ValueError
        If an invalid province code is provided.
    """
    province_code_upper = province_code.upper()
    if province_code_upper not in PROVINCES:
        raise ValueError(f"Invalid province code: {province_code}. Valid codes: {list(PROVINCES.keys())}")

    is_bc = province_code_upper == 'BC'
    tables = load_tables(province_code)

    housing_thousand, housing_stock_by_type = extract_housing_stock(tables)
    building_shares = extract_building_shares(tables)
    floorspace_per_building = extract_floorspace_per_building(tables, housing_stock_by_type)
    appliances_per_household, appliance_types = extract_appliances(tables)
    vintage_bins_lowmed, vintage_bins_high = extract_vintages(tables, building_shares, province_code)
    heating_data = extract_heating_technologies(tables, building_shares, province_code)
    cooling_share_data = extract_cooling_technologies(tables, province_code)
    water_heating_data = extract_water_heating(tables, building_shares, heating_data, province_code)

    result = {
        'province_code': province_code_upper,
        'province_name': PROVINCES[province_code_upper],
        'housing_thousand': housing_thousand,
        'housing_stock_by_type': housing_stock_by_type,
        'building_shares': building_shares,
        'share_data': building_shares,  # Alias
        'floorspace_per_building': floorspace_per_building,
        'fs_per_building': floorspace_per_building,  # Alias
        'appliances_per_household': appliances_per_household,
        'appliance_types': appliance_types,
        'vintage_bins_lowmed': vintage_bins_lowmed,
        'vintage_bins_high': vintage_bins_high,
        'cooling_share_data': cooling_share_data,
        'heating_lowmed_cold': heating_data['heating_lowmed_cold'],
        'heating_high_cold': heating_data['heating_high_cold'],
        'heating_tech_cold': heating_data['heating_tech_cold'],
        'wh_lowmed': water_heating_data['wh_lowmed'],
        'wh_high': water_heating_data['wh_high'],
        'wh_tech_lowmed': water_heating_data['wh_tech_lowmed'],
        'wh_tech_high': water_heating_data['wh_tech_high'],
    }

    if is_bc:
        result.update({
            'heating_lowmed_marine': heating_data['heating_lowmed_marine'],
            'heating_high_marine': heating_data['heating_high_marine'],
            'heating_tech_marine': heating_data['heating_tech_marine'],
        })

    if apply_projections:
        if projection_params is None:
            projection_params = load_projection_params()
        result = apply_extensions(result, province_code, projection_params)

    return result


# ==============================================================================
# BATCH EXTRACTION FOR ALL PROVINCES
# ==============================================================================

def extract_all_provinces(
    province_codes: Optional[List[str]] = None,
    apply_projections: bool = True,
) -> Dict:
    """
    Extract data for multiple provinces at once.

    Parameters
    ----------
    province_codes : list of str, optional
        Province codes to extract. If None, extracts all provinces.
    apply_projections : bool
        Whether to apply projection extensions (default True).

    Returns
    -------
    dict
        Dictionary with province codes as keys and extracted data as values.
    """
    if province_codes is None:
        province_codes = list(PROVINCES.keys())

    projection_params = load_projection_params() if apply_projections else None

    results = {}
    failed = []

    for prov in province_codes:
        try:
            print(f"Extracting data for {prov}...")
            results[prov] = extract_all_data(prov, apply_projections, projection_params)
            print(f"✅ {prov} — {PROVINCES[prov.upper()]} complete")
        except Exception as e:
            print(f"❌ {prov} — {PROVINCES[prov.upper()]} failed: {e}")
            failed.append((prov, str(e)))

    if failed:
        print(f"\n⚠️  Failed provinces: {', '.join(p for p, _ in failed)}")

    return results


# ==============================================================================
# CSV EXPORT FUNCTIONS
# ==============================================================================

def flatten_dict_to_rows(
    data: Dict,
    variable_name: str,
    parameter: str,
    unit: str,
    source: str = "CEUD",
) -> List[Dict]:
    """
    Flatten a dictionary of data into rows for CSV export.

    Handles both simple dicts (year → value) and nested dicts
    (category → year → value).

    Parameters
    ----------
    data : dict
        The data dictionary to flatten.
    variable_name : str
        Name of the variable.
    parameter : str
        Parameter type (e.g., 'service_request', 'market_share_total').
    unit : str
        Unit of measurement.
    source : str
        Data source (default 'CEUD').

    Returns
    -------
    list of dict
        Row dictionaries with keys: variable, category, parameter, unit, source, year, value.
    """
    rows = []

    if not isinstance(data, dict):
        return rows

    first_value = next(iter(data.values())) if data else None

    if isinstance(first_value, dict):
        for category, year_values in data.items():
            if isinstance(year_values, dict):
                for year, value in year_values.items():
                    if value is not None:
                        rows.append({
                            'variable': variable_name,
                            'category': category,
                            'parameter': parameter,
                            'unit': unit,
                            'source': source,
                            'year': year,
                            'value': value,
                        })
    else:
        for year, value in data.items():
            if value is not None:
                rows.append({
                    'variable': variable_name,
                    'category': None,
                    'parameter': parameter,
                    'unit': unit,
                    'source': source,
                    'year': year,
                    'value': value,
                })

    return rows


VARIABLE_CONFIGS = [
    {'key': 'housing_thousand', 'name': 'housing_thousand', 'parameter': 'service_request', 'unit': 'household'},
    {'key': 'building_shares', 'name': 'building_shares', 'parameter': 'market_share_total', 'unit': '%'},
    {'key': 'floorspace_per_building', 'name': 'floorspace_per_building', 'parameter': 'service_request', 'unit': 'm2/building'},
    {'key': 'appliances_per_household', 'name': 'appliances_per_household', 'parameter': 'service_request', 'unit': 'unit/building'},
    {'key': 'appliance_types', 'name': 'appliance_types', 'parameter': 'market_share_total', 'unit': 'unit/building'},
    {'key': 'vintage_bins_lowmed', 'name': 'vintage_bins_low_med', 'parameter': 'market_share_total', 'unit': '% of m2'},
    {'key': 'vintage_bins_high', 'name': 'vintage_bins_high', 'parameter': 'market_share_total', 'unit': '% of m2'},
    {'key': 'heating_lowmed_cold', 'name': 'heating_data_lowmed_cold', 'parameter': 'market_share_total', 'unit': '% of GJ of heat'},
    {'key': 'heating_high_cold', 'name': 'heating_data_high_cold', 'parameter': 'market_share_total', 'unit': '% of GJ of heat'},
    {'key': 'heating_tech_cold', 'name': 'heating_data_tech_cold', 'parameter': 'market_share_total', 'unit': '% of GJ of heat'},
    {'key': 'heating_lowmed_marine', 'name': 'heating_data_lowmed_marine', 'parameter': 'market_share_total', 'unit': '% of GJ of heat'},
    {'key': 'heating_high_marine', 'name': 'heating_data_high_marine', 'parameter': 'market_share_total', 'unit': '% of GJ of heat'},
    {'key': 'heating_tech_marine', 'name': 'heating_data_tech_marine', 'parameter': 'market_share_total', 'unit': '% of GJ of heat'},
    {'key': 'cooling_share_data', 'name': 'cooling_share_data', 'parameter': 'service_request', 'unit': 'GJ/GJ'},
    {'key': 'wh_lowmed', 'name': 'wh_lowmed', 'parameter': 'service_request', 'unit': 'GJ/GJ'},
    {'key': 'wh_high', 'name': 'wh_high', 'parameter': 'service_request', 'unit': 'GJ/GJ'},
    {'key': 'wh_tech_lowmed', 'name': 'wh_tech_lowmed', 'parameter': 'market_share_total', 'unit': '% of GJ of water heat'},
    {'key': 'wh_tech_high', 'name': 'wh_tech_high', 'parameter': 'market_share_total', 'unit': '% of GJ of water heat'},
]


def flatten_province_data(data: Dict, province_code: str) -> List[Dict]:
    """
    Flatten all variables for a single province into a list of row dicts,
    adding a 'province' column to each row.

    Parameters
    ----------
    data : dict
        Extracted data from extract_all_data().
    province_code : str
        Province code to tag each row with.

    Returns
    -------
    list of dict
        Row dictionaries with keys: province, variable, category, parameter, unit, source, year, value.
    """
    all_rows = []
    for config in VARIABLE_CONFIGS:
        key = config['key']
        if key not in data:
            continue
        rows = flatten_dict_to_rows(
            data[key],
            config['name'],
            config['parameter'],
            config['unit'],
        )
        for row in rows:
            row['province'] = province_code.upper()
        all_rows.extend(rows)
    return all_rows


def export_to_csv(data: Dict, output_file: Path, province_code: Optional[str] = None) -> str:
    """
    Export extracted data for a single province to CSV format.

    Parameters
    ----------
    data : dict
        Extracted data from extract_all_data().
    output_file : Path
        Path to output CSV file.
    province_code : str, optional
        Province code. Uses data['province_code'] if not provided.

    Returns
    -------
    str
        Path to the saved CSV file.
    """
    if province_code is None:
        province_code = data.get('province_code', 'UNKNOWN')

    all_rows = flatten_province_data(data, province_code)

    df = pd.DataFrame(all_rows)

    if not df.empty:
        df = df.sort_values(['province', 'variable', 'category', 'year'], na_position='first')

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"  ✅ Saved {len(df):,} rows to {output_path}")

    return str(output_path)


def main(
    province_codes: Optional[List[str]] = None,
    output_dir: Path = OUTPUT_DIR,
    apply_projections: bool = True,
    export_csv: bool = True,
) -> Dict:
    """
    Main execution function. Extracts data for all provinces and optionally
    exports to CSV files.

    Parameters
    ----------
    province_codes : list of str, optional
        Province codes to extract. If None, extracts all provinces.
    output_dir : Path
        Directory to save CSV files.
    apply_projections : bool
        Whether to apply projection extensions (default True).
    export_csv : bool
        Whether to export data to CSV files (default True).

    Returns
    -------
    dict
        Dictionary with province codes as keys and extracted data as values.
    """
    if province_codes is None:
        province_codes = list(PROVINCES.keys())

    print("=" * 80)
    print("RESIDENTIAL DATA EXTRACTION")
    print("=" * 80)
    print(f"Provinces:         {', '.join(province_codes)}")
    print(f"Apply projections: {apply_projections}")
    print(f"Export to CSV:     {export_csv}")
    if export_csv:
        print(f"Output directory:  {output_dir}")
    print("=" * 80)

    if export_csv:
        output_dir.mkdir(parents=True, exist_ok=True)

    projection_params = load_projection_params() if apply_projections else None

    results = {}
    failed = []
    all_rows = []

    for prov in province_codes:
        try:
            print(f"\n{prov} — {PROVINCES[prov.upper()]}:")
            data = extract_all_data(prov, apply_projections, projection_params)
            results[prov] = data
            print(f"  ✅ Extraction complete")

            if export_csv:
                rows = flatten_province_data(data, prov)
                all_rows.extend(rows)

        except Exception as e:
            print(f"  ❌ Failed: {e}")
            failed.append((prov, str(e)))

    if export_csv and all_rows:
        output_file = output_dir / "residential.csv"
        df = pd.DataFrame(all_rows)
        df = df.sort_values(['province', 'variable', 'category', 'year'], na_position='first')
        df.to_csv(output_file, index=False)
        print(f"\n  ✅ Saved {len(df):,} rows to {output_file}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✅ Successful: {len(results)}/{len(province_codes)} provinces")

    if failed:
        print(f"❌ Failed: {len(failed)} provinces")
        for prov, error in failed:
            print(f"   • {prov}: {error}")

    print("=" * 80)

    return results


if __name__ == "__main__":
    results = main()
