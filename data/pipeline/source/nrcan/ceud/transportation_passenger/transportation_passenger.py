"""
Transportation Passenger Pipeline

Extracts passenger-kilometres (kpkm = thousands of passenger-km) from the
NRCan CEUD (Comprehensive Energy Use Database) for all Canadian provinces,
projects activity to 2100, and produces historical technology market shares.

Key behavioural notes
---------------------
- BC uses tran_bct_e.xls (BC + Territories combined in CEUD).
- All output is in long format: (province, variable, category, parameter, unit, year, value)
- Unit for all pkm variables: 'kpkm' (thousands of passenger-km).
  CEUD reports pkm in millions; values are multiplied by 1000 on output.
- Technology market shares end at LAST_HIST_YEAR (from CONTROLS).
- k*pkm activity and mode-level k*pkm extend to 2100 via CAGR projection.

Variables extracted / derived
------------------------------
Activity (k*pkm, to 2100):
  car_kpkm, lt_kpkm, school_bus_kpkm, urban_transit_kpkm,
  intercity_bus_kpkm, motorcycle_kpkm, rail_kpkm, air_kpkm,
  walk_cycle_urban_kpkm, total_passenger_kpkm,
  urban_kpkm, intercity_land_kpkm, intercity_air_kpkm

Mode shares (ratio, to 2100):
  Mode.Urban, Mode.Intercity Land, Mode.Intercity Air

Urban tech shares (fraction, to LAST_HIST_YEAR):
  Walk Cycle Urban, Passenger Vehicle Urban SOV, Passenger Vehicle Urban HOV,
  Public Transit Urban

Intercity Land tech shares (fraction, to LAST_HIST_YEAR):
  Bus Intercity, Rail Intercity, Passenger Vehicle Intercity

Vehicle size shares (fraction, to LAST_HIST_YEAR):
  Car_small, Car_large, Light truck_small, Light truck_large

Motor tech shares (fraction, to LAST_HIST_YEAR):
  Gasoline Existing, Gasoline Standard, Hybrid, Plug-in Hybrid,
  BEV 500, BEV 800, Fuel Cell 650

Transit decomposition (fraction/ratio, to LAST_HIST_YEAR):
  PB_RT_ratio  (Public Bus k*pkm / Rapid Transit k*pkm)
  Bus Urban Diesel, Bus Urban NG, Bus Urban Electric, Ferry Urban

Intercity Bus fuel shares (fraction, to LAST_HIST_YEAR):
  Bus Intercity Diesel, Bus Intercity Gasoline

Intercity Rail tech shares (fraction, to LAST_HIST_YEAR):
  Rail Intercity Diesel, Rail Intercity Diesel Efficient,
  Rail Intercity Hybrid Biodiesel, Rail Intercity Hydrogen,
  Rail Intercity Electric
"""

from pathlib import Path
from typing import Optional
import sys

import polars as pl
import pandas as pd
import numpy as np

_current_file = Path(__file__)
_project_root = _current_file.parent.parent.parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from mappings_conversions.control import CONTROLS
from pipeline.utils.extractors.nrcan_ceud import row_to_series
from pipeline.utils.controls_conversions import BASE_PATH as _CIMS_BASE

# ==============================================================================
# CONFIGURATION
# ==============================================================================

BASE_PATH       = _CIMS_BASE / 'raw_data/nrcan/ceud/transportation'
CAN_FILE        = BASE_PATH / 'tran_ca_e.xls'
ASSUMPTIONS_CSV = _CIMS_BASE / 'raw_data/assumptions/transport_passenger_assumptions.csv'
STATSCAN_POP_PATH = _CIMS_BASE / 'raw_data/stats_can/population/1710000901.csv'
OUTPUT_DIR      = _CIMS_BASE / 'processed_data/nrcan/ceud'
LAST_HIST_YEAR  = CONTROLS["last_data_year"]["ceud"]

# National CEUD table numbers (tran_ca_e.xls -- different numbering from provincial)
CAN_TABLE_AIR  = 20  # contains "Energy Intensity1 (MJ/Pkm)"
CAN_TABLE_RAIL = 25  # contains "Passenger Rail Transportation Energy Intensity (MJ/Pkm)"

# BC uses the combined BC+Territories CEUD file (tran_bct_e.xls).
# Territories (YT, NT, NU) are absorbed into BC; their activity is negligible
# and was zero in the old code.
PROVINCES = {
    'AB': 'Alberta',
    'BC': 'British Columbia and Territories',
    'MB': 'Manitoba',
    'NB': 'New Brunswick',
    'NL': 'Newfoundland and Labrador',
    'NS': 'Nova Scotia',
    'ON': 'Ontario',
    'PE': 'Prince Edward Island',
    'QC': 'Quebec',
    'SK': 'Saskatchewan',
}

# BC maps to the BCT CEUD file; all other provinces map 1:1 by lowercase code.
_PROV_FILE_CODE = {'BC': 'bct'}

# Provincial table numbers (same across all provincial files)
TABLE_PASSENGER_AIR_FUEL   = 14  # Passenger air energy by fuel source
TABLE_PASSENGER_RAIL       = 17  # Passenger rail energy
TABLE_CAR_FUEL             = 20  # Car energy + passenger-km
TABLE_CAR_EXPLANATORY      = 21  # Car stock + average distance (vkm)
TABLE_PASSENGER_TRUCK_FUEL = 25  # PLT energy + passenger-km
TABLE_SCHOOL_BUS_FUEL      = 28  # School bus energy + passenger-km
TABLE_URBAN_TRANSIT_FUEL   = 29  # Urban transit energy + passenger-km
TABLE_INTERCITY_BUS_FUEL   = 30  # Inter-city bus energy + passenger-km
TABLE_BUS_EXPLANATORY      = 31  # Bus stock + average distance (all bus modes)
TABLE_MOTORCYCLE           = 32  # Motorcycle energy + passenger-km
TABLE_TRUCK_EXPLANATORY    = 37  # Light truck stock + average distance (vkm)


# ==============================================================================
# PROJECTION PARAMETERS (from trans_ceud_old.py)
# ==============================================================================

PROJ_HORIZON = 2100

# Province codes already match KPKM_ASSUMPTIONS keys directly; no remapping needed.


# LDV size split constants (applied to car / light-truck k*pkm fractions)
LDV_SMALL_FRAC = 0.33
LDV_LARGE_FRAC = 0.67

# Fraction of urban transit electricity allocated to Rapid Transit (vs Public Bus).
# BC = 0.67; all other provinces = 1.0 (all electric transit is rapid transit).
RAPID_TRANSIT_SHARE: dict[str, float] = {
    'BC': 0.670, 'AB': 1.0, 'SK': 1.0, 'MB': 1.0, 'ON': 1.0,
    'QC': 1.0,  'NB': 1.0, 'NS': 1.0, 'PE': 1.0, 'NL': 1.0,
}
# Benchmark-year Public Bus / Rapid Transit splits by region, used to interpolate
# annual PB/RT ratios within the historical period.
TRANSIT_SPLITS_BY_REGION: dict = {
    'BC': {
        'PB': {2000: 0.969135, 2005: 0.961025, 2010: 0.939984, 2015: 0.937402,
               2020: 0.869475, 2023: 0.922696},
        'RT': {2000: 0.030865, 2005: 0.038975, 2010: 0.060016, 2015: 0.062598,
               2020: 0.130525, 2023: 0.077304},
    },
    'AB': {
        'PB': {2000: 0.969459, 2005: 0.937292, 2010: 0.965297, 2015: 0.957658,
               2020: 0.927593, 2023: 0.951116},
        'RT': {2000: 0.030541, 2005: 0.062708, 2010: 0.034703, 2015: 0.042342,
               2020: 0.072407, 2023: 0.048884},
    },
    'SK': {
        'PB': {2000: 1.0, 2005: 0.998251, 2010: 0.999638, 2015: 1.0, 2020: 1.0, 2023: 1.0},
        'RT': {2000: 0.0, 2005: 0.001749, 2010: 0.000362, 2015: 0.0, 2020: 0.0, 2023: 0.0},
    },
    'MB': {
        'PB': {2000: 1.0, 2005: 1.0, 2010: 1.0, 2015: 1.0, 2020: 1.0, 2023: 1.0},
        'RT': {2000: 0.0, 2005: 0.0,   2010: 0.0, 2015: 0.0, 2020: 0.0, 2023: 0.0},
    },
    'ON': {
        'PB': {2000: 0.918187, 2005: 0.919065, 2010: 0.943391, 2015: 0.914087,
               2020: 0.860819, 2023: 0.894124},
        'RT': {2000: 0.081813, 2005: 0.080935, 2010: 0.056609, 2015: 0.085913,
               2020: 0.139181, 2023: 0.105876},
    },
    'QC': {
        'PB': {2000: 0.856522, 2005: 0.855182, 2010: 0.881854, 2015: 0.831281,
               2020: 0.728068, 2023: 0.804461},
        'RT': {2000: 0.143478, 2005: 0.144818, 2010: 0.118146, 2015: 0.168719,
               2020: 0.271932, 2023: 0.195539},
    },
    'AT': {
        'PB': {2000: 1.0, 2005: 0.996659, 2010: 1.0, 2015: 0.999608,
               2020: 0.999202, 2023: 0.999454},
        'RT': {2000: 0.0, 2005: 0.003341, 2010: 0.0, 2015: 0.000392,
               2020: 0.000798, 2023: 0.000546},
    },
}

# Ferry Urban anchor values for BC (k*pkm); all other provinces are 0.
FERRY_URBAN_BC_ANCHORS: dict[int, float] = {
    2000: 3.24 * 5471900 / 1000.0,
    2005: 3.24 * 5016000 / 1000.0,
    2010: 3.24 * 6735200 / 1000.0,
    2016: 3.24 * 5442000 / 1000.0,
    2019: 3.24 * 6263400 / 1000.0,
    2020: 3.24 * 2305800 / 1000.0,
    2021: 3.24 * 2553200 / 1000.0,
    2022: 3.24 * 4245700 / 1000.0,
}

# Technology market share defaults (fraction 0-1, historical base year).
MOTOR_TECH_DEFAULTS: dict[str, float] = {
    'Gasoline Existing':  1.0,
    'Gasoline Standard':  0.0,
    'Gasoline Efficient': 0.0,
    'Hybrid':             0.0,
    'Plug-in Hybrid':     0.0,
    'BEV 500':            0.0,
    'BEV 800':            0.0,
    'Fuel Cell 650':      0.0,
}

AIR_TECH_DEFAULTS: dict[str, float] = {
    'Air Intercity':           1.0,
    'Air Intercity Efficient': 0.0,
    'Air Intercity Electric':  0.0,
    'Air Intercity Hydrogen':  0.0,
}

INTERCITY_BUS_FUEL_DEFAULTS: dict[str, float] = {
    'Bus Intercity Hybrid':           0.0,
    'Bus Intercity Hybrid Biodiesel': 0.0,
    'Bus Intercity NG':               0.0,
    'Bus Intercity Hydrogen':         0.0,
}

RAIL_TECH_DEFAULTS: dict[str, float] = {
    'Rail Intercity Diesel':           1.0,
    'Rail Intercity Diesel Efficient': 0.0,
    'Rail Intercity Hybrid Biodiesel': 0.0,
    'Rail Intercity Hydrogen':         0.0,
    'Rail Intercity Electric':         0.0,
}

# Fuel label candidates for robust extraction from CEUD provincial tables
_TRANSIT_FUEL_LABELS: dict[str, list[str]] = {
    'diesel':   ['Diesel fuel oil', 'Diesel Fuel Oil', 'diesel fuel oil'],
    'ng':       ['Natural gas',     'Natural Gas',     'natural gas'],
    'elec':     ['Electricity',     'electricity'],
}
_INTERCITY_BUS_FUEL_LABELS: dict[str, list[str]] = {
    'diesel':   ['Diesel fuel oil', 'Diesel Fuel Oil', 'diesel fuel oil'],
    'gasoline': ['Motor gasoline',  'Motor Gasoline',  'motor gasoline'],
}


# ==============================================================================
# HELPERS
# ==============================================================================


def _long(province: str, variable: str, category: str, parameter: str,
          unit: str, series: pd.Series, source: str = 'CEUD') -> pl.DataFrame:
    """Convert a year-indexed pd.Series to a Polars long-format DataFrame."""
    years  = [int(y)   for y, v in series.items() if pd.notna(v)]
    values = [float(v) for _, v in series.items() if pd.notna(v)]
    n = len(years)
    return pl.DataFrame(
        {
            'province':  [province]  * n,
            'variable':  [variable]  * n,
            'category':  [category]  * n,
            'parameter': [parameter] * n,
            'unit':      [unit]      * n,
            'source':    [source]    * n,
            'year':      years,
            'value':     values,
        },
        schema_overrides={'year': pl.Int32, 'value': pl.Float64},
    )


def _load_can_table(sheet_name: str) -> pl.DataFrame:
    """Load and numeric-cast a sheet from the national CEUD file."""
    df = pl.read_excel(str(CAN_FILE), sheet_name=sheet_name, has_header=False)
    cast_exprs = [
        pl.col(c).str.strip_chars().cast(pl.Float64, strict=False)
        for c in df.columns[2:]
        if df[c].dtype in (pl.String, pl.Utf8)
    ]
    return df.with_columns(cast_exprs) if cast_exprs else df


def _load_can_intensities() -> dict[str, pd.Series]:
    """
    Read year-indexed rail and air energy data from the national CEUD file.

    Sources (tran_ca_e.xls):
        Rail -- Table 25: "Passenger Rail Transportation Energy Intensity (MJ/Pkm)"
        Air  -- Table 20: total national energy (PJ) and pkm (millions)

    Air intensity is NOT read directly from the table because Table 20 reports the
    total (domestic + international) intensity.  The domestic-calibrated intensity
    is computed later in load_projection_params() once the domestic share fractions
    are available from the assumptions CSV.
    """
    if not CAN_FILE.exists():
        return {}
    result: dict[str, pd.Series] = {}
    try:
        t = _load_can_table(f"Table {CAN_TABLE_RAIL}")
        result['rail_intensity_mj_per_kpkm'] = row_to_series(
            t, "Passenger Rail Transportation Energy Intensity (MJ/Pkm)"
        )
    except Exception:
        pass
    try:
        t = _load_can_table(f"Table {CAN_TABLE_AIR}")
        result['_air_total_energy_pj']  = row_to_series(
            t, "Passenger Air Transportation Energy Use (PJ)"
        )
        result['_air_total_pkm_mpkm'] = row_to_series(
            t, "Passenger-kilometres1 (millions)"
        )
    except Exception:
        pass
    return result


def _load_can_ldv_occupancy() -> dict[str, pd.Series]:
    """
    Load national car and light truck occupancy (Mpkm / Mvkm) from tran_ca_e.xls.

    Provincial car and lt pkm are derived as:
        kpkm = stock_thousands * avg_vkm_km_per_veh * occupancy_Mpkm_per_Mvkm
    This matches the old trans_ceud_old.py approach, which avoids reading pkm
    directly from provincial CEUD tables (those use higher provincial occupancy
    rates and overstate activity).

    Source sheets (polars 0-indexed, year header at row 2, data cols start at 2):
        Passenger1 row 26 -- Car activity (millions pkm)
        Passenger1 row 27 -- Light Truck activity (millions pkm)
        Passenger4 row 9  -- Car stock (thousands vehicles)
        Passenger4 row 10 -- Light Truck stock (thousands vehicles)
        Passenger4 row 13 -- Car average annual distance (km / vehicle)
        Passenger4 row 14 -- Light Truck average annual distance (km / vehicle)
    """
    if not CAN_FILE.exists():
        return {}

    def _rows(sheet: str, indices: list[int]) -> dict[int, pd.Series]:
        df  = _load_can_table(sheet)
        arr = df.to_numpy()
        # Year header: scan rows 0-5 for first col-2 value that looks like a year
        yr_cols: dict[int, int] = {}
        for header_row in range(min(6, arr.shape[0])):
            try:
                yr = int(float(arr[header_row, 2]))
                if 1990 <= yr <= 2040:
                    for c in range(2, arr.shape[1]):
                        try:
                            y = int(float(arr[header_row, c]))
                            if 1990 <= y <= 2040:
                                yr_cols[y] = c
                        except (ValueError, TypeError):
                            continue
                    break
            except (ValueError, TypeError):
                continue

        def _f(v) -> float:
            try:
                return float(v)
            except (ValueError, TypeError):
                return np.nan

        return {idx: pd.Series({yr: _f(arr[idx, c]) for yr, c in yr_cols.items()})
                for idx in indices}

    try:
        p1 = _rows('Passenger1', [26, 27])
        p4 = _rows('Passenger4', [9, 10, 13, 14])

        result: dict[str, pd.Series] = {}
        for label, pkm_s, stock_s, avkm_s in [
            ('car_occupancy', p1[26], p4[9],  p4[13]),
            ('lt_occupancy',  p1[27], p4[10], p4[14]),
        ]:
            common = sorted(set(pkm_s.index) & set(stock_s.index) & set(avkm_s.index))
            if not common:
                continue
            total_dist = (
                stock_s.reindex(common) * avkm_s.reindex(common) / 1000.0
            ).replace(0, np.nan)
            occ = (pkm_s.reindex(common) / total_dist).dropna()
            if not occ.empty:
                result[label] = occ
        return result

    except Exception:
        return {}


def _series_from_df(df: pl.DataFrame, variable: str, category: str = '') -> pd.Series:
    """Extract a year-sorted pd.Series from the long-format DataFrame."""
    mask = pl.col('variable') == variable
    if category:
        mask = mask & (pl.col('category') == category)
    sub = df.filter(mask).sort('year')
    if len(sub) == 0:
        return pd.Series(dtype=float)
    return pd.Series(sub['value'].to_list(), index=sub['year'].to_list())


def _try_fuel_row(table: pl.DataFrame, candidates: list[str]) -> pd.Series:
    """Try multiple label variants to extract a fuel energy row; returns empty Series on failure."""
    for label in candidates:
        try:
            s = row_to_series(table, label)
            if not s.dropna().empty:
                return s
        except Exception:
            continue
    return pd.Series(dtype=float)


def _compute_ferry_urban_bc(hist_years: list[int]) -> pd.Series:
    """
    Compute Ferry Urban k*pkm for BC using anchor values and linear interpolation.
    All years after 2022 are 0.
    """
    anchors = FERRY_URBAN_BC_ANCHORS
    anchor_yrs = sorted(anchors.keys())

    result = pd.Series(np.nan, index=hist_years, dtype=float)

    def _linseg(y0: int, y1: int) -> None:
        v0, v1 = anchors[y0], anchors[y1]
        n = y1 - y0
        for yy in range(y0, y1 + 1):
            if yy in result.index:
                result[yy] = v0 + (v1 - v0) * (yy - y0) / n

    for i in range(len(anchor_yrs) - 1):
        _linseg(anchor_yrs[i], anchor_yrs[i + 1])

    # Fill any remaining years -? last anchor with last anchor value
    last_anchor = anchor_yrs[-1]
    for y in hist_years:
        if y > last_anchor:
            result[y] = 0.0
        elif pd.isna(result.get(y)):
            result[y] = 0.0

    return result.dropna()


def _interpolate_splits(splits_dict: dict[int, float], years: list[int]) -> pd.Series:
    """
    Linearly interpolate a benchmark-year dict to annual values.
    Years before first key use first value; years after last key use last value.
    """
    if not splits_dict:
        return pd.Series(0.0, index=years)
    bench_years = sorted(splits_dict.keys())
    result = {}
    for y in years:
        if y <= bench_years[0]:
            result[y] = splits_dict[bench_years[0]]
        elif y >= bench_years[-1]:
            result[y] = splits_dict[bench_years[-1]]
        else:
            i = 0
            while i < len(bench_years) - 1 and bench_years[i + 1] <= y:
                i += 1
            y0, y1 = bench_years[i], bench_years[i + 1]
            v0, v1 = splits_dict[y0], splits_dict[y1]
            result[y] = v0 + (v1 - v0) * (y - y0) / (y1 - y0)
    return pd.Series(result)


# ==============================================================================
# PARAMETER LOADING
# ==============================================================================


def load_projection_params(assumptions_csv: Path = ASSUMPTIONS_CSV,
                           activity_cagr_csv: Path = None) -> dict:
    """
    Load transportation passenger parameters.

    Rail and air intensities come from tran_ca_e.xls (national CEUD).
    Walk/cycle share, urban vehicle share, and aviation domestic shares
    come from transport_passenger_assumptions.csv.

    Returns
    -------
    dict with keys:
        'rail_intensity_mj_per_kpkm'  -- pd.Series (national CEUD Table 25)
        'air_intensity_mj_per_kpkm'   -- pd.Series (national CEUD Table 20)
        'walk_cycle_share'           -- float, fraction of total urban travel
        'urban_vehicle_share'        -- float, urban fraction of car + PLT pkm
        'aviation_domestic_share'    -- dict {province_code: float}
    """
    params: dict = _load_can_intensities()
    params.update(_load_can_ldv_occupancy())

    try:
        raw = pd.read_csv(assumptions_csv, header=None, dtype=str)
    except FileNotFoundError:
        return params

    arr = raw.fillna('').to_numpy()

    def cell(row, col: int) -> str:
        return str(row[col]).strip() if col < len(row) else ''

    def parse_pct(v: str) -> Optional[float]:
        s = v.strip().rstrip('%')
        try:
            return float(s) / 100.0
        except ValueError:
            return None

    # walk/cycle share: row where col 5 = "Walking and Cycling", col 8 = 0.89%
    for row in arr:
        if cell(row, 5) == 'Walking and Cycling':
            val = parse_pct(cell(row, 8))
            if val is not None:
                params['walk_cycle_share'] = val
                break

    # urban vehicle share: row where col 5 = "Urban" and col 9 mentions "pkm"
    for row in arr:
        if cell(row, 5) == 'Urban' and 'pkm' in cell(row, 9).lower():
            val = parse_pct(cell(row, 8))
            if val is not None:
                params['urban_vehicle_share'] = val
                break

    # LDV vehicle size fractions: col 3 = "Vehicle size" header; data rows have
    # col 5 = "Small"/"Large", col 8 = percentage value
    _in_vehicle_size = False
    for row in arr:
        c3, c5 = cell(row, 3), cell(row, 5)
        if c3 == 'Vehicle size':
            _in_vehicle_size = True
            continue
        if _in_vehicle_size and c3 and c3 != 'Vehicle size':
            _in_vehicle_size = False
        if _in_vehicle_size:
            if c5 == 'Small':
                val = parse_pct(cell(row, 8))
                if val is not None:
                    params['ldv_small_frac'] = val
            elif c5 == 'Large':
                val = parse_pct(cell(row, 8))
                if val is not None:
                    params['ldv_large_frac'] = val

    # Rapid transit share of electric transit: col 3 = "Shares of electricity in transit"
    # header; data rows have col 4 = province code, col 9 = Rapid transit %
    _in_transit_elec = False
    rapid_transit_share: dict[str, float] = {}
    for row in arr:
        c3, c4 = cell(row, 3), cell(row, 4)
        if c3 == 'Shares of electricity in transit':
            _in_transit_elec = True
            continue
        if _in_transit_elec and c3 and c3 != 'Shares of electricity in transit':
            _in_transit_elec = False
        if _in_transit_elec and c4 and c4.isupper() and len(c4) <= 3:
            val = parse_pct(cell(row, 9))
            if val is not None:
                rapid_transit_share[c4] = val
    if rapid_transit_share:
        params['rapid_transit_share'] = rapid_transit_share

    # LDV SOV/HOV pkm shares: col 2 = "pkm %", col 3 = "LDV", col 5 = "SOV"/"HOV"
    for row in arr:
        if cell(row, 2) == 'pkm %' and cell(row, 3) == 'LDV':
            if cell(row, 5) == 'SOV':
                val = parse_pct(cell(row, 8))
                if val is not None:
                    params['ldv_sov_kpkm_share'] = val
            elif cell(row, 5) == 'HOV':
                val = parse_pct(cell(row, 8))
                if val is not None:
                    params['ldv_hov_kpkm_share'] = val

    # aviation domestic pkm share (national CAN only) -- row: col1='23-10-0220',
    # col3='Domestic', col4 empty, col5='Canada', col8=value (e.g. '27%').
    # This is the fraction of total aviation pkm that is domestic.
    for row in arr:
        if (cell(row, 3) == 'Domestic' and not cell(row, 4)
                and 'Canada' in cell(row, 5)):
            val = parse_pct(cell(row, 8))
            if val is not None:
                params['aviation_pkm_share_domestic'] = val
                break

    # aviation domestic energy shares: "Domestic" in col 3 opens section,
    # "International" closes it; province codes (-?3 uppercase) in col 4,
    # plus CAN (col3='Domestic', col4='CAN') as the national header row.
    aviation_domestic: dict[str, float] = {}
    in_domestic = False
    for row in arr:
        c3, c4 = cell(row, 3), cell(row, 4)
        if c3 == 'Domestic':
            in_domestic = True
        elif c3 == 'International':
            in_domestic = False
        if in_domestic and c4 and c4.isupper() and len(c4) <= 3:
            val = parse_pct(cell(row, 8))
            if val is not None:
                aviation_domestic[c4] = val
    if aviation_domestic:
        params['aviation_domestic_share'] = aviation_domestic

    # Compute domestic-calibrated air intensity from national totals + domestic shares.
    # Table 20 reports total (domestic+international) energy and pkm.  Domestic
    # short-haul flights consume more fuel per pkm than long-haul international flights,
    # so using total intensity overstates domestic pkm.  Replicate old code's approach:
    #   intensity_domestic = (total_energy - energy_dom_share) / (total_pkm - pkm_dom_share) - 1e3
    # where -1e3 converts PJ/Mpkm to MJ/pkm (1 PJ/Mpkm = 1e9 MJ / 1e6 pkm = 1e3 MJ/pkm).
    total_energy = params.pop('_air_total_energy_pj',  pd.Series(dtype=float))
    total_pkm    = params.pop('_air_total_pkm_mpkm',   pd.Series(dtype=float))
    if not total_energy.empty and not total_pkm.empty:
        e_share = aviation_domestic.get('CAN', 1.0)
        p_share = params.get('aviation_pkm_share_domestic', 1.0)
        if p_share > 0:
            domestic_intensity = (
                total_energy * e_share
                / total_pkm.reindex(total_energy.index).replace(0, np.nan)
                / p_share
                * 1e3
            ).dropna()
            if not domestic_intensity.empty:
                params['air_intensity_mj_per_kpkm'] = domestic_intensity

    # --- CAGR projection assumptions (k*pkm) ---
    # Mode sections are identified by col2='k*pkm' and col3=mode name.
    # Province data rows have a 2-3 char uppercase code in col4.
    # Columns: 6=hist_cagr, 8=ref_multiplier, 9=ref_cagr_2023, 10=ref_cagr_2051.
    _KPKM_MODES = {
        'Cars', 'Light Trucks', 'Motorcycle', 'School Bus',
        'Transit', 'Intercity Bus', 'Rail', 'Aviation',
    }
    kpkm_assumptions: dict = {}
    _cur_mode: str | None = None
    for row in arr:
        c2, c3, c4 = cell(row, 2), cell(row, 3), cell(row, 4)
        if c2 == 'k*pkm' and c3 in _KPKM_MODES:
            _cur_mode = c3
            kpkm_assumptions.setdefault(_cur_mode, {})
            continue
        if _cur_mode and c4 and c4.isupper() and len(c4) <= 3:
            ref_mult = parse_pct(cell(row, 8))
            cagr_23  = parse_pct(cell(row, 9))
            cagr_51  = parse_pct(cell(row, 10))
            kpkm_assumptions[_cur_mode][c4] = {
                'reference_multiplier': ref_mult if ref_mult is not None else 0.0,
                'reference_cagr_2023':  cagr_23  if cagr_23  is not None else 0.0,
                'reference_cagr_2051':  cagr_51  if cagr_51  is not None else 0.0,
            }
    if kpkm_assumptions:
        params['kpkm_assumptions'] = kpkm_assumptions

    # Transit PB/RT benchmark-year splits by region.
    # Section identified by col2 == 'Transit PB/RT'; data rows have col3 in ('PB','RT')
    # and col4 == province code. Values in cols 7-12 correspond to benchmark years
    # 2000, 2005, 2010, 2015, 2020, 2023.
    _TRANSIT_BENCH_YEARS = [2000, 2005, 2010, 2015, 2020, 2023]
    _TRANSIT_BENCH_COLS  = [7, 8, 9, 10, 11, 12]
    transit_splits: dict = {}
    _in_transit_splits = False
    for row in arr:
        c2, c3, c4 = cell(row, 2), cell(row, 3), cell(row, 4)
        if c2 == 'Transit PB/RT':
            _in_transit_splits = True
            continue
        if _in_transit_splits and c2 and c2 != 'Transit PB/RT':
            _in_transit_splits = False
        if (_in_transit_splits and c3 in ('PB', 'RT')
                and c4 and c4.isupper() and len(c4) <= 3):
            vals: dict[int, float] = {}
            for yr, ci in zip(_TRANSIT_BENCH_YEARS, _TRANSIT_BENCH_COLS):
                raw = cell(row, ci)
                if raw:
                    try:
                        vals[yr] = float(raw)
                    except ValueError:
                        pass
            if vals:
                transit_splits.setdefault(c4, {})[c3] = vals
    if transit_splits:
        params['transit_splits_by_region'] = transit_splits

    return params


# ==============================================================================
# TABLE LOADING
# ==============================================================================


def load_tables(province_code: str) -> dict:
    """
    Load CEUD Excel sheets for a province.

    Returns
    -------
    dict mapping 'Table N' -- pl.DataFrame.
    """
    file_code = _PROV_FILE_CODE.get(province_code.upper(), province_code.lower())
    file_path = BASE_PATH / f"tran_{file_code}_e.xls"
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    table_numbers = [
        TABLE_PASSENGER_AIR_FUEL,
        TABLE_PASSENGER_RAIL,
        TABLE_CAR_FUEL,
        TABLE_CAR_EXPLANATORY,
        TABLE_PASSENGER_TRUCK_FUEL,
        TABLE_SCHOOL_BUS_FUEL,
        TABLE_URBAN_TRANSIT_FUEL,
        TABLE_INTERCITY_BUS_FUEL,
        TABLE_BUS_EXPLANATORY,
        TABLE_MOTORCYCLE,
        TABLE_TRUCK_EXPLANATORY,
    ]
    table_names = [f"Table {n}" for n in sorted(set(table_numbers))]

    def load_and_clean(sheet_name: str) -> pl.DataFrame:
        df = pl.read_excel(str(file_path), sheet_name=sheet_name, has_header=False)
        cast_exprs = [
            pl.col(c).str.strip_chars().cast(pl.Float64, strict=False)
            for c in df.columns[2:]
            if df[c].dtype in (pl.String, pl.Utf8)
        ]
        return df.with_columns(cast_exprs) if cast_exprs else df

    return {name: load_and_clean(name) for name in table_names}


# ==============================================================================
# EXTRACTION -- CAR PKM
# ==============================================================================


def extract_car_kpkm(province: str, tables: dict, params: dict) -> list[pl.DataFrame]:
    """
    Car passenger-km derived from provincial stock × avg_vkm × national occupancy.

    Matches old trans_ceud_old.py: kpkm = stock_thousands * avg_vkm * can_occupancy.
    Reading pkm directly from provincial Table 20 uses provincial occupancy rates
    (higher than national) and significantly overstates activity.

    Falls back to direct Table 20 pkm reading if national occupancy is unavailable.
    """
    car_occ = params.get('car_occupancy', pd.Series(dtype=float))
    if not car_occ.empty:
        t21     = tables[f"Table {TABLE_CAR_EXPLANATORY}"]
        stock   = row_to_series(t21, "Cars", match_n=1)
        avg_vkm = row_to_series(t21, "Cars", match_n=2)
        common  = sorted(
            set(stock.dropna().index) & set(avg_vkm.dropna().index) & set(car_occ.index)
        )
        if common:
            pkm = (
                stock.reindex(common) * avg_vkm.reindex(common)
                * car_occ.reindex(common)
            ).dropna()
            if not pkm.empty:
                return [_long(province, 'car_kpkm', '', 'output', 'kpkm', pkm)]

    t   = tables[f"Table {TABLE_CAR_FUEL}"]
    pkm = row_to_series(t, "Passenger-kilometres (millions)") * 1000
    return [_long(province, 'car_kpkm', '', 'output', 'kpkm', pkm)]


# ==============================================================================
# EXTRACTION -- PLT PKM
# ==============================================================================


def extract_lt_kpkm(province: str, tables: dict, params: dict) -> list[pl.DataFrame]:
    """
    Passenger light truck pkm derived from provincial stock × avg_vkm × national occupancy.

    Matches old trans_ceud_old.py approach — see extract_car_kpkm for rationale.
    Falls back to direct Table 25 pkm reading if national occupancy is unavailable.
    """
    lt_occ = params.get('lt_occupancy', pd.Series(dtype=float))
    if not lt_occ.empty:
        t37     = tables[f"Table {TABLE_TRUCK_EXPLANATORY}"]
        stock   = row_to_series(t37, "Passenger Light Trucks", match_n=2)
        avg_vkm = row_to_series(t37, "Passenger Light Trucks", match_n=4)
        common  = sorted(
            set(stock.dropna().index) & set(avg_vkm.dropna().index) & set(lt_occ.index)
        )
        if common:
            pkm = (
                stock.reindex(common) * avg_vkm.reindex(common)
                * lt_occ.reindex(common)
            ).dropna()
            if not pkm.empty:
                return [_long(province, 'lt_kpkm', '', 'output', 'kpkm', pkm)]

    t   = tables[f"Table {TABLE_PASSENGER_TRUCK_FUEL}"]
    pkm = row_to_series(t, "Passenger-kilometres (millions)") * 1000
    return [_long(province, 'lt_kpkm', '', 'output', 'kpkm', pkm)]


# ==============================================================================
# EXTRACTION -- BUS PKM
# ==============================================================================


def extract_bus_kpkm(province: str, tables: dict) -> list[pl.DataFrame]:
    """
    Bus passenger-km as three separate variables.

    Sources:
        Table 28 -- school_bus_kpkm
        Table 29 -- urban_transit_kpkm (2022 override: value = 2019 - 0.8)
        Table 30 -- intercity_bus_kpkm

    CEUD values are in millions; output is kpkm (- 1000).
    """
    configs = [
        (TABLE_SCHOOL_BUS_FUEL,    'school_bus_kpkm',    False),
        (TABLE_URBAN_TRANSIT_FUEL, 'urban_transit_kpkm', True),
        (TABLE_INTERCITY_BUS_FUEL, 'intercity_bus_kpkm', False),
    ]
    frames = []
    for tbl_num, var_name, is_urban_transit in configs:
        t   = tables[f"Table {tbl_num}"]
        pkm = row_to_series(t, "Passenger-kilometres (millions)")

        if is_urban_transit and 2022 in pkm.index and 2019 in pkm.index and pd.notna(pkm.get(2019)):
            pkm = pkm.copy()
            pkm[2022] = pkm[2019] * 0.8

        s = (pkm * 1000).dropna()
        if not s.empty:
            frames.append(_long(province, var_name, '', 'output', 'kpkm', s))
    return frames


def extract_bus_stock(province: str, tables: dict) -> list[pl.DataFrame]:
    """
    Bus stock (thousands of vehicles) from Table 31 for all three bus modes.

    Used in apply_extensions to compute avg k*pkm per bus (output per vehicle).
    match_n=0 selects the Stock row (first occurrence of each bus label).
    """
    t31 = tables.get(f"Table {TABLE_BUS_EXPLANATORY}")
    if t31 is None:
        return []
    frames = []
    for var_name, label in [
        ('school_bus_stock', 'School Buses'),
        ('transit_stock',    'Urban Transit'),
        ('icbus_stock',      'Inter-City Buses'),
    ]:
        try:
            s = row_to_series(t31, label, match_n=0).dropna()
            if not s.empty:
                frames.append(_long(province, var_name, '', 'output', 'kveh', s))
        except Exception:
            pass
    return frames


def extract_motorcycle_kpkm(province: str, tables: dict) -> list[pl.DataFrame]:
    """
    Motorcycle passenger-km from Table 32 "Passenger-kilometres (millions)".

    CEUD values are in millions; output is kpkm (- 1000).
    """
    t   = tables[f"Table {TABLE_MOTORCYCLE}"]
    pkm = (row_to_series(t, "Passenger-kilometres (millions)") * 1000).dropna()
    if pkm.empty:
        return []
    return [_long(province, 'motorcycle_kpkm', '', 'output', 'kpkm', pkm)]


# ==============================================================================
# EXTRACTION -- VEHICLE-KM (k*vkm)
# ==============================================================================


def extract_kvkm(province: str, tables: dict) -> list[pl.DataFrame]:
    """
    Extract average annual distance per vehicle from CEUD explanatory tables.

    Sources:
        Table 21  -- Car avg vkm (match_n=2)
        Table 37  -- Passenger light truck avg vkm (match_n=4)

    match_n indices (label "Cars" in Table 21):       0=Sales, 1=Stock, 2=AvgVkm
    match_n indices ("Passenger Light Trucks" in T37): 0=Sales, 1=Sales%, 2=Stock, 3=Stock%, 4=AvgVkm
    """
    frames = []

    t21 = tables.get(f"Table {TABLE_CAR_EXPLANATORY}")
    if t21 is not None:
        try:
            avg_clean = row_to_series(t21, "Cars", match_n=2).dropna()
            if not avg_clean.empty:
                frames.append(_long(province, 'car_avg_vkm', '', 'output', 'vkm', avg_clean))
        except Exception:
            pass

    t37 = tables.get(f"Table {TABLE_TRUCK_EXPLANATORY}")
    if t37 is not None:
        try:
            lt_avg_clean = row_to_series(t37, "Passenger Light Trucks", match_n=4).dropna()
            if not lt_avg_clean.empty:
                frames.append(_long(province, 'lt_avg_vkm', '', 'output', 'vkm', lt_avg_clean))
        except Exception:
            pass

    return frames


# ==============================================================================
# EXTRACTION -- RAIL PKM
# ==============================================================================


def extract_rail_kpkm(province: str, tables: dict, params: dict) -> list[pl.DataFrame]:
    """
    Passenger rail pkm inferred from provincial energy and the national CAN intensity.

    Formula: kpkm = energy_PJ - 1e6 / intensity_MJ_per_kpkm
    Unit check: PJ - 10^6 GJ/PJ / (MJ/pkm) = PJ - 10^9 pkm ÷ 10^3 = kpkm --

    Source: Table 17 "Passenger Rail Transportation Energy Use1 (PJ)"
    """
    intensity = params.get('rail_intensity_mj_per_kpkm', pd.Series(dtype=float))
    if intensity.empty:
        return []

    t      = tables[f"Table {TABLE_PASSENGER_RAIL}"]
    energy = row_to_series(t, "Passenger Rail Transportation Energy Use1 (PJ)")
    if energy.dropna().empty:
        return []

    common = sorted(set(energy.index) & set(intensity.index))
    if not common:
        return []

    pkm = (energy.reindex(common) * 1e6 / intensity.reindex(common)).dropna()
    return [_long(province, 'rail_kpkm', 'intercity', 'output', 'kpkm', pkm)]


# ==============================================================================
# EXTRACTION -- AIR PKM
# ==============================================================================


def extract_air_kpkm(province: str, tables: dict, params: dict) -> list[pl.DataFrame]:
    """
    Passenger air pkm inferred from domestic provincial energy and the national CAN intensity.

    Formula: kpkm = energy_PJ - domestic_share - 1e6 / intensity_MJ_per_kpkm

    The domestic share (from transport_passenger_assumptions.csv) isolates domestic
    aviation energy from total provincial energy, matching the old trans_ceud_old.py
    approach which used 'energy_share_domestic' per province.

    Sources: Table 14 (Aviation Gasoline + Aviation Turbo Fuel, match_n=0)
    """
    intensity = params.get('air_intensity_mj_per_kpkm', pd.Series(dtype=float))
    if intensity.empty:
        return []

    # Domestic energy share: assumptions CSV has a single national value keyed 'CAN'.
    # Per-province overrides are used if present; otherwise fall through to the CAN value.
    dom_shares: dict = params.get('aviation_domestic_share', {})
    dom_share = dom_shares.get(province, dom_shares.get('CAN', 1.0))

    t     = tables[f"Table {TABLE_PASSENGER_AIR_FUEL}"]
    avgas = row_to_series(t, "Aviation Gasoline",   match_n=0)
    avtur = row_to_series(t, "Aviation Turbo Fuel", match_n=0)

    all_years = sorted(set(avgas.index) | set(avtur.index))
    energy = (avgas.reindex(all_years).fillna(0) + avtur.reindex(all_years).fillna(0)).replace(0, np.nan)
    if energy.dropna().empty:
        return []

    common = sorted(set(energy.index) & set(intensity.index))
    if not common:
        return []

    # Apply domestic share to isolate domestic-only aviation energy
    energy_domestic = energy.reindex(common) * dom_share
    pkm = (energy_domestic * 1e6 / intensity.reindex(common)).dropna()
    return [_long(province, 'air_kpkm', '', 'output', 'kpkm', pkm)]


# ==============================================================================
# EXTRACTION -- TRANSIT FUEL SHARES (for market share computation)
# ==============================================================================


def extract_transit_fuel_shares(tables: dict) -> dict[str, pd.Series]:
    """
    Extract urban transit fuel energy series (PJ) from Table 29.

    Returns dict with keys 'diesel', 'ng', 'elec' as pd.Series indexed by year.
    Values are raw energy in PJ; caller computes shares.
    """
    t = tables.get(f"Table {TABLE_URBAN_TRANSIT_FUEL}")
    if t is None:
        return {}
    result = {}
    for key, candidates in _TRANSIT_FUEL_LABELS.items():
        s = _try_fuel_row(t, candidates)
        result[key] = s
    return result


def extract_intercity_bus_fuel_shares(tables: dict) -> dict[str, pd.Series]:
    """
    Extract intercity bus fuel energy series (PJ) from Table 30.

    Returns dict with keys 'diesel', 'gasoline' as pd.Series indexed by year.
    """
    t = tables.get(f"Table {TABLE_INTERCITY_BUS_FUEL}")
    if t is None:
        return {}
    result = {}
    for key, candidates in _INTERCITY_BUS_FUEL_LABELS.items():
        s = _try_fuel_row(t, candidates)
        result[key] = s
    return result


# ==============================================================================
# PROJECTION + DERIVED COMPUTATION
# ==============================================================================


def apply_extensions(
    df: pl.DataFrame,
    province: str,
    params: dict,
    transit_fuel: dict | None = None,
    intercity_bus_fuel: dict | None = None,
) -> pl.DataFrame:
    """
    Project all mode k*pkm series to 2100 and compute all derived outputs:

    - Extended base mode series (historical CEUD + CAGR projection to 2100)
    - walk/cycle, total, mode shares, urban/intercity-land/air k*pkm (all to 2100)
    - Urban mode tech shares (to LAST_HIST_YEAR)
    - Intercity land tech shares (to LAST_HIST_YEAR)
    - Vehicle size shares (to LAST_HIST_YEAR)
    - Motor tech shares (to LAST_HIST_YEAR)
    - Transit decomposition: PB/RT ratio, bus fuel shares, ferry (to LAST_HIST_YEAR)
    - Intercity bus fuel shares (to LAST_HIST_YEAR)
    - Rail intercity tech shares (to LAST_HIST_YEAR)

    Parameters
    ----------
    df          : historical base mode k*pkm (from extract_* functions)
    province    : province code (e.g. 'AB', 'BC')
    params      : from load_projection_params()
    transit_fuel: output of extract_transit_fuel_shares()
    intercity_bus_fuel: output of extract_intercity_bus_fuel_shares()
    """
    if transit_fuel is None:
        transit_fuel = {}
    if intercity_bus_fuel is None:
        intercity_bus_fuel = {}

    reg = province
    all_years  = list(range(2000, PROJ_HORIZON + 1))
    hist_years = list(range(2000, LAST_HIST_YEAR + 1))

    walk_cycle_share    = params.get('walk_cycle_share', 0.24 / 26.94)
    urban_vehicle_share = params.get('urban_vehicle_share', 0.55)
    sov_pct = params.get('ldv_sov_kpkm_share', 0.516)
    hov_pct = params.get('ldv_hov_kpkm_share', 0.484)

    # ------------------------------------------------------------------
    # Step 1: Build full (2000--2100) series for each base mode via CAGR
    # ------------------------------------------------------------------
    _modes_meta = {
        'car_kpkm':            ('', 'Cars'),
        'lt_kpkm':             ('', 'Light Trucks'),
        'motorcycle_kpkm':     ('', 'Motorcycle'),
        'school_bus_kpkm':     ('', 'School Bus'),
        'urban_transit_kpkm':  ('', 'Transit'),
        'intercity_bus_kpkm':  ('', 'Intercity Bus'),
        'rail_kpkm':           ('intercity', 'Rail'),
        'air_kpkm':            ('', 'Aviation'),
    }

    full: dict[str, pd.Series] = {}

    for var, (cat, mode_name) in _modes_meta.items():
        hist = _series_from_df(df, var, cat)
        cfg  = params.get('kpkm_assumptions', {}).get(mode_name, {}).get(reg, {})
        ref_mult   = cfg.get('reference_multiplier', 0.0)
        cagr_2023  = cfg.get('reference_cagr_2023',  0.0)
        cagr_2051  = cfg.get('reference_cagr_2051',  0.0)

        s = pd.Series(np.nan, index=all_years, dtype=float)

        # copy historical values
        for y in hist.index:
            if y in s.index:
                s[y] = hist[y]

        # 2022 COVID adjustment (override with 2019 - ref_mult as projection base)
        # Only override if we don't have a more recent actual value past 2022
        base_2019 = hist.get(2019) if 2019 in hist.index else np.nan
        if (pd.notna(base_2019) and ref_mult > 0
                and LAST_HIST_YEAR <= 2022 and 2022 in s.index):
            s[2022] = base_2019 * ref_mult

        # Determine projection start: first year after LAST_HIST_YEAR
        proj_start = LAST_HIST_YEAR + 1

        for y in range(proj_start, 2051):
            prev = s.get(y - 1, np.nan)
            s[y] = prev * (1.0 + cagr_2023) if pd.notna(prev) else np.nan

        for y in range(2051, PROJ_HORIZON + 1):
            prev = s.get(y - 1, np.nan)
            s[y] = prev * (1.0 + cagr_2051) if pd.notna(prev) else np.nan

        full[var] = s

    # ------------------------------------------------------------------
    # Step 1b: Project average annual distance per vehicle (car and LT)
    # ------------------------------------------------------------------
    # Car average annual distance (vkm/vehicle) → projected, then divided by 100 for output
    _car_avg_hist = _series_from_df(df, 'car_avg_vkm', '')
    _cfg_cars = params.get('kpkm_assumptions', {}).get('Cars', {}).get(reg, {})
    _cagr_23  = _cfg_cars.get('reference_cagr_2023', 0.0)
    _cagr_51  = _cfg_cars.get('reference_cagr_2051', 0.0)
    full_car_avg = pd.Series(np.nan, index=all_years, dtype=float)
    if not _car_avg_hist.empty:
        for y in _car_avg_hist.index:
            if y in full_car_avg.index:
                full_car_avg[y] = _car_avg_hist[y]
        for y in range(proj_start, 2051):
            prev = full_car_avg.get(y - 1, np.nan)
            full_car_avg[y] = prev * (1.0 + _cagr_23) if pd.notna(prev) else np.nan
        for y in range(2051, PROJ_HORIZON + 1):
            prev = full_car_avg.get(y - 1, np.nan)
            full_car_avg[y] = prev * (1.0 + _cagr_51) if pd.notna(prev) else np.nan

    # Bus avg k*pkm per vehicle: historical = total_k*pkm / stock; projected using mode CAGR.
    # Transit stock (Table 31) covers all transit vehicles; output = total transit kpkm / stock.
    def _proj_avg(hist: pd.Series, cagr_23: float, cagr_51: float) -> pd.Series:
        s = pd.Series(np.nan, index=all_years, dtype=float)
        for y, v in hist.items():
            if y in s.index:
                s[y] = v
        for y in range(proj_start, 2051):
            prev = s.get(y - 1, np.nan)
            s[y] = prev * (1.0 + cagr_23) if pd.notna(prev) else np.nan
        for y in range(2051, PROJ_HORIZON + 1):
            prev = s.get(y - 1, np.nan)
            s[y] = prev * (1.0 + cagr_51) if pd.notna(prev) else np.nan
        return s

    def _bus_avg(kpkm_var: str, kpkm_cat: str, stock_var: str, mode_key: str) -> pd.Series:
        kpkm_hist  = _series_from_df(df, kpkm_var, kpkm_cat)
        stock_hist = _series_from_df(df, stock_var, '')
        common = sorted(set(kpkm_hist.dropna().index) & set(stock_hist.dropna().index))
        if not common:
            return pd.Series(dtype=float)
        # kpkm / stock_thousands = pkm/bus; divide by 1000 to get k*pkm/bus
        avg = (kpkm_hist.reindex(common) / stock_hist.reindex(common) / 1000.0).replace(0, np.nan).dropna()
        cfg = params.get('kpkm_assumptions', {}).get(mode_key, {}).get(reg, {})
        return _proj_avg(avg, cfg.get('reference_cagr_2023', 0.0), cfg.get('reference_cagr_2051', 0.0))

    # Transit avg k*pkm per bus: urban transit only (avg_vkm × occupancy / 1000).
    full_transit_avg_kpkm = _bus_avg('urban_transit_kpkm', '', 'transit_stock', 'Transit')

    full_icbus_avg_kpkm   = _bus_avg('intercity_bus_kpkm', '', 'icbus_stock',    'Intercity Bus')

    # Light truck average annual distance (vkm/vehicle) → projected using Cars CAGR.
    # LT k*pkm CAGR is positive (fleet growth) so applying it to per-vehicle avg_vkm
    # would incorrectly cause each truck to travel more per year.  Cars CAGR captures
    # the correct per-vehicle intensity trend (slightly declining).
    _lt_avg_hist = _series_from_df(df, 'lt_avg_vkm', '')
    full_lt_avg = pd.Series(np.nan, index=all_years, dtype=float)
    if not _lt_avg_hist.empty:
        for y in _lt_avg_hist.index:
            if y in full_lt_avg.index:
                full_lt_avg[y] = _lt_avg_hist[y]
        for y in range(proj_start, 2051):
            prev = full_lt_avg.get(y - 1, np.nan)
            full_lt_avg[y] = prev * (1.0 + _cagr_23) if pd.notna(prev) else np.nan
        for y in range(2051, PROJ_HORIZON + 1):
            prev = full_lt_avg.get(y - 1, np.nan)
            full_lt_avg[y] = prev * (1.0 + _cagr_51) if pd.notna(prev) else np.nan

    # ------------------------------------------------------------------
    # Step 2: Compute walk/cycle and total for all years
    # ------------------------------------------------------------------
    car_s     = full['car_kpkm']
    lt_s      = full['lt_kpkm']
    school_s  = full['school_bus_kpkm']
    transit_s = full['urban_transit_kpkm']
    icbus_s  = full['intercity_bus_kpkm']
    moto_s   = full['motorcycle_kpkm']
    rail_s   = full['rail_kpkm']
    air_s    = full['air_kpkm']

    walk_cycle_s = pd.Series(np.nan, index=all_years, dtype=float)
    if (walk_cycle_share and urban_vehicle_share
            and 0 < walk_cycle_share < 1):
        common_wc = sorted(
            set(car_s.dropna().index) & set(lt_s.dropna().index)
            & set(school_s.dropna().index) & set(transit_s.dropna().index)
        )
        if common_wc:
            motorised_urban = (
                (car_s.reindex(common_wc) + lt_s.reindex(common_wc)) * urban_vehicle_share
                + school_s.reindex(common_wc)
                + transit_s.reindex(common_wc)
            )
            wc = (motorised_urban / (1 - walk_cycle_share) * walk_cycle_share).dropna()
            for y, v in wc.items():
                walk_cycle_s[y] = v

    full['walk_cycle_urban_kpkm'] = walk_cycle_s

    all_mode_series = [car_s, lt_s, school_s, transit_s, icbus_s,
                       moto_s, rail_s, air_s, walk_cycle_s]
    available = [s for s in all_mode_series if s.notna().any()]

    total_s = pd.Series(0.0, index=all_years, dtype=float)
    for s in available:
        total_s = total_s.add(s.reindex(all_years).fillna(0), fill_value=0)
    total_s = total_s.replace(0, np.nan)
    full['total_passenger_kpkm'] = total_s

    # ------------------------------------------------------------------
    # Step 3: Mode shares and mode-level k*pkm
    # ------------------------------------------------------------------
    # Intercity land includes bus + rail + LDV intercity portion.
    # Motorcycle is treated as entirely urban (consistent with Step 5 pv_il formula).
    # Urban is the residual, so urban + intercity_land + intercity_air = 1.
    _ldv_all = (car_s.reindex(all_years).fillna(0) + lt_s.reindex(all_years).fillna(0))
    air_num  = air_s.reindex(all_years).fillna(0)
    land_num = (icbus_s.reindex(all_years).fillna(0)
                + rail_s.reindex(all_years).fillna(0)
                + _ldv_all * (1 - urban_vehicle_share))
    total_nn = total_s.replace(0, np.nan)

    intercity_air_share  = (air_num  / total_nn).dropna()
    intercity_land_share = (land_num / total_nn).dropna()

    common_modes = sorted(set(intercity_air_share.index) & set(intercity_land_share.index))
    if common_modes:
        urban_share = (1 - intercity_air_share.reindex(common_modes)
                       - intercity_land_share.reindex(common_modes)).dropna()
    else:
        urban_share = pd.Series(dtype=float)

    # ------------------------------------------------------------------
    # Step 4: Urban tech market shares (historical only, -? LAST_HIST_YEAR)
    # ------------------------------------------------------------------
    urban_hist = urban_share.reindex(hist_years).dropna()
    total_hist = total_s.reindex(hist_years).dropna()

    urban_total_hist = (urban_hist * total_hist.reindex(urban_hist.index)).dropna()

    ldv_hist = (car_s.reindex(urban_hist.index).fillna(0)
                + lt_s.reindex(urban_hist.index).fillna(0))
    moto_hist = moto_s.reindex(urban_hist.index).fillna(0)
    wc_hist   = walk_cycle_s.reindex(urban_hist.index).fillna(0)

    sov_share_hist = ((ldv_hist * sov_pct * urban_vehicle_share + moto_hist)
                      / urban_total_hist.replace(0, np.nan)).dropna()
    hov_share_hist = (ldv_hist * hov_pct * urban_vehicle_share
                      / urban_total_hist.replace(0, np.nan)).dropna()
    wc_share_hist  = (wc_hist / urban_total_hist.replace(0, np.nan)).dropna()

    transit_yrs_u = sorted(
        set(sov_share_hist.index) & set(hov_share_hist.index) & set(wc_share_hist.index)
    )
    transit_share_hist = (
        1 - sov_share_hist.reindex(transit_yrs_u)
          - hov_share_hist.reindex(transit_yrs_u)
          - wc_share_hist.reindex(transit_yrs_u)
    ).dropna() if transit_yrs_u else pd.Series(dtype=float)

    # Intercity land tech shares (historical)
    intercity_land_hist = intercity_land_share.reindex(hist_years).dropna()
    total_il_hist = (intercity_land_hist * total_hist.reindex(intercity_land_hist.index)).dropna()

    icbus_il = icbus_s.reindex(intercity_land_hist.index).fillna(0)
    rail_il  = rail_s.reindex(intercity_land_hist.index).fillna(0)
    pv_il    = (car_s.reindex(intercity_land_hist.index).fillna(0)
                + lt_s.reindex(intercity_land_hist.index).fillna(0)
                + moto_s.reindex(intercity_land_hist.index).fillna(0)
                - ldv_hist.reindex(intercity_land_hist.index).fillna(0) * urban_vehicle_share
                - moto_hist.reindex(intercity_land_hist.index).fillna(0)
                ).clip(lower=0)

    denom_il = (icbus_il + rail_il + pv_il).replace(0, np.nan)
    bus_intercity_share = (icbus_il / denom_il).dropna()
    rail_intercity_share = (rail_il / denom_il).dropna()
    pv_intercity_share   = (pv_il  / denom_il).dropna()

    # ------------------------------------------------------------------
    # Step 5: Vehicle size shares (historical, from car/lt k*pkm ratio)
    # ------------------------------------------------------------------
    car_hist_y  = car_s.reindex(hist_years).dropna()
    lt_hist_y   = lt_s.reindex(hist_years).dropna()
    common_ldv  = sorted(set(car_hist_y.index) & set(lt_hist_y.index))

    if common_ldv:
        ldv_total  = (car_hist_y.reindex(common_ldv) + lt_hist_y.reindex(common_ldv)).replace(0, np.nan)
        car_frac   = (car_hist_y.reindex(common_ldv) / ldv_total).dropna()
        lt_frac    = (lt_hist_y.reindex(common_ldv)  / ldv_total).dropna()
        ldv_small = params.get('ldv_small_frac', LDV_SMALL_FRAC)
        ldv_large = params.get('ldv_large_frac', LDV_LARGE_FRAC)
        car_small  = (car_frac * ldv_small).dropna()
        car_large  = (car_frac * ldv_large).dropna()
        lt_small   = (lt_frac  * ldv_small).dropna()
        lt_large   = (lt_frac  * ldv_large).dropna()
    else:
        car_small = car_large = lt_small = lt_large = pd.Series(dtype=float)

    # ------------------------------------------------------------------
    # Step 7: Transit decomposition (fuel shares + rapid transit split)
    # ------------------------------------------------------------------
    transit_hist_kpkm = transit_s.reindex(hist_years).dropna()

    def _fuel_fracs(fuel_dict: dict[str, pd.Series], keys: list[str]) -> dict[str, pd.Series]:
        """Compute fuel energy fractions from raw energy series (handle zeros)."""
        energies = {k: fuel_dict.get(k, pd.Series(dtype=float)).reindex(transit_hist_kpkm.index).fillna(0)
                    for k in keys}
        total_e  = sum(energies.values()).replace(0, np.nan)
        fracs    = {k: (energies[k] / total_e).fillna(0) for k in keys}
        return fracs

    transit_fracs = _fuel_fracs(transit_fuel, ['diesel', 'ng', 'elec'])
    transit_diesel_frac = transit_fracs.get('diesel', pd.Series(0.0, index=transit_hist_kpkm.index))
    transit_ng_frac     = transit_fracs.get('ng',     pd.Series(0.0, index=transit_hist_kpkm.index))
    transit_elec_frac   = transit_fracs.get('elec',   pd.Series(0.0, index=transit_hist_kpkm.index))

    # Fallback: if no fuel data, assume all diesel
    if transit_diesel_frac.sum() == 0 and transit_ng_frac.sum() == 0 and transit_elec_frac.sum() == 0:
        transit_diesel_frac = pd.Series(1.0, index=transit_hist_kpkm.index)

    rapid_share = params.get('rapid_transit_share', RAPID_TRANSIT_SHARE).get(reg, 1.0)

    # Ferry Urban: BC only (hard-coded anchors); others = 0
    ferry_kpkm = pd.Series(0.0, index=transit_hist_kpkm.index)
    if reg == 'BC':
        ferry_anchors = _compute_ferry_urban_bc(hist_years)
        ferry_kpkm = ferry_anchors.reindex(transit_hist_kpkm.index).fillna(0)

    elec_kpkm   = transit_hist_kpkm * transit_elec_frac
    rapid_kpkm  = (elec_kpkm * rapid_share).dropna()
    bus_elec_kpkm_raw = (elec_kpkm * (1 - rapid_share)).dropna()

    bus_diesel_kpkm  = (transit_hist_kpkm * transit_diesel_frac).dropna()
    bus_ng_kpkm      = (transit_hist_kpkm * transit_ng_frac).dropna()

    # Public Bus total = bus_diesel + bus_ng + bus_elec + ferry
    pub_bus_kpkm = (bus_diesel_kpkm.reindex(transit_hist_kpkm.index).fillna(0)
                    + bus_ng_kpkm.reindex(transit_hist_kpkm.index).fillna(0)
                    + bus_elec_kpkm_raw.reindex(transit_hist_kpkm.index).fillna(0)
                    + ferry_kpkm)

    pub_bus_total = pub_bus_kpkm.replace(0, np.nan)
    bus_diesel_share  = (bus_diesel_kpkm  / pub_bus_total).dropna()
    bus_ng_share      = (bus_ng_kpkm      / pub_bus_total).dropna()
    bus_elec_share    = (bus_elec_kpkm_raw / pub_bus_total).dropna()
    ferry_share_pub   = (ferry_kpkm        / pub_bus_total).dropna()

    # Public Bus / Rapid Transit fractions of total transit k*pkm.
    # Benchmark provinces: use interpolated splits from assumptions CSV.
    # Other provinces: derive from fuel-based kpkm.
    denom_transit = transit_hist_kpkm.replace(0, np.nan)
    _transit_splits = params.get('transit_splits_by_region', TRANSIT_SPLITS_BY_REGION)
    splits = _transit_splits.get(reg, {})
    if splits:
        pb_splits_interp = _interpolate_splits(splits.get('PB', {}), hist_years)
        rt_splits_interp = _interpolate_splits(splits.get('RT', {}), hist_years)
        common_split = sorted(set(transit_hist_kpkm.index) & set(pb_splits_interp.index))
        if common_split:
            pb_frac = pb_splits_interp.reindex(common_split).dropna()
            rt_frac = rt_splits_interp.reindex(common_split).dropna()
        else:
            pb_frac = (pub_bus_kpkm / denom_transit).dropna()
            rt_frac = (rapid_kpkm.reindex(denom_transit.index) / denom_transit).dropna()
    else:
        pb_frac = (pub_bus_kpkm / denom_transit).dropna()
        rt_frac = (rapid_kpkm.reindex(denom_transit.index) / denom_transit).dropna()

    # Last historical pb fraction — used to extend Transit.Public Bus k*pkm into projection
    last_pb_frac = pb_frac.dropna().iloc[-1] if not pb_frac.dropna().empty else np.nan

    # ------------------------------------------------------------------
    # Step 8: Intercity bus fuel shares (historical)
    # ------------------------------------------------------------------
    icbus_hist_kpkm = icbus_s.reindex(hist_years).dropna()
    ib_fracs = _fuel_fracs(
        {k: intercity_bus_fuel.get(k, pd.Series(dtype=float))
         for k in ['diesel', 'gasoline']},
        ['diesel', 'gasoline']
    )
    ib_diesel_frac   = ib_fracs.get('diesel',   pd.Series(0.0, index=icbus_hist_kpkm.index))
    ib_gasoline_frac = ib_fracs.get('gasoline', pd.Series(0.0, index=icbus_hist_kpkm.index))

    if ib_diesel_frac.reindex(icbus_hist_kpkm.index).fillna(0).sum() == 0 and \
       ib_gasoline_frac.reindex(icbus_hist_kpkm.index).fillna(0).sum() == 0:
        ib_diesel_frac = pd.Series(1.0, index=icbus_hist_kpkm.index)
        ib_gasoline_frac = pd.Series(0.0, index=icbus_hist_kpkm.index)

    ib_total = (ib_diesel_frac.reindex(icbus_hist_kpkm.index).fillna(0)
                + ib_gasoline_frac.reindex(icbus_hist_kpkm.index).fillna(0)).replace(0, np.nan)
    ib_diesel_share   = (ib_diesel_frac.reindex(icbus_hist_kpkm.index).fillna(0) / ib_total).dropna()
    ib_gasoline_share = (ib_gasoline_frac.reindex(icbus_hist_kpkm.index).fillna(0) / ib_total).dropna()

    # ------------------------------------------------------------------
    # Step 9: Assemble output long-format DataFrame
    # ------------------------------------------------------------------
    frames: list[pl.DataFrame] = []
    ldv_small = params.get('ldv_small_frac', LDV_SMALL_FRAC)
    ldv_large = params.get('ldv_large_frac', LDV_LARGE_FRAC)

    # --- total_kpkm (service request, all years) ---
    if total_s.notna().any():
        frames.append(_long(province, 'total_kpkm', '', 'service_request', 'k*pkm', total_s.dropna()))

    # --- Mode-level shares (service request, % of k*pkm, all years) ---
    for var, s in [
        ('Mode.Urban',          urban_share),
        ('Mode.Intercity Land', intercity_land_share),
        ('Mode.Intercity Air',  intercity_air_share),
    ]:
        if not s.empty:
            frames.append(_long(province, var, '', 'service_request', '% of k*pkm', s))

    # Helper: extend a historical share series to 2100 by holding last value flat.
    proj_years = list(range(proj_start, PROJ_HORIZON + 1))

    def _extend_flat(s: pd.Series) -> pd.Series:
        if s.dropna().empty:
            return s
        last = float(s.dropna().iloc[-1])
        return pd.concat([s, pd.Series(last, index=proj_years)]).sort_index()

    # --- Urban tech shares (Mode.Urban, extended to 2100) ---
    for cat, s in [
        ('Walk Cycle',            wc_share_hist),
        ('Passenger Vehicle SOV', sov_share_hist),
        ('Passenger Vehicle HOV', hov_share_hist),
        ('Public Transit',        transit_share_hist),
    ]:
        s_ext = _extend_flat(s)
        if not s_ext.empty:
            frames.append(_long(province, 'Mode.Urban', cat,
                                'market_share_total', '% of Urban k*pkm', s_ext))

    # --- Intercity land tech shares (Mode.Intercity Land, extended to 2100) ---
    for cat, s in [
        ('Bus Intercity',     bus_intercity_share),
        ('Rail Intercity',    rail_intercity_share),
        ('Passenger Vehicle', pv_intercity_share),
    ]:
        s_ext = _extend_flat(s)
        if not s_ext.empty:
            frames.append(_long(province, 'Mode.Intercity Land', cat,
                                'market_share_total', '% of Intercity Land k*pkm', s_ext))

    # --- Air intercity tech shares (Mode.Intercity Air, extended to 2100) ---
    for cat, default_val in AIR_TECH_DEFAULTS.items():
        frames.append(_long(province, 'Mode.Intercity Air', cat,
                            'market_share_total', '% of Intercity Air k*pkm',
                            pd.Series(default_val, index=all_years)))

    # --- Passenger Vehicles size shares (historical) ---
    for cat, s in [
        ('Car_small',         car_small),
        ('Car_large',         car_large),
        ('Light Truck_small', lt_small),
        ('Light Truck_large', lt_large),
    ]:
        if not s.empty:
            frames.append(_long(province, 'Passenger Vehicles', cat,
                                'market_share_total', '% of Passenger Vehicle k*pkm', s))

    # --- Passenger Vehicles avg annual distance per vehicle (k*vkm, all years) ---
    # Both small and large share the same per-vehicle average distance.
    # Output = avg_vkm / 1000 (thousands of km/vehicle = k*vkm per vehicle).
    car_avg_kvkm = (full_car_avg / 1000.0).dropna()
    lt_avg_kvkm  = (full_lt_avg  / 1000.0).dropna()
    for cat, avg_s in [
        ('Car_small',         car_avg_kvkm),
        ('Car_large',         car_avg_kvkm),
        ('Light Truck_small', lt_avg_kvkm),
        ('Light Truck_large', lt_avg_kvkm),
    ]:
        if not avg_s.empty:
            frames.append(_long(province, 'Passenger Vehicles', cat, 'output', 'k*vkm', avg_s))

    # --- Passenger Vehicle Motors tech shares (extended to 2100) ---
    for cat, default_val in MOTOR_TECH_DEFAULTS.items():
        frames.append(_long(province, 'Passenger Vehicle Motors', cat,
                            'market_share_total', '% of motors',
                            pd.Series(default_val, index=all_years)))

    # --- Passenger Vehicle Motors output: LDV fleet-weighted avg annual distance / 100 ---
    # Formula: (total_kvkm) / (total_stock_k) / 100
    # where total_kvkm = car_kvkm + lt_kvkm  (car_kpkm/car_occ + lt_kpkm/lt_occ)
    # and   total_stock_k = car_stock_k + lt_stock_k  (car_kvkm/car_avg + lt_kvkm/lt_avg)
    _car_occ_s = params.get('car_occupancy', pd.Series(dtype=float))
    _lt_occ_s  = params.get('lt_occupancy',  pd.Series(dtype=float))
    if (not _car_occ_s.empty and not _lt_occ_s.empty
            and not full_car_avg.dropna().empty and not full_lt_avg.dropna().empty):
        _car_occ_ext = pd.Series(np.nan, index=all_years, dtype=float)
        _lt_occ_ext  = pd.Series(np.nan, index=all_years, dtype=float)
        for y, v in _car_occ_s.items():
            if y in _car_occ_ext.index:
                _car_occ_ext[y] = v
        for y, v in _lt_occ_s.items():
            if y in _lt_occ_ext.index:
                _lt_occ_ext[y] = v
        _car_occ_ext = _car_occ_ext.ffill()
        _lt_occ_ext  = _lt_occ_ext.ffill()
        _common_ldv = sorted(
            set(car_s.dropna().index) & set(lt_s.dropna().index)
            & set(full_car_avg.dropna().index) & set(full_lt_avg.dropna().index)
            & set(_car_occ_ext.dropna().index) & set(_lt_occ_ext.dropna().index)
        )
        if _common_ldv:
            _car_kvkm    = car_s.reindex(_common_ldv) / _car_occ_ext.reindex(_common_ldv)
            _lt_kvkm     = lt_s.reindex(_common_ldv)  / _lt_occ_ext.reindex(_common_ldv)
            _car_stock_k = _car_kvkm / full_car_avg.reindex(_common_ldv)
            _lt_stock_k  = _lt_kvkm  / full_lt_avg.reindex(_common_ldv)
            _ldv_motor   = ((_car_kvkm + _lt_kvkm) / (_car_stock_k + _lt_stock_k) / 100.0
                            ).replace(0, np.nan).dropna()
            if not _ldv_motor.empty:
                frames.append(_long(province, 'Passenger Vehicle Motors', '', 'output',
                                    '100 vkm (avg car eq)', _ldv_motor))

    # --- Passenger Vehicles: Vehicle Motor (average annual car distance, all years, /100) ---
    vehicle_motor = (full_car_avg / 100.0).dropna()
    if not vehicle_motor.empty:
        frames.append(_long(province, 'Passenger Vehicles', 'Vehicle Motor',
                            'output', '100 vkm (avg car eq)', vehicle_motor))

    # --- Transit PB / RT splits (service request, historical) ---
    if not pb_frac.dropna().empty:
        frames.append(_long(province, 'Transit', 'Public Bus',
                            'service_request', '% of Transit k*pkm', pb_frac.dropna()))
    if not rt_frac.dropna().empty:
        frames.append(_long(province, 'Transit', 'Rapid Transit',
                            'service_request', '% of Transit k*pkm', rt_frac.dropna()))

    # --- Transit.Public Bus fuel shares (historical) ---
    for cat, s in [
        ('Bus Urban Diesel',   bus_diesel_share),
        ('Bus Urban NG',       bus_ng_share),
        ('Bus Urban Electric', bus_elec_share),
        ('Ferry Urban',        ferry_share_pub),
    ]:
        if not s.empty:
            frames.append(_long(province, 'Transit.Public Bus', cat,
                                'market_share_total', '% of Public Bus k*pkm', s))

    # --- Transit.Public Bus avg k*pkm per bus (all years) ---
    # Combined school + urban transit fleet; this is what CIMS uses as the node unit.
    transit_avg_out = full_transit_avg_kpkm.dropna()
    if not transit_avg_out.empty:
        frames.append(_long(province, 'Transit.Public Bus', '', 'output', 'k*pkm per bus', transit_avg_out))

    # --- Intercity Bus fuel shares (CEUD Diesel/Gasoline + defaults, historical) ---
    for cat, s in [
        ('Bus Intercity Diesel',   ib_diesel_share),
        ('Bus Intercity Gasoline', ib_gasoline_share),
    ]:
        if not s.empty:
            frames.append(_long(province, 'Intercity Bus', cat,
                                'market_share_total', '% of Intercity Bus k*pkm', s))
    for cat, default_val in INTERCITY_BUS_FUEL_DEFAULTS.items():
        frames.append(_long(province, 'Intercity Bus', cat,
                            'market_share_total', '% of Intercity Bus k*pkm',
                            pd.Series(default_val, index=hist_years)))

    # --- Intercity Bus avg k*pkm per bus (all years) ---
    icbus_avg_out = full_icbus_avg_kpkm.dropna()
    if not icbus_avg_out.empty:
        frames.append(_long(province, 'Intercity Bus', '', 'output', 'k*pkm per bus', icbus_avg_out))

    # --- Intercity Rail tech shares (historical, fixed defaults) ---
    for cat, default_val in RAIL_TECH_DEFAULTS.items():
        frames.append(_long(province, 'Intercity Rail', cat,
                            'market_share_total', '% of Intercity Rail k*pkm',
                            pd.Series(default_val, index=hist_years)))

    if not frames:
        return df

    return pl.concat(frames, how='diagonal_relaxed').sort(['province', 'variable', 'category', 'year'])


# ==============================================================================
# MAIN EXTRACTION FUNCTION
# ==============================================================================


def extract_all_data(
    province_code: str,
    apply_projections: bool = True,
    projection_params: Optional[dict] = None,
) -> pl.DataFrame:
    """
    Extract all passenger pkm for a province into a single long-format DataFrame.

    Returns
    -------
    pl.DataFrame
        Columns: province, variable, category, parameter, unit, source, year, value.
    """
    province = province_code.upper()
    if province not in PROVINCES:
        raise ValueError(
            f"Invalid province code: {province_code!r}. "
            f"Valid codes: {list(PROVINCES.keys())}"
        )

    if projection_params is None:
        projection_params = load_projection_params()

    tables = load_tables(province)

    frames: list[pl.DataFrame] = []
    frames += extract_car_kpkm(province, tables, projection_params)
    frames += extract_lt_kpkm(province, tables, projection_params)
    frames += extract_bus_kpkm(province, tables)
    frames += extract_bus_stock(province, tables)
    frames += extract_motorcycle_kpkm(province, tables)
    frames += extract_kvkm(province, tables)
    frames += extract_rail_kpkm(province, tables, projection_params)
    frames += extract_air_kpkm(province, tables, projection_params)

    df = pl.concat(frames, how='diagonal_relaxed')

    if apply_projections:
        transit_fuel    = extract_transit_fuel_shares(tables)
        intercity_bus_f = extract_intercity_bus_fuel_shares(tables)
        df = apply_extensions(df, province, projection_params,
                              transit_fuel=transit_fuel,
                              intercity_bus_fuel=intercity_bus_f)
    else:
        derived = compute_derived_kpkm(df, province, projection_params)
        if derived:
            df = pl.concat([df] + derived, how='diagonal_relaxed')

    return df.sort(['province', 'variable', 'category', 'year'])


# ==============================================================================
# LEGACY DERIVED PKM (used when apply_projections=False)
# ==============================================================================


def compute_derived_kpkm(df: pl.DataFrame, province: str, params: dict) -> list[pl.DataFrame]:
    """
    Compute walk/cycle urban pkm and total passenger pkm from extracted modes.
    Used only when apply_projections=False (historical-only mode).
    """
    frames: list[pl.DataFrame] = []

    car_kpkm        = _series_from_df(df, 'car_kpkm')
    lt_kpkm         = _series_from_df(df, 'lt_kpkm')
    transit_kpkm    = _series_from_df(df, 'urban_transit_kpkm')
    intercity_kpkm  = _series_from_df(df, 'intercity_bus_kpkm')
    motorcycle_kpkm = _series_from_df(df, 'motorcycle_kpkm')
    rail_kpkm       = _series_from_df(df, 'rail_kpkm', 'intercity')
    air_kpkm        = _series_from_df(df, 'air_kpkm')

    walk_cycle_share    = params.get('walk_cycle_share')
    urban_vehicle_share = params.get('urban_vehicle_share')
    walk_cycle_kpkm = pd.Series(dtype=float)

    school_kpkm     = _series_from_df(df, 'school_bus_kpkm')
    if (walk_cycle_share and urban_vehicle_share
            and 0 < walk_cycle_share < 1
            and not car_kpkm.empty and not lt_kpkm.empty
            and not school_kpkm.empty and not transit_kpkm.empty):
        common = sorted(
            set(car_kpkm.index) & set(lt_kpkm.index)
            & set(school_kpkm.index) & set(transit_kpkm.index)
        )
        if common:
            motorised_urban = (
                (car_kpkm.reindex(common) + lt_kpkm.reindex(common)) * urban_vehicle_share
                + school_kpkm.reindex(common)
                + transit_kpkm.reindex(common)
            )
            walk_cycle_kpkm = (motorised_urban / (1 - walk_cycle_share) * walk_cycle_share).dropna()
            if not walk_cycle_kpkm.empty:
                frames.append(_long(province, 'walk_cycle_urban_kpkm', '',
                                    'output', 'kpkm', walk_cycle_kpkm))

    all_modes = [car_kpkm, lt_kpkm, school_kpkm, transit_kpkm,
                 intercity_kpkm, motorcycle_kpkm, rail_kpkm, air_kpkm, walk_cycle_kpkm]
    available = [s for s in all_modes if not s.empty]
    if available:
        all_years = sorted(set.union(*[set(s.index) for s in available]))
        total = pd.Series(0.0, index=all_years)
        for s in available:
            total = total.add(s.reindex(all_years).fillna(0), fill_value=0)
        total = total.replace(0, np.nan).dropna()
        if not total.empty:
            frames.append(_long(province, 'total_passenger_kpkm', '',
                                'output', 'kpkm', total))

            yrs = total.index
            air_num  = air_kpkm.reindex(yrs).fillna(0)
            land_num = (intercity_kpkm.reindex(yrs).fillna(0)
                        + rail_kpkm.reindex(yrs).fillna(0))

            intercity_air  = (air_num  / total).dropna()
            intercity_land = (land_num / total).dropna()

            common = sorted(set(intercity_air.index) & set(intercity_land.index))
            if common:
                urban = (1
                         - intercity_air.reindex(common)
                         - intercity_land.reindex(common)).dropna()
            else:
                urban = pd.Series(dtype=float)

            if not intercity_air.empty:
                frames.append(_long(province, 'Mode.Intercity Air', '',
                                    'service_request', 'ratio', intercity_air))
            if not intercity_land.empty:
                frames.append(_long(province, 'Mode.Intercity Land', '',
                                    'service_request', 'ratio', intercity_land))
            if not urban.empty:
                frames.append(_long(province, 'Mode.Urban', '',
                                    'service_request', 'ratio', urban))

    return frames


# ==============================================================================
# BATCH EXTRACTION
# ==============================================================================


# Territories (YT, NT, NU) have no separate CEUD file -- they are combined
# with BC in tran_bct_e.xls.  We split the BCT data using population as a
# proxy: each territory's share of BCT activity = its share of BCT population.
# Market shares (technology mix) are assumed identical across the BCT region.
BCT_REGIONS = ('BC', 'YT', 'NT', 'NU')


def _load_bct_population_shares(
    pop_path: Path = STATSCAN_POP_PATH,
) -> dict[str, pd.Series]:
    """
    Compute annual population shares of BC, YT, NT, NU within the BCT region.

    Source: Statistics Canada table 17-10-0009-01 (quarterly, persons).
    Quarters are averaged to annual values.  For years beyond the data coverage
    the last observed share is held constant (territory shares are stable).

    Returns
    -------
    dict mapping province code -- pd.Series(index=year, values=share 0--1).
    Shares sum to 1 across BC + YT + NT + NU for every year.
    """
    _GEO_MAP = {
        'British Columbia':    'BC',
        'Yukon':               'YT',
        'Northwest Territories': 'NT',
        'Nunavut':             'NU',
    }

    df = (
        pl.read_csv(str(pop_path))
        .filter(pl.col('GEO').is_in(list(_GEO_MAP.keys())))
        .filter(pl.col('VALUE').is_not_null())
        .select(['REF_DATE', 'GEO', 'VALUE'])
        .with_columns([
            pl.col('REF_DATE').str.slice(0, 4).cast(pl.Int32).alias('year'),
            pl.col('GEO').replace(_GEO_MAP).alias('prov'),
            pl.col('VALUE').cast(pl.Float64),
        ])
    )

    # Average quarters to annual -- keep only complete years (all 4 quarters present)
    annual = (
        df.group_by(['year', 'prov'])
        .agg([
            pl.col('VALUE').mean().alias('pop'),
            pl.col('VALUE').count().alias('n_quarters'),
        ])
        .filter(pl.col('n_quarters') == 4)
        .sort(['prov', 'year'])
    )

    pop: dict[str, pd.Series] = {}
    for prov in BCT_REGIONS:
        sub = annual.filter(pl.col('prov') == prov)
        pop[prov] = pd.Series(
            sub['pop'].to_list(), index=sub['year'].to_list(), dtype=float
        )

    hist_years = sorted(set.union(*[set(s.index) for s in pop.values()]))
    all_years  = list(range(min(hist_years), PROJ_HORIZON + 1))

    bct_total = sum(
        pop[p].reindex(hist_years).fillna(0) for p in BCT_REGIONS
    ).replace(0, np.nan)

    shares_hist = {
        prov: (pop[prov].reindex(hist_years) / bct_total).dropna()
        for prov in BCT_REGIONS
    }

    # Hold last observed share constant for projection years
    shares_full: dict[str, pd.Series] = {}
    for prov in BCT_REGIONS:
        s = pd.Series(np.nan, index=all_years, dtype=float)
        s.update(shares_hist[prov])
        last_val = shares_hist[prov].iloc[-1] if not shares_hist[prov].empty else np.nan
        s = s.fillna(last_val)
        shares_full[prov] = s.dropna()

    return shares_full


def _split_bct(bct_df: pl.DataFrame,
               shares: dict[str, pd.Series]) -> dict[str, pl.DataFrame]:
    """
    Split a BCT-combined DataFrame into separate BC, YT, NT, NU DataFrames.

    Rows with unit 'k*pkm' or 'k*vkm' are absolute quantities and are scaled
    by each region's population share.  All other rows (shares, fractions,
    averages) are copied unchanged — the technology mix is assumed identical
    across the BCT region.
    """
    # Scale absolute quantities; copy ratios / shares / averages unchanged.
    scale_mask = pl.col('unit').is_in(['k*pkm', 'k*vkm'])

    # Build a Polars share lookup for joining
    def _share_frame(prov: str) -> pl.DataFrame:
        s = shares[prov]
        return pl.DataFrame({
            'year':      [int(y) for y in s.index],
            'pop_share': s.values.tolist(),
        }).with_columns(pl.col('year').cast(pl.Int32))

    results: dict[str, pl.DataFrame] = {}
    for prov in BCT_REGIONS:
        share_pl = _share_frame(prov)

        activity_df = (
            bct_df.filter(scale_mask)
            .join(share_pl, on='year', how='left')
            .with_columns(
                (pl.col('value') * pl.col('pop_share').fill_null(0.0)).alias('value'),
                pl.lit(prov).alias('province'),
            )
            .drop('pop_share')
        )

        non_activity_df = (
            bct_df.filter(~scale_mask)
            .with_columns(pl.lit(prov).alias('province'))
        )

        results[prov] = pl.concat(
            [activity_df, non_activity_df], how='diagonal_relaxed'
        ).sort(['variable', 'category', 'year'])

    return results


def extract_all_provinces(
    province_codes: Optional[list[str]] = None,
    apply_projections: bool = True,
) -> dict[str, pl.DataFrame]:
    """
    Extract data for all provinces and split BC (BCT) into BC + YT + NT + NU
    using population shares.

    Returns dict keyed by province code: AB BC MB NB NL NS ON PE QC SK YT NT NU.
    BC in the output is BC-only (BCT minus territories).
    """
    if province_codes is None:
        province_codes = list(PROVINCES.keys())

    params  = load_projection_params()
    results = {}
    failed  = []

    for prov in province_codes:
        try:
            results[prov] = extract_all_data(prov, apply_projections, params)
        except Exception as exc:
            failed.append((prov, str(exc)))

    # --- Split BC (BCT) into BC + YT + NT + NU via population proxy ---
    if 'BC' in results:
        try:
            bct_shares = _load_bct_population_shares()
            split = _split_bct(results['BC'], bct_shares)
            results.update(split)   # replaces 'BC' and adds 'YT', 'NT', 'NU'
        except Exception as exc:
            failed.append(('BCT-split', str(exc)))

    for prov, err in failed:
        print(f"Warning: {prov} failed: {err}")

    return results


# ==============================================================================
# MAIN
# ==============================================================================


def main(
    province_codes: Optional[list[str]] = None,
    output_dir: Path = OUTPUT_DIR,
    apply_projections: bool = True,
    export_csv: bool = True,
) -> dict[str, pl.DataFrame]:
    """Run the full transportation passenger pipeline and optionally export CSV."""
    if province_codes is None:
        province_codes = list(PROVINCES.keys())

    if export_csv:
        output_dir.mkdir(parents=True, exist_ok=True)

    results = extract_all_provinces(province_codes, apply_projections)

    if export_csv and results:
        all_frames = list(results.values())
        combined = pl.concat(all_frames, how='diagonal_relaxed')
        combined = combined.with_columns(
            pl.when(pl.col('year') <= LAST_HIST_YEAR)
            .then(pl.lit('CEUD'))
            .otherwise(pl.lit('Assumptions'))
            .alias('source')
        )
        # Market shares are historical only; cap at LAST_HIST_YEAR.
        # Output (k*pkm, k*vkm) and mode-level service_request shares are projected to 2100.
        combined = combined.filter(
            ~((pl.col('parameter') == 'market_share_total')
              & (pl.col('year') > LAST_HIST_YEAR))
        )
        combined = combined.sort(['province', 'variable', 'category', 'year'])

        output_file = output_dir / "transportation_passenger.csv"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        print(f"\nTransportation passenger extraction complete")
        print(f"   Total rows:          {combined.height:,}")
        print(f"   Provinces processed: {combined['province'].n_unique()}")
        print(f"   Variables:           {sorted(combined['variable'].unique().to_list())}")
        print(f"   Years covered:       {combined['year'].min()} -- {combined['year'].max()}")
        print(f"   Saved to:            {output_file}")
        combined = combined.rename({
            'province': 'Region', 'variable': 'Variable', 'category': 'Category',
            'parameter': 'Parameter', 'unit': 'Unit', 'source': 'Source',
            'year': 'Year', 'value': 'Value',
        })
        combined.write_csv(str(output_file))

    return results


if __name__ == "__main__":
    main()
