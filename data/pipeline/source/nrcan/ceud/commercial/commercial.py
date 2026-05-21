"""
Commercial Pipeline

Extracts and processes commercial building data from the NRCan CEUD
(Comprehensive Energy Use Database) for all Canadian regions.

Key behavioural notes
---------------------
- AT (Atlantic) groups NB + NS + PE + NL together.
- BC groups BC + Territories together.
- BC has Marine and Cold climate HVAC (80 % Marine / 20 % Cold floorspace split).
- NG heating is split by efficiency tier using ng_eff_assumptions_commercial.csv.
- Hot water technologies are NOT split by building shell type.
- All intermediate data is held as Polars DataFrames in long format:
      (region, variable, category, parameter, unit, year, value)

Suppression handling
--------------------
RESD shares (Stats Can 2510002901-commercial):
    Loaded via load_resd(), which reads all columns as strings and casts
    VALUE to Float64 with strict=False — suppressed 'x' cells become null.
    NaN demand values propagate through pandas groupby/merge so suppressed
    territory-fuel combinations produce NaN shares rather than incorrect
    values. These NaN shares are handled by the population-proxy fallback
    in the disaggregation step.

CEUD technology splits (NRCan Excel):
    Steam energy shares are occasionally suppressed for small territories.
    When a steam share exists for some years but is null for others, the
    null years are recovered as 1 − sum(all other known fuel shares),
    ensuring fuel-type shares sum to 1 for every year.
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
from pipeline.utils.extractors.nrcan_ceud import get_row_series, row_to_series, pct_series
from pipeline.utils.output_builder import pl_to_series, pl_get_scalar
from pipeline.utils.extractors.stats_can import build_population_shares
from pipeline.utils.data_extensions import (
    extend_series_trend_dampener,
    load_cagr_assumptions,
    compute_cagr,
    extend_cagr_periods,
)
from pipeline.utils.extractors.stats_can import load_resd
from pipeline.utils.controls_conversions import DATA_START, PROJECTION_END, BASE_PATH as _CIMS_BASE

# ==============================================================================
# CONFIGURATION
# ==============================================================================

BASE_PATH        = _CIMS_BASE / 'raw_data/nrcan/ceud/commercial'
ASSUMPTIONS_CSV  = _CIMS_BASE / 'raw_data/assumptions/commercial_assumptions.csv'
ACTIVITY_CAGR_CSV = _CIMS_BASE / 'raw_data/assumptions/activity_cagr_projections.csv'
NG_EFFICIENCY_CSV = _CIMS_BASE / 'raw_data/assumptions/ng_eff_assumptions_commercial.csv'
OUTPUT_DIR       = _CIMS_BASE / 'processed_data/nrcan/ceud'
RESD_CSV       = _CIMS_BASE / 'raw_data/stats_can/resd/25100029.csv'
POP_CSV        = _CIMS_BASE / 'raw_data/stats_can/population/1710000901.csv'
EFFICIENCY_XLS = _CIMS_BASE / 'raw_data/nrcan/ceud/residential/res_ca_e_32.xls'
YEARS            = list(range(DATA_START, PROJECTION_END + 1))
LAST_HIST_YEAR   = CONTROLS["last_data_year"]["ceud"]

# Per-region overrides — explicit annual rate per period, bypasses computed CAGR.
CAGR_OVERRIDES: dict[str, tuple[float, ...]] = {
}

REGIONS = {
    'AB': 'Alberta',
    'AT': 'Atlantic',
    'BC': 'British Columbia',
    'MB': 'Manitoba',
    'ON': 'Ontario',
    'QC': 'Quebec',
    'SK': 'Saskatchewan',
}

ACTIVITY_MAPPING = {
    'Wholesale Trade':                      'Wholesale',
    'Retail Trade':                         'Retail',
    'Transportation and Warehousing':       'Transportation and Warehousing',
    'Information and Cultural Industries':  'Information and Cultural',
    'Offices':                              'Offices',
    'Educational Services':                 'Educational',
    'Health Care and Social Assistance':    'Healthcare and Social Assistance',
    'Arts, Entertainment and Recreation':   'Arts Entertainment and Recreation',
    'Accommodation and Food Services':      'Accommodation and Food Services',
    'Other Services':                       'Other Services',
}

TOTAL_FLOORSPACE_TABLE = 1
FLOORSPACE_TABLES = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
HVAC_TABLE       = 4
HOT_WATER_TABLE   = 26

FILE_NAME_MAP = {'BC': 'bct', 'AT': 'atl'}


# ==============================================================================
# HELPERS
# ==============================================================================


def _long(region: str, variable: str, category: str, parameter: str,
          unit: str, series: pd.Series, source: str = 'CEUD') -> pl.DataFrame:
    """
    Convert a year-indexed pd.Series to a Polars long-format DataFrame.

    Builds directly from Python lists to avoid the pyarrow dependency that
    pl.from_pandas() requires for object-dtype string columns.
    """
    years  = [int(y)   for y, v in series.items() if pd.notna(v)]
    values = [float(v) for y, v in series.items() if pd.notna(v)]
    n = len(years)
    return pl.DataFrame({
        'region':    [region]    * n,
        'variable':  [variable]  * n,
        'category':  [category]  * n,
        'parameter': [parameter] * n,
        'unit':      [unit]      * n,
        'source':    [source]    * n,
        'year':      years,
        'value':     values,
    }).with_columns(pl.col('year').cast(pl.Int32))


# ==============================================================================
# PARAMETER LOADING
# ==============================================================================

def load_projection_params(assumptions_csv: Path = ASSUMPTIONS_CSV,
                           activity_cagr_csv: Path = ACTIVITY_CAGR_CSV) -> dict:
    """
    Load commercial projection parameters.

    Floorspace uses the shared activity CAGR file (activity_cagr_projections.csv),
    sector 'Commercial': CAGR Period, dampener periods — same pattern as activity scripts.

    Building shell uses commercial_assumptions.csv.

    Returns
    -------
    dict with keys 'floorspace' and 'building_shell_shares'.
    Empty dict if files not found.
    """
    def parse_pct(val) -> float | None:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        s = str(val).strip()
        if s.lower() in ('trend', 'remainder', ''):
            return None
        try:
            return float(s.replace('%', '').strip()) / 100.0
        except ValueError:
            return None

    def parse_year(val) -> int | None:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        s = str(val).strip().lower()
        if s == 'first':
            return LAST_HIST_YEAR + 1
        try:
            return int(float(s))
        except ValueError:
            return None

    def parse_period(row, prefix) -> tuple[int, int] | None:
        start = parse_year(row.get(f'{prefix} start'))
        end   = parse_year(row.get(f'{prefix} end'))
        if start is None or end is None:
            return None
        return start, end + 1

    try:
        raw = pd.read_csv(assumptions_csv)
    except FileNotFoundError:
        return {}

    params = {}

    # -- 1. Floorspace — activity-style CAGR (activity_cagr_projections.csv)
    cagr_start, cagr_end, periods = load_cagr_assumptions('Commercial', activity_cagr_csv)
    params['floorspace'] = {
        'cagr_start': cagr_start,
        'cagr_end':   cagr_end,
        'periods':    periods,
    }

    # -- 2. Building shell shares — trend then dampener, per activity ----------
    bs_rows = raw[raw['Variable'] == 'Building Shell']
    first_bs     = bs_rows.iloc[0]
    trend_yrs    = parse_period(first_bs, '1st period') or (LAST_HIST_YEAR + 1, 2031)
    decline1_yrs = parse_period(first_bs, '2nd period') or (2031, 2051)
    decline2_yrs = parse_period(first_bs, '3rd period') or (2051, 2101)

    activity_rates = {}
    for _, row in bs_rows.iterrows():
        activity = str(row['Variable.1']).strip()
        r1 = parse_pct(row['2nd period rate'])
        r2 = parse_pct(row['3rd period rate'])
        activity_rates[activity] = (
            r1 if r1 is not None else -0.10,
            r2 if r2 is not None else -0.10,
        )

    params['building_shell_shares'] = {
        'method': 'trend_dampener',
        'trend_start': 2000,
        'trend_end': LAST_HIST_YEAR,
        'trend_period': trend_yrs,
        'activity_rates': activity_rates,
        'decline1_years': decline1_yrs,
        'decline2_years': decline2_yrs,
    }

    return params
def load_ng_efficiency_splits(ng_efficiency_csv: Path = NG_EFFICIENCY_CSV) -> dict:
    """
    Load NG efficiency tier splits from the assumptions CSV.

    Returns
    -------
    dict  { year: {'low': float, 'medium': float, 'high': float} }
    Falls back to equal thirds if the file cannot be read.
    """
    TECH_KEY_MAP = {
        'Natural Gas_Furnace_Low Efficiency':    'low',
        'Natural Gas_Furnace_Medium Efficiency': 'medium',
        'Natural Gas_Furnace_High Efficiency':   'high',
    }
    default = {year: {'low': 1/3, 'medium': 1/3, 'high': 1/3} for year in YEARS}

    try:
        df = pl.read_csv(str(ng_efficiency_csv), has_header=True)
    except Exception:
        return default

    tech_col = df.columns[0]
    year_cols = df.columns[1:]
    splits: dict[int, dict[str, float]] = {}

    for row in df.iter_rows(named=True):
        tech_name = str(row[tech_col]).strip().lstrip('\ufeff')
        tier = TECH_KEY_MAP.get(tech_name)
        if tier is None:
            continue
        for yr_str in year_cols:
            try:
                yr = int(yr_str)
                val = row[yr_str]
                if val is not None:
                    splits.setdefault(yr, {})[tier] = float(val)
            except (ValueError, TypeError):
                continue

    if not splits:
        return default

    last_year = max(splits.keys())
    last_vals  = splits[last_year]
    for year in YEARS:
        if year not in splits:
            splits[year] = last_vals.copy()

    return splits


# ==============================================================================
# TABLE LOADING
# ==============================================================================

def load_tables(region_code: str) -> dict:
    """Load and clean CEUD Excel sheets for a commercial region."""
    region_lower = FILE_NAME_MAP.get(region_code.upper(), region_code.lower())
    file_path = BASE_PATH / f"com_{region_lower}_e.xls"

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    table_numbers = [TOTAL_FLOORSPACE_TABLE] + FLOORSPACE_TABLES + [HVAC_TABLE] + [HOT_WATER_TABLE]
    table_names = [f"Table {n}" for n in sorted(set(table_numbers))]

    def load_and_clean(sheet_name: str) -> pl.DataFrame:
        df = pl.read_excel(str(file_path), sheet_name=sheet_name, has_header=False)
        data_cols = df.columns[2:]
        cast_exprs = []
        for c in data_cols:
            if df[c].dtype in (pl.String, pl.Utf8):
                cast_exprs.append(
                    pl.col(c)
                    .str.strip_chars()
                    .cast(pl.Float64, strict=False)
                )
        if cast_exprs:
            df = df.with_columns(cast_exprs)
        return df

    return {name: load_and_clean(name) for name in table_names}


# ==============================================================================
# EXTRACTION — FLOORSPACE
# ==============================================================================

def extract_floorspace(region: str, tables: dict) -> list[pl.DataFrame]:
    """
    Extract total commercial floorspace and per-activity floorspace shares.

    Returns
    -------
    list of pl.DataFrame in long format.
    """
    t1 = tables[f"Table {TOTAL_FLOORSPACE_TABLE}"]
    total_raw = row_to_series(t1, "Total Floor Space (million m2)") * 1e6
    frames = [_long(region, 'total_floorspace', '', 'service_request', 'm2', total_raw)]

    cims_activities = list(ACTIVITY_MAPPING.values())
    for tbl_num, cims_name in zip(FLOORSPACE_TABLES, cims_activities):
        tbl = tables[f"Table {tbl_num}"]
        raw = row_to_series(tbl, "Floor Space (million m2)") * 1e6
        frames.append(_long(region, 'floorspace_by_activity', cims_name,
                            'service_request', 'm2', raw))

    return frames


# ==============================================================================
# EXTRACTION — BUILDING SHELL SHARES
# ==============================================================================

def extract_building_shell_shares(region: str,
                                   floorspace_df: pl.DataFrame) -> list[pl.DataFrame]:
    """
    Compute each activity's fraction of total floorspace.

    Parameters
    ----------
    floorspace_df : pl.DataFrame
        Output of extract_floorspace — must contain both 'total_floorspace'
        and 'floorspace_by_activity' rows.

    Returns
    -------
    list of pl.DataFrame
    """
    total_s = pl_to_series(
        floorspace_df.filter(pl.col('variable') == 'total_floorspace')
    )

    frames = []
    for cims_name in ACTIVITY_MAPPING.values():
        act_s = pl_to_series(
            floorspace_df.filter(
                (pl.col('variable') == 'floorspace_by_activity') &
                (pl.col('category') == cims_name)
            )
        )
        share = act_s / total_s.replace(0, np.nan)
        frames.append(_long(region, 'building_shell_shares', cims_name,
                            'market_share_total', '% of m2', share))

    return frames


# ==============================================================================
# EXTRACTION — HVAC TECHNOLOGIES
# ==============================================================================

def extract_hvac_technologies(region: str, tables: dict,
                               ng_splits: dict) -> list[pl.DataFrame]:
    """
    Extract HVAC technology shares from Table 24, which provides the
    region-level weighted average shares directly.

    NG total share is split into Low / Medium / High efficiency tiers
    using ng_splits. Steam is treated as NG Cogeneration.
    BC gets both Cold (Table 24) and Marine (also Table 24) climate frames.

    Parameters
    ----------
    ng_splits : dict
        Output of load_ng_efficiency_splits().

    Returns
    -------
    list of pl.DataFrame
    """
    is_bc = region.upper() == 'BC'
    tbl = tables[f"Table {HVAC_TABLE}"]

    def _safe_pct(label, n=0):
        try:
            return pct_series(tbl, label, match_n=n)
        except KeyError:
            return pd.Series(dtype=float)

    def _split_ng(ng_total: pd.Series):
        ng_low, ng_med, ng_high = {}, {}, {}
        for year, val in ng_total.items():
            if pd.isna(val):
                ng_low[year] = ng_med[year] = ng_high[year] = np.nan
            else:
                sp = ng_splits.get(int(year), {'low': 1/3, 'medium': 1/3, 'high': 1/3})
                ng_low[year]  = val * sp['low']
                ng_med[year]  = val * sp['medium']
                ng_high[year] = val * sp['high']
        return pd.Series(ng_low), pd.Series(ng_med), pd.Series(ng_high)

    # Table 24 Shares (%) section — match_n=1 selects the Shares rows
    elec  = _safe_pct("Electricity",                n=1)
    ng_t  = _safe_pct("Natural Gas",                n=1)
    lfo   = _safe_pct("Light Fuel Oil and Kerosene", n=1)
    hfo   = _safe_pct("Heavy Fuel Oil",              n=1)
    steam = _safe_pct("Steam",                       n=1)
    other = _safe_pct("Other2",                      n=1)

    # If steam exists but has suppressed (NaN) years, recover those years as
    # the remainder so shares sum to 1.
    if not steam.dropna().empty:
        known = elec.add(ng_t, fill_value=0).add(lfo, fill_value=0) \
                    .add(hfo, fill_value=0).add(other, fill_value=0)
        steam = steam.fillna((1.0 - known).clip(lower=0))

    # Steam → NG Cogeneration; Other → Propane
    ng_lo, ng_md, ng_hi = _split_ng(ng_t)

    tech_shares = {
        'Natural Gas_Furnace_Low Efficiency':       ng_lo,
        'Natural Gas_Furnace_Medium Efficiency':    ng_md,
        'Natural Gas_Furnace_High Efficiency':      ng_hi,
        'Natural Gas_Cogeneration':                 steam,
        'Light Fuel Oil_Furnace_Low Efficiency':    lfo,
        'Light Fuel Oil_Furnace_Medium Efficiency': pd.Series(dtype=float),
        'Heavy Fuel Oil_Furnace_Low Efficiency':    hfo,
        'Heavy Fuel Oil_Furnace_Medium Efficiency': pd.Series(dtype=float),
        'Propane_Furnace_Medium Efficiency':        other,
        'Propane_Furnace_High Efficiency':          pd.Series(dtype=float),
        'Electricity_Furnace_High Efficiency':      elec,
        'Electricity_GSHP':                         pd.Series(dtype=float),
        'Electricity_ASHP_Natural Gas_Backup':      pd.Series(dtype=float),
        'Electricity_ASHP_Electricity_Backup':      pd.Series(dtype=float),
        'Natural Gas_ASHP_Natural Gas_Backup':      pd.Series(dtype=float),
    }

    frames = []
    for tech, s in tech_shares.items():
        if s.dropna().empty:
            continue
        frames.append(_long(region, 'hvac_cold', tech,
                            'market_share_total', '% of GJ HVAC', s))

    if is_bc:
        marine_shares = {
            'Natural Gas_Furnace_Low Efficiency':       ng_lo,
            'Natural Gas_Furnace_Medium Efficiency':    ng_md,
            'Natural Gas_Furnace_High Efficiency':      ng_hi,
            'Natural Gas_Cogeneration':                 steam,
            'Light Fuel Oil_Furnace_Low Efficiency':    lfo,
            'Light Fuel Oil_Furnace_Medium Efficiency': pd.Series(dtype=float),
            'Heavy Fuel Oil_Furnace_Low Efficiency':    hfo,
            'Heavy Fuel Oil_Furnace_Medium Efficiency': pd.Series(dtype=float),
            'Propane_Furnace_Medium Efficiency':        other,
            'Propane_Furnace_High Efficiency':          pd.Series(dtype=float),
            'Electricity_Furnace_High Efficiency':      elec,
            'Electricity_GSHP':                         pd.Series(dtype=float),
            'Electricity_ASHP':                         pd.Series(dtype=float),
            'Natural Gas_ASHP':                         pd.Series(dtype=float),
        }
        for tech, s in marine_shares.items():
            if s.dropna().empty:
                continue
            frames.append(_long(region, 'hvac_marine', tech,
                                'market_share_total', '% of GJ HVAC', s))

    return frames


# ==============================================================================
# EXTRACTION — HOT WATER
# ==============================================================================

def extract_hot_water(region: str, tables: dict) -> list[pl.DataFrame]:
    """
    Extract commercial hot water technology shares from Table 26.

    Steam is treated as a proxy for NG medium-efficiency boilers.

    Returns
    -------
    list of pl.DataFrame
    """
    tbl = tables[f"Table {HOT_WATER_TABLE}"]

    def _hw(label, n=1):
        try:
            return pct_series(tbl, label, match_n=n)
        except KeyError:
            return pd.Series(dtype=float)

    elec  = _hw("Electricity")
    ng    = _hw("Natural Gas")
    lfo   = _hw("Light Fuel Oil and Kerosene")
    hfo   = _hw("Heavy Fuel Oil")
    steam = _hw("Steam")
    other = _hw("Other2")

    if not steam.dropna().empty:
        known = elec.add(ng, fill_value=0).add(lfo, fill_value=0) \
                    .add(hfo, fill_value=0).add(other, fill_value=0)
        steam = steam.fillna((1.0 - known).clip(lower=0))

    # Steam → NG medium efficiency proxy
    ng_med = ng.add(steam, fill_value=0)

    hot_water_tech = {
        'Electricity_Boiler_High Efficiency':       elec,
        'Natural Gas_Boiler_Medium Efficiency':     ng_med,
        'Light Fuel Oil_Boiler_Medium Efficiency':  lfo,
        'Heavy Fuel Oil_Boiler_Medium Efficiency':  hfo,
        'Propane_Boiler_Medium Efficiency':         other,
        'Electricity_ASHP':                         pd.Series(dtype=float),
    }

    frames = []
    for tech, s in hot_water_tech.items():
        if not s.dropna().empty:
            frames.append(_long(region, 'hot_water_tech', tech,
                                'market_share_total', '% of GJ hot water', s))

    return frames


# ==============================================================================
# PROJECTION EXTENSIONS
# ==============================================================================

def apply_extensions(df: pl.DataFrame, region: str, params: dict) -> pl.DataFrame:
    """
    Apply projection extensions to a region's long-format DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Region data in long format (historical only).
    region : str
        Upper-case region code.
    params : dict
        Output of load_projection_params().

    Returns
    -------
    pl.DataFrame
        Historical data combined with projected rows through 2100.
    """
    if not params:
        return df

    frames = [df]

    def _series(variable: str, category: str) -> pd.Series:
        subset = df.filter(
            (pl.col('variable') == variable) & (pl.col('category') == category)
        )
        return pl_to_series(subset).sort_index()

    def _apply(variable: str, category: str, fn, **kwargs) -> pl.DataFrame:
        s = _series(variable, category)
        if s.dropna().empty:
            return pl.DataFrame()
        max_hist = int(s.dropna().index.max())
        extended = fn(s, base_year=max_hist, **kwargs)
        new = extended[extended.index > max_hist]
        if new.empty:
            return pl.DataFrame()
        meta = df.filter(
            (pl.col('variable') == variable) & (pl.col('category') == category)
        ).head(1)
        if len(meta) == 0:
            return pl.DataFrame()
        parameter = pl_get_scalar(meta, 'parameter')
        unit      = pl_get_scalar(meta, 'unit')
        return _long(region, variable, category, parameter, unit, new)

    # 1. Total floorspace — activity-style CAGR (same pattern as activity scripts)
    fs_params = params.get('floorspace', {})
    if fs_params:
        _cagr_start = fs_params['cagr_start']
        _cagr_end   = fs_params['cagr_end']
        _periods    = fs_params['periods']

        def _extend_floorspace(series, base_year,
                               cs=_cagr_start, ce=_cagr_end, p=_periods):
            raw_cagr = compute_cagr(series, cs, min(ce, base_year))
            base_val = float(series[base_year])
            projected = extend_cagr_periods(
                base_val, raw_cagr, p, CAGR_OVERRIDES.get(region)
            )
            return pd.concat([series, projected]).sort_index()

        frames.append(_apply('total_floorspace', '', _extend_floorspace))

    # 2. Building shell shares — trend then dampener, per activity
    bs_params = params.get('building_shell_shares', {})
    if bs_params:
        trend_kwargs_base = dict(
            trend_start  = bs_params.get('trend_start', 2000),
            trend_end    = bs_params.get('trend_end', LAST_HIST_YEAR),
            trend_period = bs_params.get('trend_period', (LAST_HIST_YEAR + 1, 2031)),
        )
        activity_rates = bs_params.get('activity_rates', {})
        decline1_yrs   = bs_params.get('decline1_years', (2031, 2051))
        decline2_yrs   = bs_params.get('decline2_years', (2051, 2101))

        activities = list(ACTIVITY_MAPPING.values())
        declining  = [a for a in activities if a != 'Other Services']

        projected_declining: dict[str, pd.Series] = {}
        for activity in declining:
            s = _series('building_shell_shares', activity)
            if s.dropna().empty:
                continue
            r1, r2 = activity_rates.get(activity, (-0.10, -0.10))
            td_kwargs = {
                **trend_kwargs_base,
                'decline_periods': [
                    (decline1_yrs[0], decline1_yrs[1], r1),
                    (decline2_yrs[0], decline2_yrs[1], r2),
                ],
            }
            max_hist_bs = int(s.dropna().index.max())
            ext = extend_series_trend_dampener(s, base_year=max_hist_bs, **td_kwargs)
            projected_declining[activity] = ext
            frames.append(_apply('building_shell_shares', activity,
                                 extend_series_trend_dampener, **td_kwargs))

        # Other Services = 1 - sum(all other activities)
        if projected_declining:
            max_hist = int(df.filter(pl.col('variable') == 'building_shell_shares')
                           ['year'].max())
            other_vals: dict[int, float] = {}
            for yr in range(max_hist + 1, PROJECTION_END + 1):
                total_other = 0.0
                for s in projected_declining.values():
                    v = s.get(yr) if yr in s.index else np.nan
                    if pd.notna(v):
                        total_other += float(v)
                other_vals[yr] = max(0.0, 1.0 - total_other)
            other_series = pd.Series(other_vals)
            meta = df.filter(
                (pl.col('variable') == 'building_shell_shares') &
                (pl.col('category') == 'Other Services')
            ).head(1)
            if len(meta) > 0:
                frames.append(_long(region, 'building_shell_shares', 'Other Services',
                                    pl_get_scalar(meta, 'parameter'),
                                    pl_get_scalar(meta, 'unit'),
                                    other_series))

    return pl.concat([f for f in frames if f is not None and len(f) > 0],
                     how='diagonal_relaxed')


# ==============================================================================
# MAIN EXTRACTION FUNCTION
# ==============================================================================

def extract_all_data(
    region_code: str,
    apply_projections: bool = True,
    projection_params: Optional[dict] = None,
    ng_efficiency_splits: Optional[dict] = None,
) -> pl.DataFrame:
    """
    Extract all commercial data for a region into a single long-format DataFrame.

    Parameters
    ----------
    region_code : str
        Region identifier (e.g. 'AB', 'BC').
    apply_projections : bool
    projection_params : dict, optional
    ng_efficiency_splits : dict, optional

    Returns
    -------
    pl.DataFrame
        Columns: region, variable, category, parameter, unit, source, year, value.
    """
    region = region_code.upper()
    if region not in REGIONS:
        raise ValueError(f"Invalid region code: {region_code}. "
                         f"Valid codes: {list(REGIONS.keys())}")

    is_bc = region == 'BC'

    if ng_efficiency_splits is None:
        ng_efficiency_splits = load_ng_efficiency_splits()

    tables = load_tables(region)

    # -- Floorspace & shell shares -------------------------------------------
    floorspace_frames = extract_floorspace(region, tables)
    floorspace_df = pl.concat(floorspace_frames)

    shell_frames = extract_building_shell_shares(region, floorspace_df)
    shell_df = pl.concat(shell_frames)

    # -- HVAC ----------------------------------------------------------------
    hvac_frames = extract_hvac_technologies(region, tables, ng_efficiency_splits)

    # -- Hot water -----------------------------------------------------------
    hw_frames = extract_hot_water(region, tables)

    # Assemble — floorspace_by_activity was only needed to derive shell shares
    total_floorspace_frames = [f for f in floorspace_frames
                               if f['variable'][0] == 'total_floorspace']
    all_frames = (total_floorspace_frames + shell_frames + hvac_frames + hw_frames)
    df = pl.concat(all_frames, how='diagonal_relaxed')

    if apply_projections:
        if projection_params is None:
            projection_params = load_projection_params()
        df = apply_extensions(df, region, projection_params)

    return df.sort(['region', 'variable', 'category', 'year'])


# ==============================================================================
# BATCH EXTRACTION
# ==============================================================================

def extract_all_regions(
    region_codes: Optional[list[str]] = None,
    apply_projections: bool = True,
) -> dict[str, pl.DataFrame]:
    """
    Extract data for multiple commercial regions.

    Returns
    -------
    dict mapping region code → pl.DataFrame
    """
    if region_codes is None:
        region_codes = list(REGIONS.keys())

    params    = load_projection_params() if apply_projections else None
    ng_splits = load_ng_efficiency_splits()
    results, failed = {}, []

    for region in region_codes:
        try:
            results[region] = extract_all_data(region, apply_projections,
                                               params, ng_splits)
        except Exception as exc:
            failed.append((region, str(exc)))

    return results

# ==============================================================================
# COMMERCIAL REGION DISAGGREGATION
# ==============================================================================
#
# Splits two aggregated CEUD regions into individual provinces/territories:
#
#   AT (Atlantic) → NL, PE, NS, NB
#   BC (BC + Territories) → BC, YT, NT, NU
#
# Rules per variable:
#   total_floorspace          → split by population share
#   floorspace_by_activity    → identical shares within group (copy parent value
#                               scaled by total_floorspace split)
#   building_shell_shares     → identical across all sub-regions
#   hvac_cold                 → split by RESD fuel share, efficiency-corrected,
#                               renormalized
#   hvac_marine               → BC only (dropped for YT/NT/NU and AT provinces)
#   hot_water_tech            → split by RESD fuel share, efficiency-corrected,
#                               renormalized
#
# ==============================================================================

# RESD fuel names → CEUD commercial category names
COMM_RESD_FUEL_TO_CEUD = {
    'Natural gas': [
        'Natural Gas_Furnace_Low Efficiency',
        'Natural Gas_Furnace_Medium Efficiency',
        'Natural Gas_Furnace_High Efficiency',
        'Natural Gas_Cogeneration',
        'Natural Gas_ASHP_Natural Gas_Backup',
        'Natural Gas_ASHP',
        'Natural Gas_Boiler_Medium Efficiency',
    ],
    'Light fuel oil': [
        'Light Fuel Oil_Furnace_Low Efficiency',
        'Light Fuel Oil_Furnace_Medium Efficiency',
        'Light Fuel Oil_Boiler_Medium Efficiency',
    ],
    'Kerosene and stove oil': [
        'Light Fuel Oil_Furnace_Low Efficiency',
        'Light Fuel Oil_Furnace_Medium Efficiency',
        'Light Fuel Oil_Boiler_Medium Efficiency',
    ],
    'Heavy fuel oil': [
        'Heavy Fuel Oil_Furnace_Low Efficiency',
        'Heavy Fuel Oil_Furnace_Medium Efficiency',
        'Heavy Fuel Oil_Boiler_Medium Efficiency',
    ],
    "Gas plant natural gas liquids (NGL's)": [
        'Propane_Furnace_Medium Efficiency',
        'Propane_Furnace_High Efficiency',
        'Propane_Boiler_Medium Efficiency',
    ],
    'Primary electricity, hydro and nuclear': [
        'Electricity_Furnace_High Efficiency',
        'Electricity_ASHP_Natural Gas_Backup',
        'Electricity_ASHP_Electricity_Backup',
        'Electricity_ASHP',
        'Electricity_GSHP',
        'Electricity_Boiler_High Efficiency',
    ],
    '_wood_proxy': [],  # no wood in commercial RESD — proxy not needed
}

# Efficiency key for each commercial CEUD category (same Table 32 efficiencies)
COMM_CATEGORY_EFFICIENCY_KEY = {
    'Natural Gas_Furnace_Low Efficiency':       'ng_low',
    'Natural Gas_Furnace_Medium Efficiency':    'ng_med',
    'Natural Gas_Furnace_High Efficiency':      'ng_high',
    'Natural Gas_Cogeneration':                 'ng_med',
    'Natural Gas_ASHP_Natural Gas_Backup':      'ng_med',
    'Natural Gas_ASHP':                         'ng_med',
    'Natural Gas_Boiler_Medium Efficiency':     'ng_med',
    'Light Fuel Oil_Furnace_Low Efficiency':    'oil_low',
    'Light Fuel Oil_Furnace_Medium Efficiency': 'oil_med',
    'Light Fuel Oil_Boiler_Medium Efficiency':  'oil_med',
    'Heavy Fuel Oil_Furnace_Low Efficiency':    'oil_low',
    'Heavy Fuel Oil_Furnace_Medium Efficiency': 'oil_med',
    'Heavy Fuel Oil_Boiler_Medium Efficiency':  'oil_med',
    'Propane_Furnace_Medium Efficiency':        'other',
    'Propane_Furnace_High Efficiency':          'other',
    'Propane_Boiler_Medium Efficiency':         'other',
    'Electricity_Furnace_High Efficiency':      'elec',
    'Electricity_ASHP_Natural Gas_Backup':      'heat_pump',
    'Electricity_ASHP_Electricity_Backup':      'heat_pump',
    'Electricity_ASHP':                         'heat_pump',
    'Electricity_GSHP':                         'heat_pump',
    'Electricity_Boiler_High Efficiency':       'elec',
}

# Variables identical across sub-regions (shares copied as-is)
COMM_IDENTICAL_VARIABLES = {
    'building_shell_shares',
}

# Market share variables needing efficiency correction + renormalization
COMM_MARKET_SHARE_VARIABLES = {
    'hvac_cold',
    'hot_water_tech',
}

# hvac_marine is BC-only — dropped for all other sub-regions
MARINE_ONLY_REGIONS = {'BC'}

# Maps aggregated CEUD region → individual sub-region codes and their
# full names as they appear in the population and RESD CSVs
COMM_REGION_MAP = {
    'AT': {
        'Newfoundland and Labrador': 'NL',
        'Prince Edward Island':      'PE',
        'Nova Scotia':               'NS',
        'New Brunswick':             'NB',
    },
    'BC': {
        'Yukon':                 'YT',
        'Northwest Territories': 'NT',
        'Nunavut':               'NU',
        # BC itself stays as BC — handled separately
    },
}

COMM_PROJECTION_END = 2100


def _build_comm_resd_shares(resd_csv: Path) -> pd.DataFrame:
    """
    Load the commercial RESD CSV, aggregate Public administration +
    Commercial and other institutional, and return territorial energy
    shares by fuel and year.

    Returns
    -------
    pd.DataFrame with columns: year, geo, fuel, share
        share = this geo's fraction of the group (AT or BC-territories) total
        for this fuel/year.
    """
    df = (
        load_resd(resd_csv)
        .filter(pl.col("characteristic").is_in([
            "Public administration",
            "Commercial and other institutional",
        ]))
        .rename({"characteristic": "sector", "value": "demand_TJ"})
        .to_pandas()
    )

    # Aggregate both sectors
    agg = (
        df.groupby(['year', 'geo', 'fuel'])['demand_TJ']
        .sum()
        .reset_index()
    )

    # Assign each geo to its parent group
    at_geos   = set(COMM_REGION_MAP['AT'].keys())
    terr_geos = set(COMM_REGION_MAP['BC'].keys())

    def get_group(geo):
        if geo in at_geos:
            return 'AT'
        if geo in terr_geos:
            return 'BC_terr'
        return None

    agg['group'] = agg['geo'].apply(get_group)
    agg = agg[agg['group'].notna()].copy()

    # Group total per fuel/year within each parent group
    group_total = (
        agg.groupby(['year', 'fuel', 'group'])['demand_TJ']
        .sum()
        .rename('group_total')
        .reset_index()
    )
    agg = agg.merge(group_total, on=['year', 'fuel', 'group'])
    agg['share'] = agg['demand_TJ'] / agg['group_total'].replace(0, np.nan)

    return agg[['year', 'geo', 'fuel', 'group', 'share']].copy()


def _get_comm_efficiency(efficiencies: dict, category: str, year: int) -> float:
    """Look up efficiency for a commercial category/year."""
    key = COMM_CATEGORY_EFFICIENCY_KEY.get(category)
    if key is None:
        return 1.0
    series = efficiencies.get(key, {})
    if not series:
        return 1.0
    if year in series:
        return series[year]
    return series[max(series.keys())]


def _build_comm_cat_to_fuel_map() -> dict:
    """Invert COMM_RESD_FUEL_TO_CEUD into {category: fuel}."""
    cat_to_fuel = {}
    for fuel, cats in COMM_RESD_FUEL_TO_CEUD.items():
        for cat in cats:
            cat_to_fuel[cat] = fuel
    return cat_to_fuel

def _load_efficiencies(efficiency_xls: Path) -> dict:
    """
    Load heating system efficiencies from CEUD Table 32.

    Returns
    -------
    dict mapping efficiency_key → {year: efficiency_as_fraction}
    e.g. {'ng_low': {2000: 0.62, 2001: 0.62, ...}, ...}
    """
    df = pd.read_excel(str(efficiency_xls), sheet_name='Table 32', header=None)

    years_row = df.iloc[10]
    year_cols = {
        int(v): j for j, v in enumerate(years_row)
        if pd.notna(v) and str(v).replace('.0', '').isdigit()
    }

    def get_series(row_idx: int) -> dict[int, float]:
        row = df.iloc[row_idx]
        return {
            yr: float(row.iloc[col]) / 100.0
            for yr, col in year_cols.items()
            if pd.notna(row.iloc[col])
        }

    return {
        'ng_low':    get_series(16),
        'ng_med':    get_series(17),
        'ng_high':   get_series(18),
        'oil_low':   get_series(13),
        'oil_med':   get_series(14),
        'oil_high':  get_series(15),
        'elec':      get_series(19),
        'heat_pump': get_series(20),
        'other':     get_series(21),
        'wood':      get_series(22),
    }

def disaggregate_commercial(
    region_dfs: dict,
    resd_csv: Path,
    pop_csv: Path,
    efficiency_xls: Path,
) -> dict:
    """
    Split AT and BC CEUD commercial data into individual provinces/territories.

    Parameters
    ----------
    region_dfs : dict
        Output of extract_all_data for each region — keys are region codes
        ('AB', 'AT', 'BC', etc.), values are pl.DataFrames.
    resd_csv : Path
        Path to commercial RESD CSV (Statistics Canada table 25-10-0029-01).
    pop_csv : Path
        Path to population CSV (Statistics Canada table 17-10-0009-01).
    efficiency_xls : Path
        Path to CEUD national res_ca_e_32.xls (Table 32 efficiencies).

    Returns
    -------
    dict
        Same structure as region_dfs but with AT and BC replaced by their
        constituent provinces/territories. BC itself is kept; only the
        territory sub-regions are added.
        Keys: 'AB', 'BC', 'MB', 'NB', 'NL', 'NS', 'NT', 'NU', 'ON', 'QC',
              'PE', 'SK', 'YT'
    """
    resd_shares  = _build_comm_resd_shares(resd_csv)
    efficiencies = _load_efficiencies(efficiency_xls)
    cat_to_fuel  = _build_comm_cat_to_fuel_map()

    # Build population shares for each group
    at_pop = build_population_shares(
        pop_csv,
        regions=list(COMM_REGION_MAP['AT'].keys()),
        projection_end=COMM_PROJECTION_END,
    )
    terr_pop = build_population_shares(
        pop_csv,
        regions=list(COMM_REGION_MAP['BC'].keys()),
        projection_end=COMM_PROJECTION_END,
    )

    results = {}

    # Keep non-AT, non-BC regions unchanged
    for code, df in region_dfs.items():
        if code not in ('AT', 'BC'):
            results[code] = df

    # ------------------------------------------------------------------
    # Split AT → NL, PE, NS, NB
    # ------------------------------------------------------------------
    if 'AT' in region_dfs:
        at_df = region_dfs['AT']
        at_pd = at_df.to_pandas()
        at_pd = at_pd[at_pd['year'] <= COMM_PROJECTION_END].copy()

        at_resd = resd_shares[resd_shares['group'] == 'AT'].copy()

        for geo_name, prov_code in COMM_REGION_MAP['AT'].items():
            sub_df = at_pd.copy()
            sub_df['region'] = prov_code

            def get_resd_share_at(row, geo=geo_name):
                fuel = cat_to_fuel.get(row['category'])
                if fuel is None:
                    return np.nan
                mask = (
                    (at_resd['geo'] == geo) &
                    (at_resd['fuel'] == fuel)
                )
                available = at_resd[mask].sort_values('year')
                if available.empty:
                    return np.nan
                yr_match = available[available['year'] == row['year']]
                if not yr_match.empty:
                    return float(yr_match.iloc[0]['share'])
                return float(available.iloc[-1]['share'])

            def get_pop_share_at(row, geo=geo_name):
                try:
                    return at_pop.loc[
                        (at_pop['territory'] == geo) &
                        (at_pop['year'] == row['year']),
                        'pop_share'
                    ].iloc[0]
                except (IndexError, KeyError):
                    avail = at_pop[at_pop['territory'] == geo]
                    return float(avail.iloc[-1]['pop_share']) if not avail.empty else np.nan

            def apply_share_at(row, _get_pop=get_pop_share_at, _get_resd=get_resd_share_at):
                var = row['variable']
                if var in COMM_IDENTICAL_VARIABLES:
                    return row['value']
                if var in ('total_floorspace', 'floorspace_by_activity'):
                    return row['value'] * _get_pop(row)
                if var == 'hvac_marine':
                    return np.nan
                return row['value'] * _get_resd(row)

            sub_df['value'] = sub_df.apply(apply_share_at, axis=1)
            # Drop marine rows for Atlantic
            sub_df = sub_df[sub_df['variable'] != 'hvac_marine']
            results[prov_code] = pl.from_pandas(sub_df)

    # ------------------------------------------------------------------
    # Split BC territories → YT, NT, NU  (BC itself kept unchanged)
    # ------------------------------------------------------------------
    if 'BC' in region_dfs:
        bc_df = region_dfs['BC']
        bc_pd = bc_df.to_pandas()
        bc_pd = bc_pd[bc_pd['year'] <= COMM_PROJECTION_END].copy()

        terr_resd = resd_shares[resd_shares['group'] == 'BC_terr'].copy()

        # Build BC-only population share from full population CSV
        # BC's share of (BC + territories) total
        bc_and_terr = ['British Columbia'] + list(COMM_REGION_MAP['BC'].keys())
        bc_terr_pop = build_population_shares(
            pop_csv,
            regions=bc_and_terr,
            projection_end=COMM_PROJECTION_END,
        )

        def get_pop_share_bc_province(row):
            try:
                return bc_terr_pop.loc[
                    (bc_terr_pop['territory'] == 'British Columbia') &
                    (bc_terr_pop['year'] == row['year']),
                    'pop_share'
                ].iloc[0]
            except (IndexError, KeyError):
                avail = bc_terr_pop[bc_terr_pop['territory'] == 'British Columbia']
                return float(avail.iloc[-1]['pop_share']) if not avail.empty else np.nan

        def apply_share_bc_province(row):
            var = row['variable']
            if var in COMM_IDENTICAL_VARIABLES:
                return row['value']
            if var in ('total_floorspace', 'floorspace_by_activity'):
                return row['value'] * get_pop_share_bc_province(row)
            # BC keeps marine, hvac_cold and hot_water_tech stay as BCT values
            # (territories are cold-only; BC retains its marine/cold split)
            return row['value']

        bc_split = bc_pd.copy()
        bc_split['region'] = 'BC'
        bc_split['value'] = bc_split.apply(apply_share_bc_province, axis=1)
        results['BC'] = pl.from_pandas(bc_split)

        # Territory population shares relative to (BC + territories) total
        # so each territory gets its fraction of the full BCT floorspace
        terr_names = list(COMM_REGION_MAP['BC'].keys())
        terr_of_bct_pop = build_population_shares(
            pop_csv,
            regions=bc_and_terr,
            projection_end=COMM_PROJECTION_END,
        )

        for geo_name, terr_code in COMM_REGION_MAP['BC'].items():
            sub_df = bc_pd.copy()
            sub_df['region'] = terr_code

            def get_resd_share_bc(row, geo=geo_name):
                fuel = cat_to_fuel.get(row['category'])
                if fuel is None:
                    return np.nan
                mask = (
                    (terr_resd['geo'] == geo) &
                    (terr_resd['fuel'] == fuel)
                )
                available = terr_resd[mask].sort_values('year')
                if available.empty:
                    return np.nan
                yr_match = available[available['year'] == row['year']]
                if not yr_match.empty:
                    return float(yr_match.iloc[0]['share'])
                return float(available.iloc[-1]['share'])

            def get_pop_share_bc(row, geo=geo_name):
                try:
                    return terr_of_bct_pop.loc[
                        (terr_of_bct_pop['territory'] == geo) &
                        (terr_of_bct_pop['year'] == row['year']),
                        'pop_share'
                    ].iloc[0]
                except (IndexError, KeyError):
                    avail = terr_of_bct_pop[terr_of_bct_pop['territory'] == geo]
                    return float(avail.iloc[-1]['pop_share']) if not avail.empty else np.nan
              
            def apply_share_bc(row, _get_pop=get_pop_share_bc, _get_resd=get_resd_share_bc):
                var = row['variable']
                if var in COMM_IDENTICAL_VARIABLES:
                    return row['value']
                if var in ('total_floorspace', 'floorspace_by_activity'):
                    return row['value'] * _get_pop(row)
                if var == 'hvac_marine':
                    return np.nan
                return row['value'] * _get_resd(row)

            sub_df['value'] = sub_df.apply(apply_share_bc, axis=1)
            # Drop marine rows for territories
            sub_df = sub_df[sub_df['variable'] != 'hvac_marine']
            results[terr_code] = pl.from_pandas(sub_df)

    # ------------------------------------------------------------------
    # Efficiency correction + renormalization for market share variables
    # ------------------------------------------------------------------
    # Efficiency correction + renormalization for disaggregated regions only.
    # AT and BC sub-regions were split using RESD fuel shares, which are fuel
    # energy shares — these need efficiency correction to convert to technology
    # market shares. Non-split regions (AB, MB, ON, QC, SK) already have
    # correct market shares from Table 24 and must NOT be modified.
    disaggregated_codes = (
        set(COMM_REGION_MAP['AT'].values()) |
        set(COMM_REGION_MAP['BC'].values()) |
        {'BC'}
    )
    for code in list(results.keys()):
        if code not in disaggregated_codes:
            continue
        df_pd = results[code].to_pandas()
        changed = False

        for var in COMM_MARKET_SHARE_VARIABLES:
            var_mask = df_pd['variable'] == var
            if not var_mask.any():
                continue
            changed = True

            # Divide by efficiency to convert fuel share → useful energy share
            def eff_correct(row):
                eff = _get_comm_efficiency(efficiencies, row['category'], row['year'])
                return row['value'] / eff if eff > 0 else row['value']

            df_pd.loc[var_mask, 'value'] = df_pd[var_mask].apply(eff_correct, axis=1)

            # Renormalize per region/year
            totals = (
                df_pd[var_mask]
                .groupby(['region', 'year'])['value']
                .sum()
                .rename('total')
                .reset_index()
            )
            df_pd = df_pd.merge(totals, on=['region', 'year'], how='left')
            df_pd.loc[var_mask, 'value'] = (
                df_pd.loc[var_mask, 'value'] /
                df_pd.loc[var_mask, 'total'].replace(0, np.nan)
            )
            df_pd = df_pd.drop(columns='total')

        if changed:
            results[code] = pl.from_pandas(df_pd)

    return results

# ==============================================================================
# MAIN
# ==============================================================================

def main(
    region_codes: Optional[list[str]] = None,
    output_dir: Path = OUTPUT_DIR,
    apply_projections: bool = True,
    export_csv: bool = True,
) -> dict[str, pl.DataFrame]:
    """
    Run the full commercial pipeline and optionally export a combined CSV.
    """
    if region_codes is None:
        region_codes = list(REGIONS.keys())

    if export_csv:
        output_dir.mkdir(parents=True, exist_ok=True)

    params    = load_projection_params() if apply_projections else None
    ng_splits = load_ng_efficiency_splits()
    results, failed, all_frames = {}, [], []

    for region in region_codes:
        try:
            df = extract_all_data(region, apply_projections, params, ng_splits)
            results[region] = df
            all_frames.append(df)
        except Exception as exc:
            failed.append((region, str(exc)))

    # Disaggregate AT → NL/PE/NS/NB and BC territories → YT/NT/NU
    if not failed or any(r not in [f[0] for f in failed] for r in ['AT', 'BC']):
        results = disaggregate_commercial(results, RESD_CSV, POP_CSV, EFFICIENCY_XLS)

    all_frames = list(results.values())

    if export_csv and all_frames:
        combined = pl.concat(all_frames, how='diagonal_relaxed')
        combined = combined.with_columns(
            pl.when(pl.col('year') <= LAST_HIST_YEAR)
            .then(pl.lit('CEUD'))
            .otherwise(pl.lit('Assumptions'))
            .alias('source')
        )
        combined = combined.sort(['region', 'variable', 'category', 'year'])
        output_file = output_dir / "commercial.csv"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        print(f"\n✅ Commercial extraction complete")
        print(f"   Total rows:          {combined.height:,}")
        print(f"   Regions processed:   {combined['region'].n_unique()}")
        print(f"   Variables:           {sorted(combined['variable'].unique().to_list())}")
        print(f"   Years covered:       {combined['year'].min()} – {combined['year'].max()}")
        print(f"   Saved to:            {output_file}")
        combined = combined.rename({
            'region': 'Region', 'variable': 'Variable', 'category': 'Category',
            'parameter': 'Parameter', 'unit': 'Unit', 'source': 'Source',
            'year': 'Year', 'value': 'Value',
        })
        combined.write_csv(str(output_file))

    if failed:
        print(f"\n⚠️  Failed regions ({len(failed)}):")
        for region, err in failed:
            print(f"   • {region}: {err}")

    return results


if __name__ == "__main__":
    main()
