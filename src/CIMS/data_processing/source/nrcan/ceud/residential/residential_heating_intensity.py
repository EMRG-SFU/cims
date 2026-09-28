"""
Residential space-heating intensity by density and vintage (NRCan CEUD)

Produces the ``service_request`` of each Vintage node's *Reference* technology
onto its Heating service -- GJ of delivered heat per m2 of floor space -- for
High Density (apartments) and LowMed Density (single detached, single attached,
mobile homes) buildings, for every region and year in the CEUD residential
files.

This replaces the hand-built "REM699.CIMSAnalysis" workbook
(``AB_CIMS_Input_Res`` rows 363-378) that produced the values currently in
``raw_data/fixed_data/residential/residential_<region>.csv``.

Method (per region and year)
----------------------------
1. Space-heating fuel by building type x heating system.
   Seed = system stock by building type (Tables 22-25) x province-wide energy
   per unit of that system (Table 8 / total stock).  The seed is balanced with
   iterative proportional fitting so that building-type totals match Table 6
   and system totals match Table 8.
2. Delivered heat = fuel x system stock efficiency (Table 26).  Dual systems
   split their fuel ``DUAL_PRIMARY_SHARE`` (80 %) to the first-named fuel and
   the rest to the second, each at its own component efficiency from Table 26.
3. Vintage profile inside each building type follows Table 33 (NRCan's gross
   output thermal requirement per m2 by building type and vintage), scaled so
   that sum(intensity x floor space, Tables 19/20) reproduces that type's heat.
4. CEUD vintages are grouped into CIMS bins (same mapping as
   ``residential.extract_vintages``) and aggregated floor-weighted:
   intensity = sum(heat) / sum(floor).  High Density = apartments; LowMed =
   single detached + single attached + mobile homes.
5. Post-processing: leading years where a bin has no floor space yet are
   back-filled from the first year with data; 2021-2035 and >2035 bins are
   0.75x and 0.75^2 x the 2001-2020 bin (JCIMS assumption carried over from
   fixed_data); years after the last CEUD year are held constant to
   PROJECTION_END; the Territories file (TR) is applied to YT, NT and NU.
   No weather normalisation is applied.

Differences from the workbook (each can be reverted with ``Options``)
----------------------------------------------------------------------
- allocation='workbook': each fuel's provincial space-heating total split by
  the building type's share of *total* residential energy (all end uses), with
  per-fuel stock-weighted efficiencies and dual systems at the averaged
  efficiency on each half.  Overstated apartments by ~15 % (AB 2015).
- vintage_shape='workbook': every building type gets the all-stock vintage
  split of space-heating energy (Table 7) divided by its own floor-space
  split, which produced non-monotonic apartment intensities.
- bin_weighting='heat': LowMed intensities averaged with heat weights instead
  of floor weights.
- The workbook's coal/propane total summed the wrong rows (excluded mobile
  homes); label-based lookup here removes that class of error.

``validation/check_workbook_port.py`` runs the workbook options against the
workbook's own CEUD inputs and reproduces rows 365-373 for 2000-2020.

Heating degree-day index
------------------------
Table 1's provincial Heating Degree-Day Index (1.0 = climate normal) is
emitted alongside the intensities, historical years only.  The sector module
uses it to weather-normalise the intensity and to drive the Weather nodes
that carry year-to-year weather variation past CIMS's vintage weighting.

Output
------
processed_data/nrcan/ceud/residential_heating_intensity.csv
    Region, Variable, Category, Parameter, Unit, Source, Year, Value
    Variable  : heating_intensity_high | heating_intensity_lowmed
    Category  : vintage bin (<1960, 1961-1980, 1981-2000, 2001-2020,
                2021-2035, >2035)
    Parameter : service_request   (Reference technology -> Heating)
    Unit      : GJ/m2
  plus
    Variable  : hdd_index   Category: ''   Parameter: weather_factor
    Unit      : index       (historical years only)
BC's Marine/Cold split is left to the sector module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import polars as pl

from CIMS.data_processing.utils.extractors.nrcan_ceud import find_row_indices, _to_float
from CIMS.data_processing.utils.controls_conversions import (
    BASE_PATH, LAST_DATA_YEAR, PROJECTION_END,
)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

RAW_DIR = BASE_PATH / 'raw_data/nrcan/ceud/residential'
OUTPUT_DIR = BASE_PATH / 'processed_data/nrcan/ceud'
OUTPUT_FILE = 'residential_heating_intensity.csv'
LAST_HIST_YEAR = LAST_DATA_YEAR['ceud']

REGION_FILES = {
    'AB': 'ab', 'BC': 'bc', 'MB': 'mb', 'NB': 'nb', 'NL': 'nl', 'NS': 'ns',
    'ON': 'on', 'PE': 'pe', 'QC': 'qc', 'SK': 'sk', 'TR': 'tr',
}
TERRITORIES = ('YT', 'NT', 'NU')   # TR intensities are applied to each

BUILDING_TYPES = ('Single Detached', 'Single Attached', 'Apartments', 'Mobile Homes')
DENSITY_TYPES = {
    'heating_intensity_high':   ('Apartments',),
    'heating_intensity_lowmed': ('Single Detached', 'Single Attached', 'Mobile Homes'),
}

SINGLE_SYSTEMS = (
    'Heating Oil – Normal Efficiency', 'Heating Oil – Medium Efficiency',
    'Heating Oil – High Efficiency', 'Natural Gas – Normal Efficiency',
    'Natural Gas – Medium Efficiency', 'Natural Gas – High Efficiency',
    'Electric', 'Heat Pump', 'Other1', 'Wood',
)
# dual system -> (Table 26 block header, primary component, secondary component)
DUAL_SYSTEMS = {
    'Wood/Electric':        ('Dual Heating Systems Electric/Wood',        'Wood',        'Electricity'),
    'Wood/Heating Oil':     ('Dual Heating Systems Heating Oil/Wood',     'Wood',        'Heating Oil'),
    'Natural Gas/Electric': ('Dual Heating Systems Electric/Natural Gas', 'Natural Gas', 'Electricity'),
    'Heating Oil/Electric': ('Dual Heating Systems Electric/Heating Oil', 'Heating Oil', 'Electricity'),
}
SYSTEMS = SINGLE_SYSTEMS + tuple(DUAL_SYSTEMS)
DUAL_PRIMARY_SHARE = 0.8

STOCK_TABLES = {'Single Detached': 'Table 22', 'Single Attached': 'Table 23',
                'Apartments': 'Table 24', 'Mobile Homes': 'Table 25'}
# (table, match_n of the million-m2 block) for floor space by vintage
FLOOR_TABLES = {'Single Detached': ('Table 19', 0), 'Single Attached': ('Table 19', 2),
                'Apartments': ('Table 20', 0), 'Mobile Homes': ('Table 20', 2)}
GOTR_MATCH = {'Single Detached': 0, 'Single Attached': 1, 'Apartments': 2, 'Mobile Homes': 3}
ENERGY_BY_TYPE_TABLES = {'Single Detached': 'Table 34', 'Single Attached': 'Table 36',
                         'Apartments': 'Table 38', 'Mobile Homes': 'Table 40'}
FUELS = ('Electricity', 'Natural Gas', 'Heating Oil', 'Other', 'Wood')
TABLE5_FUEL_LABELS = {'Electricity': 'Electricity', 'Natural Gas': 'Natural Gas',
                      'Heating Oil': 'Heating Oil', 'Other': 'Other2', 'Wood': 'Wood'}

BASE_VINTAGES = ('Before 1946', '1946–1960', '1961–1977', '1978–1983', '1984–1995',
                 '1996–2000', '2001–2005', '2006–2010', '2011–2015', '2016_2020')
VINTAGE_BINS = {
    '<1960':     ('Before 1946', '1946–1960'),
    '1961-1980': ('1961–1977', '1978–1983'),
    '1981-2000': ('1984–1995', '1996–2000'),
    '2001-2020': ('2001–2005', '2006–2010', '2011–2015', '2016_2020'),
}
# JCIMS: newer code vintages relative to the 2001-2020 bin
NEW_VINTAGE_RATIOS = {'2021-2035': 0.75, '>2035': 0.75 ** 2}

SHEETS = ['Table 5', 'Table 6', 'Table 7', 'Table 8', 'Table 18', 'Table 19',
          'Table 20', 'Table 22', 'Table 23', 'Table 24', 'Table 25', 'Table 26',
          'Table 33', 'Table 34', 'Table 36', 'Table 38', 'Table 40']


@dataclass(frozen=True)
class Options:
    """Method switches. Defaults are the corrected method."""
    allocation: str = 'ipf'          # 'ipf' | 'workbook'
    vintage_shape: str = 'gotr'      # 'gotr' | 'workbook'
    bin_weighting: str = 'floor'     # 'floor' | 'heat' (workbook)
    dual_primary_share: float = DUAL_PRIMARY_SHARE


WORKBOOK_OPTIONS = Options(allocation='workbook', vintage_shape='workbook', bin_weighting='heat')


@dataclass
class RegionInputs:
    """CEUD inputs for one region, each indexed by year.

    Units: energy PJ, stock thousands, floor million m2, efficiency fraction,
    GOTR GJ/m2.  Fields marked (workbook) are only used by the workbook options.
    """
    region: str
    vintages: tuple                                   # CEUD vintage labels, oldest first
    sh_by_type: pd.DataFrame = None                   # Table 6
    sh_by_system: pd.DataFrame = None                 # Table 8
    stock: dict = field(default_factory=dict)         # type -> Tables 22-25
    eff_single: pd.DataFrame = None                   # Table 26
    eff_dual: dict = field(default_factory=dict)      # dual -> (primary, secondary) Series
    floor: dict = field(default_factory=dict)         # type -> Tables 19/20 by vintage
    gotr: dict = field(default_factory=dict)          # type -> Series by vintage (Table 33)
    sh_by_fuel: pd.DataFrame = None                   # (workbook) Table 5
    energy_by_type: pd.DataFrame = None               # (workbook) Tables 34/36/38/40
    sh_by_vintage: pd.DataFrame = None                # (workbook) Table 7
    floor_total: pd.DataFrame = None                  # (workbook) Table 18


# ==============================================================================
# LOADING
# ==============================================================================

def _year_cols(table: pl.DataFrame) -> list[tuple[int, int]]:
    """Locate the header row holding the years and return (year, column) pairs."""
    arr = table.to_numpy()
    for r in range(min(20, arr.shape[0])):
        found = []
        for c in range(arr.shape[1]):
            v = _to_float(arr[r, c])
            if not np.isnan(v) and float(v).is_integer() and 1990 <= v <= 2040:
                found.append((int(v), c))
        if len(found) >= 5:
            return found
    raise ValueError('No year header row found')


def _row(table: pl.DataFrame, idx: int) -> pd.Series:
    arr = table.to_numpy()
    return pd.Series({y: _to_float(arr[idx, c]) for y, c in _year_cols(table)})


def _series(table: pl.DataFrame, label: str, match_n: int = 0) -> pd.Series:
    idxs = find_row_indices(table, label)
    if len(idxs) <= match_n:
        raise KeyError(f"Label '{label}' not found (need #{match_n + 1}, got {len(idxs)})")
    return _row(table, idxs[match_n])


def _frame(table: pl.DataFrame, labels, match_n: int = 0) -> pd.DataFrame:
    return pd.DataFrame({lab: _series(table, lab, match_n) for lab in labels})


def _labels(table: pl.DataFrame) -> list[str]:
    return [str(v).strip() for v in table[table.columns[1]].to_list() if v is not None]


def _dual_component(table: pl.DataFrame, header: str, component: str) -> pd.Series:
    """Efficiency of one fuel inside a Table 26 dual-system block."""
    start = find_row_indices(table, header)[0]
    labels = table[table.columns[1]].to_list()
    for idx in range(start + 1, min(start + 4, len(labels))):
        if labels[idx] is not None and str(labels[idx]).strip() == component:
            return _row(table, idx) / 100.0
    raise KeyError(f"'{component}' not found under '{header}'")


def load_region(region: str, raw_dir: Path = RAW_DIR) -> RegionInputs:
    """Read every CEUD table this module needs for one region file."""
    path = raw_dir / f'res_{REGION_FILES[region]}_e.xls'
    if not path.exists():
        raise FileNotFoundError(path)
    t = pl.read_excel(str(path), sheet_name=SHEETS, has_header=False)

    last = [lab for lab in _labels(t['Table 19']) if lab.endswith('_after')]
    vintages = BASE_VINTAGES + (last[0],) if last else BASE_VINTAGES

    eff_dual = {d: (_dual_component(t['Table 26'], hdr, p), _dual_component(t['Table 26'], hdr, s))
                for d, (hdr, p, s) in DUAL_SYSTEMS.items()}

    return RegionInputs(
        region=region,
        vintages=vintages,
        sh_by_type=_frame(t['Table 6'], BUILDING_TYPES),
        sh_by_system=_frame(t['Table 8'], SYSTEMS),
        stock={bt: _frame(t[tab], SYSTEMS) for bt, tab in STOCK_TABLES.items()},
        eff_single=_frame(t['Table 26'], SINGLE_SYSTEMS) / 100.0,
        eff_dual=eff_dual,
        floor={bt: _frame(t[tab], vintages, n) for bt, (tab, n) in FLOOR_TABLES.items()},
        gotr={bt: _frame(t['Table 33'], vintages, n).replace(0.0, np.nan).ffill().bfill().iloc[-1]
              for bt, n in GOTR_MATCH.items()},
        sh_by_fuel=pd.DataFrame({f: _series(t['Table 5'], lab) for f, lab in TABLE5_FUEL_LABELS.items()}),
        energy_by_type=pd.DataFrame({bt: _row(t[tab], _first_total_row(t[tab]))
                                     for bt, tab in ENERGY_BY_TYPE_TABLES.items()}),
        sh_by_vintage=_frame(t['Table 7'], vintages),
        floor_total=_frame(t['Table 18'], BUILDING_TYPES),
    )


def _first_total_row(table: pl.DataFrame) -> int:
    for idx, v in enumerate(table[table.columns[1]].to_list()):
        if v is not None and str(v).strip().startswith('Total'):
            return idx
    raise KeyError('No Total row')


# ==============================================================================
# STEP 1-2: DELIVERED HEAT BY BUILDING TYPE
# ==============================================================================

def system_efficiencies(inp: RegionInputs, primary_share: float) -> pd.DataFrame:
    """Efficiency (fraction) per system and year; dual systems blend components."""
    eff = inp.eff_single.copy()
    for d, (prim, sec) in inp.eff_dual.items():
        eff[d] = primary_share * prim + (1.0 - primary_share) * sec
    return eff[list(SYSTEMS)]


def _ipf(seed: np.ndarray, rows: np.ndarray, cols: np.ndarray,
         tol: float = 1e-12, max_iter: int = 5000) -> np.ndarray:
    """Scale ``seed`` so its row sums equal ``rows`` and column sums ``cols``."""
    x = seed.astype(float).copy()
    for _ in range(max_iter):
        rs = x.sum(axis=1)
        x *= np.divide(rows, rs, out=np.ones_like(rs), where=rs > 0)[:, None]
        cs = x.sum(axis=0)
        x *= np.divide(cols, cs, out=np.ones_like(cs), where=cs > 0)[None, :]
        if np.max(np.abs(x.sum(axis=1) - rows)) <= tol * max(rows.sum(), 1.0):
            break
    return x


def allocate_fuel_ipf(inp: RegionInputs, year: int) -> pd.DataFrame:
    """Space-heating fuel (PJ) by building type x system for one year."""
    n = np.nan_to_num(np.array([[inp.stock[bt].at[year, s] for s in SYSTEMS]
                                for bt in BUILDING_TYPES]))
    sys_e = np.nan_to_num(inp.sh_by_system.loc[year, list(SYSTEMS)].to_numpy(float))
    type_e = np.nan_to_num(inp.sh_by_type.loc[year, list(BUILDING_TYPES)].to_numpy(float))
    if type_e.sum() > 0:
        sys_e = sys_e * type_e.sum() / sys_e.sum()   # Tables 6 and 8 share a total
    stock_tot = n.sum(axis=0)
    per_unit = np.divide(sys_e, stock_tot, out=np.zeros_like(sys_e), where=stock_tot > 0)
    seed = n * per_unit
    # energy for a system with no reported stock: spread by building-type totals
    orphan = (stock_tot == 0) & (sys_e > 0)
    if orphan.any() and type_e.sum() > 0:
        seed[:, orphan] = np.outer(type_e / type_e.sum(), sys_e[orphan])
    fitted = _ipf(seed, type_e, sys_e)
    return pd.DataFrame(fitted, index=list(BUILDING_TYPES), columns=list(SYSTEMS))


def heat_by_type_ipf(inp: RegionInputs, opts: Options) -> pd.DataFrame:
    """Delivered space heat (PJ) by year x building type."""
    eff = system_efficiencies(inp, opts.dual_primary_share)
    out = {}
    for year in inp.sh_by_type.index:
        fuel = allocate_fuel_ipf(inp, year)
        out[year] = (fuel * eff.loc[year, list(SYSTEMS)].fillna(0.0)).sum(axis=1)
    return pd.DataFrame(out).T[list(BUILDING_TYPES)]


def _workbook_fuel_efficiencies(inp: RegionInputs, bt: str) -> pd.DataFrame:
    """Per-fuel stock-weighted efficiency for one building type (workbook rows 106-131)."""
    e, n = inp.eff_single, inp.stock[bt].fillna(0.0)
    avg = {  # workbook rows 159-162: dual systems at the average of two singles
        'Wood/Electric': (e['Wood'] + e['Electric']) / 2,
        'Wood/Heating Oil': (e['Wood'] + e['Heating Oil – Medium Efficiency']) / 2,
        'Natural Gas/Electric': (e['Electric'] + e['Natural Gas – Medium Efficiency']) / 2,
        'Heating Oil/Electric': (e['Electric'] + e['Heating Oil – Medium Efficiency']) / 2,
    }
    eff_of = lambda s: avg[s] if s in avg else e[s]
    parts = {
        'Electricity': [('Electric', 1), ('Heat Pump', 1), ('Wood/Electric', .5),
                        ('Natural Gas/Electric', .5), ('Heating Oil/Electric', .5)],
        'Natural Gas': [('Natural Gas – Normal Efficiency', 1), ('Natural Gas – Medium Efficiency', 1),
                        ('Natural Gas – High Efficiency', 1), ('Natural Gas/Electric', .5)],
        'Heating Oil': [('Heating Oil – Normal Efficiency', 1), ('Heating Oil – Medium Efficiency', 1),
                        ('Heating Oil – High Efficiency', 1), ('Wood/Heating Oil', .5),
                        ('Heating Oil/Electric', .5)],
        'Other': [('Other1', 1)],
        'Wood': [('Wood', 1), ('Wood/Electric', .5), ('Wood/Heating Oil', .5)],
    }
    out = {}
    for fuel, comps in parts.items():
        num = sum(eff_of(s) * n[s] * w for s, w in comps)
        den = sum(n[s] * w for s, w in comps)
        out[fuel] = num / den.replace(0.0, np.nan)
    return pd.DataFrame(out)


def heat_by_type_workbook(inp: RegionInputs) -> pd.DataFrame:
    """Workbook rows 78-143: fuel split by share of total energy x per-fuel efficiency."""
    te = inp.energy_by_type[list(BUILDING_TYPES)]
    share = te.div(te.sum(axis=1), axis=0)
    out = {}
    for bt in BUILDING_TYPES:
        fuel = inp.sh_by_fuel[list(FUELS)].mul(share[bt], axis=0)
        eff = _workbook_fuel_efficiencies(inp, bt)[list(FUELS)]
        out[bt] = (fuel * eff).where(fuel != 0, 0.0).sum(axis=1, min_count=1)
    return pd.DataFrame(out)


# ==============================================================================
# STEP 3-4: VINTAGE SPLIT AND CIMS BINS
# ==============================================================================

def heat_and_floor_by_vintage(inp: RegionInputs, heat: pd.DataFrame,
                              opts: Options) -> pd.DataFrame:
    """Long frame: building_type, vintage, year, heat (PJ), floor (million m2)."""
    rows = []
    for bt in BUILDING_TYPES:
        floor = inp.floor[bt][list(inp.vintages)].fillna(0.0)
        if opts.vintage_shape == 'gotr':
            g = inp.gotr[bt][list(inp.vintages)].fillna(0.0)
            modelled = floor.mul(g, axis=1)                        # PJ of GOTR
            k = heat[bt] / modelled.sum(axis=1).replace(0.0, np.nan)
            v_heat = modelled.mul(k, axis=0)
        elif opts.vintage_shape == 'workbook':
            sv = inp.sh_by_vintage[list(inp.vintages)].fillna(0.0)
            v_heat = sv.div(sv.sum(axis=1), axis=0).mul(heat[bt], axis=0)
            floor = floor.div(floor.sum(axis=1), axis=0).mul(inp.floor_total[bt], axis=0)
        else:
            raise ValueError(opts.vintage_shape)
        for v in inp.vintages:
            rows.append(pd.DataFrame({'building_type': bt, 'vintage': v, 'year': floor.index,
                                      'heat': v_heat[v].to_numpy(), 'floor': floor[v].to_numpy()}))
    return pd.concat(rows, ignore_index=True)


def bin_intensities(hv: pd.DataFrame, opts: Options) -> pd.DataFrame:
    """Intensity (GJ/m2) by density variable x CIMS bin x year."""
    to_bin = {v: b for b, vs in VINTAGE_BINS.items() for v in vs}
    hv = hv.assign(bin=hv['vintage'].map(to_bin)).dropna(subset=['bin'])
    by_type = hv.groupby(['building_type', 'bin', 'year'])[['heat', 'floor']].sum()
    by_type['intensity'] = by_type['heat'] / by_type['floor'].where(by_type['floor'] > 0)
    by_type = by_type.reset_index()

    out = []
    for var, types in DENSITY_TYPES.items():
        sub = by_type[by_type['building_type'].isin(types)]
        if opts.bin_weighting == 'floor':
            g = sub.groupby(['bin', 'year'])[['heat', 'floor']].sum()
            val = g['heat'] / g['floor'].where(g['floor'] > 0)
        elif opts.bin_weighting == 'heat':
            s = sub.dropna(subset=['intensity'])
            g = s.assign(w=s['intensity'] * s['heat']).groupby(['bin', 'year'])[['w', 'heat']].sum()
            val = g['w'] / g['heat'].where(g['heat'] > 0)
        else:
            raise ValueError(opts.bin_weighting)
        out.append(val.rename('value').reset_index().assign(variable=var))
    return pd.concat(out, ignore_index=True)[['variable', 'bin', 'year', 'value']]


def compute_region(inp: RegionInputs, opts: Options = Options()) -> dict:
    """Historical intensities for one region plus intermediates for diagnostics."""
    if opts.allocation == 'ipf':
        heat = heat_by_type_ipf(inp, opts)
    elif opts.allocation == 'workbook':
        heat = heat_by_type_workbook(inp)
    else:
        raise ValueError(opts.allocation)
    hv = heat_and_floor_by_vintage(inp, heat, opts)
    return {'heat_by_type': heat, 'by_vintage': hv, 'intensity': bin_intensities(hv, opts)}


# ==============================================================================
# STEP 5: POST-PROCESSING
# ==============================================================================

def finalize(intensity: pd.DataFrame, region: str,
             last_hist_year: int = LAST_HIST_YEAR,
             projection_end: int = PROJECTION_END) -> pd.DataFrame:
    """Back-fill, add JCIMS vintages, extend constant, and shape to the output schema."""
    frames = []
    for var, g in intensity.groupby('variable'):
        wide = g.pivot(index='year', columns='bin', values='value')
        wide = wide.reindex(range(int(wide.index.min()), last_hist_year + 1))
        source = pd.DataFrame('CEUD', index=wide.index, columns=wide.columns)
        source = source.where(wide.notna(), 'Assumptions')   # back-filled years
        wide = wide.bfill()
        for b, ratio in NEW_VINTAGE_RATIOS.items():
            wide[b] = wide['2001-2020'] * ratio
            source[b] = 'JCIMS'
        future = range(last_hist_year + 1, projection_end + 1)
        wide = pd.concat([wide, pd.DataFrame([wide.loc[last_hist_year]] * len(future), index=future)])
        source = pd.concat([source, pd.DataFrame('Assumptions', index=future, columns=source.columns)])
        source.loc[list(future), list(NEW_VINTAGE_RATIOS)] = 'JCIMS'
        long = wide.stack().rename('Value').reset_index()
        long.columns = ['Year', 'Category', 'Value']
        src = source.stack().rename('Source').reset_index()
        src.columns = ['Year', 'Category', 'Source']
        frames.append(long.merge(src, on=['Year', 'Category']).assign(Variable=var))
    df = pd.concat(frames, ignore_index=True)
    df['Region'] = region
    df['Parameter'] = 'service_request'
    df['Unit'] = 'GJ/m2'
    return df[['Region', 'Variable', 'Category', 'Parameter', 'Unit', 'Source', 'Year', 'Value']]


# ==============================================================================
# HEATING DEGREE-DAY INDEX
# ==============================================================================

HDD_TABLE = 'Table 1'
HDD_LABEL = 'Heating Degree-Day Index'


def load_hdd_index(region: str, raw_dir: Path = RAW_DIR) -> pd.Series:
    """Table 1 Heating Degree-Day Index for one region file, indexed by year."""
    path = raw_dir / f'res_{REGION_FILES[region]}_e.xls'
    if not path.exists():
        raise FileNotFoundError(path)
    t = pl.read_excel(str(path), sheet_name=HDD_TABLE, has_header=False)
    return _series(t, HDD_LABEL).dropna()


def hdd_frame(hdd: pd.Series, region: str,
              last_hist_year: int = LAST_HIST_YEAR) -> pd.DataFrame:
    """Shape an HDD index series to the output schema (historical years only)."""
    hdd = hdd[hdd.index <= last_hist_year]
    return pd.DataFrame({
        'Region': region, 'Variable': 'hdd_index', 'Category': '',
        'Parameter': 'weather_factor', 'Unit': 'index', 'Source': 'CEUD',
        'Year': hdd.index.astype(int), 'Value': hdd.to_numpy(dtype=float),
    })


# ==============================================================================
# DIAGNOSTICS
# ==============================================================================

def diagnostics(inp: RegionInputs, result: dict) -> dict:
    """Closure checks on one region's result (historical years only)."""
    hv = result['by_vintage']
    heat = result['heat_by_type']
    closure = hv.groupby(['year', 'building_type'])['heat'].sum().unstack() - heat
    last_v = inp.vintages[-1]
    unbinned = hv[~hv['vintage'].isin(sum(VINTAGE_BINS.values(), ()))]
    unbinned_share = (unbinned.groupby('year')['heat'].sum()
                      / hv.groupby('year')['heat'].sum()).max()
    eff = heat.sum(axis=1) / inp.sh_by_type[list(BUILDING_TYPES)].sum(axis=1)
    return {
        'max_abs_heat_closure_PJ': float(np.nanmax(np.abs(closure.to_numpy()))),
        f'max_share_of_heat_in_{last_v}': float(unbinned_share),
        'avg_system_efficiency_range': (float(eff.min()), float(eff.max())),
    }


# ==============================================================================
# MAIN
# ==============================================================================

def extract_all(regions: Optional[list[str]] = None, opts: Options = Options(),
                raw_dir: Path = RAW_DIR, verbose: bool = True) -> pd.DataFrame:
    regions = regions or list(REGION_FILES)
    frames, failed = [], []
    for region in regions:
        try:
            inp = load_region(region, raw_dir)
            result = compute_region(inp, opts)
            df = pd.concat([finalize(result['intensity'], region),
                            hdd_frame(load_hdd_index(region, raw_dir), region)],
                           ignore_index=True)
            if verbose:
                print(f'   {region}: {diagnostics(inp, result)}')
            if region == 'TR':
                frames += [df.assign(Region=t) for t in TERRITORIES]
            else:
                frames.append(df)
        except Exception as exc:  # keep going, report at the end
            failed.append((region, repr(exc)))
    if failed:
        print('\n⚠️  Failed regions:')
        for region, err in failed:
            print(f'   • {region}: {err}')
    out = pd.concat(frames, ignore_index=True)
    order = {b: i for i, b in enumerate(list(VINTAGE_BINS) + list(NEW_VINTAGE_RATIOS))}
    return (out.assign(_o=out['Category'].map(order))
               .sort_values(['Region', 'Variable', '_o', 'Year'])
               .drop(columns='_o').reset_index(drop=True))


def main(regions: Optional[list[str]] = None, output_dir: Path = OUTPUT_DIR,
         export_csv: bool = True, opts: Options = Options()) -> pd.DataFrame:
    print('Residential space-heating intensity (CEUD)')
    df = extract_all(regions, opts)
    if export_csv:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / OUTPUT_FILE
        df.to_csv(path, index=False)
        print(f'\n✅ Residential heating intensity complete')
        print(f'   Total rows:   {len(df):,}')
        print(f'   Regions:      {df["Region"].nunique()}')
        print(f'   Years:        {df["Year"].min()} – {df["Year"].max()}')
        print(f'   Saved to:     {path}')
    return df


if __name__ == '__main__':
    main()
