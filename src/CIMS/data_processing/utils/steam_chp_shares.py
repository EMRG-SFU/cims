"""
CHP share of steam fuel, by steam sector, region, fuel and year.

MECS (US) gives each sector's cogeneration share of boiler + CHP fuel. Applied
to the Canadian steam targets it implies far more CHP electricity than
Canadian industry generates (2-11x StatCan in ON, QC, SK and MB; QC gas ~50x).
MECS is therefore only the starting point. StatCan industrial generation,
plus third-party plants, sets a cap for each province, and each sector's
Canada total is fitted to CEEDC industrial generation within those caps.

Inputs
------
raw_data/stats_can/electricity_generation/
    25100028.csv  Electricity generated from fossil fuels, annual, 1995-2021
                  (archived). Industry = 'Total industry and utilities' -
                  'Utilities'. Wood and spent pulping liquor are combined.
                  Used for 2000-2019.
    25100084.csv  Electric power generation by thermal plants, 2020-2025.
                  'Electricity producers, industries'; wood and spent pulping
                  liquor separate. Used from 2020.
raw_data/ceedc/elec_gen.csv
    CEEDC industrial generation by NAICS and fuel, Canada only, 2000-2024.
raw_data/assumptions/third_party_industrial_cogeneration.csv
    Cogeneration plants at industrial sites owned by a third party and so
    counted as utilities by StatCan (include=TRUE rows only). Their
    host-attributable generation is added to the host sector. Grandview (NB
    Petroleum Refining) was checked against TC Energy and Power Engineering;
    Cardinal and Iroquois Falls are listed but excluded (not verified).
raw_data/eia/mecs/emrg_2026_mecs-footprint-metrics.csv (techdb_steam.MECS_FILE)
    MECS cogeneration share of boiler + CHP fuel by sector and fuel, surveys
    2002-2022 (most recent survey at or before the year; 2002 before that).
    Light Industrial biomass uses the forest-products share
    (MECS_SECTOR_FOR), since sawmill wood waste is its main user.
processed_data/steam/steam_fuel_targets.csv (source/steam/steam_fuel_targets.py)
    .Steam calibration_quantity_requested by fuel, every year: the fuel each
    sector's claim is built from. Built from source data before any
    calibration script runs, so no sector reads another's calibration file.

Method (sector s, province p, fuel group g, year t)
------
    claim_s     CHP electricity sector s would make at the MECS share:
                steam fuel x MECS share / CHP fuel per GJ steam x CHP
                electricity per GJ steam (techdb_steam CHP technology)
    pool        StatCan industrial generation from group g, less spent
                pulping liquor (burned at the Pulp and Paper Recovery Furnace,
                not on .Steam). Suppressed 25-10-0028 cells are filled with
                the province total less the reported fuels, split by the
                province's 2020-2025 fuel mix.
    claims      the steam sectors' claims (techdb_steam.SECTORS)
    k           min(1, pool / sum_s claim_s); years with no pool take the
                nearest year's k; then a 3-year centred rolling median
                (K_SMOOTH_YEARS) to remove one-year spikes from filled cells
    U           CEEDC / StatCan industrial generation for Canada, by group
                and year (ownership_uplift). CEEDC counts cogeneration at
                industrial sites whoever owns the plant, about twice
                StatCan's 'industries' class (e.g. 133 vs 70 PJ of gas in
                2015).
    third_s     host-attributable generation of third-party plants at sector
                s (capacity x capacity factor x host_chp_fraction)
    max_s       CHP electricity at MAX_CHP_SHARE (0.95) of the fuel
    seed_s      the province's cap: claim_s x min(1, k x U) + third_s, no
                more than claim_s + third_s or max_s
    CEEDC fit   (_scale_to_ceedc) each sector/group/year's Canada total from
                CEEDC (CEEDC_SECTORS maps NAICS to CIMS sectors; Petroleum
                Refining is 32411 and its refinery fuel gas generation is left
                out, since still gas burns at Still Gas.Steam). Where CEEDC is
                below the sum of the caps, provinces are scaled down in
                proportion. Where it is above, provinces stay at their caps
                and the rest is recorded as ceedc_unplaced_pj (generation
                beyond what the sector's steam supports, likely condensing or
                export power), not forced in. A sector with CEEDC generation
                but no cap anywhere is spread by fuel use, up to max_s.
    chp_share_raw   CHP electricity / CHP electricity per GJ fuel / steam fuel
    chp_share   centred rolling median of chp_share_raw over
                CHP_SHARE_SMOOTH_YEARS (5) years for each sector, province and
                fuel. CHP is set as an amount of electricity, so a one-year
                dip in steam fuel spiked the raw share. After smoothing, Canada
                CHP electricity no longer matches CEEDC exactly; main() prints
                the ratio by sector.

So CEEDC sets each sector's level and StatCan (with the MECS claims and the
third-party plants) the split across provinces; provinces with no StatCan
industrial generation stay at zero. Third-party plants are in CEEDC already,
so they only move the provincial split. CEEDC_SCALING = False gives the
StatCan-only method (MECS cap, plus third-party plants).

chp_share() returns 0 for fuels with no CHP technology. In a year where the
fuel has no steam fuel of its own (it can still appear through techdb_steam's
smoothed fuel mix), it returns the nearest year's share.

Fuel groups: gas (natural gas), oil (heavy fuel oil; light fuel oil has no
CHP technology), coal, wood (solid biomass).

Output (diagnostics, every column of the working): processed_data/steam_chp/chp_shares.csv
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from CIMS.data_processing.utils.controls_conversions import BASE_PATH
from CIMS.data_processing.source.eia.mecs_steam_shares import CIMS_TO_MECS
from CIMS.data_processing.utils import techdb_steam as T

GEN_DIR      = BASE_PATH / 'raw_data/stats_can/electricity_generation'
THIRD_PARTY  = BASE_PATH / 'raw_data/assumptions/third_party_industrial_cogeneration.csv'
OUTPUT_FILE  = BASE_PATH / 'processed_data/steam_chp/chp_shares.csv'
YEARS        = range(2000, 2025)
FIRST_0084   = 2020           # use 25-10-0084 from this year (no suppression, wood vs SPL)
K_SMOOTH_YEARS = 3            # rolling-median window for k (years)
CEEDC_FILE   = BASE_PATH / 'raw_data/ceedc/elec_gen.csv'
CEEDC_SCALING = True          # scale each sector's Canada total to CEEDC
MAX_CHP_SHARE = 0.95          # most of a fuel's steam fuel that can go to CHP
CHP_SHARE_SMOOTH_YEARS = 5    # centred rolling-median window for the final CHP share (years)

# CEEDC fuel columns -> fuel group (SPL is burned at the Recovery Furnace;
# petcoke, still gas, coke oven gas etc. have no CHP technology on .Steam)
CEEDC_FUELS = {'NG': 'gas', 'HFO': 'oil', 'LFO': 'oil', 'Diesel': 'oil',
               'Coal': 'coal', 'Coke': 'coal', 'Wood': 'wood'}
# CIMS steam sector -> CEEDC NAICS rows ('-' = subtract)
CEEDC_SECTORS = {
    'Pulp and Paper':    ['322'],
    'Chemical Products': ['325'],
    'Iron and Steel':    ['3311'],
    'Metal Smelting':    ['331', '-3311'],
    # refineries only; refinery fuel gas (RFG) generation is at Still Gas.Steam
    'Petroleum Refining': ['32411'],
    # other manufacturing, incl. sawmills (321) and food (311, 312)
    'Light Industrial':  ['31-33', '-322', '-324', '-325', '-331', '-327'],
}

# Sectors that share each province's industrial generation. Add the upstream
# steam sectors here once they have steam targets and CHP technologies.
STEAM_CHP_SECTORS = list(T.SECTORS)

FUEL_GROUP = {'natural-gas': 'gas', 'heavy-fuel-oil': 'oil', 'coal': 'coal', 'biomass': 'wood'}

# Light Industrial's biomass steam is sawmill wood waste (wood products,
# NAICS 321), which MECS puts in forest products with pulp and paper, not in
# the Light Industrial MECS sectors.
MECS_SECTOR_FOR = {('Light Industrial', 'biomass'): 'Pulp and Paper'}

PROVINCES = {
    'Newfoundland and Labrador': 'NL', 'Prince Edward Island': 'PE', 'Nova Scotia': 'NS',
    'New Brunswick': 'NB', 'Quebec': 'QC', 'Ontario': 'ON', 'Manitoba': 'MB',
    'Saskatchewan': 'SK', 'Alberta': 'AB', 'British Columbia': 'BC',
}
MWH_TO_PJ = 3.6e-6
GWH_TO_PJ = 3.6e-3

G0028 = {'Natural gas': 'gas', 'Heavy fuel oil': 'oil', 'Diesel and light fuel oil': 'oil',
         'Coal': 'coal', 'Wood and spent pulping liquor': 'bio', 'Other fuels': 'other'}
G0084 = {'Natural gas': 'gas', 'Total heavy fuel oil': 'oil', 'Light fuel oil': 'oil', 'Diesel': 'oil',
         'Total coal': 'coal', 'Wood': 'wood', 'Spent pulping liquor': 'spl'}


# ── StatCan industrial generation ─────────────────────────────────────────────

def _read_0084() -> pd.Series:
    """Industrial generation (PJ) by region, group, year, 2020+, from 25-10-0084."""
    d = pd.read_csv(GEN_DIR / '25100084.csv', low_memory=False)
    d = d[(d['North American Industry Classification System (NAICS)'] == 'Electricity producers, industries')
          & (d['UOM'] == 'Megawatt hours')]
    d = d.assign(r=d['GEO'].map(PROVINCES),
                 g=d['Fuel type'].str.replace(', electricity generated', '', regex=False).map(G0084),
                 y=d['REF_DATE'].astype(int))
    d = d.dropna(subset=['r', 'g'])
    return d.groupby(['r', 'g', 'y'])['VALUE'].sum() * MWH_TO_PJ


def _read_0028() -> pd.DataFrame:
    """Industrial generation (PJ) by region and year, groups as columns, from 25-10-0028.

    Industry = total - utilities. A group cell is NaN where StatCan suppressed
    either part; 'total' is the all-fuels industrial total.
    """
    d = pd.read_csv(GEN_DIR / '25100028.csv', low_memory=False)
    d = d.assign(r=d['GEO'].map(PROVINCES), y=d['REF_DATE'].astype(int))
    d = d.dropna(subset=['r'])
    d['g'] = d['Fuel type'].map({**G0028, 'Total all fuels': 'total'})
    d = d.dropna(subset=['g'])
    v = d.groupby(['r', 'y', 'g', 'Source'])['VALUE'].sum(min_count=1).unstack('Source')
    ind = (v['Total industry and utilities'] - v['Utilities']).clip(lower=0) * GWH_TO_PJ
    return ind.unstack('g')


@lru_cache(maxsize=1)
def industrial_generation() -> pd.Series:
    """StatCan industrial generation (PJ electricity) by region, group, year.

    Groups gas, oil, coal, wood (spent pulping liquor removed).
    """
    g84 = _read_0084()
    # province fuel mix and spent pulping liquor share of wood + SPL, 2020-2025
    mix = g84.groupby(['r', 'g']).sum().unstack('g').fillna(0.0)
    mix['bio'] = mix.get('wood', 0.0) + mix.get('spl', 0.0)
    spl_share = (mix.get('spl', 0.0) / mix['bio']).where(mix['bio'] > 0, 0.0)

    w = _read_0028()
    groups = ['gas', 'oil', 'coal', 'bio', 'other']
    for c in groups:
        if c not in w:
            w[c] = float('nan')
    # fill suppressed groups: province total less reported groups, split by the 2020-25 mix
    known = w[groups].sum(axis=1, min_count=1).fillna(0.0)
    residual = (w['total'] - known).clip(lower=0)
    filled = w[groups].copy()
    for (r, y), row in w[groups].iterrows():
        missing = [c for c in groups if pd.isna(row[c])]
        if not missing or pd.isna(residual.loc[(r, y)]):
            continue
        weights = pd.Series({c: float(mix.loc[r, c]) if (r in mix.index and c in mix) else 0.0
                             for c in missing})
        weights = weights / weights.sum() if weights.sum() > 0 else pd.Series(1 / len(missing), index=missing)
        for c in missing:
            filled.loc[(r, y), c] = residual.loc[(r, y)] * weights[c]
    filled['wood'] = filled['bio'] * (1 - filled.index.get_level_values('r').map(spl_share).fillna(0.0))
    old = filled[['gas', 'oil', 'coal', 'wood']].stack().rename('pj')
    old.index.names = ['r', 'y', 'g']
    old = old.reorder_levels(['r', 'g', 'y'])
    old = old[old.index.get_level_values('y') < FIRST_0084]
    new = g84[(g84.index.get_level_values('y') >= FIRST_0084)
              & (g84.index.get_level_values('g') != 'spl')]
    return pd.concat([old, new]).sort_index()


# ── MECS cogeneration share by survey year ────────────────────────────────────

@lru_cache(maxsize=1)
def _mecs_table() -> pd.DataFrame:
    m = pd.read_csv(T.MECS_FILE)
    return m[(m['flow'] == 'input') & (m['data_source'] != 'footprint')
             & m['end_use_2'].isin(['boilers', 'cogeneration'])]


@lru_cache(maxsize=None)
def mecs_chp_share(sector: str, fuel: str, year: int) -> float:
    """MECS cogeneration share of boiler + CHP fuel for a sector and fuel.

    Most recent survey at or before the year (first survey before that). A
    fuel the survey does not report takes the sector's all-fuel share.
    MECS_SECTOR_FOR overrides the MECS sector for a sector/fuel.
    """
    sector = MECS_SECTOR_FOR.get((sector, fuel), sector)
    m = _mecs_table()
    surveys = sorted(m['mecs_year'].unique())
    survey = max([s for s in surveys if s <= year] or [surveys[0]])
    sub = m[m['name'].isin(CIMS_TO_MECS[f'.{sector}']) & (m['mecs_year'] == survey)]
    t = sub.pivot_table(index='fuel_type', columns='end_use_2', values='value', aggfunc='sum').fillna(0.0)
    for c in ('boilers', 'cogeneration'):
        if c not in t:
            t[c] = 0.0
    total = t['boilers'] + t['cogeneration']
    f = T.FUELS[fuel]['mecs']
    if f in t.index and total[f] > 0:
        return float(t.loc[f, 'cogeneration'] / total[f])
    return float(t['cogeneration'].sum() / total.sum()) if total.sum() > 0 else 0.0


# ── claims, third-party plants, shares ────────────────────────────────────────

def _chp_ratios(fuel: str) -> tuple[float, float]:
    """(fuel GJ per GJ steam, electricity GJ per GJ steam) of the fuel's CHP technology."""
    names = {n: a for n, a, _, _ in T.STEAM_TECHS}
    chp = T.STOCK_2000[fuel][1]
    spec = T.tech_spec(names[chp], fuel)
    return spec['fuel'], T._chp_electricity(names[chp])


@lru_cache(maxsize=1)
def _steam_fuel() -> pd.DataFrame:
    """.Steam fuel targets (GJ) for every steam sector, region, year, fuel."""
    from CIMS.data_processing.source.steam import steam_fuel_targets as _targets
    targets = _targets.load()
    rows = []
    for sector in STEAM_CHP_SECTORS:
        for region in _targets.regions(sector):
            for year, fuels in T.steam_fuel_targets(targets, sector, region).items():
                for fuel, gj in fuels.items():
                    rows.append((sector, region, year, fuel, gj))
    return pd.DataFrame(rows, columns=['sector', 'region', 'year', 'fuel', 'fuel_gj'])


@lru_cache(maxsize=1)
def third_party_generation() -> pd.Series:
    """Host-attributable third-party CHP generation (PJ) by sector, region, group, year."""
    if not THIRD_PARTY.exists():
        return pd.Series(dtype=float)
    p = pd.read_csv(THIRD_PARTY)
    p = p[p['include'].astype(str).str.upper() == 'TRUE']
    rows = []
    for _, r in p.iterrows():
        pj = r['capacity_mw'] * 8760 * r['capacity_factor'] * r['host_chp_fraction'] * MWH_TO_PJ
        for y in range(int(r['first_year']), int(r['last_year']) + 1):
            rows.append((r['cims_sector'], r['province'], r['fuel_group'], y, pj))
    df = pd.DataFrame(rows, columns=['sector', 'region', 'g', 'year', 'pj'])
    return df.groupby(['sector', 'region', 'g', 'year'])['pj'].sum()


@lru_cache(maxsize=1)
def chp_shares() -> pd.DataFrame:
    """CHP share of steam fuel by sector, region, fuel and year, with the working."""
    d = _steam_fuel()
    d = d[d['fuel'].isin(FUEL_GROUP) & (d['fuel_gj'] > 0)].copy()
    d['g'] = d['fuel'].map(FUEL_GROUP)
    d['mecs_share'] = [mecs_chp_share(s, f, int(y)) for s, f, y in zip(d['sector'], d['fuel'], d['year'])]
    ratios = {f: _chp_ratios(f) for f in FUEL_GROUP}
    d['claim_pj'] = [gj * sh / ratios[f][0] * ratios[f][1] / 1e6
                     for gj, sh, f in zip(d['fuel_gj'], d['mecs_share'], d['fuel'])]

    claims = d.groupby(['region', 'g', 'year'])['claim_pj'].sum().rename('claims_pj')
    pool = industrial_generation().rename('pool_pj')
    pool.index.names = ['region', 'g', 'year']
    k = pd.concat([claims, pool], axis=1).reindex(claims.index.union(pool.index))
    k['k'] = (k['pool_pj'] / k['claims_pj']).clip(upper=1.0).where(k['claims_pj'] > 0)
    # years with no pool: nearest year's k for that region and group
    k['k_raw'] = k['k']
    k['k'] = k.groupby(level=['region', 'g'])['k'].transform(lambda s: s.ffill().bfill()).fillna(1.0)
    # 3-year centred rolling median: removes one-year spikes from the
    # suppressed-cell fill (e.g. BC gas 2005) but keeps level shifts
    k['k'] = k.groupby(level=['region', 'g'])['k'].transform(
        lambda s: s.rolling(K_SMOOTH_YEARS, center=True, min_periods=1).median())
    d = d.join(k[['claims_pj', 'pool_pj', 'k_raw', 'k']], on=['region', 'g', 'year'])

    third = third_party_generation()
    d['third_party_pj'] = [float(third.get((s, r, g, y), 0.0)) for s, r, g, y
                           in zip(d['sector'], d['region'], d['g'], d['year'])]
    # most CHP electricity a fuel could give (CHP share = MAX_CHP_SHARE)
    d['max_pj'] = [gj * MAX_CHP_SHARE / ratios[f][0] * ratios[f][1] / 1e6
                   for gj, f in zip(d['fuel_gj'], d['fuel'])]
    d['ceedc_pj'] = float('nan')
    if CEEDC_SCALING:
        # cap: MECS claim limited by StatCan, uplifted by U (CEEDC / StatCan
        # industrial generation, the ownership gap), plus third-party plants
        u = ownership_uplift().rename('uplift')
        u.index.names = ['g', 'year']
        d = d.join(u, on=['g', 'year'])
        d['uplift'] = d['uplift'].fillna(1.0)
        d['seed_pj'] = (d['claim_pj'] * (d['k'] * d['uplift']).clip(upper=1.0)
                        + d['third_party_pj']).clip(upper=d['claim_pj'] + d['third_party_pj'])
        d['seed_pj'] = d['seed_pj'].clip(upper=d['max_pj'])
        d['chp_pj'] = d['seed_pj']
        d = _scale_to_ceedc(d)
    else:
        # StatCan-limited MECS claim plus third-party plants, MECS share as cap
        d['seed_pj'] = (d['k'] * d['claim_pj'] + d['third_party_pj']).clip(upper=d['claim_pj'])
        d['chp_pj'] = d['seed_pj']
    d['chp_share_raw'] = (d['chp_pj'] / d['max_pj'] * MAX_CHP_SHARE).where(d['max_pj'] > 0, 0.0)
    # Smooth the share itself, not the PJ. CHP is set as an amount of
    # electricity, so a one-year dip in a fuel's steam fuel (e.g. Chemical
    # Products ON gas 2003, 27 -> 13 PJ) or a one-year jump in the StatCan or
    # CEEDC generation (Metal Smelting BC 2008-10) spiked the share. A centred
    # rolling median over the years the fuel is used removes those spikes and
    # keeps lasting shifts. Each province's CHP electricity is no longer fitted
    # exactly to CEEDC after this (see main()).
    d = d.sort_values('year')
    d['chp_share'] = d.groupby(['sector', 'region', 'fuel'])['chp_share_raw'].transform(
        lambda s: s.rolling(CHP_SHARE_SMOOTH_YEARS, center=True, min_periods=1).median())
    return d.sort_values(['sector', 'region', 'fuel', 'year']).reset_index(drop=True)


@lru_cache(maxsize=1)
def ceedc_generation() -> pd.Series:
    """CEEDC Canada industrial generation (PJ electricity) by sector, group, year."""
    g = pd.read_csv(CEEDC_FILE, skiprows=1, dtype={'NAICS': str}).set_index(['NAICS', 'Year'])
    groups = {}
    for col, grp in CEEDC_FUELS.items():
        groups[grp] = groups.get(grp, 0.0) + g[col]
    by_code = pd.DataFrame(groups) * MWH_TO_PJ
    out = {}
    for sector, codes in CEEDC_SECTORS.items():
        total = None
        for code in codes:
            sign = -1.0 if code.startswith('-') else 1.0
            part = by_code.loc[code.lstrip('-')] * sign
            total = part if total is None else total.add(part, fill_value=0.0)
        out[sector] = total.clip(lower=0.0)
    s = pd.concat(out, names=['sector', 'year']).stack()
    s.index.names = ['sector', 'year', 'g']
    return s.reorder_levels(['sector', 'g', 'year']).sort_index()


@lru_cache(maxsize=1)
def ownership_uplift() -> pd.Series:
    """CEEDC / StatCan industrial generation by group and year (fuel-fired).

    StatCan's 'industries' class leaves out third-party-owned plants at
    industrial sites; CEEDC counts them. Applied to every province's k.
    """
    g = pd.read_csv(CEEDC_FILE, skiprows=1, dtype={'NAICS': str}).set_index(['NAICS', 'Year'])
    total = {}
    for col, grp in CEEDC_FUELS.items():
        total[grp] = total.get(grp, 0.0) + g.loc['1'][col]
    ceedc = pd.DataFrame(total).stack() * MWH_TO_PJ
    ceedc.index.names = ['year', 'g']
    statcan = industrial_generation().groupby(level=['g', 'y']).sum()
    statcan.index.names = ['g', 'year']
    ratio = ceedc.reorder_levels(['g', 'year']) / statcan
    return ratio.replace([float('inf')], float('nan')).dropna()


def _scale_to_ceedc(d: pd.DataFrame) -> pd.DataFrame:
    """Fit each sector/group/year to CEEDC's Canada total within the provincial caps.

    seed_pj is each province's cap. Where CEEDC is below the sum of caps,
    provinces are scaled down in proportion; where it is above, provinces
    stay at their caps and the rest is recorded as ceedc_unplaced_pj
    (generation CEEDC gives the sector beyond what its steam supports, e.g.
    condensing or export power). A sector with CEEDC generation but no cap
    anywhere (e.g. sawmill wood in Light Industrial, which MECS maps to
    forest products) is spread by fuel use, up to max_pj.
    """
    ceedc = ceedc_generation()
    d = d.copy()
    d['ceedc_unplaced_pj'] = 0.0
    for (sector, g, year), idx in d.groupby(['sector', 'g', 'year']).groups.items():
        target = float(ceedc.get((sector, g, year), float('nan')))
        if pd.isna(target):
            continue
        d.loc[idx, 'ceedc_pj'] = target
        cap = d.loc[idx, 'seed_pj'].to_numpy(dtype=float)
        if cap.sum() <= 0:
            cap = d.loc[idx, 'max_pj'].to_numpy(dtype=float)
        if cap.sum() <= 0:
            continue
        placed = min(target, cap.sum())
        d.loc[idx, 'chp_pj'] = cap * placed / cap.sum()
        d.loc[idx[:1], 'ceedc_unplaced_pj'] = target - placed
    d['scale'] = (d['chp_pj'] / d['seed_pj']).where(d['seed_pj'] > 0)
    return d


def chp_share(sector: str, region: str, fuel: str, year: int) -> float:
    """CHP share of a fuel's steam fuel (0 if the fuel has no CHP technology or
    no steam fuel in any year; otherwise the nearest year with steam fuel)."""
    if fuel not in FUEL_GROUP:
        return 0.0
    s = chp_shares()
    hit = s[(s['sector'] == sector) & (s['region'] == region) & (s['fuel'] == fuel)]
    if hit.empty:
        return 0.0
    # years where the fuel has no steam fuel of its own (it can still appear
    # there through techdb_steam's smoothed fuel mix) take the nearest year's share
    i = (hit['year'] - year).abs().to_numpy().argmin()
    return float(hit['chp_share'].iloc[i])


def main() -> pd.DataFrame:
    s = chp_shares()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    s.to_csv(OUTPUT_FILE, index=False)
    pd.set_option('display.width', 220)
    t = s.groupby(['sector', 'year']).apply(
        lambda g: pd.Series({'mecs': (g['fuel_gj'] * g['mecs_share']).sum() / g['fuel_gj'].sum(),
                             'scaled': (g['fuel_gj'] * g['chp_share']).sum() / g['fuel_gj'].sum()}),
        include_groups=False)
    print('CHP share of CHP-capable steam fuel, Canada')
    print(t.unstack('year').loc[:, (slice(None), [2000, 2005, 2010, 2015, 2020, 2024])].round(2).to_string())
    fit = s.assign(chp_pj_smoothed=s['chp_share'] * s['max_pj'] / MAX_CHP_SHARE)
    fit = fit.groupby(['sector', 'year'])[['chp_pj', 'chp_pj_smoothed']].sum()
    ratio = (fit['chp_pj_smoothed'] / fit['chp_pj']).unstack('year')
    print(f'\nCanada CHP electricity, smoothed share / unsmoothed (CEEDC fit), by sector')
    print(ratio.loc[:, [2000, 2005, 2010, 2015, 2020, 2024]].round(2).to_string())
    print(f'\nWrote {len(s):,} rows -> {OUTPUT_FILE}')
    return s


if __name__ == '__main__':
    main()
