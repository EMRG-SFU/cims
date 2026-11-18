"""
Steam technologies from the archetype technology database.

Replaces the technologies on each sector's main ``.Steam`` node with steam
archetypes from the technology database, at model-input time. The fixed data
files are left untouched; each sector's *_model_inputs.py calls
``replace_steam_techs`` right after flattening its fixed data.

Sectors (SECTORS): Pulp and Paper, Iron and Steel, Light Industrial, Chemical
Products, Metal Smelting, Petroleum Refining. Only the main
``CIMS.CAN.{R}.{Sector}.Steam`` node changes. Sub-nodes that supply it (Iron
and Steel ``Iron Process Gas.Steam``, the Pulp and Paper Recovery Furnace,
Petroleum Refining ``Still Gas.Steam``) are kept as they are. Petroleum
Refining's still gas, most of its steam fuel, stays on ``Still Gas.Steam``,
whose fixed-data boilers and cogen supply the main node through a negative
service request; the main node's targets cover only its own fuels (natural
gas, heavy fuel oil, petroleum coke, LPG, electricity).

Inputs
------
processed_data/technology_database/
    archetype-parameter.csv         capex, opex, lifetime, annual output,
                                    specific energy consumption, efficiency,
                                    emissions captured
    archetype-alternative-carrier.csv  fuel use on alternative fuels
    archetype-flow.csv              fuel use by commodity (fallback)
processed_data/steam/steam_fuel_targets.csv (source/steam/steam_fuel_targets.py)
    .Steam calibration_quantity_requested by fuel, 2000-2024: the same rows
    the sector calibration scripts write (EIA MECS steam
    share x CER sector demand, fuel split from the provincial RESD mix; see
    mecs_steam_calibration.py). Pulp and Paper has black liquor removed (the
    Recovery Furnace supplies that steam) and wood waste filled first; Light
    Industrial also fills wood waste first.
processed_data/steam_chp/chp_shares.csv (through steam_chp_shares.chp_share)
    CHP share of each fuel's steam fuel. MECS_FILE is defined here but read
    by steam_chp_shares.

Technologies
------------
One CIMS technology per archetype x fuel (STEAM_TECHS): conventional,
efficient, solid-fuel and electric boilers; gas-turbine, back-pressure
steam-turbine and reciprocating-engine CHP; industrial and
very-high-temperature heat pumps; CCS variants. SMR, geothermal, solar,
thermal storage and fuel cells are not used. Per GJ of steam:
    output              archetype annual-output (GJ/yr)
    fcc                 capex (2025 CAD per GJ/yr) x output
    fom                 opex-fixed (2025 CAD per GJ/yr) x output
    lifetime            archetype lifetime
    service_request     fuel: specific energy consumption for that fuel
                        (parameter for the fuel, else alternative-carrier,
                        else archetype-flow >= 1 GJ/GJ, else the archetype's
                        main fuel)
                        electricity: auxiliary use, less CHP electricity
                        (efficiency:electricity / efficiency:heat), so CHP
                        has a negative request
                        VOM: opex-variable (2025 CAD/GJ); efficiency
                        variants without one use the parent archetype's
    CCS variants        energy and costs of the parent archetype plus a
                        request to the sector's .CCS node for the CO2
                        captured (emissions-captured, t/GJ). The .CCS node
                        already holds capture capex, capture energy and
                        transport and storage, as the old NG CCS techs did.
                        Offered only in CCS_SECTORS (Chemical Products, Iron
                        and Steel, Petroleum Refining).
    proxy techs         the tech DB has no LPG or petroleum coke data, so
                        Boiler - LPG and Solid Fuel Boiler - Petroleum Coke
                        use the main-fuel energy use of the fuel-fired and
                        solid-fuel boiler archetypes.
First year available: TRL 9 techs 1930 (as in the old fixed data); industrial
heat pumps 2000; very-high-temperature heat pumps and hydrogen boilers 2025;
CCS 2030. All are unavailable from 2101.
Sector restrictions (TECH_SECTORS, offered()): cheap solid fuels are offered
only where the targets burn them; elsewhere they took new stock on fuel price
alone. Petroleum coke boilers: Chemical Products and Petroleum Refining.
Coal boilers and coal steam-turbine CHP: Pulp and Paper and Chemical Products.
Dollars: the database is in 2025 CAD; cost rows carry Unit '2025_CAD' so the
model converts them to the scenario dollar year.
Emissions are not on the technologies: steam emissions come through the fuel
nodes. No DIC, DCC, market share limits or intangible costs are written for
the new techs; FICs come from fitting.

Historical market shares
------------------------
For each region and year (stock_shares):
    1. Fuel mix: the .Steam calibration targets give steam fuel by fuel. Each
       node's fuel shares are smoothed with a centred 5-year rolling mean
       (smooth_fuel_mix, FUEL_MIX_SMOOTH_YEARS); each year keeps its own
       total. The calibration_quantity_requested targets themselves are not
       smoothed, so the market share targets' fuel split differs slightly
       from the quantity targets.
    2. Boiler vs CHP: each CHP-capable fuel (gas, heavy fuel oil, coal, solid
       biomass) is split between its stock boiler and CHP technology
       (STOCK_2000) by steam_chp_shares.chp_share: the MECS cogeneration
       share, constrained by StatCan industrial generation by province and
       fitted to CEEDC industrial generation for Canada, plus third-party
       plants at industrial hosts (CHP_IN_HISTORICAL_STOCK; False = all
       boilers).
    3. Efficient boilers (EFFICIENT_BOILER, efficient_boiler_share): the
       Efficient Boiler gas, heavy fuel oil and light fuel oil techs hold a
       share of each fuel's non-CHP boiler steam that rises linearly from 0%
       in 2000 to 20% in 2024 (EFFICIENT_YEARS, EFFICIENT_SHARE_START/END).
       The boiler fuel is unchanged; the split is made on steam output, at
       each tech's own fuel use. This is an assumption: no Canadian series
       gives industrial boiler stock by efficiency.
    4. Conversion: each tech's fuel is divided by its fuel use per GJ of
       steam and the results are normalised to sum to 1. Electricity goes to
       the electric boiler; LPG and petroleum coke to the proxy boilers.
Techs with no historical stock (0 every year): Industrial Heat Pump,
Reciprocating Engine CHP, Steam Turbine CHP - Natural Gas, and the hydrogen,
very-high-temperature heat pump and CCS techs.

    2000          market_share_total on the .Steam technologies (model input,
                  build_steam_rows). The efficient boiler share is 0. Regions
                  with no 2000 steam target get the natural gas boiler at 100%.
    every year    calibration_market_share_total rows, added to the sector's
                  calibration files by add_steam_market_share_targets (called
                  from the six *_calibration.py scripts after
                  filter_excluded_branches), for FIC fitting. Every tech
                  available in a year gets a row, 0 where it holds no stock.

Fitting FICs: output ranges from about 14,000 to 1,400,000 GJ/yr and FIC
enters the LCC as fic / output, so fit with scale_by_output=True
(optimize_ms.py); otherwise the fitted FICs barely move the LCC.

Process steam demand (the processes' service requests to .Steam) is left as
it is in the fixed data.

Other rows that name the old .Steam technologies (market share limits, DIC,
DCC, FIC) are dropped with ``drop_replaced_steam_rows``.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd
import polars as pl

from CIMS.data_processing.utils.controls_conversions import BASE_PATH

TECHDB_DIR      = BASE_PATH / 'processed_data/technology_database'
MECS_FILE       = BASE_PATH / 'raw_data/eia/mecs/emrg_2026_mecs-footprint-metrics.csv'

SECTORS = {     # CIMS sector name -> calibration folder
    'Pulp and Paper':    'pulp_and_paper',
    'Iron and Steel':    'iron_and_steel',
    'Light Industrial':  'light_industrial',
    'Chemical Products': 'chemical_products',
    'Metal Smelting':    'metal_smelting',
    'Petroleum Refining': 'petroleum_refining',
}
CCS_SECTORS = {'Chemical Products', 'Iron and Steel', 'Petroleum Refining'}
# Technologies offered only in some sectors (besides CCS, see CCS_SECTORS):
# the sectors whose .Steam calibration targets (2000-2024) burn that fuel.
# Offered elsewhere, these cheap solid fuels took new stock in 2001 on fuel
# price alone (no FICs yet), in sectors that burn none.
#   Petroleum coke: Chemical Products and Petroleum Refining only.
#   Coal: Pulp and Paper and Chemical Products only.
TECH_SECTORS = {
    'Solid Fuel Boiler - Petroleum Coke': {'Chemical Products', 'Petroleum Refining'},
    'Solid Fuel Boiler - Coal':           {'Pulp and Paper', 'Chemical Products'},
    'Steam Turbine CHP - Coal':           {'Pulp and Paper', 'Chemical Products'},
}

# fuel key -> CIMS target ({R} region, {S} sector), names in the tech DB,
# fuel name in the calibration targets, MECS fuel_type
FUELS = {
    'natural-gas':    dict(label='Natural Gas',    target='CIMS.CAN.{R}.{S}.Methane Blend',
                           db=['natural-gas'],             calib=['Natural Gas'],              mecs='natural gas'),
    'heavy-fuel-oil': dict(label='Heavy Fuel Oil', target='CIMS.Generic Fuels.Heavy Fuel Oil',
                           db=['heavy-fuel-oil', 'oil'],   calib=['Heavy Fuel Oil'],           mecs='fuel oil'),
    'light-fuel-oil': dict(label='Light Fuel Oil', target='CIMS.Generic Fuels.Light Fuel Oil',
                           db=['gas-oil', 'oil'],          calib=['Light Fuel Oil'],           mecs='fuel oil'),
    'coal':           dict(label='Coal',           target='CIMS.Generic Fuels.Thermal Coal',
                           db=['coal'],                    calib=['Coal', 'Thermal Coal'],     mecs='coal'),
    'biomass':        dict(label='Solid Biomass',  target='CIMS.Generic Fuels.Solid Biomass',
                           db=['biomass', 'biomass-solid'], calib=['Solid Biomass'],           mecs='biomass'),
    # No LPG or petroleum coke data in the tech DB: fuel_use falls back to the
    # archetype's main-fuel energy use (fuel-fired boiler for LPG, solid-fuel
    # boiler for petroleum coke).
    'lpg':            dict(label='LPG',            target='CIMS.Generic Fuels.LPG',
                           db=['lpg', 'propane'],          calib=['LPG'],                      mecs='hgl'),
    'petroleum-coke': dict(label='Petroleum Coke', target='CIMS.Generic Fuels.Petroleum Coke',
                           db=['petroleum-coke'],          calib=['Petroleum Coke'],           mecs='petroleum coke'),
    'hydrogen':       dict(label='Hydrogen',       target='CIMS.CAN.{R}.Hydrogen',
                           db=['hydrogen'],                calib=['Hydrogen'],                 mecs='hydrogen'),
    'electricity':    dict(label='Electricity',    target='CIMS.CAN.{R}.Electricity',
                           db=['electricity'],             calib=['Electricity'],              mecs='electricity'),
}
ELECTRICITY = 'CIMS.CAN.{R}.Electricity'
VOM         = 'CIMS.Generic Fuels.VOM'
CCS_NODE    = 'CIMS.CAN.{R}.{S}.CCS'
# Tech DB costs are 2025 CAD. A dollar-year Unit makes CIMS convert them to the
# scenario's dollar year (unit_conversion.normalize_currency_in_df).
CURRENCY    = '2025_CAD'

# (CIMS technology name, archetype, fuel key, first year available)
# Available years: TRL 9 = 1930 as in the old fixed data; industrial heat
# pumps / MVR 2000; very-high-temperature heat pumps (TRL 7-8) 2025;
# post-combustion capture (TRL 5-8) 2030.
STEAM_TECHS = [
    ('Boiler - Natural Gas',                     'boiler_fuel-fired',                       'natural-gas',    1930),
    ('Boiler - Heavy Fuel Oil',                  'boiler_fuel-fired',                       'heavy-fuel-oil', 1930),
    ('Boiler - Light Fuel Oil',                  'boiler_fuel-fired',                       'light-fuel-oil', 1930),
    ('Boiler - LPG',                             'boiler_fuel-fired',                       'lpg',            1930),
    ('Boiler - Hydrogen',                        'boiler_fuel-fired',                       'hydrogen',       2025),
    ('Efficient Boiler - Natural Gas',           'boiler_fuel-fired_efficiency-moderate',   'natural-gas',    1930),
    ('Efficient Boiler - Heavy Fuel Oil',        'boiler_fuel-fired_efficiency-moderate',   'heavy-fuel-oil', 1930),
    ('Efficient Boiler - Light Fuel Oil',        'boiler_fuel-fired_efficiency-moderate',   'light-fuel-oil', 1930),
    ('Boiler CCS - Natural Gas',                 'boiler_fuel-fired_ccs-diffuse',           'natural-gas',    2030),
    ('Solid Fuel Boiler - Solid Biomass',        'boiler_solid-fuel',                       'biomass',        1930),
    ('Solid Fuel Boiler - Coal',                 'boiler_solid-fuel',                       'coal',           1930),
    ('Solid Fuel Boiler - Petroleum Coke',       'boiler_solid-fuel',                       'petroleum-coke', 1930),
    ('Electric Boiler',                          'boiler_electric',                         'electricity',    1930),
    ('Gas Turbine CHP - Natural Gas',            'combustion-turbine_chp',                  'natural-gas',    1930),
    ('Gas Turbine CHP CCS - Natural Gas',        'combustion-turbine_chp_ccs-diffuse',      'natural-gas',    2030),
    ('Steam Turbine CHP - Natural Gas',          'steam-turbine_back-pressure',             'natural-gas',    1930),
    ('Steam Turbine CHP - Heavy Fuel Oil',       'steam-turbine_back-pressure',             'heavy-fuel-oil', 1930),
    ('Steam Turbine CHP - Coal',                 'steam-turbine_back-pressure',             'coal',           1930),
    ('Steam Turbine CHP - Solid Biomass',        'steam-turbine_back-pressure',             'biomass',        1930),
    ('Steam Turbine CHP CCS - Natural Gas',      'steam-turbine_back-pressure_ccs-diffuse', 'natural-gas',    2030),
    ('Steam Turbine CHP CCS - Solid Biomass',    'steam-turbine_back-pressure_ccs-diffuse', 'biomass',        2030),
    ('Reciprocating Engine CHP - Natural Gas',   'reciprocating-engine_chp',                'natural-gas',    1930),
    ('Industrial Heat Pump',                     'heat-pump_industrial',                    'electricity',    2000),
    ('High Temperature Heat Pump',               'heat-pump_very-high-temperature',         'electricity',    2025),
]

# Technologies that carry the historical stock: fuel -> boiler tech, CHP tech
STOCK_2000 = {
    'natural-gas':    ('Boiler - Natural Gas',               'Gas Turbine CHP - Natural Gas'),
    'heavy-fuel-oil': ('Boiler - Heavy Fuel Oil',            'Steam Turbine CHP - Heavy Fuel Oil'),
    'light-fuel-oil': ('Boiler - Light Fuel Oil',            None),
    'lpg':            ('Boiler - LPG',                       None),
    'coal':           ('Solid Fuel Boiler - Coal',           'Steam Turbine CHP - Coal'),
    'petroleum-coke': ('Solid Fuel Boiler - Petroleum Coke', None),
    'biomass':        ('Solid Fuel Boiler - Solid Biomass',  'Steam Turbine CHP - Solid Biomass'),
    'hydrogen':       ('Boiler - Hydrogen',                  None),
    'electricity':    ('Electric Boiler',                    None),
}

# CHP in the historical stock (2000 market_share_total and the historical
# calibration_market_share_total targets). On (2026-11-13): each fuel is split
# between its boiler and CHP technology by steam_chp_shares.chp_share: the
# MECS cogeneration share, capped by province with StatCan industrial
# generation (25-10-0028 / 25-10-0084) plus third-party plants at industrial
# hosts, and fitted to CEEDC industrial generation for Canada. Unscaled MECS
# implied 2-11x StatCan generation in ON, QC, SK and MB. False = all boilers.
CHP_IN_HISTORICAL_STOCK = True
CHP_SOURCE = ('EIA MECS cogeneration share, capped by StatCan industrial generation '
              '(25-10-0028, 25-10-0084) + third-party plants, fitted to CEEDC '
              'industrial generation')

# Market share targets use each node's fuel mix (share of each fuel in the
# .Steam fuel targets) smoothed with a centred rolling mean over this many
# years. The provincial mixes jump year to year (RESD suppressed-cell fills,
# the Pulp and Paper biomass residual after black liquor), which made the
# targets swing by up to 0.3-0.6 in a year. Each year keeps its own total
# fuel; only the split between fuels is smoothed. The .Steam
# calibration_quantity_requested targets are not changed. 1 = no smoothing.
FUEL_MIX_SMOOTH_YEARS = 5

# Efficient boilers in the historical stock: share of each fuel's (non-CHP)
# boiler steam output held by the Efficient Boiler variant, rising linearly
# from EFFICIENT_SHARE_START in EFFICIENT_YEARS[0] to EFFICIENT_SHARE_END in
# EFFICIENT_YEARS[1] (held flat outside). ASSUMPTION (2026-11-16, user's
# choice): no Canadian series gives industrial boiler stock by efficiency.
# Heat pumps and reciprocating-engine CHP stay at 0 historically.
EFFICIENT_BOILER = {
    'natural-gas':    'Efficient Boiler - Natural Gas',
    'heavy-fuel-oil': 'Efficient Boiler - Heavy Fuel Oil',
    'light-fuel-oil': 'Efficient Boiler - Light Fuel Oil',
}
EFFICIENT_YEARS = (2000, 2024)
EFFICIENT_SHARE_START = 0.0
EFFICIENT_SHARE_END = 0.20


def efficient_boiler_share(year: int) -> float:
    """Assumed Efficient Boiler share of a fuel's boiler steam in `year`."""
    y0, y1 = EFFICIENT_YEARS
    t = min(max((year - y0) / (y1 - y0), 0.0), 1.0)
    return EFFICIENT_SHARE_START + t * (EFFICIENT_SHARE_END - EFFICIENT_SHARE_START)

CCS_SUFFIX = '_ccs-diffuse'
EFFICIENT_SUFFIX = '_efficiency-moderate'


# ── technology database ───────────────────────────────────────────────────────

def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(TECHDB_DIR / name, encoding='utf-8-sig')


@lru_cache(maxsize=1)
def _techdb() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    archetypes = {a for _, a, _, _ in STEAM_TECHS}
    archetypes |= {a.removesuffix(CCS_SUFFIX) for a in archetypes}
    p = _read('archetype-parameter.csv')
    p = p[p['archetype'].isin(archetypes)].copy()
    p['target'] = p['target'].fillna('')
    alt = _read('archetype-alternative-carrier.csv')
    alt = alt[alt['archetype'].isin(archetypes) & (alt['relation'] == 'alternative fuel')]
    flow = _read('archetype-flow.csv')
    flow = flow[flow['archetype'].isin(archetypes) & (flow['direction'] == 'input')]
    return p, alt, flow


def _param(archetype: str, parameter: str, target: str = '', default=None) -> float | None:
    p, _, _ = _techdb()
    hit = p[(p['archetype'] == archetype) & (p['parameter'] == parameter) & (p['target'] == target)]
    if hit.empty or pd.isna(hit['central'].iloc[0]):
        if default is not None:
            return default
        raise KeyError(f'{archetype}: no {parameter}[{target}] in the technology database')
    return float(hit['central'].iloc[0])


def fuel_use(archetype: str, fuel: str) -> tuple[float, str]:
    """GJ of fuel per GJ of steam, and where the number came from."""
    p, alt, flow = _techdb()
    for name in FUELS[fuel]['db']:
        v = _param(archetype, 'specific-energy-consumption', name, default=float('nan'))
        if pd.notna(v):
            return v, f'specific-energy-consumption[{name}]'
    for name in FUELS[fuel]['db']:
        hit = alt[(alt['archetype'] == archetype) & (alt['carrier'] == name)]
        if not hit.empty:
            return float(hit['central'].iloc[0]), f'alternative-carrier[{name}]'
    # archetype-flow rows for blended archetypes hold fuel-mix weights (e.g.
    # 0.03 GJ/GJ of oil); only a value of at least 1 GJ/GJ is a fuel use.
    for name in FUELS[fuel]['db']:
        hit = flow[(flow['archetype'] == archetype) & (flow['commodity'] == name)
                   & (flow['central'] >= 1.0)]
        if not hit.empty:
            return float(hit['central'].iloc[0]), f'flow[{name}]'
    return _param(archetype, 'specific-energy-consumption', 'fuel'), 'specific-energy-consumption[fuel] (assumed same as main fuel)'


def _chp_electricity(archetype: str) -> float:
    """GJ of electricity per GJ of steam from a CHP archetype (0 for boilers)."""
    elec = _param(archetype, 'efficiency', 'electricity', default=0.0)
    if not elec:
        return 0.0
    return elec / _param(archetype, 'efficiency', 'heat')


@lru_cache(maxsize=None)
def tech_spec(archetype: str, fuel: str) -> dict:
    """Per-GJ-steam inputs and costs for one archetype x fuel."""
    ccs = archetype.endswith(CCS_SUFFIX)
    base = archetype.removesuffix(CCS_SUFFIX)
    parent = base.removesuffix(EFFICIENT_SUFFIX)
    output = _param(base, 'annual-output')
    vom = _param(base, 'opex-variable', default=float('nan'))
    if pd.isna(vom):    # efficiency variants carry no variable O&M: use the parent's
        vom = _param(parent, 'opex-variable', default=0.0)
    spec = {
        'archetype': archetype,
        'lifetime': _param(base, 'lifetime'),
        'output':   output,
        'fcc':      _param(base, 'capex') * output,
        'fom':      _param(base, 'opex-fixed', default=0.0) * output,
        'vom':      vom,
        'ccs_tco2': _param(archetype, 'emissions-captured', 'co2') if ccs else 0.0,
    }
    if fuel == 'electricity':
        spec['fuel'], spec['fuel_from'] = 0.0, ''
        spec['electricity'] = _param(base, 'specific-energy-consumption', 'electricity')
    else:
        spec['fuel'], spec['fuel_from'] = fuel_use(base, fuel)
        spec['electricity'] = (_param(base, 'specific-energy-consumption', 'electricity', default=0.0)
                               - _chp_electricity(base))
    return spec


# ── 2000 market shares ────────────────────────────────────────────────────────

def _chp_share(sector: str, region: str, fuel: str, year: int) -> float:
    """CHP share of a fuel's steam fuel (steam_chp_shares; imported here to avoid a cycle)."""
    from CIMS.data_processing.utils.steam_chp_shares import chp_share
    return chp_share(sector, region, fuel, year)


def _fuel_key(target: str) -> str | None:
    """Fuel key for a calibration Target such as 'CIMS.Generic Fuels.LPG', or
    None when no steam technology burns that fuel."""
    by_name = {c: k for k, f in FUELS.items() for c in f['calib']}
    return by_name.get(str(target).split('.')[-1])


def steam_fuel_targets(calib: pd.DataFrame, sector: str, region: str) -> dict[int, dict[str, float]]:
    """{year: {fuel key: GJ}} from the .Steam calibration_quantity_requested rows.

    Blank and zero targets are skipped. A target for a fuel with no steam
    technology (no entry in FUELS, e.g. Coke, Kerosene, or Black Liquor) is
    dropped with a printed warning that gives the PJ dropped, since that steam
    is then missing from the market shares and CHP claims built from these
    targets. (Printed, not warnings.warn: some pipeline modules switch Python
    warnings off at import.)
    """
    df = calib[(calib['Branch'] == f'CIMS.CAN.{region}.{sector}.Steam')
               & (calib['Parameter'] == 'calibration_quantity_requested')]
    out: dict[int, dict[str, float]] = {}
    dropped: dict[str, list] = {}
    for year, target, value in zip(df['Year'], df['Target'], df['Value']):
        value = pd.to_numeric(value, errors='coerce')
        if pd.isna(value) or value == 0:
            continue
        key = _fuel_key(target)
        if key is None:
            d = dropped.setdefault(str(target).split('.')[-1], [0.0, set()])
            d[0] += float(value)
            d[1].add(int(year))
            continue
        fuels = out.setdefault(int(year), {})
        fuels[key] = fuels.get(key, 0.0) + float(value)
    for fuel, (gj, years) in sorted(dropped.items()):
        print(f'  Warning: {sector} {region}: dropped .Steam target for {fuel!r} '
              f'(no steam technology burns it): {gj / 1e6:,.3f} PJ over {min(years)}-{max(years)}')
    return out


def smooth_fuel_mix(by_year: dict[int, dict[str, float]],
                    window: int = FUEL_MIX_SMOOTH_YEARS) -> dict[int, dict[str, float]]:
    """{year: {fuel: GJ}} with each year's fuel shares replaced by a centred
    rolling mean of the shares over `window` years; each year keeps its own
    total GJ. Years with no fuel are left out of the mean."""
    if window <= 1 or not by_year:
        return by_year
    gj = pd.DataFrame.from_dict(by_year, orient='index').fillna(0.0).sort_index()
    total = gj.sum(axis=1)
    gj = gj[total > 0]
    total = total[total > 0]
    shares = gj.div(total, axis=0).rolling(window, center=True, min_periods=1).mean()
    shares = shares.div(shares.sum(axis=1), axis=0)
    out = shares.mul(total, axis=0)
    return {int(y): {f: float(v) for f, v in row.items() if v > 0} for y, row in out.iterrows()}


@lru_cache(maxsize=None)
def _calibration_steam_by_year(sector: str, region: str) -> dict[int, dict[str, float]]:
    """{year: {fuel key: GJ}} .Steam fuel targets for one region, from the
    steam fuel targets stage (processed_data/steam/steam_fuel_targets.csv)."""
    from CIMS.data_processing.source.steam import steam_fuel_targets as _targets
    return steam_fuel_targets(_targets.load(), sector, region)


def _calibration_steam(sector: str, region: str, year: int) -> dict[str, float]:
    """.Steam fuel targets (GJ) by fuel key for one region and year."""
    return _calibration_steam_by_year(sector, region).get(year, {})


def _calibration_steam_2000(sector: str, region: str) -> dict[str, float]:
    """2000 .Steam fuel targets (GJ) by fuel key for one region."""
    return _calibration_steam(sector, region, 2000)


def offered(name: str, archetype: str, sector: str) -> bool:
    """Whether a steam technology goes on this sector's .Steam node."""
    if archetype.endswith(CCS_SUFFIX) and sector not in CCS_SECTORS:
        return False
    return sector in TECH_SECTORS.get(name, {sector})


def stock_shares(sector: str, fuel_gj: dict[str, float], region: str, year: int) -> dict[str, float]:
    """Steam output share by CIMS technology for a fuel mix (GJ by fuel key).

    Each fuel's GJ is turned into steam with its stock technology's fuel use
    (electricity: the electric boiler's electricity use). With
    CHP_IN_HISTORICAL_STOCK each fuel is first split between its boiler and
    CHP technology by that region and year's CHP share (steam_chp_shares).
    Gas and oil boiler steam is then split with the Efficient Boiler variant
    by efficient_boiler_share(year). Empty if no fuel.
    """
    names = {n: (a, f) for n, a, f, _ in STEAM_TECHS}
    steam: dict[str, float] = {}
    for fuel, gj in fuel_gj.items():
        if gj <= 0:
            continue
        boiler, chp = STOCK_2000[fuel]
        if not offered(boiler, names[boiler][0], sector):
            raise ValueError(f'{sector}: steam target has {fuel} but {boiler!r} is not '
                             'offered in this sector (see TECH_SECTORS)')
        chp_share = _chp_share(sector, region, fuel, year) if (chp and CHP_IN_HISTORICAL_STOCK) else 0.0
        for tech, part in ((boiler, 1 - chp_share), (chp, chp_share)):
            if tech and part > 0:
                spec = tech_spec(*names[tech])
                per_gj = spec['fuel'] if fuel != 'electricity' else spec['electricity']
                steam[tech] = steam.get(tech, 0.0) + gj * part / per_gj
        # Efficient boilers: move efficient_boiler_share(year) of the boiler's
        # steam to the efficient variant. The boiler fuel stays the same, so
        # the boiler steam is recomputed at the blended fuel use.
        eff = EFFICIENT_BOILER.get(fuel)
        e = efficient_boiler_share(year) if eff and offered(eff, names[eff][0], sector) else 0.0
        if e > 0 and steam.get(boiler, 0.0) > 0:
            boiler_gj = gj * (1 - chp_share)
            f_std = tech_spec(*names[boiler])['fuel']
            f_eff = tech_spec(*names[eff])['fuel']
            boiler_steam = boiler_gj / ((1 - e) * f_std + e * f_eff)
            steam[boiler] = steam[boiler] - boiler_gj / f_std + (1 - e) * boiler_steam
            steam[eff] = steam.get(eff, 0.0) + e * boiler_steam
    total = sum(steam.values())
    return {t: v / total for t, v in steam.items()} if total > 0 else {}


def market_shares_2000(sector: str, region: str) -> dict[str, float]:
    """market_share_total for 2000 by CIMS technology name (sums to 1)."""
    fuels = smooth_fuel_mix(_calibration_steam_by_year(sector, region)).get(2000, {})
    return stock_shares(sector, fuels, region, 2000) or {'Boiler - Natural Gas': 1.0}


def steam_market_share_targets(calib: pl.DataFrame, sector: str) -> pl.DataFrame:
    """calibration_market_share_total rows for every .Steam node and year.

    calib is a sector's calibration output (OUTPUT_COLS). For each region and
    year with .Steam fuel targets, every technology available that year gets a
    row: its share of steam output (stock_shares), 0 if it holds no stock.
    The fuel mix is smoothed first (smooth_fuel_mix, FUEL_MIX_SMOOTH_YEARS).
    """
    pdf = calib.to_pandas()
    pdf['Year'] = pd.to_numeric(pdf['Year'], errors='coerce')
    pdf['Value'] = pd.to_numeric(pdf['Value'], errors='coerce')
    source = (f'.Steam calibration_quantity_requested by fuel ({FUEL_MIX_SMOOTH_YEARS}-yr mean fuel mix) '
              '/ technology fuel use; efficient boilers 0-20% of gas/oil boiler steam 2000-24 (assumed); CHP split: '
              + (CHP_SOURCE if CHP_IN_HISTORICAL_STOCK else 'none (boilers only)'))
    rows = []
    for region in sorted(pdf['Region'].dropna().unique()):
        branch = f'CIMS.CAN.{region}.{sector}.Steam'
        for year, fuels in sorted(smooth_fuel_mix(steam_fuel_targets(pdf, sector, region)).items()):
            shares = stock_shares(sector, fuels, region, year)
            if not shares:
                continue
            for name, archetype, _, available in STEAM_TECHS:
                if available > year or not offered(name, archetype, sector):
                    continue
                rows.append(_row(branch, region, sector, name, 'calibration_market_share_total',
                                 shares.get(name, 0.0), '%', source=source, year=str(year)))
    if not rows:
        return pl.DataFrame(schema={c: pl.String for c in calib.columns})
    out = pl.DataFrame(rows, schema={c: pl.String for c in _row('', '', '', '', '', None)})
    return out.select(calib.columns).cast(calib.schema, strict=False)


def add_steam_market_share_targets(calib: pl.DataFrame, sector: str) -> pl.DataFrame:
    """Append steam_market_share_targets to a sector's calibration output."""
    targets = steam_market_share_targets(calib, sector)
    print(f'  Steam: {targets.height:,} calibration_market_share_total rows '
          f'({targets["Region"].n_unique()} regions)')
    return pl.concat([calib, targets], how='vertical_relaxed')


# ── CIMS rows ─────────────────────────────────────────────────────────────────

def _row(branch, region, sector, tech, parameter, value, unit='', target='', source='', year=None):
    return {
        'Branch': branch, 'Type': 'Service', 'Region': region, 'Sector': sector,
        'Service': 'Steam', 'Technology': tech, 'Parameter': parameter,
        'Context': None, 'Sub_Context': None, 'Target': target or None,
        'Source': source or None, 'Unit': unit or None,
        'Year': year, 'Value': None if value is None else str(value),
    }


def build_steam_rows(sector: str, region: str) -> list[dict]:
    """All technology rows for CIMS.CAN.{region}.{sector}.Steam."""
    branch = f'CIMS.CAN.{region}.{sector}.Steam'
    fmt = dict(R=region, S=sector)
    shares = market_shares_2000(sector, region)
    ms_source = ('2000 steam calibration fuel / technology fuel use; CHP split: '
                 + (CHP_SOURCE if CHP_IN_HISTORICAL_STOCK else 'none (boilers only)'))
    rows = []
    for name, archetype, fuel, available in STEAM_TECHS:
        if not offered(name, archetype, sector):
            continue
        s = tech_spec(archetype, fuel)
        src = f'Technology database: {archetype}'
        r = lambda *a, **k: rows.append(_row(branch, region, sector, name, *a, **k))
        r('technology', None)
        r('available', available, 'Year', source=src)
        r('unavailable', 2101, 'Year', source=src)
        r('lifetime', s['lifetime'], 'Years', source=src)
        r('market_share_total', shares.get(name, 0.0), '%', source=ms_source, year='2000')
        r('output', s['output'], 'node unit', source=src)
        r('fcc', s['fcc'], CURRENCY, source=src)
        r('fom', s['fom'], CURRENCY, source=src)
        if s['fuel']:
            r('service_request', s['fuel'], 'GJ', FUELS[fuel]['target'].format(**fmt),
              f"{src} {s['fuel_from']}")
        if s['electricity']:
            r('service_request', s['electricity'], 'GJ', ELECTRICITY.format(**fmt), src)
        if s['vom']:
            r('service_request', s['vom'], CURRENCY, VOM, src)
        if s['ccs_tco2']:
            r('service_request', s['ccs_tco2'], 'tCO2', CCS_NODE.format(**fmt),
              f'{src} emissions-captured')
    return rows


def _is_old_steam_tech(sector: str) -> pl.Expr:
    """Technology rows on CIMS.CAN.{R}.{sector}.Steam (not its sub-nodes)."""
    return (pl.col('Branch').str.starts_with('CIMS.CAN.')
            & pl.col('Branch').str.ends_with(f'.{sector}.Steam')
            & (pl.col('Branch').str.count_matches(r'\.') == 4)
            & (pl.col('Technology').fill_null('') != ''))


def replace_steam_techs(fixed: pl.DataFrame, sector: str) -> pl.DataFrame:
    """Swap the .Steam technologies in one sector's flattened fixed data.

    Each region's new rows go where that region's old .Steam technology
    block started, so the node header rows stay ahead of the technologies.
    """
    indexed = fixed.cast(pl.String).with_row_index('_i').with_columns(
        pl.col('_i').cast(pl.Float64), pl.lit(0).alias('_j'))
    old = indexed.filter(_is_old_steam_tech(sector))
    if old.is_empty():
        return fixed
    first = old.group_by('Region').agg(pl.col('_i').min())
    parts = []
    for region, i in first.sort('Region').iter_rows():
        rows = build_steam_rows(sector, region)
        parts.append(pl.DataFrame(rows, schema={c: pl.String for c in fixed.columns})
                     .with_columns(pl.lit(i).alias('_i'),
                                   pl.int_range(pl.len()).alias('_j')))
    new = pl.concat(parts)
    print(f'  Steam: {old.height:,} fixed-data technology rows replaced by '
          f'{new.height:,} technology database rows ({first.height} regions)')
    kept = indexed.filter(~_is_old_steam_tech(sector))
    out = (pl.concat([kept, new.select(kept.columns)], how='vertical_relaxed')
           .sort(['_i', '_j']).drop('_i', '_j'))
    return out


def drop_replaced_steam_rows(df: pl.DataFrame) -> pl.DataFrame:
    """Drop rows (limits, DIC, DCC, FIC) that name the replaced .Steam techs."""
    mask = pl.lit(False)
    for sector in SECTORS:
        mask = mask | _is_old_steam_tech(sector)
    return df.filter(~mask)


def summary() -> pd.DataFrame:
    """One row per technology: per-GJ inputs and costs (for review)."""
    rows = []
    for name, archetype, fuel, available in STEAM_TECHS:
        s = tech_spec(archetype, fuel)
        rows.append({'technology': name, 'available': available, **s})
    return pd.DataFrame(rows)


if __name__ == '__main__':
    pd.set_option('display.width', 250)
    print(summary().round(4).to_string())
    for sector in SECTORS:
        print(f'\n{sector}')
        for region in ['AB', 'BC', 'MB', 'NB', 'NL', 'NS', 'ON', 'PE', 'QC', 'SK']:
            ms = market_shares_2000(sector, region)
            print(f'  {region}: ' + ', '.join(f'{k} {v:.2f}' for k, v in ms.items()))
