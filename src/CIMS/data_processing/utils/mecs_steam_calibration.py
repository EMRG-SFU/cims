"""
Replace CER-based Steam node calibration targets with MECS-based targets.

Background
----------
cer_to_cims_map.csv sends each sector's whole CER 'Process Heat' end use to
the sector's .Steam node (Chemical Products, Iron and Steel, Metal Smelting,
Pulp and Paper). Process heat also includes direct-fired heat (furnaces,
kilns, heaters), so that target overstates steam. Light Industrial and
Petroleum Refining get no Steam target at all.

What this does
--------------
For one sector, on the cer_resd_demand output (Region | Node | Variable |
Unit | Source | Year | Value, GJ):

1. Drops any existing rows for Node == '{sector}.Steam'.
2. Adds new rows for '{sector}.Steam':
     steam fuel   = MECS steam_share_fuel × sector energy demand, split across
                    fuels by:
                      RESD_MIX_SECTORS: the RESD steam fuel shares
                        (resd_steam_fuel_shares.py: boiler/CHP fuels only,
                        suppressed cells filled, fuels <1% dropped);
                        capped at the sector's CER combustion fuel total
                      other sectors (Pulp and Paper, Petroleum Refining):
                        the sector's CER Process Heat combustion fuels (falls
                        back to all combustion fuels if it has no Process
                        Heat); capped at that fuel total
     electricity  = MECS steam_share_electricity × sector energy demand
   'Sector energy demand' is the CER general demand for the sector
   (vDmd + vDmd_OG), excluding feedstock (vFsDmd) — the same basis as MECS.

   Pulp and Paper (RECOVERY_FURNACE_SECTORS, BIOMASS_FIRST_SECTORS): the
   MECS steam fuel target counts pulping liquor, but in CIMS the liquor is
   burned at the Recovery Furnace node, which supplies steam to the Steam
   node through a negative service request. So the sector's Black Liquor
   (StatCan SPL) is subtracted from the target, Black Liquor is never put on
   the Steam node, and the rest of the target is filled with Solid Biomass
   (StatCan wood waste) first, then the RESD fossil mix.

   Light Industrial (BIOMASS_FIRST_SECTORS): same as Pulp and Paper without
   the liquor step. The MECS steam fuel target is filled first with the
   provincial BIOMASS_STEAM_SHARE of the wood products wood waste (StatCan;
   BC 50%, AB 0%, other provinces 100%), then the RESD fossil mix. The rest of
   the wood waste stays on the .Light Industrial sector node (direct-fired
   kilns and dryers). These shares are a provisional starting point; see the
   note at BIOMASS_STEAM_SHARE.

   Petroleum Refining (STILL_GAS_SECTORS): the target is split by the
   province's mix of refinery boiler fuels (REFINING_STEAM_FUELS) from
   stats_can/refinery_fuel.py (StatCan RESD; same approach as other sectors).
   The main .Steam node has no still gas technology, so the still gas share
   goes to '.Petroleum Refining.Still Gas.Steam' (whose boilers credit .Steam
   in the model) and the rest of the still gas to '.Still Gas.Heater'. Other
   fuels go to .Steam. Fuels whose share of the boiler-fuel mix never reaches
   1% in any year for a region are dropped and the rest prorated (as in the
   RESD steam fuel shares). Non-energy products are left out of the sector
   total.

   Biomass in the basis comes from stats_can/biomass_demand.py
   (replace_cer_biomass), the same numbers as cer_resd_demand.

The sector node's own rows are left unchanged: the sector node is a rollup
of its subtree, so its totals already include the Steam node.

Usage (in a sector calibration.py, right after cer_df = _cer_mod.main()):
    from CIMS.data_processing.utils.mecs_steam_calibration import apply_mecs_steam
    cer_df = apply_mecs_steam(cer_df, '.Chemical Products')
"""

from functools import lru_cache

import pandas as pd

import CIMS.data_processing.source.cer.cer_resd_demand as _cer_mod
import CIMS.data_processing.source.eia.mecs_steam_shares as _mecs_mod
from CIMS.data_processing.source.stats_can.resd_steam_fuel_shares import load_steam_fuel_shares
from CIMS.data_processing.source.stats_can.biomass_demand import replace_cer_biomass
from CIMS.data_processing.source.stats_can.refinery_fuel import replace_cer_refinery_fuel

SOURCE_LABEL      = 'EIA MECS share x CER/RESD'
SOURCE_LABEL_RESD = 'EIA MECS share x CER/RESD; fuel split RESD (filled)'
_PH_SUFFIX   = '.__process_heat__'
_NON_COMBUSTION = {'Electricity'}

# Sectors whose Steam fuel split comes from the RESD steam fuel shares.
RESD_MIX_SECTORS = {'.Chemical Products', '.Iron and Steel', '.Metal Smelting', '.Light Industrial',
                    '.Pulp and Paper'}

# Sectors where CER biomass goes to steam first (RESD has no provincial
# biomass). All of the sector's CER biomass counts as steam fuel, up to the
# MECS steam fuel target; the rest of the target is split by the RESD fossil
# mix. Mills burn essentially all their wood and black liquor in boilers.
BIOMASS_FIRST_SECTORS = {'.Pulp and Paper', '.Light Industrial'}

# Sectors whose steam is split by their own (StatCan refinery) boiler-fuel mix,
# with still gas sent to the Still Gas nodes (the main .Steam node cannot burn it).
STILL_GAS_SECTORS = {'.Petroleum Refining'}
STILL_GAS = 'Refinery Fuel Gas'
REFINING_STEAM_FUELS = ['Refinery Fuel Gas', 'Natural Gas', 'Petroleum Coke', 'Heavy Fuel Oil', 'LPG']
MIN_SHARE_REFINING = 0.01   # drop a refinery boiler fuel whose share of the boiler-fuel
                            # mix never reaches this in any year for that region;
                            # prorate the rest (same rule as resd_steam_fuel_shares)
NON_ENERGY = {'Other Non-Energy Products', 'Asphalt', 'Lubricants', 'Petrochemical Feedstock',
              'Naphtha Specialties'}

# Share of each sector's biomass that is steam fuel (the rest is direct heat).
#   Pulp and Paper: 1.0. MECS puts ~99% of forest products biomass in boilers
#     and cogeneration.
#   Light Industrial (wood products wood waste), by province. PROVISIONAL:
#     a starting point from a general understanding of the industry, not a
#     published source.
#       BC 0.5 : BC Interior mills mostly use direct-fired kilns (gas burners,
#                or wood-residue burners/gasifiers feeding hot gas to the
#                kiln); coastal BC mills more often use hog-fuel boilers.
#       AB 0.0 : mostly direct-fired kilns, as in the BC Interior.
#       others 1.0 : QC, ON and Atlantic mills more often heat kilns with
#                steam or thermal oil from hog-fuel boilers (older mills, and
#                mills next to a pulp mill or cogeneration plant).
#     Why provisional: no source reports how wood products wood waste is
#     split between steam and direct heat, nationally or by province. StatCan
#     25-10-0031 has no end uses; CEEDC has no provincial wood products
#     energy data since 1996; US MECS reports no end uses for wood; CIPEC
#     Table 3-3 (softwood lumber kilns: 69% hog fuel, 20% gas, 11% purchased
#     steam) does not split kilns by type or region. The shares also treat
#     all wood products like sawmills, although CEEDC (2017) puts only ~37% of
#     the sector's wood at sawmills; panel plants (OSB, MDF, particleboard)
#     mostly direct-fire their dryers.
#     More research needed: kiln and dryer heating type by mill and province
#     (FPInnovations; NRCan's "Status of Energy Use in Canadian Wood Products
#     Sector", on request from info.ind@nrcan-rncan.gc.ca), and the split for
#     panel plants. Replace these shares when a source is found.
#   Note: the Light Industrial MECS share is built from food, textiles,
#     plastics, machinery and transport equipment; the MECS footprint used
#     for the steam shares groups wood products (321) with paper as 'forest
#     products', so wood products' fossil fuels get the Light Industrial
#     share (about 9% of Light Industrial fossil fuel, CEEDC 2020).
BIOMASS_STEAM_SHARE = {
    '.Pulp and Paper':   {'default': 1.0},
    '.Light Industrial': {'BC': 0.5, 'AB': 0.0, 'default': 1.0},
}


def _bio_steam_share(sector_node: str, region: str) -> float:
    """BIOMASS_STEAM_SHARE for a sector and region ('default' if not listed)."""
    shares = BIOMASS_STEAM_SHARE.get(sector_node, {})
    return shares.get(region, shares.get('default', 1.0))

# Sectors whose Black Liquor is burned at the Recovery Furnace node, which
# supplies steam through a negative service request. Their Black Liquor is
# subtracted from the MECS steam fuel target and not put on the Steam node.
RECOVERY_FURNACE_SECTORS = {'.Pulp and Paper'}
LIQUOR_FUELS = {'Black Liquor'}

# CER fuel names treated as biomass (matched case-insensitively on these words).
BIOMASS_KEYWORDS = ('biomass', 'wood', 'liquor', 'spl', 'hog', 'waste', 'bark')


def _is_biomass(fuels: pd.Series) -> pd.Series:
    low = fuels.str.lower()
    return low.apply(lambda f: any(k in f for k in BIOMASS_KEYWORDS))


def _resd_mix(sector_node: str):
    """{(region, year): Series fuel -> share} for one sector, plus its year range."""
    df = load_steam_fuel_shares()
    df = df[df['node'] == sector_node]
    mixes = {k: g.set_index('steam_fuel')['share'] for k, g in df.groupby(['region', 'year'])}
    return mixes, int(df['year'].min()), int(df['year'].max())


@lru_cache(maxsize=1)
def _load_shares() -> pd.DataFrame:
    """MECS steam shares (Node, Year → steam_share_fuel, steam_share_electricity)."""
    path = _mecs_mod.OUTPUT_FILE
    df = pd.read_csv(path) if path.exists() else _mecs_mod.main()
    return df.pivot_table(index=['Node', 'Year'], columns='Variable',
                          values='Value', aggfunc='first').reset_index()


@lru_cache(maxsize=1)
def _load_basis() -> pd.DataFrame:
    """
    CER general (non-feedstock) demand by Region / Node / fuel / Year, GJ,
    for each steam sector node, plus its Process Heat rows under the
    pseudo-node '{node}.__process_heat__'.
    """
    cer = _cer_mod.load_cer_data()
    cer = cer[cer['demand_type'] == 'general']
    mapping, energy_map, region_map = _cer_mod._load_mapping_tables()

    sector_rows = mapping[(mapping['cer_enduse'] == '')
                          & mapping['cims_node'].isin(_mecs_mod.CIMS_TO_MECS)]
    custom_map = pd.concat([
        sector_rows[['cer_sector', 'cer_enduse', 'cims_node']],
        sector_rows.assign(cer_enduse='process heat',
                           cims_node=sector_rows['cims_node'] + _PH_SUFFIX)
                   [['cer_sector', 'cer_enduse', 'cims_node']],
    ], ignore_index=True)

    merged = _cer_mod._map_to_cims(cer, custom_map, energy_map, region_map, verbose=False)
    return replace_cer_refinery_fuel(replace_cer_biomass(_cer_mod._aggregate(merged)))


def build_mecs_steam_rows(sector_node: str) -> pd.DataFrame:
    """MECS-based Steam node rows for one sector, in cer_resd_demand format."""
    steam_node = f'{sector_node}.Steam'
    basis  = _load_basis()
    shares = _load_shares()
    shares = shares[shares['Node'] == sector_node].set_index('Year')
    if shares.empty:
        raise KeyError(f'No MECS steam share for {sector_node}')

    sector = basis[basis['Node'] == sector_node]
    ph     = basis[basis['Node'] == sector_node + _PH_SUFFIX]

    if sector_node in STILL_GAS_SECTORS:
        sector = sector[~sector['Variable'].isin(NON_ENERGY)]
    totals  = sector.groupby(['Region', 'Year'])['Value'].sum()
    comb    = sector[~sector['Variable'].isin(_NON_COMBUSTION)]
    ph_comb = ph[~ph['Variable'].isin(_NON_COMBUSTION)]

    use_resd = sector_node in RESD_MIX_SECTORS
    if sector_node in STILL_GAS_SECTORS:
        bf = comb[comb['Variable'].isin(REFINING_STEAM_FUELS)].groupby(['Region', 'Year', 'Variable'])['Value'].sum()
        bshare = bf / bf.groupby(['Region', 'Year']).transform('sum')
        max_share = bshare.groupby(['Region', 'Variable']).max()
        keep_fuels = {r: set(g.index.get_level_values('Variable')[g.values >= MIN_SHARE_REFINING])
                      for r, g in max_share.groupby(level='Region')}
    bio_first = sector_node in BIOMASS_FIRST_SECTORS
    if use_resd:
        resd_mix, y_min, y_max = _resd_mix(sector_node)
    source = SOURCE_LABEL_RESD if use_resd else SOURCE_LABEL
    rf = sector_node in RECOVERY_FURNACE_SECTORS
    if rf:
        is_liq = comb['Variable'].isin(LIQUOR_FUELS)
        liq_all, comb = comb[is_liq], comb[~is_liq]
        source += '; less Black Liquor (Recovery Furnace)'
    if bio_first:
        source += '; biomass first (x BIOMASS_STEAM_SHARE)'
        is_bio = _is_biomass(comb['Variable'])
        bio_all, comb = comb[is_bio], comb[~is_bio]
        ph_comb = ph_comb[~_is_biomass(ph_comb['Variable'])]
        names = sorted(bio_all['Variable'].unique())
        print(f'  {steam_node}: biomass fuels found: {names if names else "NONE - check BIOMASS_KEYWORDS"}')

    rows, capped, fallback, all_bio, liq_over = [], [], [], [], []
    for (region, year), total in totals.items():
        if total <= 0 or year not in shares.index:
            continue
        s_fuel = float(shares.at[year, 'steam_share_fuel'])
        s_elec = float(shares.at[year, 'steam_share_electricity'])
        target = s_fuel * total

        # Recovery Furnace sectors: the liquor's steam reaches the Steam node
        # through the furnace's negative service request, so take it off
        if rf:
            liq = liq_all[(liq_all['Region'] == region) & (liq_all['Year'] == year)]['Value'].sum()
            if liq > target:
                liq_over.append((region, year))
            target = max(target - liq, 0.0)

        # Biomass first (BIOMASS_FIRST_SECTORS): CER biomass, up to the target
        if bio_first:
            bio = (bio_all[(bio_all['Region'] == region) & (bio_all['Year'] == year)]
                   .groupby('Variable')['Value'].sum())
            bio = bio[bio > 0] * _bio_steam_share(sector_node, region)
            bio_steam = min(bio.sum(), target)
            for fuel, v in (bio_steam * bio / bio.sum()).items() if bio.sum() > 0 else []:
                rows.append((region, fuel, year, v))
            target -= bio_steam

        # Fossil (or all) fuel steam: RESD mix where available, else CER Process Heat
        comb_ry = comb[(comb['Region'] == region) & (comb['Year'] == year)]
        mix = None
        if sector_node in STILL_GAS_SECTORS:
            mix = comb_ry[comb_ry['Variable'].isin(REFINING_STEAM_FUELS)].groupby('Variable')['Value'].sum()
            mix = mix[(mix > 0) & mix.index.isin(keep_fuels.get(region, set()))]
            cap = comb_ry[comb_ry['Variable'].isin(REFINING_STEAM_FUELS)]['Value'].sum()
        elif use_resd:
            # RESD years outside its range use the nearest year's mix
            mix = resd_mix.get((region, min(max(int(year), y_min), y_max)))
            if mix is None:
                fallback.append((region, year))
            else:
                mix = mix[mix > 0]
                cap = comb_ry['Value'].sum()
        if mix is None:
            mix = ph_comb[(ph_comb['Region'] == region) & (ph_comb['Year'] == year)]
            if mix['Value'].sum() <= 0:
                mix = comb_ry
            mix = mix.groupby('Variable')['Value'].sum()
            mix = mix[mix > 0]
            cap = mix.sum()

        if bio_first and target <= 0 and comb_ry['Value'].sum() > 0:
            all_bio.append((region, year))
        if mix.sum() > 0 and target > 0:
            if target > cap:
                capped.append((region, year, target, cap))
                target = cap
            for fuel, v in (target * mix / mix.sum()).items():
                rows.append((region, fuel, year, v))
        if s_elec > 0:
            rows.append((region, 'Electricity', year, s_elec * total))

    if fallback:
        regions = sorted({r for r, _ in fallback})
        print(f'  Note: {steam_node} has no RESD fuel mix for {", ".join(regions)}; '
              f'used the CER Process Heat mix there ({len(fallback)} region-years)')
    if all_bio:
        regions = sorted({r for r, _ in all_bio})
        print(f'  Note: {steam_node} biomass alone meets the MECS steam fuel target in '
              f'{", ".join(regions)} ({len(all_bio)} region-years); no fossil steam there')

    if liq_over:
        regions = sorted({r for r, _ in liq_over})
        print(f'  Note: {steam_node} Black Liquor exceeds the MECS steam fuel target in '
              f'{", ".join(regions)} ({len(liq_over)} region-years); Steam node fuel set to 0 there')

    if capped:
        print(f'  Note: {steam_node} steam target capped at available combustion '
              f'fuel in {len(capped)} region-years (e.g. {capped[0][0]} {capped[0][1]})')

    out = pd.DataFrame(rows, columns=['Region', 'Variable', 'Year', 'Value'])
    out.insert(1, 'Node', steam_node)
    if sector_node in STILL_GAS_SECTORS:
        is_sg = out['Variable'] == STILL_GAS
        out.loc[is_sg, 'Node'] = f'{sector_node}.Still Gas.Steam'
        sg_tot = comb[comb['Variable'] == STILL_GAS].groupby(['Region', 'Year'])['Value'].sum()
        sg_steam = out[is_sg].groupby(['Region', 'Year'])['Value'].sum().reindex(sg_tot.index, fill_value=0)
        heat = (sg_tot - sg_steam).clip(lower=0).reset_index()
        heat = heat[heat['Value'] > 0]
        heat.insert(1, 'Node', f'{sector_node}.Still Gas.Heater')
        heat.insert(2, 'Variable', STILL_GAS)
        out = pd.concat([out, heat[['Region', 'Node', 'Variable', 'Year', 'Value']]], ignore_index=True)
    out.insert(3, 'Unit', 'GJ')
    out.insert(4, 'Source', source)
    return out[['Region', 'Node', 'Variable', 'Unit', 'Source', 'Year', 'Value']]


def apply_mecs_steam(cer_df: pd.DataFrame, sector_node: str) -> pd.DataFrame:
    """Swap the CER Process-Heat-based Steam rows for MECS-based ones."""
    steam_node = f'{sector_node}.Steam'
    nodes = {steam_node}
    if sector_node in STILL_GAS_SECTORS:
        nodes |= {f'{sector_node}.Still Gas.Steam', f'{sector_node}.Still Gas.Heater'}
    drop = cer_df['Node'].isin(nodes)
    new_rows = build_mecs_steam_rows(sector_node)
    print(f'  MECS steam: replaced {int(drop.sum()):,} CER rows at {", ".join(sorted(nodes))} '
          f'with {len(new_rows):,} MECS-based rows')
    return pd.concat([cer_df[~drop], new_rows], ignore_index=True)
