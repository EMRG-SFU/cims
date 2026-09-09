"""
feedstock_demand.py — Shared Feedstock Service-Block Builder
==============================================================

Builds the CIMS rows for a sector's "Feedstock" service: a single
non-competing, Fixed-Ratio service that reproduces historical CER feedstock
demand (vFsDmd-CIMS.csv, via pipeline/source/cer/cer_resd_demand.py's
load_feedstock_demand()) exactly and projects future years from ratios held
flat at their recent average.

Structure written (mirrors Commercial -> Buildings' existing "Fixed Ratio"
grouping node -- see raw_data/fixed_data/commercial/commercial_ab.csv lines
5-10 -- and Buildings' own fan-out to Lighting/Refrigeration/etc., lines
7-12: a Fixed-Ratio service requesting each of several targets directly,
with no technology layer, since there's no real technology choice here --
every feedstock fuel's share is a fact about historical fuel use, not a
cost-competed decision):

    <sector_branch>          ,service_request -> .Feedstock   (ratio = total feedstock GJ / scale, per year)
    <sector_branch>.Feedstock,service_provide (Unit=GJ) / competition=Fixed Ratio
    <sector_branch>.Feedstock,service_request -> Generic Fuels.<fuel>   (one row per fuel, per year)

The Sector -> Feedstock ratio is what carries the total historical demand:
for a historical year it is exactly `total_feedstock_value / scale_series
[year]` (summed across every fuel that year), which multiplies back out to
the exact CER total once CIMS applies it against that year's real
scale_series (e.g. floorspace) -- so Feedstock's own assessed demand is a
real, meaningful "total feedstock GJ" quantity, not a placeholder. Feedstock's
own per-fuel service_request rows then carry that fuel's SHARE of the total
for the year (fuel_value / total_value), so share * Feedstock's assessed
demand reproduces each fuel's exact historical value. For years after
`last_hist_year`, both the Sector -> Feedstock ratio and each fuel's share
use the average of their respective historical values over the last
`ratio_window` years, so demand keeps rising and falling with scale_series's
own projection while each fuel keeps roughly its recent share of the total.

Deliberately generic: nothing here is Commercial- or floorspace-specific.
Any sector's model_inputs.py can call build_feedstock_rows() with its own
sector Branch/name, its own natural scale driver (floorspace, activity,
output...), and its own slice of load_feedstock_demand()'s output.
"""

import pandas as pd
import polars as pl

OUTPUT_COLS = [
    'Branch', 'Type', 'Region', 'Sector', 'Service', 'Technology',
    'Parameter', 'Context', 'Sub_Context', 'Target', 'Source', 'Unit',
    'Year', 'Value',
]


def _empty_frame() -> pl.DataFrame:
    return pl.DataFrame({c: pl.Series([], dtype=pl.Utf8) for c in OUTPUT_COLS + ['_order']})


def _trailing_average(series: dict[int, float], last_hist_year: int, window: int) -> float | None:
    """Average of `series`'s last `window` historical years (<= last_hist_year) with data."""
    hist_years = sorted(y for y in series if y <= last_hist_year)
    window_years = hist_years[-window:] if hist_years else []
    if not window_years:
        return None
    return sum(series[y] for y in window_years) / len(window_years)


def build_feedstock_rows(
    sector_branch: str,
    sector_name: str,
    region: str,
    feedstock: pd.DataFrame,
    scale_series: pd.Series,
    scale_unit: str,
    last_hist_year: int,
    ratio_window: int = 5,
    fuel_target: dict[str, str] | None = None,
    node_name: str = 'Feedstock',
    start_order: float = 0.0,
) -> pl.DataFrame:
    """
    Build the Feedstock service-block rows for one sector/region.

    Parameters
    ----------
    sector_branch : str
        e.g. 'CIMS.CAN.AB.Commercial' -- the Sector node the Feedstock
        block hangs off of. Must already exist in the sector's own fixed
        data with its own service_provide/competition rows; this function
        only ADDS one more service_request row there.
    sector_name : str
        e.g. 'Commercial' -- written to the Sector column of every row.
    region : str
        CIMS region code, e.g. 'AB'.
    feedstock : pd.DataFrame
        This sector+region's slice of cer_resd_demand.load_feedstock_demand()
        -- must have at least 'Variable' (fuel name), 'Year', 'Value'
        columns. Caller is responsible for filtering to the right Region
        and Node before calling.
    scale_series : pd.Series
        Year (int) -> this region's natural scale driver for the sector
        (e.g. total_floorspace in m2 for Commercial/Residential). Must
        cover both the historical and projection years the output should
        span; years missing here get no feedstock rows.
    scale_unit : str
        Unit of scale_series, e.g. 'm2' -- used only for the Sector ->
        Feedstock row's Unit column (matches how Commercial -> Buildings
        already works, whose Unit records the ratio's denominator).
    last_hist_year : int
        Years at or before this use the exact CER-implied ratio/share;
        years after it use the trailing average (see ratio_window).
    ratio_window : int
        How many of the most recent historical years (with data) to
        average for the projected ratio/share. Modifiable per sector/run.
    fuel_target : dict[str, str] or None
        CIMS fuel name -> Generic Fuels node name, for fuels whose CIMS
        fuel name differs from their Generic Fuels target. Defaults to
        identity (fuel name used as-is).
    node_name : str
        Display name of the Feedstock service node. Default 'Feedstock'.
    start_order : float
        `_order` anchor for the whole block; every row in the returned
        frame gets an `_order` at or after this value, so the caller can
        anchor it (e.g. after everything else in the sector's fixed data).

    Returns
    -------
    pl.DataFrame
        Rows in OUTPUT_COLS + '_order' shape, ready to concat into the
        sector's own assembled output.
    """
    if feedstock is None or feedstock.empty:
        return _empty_frame()

    all_years = sorted(int(y) for y in scale_series.dropna().index)
    if not all_years:
        return _empty_frame()

    scale_branch = f'{sector_branch}.{node_name}'
    fuels = sorted(feedstock['Variable'].dropna().unique())

    fdata = (
        feedstock
        .assign(Year=lambda d: d['Year'].astype(int), Value=lambda d: d['Value'].astype(float))
    )

    # -- per-fuel, per-year value, and each year's total across fuels ---------
    by_fuel: dict[str, dict[int, float]] = {
        fuel: dict(zip(sub['Year'], sub['Value']))
        for fuel, sub in fdata.groupby('Variable')
    }
    total_by_year: dict[int, float] = {}
    for fuel_years in by_fuel.values():
        for year, value in fuel_years.items():
            total_by_year[year] = total_by_year.get(year, 0.0) + value

    # -- Sector -> Feedstock: total-feedstock-per-scale ratio, one per year ---
    total_ratio: dict[int, float] = {}
    for year, value in total_by_year.items():
        denom = scale_series.get(year)
        if denom is None or pd.isna(denom) or not denom:
            continue
        total_ratio[year] = value / float(denom)

    if not total_ratio:
        return _empty_frame()

    avg_total_ratio = _trailing_average(total_ratio, last_hist_year, ratio_window)

    rows: list[dict] = []
    for year in all_years:
        if year <= last_hist_year and year in total_ratio:
            value, source = total_ratio[year], 'CER/RESD'
        elif year > last_hist_year and avg_total_ratio is not None:
            value, source = avg_total_ratio, 'Assumptions'
        else:
            continue
        rows.append({
            'Branch': sector_branch, 'Type': 'Service', 'Region': region,
            'Sector': sector_name, 'Service': '', 'Technology': '',
            'Parameter': 'service_request', 'Context': '', 'Sub_Context': '',
            'Target': scale_branch, 'Source': source, 'Unit': scale_unit,
            'Year': str(year), 'Value': str(value), '_order': start_order,
        })

    # ── Feedstock service header: Fixed Ratio, no technology layer -- there's
    # no real technology choice here, just a fixed (CER-implied) fan-out to
    # each fuel, exactly like Commercial -> Buildings' own header. ──────────
    rows.append({
        'Branch': scale_branch, 'Type': 'Service', 'Region': region,
        'Sector': sector_name, 'Service': node_name, 'Technology': '',
        'Parameter': 'service_provide', 'Context': '', 'Sub_Context': '',
        'Target': '', 'Source': '', 'Unit': 'GJ',
        'Year': '', 'Value': '', '_order': start_order + 0.001,
    })
    rows.append({
        'Branch': scale_branch, 'Type': 'Service', 'Region': region,
        'Sector': sector_name, 'Service': node_name, 'Technology': '',
        'Parameter': 'competition', 'Context': '', 'Sub_Context': '',
        'Target': '', 'Source': '', 'Unit': '',
        'Year': '', 'Value': 'Fixed Ratio', '_order': start_order + 0.002,
    })

    # ── one service_request row per fuel, per year: that fuel's share ──────
    for fi, fuel in enumerate(fuels):
        target_fuel = (fuel_target or {}).get(fuel, fuel)
        target_node = f'CIMS.Generic Fuels.{target_fuel}'
        fuel_order = start_order + 0.003 + 0.0001 * fi

        share: dict[int, float] = {}
        for year, value in by_fuel[fuel].items():
            total = total_by_year.get(year)
            if not total:
                continue
            share[year] = value / total

        avg_share = _trailing_average(share, last_hist_year, ratio_window)

        for year in all_years:
            if year <= last_hist_year and year in share:
                value, source = share[year], 'CER/RESD'
            elif year > last_hist_year and avg_share is not None:
                value, source = avg_share, 'Assumptions'
            else:
                continue
            rows.append({
                'Branch': scale_branch, 'Type': 'Service', 'Region': region,
                'Sector': sector_name, 'Service': node_name, 'Technology': '',
                'Parameter': 'service_request', 'Context': '', 'Sub_Context': '',
                'Target': target_node, 'Source': source, 'Unit': 'GJ',
                'Year': str(year), 'Value': str(value), '_order': fuel_order,
            })

    return pl.DataFrame(rows) if rows else _empty_frame()


def build_feedstock_rows_all_regions(
    sector_name: str,
    feedstock_demand: pd.DataFrame,
    scale_by_region: dict[str, pd.Series],
    scale_unit: str,
    ratio_window: int = 5,
    **kwargs,
) -> pl.DataFrame:
    """
    Call build_feedstock_rows() once per region, concatenating the results.

    For sectors whose model_inputs.py has a single top-level
    `CIMS.CAN.{region}.<Sector>` branch and a pre-existing, already-
    CAGR-extended per-region scale driver (the common shape almost every
    sector besides Commercial has), this collapses the "loop over regions,
    build a scale_series, call build_feedstock_rows(), concat" boilerplate
    that would otherwise be repeated identically in every sector's
    model_inputs.py.

    Parameters
    ----------
    sector_name : str
        e.g. 'Iron and Steel' -- used to build each region's
        `CIMS.CAN.{region}.{sector_name}` sector_branch and written to the
        Sector column.
    feedstock_demand : pd.DataFrame
        This sector's slice of cer_resd_demand.load_feedstock_demand()
        (already filtered to the sector's own CIMS node(s), and already
        summed across nodes if the sector maps to more than one -- see
        Petroleum Crude). Must have 'Region', 'Variable', 'Year', 'Value'.
    scale_by_region : dict[str, pd.Series]
        Region code -> Year (int) -> scale value, one entry per region this
        sector models. Regions absent here, or present but empty of
        feedstock data, are silently skipped (no Feedstock rows written for
        them) rather than erroring.
    scale_unit, ratio_window, **kwargs
        Passed through to build_feedstock_rows() for every region (e.g.
        fuel_target, node_name, start_order).

    Returns
    -------
    pl.DataFrame
        Rows in OUTPUT_COLS + '_order' shape, ready to concat into the
        sector's own assembled output.
    """
    if feedstock_demand is None or feedstock_demand.empty:
        return _empty_frame()

    last_hist_year = int(pd.to_numeric(feedstock_demand['Year']).max())

    frames = []
    for region, scale_series in scale_by_region.items():
        feedstock_region = feedstock_demand.loc[
            feedstock_demand['Region'] == region, ['Variable', 'Year', 'Value']
        ]
        if feedstock_region.empty:
            continue
        frames.append(build_feedstock_rows(
            sector_branch=f'CIMS.CAN.{region}.{sector_name}',
            sector_name=sector_name,
            region=region,
            feedstock=feedstock_region,
            scale_series=scale_series,
            scale_unit=scale_unit,
            last_hist_year=last_hist_year,
            ratio_window=ratio_window,
            **kwargs,
        ))

    return pl.concat(frames, how='diagonal_relaxed') if frames else _empty_frame()
