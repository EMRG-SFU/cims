"""
Biomass energy use by province and year for the CIMS counterfactual data.

Builds one clean table of actual biomass energy use (TJ), 2000-2024:

    StatCan fuel                  CIMS fuel       CIMS sector
    ----------------------------  --------------  ----------------
    Spent pulping liquor (SPL)    Black Liquor    Pulp and Paper
    Wood waste, pulp and paper    Solid Biomass   Pulp and Paper
    Wood waste, wood products     Solid Biomass   Light Industrial

Every published StatCan value is kept, suppressed cells are filled with the
rules below, and provinces add up to the StatCan Canada total in every year
(after the 2000-2003 adjustment). This script only prepares the amounts; the
sector calibration scripts decide which CIMS node each amount goes to.

Inputs
------
raw_data/stats_can/resd/2510003101-eng.csv                 [S1]
raw_data/stats_can/resd/CEEDC_national energy_2000-2024.xlsx [S2]
raw_data/stats_can/activity/36100488.csv                   [S3]

Output: processed_data/stats_can/biomass_demand.csv
    Columns: region, year, statcan_fuel, cims_fuel, cims_sector, tj, method

Pipeline use:
    from CIMS.data_processing.source.stats_can.biomass_demand import load_biomass_demand
    biomass = load_biomass_demand()   # pandas, CIMS region codes (AB, BC, ...)

    # Swap CER biomass for this table in a cer_resd_demand-style frame (GJ).
    # Called by cer_resd_demand.main() and mecs_steam_calibration._load_basis().
    from CIMS.data_processing.source.stats_can.biomass_demand import replace_cer_biomass
    cer_df = replace_cer_biomass(cer_df)

Where the amounts go (replace_cer_biomass)
------------------------------------------
    .Pulp and Paper                Black Liquor (SPL), Solid Biomass (wood waste)
    .Light Industrial              Solid Biomass (wood products wood waste)
Black Liquor goes on the sector node only. There is no Recovery Furnace
counterfactual; the Recovery Furnace is calibrated inside the model against
the sector's Black Liquor. Black Liquor is NOT put at .Pulp and Paper.Steam:
the Recovery Furnace supplies steam to the Steam node through its negative
service request. The Steam node's
fuel rows are built by mecs_steam_calibration, which subtracts SPL from the
MECS steam fuel target so the liquor's steam is not counted twice.
Steam nodes (built by mecs_steam_calibration):
    .Pulp and Paper.Steam     Solid Biomass first, up to the MECS steam fuel
                              target less SPL; all P&P wood waste is treated as
                              steam fuel (MECS: ~99% of forest products biomass
                              goes to boilers and cogeneration).
    .Light Industrial.Steam   provincial share of wood products wood waste first
                              (BC 50%, AB 0%, others 100%; provisional), up to
                              the MECS steam fuel target, then fossil fuels
                              (BIOMASS_STEAM_SHARE in mecs_steam_calibration).
CER biomass rows at these sectors' nodes are dropped first. Regions with no
StatCan biomass (PE, territories) end up with none.

Standalone:
    python biomass_demand.py <2510003101-eng.csv> <CEEDC.xlsx> <36100488.csv> <output.csv>


DATA SOURCES
------------
[S1] Statistics Canada. Table 25-10-0031-01, Consumption of solid wood waste
     and spent pulping liquor for energy production, annual (release
     2025-11-13). Provincial totals for both fuels. NL and NS are published
     combined.
     https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=2510003101
[S2] CEEDC national industrial energy use, 2000-2024 ('Data' sheet, Canada,
     'Wood' column, GJ). National share of wood waste burned by wood products:
     NAICS 321 / (NAICS 321 + NAICS 322).
[S3] Statistics Canada. Table 36-10-0488-01, Output by industry, provinces and
     territories, 'Sawmills and wood preservation [BS321100]'. Provincial
     proxy for wood products wood waste and for the NL vs NS wood products
     split. Ends in 2022; 2023-2024 use the 2022 shares.


WOOD WASTE METHOD
-----------------
W1. Suppressed MB and SK (2000-2018): MB + SK = Canada minus the published
    regions [S1], split by the pooled 2019-2021 MB share (58.7%), the first
    years both are published.
W2. 2000-2003: each region gets its own 2004-minus-2003 step added, removing
    the 2003->2004 break in [S1] (national step: 46,229 TJ). Adjustment
    suggested by Brad, as the 2000-2003 values looked low compared with 2004
    onward when the series was plotted.
W3. Sector split: wood products = Canada total x national wood products
    share [S2] x the region's share of sawmill output [S3] (shares taken over
    the eight StatCan regions, NL and NS combined). Pulp and paper is the
    remainder.
W4. If wood products would exceed a region's total, it is capped at the
    total and the excess is given to the other regions in proportion to
    their wood products, so national wood products always matches [S2]. This
    happens once: NL/NS in 2022 (243 TJ).
W5. NL/NS: wood products split by NL vs NS sawmill output [S3]; pulp and
    paper split by the number of mills operating each year (MILLS below).
    The mill-count split assumes each mill burns the same amount of wood.


SPL METHOD
----------
SPL comes only from chemical pulping (kraft, sulphite, semi-chemical).
Mechanical pulping (groundwood, TMP, CTMP, BCTMP) and recycled-fibre mills
produce no spent liquor [M1, M8]. All SPL is pulp and paper.

QC, ON, AB and BC are published every year [S1]. NS, NB, MB and SK are about
10% of national SPL and are filled from the residual (Canada minus QC, ON, AB,
BC) as follows:

P1. Published values in [S1] are always used as is.
P2. NL has no chemical pulp mill [M2], so all NL/NS SPL is assigned to NS.
P3. NS: 0 from 2020 (Northern Pulp, the only NS chemical mill left after
    Port Hawkesbury replaced its sulphite plant around 2005 [M9], closed in
    January 2020 [M4]). 2019 = residual - NB - MB. Suppressed years (2007,
    2010-2018) are interpolated between neighbouring values.
P4. SK: 0 wherever not published. Its only kraft mill (Prince Albert) closed
    in 2006 [M6, M7]; [S1] publishes 0 for 2017-2020. SK SPL for 2000-2006 is
    not estimated and falls into NB (see LIMITATIONS).
P5. MB: from 2020, residual - NB (NS and SK are 0). Before 2020, held at the
    2020-2024 average; its one kraft mill (The Pas) has run since 1970 [M5].
    In 2007, when NB is published, MB takes whatever is left over.
P6. NB: published where available, otherwise the remainder (residual - NS -
    MB - SK).
P7. 2000-2003: same step adjustment as W2 (set ADJUST_2000_2003['SPL']).


MILLS (NL/NS pulp and paper wood waste split, W5)
-------------------------------------------------
Mill-years count when the mill ran most of the year. Port Hawkesbury is not
counted after its 2011 idling: after it restarted, its steam comes from the
Nova Scotia Power biomass plant next door, an electric utility.
    NL  Corner Brook (Kruger), newsprint ........... 2000-2024  [M2]
    NL  Stephenville (Abitibi), newsprint .......... 2000-2005  [M2]
    NL  Grand Falls-Windsor (Abitibi), newsprint ... 2000-2008  [M2]
    NS  Northern Pulp, Pictou, kraft ............... 2000-2019  [M4]
    NS  Port Hawkesbury, paper ..................... 2000-2011
    NS  Bowater Mersey, Liverpool, newsprint ....... 2000-2011  [M3]
    NS  Minas Basin, Hantsport, paperboard ......... 2000-2011


MILL REFERENCES
---------------
[M1] Natural Resources Canada, Canadian Forest Service forestry glossary,
     "black liquor": residue from the kraft pulping process.
     https://cfs.nrcan.gc.ca/terms/read/366
[M2] Government of Newfoundland and Labrador, Pulp and Paper Mills. Corner
     Brook: TMP newsprint, operating. Grand Falls: closed Feb 2009.
     Stephenville: closed Oct 2005.
     https://www.gov.nl.ca/eccc/env-protection/ics/pulp
[M3] Wikipedia, Bowater Mersey Paper Company: Brooklyn NS mill was a TMP
     newsprint mill; idled June 16, 2012.
     https://en.wikipedia.org/wiki/Bowater_Mersey_Paper_Company_Limited
[M4] PaperAge, Dec 20, 2019: Northern Pulp (NS), 280,000 t/yr NBSK kraft
     pulp, closing by the end of January 2020.
     https://www.paperage.com/2019news/12_20_2019northern_pulp_to_close.html
[M5] Pulp & Paper Canada, "Modernization: it's all in the bag for Tolko":
     The Pas MB, kraft process, unbleached kraft paper, ~155,000 t/yr,
     started 1970.
     https://www.pulpandpapercanada.com/modernization-it-s-all-in-the-bag-for-tolko-1000109843/
[M6] Weyerhaeuser, Oct 4, 2005: indefinite closure of Prince Albert pulp and
     paper from Jan 2006; the pulp mill runs until spring 2006.
     https://investor.weyerhaeuser.com/2005-10-04-Weyerhaeuser-Announces-Indefinite-Closure-at-Prince-Albert-Pulp-Paper
[M7] PaperAge, Feb 8, 2021: Prince Albert produced NBSK (kraft) pulp; closed
     in 2006.
     https://paperage.com/2021news/02_08_2021paper_excellence_prince_albert_pulp.html
[M8] Domtar, Meadow Lake Mill: BCTMP (mechanical), 405,000 t/yr, since 1992.
     https://www.domtar.com/wp-content/uploads/2024/10/Meadow-Lake-Mill.pdf
[M9] Valmet, Feb 24, 2005: new TMP line at Port Hawkesbury NS replaces its
     stone groundwood and high-yield sulphite plants.
     https://www.valmet.com/news/news-archive-1999-2013/pulp-and-paper-news/stora-enso-starts-up-new-tmp-mill-at-port-hawkesbury-in-canada/


LIMITATIONS
-----------
- NB absorbs any error in the other SPL rules (likely a few PJ in some
  years), including SK's Prince Albert mill in 2000-2006 and real changes in
  MB.
- MB SPL before 2020 is flat.
- All StatCan wood waste is split between pulp and paper and wood products.
  CEEDC shows about 1.5% of industrial wood burned elsewhere (food, cement,
  metals, chemicals); it is folded into the two sectors.
- NL pulp and paper wood waste from 2020 is lower than ECCC GHGRP implies for
  Corner Brook. GHGRP and [S1] do not reconcile, so [S1] is kept.
- The 2000-2003 adjustment assumes the 2003->2004 jump in [S1] is a
  reporting break. [S1] runs 20-47% above StatCan's industrial energy survey
  in 2004-2015, consistent with a reporting change. For SPL the adjustment
  raises Canada 2000 from 318 to 368 PJ, the highest year in the series;
  SPL does not show as clean a step as wood waste. Kept for consistency
  between the two fuels.
- The pulping process at Stephenville and Grand Falls (NL) is not confirmed
  from a primary source. Both closed by 2009, so this only affects whether any
  of the published 2000-2009 NL/NS SPL belongs to NL.
"""

import sys
from functools import lru_cache

import polars as pl

# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
YEARS = (2000, 2024)            # output years, inclusive
ADJUST_2000_2003 = {"Wood": True, "SPL": True}   # W2 / P7
MBSK_SHARE_YEARS = (2019, 2021)  # W1: years for the pooled MB share
MB_SPL_YEARS = (2020, 2024)      # P5: years for the MB SPL average
SPL_ZERO_FROM = {"NS": 2020}     # P3: first year with no chemical mill

# 25-10-0031 geography -> region code (NL and NS are published combined)
REGIONS = {
    "Newfoundland and Labrador and Nova Scotia": "NL/NS",
    "New Brunswick": "NB", "Quebec": "QC", "Ontario": "ON", "Manitoba": "MB",
    "Saskatchewan": "SK", "Alberta": "AB", "British Columbia": "BC",
    "Canada": "CAN",
}
PROVS = ["NL/NS", "NB", "QC", "ON", "MB", "SK", "AB", "BC"]
SPL_BIG4 = ["QC", "ON", "AB", "BC"]   # SPL published every year

# 36-10-0488 geography -> region code for the sawmill shares
SAWMILL_GEO = {
    "Newfoundland and Labrador": "NL", "Nova Scotia": "NS",
    "New Brunswick": "NB", "Quebec": "QC", "Ontario": "ON", "Manitoba": "MB",
    "Saskatchewan": "SK", "Alberta": "AB", "British Columbia": "BC",
}
SAWMILL_INDUSTRY = "Sawmills and wood preservation [BS321100]"

# W5: pulp and paper mills burning their own wood, (region, first year, last year)
MILLS = {
    "Corner Brook":        ("NL", 2000, 2024),
    "Stephenville":        ("NL", 2000, 2005),
    "Grand Falls-Windsor": ("NL", 2000, 2008),
    "Northern Pulp":       ("NS", 2000, 2019),
    "Port Hawkesbury":     ("NS", 2000, 2011),
    "Bowater Mersey":      ("NS", 2000, 2011),
    "Minas Basin":         ("NS", 2000, 2011),
}

YEAR_LIST = list(range(YEARS[0], YEARS[1] + 1))


# --------------------------------------------------------------------------
# Readers
# --------------------------------------------------------------------------
def read_statcan_biomass(path: str) -> dict[str, dict[str, dict[int, float | None]]]:
    """25-10-0031 -> {fuel: {region: {year: TJ or None}}}, fuel 'Wood' or 'SPL'.
    Suppressed ('x') and not-available ('..') cells are None."""
    import csv
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    head = next(i for i, r in enumerate(rows) if r and r[0] == "Geography")
    years = [int(y) for y in rows[head][3:]]
    out = {"Wood": {}, "SPL": {}}
    geo = None
    for r in rows[head + 1:]:
        if len(r) < 4 or r[1] not in ("Solid wood waste", "Spent pulping liquor"):
            break  # end of table (symbol legend follows)
        geo = r[0] or geo
        fuel = "Wood" if r[1] == "Solid wood waste" else "SPL"
        vals = {}
        for y, v in zip(years, r[3:]):
            v = v.replace(",", "").strip()
            vals[y] = float(v) if v not in ("x", "..", "") else None
        out[fuel][REGIONS[geo]] = vals
    return out


def read_ceedc_wp_share(path: str) -> dict[int, float]:
    """[S2] National wood products share of wood waste: 321 / (321 + 322)."""
    df = pl.read_excel(path, sheet_name="Data", has_header=True, infer_schema_length=0)
    df = (
        df.filter((pl.col("Region") == "Canada") & pl.col("NAICS").is_in(["321", "322"]))
        .select(pl.col("Year").cast(pl.Int32), "NAICS", pl.col("Wood").cast(pl.Float64))
    )
    wide = df.pivot(on="NAICS", index="Year", values="Wood")
    return {r["Year"]: r["321"] / (r["321"] + r["322"]) for r in wide.iter_rows(named=True)}


def read_sawmill_output(path: str) -> dict[int, dict[str, float]]:
    """[S3] Sawmill output ($M) by region code and year."""
    df = pl.read_csv(path, infer_schema_length=0)
    df = (
        df.filter((pl.col("Industry") == SAWMILL_INDUSTRY) & pl.col("GEO").is_in(list(SAWMILL_GEO)))
        .select(pl.col("REF_DATE").cast(pl.Int32).alias("year"),
                pl.col("GEO").replace_strict(SAWMILL_GEO).alias("region"),
                pl.col("VALUE").cast(pl.Float64).alias("value"))
    )
    out: dict[int, dict[str, float]] = {}
    for r in df.iter_rows(named=True):
        out.setdefault(r["year"], {})[r["region"]] = r["value"]
    last = max(out)
    for y in YEAR_LIST:  # 2023-2024: hold the last year's output
        if y not in out:
            out[y] = dict(out[last])
    return out


# --------------------------------------------------------------------------
# Wood waste
# --------------------------------------------------------------------------
def fill_wood(raw: dict) -> tuple[dict, dict]:
    """W1. Returns (TJ, method) as {region: {year: ...}}."""
    tj = {r: dict(raw[r]) for r in PROVS + ["CAN"]}
    method = {r: {y: "published" for y in YEAR_LIST} for r in PROVS}
    mb = sum(tj["MB"][y] for y in range(MBSK_SHARE_YEARS[0], MBSK_SHARE_YEARS[1] + 1))
    sk = sum(tj["SK"][y] for y in range(MBSK_SHARE_YEARS[0], MBSK_SHARE_YEARS[1] + 1))
    mb_share = mb / (mb + sk)
    for y in YEAR_LIST:
        if tj["MB"][y] is None or tj["SK"][y] is None:
            pub = sum(tj[r][y] for r in PROVS if r not in ("MB", "SK"))
            resid = tj["CAN"][y] - pub
            tj["MB"][y], tj["SK"][y] = resid * mb_share, resid * (1 - mb_share)
            note = f"filled: residual x MB/SK {MBSK_SHARE_YEARS[0]}-{MBSK_SHARE_YEARS[1]} share"
            method["MB"][y] = f"{note} ({mb_share:.3f})"
            method["SK"][y] = f"{note} ({1 - mb_share:.3f})"
    return tj, method


def adjust_2000_2003(tj: dict, method: dict, regions: list[str]) -> None:
    """W2 / P7. Add each region's own 2004-minus-2003 step to 2000-2003."""
    for r in regions:
        step = tj[r][2004] - tj[r][2003]
        for y in range(2000, 2004):
            tj[r][y] += step
            method[r][y] += f"; +2004-2003 step ({step:,.0f} TJ)"


def split_sectors(wood: dict, wp_share: dict, saw: dict) -> tuple[dict, dict, dict]:
    """W3, W4. Returns (wood products, pulp and paper, flag) as {region: {year: ...}}."""
    wp = {r: {} for r in PROVS}
    flag = {r: {y: "" for y in YEAR_LIST} for r in PROVS}
    for y in YEAR_LIST:
        s = dict(saw[y])
        s["NL/NS"] = s.pop("NL") + s.pop("NS")
        s_tot = sum(s[r] for r in PROVS)
        total = sum(wood[r][y] for r in PROVS)
        target = total * wp_share[y]
        want = {r: target * s[r] / s_tot for r in PROVS}
        # W4: cap at each region's total; pass any excess to the uncapped regions
        capped: set[str] = set()
        while True:
            over = {r for r in PROVS if r not in capped and want[r] > wood[r][y]}
            if not over:
                break
            capped |= over
            excess = sum(want[r] - wood[r][y] for r in over)
            for r in over:
                want[r] = wood[r][y]
            free = [r for r in PROVS if r not in capped]
            base = sum(want[r] for r in free)
            for r in free:
                add = excess * want[r] / base
                want[r] += add
                flag[r][y] = f"received {add:,.1f} TJ of capped wood products"
        for r in capped:
            flag[r][y] = "wood products exceeded total; capped at total"
        for r in PROVS:
            wp[r][y] = want[r]
    pp = {r: {y: wood[r][y] - wp[r][y] for y in YEAR_LIST} for r in PROVS}
    return wp, pp, flag


def mill_count_share(year: int) -> float:
    """W5. NL share of NL+NS pulp and paper mills operating in `year`."""
    n = {"NL": 0, "NS": 0}
    for region, first, last in MILLS.values():
        if first <= year <= last:
            n[region] += 1
    return n["NL"] / (n["NL"] + n["NS"])


def split_nl_ns(wp: dict, pp: dict, flag: dict, method: dict, saw: dict) -> None:
    """W5. Replace the NL/NS region with NL and NS, in place."""
    for d in (wp, pp, flag, method):
        d["NL"], d["NS"] = {}, {}
    for y in YEAR_LIST:
        wp_nl = saw[y]["NL"] / (saw[y]["NL"] + saw[y]["NS"])
        pp_nl = mill_count_share(y)
        wp["NL"][y], wp["NS"][y] = wp["NL/NS"][y] * wp_nl, wp["NL/NS"][y] * (1 - wp_nl)
        pp["NL"][y], pp["NS"][y] = pp["NL/NS"][y] * pp_nl, pp["NL/NS"][y] * (1 - pp_nl)
        note = (f"; NL/NS split, NL share: wood products {wp_nl:.3f} (sawmill output), "
                f"pulp and paper {pp_nl:.3f} (mill count)")
        for r in ("NL", "NS"):
            flag[r][y] = flag["NL/NS"][y]
            method[r][y] = method["NL/NS"][y] + note
    for d in (wp, pp, flag, method):
        del d["NL/NS"]


# --------------------------------------------------------------------------
# Spent pulping liquor
# --------------------------------------------------------------------------
def fill_spl(raw: dict) -> tuple[dict, dict]:
    """P1-P6. Returns (TJ, method) as {region: {year: ...}} for NS, NB, MB, SK,
    QC, ON, AB, BC, NL (NL is always 0)."""
    pub = {r: dict(raw[r]) for r in PROVS + ["CAN"]}
    pub["NS"] = pub.pop("NL/NS")                                      # P2
    resid = {y: pub["CAN"][y] - sum(pub[r][y] for r in SPL_BIG4) for y in YEAR_LIST}

    tj = {r: {} for r in ["NL", "NS", "NB", "MB", "SK"] + SPL_BIG4}
    method = {r: {} for r in tj}
    for r in SPL_BIG4:
        for y in YEAR_LIST:
            tj[r][y], method[r][y] = pub[r][y], "published"
    for y in YEAR_LIST:
        tj["NL"][y], method["NL"][y] = 0.0, "0: no chemical pulp mill"

    # P4: SK published or 0
    for y in YEAR_LIST:
        if pub["SK"][y] is not None:
            tj["SK"][y], method["SK"][y] = pub["SK"][y], "published"
        else:
            tj["SK"][y], method["SK"][y] = 0.0, "0: not published (Prince Albert kraft mill closed 2006)"

    # P3 (part): NS published, or 0 after closure
    ns: dict[int, float | None] = {}
    for y in YEAR_LIST:
        if pub["NS"][y] is not None:
            ns[y], method["NS"][y] = pub["NS"][y], "published (NL/NS)"
        elif y >= SPL_ZERO_FROM["NS"]:
            ns[y], method["NS"][y] = 0.0, "0: Northern Pulp closed Jan 2020"
        else:
            ns[y] = None

    # P5: MB from 2020 is the residual minus NB (NS, SK are 0); before, the average
    mb_years = range(MB_SPL_YEARS[0], MB_SPL_YEARS[1] + 1)
    for y in mb_years:
        if pub["NB"][y] is None or ns[y] != 0.0 or tj["SK"][y] != 0.0:
            raise ValueError(f"MB SPL rule P5 needs NB published and NS, SK = 0 in {y}")
        tj["MB"][y] = resid[y] - pub["NB"][y]
        method["MB"][y] = "residual - NB (NS, SK = 0)"
    mb_avg = sum(tj["MB"][y] for y in mb_years) / len(mb_years)
    for y in YEAR_LIST:
        if y < MB_SPL_YEARS[0]:
            tj["MB"][y] = mb_avg
            method["MB"][y] = f"{MB_SPL_YEARS[0]}-{MB_SPL_YEARS[1]} average"

    # P3: the last year before closure is the residual minus NB and MB, if NB is published
    last = SPL_ZERO_FROM["NS"] - 1
    if ns[last] is None and pub["NB"][last] is not None:
        ns[last] = resid[last] - pub["NB"][last] - tj["MB"][last] - tj["SK"][last]
        method["NS"][last] = "residual - NB - MB"
    # P3: interpolate the remaining gaps
    s = pl.Series([ns[y] for y in YEAR_LIST], dtype=pl.Float64).interpolate()
    for y, v in zip(YEAR_LIST, s.to_list()):
        if ns[y] is None:
            method["NS"][y] = "interpolated"
        tj["NS"][y] = v

    # P6: NB published or remainder; P5: if NB is published before 2020, MB takes the rest
    for y in YEAR_LIST:
        if pub["NB"][y] is not None:
            tj["NB"][y], method["NB"][y] = pub["NB"][y], "published"
            if y < MB_SPL_YEARS[0] and method["NS"][y] != "residual - NB - MB":
                tj["MB"][y] = resid[y] - pub["NB"][y] - tj["NS"][y] - tj["SK"][y]
                method["MB"][y] = "residual - NB - NS - SK (NB published)"
        else:
            tj["NB"][y] = resid[y] - tj["NS"][y] - tj["MB"][y] - tj["SK"][y]
            method["NB"][y] = "remainder: residual - NS - MB - SK"
    return tj, method


# --------------------------------------------------------------------------
# Build and check
# --------------------------------------------------------------------------
def build(statcan_path: str, ceedc_path: str, sawmill_path: str) -> pl.DataFrame:
    raw = read_statcan_biomass(statcan_path)
    wp_share = read_ceedc_wp_share(ceedc_path)
    saw = read_sawmill_output(sawmill_path)

    # Wood waste
    wood, wmethod = fill_wood(raw["Wood"])
    if ADJUST_2000_2003["Wood"]:
        adjust_2000_2003(wood, wmethod, PROVS)
    wp, pp, flag = split_sectors(wood, wp_share, saw)
    split_nl_ns(wp, pp, flag, wmethod, saw)

    # SPL
    spl, smethod = fill_spl(raw["SPL"])
    if ADJUST_2000_2003["SPL"]:
        adjust_2000_2003(spl, smethod, [r for r in spl if r != "NL"])

    rows = []
    for r in wp:
        for y in YEAR_LIST:
            m = wmethod[r][y] + (f"; {flag[r][y]}" if flag[r][y] else "")
            rows.append((r, y, "Solid wood waste", "Solid Biomass", "Light Industrial", wp[r][y], m))
            rows.append((r, y, "Solid wood waste", "Solid Biomass", "Pulp and Paper", pp[r][y], m))
    for r in spl:
        for y in YEAR_LIST:
            rows.append((r, y, "Spent pulping liquor", "Black Liquor", "Pulp and Paper", spl[r][y], smethod[r][y]))
    out = pl.DataFrame(rows, orient="row", schema={
        "region": pl.Utf8, "year": pl.Int32, "statcan_fuel": pl.Utf8, "cims_fuel": pl.Utf8,
        "cims_sector": pl.Utf8, "tj": pl.Float64, "method": pl.Utf8})
    check(out, raw, wp_share)
    return out.sort("region", "year", "cims_fuel", "cims_sector")


def check(out: pl.DataFrame, raw: dict, wp_share: dict, tol: float = 2.0) -> None:
    """Raise if any of the add-up rules fail. `tol` (TJ) allows for rounding in
    [S1], where published provinces can differ from Canada by 1 TJ."""
    if out["tj"].is_null().any() or (out["tj"] < -tol).any():
        raise ValueError("missing or negative values in biomass table")
    for fuel, key in (("Solid wood waste", "Wood"), ("Spent pulping liquor", "SPL")):
        tot = out.filter(pl.col("statcan_fuel") == fuel).group_by("year").agg(pl.col("tj").sum())
        can = raw[key]["CAN"]
        for y, v in tot.iter_rows():
            if y >= 2004 or not ADJUST_2000_2003[key]:
                if abs(v - can[y]) > tol:
                    raise ValueError(f"{fuel} {y}: provinces {v:,.1f} != Canada {can[y]:,.1f}")
    lw = out.filter((pl.col("statcan_fuel") == "Solid wood waste")).group_by("year", "cims_sector").agg(pl.col("tj").sum())
    for y in YEAR_LIST:
        w = {s: v for yy, s, v in lw.iter_rows() if yy == y}
        share = w["Light Industrial"] / (w["Light Industrial"] + w["Pulp and Paper"])
        if abs(share - wp_share[y]) > 1e-6:
            raise ValueError(f"wood products share {y}: {share:.6f} != CEEDC {wp_share[y]:.6f}")
    # published provincial values are unchanged (2004 onward, or all years if not adjusted)
    for key, fuel in (("Wood", "Solid wood waste"), ("SPL", "Spent pulping liquor")):
        tot = out.filter(pl.col("statcan_fuel") == fuel).group_by("region", "year").agg(pl.col("tj").sum())
        got = {(r, y): v for r, y, v in tot.iter_rows()}
        for reg in PROVS:
            for y in YEAR_LIST:
                pv = raw[key][reg][y]
                if pv is None or (ADJUST_2000_2003[key] and y < 2004):
                    continue
                v = got[("NL", y)] + got[("NS", y)] if reg == "NL/NS" else got[(reg, y)]
                if abs(v - pv) > tol:
                    raise ValueError(f"{fuel} {reg} {y}: {v:,.1f} != published {pv:,.1f}")


# --------------------------------------------------------------------------
# Swap into CER demand
# --------------------------------------------------------------------------
USE_STATCAN_BIOMASS = True   # False: leave CER biomass untouched

SECTOR_NODES = (".Pulp and Paper", ".Light Industrial")
CER_BIOMASS_FUELS = {"Black Liquor", "Solid Biomass"}   # CER 'Biomass' after map_fuel_names
SOURCE_LABEL = "StatCan 25-10-0031 (biomass_demand)"

# (cims_fuel, cims_sector) in the biomass table -> CIMS nodes that get it
_PLACEMENT = {
    ("Black Liquor", "Pulp and Paper"):     [".Pulp and Paper"],
    ("Solid Biomass", "Pulp and Paper"):    [".Pulp and Paper"],
    ("Solid Biomass", "Light Industrial"):  [".Light Industrial"],
}


def replace_cer_biomass(cer_df):
    """Replace CER biomass with the StatCan biomass table in a frame shaped
    like cer_resd_demand output (Region | Node | Variable | Unit | Source |
    Year | Value, GJ). Rows at other sectors are untouched. Only years in
    the biomass table are replaced."""
    import pandas as pd
    if not USE_STATCAN_BIOMASS:
        return cer_df
    bio = load_biomass_demand()
    years = set(bio["year"])
    in_sector = cer_df["Node"].str.startswith(SECTOR_NODES)
    drop = in_sector & cer_df["Variable"].isin(CER_BIOMASS_FUELS) & cer_df["Year"].astype(int).isin(years)
    rows = []
    for (fuel, sector), nodes in _PLACEMENT.items():
        b = bio[(bio["cims_fuel"] == fuel) & (bio["cims_sector"] == sector)]
        b = b.groupby(["region", "year"], as_index=False)["tj"].sum()
        for node in nodes:
            rows.append(pd.DataFrame({
                "Region": b["region"], "Node": node, "Variable": fuel, "Unit": "GJ",
                "Source": SOURCE_LABEL, "Year": b["year"], "Value": b["tj"] * 1000.0,
            }))
    new = pd.concat(rows, ignore_index=True)
    new["Year"] = new["Year"].astype(cer_df["Year"].dtype)
    out = pd.concat([cer_df[~drop], new], ignore_index=True)
    return out.sort_values(["Region", "Node", "Variable", "Year"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# Pipeline entry point
# --------------------------------------------------------------------------
def _paths():
    from CIMS.data_processing.utils.controls_conversions import BASE_PATH
    return (BASE_PATH / "raw_data/stats_can/resd/2510003101-eng.csv",
            BASE_PATH / "raw_data/stats_can/resd/CEEDC_national energy_2000-2024.xlsx",
            BASE_PATH / "raw_data/stats_can/activity/36100488.csv",
            BASE_PATH / "processed_data/stats_can/biomass_demand.csv")


@lru_cache(maxsize=1)
def load_biomass_demand():
    """Biomass table as pandas. Region codes already match CIMS (NL, NS, ...).
    Rebuilt from the raw files on every new session and saved to
    processed_data for reference."""
    *src, dst = _paths()
    out = build(*map(str, src))
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(dst)
    import pandas as pd
    return pd.DataFrame(out.to_dict(as_series=False))  # no pyarrow needed


if __name__ == "__main__":
    if len(sys.argv) == 5:
        out = build(*sys.argv[1:4])
        dst = sys.argv[4]
    else:
        *src, dst = _paths()
        out = build(*map(str, src))
        dst.parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(dst)
    print(f"wrote {out.height} rows -> {dst}")
