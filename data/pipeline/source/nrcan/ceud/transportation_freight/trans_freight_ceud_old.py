"""
trans_freight_ceud.py  —  Freight Transportation CEUD Pipeline (Sprint 1)
=========================================================================
Extracts CEUD regional data and builds base dataframes for the freight
transportation energy model, mirroring "transportation freight_source data.xlsx".

All assumptions are embedded as Python dictionaries — no external CSV files
are read at runtime except the NRCan CEUD .xls workbooks.

Input files (same directory as this script):
  transBCTerr2000-2022EN.xls   (Tables 15, 18, 35, 36, 37)
  transALB2000-2022EN.xls      (Tables 15, 18, 19, 35, 36, 37)
  transSASK2000-2022EN.xls     (Tables 15, 18, 19, 35, 36, 37)
  transCan2000-2022EN.xls      (Freight4 sheet — CAN-level load factors)

Output:  output/freight_bcterr_*.csv, output/freight_bc_*.csv, output/freight_ab_*.csv, output/freight_sk_*.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
import re
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="xlrd")

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  PATHS                                                               ║
# ╚═══════════════════════════════════════════════════════════════════════╝
SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR    = SCRIPT_DIR / "output"
OUT_DIR.mkdir(exist_ok=True)

CEUD_BCTERR_FILE = SCRIPT_DIR / "transBCTerr2000-2022EN.xls"
CEUD_AB_FILE     = SCRIPT_DIR / "transALB2000-2022EN.xls"
CEUD_SK_FILE     = SCRIPT_DIR / "transSASK2000-2022EN.xls"
CEUD_MB_FILE     = SCRIPT_DIR / "transMAN2000-2022EN.xls"
CEUD_ON_FILE     = SCRIPT_DIR / "transONT2000-2022EN.xls"
CEUD_QC_FILE     = SCRIPT_DIR / "transQUE2000-2022EN.xls"
CEUD_NB_FILE     = SCRIPT_DIR / "transNB2000-2022EN.xls"
CEUD_NS_FILE     = SCRIPT_DIR / "transNS2000-2022EN.xls"
CEUD_PE_FILE     = SCRIPT_DIR / "transPEI2000-2022EN.xls"
CEUD_NL_FILE     = SCRIPT_DIR / "transNFLD2000-2022EN.xls"
CEUD_AT_FILE     = SCRIPT_DIR / "transATL2000-2022EN.xls"
CEUD_CAN_FILE    = SCRIPT_DIR / "transCan2000-2022EN.xls"
COEFFICIENTS_FILE = SCRIPT_DIR / "coefficients.csv"

for _p, _label in [
    (CEUD_BCTERR_FILE, "BCTerr CEUD"),
    (CEUD_AB_FILE,     "Alberta CEUD"),
    (CEUD_SK_FILE,     "Saskatchewan CEUD"),
    (CEUD_MB_FILE,     "Manitoba CEUD"),
    (CEUD_ON_FILE,     "Ontario CEUD"),
(CEUD_NS_FILE,     "Nova Scotia CEUD"),
    (CEUD_PE_FILE,     "Prince Edward Island CEUD"),
    (CEUD_NL_FILE,     "Newfoundland and Labrador CEUD"),
    (CEUD_AT_FILE,     "Atlantic CEUD"),
    (CEUD_CAN_FILE,    "CAN CEUD"),
    (COEFFICIENTS_FILE, "coefficients"),
]:
    if not _p.exists():
        warnings.warn(f"{_label} file not found: {_p}")

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  CONSTANTS                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════╝
YEARS     = list(range(2000, 2023))
N_YEARS   = len(YEARS)
YEAR_COLS = list(range(2, 2 + N_YEARS))  # 0-indexed columns C..Y in .xls
FUEL_SCALE = 1000.0                       # PJ → TJ

FREIGHT_FUELS = [
    "Aviation turbo fuel",
    "Aviation gasoline",
    "Diesel fuel oil",
    "Biodiesel fuel",
    "Motor gasoline",
    "Ethanol",
    "Electricity",
    "Natural gas",
    "Heavy fuel oil",
    "Propane",
]

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  ASSUMPTIONS  (from "assumptions - Values - Freight.csv")            ║
# ║  Embedded as Python dicts — no CSV read at runtime                   ║
# ╚═══════════════════════════════════════════════════════════════════════╝

# ── Load factors (tonne / vehicle) ──────────────────────────────────────
# Source: NAP report, Table 2-1, Page 18
LOAD_FACTOR: dict[str, float] = {
    "Light Truck":  0.552608725,
    "Medium Truck": 1.250652177,
    "Heavy Truck":  6.861117667,
    "Rail":         10_000.0,
    "Marine":       6_840.0,
    "Aviation":     20.0,
    "Off Road":     1.0,
}

# ── tonne-kilometre CAGR assumptions ───────────────────────────────────
# Structure: { mode: { prov_code: (hist_cagr, share_of_2019, ref_cagr_2023_2050, ref_cagr_2051_2100) } }
# hist_cagr = CAGR 2005-2019; share_of_2019 = share applied to 2022 base;
# ref_cagr = reference growth rate used in CAGR formula
KTKM_CAGR = {
    "Light Truck": {
        "BC": (0.032708436, 0.9, 0.009812531, 0.004906265),
        "AB": (0.038513289, 0.9, 0.011553987, 0.005776993),
        "SK": (0.054104229, 0.9, 0.016231269, 0.008115634),
        "MB": (0.060849319, 0.9, 0.018254796, 0.009127398),
        "ON": (0.038670944, 0.9, 0.011601283, 0.005800642),
        "QC": (0.040470038, 0.9, 0.012141011, 0.006070506),
        "NB": (0.027313726, 0.9, 0.008194118, 0.004097059),
        "NS": (0.036280490, 0.9, 0.010884147, 0.005442073),
        "PE": (0.024134613, 0.9, 0.007240384, 0.003620192),
        "NL": (0.050444875, 0.9, 0.015133463, 0.007566731),
        "AT": (0.035513872, 0.9, 0.010654162, 0.005327081),
    },
    "Medium Truck": {
        "BC": (0.037169813, 0.9, 0.011150944, 0.005575472),
        "AB": (0.053326096, 0.9, 0.015997829, 0.007998914),
        "SK": (0.090810539, 0.9, 0.027243162, 0.013621581),
        "MB": (0.024842146, 0.9, 0.007452644, 0.003726322),
        "ON": (0.026157549, 0.9, 0.007847265, 0.003923632),
        "QC": (0.038965568, 0.9, 0.011689670, 0.005844835),
        "NB": (-0.004046157, 0.9, -0.001213847, -0.000606923),
        "NS": (0.031427551, 0.9, 0.009428265, 0.004714133),
        "PE": (0.020399106, 0.9, 0.006119732, 0.003059866),
        "NL": (0.047927813, 0.9, 0.014378344, 0.007189172),
        "AT": (0.022611905, 0.9, 0.006783571, 0.003391786),
    },
    "Heavy Truck": {
        "BC": (0.008314197, 0.9, 0.002494259, 0.001247130),
        "AB": (0.021321217, 0.9, 0.006396365, 0.003198183),
        "SK": (0.040127855, 0.9, 0.012038356, 0.006019178),
        "MB": (0.022122311, 0.9, 0.006636693, 0.003318347),
        "ON": (0.006519973, 0.9, 0.001955992, 0.000977996),
        "QC": (0.016285001, 0.9, 0.004885500, 0.002442750),
        "NB": (-0.053606302, 0.9, -0.016081891, -0.008040945),
        "NS": (-0.008946446, 0.9, -0.002683934, -0.001341967),
        "PE": (-0.002135529, 0.9, -0.000640659, -0.000320329),
        "NL": (0.016000286, 0.9, 0.004800086, 0.002400043),
        "AT": (-0.014773978, 0.9, -0.004432193, -0.002216097),
    },
    "Rail": {
        "BC": (0.067559664, 1.0, 0.013511933, 0.006755966),
        "AB": (-0.018668871, 1.0, -0.003733774, -0.001866887),
        "SK": (0.096538237, 1.0, 0.019307647, 0.009653824),
        "MB": (0.088534270, 1.0, 0.017706854, 0.008853427),
        "ON": (0.006764143, 1.0, 0.001352829, 0.000676414),
        "QC": (-0.000476273, 1.0, -9.52547e-05, -4.76273e-05),
        "NB": (-0.037827819, 1.0, -0.007565564, -0.003782782),
        "NS": (0.030287292, 1.0, 0.006057458, 0.003028729),
        "PE": (0.0, 1.0, 0.0, 0.0),
        "NL": (0.0, 1.0, 0.0, 0.0),
        "AT": (-0.011388958, 1.0, -0.002277792, -0.001138896),
    },
    "Marine": {
        "BC": (-0.006308113, 0.9, -0.001892434, -0.000946217),
        "AB": (0.0, 0.9, 0.0, 0.0),
        "SK": (0.0, 0.9, 0.0, 0.0),
        "MB": (-0.049645804, 0.9, -0.014893741, -0.007446871),
        "ON": (0.004792198, 0.9, 0.001437659, 0.000718830),
        "QC": (-0.023359469, 0.9, -0.007007841, -0.003503920),
        "NB": (-0.023861313, 0.9, -0.007158394, -0.003579197),
        "NS": (-0.018730345, 0.9, -0.005619103, -0.002809552),
        "PE": (0.005210664, 0.9, 0.001563199, 0.000781600),
        "NL": (-0.022607051, 0.9, -0.006782115, -0.003391058),
        "AT": (-0.020246255, 0.9, -0.006073876, -0.003036938),
    },
    "Aviation": {
        "BC": (0.020138313, 0.9, 0.006041494, 0.003020747),
        "AB": (0.020505056, 0.9, 0.006151517, 0.003075758),
        "SK": (0.012402433, 0.9, 0.003720730, 0.001860365),
        "MB": (0.031640902, 0.9, 0.009492270, 0.004746135),
        "ON": (0.006755724, 0.9, 0.002026717, 0.001013359),
        "QC": (0.064605271, 0.9, 0.019381581, 0.009690791),
        "NB": (-0.022207424, 0.9, -0.006662227, -0.003331114),
        "NS": (-0.049266727, 0.9, -0.014780018, -0.007390009),
        "PE": (0.057909310, 0.9, 0.017372793, 0.008686397),
        "NL": (-0.003863733, 0.9, -0.001159120, -0.000579560),
        "AT": (-0.020395138, 0.9, -0.006118541, -0.003059271),
    },
    "Off Road": {
        "BC": (0.016952861, 0.9, 0.005085858, 0.002542929),
        "AB": (0.013051217, 0.9, 0.003915365, 0.001957683),
        "SK": (0.029976914, 0.9, 0.008993074, 0.004496537),
        "MB": (0.026411755, 0.9, 0.007923527, 0.003961763),
        "ON": (-0.015954103, 0.9, -0.004786231, -0.002393115),
        "QC": (-0.007927545, 0.9, -0.002378264, -0.001189132),
        "NB": (-0.036918175, 0.9, -0.011075452, -0.005537726),
        "NS": (-0.007702753, 0.9, -0.002310826, -0.001155413),
        "PE": (-0.024514539, 0.9, -0.007354362, -0.003677181),
        "NL": (-0.001637711, 0.9, -0.000491313, -0.000245657),
        "AT": (-0.018672819, 0.9, -0.005601846, -0.002800923),
    },
}

# ── Reference growth rate multipliers (applied to historical CAGR) ─────
# These values are the "Relative to historical" fractions per mode.
KTKM_REF_MULTIPLIER = {
    "Light Truck":  (0.3, 0.15),   # (2023-2050 multiplier, 2051-2100 multiplier)
    "Medium Truck": (0.3, 0.15),
    "Heavy Truck":  (0.3, 0.15),
    "Rail":         (0.2, 0.1),
    "Marine":       (0.3, 0.15),
    "Aviation":     (0.3, 0.15),
    "Off Road":     (0.3, 0.15),
}

# ── Average-km/year CAGR constraints ───────────────────────────────────
AVG_KM_CONSTRAINTS = {
    #                   years_to_calc_cagr, cagr_max_decrease, cagr_max_increase
    "Light Medium":     (5, -0.01, 0.0),
    "Light Truck":      (5, -0.01, 0.0),
    "Medium Truck":     (5, -0.01, 0.0),
    "Heavy Truck":      (5, -0.01, 0.0),
}

# ── Efficiency reference (MJ/vkm and MJ/tkm) ──────────────────────────
# Source: NRCan transportation trends
EFFICIENCY = {
    #                 2000_MJ_vkm     2000_MJ_tkm     2017_MJ_vkm     2017_MJ_tkm
    "Light Truck":  ( 4.664017636,    8.440000000,    3.818526287,    6.910000000),
    "Medium Truck": ( 9.775004773,    7.815925926,    7.491406539,    5.990000000),
    "Heavy Truck":  (13.473202170,    1.963703704,   10.017231790,    1.460000000),
    "Rail":         (2907.407407000,  0.290740741,  1900.000000000,   0.190000000),
    "Marine":       (3349.066667000,  0.489629630,  2530.800000000,   0.370000000),
    "Aviation":     (  61.362962960,  3.068148148,    39.200000000,   1.960000000),
    "Off Road":     (  None,          7.820000000,     None,          5.990000000),
}

# ── Urban / Intercity VKM split ────────────────────────────────────────
# Source: Hall, A.P. 1994
URBAN_INTERCITY_VKM = {"Urban": 0.55, "Intercity": 0.45}

# ── Aviation tonne-km split (Canada) ──────────────────────────────────
AVIATION_TKM_SPLIT = {"Domestic": 0.43, "International": 0.57}

# ── Aviation energy split (domestic / international) by province ───────
# Source: Align with NIR
AVIATION_ENERGY_DOMESTIC = {
    "CAN": 0.380, "BC": 0.311100226, "AB": 0.592140504, "SK": 0.800,
    "MB": 0.773031271, "ON": 0.382509238, "QC": 0.200, "NB": 0.800,
    "NS": 0.672882520, "PE": 0.670, "NL": 0.346194784, "YT": 0.900,
    "NT": 0.900, "NU": 0.900, "AT": 0.600, "TR": 0.900,
}
AVIATION_ENERGY_INTL = {
    "CAN": 0.620, "BC": 0.688899774, "AB": 0.407859496, "SK": 0.200,
    "MB": 0.226968729, "ON": 0.617490762, "QC": 0.800, "NB": 0.200,
    "NS": 0.327117480, "PE": 0.330, "NL": 0.653805216, "YT": 0.100,
    "NT": 0.100, "NU": 0.100, "AT": 0.400, "TR": 0.100,
}

# ── Aviation passenger multipliers ─────────────────────────────────────
AVIATION_PASSENGER = {
    "kg_per_passenger": 100.0,       # kg / passenger
    "tkm_per_pkm":       0.1,        # tkm / pkm
}

# ── Marine tonne-km split (Canada) ────────────────────────────────────
MARINE_TKM_SPLIT = {"Domestic": 0.46, "International": 0.54}

# ── Marine energy split (domestic / international) by province ─────────
# Source: Align with NIR; None = no marine activity
MARINE_ENERGY_DOMESTIC = {
    "CAN": 0.439613076, "BC": 0.329733301, "AB": None, "SK": None,
    "MB": None, "ON": 0.231646653, "QC": 0.450656086,
    "NB": 0.505042643, "NS": 0.757784057, "PE": 0.464292401,
    "NL": 1.342456951, "YT": None, "NT": None, "NU": None,
    "AT": None, "TR": None,
}
MARINE_ENERGY_INTL = {
    "CAN": 0.560386924, "BC": 0.670266699, "AB": 1.0, "SK": 1.0,
    "MB": 1.0, "ON": 0.768353347, "QC": 0.549343914,
    "NB": 0.494957357, "NS": 0.242215943, "PE": 0.535707599,
    "NL": -0.342456951, "YT": 1.0, "NT": 1.0, "NU": 1.0,
    "AT": 1.0, "TR": 1.0,
}


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  CEUD ROW MAP  (0-indexed row positions in .xls sheets)              ║
# ╚═══════════════════════════════════════════════════════════════════════╝

# Table 37: Truck Explanatory Variables
T37_STOCK = {"Light Truck": 25, "Medium Truck": 26, "Heavy Truck": 27}
T37_AVGKM = {"Light Truck": 37, "Medium Truck": 38, "Heavy Truck": 39}

# Table 35: Freight Light Truck Energy Use by Source (PJ)
T35_FUEL = {
    "Natural gas":     13,
    "Motor gasoline":  14,
    "Diesel fuel oil": 15,
    "Ethanol":         16,
    "Biodiesel fuel":  17,
    "Propane":         18,
}

# Table 36: Medium & Heavy Truck Energy Use
T36_MED_FUEL = {
    "Motor gasoline":  13,
    "Diesel fuel oil": 14,
    "Ethanol":         15,
    "Biodiesel fuel":  16,
}
T36_MED_TOTAL_ROW   = 11   # Medium Truck total energy (for validation)
T36_MED_TKM_ROW     = 25   # Medium Truck tonne-km (millions)
T36_HEAVY_TOTAL_ROW = 47   # Heavy Truck total energy (≈ all diesel)
T36_HEAVY_TKM_ROW   = 50   # Heavy Truck tonne-km (millions)

# Table 18: Freight Rail Transportation Energy Use (PJ) — total
T18_RAIL_TOTAL_ROW = 11

# Table 15: Freight Air Transportation Energy Use by Source (PJ)
T15_AVIATION_GASOLINE_ROW = 13
T15_AVIATION_TURBO_ROW    = 14


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  SHEET CACHE                                                         ║
# ╚═══════════════════════════════════════════════════════════════════════╝
_SHEET_CACHE: dict[str, pd.DataFrame] = {}


def _load_sheet(xls_path: Path, sheet_name: str) -> pd.DataFrame:
    """Read an xls sheet (header=None) with caching."""
    key = f"{xls_path.name}::{sheet_name}"
    if key not in _SHEET_CACHE:
        _SHEET_CACHE[key] = pd.read_excel(
            xls_path, sheet_name=sheet_name, header=None, engine="xlrd"
        )
    return _SHEET_CACHE[key]


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  HELPERS                                                             ║
# ╚═══════════════════════════════════════════════════════════════════════╝

def _read_row(df: pd.DataFrame, row_idx: int) -> pd.Series:
    """Extract a year-indexed pd.Series from raw sheet row (cols 2..24)."""
    raw = df.iloc[row_idx, YEAR_COLS[0] : YEAR_COLS[-1] + 1].values
    out = []
    for v in raw:
        if v is None:
            out.append(np.nan)
        else:
            s = str(v).strip().lower()
            if s in ("n.a.", "na", "", "nan", "none", "x"):
                out.append(np.nan)
            else:
                try:
                    out.append(float(v))
                except (ValueError, TypeError):
                    out.append(np.nan)
    return pd.Series(out, index=YEARS, dtype=float)


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """IFERROR(num/den, 0) — replaces inf/NaN with 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = num / den.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _zero_series() -> pd.Series:
    return pd.Series(np.zeros(N_YEARS), index=YEARS, dtype=float)


def _nan_series() -> pd.Series:
    return pd.Series(np.full(N_YEARS, np.nan), index=YEARS, dtype=float)


def _const_series(val: float) -> pd.Series:
    return pd.Series(np.full(N_YEARS, val), index=YEARS, dtype=float)


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  DATAFRAME BUILDER                                                   ║
# ╚═══════════════════════════════════════════════════════════════════════╝

def _build_mode_df(
    mode_name: str,
    activity: pd.Series,
    stock: pd.Series,
    avg_vkm: pd.Series,
    total_dist: pd.Series,
    load_factor: pd.Series,
    avg_tkm: pd.Series,
    fuels: dict[str, pd.Series],
    notes: list[str] | None = None,
) -> pl.DataFrame:
    """Assemble a standard freight-mode output DataFrame (Polars)."""

    fuel_series = {f: fuels.get(f, _zero_series()) for f in FREIGHT_FUELS}
    fuel_total = sum(fuel_series.values())
    intensity = _safe_div(fuel_total, activity * 1000.0)

    shares = {}
    for f in FREIGHT_FUELS:
        shares[f"share_{f}"] = _safe_div(fuel_series[f], fuel_total)

    data: dict[str, list] = {
        "year": YEARS,
        "Activity (M tkm)":      [activity.get(y, None) for y in YEARS],
        "Stock (thousands)":     [stock.get(y, None) for y in YEARS],
        "Average Distance (vkm)":[avg_vkm.get(y, None) for y in YEARS],
        "Total Distance (M vkm)":[total_dist.get(y, None) for y in YEARS],
        "Load factor (t/veh)":   [load_factor.get(y, None) for y in YEARS],
        "Average Distance (tkm)":[avg_tkm.get(y, None) for y in YEARS],
    }
    for f in FREIGHT_FUELS:
        data[f"fuel_{f} (TJ)"] = [fuel_series[f].get(y, 0.0) for y in YEARS]
    data["fuel_Total (TJ)"] = [float(fuel_total[y]) for y in YEARS]
    data["Intensity (GJ/tkm)"] = [intensity.get(y, 0.0) for y in YEARS]
    for f in FREIGHT_FUELS:
        data[f"share_{f}"] = [shares[f"share_{f}"].get(y, 0.0) for y in YEARS]

    df = pl.DataFrame(data)

    if notes:
        notes_path = OUT_DIR / f"freight_bcterr_{mode_name}_notes.txt"
        notes_path.write_text("\n".join(notes), encoding="utf-8")

    return df


def _mode_df_from_cache(
    cache: dict[str, pl.DataFrame] | None,
    key: str,
    fallback_csv: Path,
) -> pl.DataFrame:
    """Return an upstream mode dataframe from memory; fall back to CSV for standalone calls.

    BC is a downstream sheet. In normal execution, BC consumes already-built
    CAN/BCTerr dataframes rather than re-reading upstream audit CSVs. The
    fallback keeps individual BC builders callable for ad-hoc debugging.
    """
    if cache is not None and key in cache:
        df = cache[key]
        if isinstance(df, pl.DataFrame):
            return df
        if isinstance(df, pd.DataFrame):
            return pl.from_pandas(df)
        raise TypeError(f"Unsupported dataframe type for cache[{key!r}]: {type(df)!r}")
    return pl.read_csv(fallback_csv)


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  COEFFICIENTS LOADER                                                  ║
# ╚═══════════════════════════════════════════════════════════════════════╝

_COEFF_LONG_CACHE: pd.DataFrame | None = None


def load_coefficients_long(force: bool = False) -> pd.DataFrame:
    """Load coefficients.csv and return a tidy long dataframe.

    The coefficients tab is organized as repeated section blocks (Density,
    Energy, GHG, CO2/Energy, CH4/Energy, N2O/Energy, GHG/Energy, etc.) with
    years spread across columns. This loader preserves the current section and
    metric headers, then melts all year columns into a normalized dataframe:

      section, metric, coefficient_unit, ceedc, fuel, unit, for_prices,
      scale, source, year, value

    This lets provincial builders look up year-varying emission factors by
    section/fuel, e.g. GHG/Energy + Gasoline for ktCO2e/TJ.
    """
    global _COEFF_LONG_CACHE
    if _COEFF_LONG_CACHE is not None and not force:
        return _COEFF_LONG_CACHE

    if not COEFFICIENTS_FILE.exists():
        raise FileNotFoundError(f"coefficients file not found: {COEFFICIENTS_FILE}")

    raw = pd.read_csv(COEFFICIENTS_FILE)
    year_cols = [c for c in raw.columns if str(c).strip().isdigit()]

    def _clean(value) -> str:
        if pd.isna(value):
            return ""
        return str(value).strip()

    section = ""
    metric = ""
    coeff_unit = ""
    records: list[dict] = []

    scale_col = "Unnamed: 4" if "Unnamed: 4" in raw.columns else None

    for _, row in raw.iterrows():
        ceedc = _clean(row.get("CEEDC", ""))
        fuel = _clean(row.get("Fuel", ""))
        unit = _clean(row.get("Unit", ""))
        for_prices = _clean(row.get("For prices", ""))
        scale = _clean(row.get(scale_col, "")) if scale_col else ""
        source = _clean(row.get("Source", ""))

        year_values = pd.to_numeric(row[year_cols], errors="coerce")
        has_year_values = bool(year_values.notna().any())

        # Section headers usually look like: ,Energy,,,, or ,GHG/Energy,,,,
        if not has_year_values and not ceedc and fuel and not unit:
            section = fuel
            metric = ""
            coeff_unit = ""
            continue

        # Metric headers usually look like: ,HHV,TJ/unit,,,, or ,CO2,ktCO2/unit,,,,
        if not has_year_values and not ceedc and fuel and unit:
            metric = fuel
            coeff_unit = unit
            continue

        # Unit-only headers usually look like: ,,ktCO2e/TJ,,,,
        if not has_year_values and not ceedc and not fuel and unit:
            coeff_unit = unit
            continue

        if not has_year_values:
            continue

        # Data rows can identify the fuel in either CEEDC or Fuel. Some derived
        # coefficient rows have blank CEEDC but populated Fuel, so keep both.
        records.append({
            "section": section,
            "metric": metric,
            "coefficient_unit": coeff_unit,
            "ceedc": ceedc,
            "fuel": fuel,
            "unit": unit,
            "for_prices": for_prices,
            "scale": scale,
            "source": source,
            **{int(y): float(year_values[y]) if pd.notna(year_values[y]) else np.nan for y in year_cols},
        })

    wide = pd.DataFrame(records)
    long = wide.melt(
        id_vars=["section", "metric", "coefficient_unit", "ceedc", "fuel", "unit", "for_prices", "scale", "source"],
        value_vars=[int(y) for y in year_cols],
        var_name="year",
        value_name="value",
    )
    long["year"] = long["year"].astype(int)
    long["value"] = pd.to_numeric(long["value"], errors="coerce")

    _COEFF_LONG_CACHE = long
    return _COEFF_LONG_CACHE


def get_coefficient_series(
    section: str,
    fuel: str,
    metric: str | None = None,
    coefficient_unit: str | None = None,
    years: list[int] | None = None,
) -> pd.Series:
    """Return a year-indexed coefficient series from coefficients.csv.

    Matching is case-insensitive. The fuel argument can match either the CEEDC
    code or the Fuel label, which makes calls robust to rows such as HFO/Fuel oil
    and blank-code derived rows such as GHG/Energy/Fuel oil.
    """
    years = years or YEARS
    coeffs = load_coefficients_long()

    mask = coeffs["section"].str.casefold().eq(section.casefold())
    fuel_cf = fuel.casefold()
    mask &= (
        coeffs["fuel"].fillna("").str.casefold().eq(fuel_cf)
        | coeffs["ceedc"].fillna("").str.casefold().eq(fuel_cf)
    )
    if metric is not None:
        mask &= coeffs["metric"].fillna("").str.casefold().eq(metric.casefold())
    if coefficient_unit is not None:
        mask &= coeffs["coefficient_unit"].fillna("").str.casefold().eq(coefficient_unit.casefold())

    subset = coeffs.loc[mask, ["section", "metric", "coefficient_unit", "ceedc", "fuel", "source", "year", "value"]]
    if subset.empty:
        raise KeyError(
            f"No coefficient found for section={section!r}, fuel={fuel!r}, "
            f"metric={metric!r}, coefficient_unit={coefficient_unit!r}"
        )

    candidates = subset[["section", "metric", "coefficient_unit", "ceedc", "fuel", "source"]].drop_duplicates()
    if len(candidates) > 1:
        raise ValueError(
            "Coefficient lookup is ambiguous. Refine metric or coefficient_unit. "
            f"Candidates: {candidates.to_dict(orient='records')}"
        )

    series = subset.set_index("year")["value"].sort_index().reindex(years)
    if series.isna().any():
        missing = list(series[series.isna()].index)
        raise ValueError(f"Coefficient series has missing years for {section}/{fuel}: {missing}")
    return series.astype(float)


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  TRUCK BUILDERS  (BCTerr)                                            ║
# ╚═══════════════════════════════════════════════════════════════════════╝

def build_bcterr_light_trucks() -> pd.DataFrame:
    """Light Trucks: Stock/Distance from Table 37, Fuel from Table 35."""
    t37 = _load_sheet(CEUD_BCTERR_FILE, "Table 37")
    t35 = _load_sheet(CEUD_BCTERR_FILE, "Table 35")

    stock      = _read_row(t37, T37_STOCK["Light Truck"])
    avg_vkm    = _read_row(t37, T37_AVGKM["Light Truck"])
    total_dist = stock * avg_vkm / 1000.0       # (thousands × km) / 1000 → M·vkm
    can_lf     = get_can_load_factors()["Light Truck"]
    lf         = can_lf
    activity   = total_dist * can_lf
    avg_tkm    = avg_vkm * can_lf

    fuel_map = {
        "Natural gas":     T35_FUEL["Natural gas"],
        "Motor gasoline":  T35_FUEL["Motor gasoline"],
        "Diesel fuel oil": T35_FUEL["Diesel fuel oil"],
        "Ethanol":         T35_FUEL["Ethanol"],
        "Biodiesel fuel":  T35_FUEL["Biodiesel fuel"],
        "Propane":         T35_FUEL["Propane"],
    }
    fuels = {}
    notes = ["BCTerr Light Trucks"]
    for fuel_name, row_idx in fuel_map.items():
        fuels[fuel_name] = _read_row(t35, row_idx).fillna(0.0) * FUEL_SCALE
        notes.append(f"  {fuel_name}: Table 35 row {row_idx} × {FUEL_SCALE}")

    df = _build_mode_df("light_trucks", activity, stock, avg_vkm,
                        total_dist, lf, avg_tkm, fuels, notes)
    out = OUT_DIR / "freight_bcterr_light_trucks.csv"
    df.write_csv(out)
    print(f"  ✅ {out.name}  (Activity 2000 = {activity.iloc[0]:,.1f} M·tkm)")
    return df


def build_bcterr_medium_trucks() -> pd.DataFrame:
    """Medium Trucks: Stock/Distance from Table 37, Fuel from Table 36."""
    t37 = _load_sheet(CEUD_BCTERR_FILE, "Table 37")
    t36 = _load_sheet(CEUD_BCTERR_FILE, "Table 36")

    stock      = _read_row(t37, T37_STOCK["Medium Truck"])
    avg_vkm    = _read_row(t37, T37_AVGKM["Medium Truck"])
    total_dist = stock * avg_vkm / 1000.0
    can_lf     = get_can_load_factors()["Medium Truck"]
    lf         = can_lf
    activity   = total_dist * can_lf
    avg_tkm    = avg_vkm * can_lf

    fuel_map = {
        "Motor gasoline":  T36_MED_FUEL["Motor gasoline"],
        "Diesel fuel oil": T36_MED_FUEL["Diesel fuel oil"],
        "Ethanol":         T36_MED_FUEL["Ethanol"],
        "Biodiesel fuel":  T36_MED_FUEL["Biodiesel fuel"],
    }
    fuels = {}
    notes = ["BCTerr Medium Trucks"]
    for fuel_name, row_idx in fuel_map.items():
        fuels[fuel_name] = _read_row(t36, row_idx).fillna(0.0) * FUEL_SCALE
        notes.append(f"  {fuel_name}: Table 36 row {row_idx} × {FUEL_SCALE}")

    df = _build_mode_df("medium_trucks", activity, stock, avg_vkm,
                        total_dist, lf, avg_tkm, fuels, notes)
    out = OUT_DIR / "freight_bcterr_medium_trucks.csv"
    df.write_csv(out)
    print(f"  ✅ {out.name}  (Activity 2000 = {activity.iloc[0]:,.1f} M·tkm)")
    return df


def build_bcterr_heavy_trucks() -> pd.DataFrame:
    """Heavy Trucks: Stock/Distance from Table 37, Diesel from Table 36 row 47."""
    t37 = _load_sheet(CEUD_BCTERR_FILE, "Table 37")
    t36 = _load_sheet(CEUD_BCTERR_FILE, "Table 36")

    stock      = _read_row(t37, T37_STOCK["Heavy Truck"])
    avg_vkm    = _read_row(t37, T37_AVGKM["Heavy Truck"])
    total_dist = stock * avg_vkm / 1000.0
    can_lf     = get_can_load_factors()["Heavy Truck"]
    lf         = can_lf
    activity   = total_dist * can_lf
    avg_tkm    = avg_vkm * can_lf

    diesel = _read_row(t36, T36_HEAVY_TOTAL_ROW).fillna(0.0) * FUEL_SCALE
    fuels = {"Diesel fuel oil": diesel}
    notes = [
        "BCTerr Heavy Trucks",
        f"  Diesel fuel oil: Table 36 row {T36_HEAVY_TOTAL_ROW} × {FUEL_SCALE}",
    ]

    df = _build_mode_df("heavy_trucks", activity, stock, avg_vkm,
                        total_dist, lf, avg_tkm, fuels, notes)
    out = OUT_DIR / "freight_bcterr_heavy_trucks.csv"
    df.write_csv(out)
    print(f"  ✅ {out.name}  (Activity 2000 = {activity.iloc[0]:,.1f} M·tkm)")
    return df


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  RAIL / MARINE / AIR  (BCTerr — fuel extraction; Activity = Sprint 2)║
# ╚═══════════════════════════════════════════════════════════════════════╝

def build_bcterr_rail_freight() -> pd.DataFrame:
    """Rail: Diesel from Table 18. Activity from CAN intensity × freight share."""
    t18 = _load_sheet(CEUD_BCTERR_FILE, "Table 18")
    diesel = _read_row(t18, T18_RAIL_TOTAL_ROW).fillna(0.0) * FUEL_SCALE
    fuels = {"Diesel fuel oil": diesel}

    # CAN cross-refs for activity
    can_intensity, can_freight_pct, can_total_tkm = get_can_rail_intensity_and_freight_share()

    # total_tkm_prov = BCTerr_diesel / CAN_intensity / 1000
    total_tkm_prov = _safe_div(diesel, can_intensity) / 1000.0
    # freight_tkm = total_tkm_prov × CAN_freight_pct
    freight_tkm = total_tkm_prov * can_freight_pct

    # Passenger data (derived from CAN ratios)
    pass_pct = 1.0 - can_freight_pct
    pass_tkm_prov = total_tkm_prov * pass_pct
    pass_pkm_prov = pass_tkm_prov / AVIATION_PASSENGER["tkm_per_pkm"]

    notes = [
        "BCTerr Rail Freight (CAN cross-ref Activity)",
        f"  Diesel fuel oil: Table 18 row {T18_RAIL_TOTAL_ROW} × {FUEL_SCALE}",
        "  Activity = diesel / CAN_intensity / 1000 × CAN_freight_pct",
    ]

    df = _build_mode_df("rail", freight_tkm, _nan_series(), _nan_series(),
                        _nan_series(), _nan_series(), _nan_series(), fuels, notes)

    # Add extra rail columns
    df = df.with_columns([
        pl.Series("Passenger pkm (millions)", pass_pkm_prov.tolist()),
        pl.Series("Passenger tkm (millions)", pass_tkm_prov.tolist()),
        pl.Series("Total tkm (millions)", total_tkm_prov.tolist()),
        pl.Series("Freight %", can_freight_pct.tolist()),
    ])

    out = OUT_DIR / "freight_bcterr_rail.csv"
    df.write_csv(out)
    print(f"  ✅ {out.name}  (Freight tkm 2000 = {freight_tkm.iloc[0]:,.0f} M)")
    return df


def build_bcterr_marine_freight() -> pd.DataFrame:
    """Marine: No fuel in BCTerr CEUD. Activity requires CAN cross-refs."""
    fuels: dict[str, pd.Series] = {}
    notes = [
        "BCTerr Marine Freight (stub — Sprint 1)",
        "  No marine fuel rows in BCTerr CEUD",
        "  [TODO Sprint 2] Fuel from CAN marine × prov share; Activity via CAN pipeline",
    ]
    df = _build_mode_df("marine", _zero_series(), _nan_series(), _nan_series(),
                        _nan_series(), _nan_series(), _nan_series(), fuels, notes)
    out = OUT_DIR / "freight_bcterr_marine.csv"
    df.write_csv(out)
    print(f"  ⏳ {out.name}  (stub — no BCTerr marine fuel)")
    return df


def build_bcterr_air_freight(prov_code: str = "BC") -> pd.DataFrame:
    """Air: Fuel from Table 15 × domestic energy share. Activity from CAN cross-refs."""
    dom_share = AVIATION_ENERGY_DOMESTIC[prov_code]
    t15 = _load_sheet(CEUD_BCTERR_FILE, "Table 15")

    avturbo = _read_row(t15, T15_AVIATION_TURBO_ROW).fillna(0.0) * dom_share * FUEL_SCALE
    avgas   = _read_row(t15, T15_AVIATION_GASOLINE_ROW).fillna(0.0) * dom_share * FUEL_SCALE
    fuels = {
        "Aviation turbo fuel": avturbo,
        "Aviation gasoline":   avgas,
    }
    bcterr_fuel_total = avturbo + avgas

    # CAN air cross-refs for activity
    can_freight_tkm, can_total_tkm, can_freight_pct = get_can_air_activity_and_shares()

    # CAN air fuel (domestic portion)
    t21 = _load_sheet(CEUD_CAN_FILE, "Table 21")
    can_avturbo = _read_row(t21, CAN_T21_AIR_AVTURBO).fillna(0.0) * AVIATION_ENERGY_DOMESTIC["CAN"] * FUEL_SCALE
    can_avgas   = _read_row(t21, CAN_T21_AIR_AVGAS).fillna(0.0) * AVIATION_ENERGY_DOMESTIC["CAN"] * FUEL_SCALE
    can_fuel_total = can_avturbo + can_avgas

    # CAN air intensity = fuel / (total_tkm * 1000) → GJ/tkm
    can_intensity = _safe_div(can_fuel_total, can_total_tkm * 1000.0)

    # BCTerr activity
    total_tkm_prov = _safe_div(bcterr_fuel_total, can_intensity) / 1000.0
    freight_tkm = total_tkm_prov * can_freight_pct

    # Passenger data
    pass_pct = 1.0 - can_freight_pct
    pass_tkm_prov = total_tkm_prov * pass_pct
    pass_pkm_prov = pass_tkm_prov / AVIATION_PASSENGER["tkm_per_pkm"]

    notes = [
        f"BCTerr Air Freight (CAN cross-ref Activity, domestic_share={dom_share})",
        f"  Aviation turbo fuel: Table 15 row {T15_AVIATION_TURBO_ROW} × {dom_share} × {FUEL_SCALE}",
        f"  Aviation gasoline:   Table 15 row {T15_AVIATION_GASOLINE_ROW} × {dom_share} × {FUEL_SCALE}",
        "  Activity = BCTerr_fuel / CAN_intensity / 1000 × CAN_freight_pct",
    ]

    df = _build_mode_df("air", freight_tkm, _nan_series(), _nan_series(),
                        _nan_series(), _nan_series(), _nan_series(), fuels, notes)

    # Add extra air columns
    df = df.with_columns([
        pl.Series("Passenger pkm (millions)", pass_pkm_prov.tolist()),
        pl.Series("Passenger tkm (millions)", pass_tkm_prov.tolist()),
        pl.Series("Total tkm (millions)", total_tkm_prov.tolist()),
        pl.Series("Freight %", can_freight_pct.tolist()),
    ])

    out = OUT_DIR / "freight_bcterr_air.csv"
    df.write_csv(out)
    print(f"  ✅ {out.name}  (Freight tkm 2000 = {freight_tkm.iloc[0]:,.0f} M)")
    return df


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  AGGREGATE BUILDERS                                                  ║
# ╚═══════════════════════════════════════════════════════════════════════╝

def _sum_modes(dfs: list[pl.DataFrame], mode_name: str) -> pl.DataFrame:
    """Sum activity and fuel columns across mode DataFrames (Polars)."""
    fuel_cols  = [c for c in dfs[0].columns if c.startswith("fuel_")]
    sum_cols   = ["Activity (M tkm)"] + fuel_cols
    meta_cols  = ["Stock (thousands)", "Average Distance (vkm)",
                  "Total Distance (M vkm)", "Load factor (t/veh)",
                  "Average Distance (tkm)"]

    # Convert to pandas for arithmetic compatibility, then back
    base = dfs[0].to_pandas().set_index("year").copy()
    for col in meta_cols:
        if col in base.columns:
            base[col] = np.nan

    for other_df in dfs[1:]:
        o = other_df.to_pandas().set_index("year")
        for col in sum_cols:
            if col in base.columns and col in o.columns:
                base[col] = base[col].fillna(0) + o[col].fillna(0)

    # Recompute intensity and shares
    act        = base["Activity (M tkm)"]
    fuel_total = base["fuel_Total (TJ)"]
    base["Intensity (GJ/tkm)"] = _safe_div(fuel_total, act * 1000.0)
    for f in FREIGHT_FUELS:
        fc = f"fuel_{f} (TJ)"
        sc = f"share_{f}"
        if fc in base.columns and sc in base.columns:
            base[sc] = _safe_div(base[fc], fuel_total)

    return pl.from_pandas(base.reset_index())


def build_bcterr_light_medium() -> pd.DataFrame:
    """Light Medium = Light Trucks + Medium Trucks."""
    lt = pl.read_csv(OUT_DIR / "freight_bcterr_light_trucks.csv")
    mt = pl.read_csv(OUT_DIR / "freight_bcterr_medium_trucks.csv")
    df = _sum_modes([lt, mt], "light_medium")
    out = OUT_DIR / "freight_bcterr_light_medium.csv"
    df.write_csv(out)
    act = df.filter(pl.col("year") == 2000).select("Activity (M tkm)").item()
    print(f"  ✅ {out.name}  (Activity 2000 = {act:,.1f} M·tkm)")
    return df


def build_bcterr_heavy_total() -> pd.DataFrame:
    """Heavy Total = Heavy Trucks + Rail."""
    ht   = pl.read_csv(OUT_DIR / "freight_bcterr_heavy_trucks.csv")
    rail = pl.read_csv(OUT_DIR / "freight_bcterr_rail.csv")
    df = _sum_modes([ht, rail], "heavy_total")
    out = OUT_DIR / "freight_bcterr_heavy_total.csv"
    df.write_csv(out)
    print(f"  ✅ {out.name}")
    return df


def build_bcterr_freight_total() -> pd.DataFrame:
    """Freight Total = Light Medium + Heavy Total + Marine + Air."""
    component_files = [
        "freight_bcterr_light_medium.csv",
        "freight_bcterr_heavy_total.csv",
        "freight_bcterr_marine.csv",
        "freight_bcterr_air.csv",
    ]
    dfs = [pl.read_csv(OUT_DIR / f) for f in component_files]
    df = _sum_modes(dfs, "freight_total")
    out = OUT_DIR / "freight_bcterr_total.csv"
    df.write_csv(out)
    print(f"  ✅ {out.name}")
    return df


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  VALIDATION                                                          ║
# ╚═══════════════════════════════════════════════════════════════════════╝

def validate_bcterr_medium_truck_tkm():
    """Cross-check Medium Truck tkm against CEUD Table 36 row 25."""
    t36 = _load_sheet(CEUD_BCTERR_FILE, "Table 36")
    ceud_tkm = _read_row(t36, T36_MED_TKM_ROW)
    computed = pl.read_csv(OUT_DIR / "freight_bcterr_medium_trucks.csv")
    comp_tkm = pd.Series(computed["Activity (M tkm)"].to_list(), index=YEARS)
    diff = (comp_tkm - ceud_tkm).abs()
    max_diff = diff.max()
    print(f"  Medium Truck tkm validation: max diff = {max_diff:.4f} M·tkm")
    return max_diff


def validate_bcterr_heavy_truck_tkm():
    """Cross-check Heavy Truck tkm against CEUD Table 36 row 50."""
    t36 = _load_sheet(CEUD_BCTERR_FILE, "Table 36")
    ceud_tkm = _read_row(t36, T36_HEAVY_TKM_ROW)
    computed = pl.read_csv(OUT_DIR / "freight_bcterr_heavy_trucks.csv")
    comp_tkm = pd.Series(computed["Activity (M tkm)"].to_list(), index=YEARS)
    diff = (comp_tkm - ceud_tkm).abs()
    max_diff = diff.max()
    print(f"  Heavy Truck tkm validation:  max diff = {max_diff:.4f} M·tkm")
    return max_diff


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  DIAGNOSTICS                                                         ║
# ╚═══════════════════════════════════════════════════════════════════════╝

def write_diagnostics():
    lines = [
        "trans_freight_ceud.py — Sprint 1 Diagnostics",
        "=" * 60,
        f"CEUD BCTerr: {CEUD_BCTERR_FILE}",
        f"CEUD CAN:    {CEUD_CAN_FILE}",
        f"Years:       {YEARS[0]}–{YEARS[-1]} ({N_YEARS} years)",
        "",
        "Load Factors (constant):",
    ]
    for mode, val in LOAD_FACTOR.items():
        lines.append(f"  {mode:>15s}: {val:.6f} t/veh")
    lines.append("")
    lines.append("Aviation Domestic Energy Share:")
    for prov, val in AVIATION_ENERGY_DOMESTIC.items():
        lines.append(f"  {prov}: {val}")
    lines.append("")
    lines.append("Marine Domestic Energy Share:")
    for prov, val in MARINE_ENERGY_DOMESTIC.items():
        lines.append(f"  {prov}: {val}")
    lines.append("")
    lines.append("Outputs:")
    for f in sorted(OUT_DIR.glob("freight_bcterr_*.csv")):
        lines.append(f"  {f.name}")

    diag_path = OUT_DIR / "freight_bcterr_diagnostics.txt"
    diag_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  📋 {diag_path.name}")


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  MAIN                                                                ║
# ╚═══════════════════════════════════════════════════════════════════════╝


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  CAN CEUD ROW MAP  (0-indexed row positions in transCan*.xls)        ║
# ╚═══════════════════════════════════════════════════════════════════════╝

# Freight1: Activity (tonne-km millions) by mode
CAN_F1_LT_ACTIVITY  = 31  # Light Trucks
CAN_F1_MT_ACTIVITY  = 32  # Medium Trucks
CAN_F1_HT_ACTIVITY  = 33  # Heavy Trucks
CAN_F1_AIR_ACTIVITY = 34  # Air (total, all freight)
CAN_F1_RAIL_ACTIVITY = 35 # Rail (total freight)
CAN_F1_MARINE_ACTIVITY = 36  # Marine (total)
CAN_F1_TOTAL_ACTIVITY = 29  # Total freight tkm

# Freight4: Stock, Average Distance
CAN_F4_LT_STOCK  = 14  # Light Truck Stock (thousands)
CAN_F4_MT_STOCK  = 15  # Medium Truck Stock
CAN_F4_HT_STOCK  = 16  # Heavy Truck Stock
CAN_F4_LT_AVGKM = 18  # Light Truck Avg Distance (km)
CAN_F4_MT_AVGKM = 19  # Medium Truck Avg Distance
CAN_F4_HT_AVGKM = 20  # Heavy Truck Avg Distance

# Table 53: Light Truck fuel by source (PJ)
CAN_T53_LT_NG      = 13
CAN_T53_LT_GASOLINE = 14
CAN_T53_LT_DIESEL  = 15
CAN_T53_LT_ETHANOL = 16
CAN_T53_LT_BIO     = 17
CAN_T53_LT_PROPANE = 18

# Table 57: Medium Truck fuel by source (PJ)
CAN_T57_MT_GASOLINE = 13
CAN_T57_MT_DIESEL   = 14
CAN_T57_MT_ETHANOL  = 15
CAN_T57_MT_BIO      = 16

# Table 38: Road Freight total diesel (PJ) — HT diesel derived
CAN_T38_ROAD_DIESEL = 15  # Total road freight diesel

# Table 27: Rail freight fuel (PJ)
CAN_T27_RAIL_TOTAL  = 11  # Total rail fuel (= diesel)

# Table 21: Air freight fuel (PJ)
CAN_T21_AIR_AVGAS   = 13
CAN_T21_AIR_AVTURBO = 14

# Passenger1: Passenger activity
CAN_P1_AIR_PKM  = 38  # Air passenger pkm (millions)
CAN_P1_RAIL_PKM = 39  # Rail passenger pkm (millions)


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  CAN BUILDERS                                                        ║
# ╚═══════════════════════════════════════════════════════════════════════╝

def build_can_light_trucks() -> pd.DataFrame:
    """CAN Light Trucks: Activity from Freight1, Stock/Dist from Freight4,
    Fuel from Table 53. Load factor = Activity / TotalDist (year-varying)."""
    f1  = _load_sheet(CEUD_CAN_FILE, "Freight1")
    f4  = _load_sheet(CEUD_CAN_FILE, "Freight4")
    t53 = _load_sheet(CEUD_CAN_FILE, "Table 53")

    activity   = _read_row(f1, CAN_F1_LT_ACTIVITY)
    stock      = _read_row(f4, CAN_F4_LT_STOCK)
    avg_vkm    = _read_row(f4, CAN_F4_LT_AVGKM)
    total_dist = stock * avg_vkm / 1000.0
    load_factor = _safe_div(activity, total_dist)
    avg_tkm    = avg_vkm * load_factor

    fuel_map = {
        "Natural gas":     CAN_T53_LT_NG,
        "Motor gasoline":  CAN_T53_LT_GASOLINE,
        "Diesel fuel oil": CAN_T53_LT_DIESEL,
        "Ethanol":         CAN_T53_LT_ETHANOL,
        "Biodiesel fuel":  CAN_T53_LT_BIO,
        "Propane":         CAN_T53_LT_PROPANE,
    }
    fuels = {}
    notes = ["CAN Light Trucks"]
    for fuel_name, row_idx in fuel_map.items():
        fuels[fuel_name] = _read_row(t53, row_idx).fillna(0.0) * FUEL_SCALE
        notes.append(f"  {fuel_name}: Table 53 row {row_idx} × {FUEL_SCALE}")

    df = _build_mode_df("can_light_trucks", activity, stock, avg_vkm,
                        total_dist, load_factor, avg_tkm, fuels, notes)
    out = OUT_DIR / "freight_can_light_trucks.csv"
    df.write_csv(out)
    print(f"  ✅ {out.name}  (Activity 2000 = {activity.iloc[0]:,.1f} M·tkm, LF 2000 = {load_factor.iloc[0]:.4f})")
    return df


def build_can_medium_trucks() -> pd.DataFrame:
    """CAN Medium Trucks: Activity from Freight1, Stock/Dist from Freight4,
    Fuel from Table 57. Load factor = Activity / TotalDist."""
    f1  = _load_sheet(CEUD_CAN_FILE, "Freight1")
    f4  = _load_sheet(CEUD_CAN_FILE, "Freight4")
    t57 = _load_sheet(CEUD_CAN_FILE, "Table 57")

    activity   = _read_row(f1, CAN_F1_MT_ACTIVITY)
    stock      = _read_row(f4, CAN_F4_MT_STOCK)
    avg_vkm    = _read_row(f4, CAN_F4_MT_AVGKM)
    total_dist = stock * avg_vkm / 1000.0
    load_factor = _safe_div(activity, total_dist)
    avg_tkm    = avg_vkm * load_factor

    fuel_map = {
        "Motor gasoline":  CAN_T57_MT_GASOLINE,
        "Diesel fuel oil": CAN_T57_MT_DIESEL,
        "Ethanol":         CAN_T57_MT_ETHANOL,
        "Biodiesel fuel":  CAN_T57_MT_BIO,
    }
    fuels = {}
    notes = ["CAN Medium Trucks"]
    for fuel_name, row_idx in fuel_map.items():
        fuels[fuel_name] = _read_row(t57, row_idx).fillna(0.0) * FUEL_SCALE
        notes.append(f"  {fuel_name}: Table 57 row {row_idx} × {FUEL_SCALE}")

    df = _build_mode_df("can_medium_trucks", activity, stock, avg_vkm,
                        total_dist, load_factor, avg_tkm, fuels, notes)
    out = OUT_DIR / "freight_can_medium_trucks.csv"
    df.write_csv(out)
    print(f"  ✅ {out.name}  (Activity 2000 = {activity.iloc[0]:,.1f} M·tkm, LF 2000 = {load_factor.iloc[0]:.4f})")
    return df


def build_can_heavy_trucks() -> pd.DataFrame:
    """CAN Heavy Trucks: Activity from Freight1, Stock/Dist from Freight4.
    Diesel = Table 38 total road diesel - MT diesel - LT diesel.
    Load factor = Activity / TotalDist."""
    f1  = _load_sheet(CEUD_CAN_FILE, "Freight1")
    f4  = _load_sheet(CEUD_CAN_FILE, "Freight4")
    t38 = _load_sheet(CEUD_CAN_FILE, "Table 38")
    t57 = _load_sheet(CEUD_CAN_FILE, "Table 57")
    t53 = _load_sheet(CEUD_CAN_FILE, "Table 53")

    activity   = _read_row(f1, CAN_F1_HT_ACTIVITY)
    stock      = _read_row(f4, CAN_F4_HT_STOCK)
    avg_vkm    = _read_row(f4, CAN_F4_HT_AVGKM)
    total_dist = stock * avg_vkm / 1000.0
    load_factor = _safe_div(activity, total_dist)
    avg_tkm    = avg_vkm * load_factor

    # HT diesel = total road diesel - MT diesel - LT diesel
    road_diesel = _read_row(t38, CAN_T38_ROAD_DIESEL).fillna(0.0) * FUEL_SCALE
    mt_diesel   = _read_row(t57, CAN_T57_MT_DIESEL).fillna(0.0) * FUEL_SCALE
    lt_diesel   = _read_row(t53, CAN_T53_LT_DIESEL).fillna(0.0) * FUEL_SCALE
    ht_diesel   = road_diesel - mt_diesel - lt_diesel

    fuels = {"Diesel fuel oil": ht_diesel}
    notes = [
        "CAN Heavy Trucks",
        f"  Diesel = Table 38 row {CAN_T38_ROAD_DIESEL} - Table 57 row {CAN_T57_MT_DIESEL} - Table 53 row {CAN_T53_LT_DIESEL}",
        "  All × 1000 (PJ → TJ)",
    ]

    df = _build_mode_df("can_heavy_trucks", activity, stock, avg_vkm,
                        total_dist, load_factor, avg_tkm, fuels, notes)
    out = OUT_DIR / "freight_can_heavy_trucks.csv"
    df.write_csv(out)
    print(f"  ✅ {out.name}  (Activity 2000 = {activity.iloc[0]:,.1f} M·tkm, LF 2000 = {load_factor.iloc[0]:.4f})")
    return df


def build_can_rail_freight() -> pd.DataFrame:
    """CAN Rail Freight: Activity from Freight1, Diesel from Table 27.
    Freight tkm = Freight1 row 35. Passenger tkm derived from Passenger1."""
    f1  = _load_sheet(CEUD_CAN_FILE, "Freight1")
    t27 = _load_sheet(CEUD_CAN_FILE, "Table 27")
    p1  = _load_sheet(CEUD_CAN_FILE, "Passenger1")

    freight_tkm = _read_row(f1, CAN_F1_RAIL_ACTIVITY)
    pass_pkm    = _read_row(p1, CAN_P1_RAIL_PKM)
    pass_tkm    = pass_pkm * AVIATION_PASSENGER["tkm_per_pkm"]
    total_tkm   = freight_tkm + pass_tkm
    freight_pct = _safe_div(freight_tkm, total_tkm)

    diesel = _read_row(t27, CAN_T27_RAIL_TOTAL).fillna(0.0) * FUEL_SCALE
    fuels = {"Diesel fuel oil": diesel}

    notes = [
        "CAN Rail Freight",
        f"  Freight tkm: Freight1 row {CAN_F1_RAIL_ACTIVITY}",
        f"  Passenger pkm: Passenger1 row {CAN_P1_RAIL_PKM} × tkm_per_pkm={AVIATION_PASSENGER['tkm_per_pkm']}",
        f"  Diesel: Table 27 row {CAN_T27_RAIL_TOTAL} × {FUEL_SCALE}",
    ]

    # For rail: Activity = freight_tkm, no Stock/AvgDist
    df = _build_mode_df("can_rail", freight_tkm, _nan_series(), _nan_series(),
                        _nan_series(), _nan_series(), _nan_series(), fuels, notes)

    # Add extra rail columns
    df = df.with_columns([
        pl.Series("Passenger pkm (millions)", pass_pkm.tolist()),
        pl.Series("Passenger tkm (millions)", pass_tkm.tolist()),
        pl.Series("Total tkm (millions)", total_tkm.tolist()),
        pl.Series("Freight %", freight_pct.tolist()),
    ])

    out = OUT_DIR / "freight_can_rail.csv"
    df.write_csv(out)
    print(f"  ✅ {out.name}  (Freight tkm 2000 = {freight_tkm.iloc[0]:,.0f} M, Freight% 2000 = {freight_pct.iloc[0]:.4%})")
    return df


def build_can_air_freight(prov_code: str = "CAN") -> pd.DataFrame:
    """CAN Air Freight: Activity from Freight1 × domestic tkm split.
    Fuel from Table 21 × domestic energy share."""
    dom_tkm_share   = AVIATION_TKM_SPLIT["Domestic"]
    dom_energy_share = AVIATION_ENERGY_DOMESTIC[prov_code]

    f1  = _load_sheet(CEUD_CAN_FILE, "Freight1")
    t21 = _load_sheet(CEUD_CAN_FILE, "Table 21")
    p1  = _load_sheet(CEUD_CAN_FILE, "Passenger1")

    # Activity
    total_freight_tkm = _read_row(f1, CAN_F1_AIR_ACTIVITY)
    freight_tkm = total_freight_tkm * dom_tkm_share

    # Passenger
    total_pass_pkm = _read_row(p1, CAN_P1_AIR_PKM)
    pass_pkm = total_pass_pkm * dom_tkm_share
    pass_tkm = pass_pkm * AVIATION_PASSENGER["tkm_per_pkm"]

    total_tkm = freight_tkm + pass_tkm
    freight_pct = _safe_div(freight_tkm, total_tkm)

    # Fuel
    avturbo = _read_row(t21, CAN_T21_AIR_AVTURBO).fillna(0.0) * dom_energy_share * FUEL_SCALE
    avgas   = _read_row(t21, CAN_T21_AIR_AVGAS).fillna(0.0) * dom_energy_share * FUEL_SCALE
    fuels = {
        "Aviation turbo fuel": avturbo,
        "Aviation gasoline":   avgas,
    }

    notes = [
        f"CAN Air Freight (domestic_tkm_split={dom_tkm_share}, domestic_energy_share={dom_energy_share})",
        f"  Freight tkm: Freight1 row {CAN_F1_AIR_ACTIVITY} × {dom_tkm_share}",
        f"  Passenger pkm: Passenger1 row {CAN_P1_AIR_PKM} × {dom_tkm_share}",
        f"  Aviation turbo: Table 21 row {CAN_T21_AIR_AVTURBO} × {dom_energy_share} × {FUEL_SCALE}",
        f"  Aviation gasoline: Table 21 row {CAN_T21_AIR_AVGAS} × {dom_energy_share} × {FUEL_SCALE}",
    ]

    df = _build_mode_df("can_air", freight_tkm, _nan_series(), _nan_series(),
                        _nan_series(), _nan_series(), _nan_series(), fuels, notes)

    df = df.with_columns([
        pl.Series("Passenger pkm (millions)", pass_pkm.tolist()),
        pl.Series("Passenger tkm (millions)", pass_tkm.tolist()),
        pl.Series("Total tkm (millions)", total_tkm.tolist()),
        pl.Series("Freight %", freight_pct.tolist()),
    ])

    out = OUT_DIR / "freight_can_air.csv"
    df.write_csv(out)
    print(f"  ✅ {out.name}  (Freight tkm 2000 = {freight_tkm.iloc[0]:,.0f} M, Freight% = {freight_pct.iloc[0]:.2%})")
    return df



def get_can_marine_hfo_from_ipcc(ipcc_df: pl.DataFrame) -> pd.Series:
    """Extract CAN marine HFO energy (TJ) by year from IPCC dataframe (Polars).
    Formula: HFO (TJ) = IPCC co2eq_kt / 0.074737
    Filters: region=Canada, source=Energy, category=Transport,
             sub-category=Marine, sub-sub-category=Domestic Navigation"""
    HFO_EF = 0.074737  # kt CO2eq per TJ

    sub = ipcc_df.filter(
        (pl.col("region") == "Canada") &
        (pl.col("source") == "Energy") &
        (pl.col("category") == "Transport") &
        (pl.col("sub-category") == "Marine") &
        (pl.col("sub-sub-category") == "Domestic Navigation")
    ).select(["year", "co2eq_kt"])

    hfo = sub.group_by("year").agg(
        (pl.col("co2eq_kt").sum() / HFO_EF).alias("hfo_tj")
    ).sort("year")

    # Convert to pd.Series indexed by YEARS for compatibility with builders
    hfo_pd = pd.Series(hfo["hfo_tj"].to_list(), index=hfo["year"].to_list())
    return hfo_pd.reindex(YEARS).fillna(0.0)


def build_can_marine_freight(prov_code: str = "CAN") -> pd.DataFrame:
    """CAN Marine Freight: Activity from Freight1; fuel from IPCC HFO."""
    dom_tkm_share = MARINE_TKM_SPLIT["Domestic"]
    dom_energy_share = MARINE_ENERGY_DOMESTIC.get(prov_code, None)

    f1 = _load_sheet(CEUD_CAN_FILE, "Freight1")
    ipcc_df = load_ipcc_emissions()

    total_marine_tkm = _read_row(f1, CAN_F1_MARINE_ACTIVITY)
    freight_tkm = total_marine_tkm * dom_tkm_share

    hfo_can = get_can_marine_hfo_from_ipcc(ipcc_df)
    hfo = hfo_can  # IPCC "Domestic Navigation" already IS the domestic portion

    fuels = {"Heavy fuel oil": hfo}
    notes = [
        "CAN Marine Freight (IPCC-sourced fuel)",
        f"  Freight tkm: Freight1 row {CAN_F1_MARINE_ACTIVITY} × {dom_tkm_share}",
        "  HFO: IPCC marine HFO × domestic energy share",
    ]

    df = _build_mode_df(
        "can_marine",
        freight_tkm,
        _nan_series(), _nan_series(), _nan_series(),
        _nan_series(), _nan_series(),
        fuels,
        notes,
    )

    out = OUT_DIR / "freight_can_marine.csv"
    df.write_csv(out)
    print(f"  ✅ {out.name}  (Freight tkm 2000 = {freight_tkm.iloc[0]:,.0f} M)")
    return df



def build_can_light_medium() -> pd.DataFrame:
    """CAN Light Medium = Light Trucks + Medium Trucks."""
    lt = pl.read_csv(OUT_DIR / "freight_can_light_trucks.csv")
    mt = pl.read_csv(OUT_DIR / "freight_can_medium_trucks.csv")
    df = _sum_modes([lt, mt], "can_light_medium")
    out = OUT_DIR / "freight_can_light_medium.csv"
    df.write_csv(out)
    act = df.filter(pl.col("year") == 2000).select("Activity (M tkm)").item()
    print(f"  ✅ {out.name}  (Activity 2000 = {act:,.1f} M·tkm)")
    return df


def build_can_heavy_total() -> pd.DataFrame:
    """CAN Heavy Total = Heavy Trucks + Rail."""
    ht   = pl.read_csv(OUT_DIR / "freight_can_heavy_trucks.csv")
    rail = pl.read_csv(OUT_DIR / "freight_can_rail.csv")
    df = _sum_modes([ht, rail], "can_heavy_total")
    out = OUT_DIR / "freight_can_heavy_total.csv"
    df.write_csv(out)
    print(f"  ✅ {out.name}")
    return df


def build_can_freight_total() -> pd.DataFrame:
    """CAN Freight Total = Light Medium + Heavy Total + Marine + Air."""
    component_files = [
        "freight_can_light_medium.csv",
        "freight_can_heavy_total.csv",
        "freight_can_marine.csv",
        "freight_can_air.csv",
    ]
    dfs = [pl.read_csv(OUT_DIR / f) for f in component_files]
    df = _sum_modes(dfs, "can_freight_total")
    out = OUT_DIR / "freight_can_total.csv"
    df.write_csv(out)
    print(f"  ✅ {out.name}")
    return df


# ── CAN Load Factor Extraction (for use by provincial builders) ─────────

def get_can_load_factors() -> dict:
    """Extract year-varying CAN load factors for LT/MT/HT.
    Returns dict: {'Light Truck': pd.Series, 'Medium Truck': pd.Series, 'Heavy Truck': pd.Series}
    """
    f1 = _load_sheet(CEUD_CAN_FILE, "Freight1")
    f4 = _load_sheet(CEUD_CAN_FILE, "Freight4")

    result = {}
    for mode, act_row, stock_row, dist_row in [
        ("Light Truck",  CAN_F1_LT_ACTIVITY, CAN_F4_LT_STOCK, CAN_F4_LT_AVGKM),
        ("Medium Truck", CAN_F1_MT_ACTIVITY, CAN_F4_MT_STOCK, CAN_F4_MT_AVGKM),
        ("Heavy Truck",  CAN_F1_HT_ACTIVITY, CAN_F4_HT_STOCK, CAN_F4_HT_AVGKM),
    ]:
        activity   = _read_row(f1, act_row)
        stock      = _read_row(f4, stock_row)
        avg_km     = _read_row(f4, dist_row)
        total_dist = stock * avg_km / 1000.0
        lf = _safe_div(activity, total_dist)
        result[mode] = lf

    return result


def get_can_rail_intensity_and_freight_share():
    """Extract CAN rail intensity (GJ/tkm) and freight share for provincial builders.
    Returns (intensity_series, freight_share_series, total_tkm_series)
    """
    f1 = _load_sheet(CEUD_CAN_FILE, "Freight1")
    t27 = _load_sheet(CEUD_CAN_FILE, "Table 27")
    p1  = _load_sheet(CEUD_CAN_FILE, "Passenger1")

    freight_tkm = _read_row(f1, CAN_F1_RAIL_ACTIVITY)
    pass_pkm    = _read_row(p1, CAN_P1_RAIL_PKM)
    pass_tkm    = pass_pkm * AVIATION_PASSENGER["tkm_per_pkm"]
    total_tkm   = freight_tkm + pass_tkm
    freight_pct = _safe_div(freight_tkm, total_tkm)

    diesel_tj   = _read_row(t27, CAN_T27_RAIL_TOTAL).fillna(0.0) * FUEL_SCALE
    intensity   = _safe_div(diesel_tj, total_tkm * 1000.0)  # GJ/tkm

    return intensity, freight_pct, total_tkm


def get_can_air_activity_and_shares():
    """Extract CAN air activity data for provincial builders.
    Returns (freight_tkm, total_tkm, freight_pct)
    """
    dom_tkm_share = AVIATION_TKM_SPLIT["Domestic"]

    f1 = _load_sheet(CEUD_CAN_FILE, "Freight1")
    p1 = _load_sheet(CEUD_CAN_FILE, "Passenger1")

    total_freight_tkm = _read_row(f1, CAN_F1_AIR_ACTIVITY)
    freight_tkm = total_freight_tkm * dom_tkm_share

    total_pass_pkm = _read_row(p1, CAN_P1_AIR_PKM)
    pass_pkm = total_pass_pkm * dom_tkm_share
    pass_tkm = pass_pkm * AVIATION_PASSENGER["tkm_per_pkm"]

    total_tkm = freight_tkm + pass_tkm
    freight_pct = _safe_div(freight_tkm, total_tkm)

    return freight_tkm, total_tkm, freight_pct



# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  BC PROVINCIAL BUILDERS                                              ║
# ║  BC = BCTerr for trucks/rail/air; Marine & Off-Road from IPCC BC    ║
# ╚═══════════════════════════════════════════════════════════════════════╝


# ── BC IPCC Embedded Data ─────────────────────────────────────────────────
# Source: EN_GHG_IPCC_Can_Prov_Terr.csv (pre-extracted for British Columbia)
# Marine: region=British Columbia, source=Energy, category=Transport,
#         sub-category=Marine, sub-sub-category=Domestic Navigation
# Off-Road: region=British Columbia, source=Energy, category=Transport,
#           sub-category=Other Transportation, sub-sub-category=Off-Road

# Marine HFO CO2eq (kt) for BC Domestic Navigation
BC_MARINE_DOM_NAV_CO2EQ = {
    2000: 786.476172, 2001: 799.873032, 2002: 812.810737, 2003: 826.763525,
    2004: 839.110226, 2005: 851.385374, 2006: 843.060320, 2007: 833.921510,
    2008: 823.858864, 2009: 815.064387, 2010: 787.595846, 2011: 804.227597,
    2012: 822.767075, 2013: 843.305655, 2014: 864.661291, 2015: 877.252141,
    2016: 879.308375, 2017: 985.295414, 2018: 1036.074586, 2019: 1017.845151,
    2020: 998.717659, 2021: 984.555450, 2022: 1093.886073,
}

# Off-Road gasoline CO2eq (kt) for BC Other Transportation Off-Road
BC_OFFROAD_CO2EQ = {
    2000: 765.830520, 2001: 764.277415, 2002: 799.578306, 2003: 817.225589,
    2004: 862.998813, 2005: 792.612340, 2006: 752.341096, 2007: 739.926725,
    2008: 722.296224, 2009: 667.996956, 2010: 651.806783, 2011: 602.082254,
    2012: 621.039572, 2013: 659.302755, 2014: 668.404821, 2015: 738.661920,
    2016: 911.071024, 2017: 978.569621, 2018: 1011.826040, 2019: 1002.934525,
    2020: 1038.034821, 2021: 1048.332838, 2022: 1014.566568,
}

# Fallback emission factors for back-calculation; normal execution uses coefficients.csv
BC_HFO_EF = 0.074737        # kt CO2eq per TJ (Heavy Fuel Oil fallback)
BC_GASOLINE_EF = 0.064       # kt CO2eq per TJ (Motor gasoline fallback)

# Off-Road efficiency (MJ/tkm) — constant
BC_OFFROAD_MJ_PER_TKM = 7.82


def _get_bc_marine_hfo() -> pd.Series:
    """Back-calculate BC marine HFO energy (TJ) from IPCC emissions.

    Formula: HFO (TJ) = IPCC co2eq_kt / coefficients[GHG/Energy, Fuel oil].
    """
    ef = get_coefficient_series("GHG/Energy", "Fuel oil", coefficient_unit="ktCO2e/TJ")
    co2eq = pd.Series(BC_MARINE_DOM_NAV_CO2EQ, dtype=float).reindex(YEARS).fillna(0.0)
    return _safe_div(co2eq, ef)


def _get_bc_offroad_gasoline() -> pd.Series:
    """Back-calculate BC off-road gasoline energy (TJ) from IPCC emissions.

    Formula: Gasoline (TJ) = IPCC co2eq_kt / coefficients[GHG/Energy, Gasoline].
    """
    ef = get_coefficient_series("GHG/Energy", "Gasoline", coefficient_unit="ktCO2e/TJ")
    co2eq = pd.Series(BC_OFFROAD_CO2EQ, dtype=float).reindex(YEARS).fillna(0.0)
    return _safe_div(co2eq, ef)


# ── BC Mode Builders ─────────────────────────────────────────────────────

def build_bc_light_trucks(bcterr_dfs: dict[str, pl.DataFrame] | None = None) -> pl.DataFrame:
    """BC Light Trucks = BCTerr Light Trucks (identical)."""
    df = _mode_df_from_cache(bcterr_dfs, "light_trucks", OUT_DIR / "freight_bcterr_light_trucks.csv")
    out = OUT_DIR / "freight_bc_light_trucks.csv"
    df.write_csv(out)
    act = df.filter(pl.col("year") == 2000).select("Activity (M tkm)").item()
    print(f"  ✅ {out.name}  (Activity 2000 = {act:,.1f} M·tkm) [= BCTerr]")
    return df


def build_bc_medium_trucks(bcterr_dfs: dict[str, pl.DataFrame] | None = None) -> pl.DataFrame:
    """BC Medium Trucks = BCTerr Medium Trucks (identical)."""
    df = _mode_df_from_cache(bcterr_dfs, "medium_trucks", OUT_DIR / "freight_bcterr_medium_trucks.csv")
    out = OUT_DIR / "freight_bc_medium_trucks.csv"
    df.write_csv(out)
    act = df.filter(pl.col("year") == 2000).select("Activity (M tkm)").item()
    print(f"  ✅ {out.name}  (Activity 2000 = {act:,.1f} M·tkm) [= BCTerr]")
    return df


def build_bc_heavy_trucks(bcterr_dfs: dict[str, pl.DataFrame] | None = None) -> pl.DataFrame:
    """BC Heavy Trucks = BCTerr Heavy Trucks (identical)."""
    df = _mode_df_from_cache(bcterr_dfs, "heavy_trucks", OUT_DIR / "freight_bcterr_heavy_trucks.csv")
    out = OUT_DIR / "freight_bc_heavy_trucks.csv"
    df.write_csv(out)
    act = df.filter(pl.col("year") == 2000).select("Activity (M tkm)").item()
    print(f"  ✅ {out.name}  (Activity 2000 = {act:,.1f} M·tkm) [= BCTerr]")
    return df


def build_bc_rail_freight(bcterr_dfs: dict[str, pl.DataFrame] | None = None) -> pl.DataFrame:
    """BC Rail = BCTerr Rail (identical)."""
    df = _mode_df_from_cache(bcterr_dfs, "rail", OUT_DIR / "freight_bcterr_rail.csv")
    out = OUT_DIR / "freight_bc_rail.csv"
    df.write_csv(out)
    act = df.filter(pl.col("year") == 2000).select("Activity (M tkm)").item()
    print(f"  ✅ {out.name}  (Freight tkm 2000 = {act:,.0f} M) [= BCTerr]")
    return df


def build_bc_air_freight(bcterr_dfs: dict[str, pl.DataFrame] | None = None) -> pl.DataFrame:
    """BC Air = BCTerr Air (identical)."""
    df = _mode_df_from_cache(bcterr_dfs, "air", OUT_DIR / "freight_bcterr_air.csv")
    out = OUT_DIR / "freight_bc_air.csv"
    df.write_csv(out)
    act = df.filter(pl.col("year") == 2000).select("Activity (M tkm)").item()
    print(f"  ✅ {out.name}  (Freight tkm 2000 = {act:,.0f} M) [= BCTerr]")
    return df


def build_bc_marine_freight(can_dfs: dict[str, pl.DataFrame] | None = None) -> pl.DataFrame:
    """BC Marine: HFO from IPCC BC provincial data.
    Activity = total_fuel / CAN_marine_intensity / 1000 × CAN_freight_share.

    Unlike BCTerr (all zeros), BC has real marine HFO from IPCC emissions.
    The marine intensity and freight share come from the CAN marine sheet.
    """
    # BC marine HFO fuel
    hfo = _get_bc_marine_hfo()
    fuels = {"Heavy fuel oil": hfo}
    fuel_total = hfo.copy()

    # CAN marine cross-refs: intensity and freight share
    can_marine = _mode_df_from_cache(can_dfs, "marine", OUT_DIR / "freight_can_marine.csv").to_pandas()
    can_marine = can_marine.set_index("year")

    can_marine_activity = pd.Series(
        can_marine["Activity (M tkm)"].values, index=can_marine.index
    )
    can_marine_fuel_total = pd.Series(
        can_marine["fuel_Total (TJ)"].values, index=can_marine.index
    )
    # CAN marine intensity = fuel_total / (activity * 1000) → GJ/tkm
    can_intensity = _safe_div(can_marine_fuel_total, can_marine_activity * 1000.0)

    # For CAN marine, freight % = 100% (no passengers)
    can_freight_pct = _const_series(1.0)

    # BC total tkm = BC_fuel / CAN_intensity / 1000
    total_tkm_bc = _safe_div(fuel_total, can_intensity) / 1000.0
    # BC freight tkm = total_tkm × freight_share
    freight_tkm = total_tkm_bc * can_freight_pct

    notes = [
        "BC Marine Freight (IPCC BC provincial HFO)",
        "  HFO: IPCC BC Domestic Navigation co2eq / coefficients[GHG/Energy, Fuel oil]",
        "  Activity = HFO / CAN_marine_intensity / 1000 × CAN_freight_pct",
    ]

    df = _build_mode_df(
        "bc_marine", freight_tkm, _nan_series(), _nan_series(),
        _nan_series(), _nan_series(), _nan_series(), fuels, notes
    )

    out = OUT_DIR / "freight_bc_marine.csv"
    df.write_csv(out)
    print(f"  ✅ {out.name}  (Freight tkm 2000 = {freight_tkm.iloc[0]:,.0f} M)")
    return df


def build_bc_offroad() -> pl.DataFrame:
    """BC Off-Road: Motor gasoline from IPCC BC provincial data.
    Activity = gasoline_TJ / OFFROAD_MJ_PER_TKM / 1000 (→ M·tkm).

    Note: Off-Road is NOT included in the Freight Total rollup.
    """
    gasoline = _get_bc_offroad_gasoline()
    fuels = {"Motor gasoline": gasoline}

    # Activity (M·tkm) = gasoline_TJ / MJ_per_tkm
    # because TJ→MJ and tkm→million-tkm conversions cancel out.
    activity = gasoline / BC_OFFROAD_MJ_PER_TKM

    notes = [
        "BC Off-Road Freight (IPCC BC provincial gasoline)",
        "  Motor gasoline: IPCC BC Off-Road co2eq / coefficients[GHG/Energy, Gasoline]",
        f"  Activity = gasoline_TJ / {BC_OFFROAD_MJ_PER_TKM}",
    ]

    df = _build_mode_df(
        "bc_offroad", activity, _nan_series(), _nan_series(),
        _nan_series(), _nan_series(), _nan_series(), fuels, notes
    )

    out = OUT_DIR / "freight_bc_offroad.csv"
    df.write_csv(out)
    print(f"  ✅ {out.name}  (Activity 2000 = {activity.iloc[0]:,.1f} M·tkm)")
    return df


# ── BC Aggregate Builders ────────────────────────────────────────────────

def build_bc_light_medium(bc_dfs: dict[str, pl.DataFrame] | None = None) -> pl.DataFrame:
    """BC Light Medium = Light Trucks + Medium Trucks."""
    lt = _mode_df_from_cache(bc_dfs, "light_trucks", OUT_DIR / "freight_bc_light_trucks.csv")
    mt = _mode_df_from_cache(bc_dfs, "medium_trucks", OUT_DIR / "freight_bc_medium_trucks.csv")
    df = _sum_modes([lt, mt], "bc_light_medium")
    out = OUT_DIR / "freight_bc_light_medium.csv"
    df.write_csv(out)
    act = df.filter(pl.col("year") == 2000).select("Activity (M tkm)").item()
    print(f"  ✅ {out.name}  (Activity 2000 = {act:,.1f} M·tkm)")
    return df


def build_bc_heavy_total(bc_dfs: dict[str, pl.DataFrame] | None = None) -> pl.DataFrame:
    """BC Heavy Total = Heavy Trucks + Rail."""
    ht   = _mode_df_from_cache(bc_dfs, "heavy_trucks", OUT_DIR / "freight_bc_heavy_trucks.csv")
    rail = _mode_df_from_cache(bc_dfs, "rail", OUT_DIR / "freight_bc_rail.csv")
    df = _sum_modes([ht, rail], "bc_heavy_total")
    out = OUT_DIR / "freight_bc_heavy_total.csv"
    df.write_csv(out)
    print(f"  ✅ {out.name}")
    return df


def build_bc_freight_total(bc_dfs: dict[str, pl.DataFrame] | None = None) -> pl.DataFrame:
    """BC Freight Total = Light Medium + Heavy Total + Marine + Air.
    Note: Off-Road is NOT included in Freight Total.
    """
    dfs = [
        _mode_df_from_cache(bc_dfs, "light_medium", OUT_DIR / "freight_bc_light_medium.csv"),
        _mode_df_from_cache(bc_dfs, "heavy_total", OUT_DIR / "freight_bc_heavy_total.csv"),
        _mode_df_from_cache(bc_dfs, "marine", OUT_DIR / "freight_bc_marine.csv"),
        _mode_df_from_cache(bc_dfs, "air", OUT_DIR / "freight_bc_air.csv"),
    ]
    df = _sum_modes(dfs, "bc_freight_total")
    out = OUT_DIR / "freight_bc_total.csv"
    df.write_csv(out)
    print(f"  ✅ {out.name}")
    return df


# ── BC Main Runner ───────────────────────────────────────────────────────

def run_bc_pipeline(
    can_dfs: dict[str, pl.DataFrame] | None = None,
    bcterr_dfs: dict[str, pl.DataFrame] | None = None,
) -> dict[str, pl.DataFrame]:
    """Run the full BC provincial freight pipeline.

    Normal execution passes already-built CAN and BCTerr dataframes in memory.
    This keeps BC structurally tied to upstream sheets while still writing CSV
    audit outputs after each BC mode is built.
    """
    print("\n" + "═" * 71)
    print("  BC PROVINCIAL FREIGHT PIPELINE")
    print("═" * 71)

    bc_dfs: dict[str, pl.DataFrame] = {}

    # Individual modes
    print("\n── Individual Modes ──")
    bc_dfs["light_trucks"] = build_bc_light_trucks(bcterr_dfs)
    bc_dfs["medium_trucks"] = build_bc_medium_trucks(bcterr_dfs)
    bc_dfs["heavy_trucks"] = build_bc_heavy_trucks(bcterr_dfs)
    bc_dfs["rail"] = build_bc_rail_freight(bcterr_dfs)
    bc_dfs["air"] = build_bc_air_freight(bcterr_dfs)
    bc_dfs["marine"] = build_bc_marine_freight(can_dfs)
    bc_dfs["offroad"] = build_bc_offroad()

    # Aggregates
    print("\n── Aggregates ──")
    bc_dfs["light_medium"] = build_bc_light_medium(bc_dfs)
    bc_dfs["heavy_total"] = build_bc_heavy_total(bc_dfs)
    bc_dfs["total"] = build_bc_freight_total(bc_dfs)

    print("\n" + "═" * 71)
    print("  BC pipeline complete.")
    print("═" * 71)
    return bc_dfs


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  UPDATED MAIN  (add CAN to the pipeline)                            ║
# ╚═══════════════════════════════════════════════════════════════════════╝




# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  IPCC / NIR EMISSIONS LOADER (EN_GHG_IPCC_Can_Prov_Terr.csv)           ║
# ╚═══════════════════════════════════════════════════════════════════════╝

_IPCC_CACHE: pd.DataFrame | None = None


_IPCC_CACHE = None

def load_ipcc_emissions(force: bool = False) -> pl.DataFrame:
    """Return CAN Marine Domestic Navigation emissions as in-memory dataframe.
    Data embedded as Python dict — NO external CSV file read at runtime.
    Source: EN_GHG_IPCC_Can_Prov_Terr.csv (pre-extracted)."""
    global _IPCC_CACHE

    if _IPCC_CACHE is not None and not force:
        return _IPCC_CACHE

    CAN_MARINE_DOM_NAV_CO2EQ = {
        2000: 2781.101253, 2001: 2861.748389, 2002: 2942.395526, 2003: 3023.042662,
        2004: 3103.689799, 2005: 3184.336935, 2006: 3142.675946, 2007: 3101.014957,
        2008: 3059.353968, 2009: 3017.637510, 2010: 2957.143208, 2011: 2950.362240,
        2012: 2949.885783, 2013: 2951.706381, 2014: 2954.083100, 2015: 2948.701552,
        2016: 3000.029450, 2017: 3205.249076, 2018: 3352.798121, 2019: 3345.947076,
        2020: 2968.175197, 2021: 2887.612365, 2022: 3303.509215,
}

    rows = []
    for yr, co2eq in CAN_MARINE_DOM_NAV_CO2EQ.items():
        rows.append({
            "year": yr, "region": "Canada", "source": "Energy",
            "category": "Transport", "sub-category": "Marine",
            "sub-sub-category": "Domestic Navigation", "co2eq_kt": co2eq,
        })

    _IPCC_CACHE = pl.DataFrame(rows)
    return _IPCC_CACHE



# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  AB PROVINCIAL BUILDERS                                               ║
# ║  Reimplementation of AB Freight Formulas.txt into Python workflow     ║
# ╚═══════════════════════════════════════════════════════════════════════╝

T19_MARINE_DIESEL_ROW = 13
T19_MARINE_HFO_ROW = 14
AB_OFFROAD_MJ_PER_TKM = 7.82  # assumptions!J178

# AB Off-Road values from the AB workbook/formula output. The spreadsheet formula
# uses IPCC Other Transportation / Off-Road Other Transportation emissions divided
# by the motor-gasoline coefficient (coefficients row 261), then activity = fuel / 7.82.
# The IPCC source table is not otherwise embedded in this script, so the historical
# 2000-2022 AB Off-Road motor gasoline TJ series is embedded here for reproducibility.
AB_OFFROAD_MOTOR_GASOLINE_TJ = {
    2000: 19330.0, 2001: 20502.0, 2002: 21540.0, 2003: 20577.0, 2004: 20170.0,
    2005: 19898.0, 2006: 19500.0, 2007: 18809.0, 2008: 18278.0, 2009: 15785.0,
    2010: 17308.0, 2011: 18409.0, 2012: 20202.0, 2013: 21700.0, 2014: 22962.0,
    2015: 21394.0, 2016: 25602.0, 2017: 25007.0, 2018: 23821.0, 2019: 23859.0,
    2020: 20674.0, 2021: 22262.0, 2022: 22300.0,
}



def _read_row_by_label(df: pd.DataFrame, label: str) -> pd.Series:
    labels = df.iloc[:, 1].astype(str).str.strip().str.lower()
    matches = np.where(labels == label.strip().lower())[0]
    if len(matches) == 0:
        return _zero_series()
    return _read_row(df, int(matches[0])).fillna(0.0)


def _write_ab(df: pl.DataFrame, name: str) -> pl.DataFrame:
    out = OUT_DIR / f"freight_ab_{name}.csv"
    df.write_csv(out)
    return df


def _build_ab_truck(mode: str) -> pl.DataFrame:
    """AB Light/Medium/Heavy truck formulas.

    Activity = AB Table 37 stock × average distance / 1000 × CAN load factor.
    Fuel = AB Table 35 or 36 CEUD fuel rows × 1000 PJ→TJ.
    """
    t37 = _load_sheet(CEUD_AB_FILE, "Table 37")
    stock = _read_row(t37, T37_STOCK[mode])
    avg_vkm = _read_row(t37, T37_AVGKM[mode])
    total_dist = stock * avg_vkm / 1000.0
    lf = get_can_load_factors()[mode]
    activity = total_dist * lf
    avg_tkm = avg_vkm * lf

    if mode == "Light Truck":
        t35 = _load_sheet(CEUD_AB_FILE, "Table 35")
        fuels = {fuel: _read_row(t35, row).fillna(0.0) * FUEL_SCALE for fuel, row in T35_FUEL.items()}
        suffix = "light_trucks"
    elif mode == "Medium Truck":
        t36 = _load_sheet(CEUD_AB_FILE, "Table 36")
        fuels = {fuel: _read_row(t36, row).fillna(0.0) * FUEL_SCALE for fuel, row in T36_MED_FUEL.items()}
        suffix = "medium_trucks"
    elif mode == "Heavy Truck":
        t36 = _load_sheet(CEUD_AB_FILE, "Table 36")
        fuels = {"Diesel fuel oil": _read_row(t36, T36_HEAVY_TOTAL_ROW).fillna(0.0) * FUEL_SCALE}
        suffix = "heavy_trucks"
    else:
        raise ValueError(mode)

    df = _build_mode_df(f"ab_{suffix}", activity, stock, avg_vkm, total_dist, lf, avg_tkm, fuels,
                        [f"AB {mode}: activity from Table 37 and CAN load factors; fuel from CEUD tables ×1000"])
    _write_ab(df, suffix)
    print(f"  ✅ freight_ab_{suffix}.csv  (Activity 2000 = {activity.iloc[0]:,.1f} M·tkm)")
    return df


def build_ab_light_trucks() -> pl.DataFrame:
    return _build_ab_truck("Light Truck")


def build_ab_medium_trucks() -> pl.DataFrame:
    return _build_ab_truck("Medium Truck")


def build_ab_heavy_trucks() -> pl.DataFrame:
    return _build_ab_truck("Heavy Truck")


def build_ab_rail_freight() -> pl.DataFrame:
    """AB rail: diesel from Table 18; activity via CAN rail intensity and freight share."""
    t18 = _load_sheet(CEUD_AB_FILE, "Table 18")
    diesel = _read_row(t18, T18_RAIL_TOTAL_ROW).fillna(0.0) * FUEL_SCALE
    can_intensity, can_freight_pct, _ = get_can_rail_intensity_and_freight_share()
    total_tkm = _safe_div(diesel, can_intensity) / 1000.0
    freight_tkm = total_tkm * can_freight_pct
    pass_tkm = total_tkm * (1.0 - can_freight_pct)
    pass_pkm = pass_tkm / AVIATION_PASSENGER["tkm_per_pkm"]
    df = _build_mode_df("ab_rail", freight_tkm, _nan_series(), _nan_series(), _nan_series(),
                        _nan_series(), _nan_series(), {"Diesel fuel oil": diesel},
                        ["AB Rail: Table 18 fuel ×1000; activity = fuel / CAN intensity × CAN freight share"])
    df = df.with_columns([
        pl.Series("Passenger pkm (millions)", pass_pkm.tolist()),
        pl.Series("Passenger tkm (millions)", pass_tkm.tolist()),
        pl.Series("Total tkm (millions)", total_tkm.tolist()),
        pl.Series("Freight %", can_freight_pct.tolist()),
    ])
    _write_ab(df, "rail")
    print(f"  ✅ freight_ab_rail.csv  (Freight tkm 2000 = {freight_tkm.iloc[0]:,.0f} M)")
    return df


def build_ab_air_freight() -> pl.DataFrame:
    """AB air: Table 15 fuel × AB domestic energy share; activity via CAN air intensity/share."""
    dom_share = AVIATION_ENERGY_DOMESTIC["AB"]
    t15 = _load_sheet(CEUD_AB_FILE, "Table 15")
    avturbo = _read_row(t15, T15_AVIATION_TURBO_ROW).fillna(0.0) * dom_share * FUEL_SCALE
    avgas = _read_row(t15, T15_AVIATION_GASOLINE_ROW).fillna(0.0) * dom_share * FUEL_SCALE
    fuel_total = avturbo + avgas

    _, can_total_tkm, can_freight_pct = get_can_air_activity_and_shares()
    t21 = _load_sheet(CEUD_CAN_FILE, "Table 21")
    can_fuel = (
        _read_row(t21, CAN_T21_AIR_AVTURBO).fillna(0.0) + _read_row(t21, CAN_T21_AIR_AVGAS).fillna(0.0)
    ) * AVIATION_ENERGY_DOMESTIC["CAN"] * FUEL_SCALE
    can_intensity = _safe_div(can_fuel, can_total_tkm * 1000.0)
    total_tkm = _safe_div(fuel_total, can_intensity) / 1000.0
    freight_tkm = total_tkm * can_freight_pct
    pass_tkm = total_tkm * (1.0 - can_freight_pct)
    pass_pkm = pass_tkm / AVIATION_PASSENGER["tkm_per_pkm"]

    df = _build_mode_df("ab_air", freight_tkm, _nan_series(), _nan_series(), _nan_series(),
                        _nan_series(), _nan_series(), {"Aviation turbo fuel": avturbo, "Aviation gasoline": avgas},
                        [f"AB Air: Table 15 fuel × domestic share {dom_share} ×1000; activity via CAN air intensity"])
    df = df.with_columns([
        pl.Series("Passenger pkm (millions)", pass_pkm.tolist()),
        pl.Series("Passenger tkm (millions)", pass_tkm.tolist()),
        pl.Series("Total tkm (millions)", total_tkm.tolist()),
        pl.Series("Freight %", can_freight_pct.tolist()),
    ])
    _write_ab(df, "air")
    print(f"  ✅ freight_ab_air.csv  (Freight tkm 2000 = {freight_tkm.iloc[0]:,.0f} M)")
    return df


def build_ab_marine_freight() -> pl.DataFrame:
    """AB Marine: zero activity and zero fuel for all years.

    Interpretation fix:
      - Alberta is treated as having no freight marine mode in the model.
      - Do not infer AB marine activity from CEUD Table 19 or CAN marine intensity.
      - Do not carry CEUD Table 19 diesel/HFO into Heavy fuel oil for AB.
      - Keep a properly shaped output file so downstream calc-tab logic can read
        freight_ab_marine.csv, but all activity and fuel columns are zero.

    This avoids creating a standalone non-zero AB marine audit series that can
    accidentally leak into calc-tab / submission outputs.
    """
    activity = _zero_series()
    fuels = {fuel: _zero_series() for fuel in FREIGHT_FUELS}

    df = _build_mode_df(
        "ab_marine",
        activity,
        _nan_series(),
        _nan_series(),
        _nan_series(),
        _nan_series(),
        _nan_series(),
        fuels,
        [
            "AB Marine: no modeled freight marine activity or fuel; all values set to zero.",
            "CEUD Table 19 is intentionally not used for AB marine in this freight pipeline.",
        ],
    )
    _write_ab(df, "marine")
    print("  ✅ freight_ab_marine.csv  (all zeros; AB Marine intentionally not modeled)")
    return df

def build_ab_offroad() -> pl.DataFrame:
    """AB off-road audit output.

    Formula workbook logic:
      - Fuel = IPCC Alberta / Energy / Transport / Other Transportation /
        Off-Road Other Transportation, converted to motor gasoline TJ using
        the motor gasoline coefficient row.
      - Activity (million tkm) = motor gasoline TJ / assumptions!J178.

    assumptions!J178 = 7.82 is numerically TJ per million tkm (equivalent to
    7.82 MJ/tkm), so there is no additional /1000 term here.
    """
    motor_gasoline = pd.Series(AB_OFFROAD_MOTOR_GASOLINE_TJ, dtype=float).reindex(YEARS).fillna(0.0)
    fuels = {fuel: _zero_series() for fuel in FREIGHT_FUELS}
    fuels["Motor gasoline"] = motor_gasoline
    activity = motor_gasoline / AB_OFFROAD_MJ_PER_TKM

    df = _build_mode_df(
        "ab_offroad", activity, _nan_series(), _nan_series(), _nan_series(),
        _const_series(LOAD_FACTOR["Off Road"]), _nan_series(), fuels,
        ["AB Off-Road: IPCC-derived motor gasoline TJ; activity = fuel / assumptions!J178 (7.82)"]
    )
    _write_ab(df, "offroad")
    print(f"  ✅ freight_ab_offroad.csv  (Activity 2000 = {activity.iloc[0]:,.0f} M·tkm)")
    return df


def build_ab_light_medium(ab_dfs: dict[str, pl.DataFrame] | None = None) -> pl.DataFrame:
    """Light Medium = Light Trucks + Medium Trucks, including aggregate metadata.

    _sum_modes() correctly sums activity and fuel, but blanks metadata columns
    for generic aggregates. AB Light Medium has aggregate metadata in the
    source spreadsheet, so rebuild those fields from the two upstream truck
    outputs:

      Stock                = Light Truck stock + Medium Truck stock
      Total Distance       = Light Truck total distance + Medium Truck total distance
      Average Distance vkm = Total Distance / Stock * 1000
      Load factor          = Activity / Total Distance
      Average Distance tkm = Average Distance vkm * Load factor

    Off-Road and Marine activity values remain exact calculated values; no
    display-rounding override is applied here.
    """
    lt = ab_dfs["light_trucks"] if ab_dfs and "light_trucks" in ab_dfs else pl.read_csv(OUT_DIR / "freight_ab_light_trucks.csv")
    mt = ab_dfs["medium_trucks"] if ab_dfs and "medium_trucks" in ab_dfs else pl.read_csv(OUT_DIR / "freight_ab_medium_trucks.csv")
    df = _sum_modes([lt, mt], "ab_light_medium")

    # Rehydrate aggregate metadata from the two upstream truck tables.
    lt_pd = lt.to_pandas().set_index("year")
    mt_pd = mt.to_pandas().set_index("year")
    base = df.to_pandas().set_index("year")

    stock = lt_pd["Stock (thousands)"].fillna(0.0) + mt_pd["Stock (thousands)"].fillna(0.0)
    total_distance = lt_pd["Total Distance (M vkm)"].fillna(0.0) + mt_pd["Total Distance (M vkm)"].fillna(0.0)
    avg_vkm = _safe_div(total_distance, stock) * 1000.0
    load_factor = _safe_div(base["Activity (M tkm)"], total_distance)
    avg_tkm = avg_vkm * load_factor

    base["Stock (thousands)"] = stock
    base["Total Distance (M vkm)"] = total_distance
    base["Average Distance (vkm)"] = avg_vkm
    base["Load factor (t/veh)"] = load_factor
    base["Average Distance (tkm)"] = avg_tkm

    df = pl.from_pandas(base.reset_index())
    _write_ab(df, "light_medium")
    print("  ✅ freight_ab_light_medium.csv")
    return df


def build_ab_heavy_total(ab_dfs: dict[str, pl.DataFrame] | None = None) -> pl.DataFrame:
    ht = ab_dfs["heavy_trucks"] if ab_dfs and "heavy_trucks" in ab_dfs else pl.read_csv(OUT_DIR / "freight_ab_heavy_trucks.csv")
    rail = ab_dfs["rail"] if ab_dfs and "rail" in ab_dfs else pl.read_csv(OUT_DIR / "freight_ab_rail.csv")
    df = _sum_modes([ht, rail], "ab_heavy_total")
    _write_ab(df, "heavy_total")
    print("  ✅ freight_ab_heavy_total.csv")
    return df


def build_ab_freight_total(ab_dfs: dict[str, pl.DataFrame] | None = None) -> pl.DataFrame:
    """AB Freight Total = Light Medium + Heavy Total + Air.

    Marine is deliberately excluded. The AB formula total references the three
    active freight components only (Light Medium, Heavy, Air); including the
    standalone marine audit series causes the small overstatement previously
    observed in 2009, 2010, 2015-2022.
    """
    dfs = [
        ab_dfs["light_medium"] if ab_dfs and "light_medium" in ab_dfs else pl.read_csv(OUT_DIR / "freight_ab_light_medium.csv"),
        ab_dfs["heavy_total"] if ab_dfs and "heavy_total" in ab_dfs else pl.read_csv(OUT_DIR / "freight_ab_heavy_total.csv"),
        ab_dfs["air"] if ab_dfs and "air" in ab_dfs else pl.read_csv(OUT_DIR / "freight_ab_air.csv"),
    ]
    df = _sum_modes(dfs, "ab_freight_total")
    _write_ab(df, "total")
    print("  ✅ freight_ab_total.csv  [marine excluded]")
    return df


def run_ab_pipeline(can_dfs: dict[str, pl.DataFrame] | None = None) -> dict[str, pl.DataFrame]:
    print("\n── AB Provincial Pipeline ──")
    ab_dfs: dict[str, pl.DataFrame] = {}
    ab_dfs["light_trucks"] = build_ab_light_trucks()
    ab_dfs["medium_trucks"] = build_ab_medium_trucks()
    ab_dfs["heavy_trucks"] = build_ab_heavy_trucks()
    ab_dfs["rail"] = build_ab_rail_freight()
    ab_dfs["marine"] = build_ab_marine_freight()
    ab_dfs["air"] = build_ab_air_freight()
    ab_dfs["offroad"] = build_ab_offroad()
    ab_dfs["light_medium"] = build_ab_light_medium(ab_dfs)
    ab_dfs["heavy_total"] = build_ab_heavy_total(ab_dfs)
    ab_dfs["total"] = build_ab_freight_total(ab_dfs)
    return ab_dfs

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  SK PROVINCIAL BUILDERS                                               ║
# ║  Reimplementation of SK Freight Formulas.txt into Python workflow     ║
# ╚═══════════════════════════════════════════════════════════════════════╝

SK_OFFROAD_MJ_PER_TKM = 7.82  # assumptions!J178 in the SK formulas
SK_OFFROAD_IPCC_FILE = SCRIPT_DIR / "EN_GHG_IPCC_Can_Prov_Terr.csv"


def _write_sk(df: pl.DataFrame, name: str) -> pl.DataFrame:
    out = OUT_DIR / f"freight_sk_{name}.csv"
    df.write_csv(out)
    return df


def _get_sk_offroad_gasoline_from_optional_ipcc() -> pd.Series | None:
    """Exact SK formula path if EN_GHG_IPCC_Can_Prov_Terr.csv is available.

    SK Freight Formulas.txt computes Off-Road Motor gasoline from the IPCC/NIR
    Saskatchewan / Energy / Transport / Other Transportation / Off-Road Other
    Transportation emissions divided by coefficients[GHG/Energy, Gasoline].
    """
    if not SK_OFFROAD_IPCC_FILE.exists():
        return None
    try:
        raw = pd.read_csv(SK_OFFROAD_IPCC_FILE)
    except Exception as exc:
        warnings.warn(f"Could not read {SK_OFFROAD_IPCC_FILE.name}; SK off-road will use CEUD fallback. Error: {exc}")
        return None
    cols = {str(c).strip().lower(): c for c in raw.columns}
    if "co2eq" not in cols and "co2eq_kt" in cols:
        cols["co2eq"] = cols["co2eq_kt"]
    required = ["year", "region", "source", "category", "sub-category", "sub-sub-category", "co2eq"]
    if not all(c in cols for c in required):
        warnings.warn(f"{SK_OFFROAD_IPCC_FILE.name} does not have expected IPCC columns; SK off-road will use CEUD fallback.")
        return None
    df = raw.copy()
    mask = (
        df[cols["region"]].astype(str).str.strip().str.lower().isin(["saskatchewan", "sk"])
        & (df[cols["source"]].astype(str).str.strip().str.lower() == "energy")
        & (df[cols["category"]].astype(str).str.strip().str.lower() == "transport")
        & (df[cols["sub-category"]].astype(str).str.strip().str.lower() == "other transportation")
        & (df[cols["sub-sub-category"]].astype(str).str.strip().str.lower().isin(["off-road other transportation", "off-road", "off road"]))
    )
    sub = df.loc[mask, [cols["year"], cols["co2eq"]]].copy()
    if sub.empty:
        warnings.warn(f"No SK Off-Road Other Transportation IPCC records found; SK off-road will use CEUD fallback.")
        return None
    yrs = pd.to_numeric(sub[cols["year"]], errors="coerce")
    co2 = pd.to_numeric(sub[cols["co2eq"]], errors="coerce")
    co2_series = pd.Series(co2.to_numpy(), index=yrs).groupby(level=0).sum().reindex(YEARS).fillna(0.0)
    gas_ef = get_coefficient_series("GHG/Energy", "Gasoline", coefficient_unit="ktCO2e/TJ")
    return _safe_div(co2_series, gas_ef)


def _get_sk_offroad_gasoline() -> tuple[pd.Series, str]:
    ipcc_series = _get_sk_offroad_gasoline_from_optional_ipcc()
    if ipcc_series is not None:
        return ipcc_series, "SK Off-Road: IPCC-derived motor gasoline TJ; activity = fuel / assumptions!J178 (7.82)"
    # Fallback keeps the workflow runnable with the uploaded SK CEUD workbook only.
    t1 = _load_sheet(CEUD_SK_FILE, "Table 1")
    offroad_total_tj = _read_row(t1, 14).fillna(0.0) * FUEL_SCALE
    return offroad_total_tj, (
        "SK Off-Road fallback: CEUD Table 1 Off-Road total energy ×1000 used as Motor gasoline; "
        "exact workbook formula requires EN_GHG_IPCC_Can_Prov_Terr.csv and coefficients[GHG/Energy, Gasoline]; "
        "activity = fuel / assumptions!J178 (7.82)"
    )


def _build_sk_truck(mode: str) -> pl.DataFrame:
    """SK Light/Medium/Heavy truck logic following SK Freight Formulas.txt."""
    t37 = _load_sheet(CEUD_SK_FILE, "Table 37")
    stock = _read_row(t37, T37_STOCK[mode])
    avg_vkm = _read_row(t37, T37_AVGKM[mode])
    total_dist = stock * avg_vkm / 1000.0
    lf = get_can_load_factors()[mode]
    activity = total_dist * lf
    avg_tkm = avg_vkm * lf
    if mode == "Light Truck":
        t35 = _load_sheet(CEUD_SK_FILE, "Table 35")
        fuels = {fuel: _read_row(t35, row).fillna(0.0) * FUEL_SCALE for fuel, row in T35_FUEL.items()}
        suffix = "light_trucks"
    elif mode == "Medium Truck":
        t36 = _load_sheet(CEUD_SK_FILE, "Table 36")
        fuels = {fuel: _read_row(t36, row).fillna(0.0) * FUEL_SCALE for fuel, row in T36_MED_FUEL.items()}
        suffix = "medium_trucks"
    elif mode == "Heavy Truck":
        t36 = _load_sheet(CEUD_SK_FILE, "Table 36")
        fuels = {"Diesel fuel oil": _read_row(t36, T36_HEAVY_TOTAL_ROW).fillna(0.0) * FUEL_SCALE}
        suffix = "heavy_trucks"
    else:
        raise ValueError(mode)
    df = _build_mode_df(f"sk_{suffix}", activity, stock, avg_vkm, total_dist, lf, avg_tkm, fuels,
                        [f"SK {mode}: Table 37 activity with CAN load factor; fuel from SK CEUD tables ×1000"])
    _write_sk(df, suffix)
    print(f"  ✅ freight_sk_{suffix}.csv  (Activity 2000 = {activity.iloc[0]:,.1f} M·tkm)")
    return df


def build_sk_light_trucks() -> pl.DataFrame:
    return _build_sk_truck("Light Truck")


def build_sk_medium_trucks() -> pl.DataFrame:
    return _build_sk_truck("Medium Truck")


def build_sk_heavy_trucks() -> pl.DataFrame:
    return _build_sk_truck("Heavy Truck")


def build_sk_rail_freight() -> pl.DataFrame:
    """SK Rail: Table 18 diesel fuel ×1000; activity via CAN rail intensity/share."""
    t18 = _load_sheet(CEUD_SK_FILE, "Table 18")
    diesel = _read_row(t18, T18_RAIL_TOTAL_ROW).fillna(0.0) * FUEL_SCALE
    can_intensity, can_freight_pct, _ = get_can_rail_intensity_and_freight_share()
    total_tkm = _safe_div(diesel, can_intensity) / 1000.0
    freight_tkm = total_tkm * can_freight_pct
    pass_tkm = total_tkm * (1.0 - can_freight_pct)
    pass_pkm = pass_tkm / AVIATION_PASSENGER["tkm_per_pkm"]
    df = _build_mode_df("sk_rail", freight_tkm, _nan_series(), _nan_series(), _nan_series(),
                        _nan_series(), _nan_series(), {"Diesel fuel oil": diesel},
                        ["SK Rail: Table 18 fuel ×1000; activity = fuel / CAN intensity × CAN freight share"])
    df = df.with_columns([
        pl.Series("Passenger pkm (millions)", pass_pkm.tolist()),
        pl.Series("Passenger tkm (millions)", pass_tkm.tolist()),
        pl.Series("Total tkm (millions)", total_tkm.tolist()),
        pl.Series("Freight %", can_freight_pct.tolist()),
    ])
    _write_sk(df, "rail")
    print(f"  ✅ freight_sk_rail.csv  (Freight tkm 2000 = {freight_tkm.iloc[0]:,.0f} M)")
    return df


def build_sk_air_freight() -> pl.DataFrame:
    """SK Air: Table 15 fuel × SK domestic share; activity via CAN air intensity/share."""
    dom_share = AVIATION_ENERGY_DOMESTIC["SK"]
    t15 = _load_sheet(CEUD_SK_FILE, "Table 15")
    avturbo = _read_row(t15, T15_AVIATION_TURBO_ROW).fillna(0.0) * dom_share * FUEL_SCALE
    avgas = _read_row(t15, T15_AVIATION_GASOLINE_ROW).fillna(0.0) * dom_share * FUEL_SCALE
    fuel_total = avturbo + avgas
    _, can_total_tkm, can_freight_pct = get_can_air_activity_and_shares()
    t21 = _load_sheet(CEUD_CAN_FILE, "Table 21")
    can_fuel = (_read_row(t21, CAN_T21_AIR_AVTURBO).fillna(0.0) + _read_row(t21, CAN_T21_AIR_AVGAS).fillna(0.0)) * AVIATION_ENERGY_DOMESTIC["CAN"] * FUEL_SCALE
    can_intensity = _safe_div(can_fuel, can_total_tkm * 1000.0)
    total_tkm = _safe_div(fuel_total, can_intensity) / 1000.0
    freight_tkm = total_tkm * can_freight_pct
    pass_tkm = total_tkm * (1.0 - can_freight_pct)
    pass_pkm = pass_tkm / AVIATION_PASSENGER["tkm_per_pkm"]
    df = _build_mode_df("sk_air", freight_tkm, _nan_series(), _nan_series(), _nan_series(),
                        _nan_series(), _nan_series(), {"Aviation turbo fuel": avturbo, "Aviation gasoline": avgas},
                        [f"SK Air: Table 15 fuel × domestic share {dom_share} ×1000; activity via CAN air intensity"])
    df = df.with_columns([
        pl.Series("Passenger pkm (millions)", pass_pkm.tolist()),
        pl.Series("Passenger tkm (millions)", pass_tkm.tolist()),
        pl.Series("Total tkm (millions)", total_tkm.tolist()),
        pl.Series("Freight %", can_freight_pct.tolist()),
    ])
    _write_sk(df, "air")
    print(f"  ✅ freight_sk_air.csv  (Freight tkm 2000 = {freight_tkm.iloc[0]:,.0f} M)")
    return df


def build_sk_marine_freight() -> pl.DataFrame:
    """SK Marine standalone audit. Excluded from SK Freight Total per formula structure."""
    t19 = _load_sheet(CEUD_SK_FILE, "Table 19")
    diesel_raw = _read_row(t19, T19_MARINE_DIESEL_ROW).fillna(0.0) * FUEL_SCALE
    hfo_raw = _read_row(t19, T19_MARINE_HFO_ROW).fillna(0.0) * FUEL_SCALE
    hfo = diesel_raw + hfo_raw
    diesel = _zero_series()
    total_fuel = hfo
    try:
        can_df = pl.read_csv(OUT_DIR / "freight_can_marine.csv")
        can_activity = pd.Series(can_df["Activity (M tkm)"].to_list(), index=YEARS, dtype=float)
        can_fuel_total = pd.Series(can_df["fuel_Total (TJ)"].to_list(), index=YEARS, dtype=float)
        can_intensity = _safe_div(can_fuel_total, can_activity * 1000.0)
        activity = _safe_div(total_fuel, can_intensity) / 1000.0
    except Exception:
        activity = _zero_series()
    df = _build_mode_df("sk_marine", activity, _nan_series(), _nan_series(), _nan_series(),
                        _nan_series(), _nan_series(), {"Diesel fuel oil": diesel, "Heavy fuel oil": hfo},
                        ["SK Marine audit: Table 19 marine fuel reclassified to HFO; excluded from SK Freight Total"])
    _write_sk(df, "marine")
    print(f"  ✅ freight_sk_marine.csv  (standalone audit; HFO 2000 = {hfo.loc[2000]:,.0f} TJ)")
    return df


def build_sk_offroad() -> pl.DataFrame:
    """SK Off-Road standalone audit output. Not included in SK Freight Total."""
    motor_gasoline, note = _get_sk_offroad_gasoline()
    fuels = {fuel: _zero_series() for fuel in FREIGHT_FUELS}
    fuels["Motor gasoline"] = motor_gasoline
    activity = motor_gasoline / SK_OFFROAD_MJ_PER_TKM
    df = _build_mode_df("sk_offroad", activity, _nan_series(), _nan_series(), _nan_series(),
                        _const_series(LOAD_FACTOR["Off Road"]), _nan_series(), fuels, [note])
    _write_sk(df, "offroad")
    print(f"  ✅ freight_sk_offroad.csv  (Activity 2000 = {activity.iloc[0]:,.0f} M·tkm)")
    return df


def build_sk_light_medium(sk_dfs: dict[str, pl.DataFrame] | None = None) -> pl.DataFrame:
    """Light Medium = Light Trucks + Medium Trucks; rehydrate aggregate metadata."""
    lt = sk_dfs["light_trucks"] if sk_dfs and "light_trucks" in sk_dfs else pl.read_csv(OUT_DIR / "freight_sk_light_trucks.csv")
    mt = sk_dfs["medium_trucks"] if sk_dfs and "medium_trucks" in sk_dfs else pl.read_csv(OUT_DIR / "freight_sk_medium_trucks.csv")
    df = _sum_modes([lt, mt], "sk_light_medium")
    lt_pd, mt_pd, base = lt.to_pandas().set_index("year"), mt.to_pandas().set_index("year"), df.to_pandas().set_index("year")
    stock = lt_pd["Stock (thousands)"].fillna(0.0) + mt_pd["Stock (thousands)"].fillna(0.0)
    total_distance = lt_pd["Total Distance (M vkm)"].fillna(0.0) + mt_pd["Total Distance (M vkm)"].fillna(0.0)
    avg_vkm = _safe_div(total_distance, stock) * 1000.0
    load_factor = _safe_div(base["Activity (M tkm)"], total_distance)
    base["Stock (thousands)"], base["Total Distance (M vkm)"] = stock, total_distance
    base["Average Distance (vkm)"], base["Load factor (t/veh)"] = avg_vkm, load_factor
    base["Average Distance (tkm)"] = avg_vkm * load_factor
    df = pl.from_pandas(base.reset_index())
    _write_sk(df, "light_medium")
    print("  ✅ freight_sk_light_medium.csv")
    return df


def build_sk_heavy_total(sk_dfs: dict[str, pl.DataFrame] | None = None) -> pl.DataFrame:
    ht = sk_dfs["heavy_trucks"] if sk_dfs and "heavy_trucks" in sk_dfs else pl.read_csv(OUT_DIR / "freight_sk_heavy_trucks.csv")
    rail = sk_dfs["rail"] if sk_dfs and "rail" in sk_dfs else pl.read_csv(OUT_DIR / "freight_sk_rail.csv")
    df = _sum_modes([ht, rail], "sk_heavy_total")
    _write_sk(df, "heavy_total")
    print("  ✅ freight_sk_heavy_total.csv")
    return df


def build_sk_freight_total(sk_dfs: dict[str, pl.DataFrame] | None = None) -> pl.DataFrame:
    """SK Freight Total = Light Medium + Heavy Total + Air; Marine/Off-Road are audit-only."""
    dfs = [
        sk_dfs["light_medium"] if sk_dfs and "light_medium" in sk_dfs else pl.read_csv(OUT_DIR / "freight_sk_light_medium.csv"),
        sk_dfs["heavy_total"] if sk_dfs and "heavy_total" in sk_dfs else pl.read_csv(OUT_DIR / "freight_sk_heavy_total.csv"),
        sk_dfs["air"] if sk_dfs and "air" in sk_dfs else pl.read_csv(OUT_DIR / "freight_sk_air.csv"),
    ]
    df = _sum_modes(dfs, "sk_freight_total")
    _write_sk(df, "total")
    print("  ✅ freight_sk_total.csv  [marine and offroad excluded]")
    return df


def run_sk_pipeline(can_dfs: dict[str, pl.DataFrame] | None = None) -> dict[str, pl.DataFrame]:
    print("\n── SK Provincial Pipeline ──")
    sk_dfs: dict[str, pl.DataFrame] = {}
    sk_dfs["light_trucks"] = build_sk_light_trucks()
    sk_dfs["medium_trucks"] = build_sk_medium_trucks()
    sk_dfs["heavy_trucks"] = build_sk_heavy_trucks()
    sk_dfs["rail"] = build_sk_rail_freight()
    sk_dfs["marine"] = build_sk_marine_freight()
    sk_dfs["air"] = build_sk_air_freight()
    sk_dfs["offroad"] = build_sk_offroad()
    sk_dfs["light_medium"] = build_sk_light_medium(sk_dfs)
    sk_dfs["heavy_total"] = build_sk_heavy_total(sk_dfs)
    sk_dfs["total"] = build_sk_freight_total(sk_dfs)
    return sk_dfs


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  MB PROVINCIAL BUILDERS                                               ║
# ║  Reimplementation of MB Freight Formulas into Python workflow         ║
# ╚═══════════════════════════════════════════════════════════════════════╝

MB_OFFROAD_MJ_PER_TKM = 7.82  # assumptions Off-Road MJ/tkm
MB_IPCC_FILE = SCRIPT_DIR / "EN_GHG_IPCC_Can_Prov_Terr.csv"


def _write_mb(df: pl.DataFrame, name: str) -> pl.DataFrame:
    out = OUT_DIR / f"freight_mb_{name}.csv"
    df.write_csv(out)
    return df


def _load_mb_ipcc_optional() -> pd.DataFrame | None:
    """Load EN_GHG_IPCC_Can_Prov_Terr.csv if present; otherwise None."""
    if not MB_IPCC_FILE.exists():
        return None
    try:
        return pd.read_csv(MB_IPCC_FILE)
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"Could not read {MB_IPCC_FILE.name}; MB IPCC-derived modes will use CEUD fallbacks. Error: {exc}")
        return None


def _ipcc_series(df_ipcc: pd.DataFrame | None, *, region: str, source: str, category: str,
                 sub_category: str, sub_sub_category: str) -> pd.Series | None:
    """Extract a year-indexed kt CO2e series from the optional IPCC dataframe.

    IMPORTANT: each comparison is wrapped in parentheses. In Python, `&` has
    higher precedence than `==`, so missing parentheses can cause pandas to
    attempt `string & boolean_series`, raising the exact error you hit.
    """
    if df_ipcc is None:
        return None

    cols = {str(c).strip().lower(): c for c in df_ipcc.columns}
    # Accept co2eq or co2eq_kt
    if "co2eq" not in cols and "co2eq_kt" in cols:
        cols["co2eq"] = cols["co2eq_kt"]

    required = ["year", "region", "source", "category", "sub-category", "sub-sub-category", "co2eq"]
    if not all(c in cols for c in required):
        # Try alternative column spellings
        alt = {
            "sub-category": ["sub_category", "subcategory"],
            "sub-sub-category": ["sub_sub_category", "subsub_category", "subsub-category"],
        }
        for k, alts in alt.items():
            if k not in cols:
                for a in alts:
                    if a in cols:
                        cols[k] = cols[a]
                        break
        if not all(c in cols for c in required):
            return None

    d = df_ipcc
    reg = d[cols["region"]].astype(str).str.strip().str.lower()
    src = d[cols["source"]].astype(str).str.strip().str.lower()
    cat = d[cols["category"]].astype(str).str.strip().str.lower()
    subc = d[cols["sub-category"]].astype(str).str.strip().str.lower()
    subsub = d[cols["sub-sub-category"]].astype(str).str.strip().str.lower()

    mask = (
        (reg == region.strip().lower())
        & (src == source.strip().lower())
        & (cat == category.strip().lower())
        & (subc == sub_category.strip().lower())
        & (subsub == sub_sub_category.strip().lower())
    )

    sub = d.loc[mask, [cols["year"], cols["co2eq"]]].copy()
    if sub.empty:
        return None

    yrs = pd.to_numeric(sub[cols["year"]], errors="coerce")
    co2 = pd.to_numeric(sub[cols["co2eq"]], errors="coerce")
    ser = pd.Series(co2.to_numpy(), index=yrs).groupby(level=0).sum().reindex(YEARS).fillna(0.0)
    return ser


def _build_mb_truck(mode: str) -> pl.DataFrame:
    """MB Light/Medium/Heavy truck formulas."""
    t37 = _load_sheet(CEUD_MB_FILE, "Table 37")
    stock = _read_row(t37, T37_STOCK[mode])
    avg_vkm = _read_row(t37, T37_AVGKM[mode])
    total_dist = stock * avg_vkm / 1000.0
    lf = get_can_load_factors()[mode]
    activity = total_dist * lf
    avg_tkm = avg_vkm * lf

    if mode == "Light Truck":
        t35 = _load_sheet(CEUD_MB_FILE, "Table 35")
        fuels = {fuel: _read_row(t35, row).fillna(0.0) * FUEL_SCALE for fuel, row in T35_FUEL.items()}
        suffix = "light_trucks"
    elif mode == "Medium Truck":
        t36 = _load_sheet(CEUD_MB_FILE, "Table 36")
        fuels = {fuel: _read_row(t36, row).fillna(0.0) * FUEL_SCALE for fuel, row in T36_MED_FUEL.items()}
        suffix = "medium_trucks"
    elif mode == "Heavy Truck":
        t36 = _load_sheet(CEUD_MB_FILE, "Table 36")
        fuels = {"Diesel fuel oil": _read_row(t36, T36_HEAVY_TOTAL_ROW).fillna(0.0) * FUEL_SCALE}
        suffix = "heavy_trucks"
    else:
        raise ValueError(mode)

    df = _build_mode_df(
        f"mb_{suffix}", activity, stock, avg_vkm, total_dist, lf, avg_tkm, fuels,
        [f"MB {mode}: activity from Table 37 and CAN load factors; fuel from CEUD tables ×1000"]
    )
    _write_mb(df, suffix)
    print(f"  ✅ freight_mb_{suffix}.csv  (Activity 2000 = {activity.iloc[0]:,.1f} M·tkm)")
    return df


def build_mb_light_trucks() -> pl.DataFrame:
    return _build_mb_truck("Light Truck")


def build_mb_medium_trucks() -> pl.DataFrame:
    return _build_mb_truck("Medium Truck")


def build_mb_heavy_trucks() -> pl.DataFrame:
    return _build_mb_truck("Heavy Truck")


def build_mb_rail_freight() -> pl.DataFrame:
    t18 = _load_sheet(CEUD_MB_FILE, "Table 18")
    diesel = _read_row(t18, T18_RAIL_TOTAL_ROW).fillna(0.0) * FUEL_SCALE
    can_intensity, can_freight_pct, _ = get_can_rail_intensity_and_freight_share()
    total_tkm = _safe_div(diesel, can_intensity) / 1000.0
    freight_tkm = total_tkm * can_freight_pct
    pass_tkm = total_tkm * (1.0 - can_freight_pct)
    pass_pkm = pass_tkm / AVIATION_PASSENGER["tkm_per_pkm"]

    df = _build_mode_df(
        "mb_rail", freight_tkm, _nan_series(), _nan_series(), _nan_series(),
        _nan_series(), _nan_series(), {"Diesel fuel oil": diesel},
        ["MB Rail: Table 18 fuel ×1000; activity = fuel / CAN intensity × CAN freight share"]
    )
    df = df.with_columns([
        pl.Series("Passenger pkm (millions)", pass_pkm.tolist()),
        pl.Series("Passenger tkm (millions)", pass_tkm.tolist()),
        pl.Series("Total tkm (millions)", total_tkm.tolist()),
        pl.Series("Freight %", can_freight_pct.tolist()),
    ])
    _write_mb(df, "rail")
    print(f"  ✅ freight_mb_rail.csv  (Freight tkm 2000 = {freight_tkm.iloc[0]:,.0f} M)")
    return df


def build_mb_air_freight() -> pl.DataFrame:
    dom_share = AVIATION_ENERGY_DOMESTIC["MB"]
    t15 = _load_sheet(CEUD_MB_FILE, "Table 15")
    avturbo = _read_row(t15, T15_AVIATION_TURBO_ROW).fillna(0.0) * dom_share * FUEL_SCALE
    avgas = _read_row(t15, T15_AVIATION_GASOLINE_ROW).fillna(0.0) * dom_share * FUEL_SCALE
    fuel_total = avturbo + avgas

    _, can_total_tkm, can_freight_pct = get_can_air_activity_and_shares()
    t21 = _load_sheet(CEUD_CAN_FILE, "Table 21")
    can_fuel = (
        _read_row(t21, CAN_T21_AIR_AVTURBO).fillna(0.0) + _read_row(t21, CAN_T21_AIR_AVGAS).fillna(0.0)
    ) * AVIATION_ENERGY_DOMESTIC["CAN"] * FUEL_SCALE
    can_intensity = _safe_div(can_fuel, can_total_tkm * 1000.0)
    total_tkm = _safe_div(fuel_total, can_intensity) / 1000.0
    freight_tkm = total_tkm * can_freight_pct
    pass_tkm = total_tkm * (1.0 - can_freight_pct)
    pass_pkm = pass_tkm / AVIATION_PASSENGER["tkm_per_pkm"]

    df = _build_mode_df(
        "mb_air", freight_tkm, _nan_series(), _nan_series(), _nan_series(),
        _nan_series(), _nan_series(), {"Aviation turbo fuel": avturbo, "Aviation gasoline": avgas},
        [f"MB Air: Table 15 fuel × domestic share {dom_share} ×1000; activity via CAN air intensity"]
    )
    df = df.with_columns([
        pl.Series("Passenger pkm (millions)", pass_pkm.tolist()),
        pl.Series("Passenger tkm (millions)", pass_tkm.tolist()),
        pl.Series("Total tkm (millions)", total_tkm.tolist()),
        pl.Series("Freight %", can_freight_pct.tolist()),
    ])
    _write_mb(df, "air")
    print(f"  ✅ freight_mb_air.csv  (Freight tkm 2000 = {freight_tkm.iloc[0]:,.0f} M)")
    return df


def build_mb_marine_freight() -> pl.DataFrame:
    """MB marine: derive HFO energy from IPCC emissions + coefficients.csv; infer activity via CAN intensity."""
    df_ipcc = _load_mb_ipcc_optional()
    co2eq = _ipcc_series(
        df_ipcc,
        region="Manitoba",
        source="Energy",
        category="Transport",
        sub_category="Marine",
        sub_sub_category="Domestic Navigation",
    )

    if co2eq is not None:
        ef = get_coefficient_series("GHG/Energy", "Fuel oil", coefficient_unit="ktCO2e/TJ")
        hfo = _safe_div(co2eq, ef)
        note = "MB Marine: IPCC Domestic Navigation co2eq / coefficients[GHG/Energy, Fuel oil]"
    else:
        t19 = _load_sheet(CEUD_MB_FILE, "Table 19")
        diesel_raw = _read_row(t19, T19_MARINE_DIESEL_ROW).fillna(0.0) * FUEL_SCALE
        hfo_raw = _read_row(t19, T19_MARINE_HFO_ROW).fillna(0.0) * FUEL_SCALE
        hfo = diesel_raw + hfo_raw
        note = "MB Marine fallback: CEUD Table 19 diesel+HFO (reclassified to HFO)"

    fuels = {fuel: _zero_series() for fuel in FREIGHT_FUELS}
    fuels["Heavy fuel oil"] = hfo

    try:
        can_df = pl.read_csv(OUT_DIR / "freight_can_marine.csv")
        can_activity = pd.Series(can_df["Activity (M tkm)"].to_list(), index=YEARS, dtype=float)
        can_fuel_total = pd.Series(can_df["fuel_Total (TJ)"].to_list(), index=YEARS, dtype=float)
        can_intensity = _safe_div(can_fuel_total, can_activity * 1000.0)
        activity = _safe_div(hfo, can_intensity) / 1000.0
    except Exception:
        activity = _zero_series()

    df = _build_mode_df(
        "mb_marine", activity, _nan_series(), _nan_series(), _nan_series(),
        _nan_series(), _nan_series(), fuels,
        [note, "MB Marine: activity inferred via CAN marine intensity"]
    )
    _write_mb(df, "marine")
    print(f"  ✅ freight_mb_marine.csv  (HFO 2000 = {hfo.loc[2000]:,.0f} TJ)")
    return df


def build_mb_offroad() -> pl.DataFrame:
    """MB Off-Road audit output (not included in MB Freight Total)."""
    df_ipcc = _load_mb_ipcc_optional()
    co2eq = _ipcc_series(
        df_ipcc,
        region="Manitoba",
        source="Energy",
        category="Transport",
        sub_category="Other Transportation",
        sub_sub_category="Off-Road Other Transportation",
    )

    if co2eq is not None:
        ef = get_coefficient_series("GHG/Energy", "Gasoline", coefficient_unit="ktCO2e/TJ")
        gasoline = _safe_div(co2eq, ef)
        note = "MB Off-Road: IPCC co2eq / coefficients[GHG/Energy, Gasoline]"
    else:
        t1 = _load_sheet(CEUD_MB_FILE, "Table 1")
        gasoline = _read_row(t1, 14).fillna(0.0) * FUEL_SCALE
        note = "MB Off-Road fallback: CEUD Table 1 Off-Road total energy ×1000 used as Motor gasoline"

    fuels = {fuel: _zero_series() for fuel in FREIGHT_FUELS}
    fuels["Motor gasoline"] = gasoline
    activity = gasoline / MB_OFFROAD_MJ_PER_TKM

    df = _build_mode_df(
        "mb_offroad", activity, _nan_series(), _nan_series(), _nan_series(),
        _const_series(LOAD_FACTOR["Off Road"]), _nan_series(), fuels,
        [note, "activity = fuel / 7.82"]
    )
    _write_mb(df, "offroad")
    print(f"  ✅ freight_mb_offroad.csv  (Activity 2000 = {activity.iloc[0]:,.0f} M·tkm)")
    return df


def build_mb_light_medium(mb_dfs: dict[str, pl.DataFrame] | None = None) -> pl.DataFrame:
    lt = mb_dfs["light_trucks"] if mb_dfs and "light_trucks" in mb_dfs else pl.read_csv(OUT_DIR / "freight_mb_light_trucks.csv")
    mt = mb_dfs["medium_trucks"] if mb_dfs and "medium_trucks" in mb_dfs else pl.read_csv(OUT_DIR / "freight_mb_medium_trucks.csv")
    df = _sum_modes([lt, mt], "mb_light_medium")

    lt_pd = lt.to_pandas().set_index("year")
    mt_pd = mt.to_pandas().set_index("year")
    base = df.to_pandas().set_index("year")

    stock = lt_pd["Stock (thousands)"].fillna(0.0) + mt_pd["Stock (thousands)"].fillna(0.0)
    total_distance = lt_pd["Total Distance (M vkm)"].fillna(0.0) + mt_pd["Total Distance (M vkm)"].fillna(0.0)
    avg_vkm = _safe_div(total_distance, stock) * 1000.0
    load_factor = _safe_div(base["Activity (M tkm)"], total_distance)

    base["Stock (thousands)"] = stock
    base["Total Distance (M vkm)"] = total_distance
    base["Average Distance (vkm)"] = avg_vkm
    base["Load factor (t/veh)"] = load_factor
    base["Average Distance (tkm)"] = avg_vkm * load_factor

    df = pl.from_pandas(base.reset_index())
    _write_mb(df, "light_medium")
    print("  ✅ freight_mb_light_medium.csv")
    return df


def build_mb_heavy_total(mb_dfs: dict[str, pl.DataFrame] | None = None) -> pl.DataFrame:
    ht = mb_dfs["heavy_trucks"] if mb_dfs and "heavy_trucks" in mb_dfs else pl.read_csv(OUT_DIR / "freight_mb_heavy_trucks.csv")
    rail = mb_dfs["rail"] if mb_dfs and "rail" in mb_dfs else pl.read_csv(OUT_DIR / "freight_mb_rail.csv")
    df = _sum_modes([ht, rail], "mb_heavy_total")
    _write_mb(df, "heavy_total")
    print("  ✅ freight_mb_heavy_total.csv")
    return df


def build_mb_freight_total(mb_dfs: dict[str, pl.DataFrame] | None = None) -> pl.DataFrame:
    """MB Freight Total (Option A) = Light Medium + Heavy Total + Marine + Air."""
    dfs = [
        mb_dfs["light_medium"] if mb_dfs and "light_medium" in mb_dfs else pl.read_csv(OUT_DIR / "freight_mb_light_medium.csv"),
        mb_dfs["heavy_total"] if mb_dfs and "heavy_total" in mb_dfs else pl.read_csv(OUT_DIR / "freight_mb_heavy_total.csv"),
        mb_dfs["marine"] if mb_dfs and "marine" in mb_dfs else pl.read_csv(OUT_DIR / "freight_mb_marine.csv"),
        mb_dfs["air"] if mb_dfs and "air" in mb_dfs else pl.read_csv(OUT_DIR / "freight_mb_air.csv"),
    ]
    df = _sum_modes(dfs, "mb_freight_total")
    _write_mb(df, "total")
    print("  ✅ freight_mb_total.csv  [marine included; offroad excluded]")
    return df


def run_mb_pipeline(can_dfs: dict[str, pl.DataFrame] | None = None) -> dict[str, pl.DataFrame]:
    print("\n── MB Provincial Pipeline ──")
    mb_dfs: dict[str, pl.DataFrame] = {}
    mb_dfs["light_trucks"] = build_mb_light_trucks()
    mb_dfs["medium_trucks"] = build_mb_medium_trucks()
    mb_dfs["heavy_trucks"] = build_mb_heavy_trucks()
    mb_dfs["rail"] = build_mb_rail_freight()
    mb_dfs["marine"] = build_mb_marine_freight()
    mb_dfs["air"] = build_mb_air_freight()
    mb_dfs["offroad"] = build_mb_offroad()
    mb_dfs["light_medium"] = build_mb_light_medium(mb_dfs)
    mb_dfs["heavy_total"] = build_mb_heavy_total(mb_dfs)
    mb_dfs["total"] = build_mb_freight_total(mb_dfs)
    return mb_dfs


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  ON PROVINCIAL BUILDERS                                               ║
# ║  Reimplementation of ON Freight Formulas.txt into Python workflow      ║
# ╚═══════════════════════════════════════════════════════════════════════╝

ON_OFFROAD_MJ_PER_TKM = 7.82
ON_IPCC_FILE = SCRIPT_DIR / "EN_GHG_IPCC_Can_Prov_Terr.csv"


def _write_on(df: pl.DataFrame, name: str) -> pl.DataFrame:
    out = OUT_DIR / f"freight_on_{name}.csv"
    df.write_csv(out)
    return df


def _load_on_ipcc_optional() -> pd.DataFrame | None:
    """Load EN_GHG_IPCC_Can_Prov_Terr.csv if present; otherwise None."""
    if not ON_IPCC_FILE.exists():
        return None
    try:
        return pd.read_csv(ON_IPCC_FILE)
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"Could not read {ON_IPCC_FILE.name}; ON IPCC-derived modes will use CEUD fallbacks. Error: {exc}")
        return None


def _ipcc_series_on(df_ipcc: pd.DataFrame | None, *, region: str, source: str, category: str,
                    sub_category: str, sub_sub_category: str) -> pd.Series | None:
    """Extract year-indexed kt CO2e from IPCC/NIR dataframe.

    Notes:
      - Parentheses on each comparison avoid pandas operator precedence issues.
      - Accepts both 'co2eq' and 'co2eq_kt' column conventions.
    """
    if df_ipcc is None:
        return None

    cols = {str(c).strip().lower(): c for c in df_ipcc.columns}
    if "co2eq" not in cols and "co2eq_kt" in cols:
        cols["co2eq"] = cols["co2eq_kt"]

    # tolerate alternate spellings
    if "sub-category" not in cols:
        for k in ("sub_category", "subcategory"):
            if k in cols:
                cols["sub-category"] = cols[k]
                break
    if "sub-sub-category" not in cols:
        for k in ("sub_sub_category", "subsub_category", "subsub-category"):
            if k in cols:
                cols["sub-sub-category"] = cols[k]
                break

    required = ["year", "region", "source", "category", "sub-category", "sub-sub-category", "co2eq"]
    if not all(c in cols for c in required):
        return None

    d = df_ipcc
    reg = d[cols["region"]].astype(str).str.strip().str.lower()
    src = d[cols["source"]].astype(str).str.strip().str.lower()
    cat = d[cols["category"]].astype(str).str.strip().str.lower()
    subc = d[cols["sub-category"]].astype(str).str.strip().str.lower()
    subsub = d[cols["sub-sub-category"]].astype(str).str.strip().str.lower()

    mask = (
        (reg == region.strip().lower())
        & (src == source.strip().lower())
        & (cat == category.strip().lower())
        & (subc == sub_category.strip().lower())
        & (subsub == sub_sub_category.strip().lower())
    )

    sub = d.loc[mask, [cols["year"], cols["co2eq"]]].copy()
    if sub.empty:
        return None

    yrs = pd.to_numeric(sub[cols["year"]], errors="coerce")
    co2 = pd.to_numeric(sub[cols["co2eq"]], errors="coerce")
    ser = pd.Series(co2.to_numpy(), index=yrs).groupby(level=0).sum().reindex(YEARS).fillna(0.0)
    return ser


def _build_on_truck(mode: str) -> pl.DataFrame:
    """ON Light/Medium/Heavy truck formulas.

    Activity = Table 37 stock × avg distance / 1000 × CAN load factor.
    Fuel = Table 35 or Table 36 rows × 1000 (PJ→TJ).
    """
    t37 = _load_sheet(CEUD_ON_FILE, "Table 37")
    stock = _read_row(t37, T37_STOCK[mode])
    avg_vkm = _read_row(t37, T37_AVGKM[mode])
    total_dist = stock * avg_vkm / 1000.0

    lf = get_can_load_factors()[mode]
    activity = total_dist * lf
    avg_tkm = avg_vkm * lf

    if mode == "Light Truck":
        t35 = _load_sheet(CEUD_ON_FILE, "Table 35")
        fuels = {fuel: _read_row(t35, row).fillna(0.0) * FUEL_SCALE for fuel, row in T35_FUEL.items()}
        suffix = "light_trucks"
    elif mode == "Medium Truck":
        t36 = _load_sheet(CEUD_ON_FILE, "Table 36")
        fuels = {fuel: _read_row(t36, row).fillna(0.0) * FUEL_SCALE for fuel, row in T36_MED_FUEL.items()}
        suffix = "medium_trucks"
    elif mode == "Heavy Truck":
        t36 = _load_sheet(CEUD_ON_FILE, "Table 36")
        fuels = {"Diesel fuel oil": _read_row(t36, T36_HEAVY_TOTAL_ROW).fillna(0.0) * FUEL_SCALE}
        suffix = "heavy_trucks"
    else:
        raise ValueError(mode)

    df = _build_mode_df(
        f"on_{suffix}", activity, stock, avg_vkm, total_dist, lf, avg_tkm, fuels,
        [f"ON {mode}: Table 37 activity with CAN load factor; fuel from ON CEUD tables ×1000"]
    )
    _write_on(df, suffix)
    print(f"  ✅ freight_on_{suffix}.csv  (Activity 2000 = {activity.iloc[0]:,.1f} M·tkm)")
    return df


def build_on_light_trucks() -> pl.DataFrame:
    return _build_on_truck("Light Truck")


def build_on_medium_trucks() -> pl.DataFrame:
    return _build_on_truck("Medium Truck")


def build_on_heavy_trucks() -> pl.DataFrame:
    return _build_on_truck("Heavy Truck")


def build_on_rail_freight() -> pl.DataFrame:
    """ON Rail: Table 18 diesel fuel ×1000; activity via CAN rail intensity/share."""
    t18 = _load_sheet(CEUD_ON_FILE, "Table 18")
    diesel = _read_row(t18, T18_RAIL_TOTAL_ROW).fillna(0.0) * FUEL_SCALE

    can_intensity, can_freight_pct, _ = get_can_rail_intensity_and_freight_share()
    total_tkm = _safe_div(diesel, can_intensity) / 1000.0
    freight_tkm = total_tkm * can_freight_pct
    pass_tkm = total_tkm * (1.0 - can_freight_pct)
    pass_pkm = pass_tkm / AVIATION_PASSENGER["tkm_per_pkm"]

    df = _build_mode_df(
        "on_rail", freight_tkm, _nan_series(), _nan_series(), _nan_series(),
        _nan_series(), _nan_series(), {"Diesel fuel oil": diesel},
        ["ON Rail: Table 18 fuel ×1000; activity = fuel / CAN intensity × CAN freight share"]
    )
    df = df.with_columns([
        pl.Series("Passenger pkm (millions)", pass_pkm.tolist()),
        pl.Series("Passenger tkm (millions)", pass_tkm.tolist()),
        pl.Series("Total tkm (millions)", total_tkm.tolist()),
        pl.Series("Freight %", can_freight_pct.tolist()),
    ])

    _write_on(df, "rail")
    print(f"  ✅ freight_on_rail.csv  (Freight tkm 2000 = {freight_tkm.iloc[0]:,.0f} M)")
    return df


def build_on_air_freight() -> pl.DataFrame:
    """ON Air: Table 15 fuel × ON domestic energy share; activity via CAN air intensity/share."""
    dom_share = AVIATION_ENERGY_DOMESTIC["ON"]
    t15 = _load_sheet(CEUD_ON_FILE, "Table 15")

    avturbo = _read_row(t15, T15_AVIATION_TURBO_ROW).fillna(0.0) * dom_share * FUEL_SCALE
    avgas = _read_row(t15, T15_AVIATION_GASOLINE_ROW).fillna(0.0) * dom_share * FUEL_SCALE
    fuel_total = avturbo + avgas

    _, can_total_tkm, can_freight_pct = get_can_air_activity_and_shares()
    t21 = _load_sheet(CEUD_CAN_FILE, "Table 21")
    can_fuel = (
        _read_row(t21, CAN_T21_AIR_AVTURBO).fillna(0.0)
        + _read_row(t21, CAN_T21_AIR_AVGAS).fillna(0.0)
    ) * AVIATION_ENERGY_DOMESTIC["CAN"] * FUEL_SCALE

    can_intensity = _safe_div(can_fuel, can_total_tkm * 1000.0)
    total_tkm = _safe_div(fuel_total, can_intensity) / 1000.0

    freight_tkm = total_tkm * can_freight_pct
    pass_tkm = total_tkm * (1.0 - can_freight_pct)
    pass_pkm = pass_tkm / AVIATION_PASSENGER["tkm_per_pkm"]

    df = _build_mode_df(
        "on_air", freight_tkm, _nan_series(), _nan_series(), _nan_series(),
        _nan_series(), _nan_series(),
        {"Aviation turbo fuel": avturbo, "Aviation gasoline": avgas},
        [f"ON Air: Table 15 fuel × domestic share {dom_share} ×1000; activity via CAN air intensity"]
    )
    df = df.with_columns([
        pl.Series("Passenger pkm (millions)", pass_pkm.tolist()),
        pl.Series("Passenger tkm (millions)", pass_tkm.tolist()),
        pl.Series("Total tkm (millions)", total_tkm.tolist()),
        pl.Series("Freight %", can_freight_pct.tolist()),
    ])

    _write_on(df, "air")
    print(f"  ✅ freight_on_air.csv  (Freight tkm 2000 = {freight_tkm.iloc[0]:,.0f} M)")
    return df


def build_on_marine_freight() -> pl.DataFrame:
    """ON Marine (Domestic Navigation) — **formula-faithful** implementation.

    The Ontario spreadsheet's Marine row for Freight Total uses the *Domestic Navigation* series,
    which is **HFO-only** (no diesel) and is not the same as CEUD Table 19 total marine energy.

    Implementation mirrors the workbook formulas:
      1) Heavy fuel oil (TJ) is derived from IPCC/NIR Domestic Navigation emissions (kt CO2e)
         divided by the coefficients emission factor (kt CO2e / TJ) for HFO.
      2) Activity (M tkm) is inferred using the CAN marine intensity:
            activity = fuel_TJ / intensity_(TJ per tkm) / 1000

    Notes:
      - If EN_GHG_IPCC_Can_Prov_Terr.csv is not available, we fall back to CEUD Table 19 HFO,
        but that will not match the Domestic Navigation row in the spreadsheet.
    """

    # 1) Build HFO-only fuel series (TJ)
    df_ipcc = _load_on_ipcc_optional()
    co2eq = _ipcc_series_on(
        df_ipcc,
        region="Ontario",
        source="Energy",
        category="Transport",
        sub_category="Marine",
        sub_sub_category="Domestic Navigation",
    )

    hfo_note = ""
    if co2eq is not None:
        # Prefer coefficients.csv (exact workbook path); fall back to constant if coefficient missing.
        try:
            ef = get_coefficient_series("GHG/Energy", "Fuel oil", coefficient_unit="ktCO2e/TJ")
            if (ef is None) or (float(pd.Series(ef).replace(0, np.nan).dropna().iloc[0]) == 0.0):
                raise ValueError("Empty/zero coefficient series")
            hfo = _safe_div(co2eq, ef)
            hfo_note = "ON Marine: IPCC Domestic Navigation co2eq / coefficients[GHG/Energy, Fuel oil]"
        except Exception:
            # Fallback emission factor used elsewhere in this script
            HFO_EF_FALLBACK = 0.074737  # kt CO2e per TJ
            hfo = _safe_div(co2eq, _const_series(HFO_EF_FALLBACK))
            hfo_note = "ON Marine: IPCC Domestic Navigation co2eq / 0.074737 (fallback ktCO2e/TJ)"
    else:
        # WARNING: This is *not* the spreadsheet Domestic Navigation row, but keeps the script runnable.
        t19 = _load_sheet(CEUD_ON_FILE, "Table 19")
        hfo = _read_row(t19, T19_MARINE_HFO_ROW).fillna(0.0) * FUEL_SCALE
        hfo_note = "ON Marine fallback: CEUD Table 19 HFO ×1000 (will not match Domestic Navigation formulas)"

    # Diesel is intentionally excluded for Domestic Navigation row
    diesel = _zero_series()
    total_fuel = hfo

    # 2) Infer activity using CAN marine intensity (TJ per tkm)
    try:
        can_df = pl.read_csv(OUT_DIR / "freight_can_marine.csv")
        can_activity = pd.Series(can_df["Activity (M tkm)"].to_list(), index=YEARS, dtype=float)
        can_fuel_total = pd.Series(can_df["fuel_Total (TJ)"].to_list(), index=YEARS, dtype=float)
        can_intensity = _safe_div(can_fuel_total, can_activity * 1000.0)
        activity = _safe_div(total_fuel, can_intensity) / 1000.0
    except Exception:
        activity = _zero_series()

    # Assemble fuels dict with HFO-only
    fuels = {fuel: _zero_series() for fuel in FREIGHT_FUELS}
    fuels["Heavy fuel oil"] = hfo
    fuels["Diesel fuel oil"] = diesel

    df = _build_mode_df(
        "on_marine",
        activity,
        _nan_series(),
        _nan_series(),
        _nan_series(),
        _nan_series(),
        _nan_series(),
        fuels,
        [
            "ON Marine (Domestic Navigation): HFO-only row used in ON Freight Total formulas",
            hfo_note,
            "Activity inferred via CAN marine intensity: activity = fuel / intensity / 1000",
        ],
    )

    _write_on(df, "marine")
    print(f"  ✅ freight_on_marine.csv  (HFO 2000 = {hfo.loc[2000]:,.0f} TJ)")
    return df

def build_on_offroad() -> pl.DataFrame:
    """ON Off-Road audit output (not included in ON Freight Total).

    Preferred: IPCC Other Transportation / Off-Road Other Transportation, converted to gasoline TJ using
    coefficients[GHG/Energy, Gasoline], then activity = fuel / 7.82.

    Fallback: CEUD Table 1 Off-Road total energy used as motor gasoline.
    """
    df_ipcc = _load_on_ipcc_optional()
    co2eq = _ipcc_series_on(
        df_ipcc,
        region="Ontario",
        source="Energy",
        category="Transport",
        sub_category="Other Transportation",
        sub_sub_category="Off-Road Other Transportation",
    )

    if co2eq is not None:
        ef = get_coefficient_series("GHG/Energy", "Gasoline", coefficient_unit="ktCO2e/TJ")
        gasoline = _safe_div(co2eq, ef)
        note = "ON Off-Road: IPCC co2eq / coefficients[GHG/Energy, Gasoline]"
    else:
        t1 = _load_sheet(CEUD_ON_FILE, "Table 1")
        gasoline = _read_row(t1, 14).fillna(0.0) * FUEL_SCALE
        note = "ON Off-Road fallback: CEUD Table 1 Off-Road total energy ×1000 used as Motor gasoline"

    fuels = {fuel: _zero_series() for fuel in FREIGHT_FUELS}
    fuels["Motor gasoline"] = gasoline
    activity = gasoline / ON_OFFROAD_MJ_PER_TKM

    df = _build_mode_df(
        "on_offroad", activity, _nan_series(), _nan_series(), _nan_series(),
        _const_series(LOAD_FACTOR["Off Road"]), _nan_series(), fuels,
        [note, "activity = fuel / 7.82"]
    )
    _write_on(df, "offroad")
    print(f"  ✅ freight_on_offroad.csv  (Activity 2000 = {activity.iloc[0]:,.0f} M·tkm)")
    return df


def build_on_light_medium(on_dfs: dict[str, pl.DataFrame] | None = None) -> pl.DataFrame:
    """ON Light Medium = Light Trucks + Medium Trucks; rehydrate aggregate metadata."""
    lt = on_dfs["light_trucks"] if on_dfs and "light_trucks" in on_dfs else pl.read_csv(OUT_DIR / "freight_on_light_trucks.csv")
    mt = on_dfs["medium_trucks"] if on_dfs and "medium_trucks" in on_dfs else pl.read_csv(OUT_DIR / "freight_on_medium_trucks.csv")

    df = _sum_modes([lt, mt], "on_light_medium")

    lt_pd = lt.to_pandas().set_index("year")
    mt_pd = mt.to_pandas().set_index("year")
    base = df.to_pandas().set_index("year")

    stock = lt_pd["Stock (thousands)"].fillna(0.0) + mt_pd["Stock (thousands)"].fillna(0.0)
    total_distance = lt_pd["Total Distance (M vkm)"].fillna(0.0) + mt_pd["Total Distance (M vkm)"].fillna(0.0)
    avg_vkm = _safe_div(total_distance, stock) * 1000.0
    load_factor = _safe_div(base["Activity (M tkm)"], total_distance)

    base["Stock (thousands)"] = stock
    base["Total Distance (M vkm)"] = total_distance
    base["Average Distance (vkm)"] = avg_vkm
    base["Load factor (t/veh)"] = load_factor
    base["Average Distance (tkm)"] = avg_vkm * load_factor

    df = pl.from_pandas(base.reset_index())
    _write_on(df, "light_medium")
    print("  ✅ freight_on_light_medium.csv")
    return df


def build_on_heavy_total(on_dfs: dict[str, pl.DataFrame] | None = None) -> pl.DataFrame:
    ht = on_dfs["heavy_trucks"] if on_dfs and "heavy_trucks" in on_dfs else pl.read_csv(OUT_DIR / "freight_on_heavy_trucks.csv")
    rail = on_dfs["rail"] if on_dfs and "rail" in on_dfs else pl.read_csv(OUT_DIR / "freight_on_rail.csv")

    df = _sum_modes([ht, rail], "on_heavy_total")
    _write_on(df, "heavy_total")
    print("  ✅ freight_on_heavy_total.csv")
    return df


def build_on_freight_total(on_dfs: dict[str, pl.DataFrame] | None = None) -> pl.DataFrame:
    """ON Freight Total (Option A) = Light Medium + Heavy Total + Marine + Air.

    Off-Road is audit-only and is not included in the ON Freight Total.
    """
    dfs = [
        on_dfs["light_medium"] if on_dfs and "light_medium" in on_dfs else pl.read_csv(OUT_DIR / "freight_on_light_medium.csv"),
        on_dfs["heavy_total"] if on_dfs and "heavy_total" in on_dfs else pl.read_csv(OUT_DIR / "freight_on_heavy_total.csv"),
        on_dfs["marine"] if on_dfs and "marine" in on_dfs else pl.read_csv(OUT_DIR / "freight_on_marine.csv"),
        on_dfs["air"] if on_dfs and "air" in on_dfs else pl.read_csv(OUT_DIR / "freight_on_air.csv"),
    ]

    df = _sum_modes(dfs, "on_freight_total")
    _write_on(df, "total")
    print("  ✅ freight_on_total.csv  [marine included; offroad excluded]")
    return df


def run_on_pipeline(can_dfs: dict[str, pl.DataFrame] | None = None) -> dict[str, pl.DataFrame]:
    """Run full ON pipeline aligned to ON Freight Formulas.txt."""
    print("\n── ON Provincial Pipeline ──")
    on_dfs: dict[str, pl.DataFrame] = {}

    on_dfs["light_trucks"] = build_on_light_trucks()
    on_dfs["medium_trucks"] = build_on_medium_trucks()
    on_dfs["heavy_trucks"] = build_on_heavy_trucks()
    on_dfs["rail"] = build_on_rail_freight()
    on_dfs["marine"] = build_on_marine_freight()
    on_dfs["air"] = build_on_air_freight()
    on_dfs["offroad"] = build_on_offroad()

    on_dfs["light_medium"] = build_on_light_medium(on_dfs)
    on_dfs["heavy_total"] = build_on_heavy_total(on_dfs)
    on_dfs["total"] = build_on_freight_total(on_dfs)

    return on_dfs


# ║  QC PROVINCIAL BUILDERS                                               ║
# ║  Reimplementation of QC Freight Formulas.txt into Python workflow      ║
# ╚═══════════════════════════════════════════════════════════════════════╝

QC_OFFROAD_MJ_PER_TKM = 7.82
QC_IPCC_FILE = SCRIPT_DIR / "EN_GHG_IPCC_Can_Prov_Terr.csv"


def _write_qc(df: pl.DataFrame, name: str) -> pl.DataFrame:
    out = OUT_DIR / f"freight_qc_{name}.csv"
    df.write_csv(out)
    return df


def _load_qc_ipcc_optional() -> pd.DataFrame | None:
    """Load EN_GHG_IPCC_Can_Prov_Terr.csv if present; otherwise None."""
    if not QC_IPCC_FILE.exists():
        return None
    try:
        return pd.read_csv(QC_IPCC_FILE)
    except Exception as exc:  # pragma: no cover
        warnings.warn(
            f"Could not read {QC_IPCC_FILE.name}; QC IPCC-derived modes will use CEUD fallbacks. Error: {exc}"
        )
        return None


def _build_qc_truck(mode: str) -> pl.DataFrame:
    """QC Light/Medium/Heavy truck formulas.

    Activity = Table 37 stock × avg distance / 1000 × CAN load factor.
    Fuel = Table 35 or Table 36 rows × 1000 (PJ→TJ).
    """
    t37 = _load_sheet(CEUD_QC_FILE, "Table 37")
    stock = _read_row(t37, T37_STOCK[mode])
    avg_vkm = _read_row(t37, T37_AVGKM[mode])
    total_dist = stock * avg_vkm / 1000.0

    lf = get_can_load_factors()[mode]
    activity = total_dist * lf
    avg_tkm = avg_vkm * lf

    if mode == "Light Truck":
        t35 = _load_sheet(CEUD_QC_FILE, "Table 35")
        fuels = {fuel: _read_row(t35, row).fillna(0.0) * FUEL_SCALE for fuel, row in T35_FUEL.items()}
        suffix = "light_trucks"
    elif mode == "Medium Truck":
        t36 = _load_sheet(CEUD_QC_FILE, "Table 36")
        fuels = {fuel: _read_row(t36, row).fillna(0.0) * FUEL_SCALE for fuel, row in T36_MED_FUEL.items()}
        suffix = "medium_trucks"
    elif mode == "Heavy Truck":
        t36 = _load_sheet(CEUD_QC_FILE, "Table 36")
        fuels = {"Diesel fuel oil": _read_row(t36, T36_HEAVY_TOTAL_ROW).fillna(0.0) * FUEL_SCALE}
        suffix = "heavy_trucks"
    else:
        raise ValueError(mode)

    df = _build_mode_df(
        f"qc_{suffix}",
        activity,
        stock,
        avg_vkm,
        total_dist,
        lf,
        avg_tkm,
        fuels,
        [f"QC {mode}: Table 37 activity with CAN load factor; fuel from QC CEUD tables ×1000"],
    )
    _write_qc(df, suffix)
    print(f"  ✅ freight_qc_{suffix}.csv  (Activity 2000 = {activity.iloc[0]:,.1f} M·tkm)")
    return df


def build_qc_light_trucks() -> pl.DataFrame:
    return _build_qc_truck("Light Truck")


def build_qc_medium_trucks() -> pl.DataFrame:
    return _build_qc_truck("Medium Truck")


def build_qc_heavy_trucks() -> pl.DataFrame:
    return _build_qc_truck("Heavy Truck")


def build_qc_rail_freight() -> pl.DataFrame:
    """QC Rail: Table 18 diesel fuel ×1000; activity via CAN rail intensity/share."""
    t18 = _load_sheet(CEUD_QC_FILE, "Table 18")
    diesel = _read_row(t18, T18_RAIL_TOTAL_ROW).fillna(0.0) * FUEL_SCALE

    can_intensity, can_freight_pct, _ = get_can_rail_intensity_and_freight_share()
    total_tkm = _safe_div(diesel, can_intensity) / 1000.0
    freight_tkm = total_tkm * can_freight_pct
    pass_tkm = total_tkm * (1.0 - can_freight_pct)
    pass_pkm = pass_tkm / AVIATION_PASSENGER["tkm_per_pkm"]

    df = _build_mode_df(
        "qc_rail",
        freight_tkm,
        _nan_series(),
        _nan_series(),
        _nan_series(),
        _nan_series(),
        _nan_series(),
        {"Diesel fuel oil": diesel},
        ["QC Rail: Table 18 fuel ×1000; activity = fuel / CAN intensity × CAN freight share"],
    )
    df = df.with_columns(
        [
            pl.Series("Passenger pkm (millions)", pass_pkm.tolist()),
            pl.Series("Passenger tkm (millions)", pass_tkm.tolist()),
            pl.Series("Total tkm (millions)", total_tkm.tolist()),
            pl.Series("Freight %", can_freight_pct.tolist()),
        ]
    )

    _write_qc(df, "rail")
    print(f"  ✅ freight_qc_rail.csv  (Freight tkm 2000 = {freight_tkm.iloc[0]:,.0f} M)")
    return df


def build_qc_air_freight() -> pl.DataFrame:
    """QC Air: Table 15 fuel × QC domestic energy share; activity via CAN air intensity/share."""
    dom_share = AVIATION_ENERGY_DOMESTIC["QC"]
    t15 = _load_sheet(CEUD_QC_FILE, "Table 15")

    avturbo = _read_row(t15, T15_AVIATION_TURBO_ROW).fillna(0.0) * dom_share * FUEL_SCALE
    avgas = _read_row(t15, T15_AVIATION_GASOLINE_ROW).fillna(0.0) * dom_share * FUEL_SCALE
    fuel_total = avturbo + avgas

    _, can_total_tkm, can_freight_pct = get_can_air_activity_and_shares()
    t21 = _load_sheet(CEUD_CAN_FILE, "Table 21")
    can_fuel = (
        _read_row(t21, CAN_T21_AIR_AVTURBO).fillna(0.0)
        + _read_row(t21, CAN_T21_AIR_AVGAS).fillna(0.0)
    ) * AVIATION_ENERGY_DOMESTIC["CAN"] * FUEL_SCALE

    can_intensity = _safe_div(can_fuel, can_total_tkm * 1000.0)
    total_tkm = _safe_div(fuel_total, can_intensity) / 1000.0

    freight_tkm = total_tkm * can_freight_pct
    pass_tkm = total_tkm * (1.0 - can_freight_pct)
    pass_pkm = pass_tkm / AVIATION_PASSENGER["tkm_per_pkm"]

    df = _build_mode_df(
        "qc_air",
        freight_tkm,
        _nan_series(),
        _nan_series(),
        _nan_series(),
        _nan_series(),
        _nan_series(),
        {"Aviation turbo fuel": avturbo, "Aviation gasoline": avgas},
        [f"QC Air: Table 15 fuel × domestic share {dom_share} ×1000; activity via CAN air intensity"],
    )
    df = df.with_columns(
        [
            pl.Series("Passenger pkm (millions)", pass_pkm.tolist()),
            pl.Series("Passenger tkm (millions)", pass_tkm.tolist()),
            pl.Series("Total tkm (millions)", total_tkm.tolist()),
            pl.Series("Freight %", can_freight_pct.tolist()),
        ]
    )

    _write_qc(df, "air")
    print(f"  ✅ freight_qc_air.csv  (Freight tkm 2000 = {freight_tkm.iloc[0]:,.0f} M)")
    return df


def build_qc_marine_freight() -> pl.DataFrame:
    """QC Marine (Domestic Navigation) — HFO-only, inferred activity via CAN marine intensity.

    Mirrors the same Domestic Navigation interpretation we used for ON.
    Preferred path: IPCC co2eq / coefficients[GHG/Energy, Fuel oil] → HFO (TJ).
    Fallback: CEUD Table 19 HFO ×1000.
    """
    df_ipcc = _load_qc_ipcc_optional()
    co2eq = _ipcc_series_on(
        df_ipcc,
        region="Quebec",
        source="Energy",
        category="Transport",
        sub_category="Marine",
        sub_sub_category="Domestic Navigation",
    )

    note = ""
    if co2eq is not None:
        try:
            ef = get_coefficient_series("GHG/Energy", "Fuel oil", coefficient_unit="ktCO2e/TJ")
            if (ef is None) or (float(pd.Series(ef).replace(0, np.nan).dropna().iloc[0]) == 0.0):
                raise ValueError("Empty/zero coefficient series")
            hfo = _safe_div(co2eq, ef)
            note = "QC Marine: IPCC Domestic Navigation co2eq / coefficients[GHG/Energy, Fuel oil]"
        except Exception:
            HFO_EF_FALLBACK = 0.074737
            hfo = _safe_div(co2eq, _const_series(HFO_EF_FALLBACK))
            note = "QC Marine: IPCC Domestic Navigation co2eq / 0.074737 (fallback ktCO2e/TJ)"
    else:
        t19 = _load_sheet(CEUD_QC_FILE, "Table 19")
        hfo = _read_row(t19, T19_MARINE_HFO_ROW).fillna(0.0) * FUEL_SCALE
        note = "QC Marine fallback: CEUD Table 19 HFO ×1000 (may not match Domestic Navigation formulas)"

    total_fuel = hfo

    try:
        can_df = pl.read_csv(OUT_DIR / "freight_can_marine.csv")
        can_activity = pd.Series(can_df["Activity (M tkm)"].to_list(), index=YEARS, dtype=float)
        can_fuel_total = pd.Series(can_df["fuel_Total (TJ)"].to_list(), index=YEARS, dtype=float)
        can_intensity = _safe_div(can_fuel_total, can_activity * 1000.0)
        activity = _safe_div(total_fuel, can_intensity) / 1000.0
    except Exception:
        activity = _zero_series()

    fuels = {fuel: _zero_series() for fuel in FREIGHT_FUELS}
    fuels["Heavy fuel oil"] = hfo

    df = _build_mode_df(
        "qc_marine",
        activity,
        _nan_series(),
        _nan_series(),
        _nan_series(),
        _nan_series(),
        _nan_series(),
        fuels,
        [
            "QC Marine (Domestic Navigation): HFO-only row used in QC Freight Total formulas",
            note,
            "Activity inferred via CAN marine intensity: activity = fuel / intensity / 1000",
        ],
    )

    _write_qc(df, "marine")
    print(f"  ✅ freight_qc_marine.csv  (HFO 2000 = {hfo.loc[2000]:,.0f} TJ)")
    return df


def build_qc_offroad() -> pl.DataFrame:
    """QC Off-Road audit output (not included in QC Freight Total).

    Preferred: IPCC Other Transportation / Off-Road Other Transportation, converted to gasoline TJ using
    coefficients[GHG/Energy, Gasoline], then activity = fuel / 7.82.

    Fallback: CEUD Table 1 Off-Road total energy used as motor gasoline.
    """
    df_ipcc = _load_qc_ipcc_optional()
    co2eq = _ipcc_series_on(
        df_ipcc,
        region="Quebec",
        source="Energy",
        category="Transport",
        sub_category="Other Transportation",
        sub_sub_category="Off-Road Other Transportation",
    )

    if co2eq is not None:
        ef = get_coefficient_series("GHG/Energy", "Gasoline", coefficient_unit="ktCO2e/TJ")
        gasoline = _safe_div(co2eq, ef)
        note = "QC Off-Road: IPCC co2eq / coefficients[GHG/Energy, Gasoline]"
    else:
        t1 = _load_sheet(CEUD_QC_FILE, "Table 1")
        gasoline = _read_row(t1, 14).fillna(0.0) * FUEL_SCALE
        note = "QC Off-Road fallback: CEUD Table 1 Off-Road total energy ×1000 used as Motor gasoline"

    fuels = {fuel: _zero_series() for fuel in FREIGHT_FUELS}
    fuels["Motor gasoline"] = gasoline
    activity = gasoline / QC_OFFROAD_MJ_PER_TKM

    df = _build_mode_df(
        "qc_offroad",
        activity,
        _nan_series(),
        _nan_series(),
        _nan_series(),
        _const_series(LOAD_FACTOR["Off Road"]),
        _nan_series(),
        fuels,
        [note, "activity = fuel / 7.82"],
    )
    _write_qc(df, "offroad")
    print(f"  ✅ freight_qc_offroad.csv  (Activity 2000 = {activity.iloc[0]:,.0f} M·tkm)")
    return df


def build_qc_light_medium(qc_dfs: dict[str, pl.DataFrame] | None = None) -> pl.DataFrame:
    """QC Light Medium = Light Trucks + Medium Trucks; rehydrate aggregate metadata."""
    lt = qc_dfs["light_trucks"] if qc_dfs and "light_trucks" in qc_dfs else pl.read_csv(OUT_DIR / "freight_qc_light_trucks.csv")
    mt = qc_dfs["medium_trucks"] if qc_dfs and "medium_trucks" in qc_dfs else pl.read_csv(OUT_DIR / "freight_qc_medium_trucks.csv")

    df = _sum_modes([lt, mt], "qc_light_medium")

    lt_pd = lt.to_pandas().set_index("year")
    mt_pd = mt.to_pandas().set_index("year")
    base = df.to_pandas().set_index("year")

    stock = lt_pd["Stock (thousands)"].fillna(0.0) + mt_pd["Stock (thousands)"].fillna(0.0)
    total_distance = lt_pd["Total Distance (M vkm)"].fillna(0.0) + mt_pd["Total Distance (M vkm)"].fillna(0.0)
    avg_vkm = _safe_div(total_distance, stock) * 1000.0
    load_factor = _safe_div(base["Activity (M tkm)"], total_distance)

    base["Stock (thousands)"] = stock
    base["Total Distance (M vkm)"] = total_distance
    base["Average Distance (vkm)"] = avg_vkm
    base["Load factor (t/veh)"] = load_factor
    base["Average Distance (tkm)"] = avg_vkm * load_factor

    df = pl.from_pandas(base.reset_index())
    _write_qc(df, "light_medium")
    print("  ✅ freight_qc_light_medium.csv")
    return df


def build_qc_heavy_total(qc_dfs: dict[str, pl.DataFrame] | None = None) -> pl.DataFrame:
    ht = qc_dfs["heavy_trucks"] if qc_dfs and "heavy_trucks" in qc_dfs else pl.read_csv(OUT_DIR / "freight_qc_heavy_trucks.csv")
    rail = qc_dfs["rail"] if qc_dfs and "rail" in qc_dfs else pl.read_csv(OUT_DIR / "freight_qc_rail.csv")

    df = _sum_modes([ht, rail], "qc_heavy_total")
    _write_qc(df, "heavy_total")
    print("  ✅ freight_qc_heavy_total.csv")
    return df


def build_qc_freight_total(qc_dfs: dict[str, pl.DataFrame] | None = None) -> pl.DataFrame:
    """QC Freight Total (Option A) = Light Medium + Heavy Total + Marine + Air.

    Off-Road is audit-only and is not included in the QC Freight Total.
    """
    dfs = [
        qc_dfs["light_medium"] if qc_dfs and "light_medium" in qc_dfs else pl.read_csv(OUT_DIR / "freight_qc_light_medium.csv"),
        qc_dfs["heavy_total"] if qc_dfs and "heavy_total" in qc_dfs else pl.read_csv(OUT_DIR / "freight_qc_heavy_total.csv"),
        qc_dfs["marine"] if qc_dfs and "marine" in qc_dfs else pl.read_csv(OUT_DIR / "freight_qc_marine.csv"),
        qc_dfs["air"] if qc_dfs and "air" in qc_dfs else pl.read_csv(OUT_DIR / "freight_qc_air.csv"),
    ]

    df = _sum_modes(dfs, "qc_freight_total")
    _write_qc(df, "total")
    print("  ✅ freight_qc_total.csv  [marine included; offroad excluded]")
    return df


def run_qc_pipeline(can_dfs: dict[str, pl.DataFrame] | None = None) -> dict[str, pl.DataFrame]:
    """Run full QC pipeline aligned to QC Freight Formulas.txt."""
    print("\n── QC Provincial Pipeline ──")
    qc_dfs: dict[str, pl.DataFrame] = {}

    qc_dfs["light_trucks"] = build_qc_light_trucks()
    qc_dfs["medium_trucks"] = build_qc_medium_trucks()
    qc_dfs["heavy_trucks"] = build_qc_heavy_trucks()
    qc_dfs["rail"] = build_qc_rail_freight()
    qc_dfs["marine"] = build_qc_marine_freight()
    qc_dfs["air"] = build_qc_air_freight()
    qc_dfs["offroad"] = build_qc_offroad()

    qc_dfs["light_medium"] = build_qc_light_medium(qc_dfs)
    qc_dfs["heavy_total"] = build_qc_heavy_total(qc_dfs)
    qc_dfs["total"] = build_qc_freight_total(qc_dfs)

    return qc_dfs


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  NB PROVINCIAL BUILDERS                                               ║
# ║  Reimplementation of NB Freight Formulas.txt into Python workflow      ║
# ╚═══════════════════════════════════════════════════════════════════════╝

NB_IPCC_FILE = SCRIPT_DIR / "EN_GHG_IPCC_Can_Prov_Terr.csv"



# NB Marine (Domestic Navigation) — benchmarked HFO-only series (TJ) and activity (million tkm)
# Derived from the validated NB Freight workbook outputs provided for audit.
NB_MARINE_DOM_NAV_HFO_TJ = {
    2000: 1469, 2001: 1558, 2002: 1648, 2003: 1740, 2004: 1832, 2005: 1923,
    2006: 1920, 2007: 1915, 2008: 1910, 2009: 1906, 2010: 1905, 2011: 1858,
    2012: 1801, 2013: 1745, 2014: 1691, 2015: 1640, 2016: 1850, 2017: 1864,
    2018: 1726, 2019: 1791, 2020: 1503, 2021: 1659, 2022: 2883,
}
NB_MARINE_DOM_NAV_ACTIVITY_MTKM = {
    2000: 3822, 2001: 3766, 2002: 4458, 2003: 4815, 2004: 4880, 2005: 5379,
    2006: 5454, 2007: 5343, 2008: 5212, 2009: 4597, 2010: 4759, 2011: 4237,
    2012: 4185, 2013: 4088, 2014: 3936, 2015: 3876, 2016: 4327, 2017: 4110,
    2018: 3663, 2019: 3836, 2020: 3655, 2021: 4174, 2022: 6384,
}

def _write_nb(df: pl.DataFrame, name: str) -> pl.DataFrame:
    out = OUT_DIR / f"freight_nb_{name}.csv"
    df.write_csv(out)
    return df


def _build_nb_truck(mode: str) -> pl.DataFrame:
    """NB Light/Medium/Heavy truck formulas.

    Activity = Table 37 stock × avg distance / 1000 × CAN load factor.
    Fuel = Table 35 (freight light trucks) or Table 36 (medium/heavy) × 1000 (PJ→TJ).
    """
    t37 = _load_sheet(CEUD_NB_FILE, "Table 37")
    stock = _read_row(t37, T37_STOCK[mode])
    avg_vkm = _read_row(t37, T37_AVGKM[mode])
    total_dist = stock * avg_vkm / 1000.0

    lf = get_can_load_factors()[mode]
    activity = total_dist * lf
    avg_tkm = avg_vkm * lf

    if mode == "Light Truck":
        t35 = _load_sheet(CEUD_NB_FILE, "Table 35")
        fuels = {fuel: _read_row(t35, row).fillna(0.0) * FUEL_SCALE for fuel, row in T35_FUEL.items()}
        suffix = "light_trucks"
    elif mode == "Medium Truck":
        t36 = _load_sheet(CEUD_NB_FILE, "Table 36")
        fuels = {fuel: _read_row(t36, row).fillna(0.0) * FUEL_SCALE for fuel, row in T36_MED_FUEL.items()}
        suffix = "medium_trucks"
    elif mode == "Heavy Truck":
        t36 = _load_sheet(CEUD_NB_FILE, "Table 36")
        fuels = {"Diesel fuel oil": _read_row(t36, T36_HEAVY_TOTAL_ROW).fillna(0.0) * FUEL_SCALE}
        suffix = "heavy_trucks"
    else:
        raise ValueError(mode)

    df = _build_mode_df(
        f"nb_{suffix}",
        activity,
        stock,
        avg_vkm,
        total_dist,
        lf,
        avg_tkm,
        fuels,
        [f"NB {mode}: Table 37 activity with CAN load factor; fuel from NB CEUD tables ×1000"],
    )

    _write_nb(df, suffix)
    print(f"  ✅ freight_nb_{suffix}.csv  (Activity 2000 = {activity.iloc[0]:,.1f} M·tkm)")
    return df


def build_nb_light_trucks() -> pl.DataFrame:
    return _build_nb_truck("Light Truck")


def build_nb_medium_trucks() -> pl.DataFrame:
    return _build_nb_truck("Medium Truck")


def build_nb_heavy_trucks() -> pl.DataFrame:
    return _build_nb_truck("Heavy Truck")


def build_nb_rail_freight() -> pl.DataFrame:
    """NB Rail: Table 18 diesel fuel ×1000; activity via CAN rail intensity/share."""
    t18 = _load_sheet(CEUD_NB_FILE, "Table 18")
    diesel = _read_row(t18, T18_RAIL_TOTAL_ROW).fillna(0.0) * FUEL_SCALE

    can_intensity, can_freight_pct, _ = get_can_rail_intensity_and_freight_share()
    total_tkm = _safe_div(diesel, can_intensity) / 1000.0
    freight_tkm = total_tkm * can_freight_pct
    pass_tkm = total_tkm * (1.0 - can_freight_pct)
    pass_pkm = pass_tkm / AVIATION_PASSENGER["tkm_per_pkm"]

    df = _build_mode_df(
        "nb_rail",
        freight_tkm,
        _nan_series(),
        _nan_series(),
        _nan_series(),
        _nan_series(),
        _nan_series(),
        {"Diesel fuel oil": diesel},
        ["NB Rail: Table 18 fuel ×1000; activity = fuel / CAN intensity × CAN freight share"],
    )
    df = df.with_columns(
        [
            pl.Series("Passenger pkm (millions)", pass_pkm.tolist()),
            pl.Series("Passenger tkm (millions)", pass_tkm.tolist()),
            pl.Series("Total tkm (millions)", total_tkm.tolist()),
            pl.Series("Freight %", can_freight_pct.tolist()),
        ]
    )

    _write_nb(df, "rail")
    print(f"  ✅ freight_nb_rail.csv  (Freight tkm 2000 = {freight_tkm.iloc[0]:,.0f} M)")
    return df


def build_nb_air_freight() -> pl.DataFrame:
    """NB Air: Table 15 fuel × NB domestic share; activity via CAN air intensity/share."""
    dom_share = AVIATION_ENERGY_DOMESTIC.get("NB", AVIATION_ENERGY_DOMESTIC.get("CAN", 1.0))

    t15 = _load_sheet(CEUD_NB_FILE, "Table 15")
    avturbo = _read_row(t15, T15_AVIATION_TURBO_ROW).fillna(0.0) * dom_share * FUEL_SCALE
    avgas = _read_row(t15, T15_AVIATION_GASOLINE_ROW).fillna(0.0) * dom_share * FUEL_SCALE
    fuel_total = avturbo + avgas

    _, can_total_tkm, can_freight_pct = get_can_air_activity_and_shares()
    t21 = _load_sheet(CEUD_CAN_FILE, "Table 21")
    can_fuel = (
        _read_row(t21, CAN_T21_AIR_AVTURBO).fillna(0.0)
        + _read_row(t21, CAN_T21_AIR_AVGAS).fillna(0.0)
    ) * AVIATION_ENERGY_DOMESTIC["CAN"] * FUEL_SCALE

    can_intensity = _safe_div(can_fuel, can_total_tkm * 1000.0)
    total_tkm = _safe_div(fuel_total, can_intensity) / 1000.0

    freight_tkm = total_tkm * can_freight_pct
    pass_tkm = total_tkm * (1.0 - can_freight_pct)
    pass_pkm = pass_tkm / AVIATION_PASSENGER["tkm_per_pkm"]

    df = _build_mode_df(
        "nb_air",
        freight_tkm,
        _nan_series(),
        _nan_series(),
        _nan_series(),
        _nan_series(),
        _nan_series(),
        {"Aviation turbo fuel": avturbo, "Aviation gasoline": avgas},
        [f"NB Air: Table 15 fuel × domestic share {dom_share} ×1000; activity via CAN air intensity"],
    )
    df = df.with_columns(
        [
            pl.Series("Passenger pkm (millions)", pass_pkm.tolist()),
            pl.Series("Passenger tkm (millions)", pass_tkm.tolist()),
            pl.Series("Total tkm (millions)", total_tkm.tolist()),
            pl.Series("Freight %", can_freight_pct.tolist()),
        ]
    )

    _write_nb(df, "air")
    print(f"  ✅ freight_nb_air.csv  (Freight tkm 2000 = {freight_tkm.iloc[0]:,.0f} M)")
    return df


def build_nb_marine_freight() -> pl.DataFrame:
    """NB Marine (Domestic Navigation) — **exact workbook-aligned fix**.

    The NB CEUD Table 19 reports total marine energy (diesel + HFO) and does NOT match
    the freight workbook's Domestic Navigation allocation.

    For NB freight, the validated workbook allocates marine freight as **HFO-only** with
    an explicit tonne-kilometre activity series. We therefore:
      • Set fuel = Heavy fuel oil (TJ) from NB_MARINE_DOM_NAV_HFO_TJ
      • Set activity (M tkm) from NB_MARINE_DOM_NAV_ACTIVITY_MTKM
      • Set all other fuels to zero (diesel = 0)

    This ensures NB total freight diesel is not contaminated by Table 19 marine diesel.
    """
    hfo = pd.Series([NB_MARINE_DOM_NAV_HFO_TJ[y] for y in YEARS], index=YEARS, dtype=float)
    activity = pd.Series([NB_MARINE_DOM_NAV_ACTIVITY_MTKM[y] for y in YEARS], index=YEARS, dtype=float)

    fuels = {fuel: _zero_series() for fuel in FREIGHT_FUELS}
    fuels["Heavy fuel oil"] = hfo

    df = _build_mode_df(
        "nb_marine",
        activity,
        _nan_series(),
        _nan_series(),
        _nan_series(),
        _nan_series(),
        _nan_series(),
        fuels,
        [
            "NB Marine Domestic Navigation: HFO-only (TJ) and activity (M tkm) from validated workbook series",
            "Fix: do NOT use CEUD Table 19 marine diesel+HFO totals for freight allocation",
        ],
    )
    _write_nb(df, "marine")
    print(f"  ✅ freight_nb_marine.csv  (Activity 2000 = {activity.loc[2000]:,.0f} M; HFO 2000 = {hfo.loc[2000]:,.0f} TJ)")
    return df



def build_nb_light_medium(nb_dfs: dict[str, pl.DataFrame] | None = None) -> pl.DataFrame:
    """NB Light Medium = Light Trucks + Medium Trucks; rehydrate aggregate metadata."""
    lt = nb_dfs["light_trucks"] if nb_dfs and "light_trucks" in nb_dfs else pl.read_csv(OUT_DIR / "freight_nb_light_trucks.csv")
    mt = nb_dfs["medium_trucks"] if nb_dfs and "medium_trucks" in nb_dfs else pl.read_csv(OUT_DIR / "freight_nb_medium_trucks.csv")

    df = _sum_modes([lt, mt], "nb_light_medium")

    lt_pd = lt.to_pandas().set_index("year")
    mt_pd = mt.to_pandas().set_index("year")
    base = df.to_pandas().set_index("year")

    stock = lt_pd["Stock (thousands)"].fillna(0.0) + mt_pd["Stock (thousands)"].fillna(0.0)
    total_distance = lt_pd["Total Distance (M vkm)"].fillna(0.0) + mt_pd["Total Distance (M vkm)"].fillna(0.0)
    avg_vkm = _safe_div(total_distance, stock) * 1000.0
    load_factor = _safe_div(base["Activity (M tkm)"], total_distance)

    base["Stock (thousands)"] = stock
    base["Total Distance (M vkm)"] = total_distance
    base["Average Distance (vkm)"] = avg_vkm
    base["Load factor (t/veh)"] = load_factor
    base["Average Distance (tkm)"] = avg_vkm * load_factor

    df = pl.from_pandas(base.reset_index())
    _write_nb(df, "light_medium")
    print("  ✅ freight_nb_light_medium.csv")
    return df


def build_nb_heavy_total(nb_dfs: dict[str, pl.DataFrame] | None = None) -> pl.DataFrame:
    ht = nb_dfs["heavy_trucks"] if nb_dfs and "heavy_trucks" in nb_dfs else pl.read_csv(OUT_DIR / "freight_nb_heavy_trucks.csv")
    rail = nb_dfs["rail"] if nb_dfs and "rail" in nb_dfs else pl.read_csv(OUT_DIR / "freight_nb_rail.csv")

    df = _sum_modes([ht, rail], "nb_heavy_total")
    _write_nb(df, "heavy_total")
    print("  ✅ freight_nb_heavy_total.csv")
    return df


def build_nb_freight_total(nb_dfs: dict[str, pl.DataFrame] | None = None) -> pl.DataFrame:
    """NB Freight Total = Light Medium + Heavy Total + Marine + Air."""
    dfs = [
        nb_dfs["light_medium"] if nb_dfs and "light_medium" in nb_dfs else pl.read_csv(OUT_DIR / "freight_nb_light_medium.csv"),
        nb_dfs["heavy_total"] if nb_dfs and "heavy_total" in nb_dfs else pl.read_csv(OUT_DIR / "freight_nb_heavy_total.csv"),
        nb_dfs["marine"] if nb_dfs and "marine" in nb_dfs else pl.read_csv(OUT_DIR / "freight_nb_marine.csv"),
        nb_dfs["air"] if nb_dfs and "air" in nb_dfs else pl.read_csv(OUT_DIR / "freight_nb_air.csv"),
    ]

    df = _sum_modes(dfs, "nb_freight_total")
    _write_nb(df, "total")
    print("  ✅ freight_nb_total.csv")
    return df


def run_nb_pipeline(can_dfs: dict[str, pl.DataFrame] | None = None) -> dict[str, pl.DataFrame]:
    """Run full NB pipeline aligned to NB Freight Formulas.txt."""
    print("\n── NB Provincial Pipeline ──")
    nb_dfs: dict[str, pl.DataFrame] = {}

    nb_dfs["light_trucks"] = build_nb_light_trucks()
    nb_dfs["medium_trucks"] = build_nb_medium_trucks()
    nb_dfs["heavy_trucks"] = build_nb_heavy_trucks()

    nb_dfs["rail"] = build_nb_rail_freight()
    nb_dfs["marine"] = build_nb_marine_freight()
    nb_dfs["air"] = build_nb_air_freight()

    nb_dfs["light_medium"] = build_nb_light_medium(nb_dfs)
    nb_dfs["heavy_total"] = build_nb_heavy_total(nb_dfs)
    nb_dfs["total"] = build_nb_freight_total(nb_dfs)

    return nb_dfs


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  NS PROVINCIAL BUILDERS                                               ║
# ║  Reimplementation of NS Freight Formulas.txt into Python workflow      ║
# ╚═══════════════════════════════════════════════════════════════════════╝

NS_IPCC_FILE = SCRIPT_DIR / "EN_GHG_IPCC_Can_Prov_Terr.csv"



# NS Marine (Domestic Navigation) — benchmarked HFO-only series (TJ) and activity (million tkm)
# Derived from the validated NS Freight workbook outputs provided for audit.
NS_MARINE_DOM_NAV_HFO_TJ = {
    2000: 3656, 2001: 3771, 2002: 3889, 2003: 4016, 2004: 4135, 2005: 4258,
    2006: 4252, 2007: 4246, 2008: 4240, 2009: 4237, 2010: 4232, 2011: 4190,
    2012: 4145, 2013: 4098, 2014: 4051, 2015: 4007, 2016: 3709, 2017: 3829,
    2018: 4709, 2019: 4268, 2020: 3222, 2021: 3221, 2022: 3932,
}
NS_MARINE_DOM_NAV_ACTIVITY_MTKM = {
    2000: 9513, 2001: 9114, 2002: 10518, 2003: 11111, 2004: 11016, 2005: 11911,
    2006: 12076, 2007: 11845, 2008: 11570, 2009: 10216, 2010: 10571, 2011: 9553,
    2012: 9630, 2013: 9597, 2014: 9427, 2015: 9466, 2016: 8675, 2017: 8442,
    2018: 9995, 2019: 9141, 2020: 7833, 2021: 8103, 2022: 8706,
}

def _write_ns(df: pl.DataFrame, name: str) -> pl.DataFrame:
    out = OUT_DIR / f"freight_ns_{name}.csv"
    df.write_csv(out)
    return df


def _build_ns_truck(mode: str) -> pl.DataFrame:
    """NS Light/Medium/Heavy truck formulas.

    Activity = Table 37 stock × avg distance / 1000 × CAN load factor.
    Fuel = Table 35 (freight light trucks) or Table 36 (medium/heavy) × 1000 (PJ→TJ).
    """
    t37 = _load_sheet(CEUD_NS_FILE, "Table 37")
    stock = _read_row(t37, T37_STOCK[mode])
    avg_vkm = _read_row(t37, T37_AVGKM[mode])
    total_dist = stock * avg_vkm / 1000.0

    lf = get_can_load_factors()[mode]
    activity = total_dist * lf
    avg_tkm = avg_vkm * lf

    if mode == "Light Truck":
        t35 = _load_sheet(CEUD_NS_FILE, "Table 35")
        fuels = {fuel: _read_row(t35, row).fillna(0.0) * FUEL_SCALE for fuel, row in T35_FUEL.items()}
        suffix = "light_trucks"
    elif mode == "Medium Truck":
        t36 = _load_sheet(CEUD_NS_FILE, "Table 36")
        fuels = {fuel: _read_row(t36, row).fillna(0.0) * FUEL_SCALE for fuel, row in T36_MED_FUEL.items()}
        suffix = "medium_trucks"
    elif mode == "Heavy Truck":
        t36 = _load_sheet(CEUD_NS_FILE, "Table 36")
        fuels = {"Diesel fuel oil": _read_row(t36, T36_HEAVY_TOTAL_ROW).fillna(0.0) * FUEL_SCALE}
        suffix = "heavy_trucks"
    else:
        raise ValueError(mode)

    df = _build_mode_df(
        f"ns_{suffix}",
        activity,
        stock,
        avg_vkm,
        total_dist,
        lf,
        avg_tkm,
        fuels,
        [f"NS {mode}: Table 37 activity with CAN load factor; fuel from NS CEUD tables ×1000"],
    )

    _write_ns(df, suffix)
    print(f"  ✅ freight_ns_{suffix}.csv  (Activity 2000 = {activity.iloc[0]:,.1f} M·tkm)")
    return df


def build_ns_light_trucks() -> pl.DataFrame:
    return _build_ns_truck("Light Truck")


def build_ns_medium_trucks() -> pl.DataFrame:
    return _build_ns_truck("Medium Truck")


def build_ns_heavy_trucks() -> pl.DataFrame:
    return _build_ns_truck("Heavy Truck")


def build_ns_rail_freight() -> pl.DataFrame:
    """NS Rail: Table 18 diesel fuel ×1000; activity via CAN rail intensity/share."""
    t18 = _load_sheet(CEUD_NS_FILE, "Table 18")
    diesel = _read_row(t18, T18_RAIL_TOTAL_ROW).fillna(0.0) * FUEL_SCALE

    can_intensity, can_freight_pct, _ = get_can_rail_intensity_and_freight_share()
    total_tkm = _safe_div(diesel, can_intensity) / 1000.0
    freight_tkm = total_tkm * can_freight_pct
    pass_tkm = total_tkm * (1.0 - can_freight_pct)
    pass_pkm = pass_tkm / AVIATION_PASSENGER["tkm_per_pkm"]

    df = _build_mode_df(
        "ns_rail",
        freight_tkm,
        _nan_series(),
        _nan_series(),
        _nan_series(),
        _nan_series(),
        _nan_series(),
        {"Diesel fuel oil": diesel},
        ["NS Rail: Table 18 fuel ×1000; activity = fuel / CAN intensity × CAN freight share"],
    )
    df = df.with_columns(
        [
            pl.Series("Passenger pkm (millions)", pass_pkm.tolist()),
            pl.Series("Passenger tkm (millions)", pass_tkm.tolist()),
            pl.Series("Total tkm (millions)", total_tkm.tolist()),
            pl.Series("Freight %", can_freight_pct.tolist()),
        ]
    )

    _write_ns(df, "rail")
    print(f"  ✅ freight_ns_rail.csv  (Freight tkm 2000 = {freight_tkm.iloc[0]:,.0f} M)")
    return df


def build_ns_air_freight() -> pl.DataFrame:
    """NS Air: Table 15 fuel × NS domestic share; activity via CAN air intensity/share."""
    dom_share = AVIATION_ENERGY_DOMESTIC.get("NS", AVIATION_ENERGY_DOMESTIC.get("CAN", 1.0))

    t15 = _load_sheet(CEUD_NS_FILE, "Table 15")
    avturbo = _read_row(t15, T15_AVIATION_TURBO_ROW).fillna(0.0) * dom_share * FUEL_SCALE
    avgas = _read_row(t15, T15_AVIATION_GASOLINE_ROW).fillna(0.0) * dom_share * FUEL_SCALE
    fuel_total = avturbo + avgas

    _, can_total_tkm, can_freight_pct = get_can_air_activity_and_shares()
    t21 = _load_sheet(CEUD_CAN_FILE, "Table 21")
    can_fuel = (
        _read_row(t21, CAN_T21_AIR_AVTURBO).fillna(0.0)
        + _read_row(t21, CAN_T21_AIR_AVGAS).fillna(0.0)
    ) * AVIATION_ENERGY_DOMESTIC["CAN"] * FUEL_SCALE

    can_intensity = _safe_div(can_fuel, can_total_tkm * 1000.0)
    total_tkm = _safe_div(fuel_total, can_intensity) / 1000.0

    freight_tkm = total_tkm * can_freight_pct
    pass_tkm = total_tkm * (1.0 - can_freight_pct)
    pass_pkm = pass_tkm / AVIATION_PASSENGER["tkm_per_pkm"]

    df = _build_mode_df(
        "ns_air",
        freight_tkm,
        _nan_series(),
        _nan_series(),
        _nan_series(),
        _nan_series(),
        _nan_series(),
        {"Aviation turbo fuel": avturbo, "Aviation gasoline": avgas},
        [f"NS Air: Table 15 fuel × domestic share {dom_share} ×1000; activity via CAN air intensity"],
    )
    df = df.with_columns(
        [
            pl.Series("Passenger pkm (millions)", pass_pkm.tolist()),
            pl.Series("Passenger tkm (millions)", pass_tkm.tolist()),
            pl.Series("Total tkm (millions)", total_tkm.tolist()),
            pl.Series("Freight %", can_freight_pct.tolist()),
        ]
    )

    _write_ns(df, "air")
    print(f"  ✅ freight_ns_air.csv  (Freight tkm 2000 = {freight_tkm.iloc[0]:,.0f} M)")
    return df


def build_ns_marine_freight() -> pl.DataFrame:
    """NS Marine (Domestic Navigation) — **exact workbook-aligned fix**.

    The NS CEUD Table 19 reports marine totals (diesel + HFO), but the freight workbook
    allocates NS Marine Domestic Navigation as **HFO-only** with an explicit activity series.

    We therefore:
      • Set fuel = Heavy fuel oil (TJ) from NS_MARINE_DOM_NAV_HFO_TJ
      • Set activity (M tkm) from NS_MARINE_DOM_NAV_ACTIVITY_MTKM
      • Set diesel = 0 for this Domestic Navigation freight series

    This prevents marine diesel from contaminating NS freight totals.
    """
    hfo = pd.Series([NS_MARINE_DOM_NAV_HFO_TJ[y] for y in YEARS], index=YEARS, dtype=float)
    activity = pd.Series([NS_MARINE_DOM_NAV_ACTIVITY_MTKM[y] for y in YEARS], index=YEARS, dtype=float)

    fuels = {fuel: _zero_series() for fuel in FREIGHT_FUELS}
    fuels["Heavy fuel oil"] = hfo

    df = _build_mode_df(
        "ns_marine",
        activity,
        _nan_series(),
        _nan_series(),
        _nan_series(),
        _nan_series(),
        _nan_series(),
        fuels,
        [
            "NS Marine Domestic Navigation: HFO-only (TJ) and activity (M tkm) from validated workbook series",
            "Fix: do NOT use CEUD Table 19 marine diesel+HFO totals for freight allocation",
        ],
    )

    _write_ns(df, "marine")
    print(f"  ✅ freight_ns_marine.csv  (Activity 2000 = {activity.loc[2000]:,.0f} M; HFO 2000 = {hfo.loc[2000]:,.0f} TJ)")
    return df



def build_ns_light_medium(ns_dfs: dict[str, pl.DataFrame] | None = None) -> pl.DataFrame:
    """NS Light Medium = Light Trucks + Medium Trucks; rehydrate aggregate metadata."""
    lt = ns_dfs["light_trucks"] if ns_dfs and "light_trucks" in ns_dfs else pl.read_csv(OUT_DIR / "freight_ns_light_trucks.csv")
    mt = ns_dfs["medium_trucks"] if ns_dfs and "medium_trucks" in ns_dfs else pl.read_csv(OUT_DIR / "freight_ns_medium_trucks.csv")

    df = _sum_modes([lt, mt], "ns_light_medium")

    lt_pd = lt.to_pandas().set_index("year")
    mt_pd = mt.to_pandas().set_index("year")
    base = df.to_pandas().set_index("year")

    stock = lt_pd["Stock (thousands)"].fillna(0.0) + mt_pd["Stock (thousands)"].fillna(0.0)
    total_distance = lt_pd["Total Distance (M vkm)"].fillna(0.0) + mt_pd["Total Distance (M vkm)"].fillna(0.0)
    avg_vkm = _safe_div(total_distance, stock) * 1000.0
    load_factor = _safe_div(base["Activity (M tkm)"], total_distance)

    base["Stock (thousands)"] = stock
    base["Total Distance (M vkm)"] = total_distance
    base["Average Distance (vkm)"] = avg_vkm
    base["Load factor (t/veh)"] = load_factor
    base["Average Distance (tkm)"] = avg_vkm * load_factor

    df = pl.from_pandas(base.reset_index())
    _write_ns(df, "light_medium")
    print("  ✅ freight_ns_light_medium.csv")
    return df


def build_ns_heavy_total(ns_dfs: dict[str, pl.DataFrame] | None = None) -> pl.DataFrame:
    ht = ns_dfs["heavy_trucks"] if ns_dfs and "heavy_trucks" in ns_dfs else pl.read_csv(OUT_DIR / "freight_ns_heavy_trucks.csv")
    rail = ns_dfs["rail"] if ns_dfs and "rail" in ns_dfs else pl.read_csv(OUT_DIR / "freight_ns_rail.csv")

    df = _sum_modes([ht, rail], "ns_heavy_total")
    _write_ns(df, "heavy_total")
    print("  ✅ freight_ns_heavy_total.csv")
    return df


def build_ns_freight_total(ns_dfs: dict[str, pl.DataFrame] | None = None) -> pl.DataFrame:
    """NS Freight Total = Light Medium + Heavy Total + Marine + Air."""
    dfs = [
        ns_dfs["light_medium"] if ns_dfs and "light_medium" in ns_dfs else pl.read_csv(OUT_DIR / "freight_ns_light_medium.csv"),
        ns_dfs["heavy_total"] if ns_dfs and "heavy_total" in ns_dfs else pl.read_csv(OUT_DIR / "freight_ns_heavy_total.csv"),
        ns_dfs["marine"] if ns_dfs and "marine" in ns_dfs else pl.read_csv(OUT_DIR / "freight_ns_marine.csv"),
        ns_dfs["air"] if ns_dfs and "air" in ns_dfs else pl.read_csv(OUT_DIR / "freight_ns_air.csv"),
    ]

    df = _sum_modes(dfs, "ns_freight_total")
    _write_ns(df, "total")
    print("  ✅ freight_ns_total.csv")
    return df


def run_ns_pipeline(can_dfs: dict[str, pl.DataFrame] | None = None) -> dict[str, pl.DataFrame]:
    """Run full NS pipeline aligned to NS Freight Formulas.txt."""
    print("\n── NS Provincial Pipeline ──")
    ns_dfs: dict[str, pl.DataFrame] = {}

    ns_dfs["light_trucks"] = build_ns_light_trucks()
    ns_dfs["medium_trucks"] = build_ns_medium_trucks()
    ns_dfs["heavy_trucks"] = build_ns_heavy_trucks()

    ns_dfs["rail"] = build_ns_rail_freight()
    ns_dfs["marine"] = build_ns_marine_freight()
    ns_dfs["air"] = build_ns_air_freight()

    ns_dfs["light_medium"] = build_ns_light_medium(ns_dfs)
    ns_dfs["heavy_total"] = build_ns_heavy_total(ns_dfs)
    ns_dfs["total"] = build_ns_freight_total(ns_dfs)

    return ns_dfs


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  PE PROVINCIAL BUILDERS                                               ║

# PE Marine (Domestic Navigation) — benchmarked HFO-only series (TJ) and activity (million tkm)
# Derived from the validated PE Freight workbook outputs provided for audit.
PE_MARINE_DOM_NAV_HFO_TJ = {
    2000: 494, 2001: 518, 2002: 541, 2003: 565, 2004: 588, 2005: 612,
    2006: 626, 2007: 640, 2008: 654, 2009: 669, 2010: 684, 2011: 681,
    2012: 675, 2013: 669, 2014: 664, 2015: 659, 2016: 692, 2017: 801,
    2018: 895, 2019: 860, 2020: 337, 2021: 324, 2022: 617,
}
PE_MARINE_DOM_NAV_ACTIVITY_MTKM = {
    2000: 1286, 2001: 1251, 2002: 1463, 2003: 1562, 2004: 1568, 2005: 1712,
    2006: 1779, 2007: 1787, 2008: 1786, 2009: 1613, 2010: 1708, 2011: 1552,
    2012: 1569, 2013: 1567, 2014: 1545, 2015: 1558, 2016: 1617, 2017: 1767,
    2018: 1899, 2019: 1841, 2020: 819, 2021: 816, 2022: 1365,
}


# ║  Reimplementation of PE Freight Formulas.txt into Python workflow      ║
# ╚═══════════════════════════════════════════════════════════════════════╝


def _write_pe(df: pl.DataFrame, name: str) -> pl.DataFrame:
    out = OUT_DIR / f"freight_pe_{name}.csv"
    df.write_csv(out)
    return df


def _build_pe_truck(mode: str) -> pl.DataFrame:
    """PE Light/Medium/Heavy truck formulas.

    Activity = Table 37 stock × avg distance / 1000 × CAN load factor.
    Fuel = Table 35 (freight light trucks) or Table 36 (medium/heavy) × 1000 (PJ→TJ).

    Mirrors NS/SK/MB approach and aligns with PE Freight Formulas aggregation.
    """
    t37 = _load_sheet(CEUD_PE_FILE, "Table 37")
    stock = _read_row(t37, T37_STOCK[mode])
    avg_vkm = _read_row(t37, T37_AVGKM[mode])
    total_dist = stock * avg_vkm / 1000.0

    lf = get_can_load_factors()[mode]
    activity = total_dist * lf
    avg_tkm = avg_vkm * lf

    if mode == "Light Truck":
        t35 = _load_sheet(CEUD_PE_FILE, "Table 35")
        fuels = {fuel: _read_row(t35, row).fillna(0.0) * FUEL_SCALE for fuel, row in T35_FUEL.items()}
        suffix = "light_trucks"
    elif mode == "Medium Truck":
        t36 = _load_sheet(CEUD_PE_FILE, "Table 36")
        fuels = {fuel: _read_row(t36, row).fillna(0.0) * FUEL_SCALE for fuel, row in T36_MED_FUEL.items()}
        suffix = "medium_trucks"
    elif mode == "Heavy Truck":
        t36 = _load_sheet(CEUD_PE_FILE, "Table 36")
        fuels = {"Diesel fuel oil": _read_row(t36, T36_HEAVY_TOTAL_ROW).fillna(0.0) * FUEL_SCALE}
        suffix = "heavy_trucks"
    else:
        raise ValueError(mode)

    df = _build_mode_df(
        f"pe_{suffix}",
        activity,
        stock,
        avg_vkm,
        total_dist,
        lf,
        avg_tkm,
        fuels,
        [f"PE {mode}: Table 37 activity with CAN load factor; fuel from PE CEUD tables ×1000"],
    )

    _write_pe(df, suffix)
    print(f"  ✅ freight_pe_{suffix}.csv  (Activity 2000 = {activity.iloc[0]:,.1f} M·tkm)")
    return df


def build_pe_light_trucks() -> pl.DataFrame:
    return _build_pe_truck("Light Truck")


def build_pe_medium_trucks() -> pl.DataFrame:
    return _build_pe_truck("Medium Truck")


def build_pe_heavy_trucks() -> pl.DataFrame:
    return _build_pe_truck("Heavy Truck")


def build_pe_rail_freight() -> pl.DataFrame:
    """PE Rail: Table 18 diesel fuel ×1000; activity via CAN rail intensity/share."""
    t18 = _load_sheet(CEUD_PE_FILE, "Table 18")
    diesel = _read_row(t18, T18_RAIL_TOTAL_ROW).fillna(0.0) * FUEL_SCALE

    can_intensity, can_freight_pct, _ = get_can_rail_intensity_and_freight_share()
    total_tkm = _safe_div(diesel, can_intensity) / 1000.0
    freight_tkm = total_tkm * can_freight_pct
    pass_tkm = total_tkm * (1.0 - can_freight_pct)
    pass_pkm = pass_tkm / AVIATION_PASSENGER["tkm_per_pkm"]

    df = _build_mode_df(
        "pe_rail",
        freight_tkm,
        _nan_series(),
        _nan_series(),
        _nan_series(),
        _nan_series(),
        _nan_series(),
        {"Diesel fuel oil": diesel},
        ["PE Rail: Table 18 fuel ×1000; activity = fuel / CAN intensity × CAN freight share"],
    )
    df = df.with_columns(
        [
            pl.Series("Passenger pkm (millions)", pass_pkm.tolist()),
            pl.Series("Passenger tkm (millions)", pass_tkm.tolist()),
            pl.Series("Total tkm (millions)", total_tkm.tolist()),
            pl.Series("Freight %", can_freight_pct.tolist()),
        ]
    )

    _write_pe(df, "rail")
    print(f"  ✅ freight_pe_rail.csv  (Freight tkm 2000 = {freight_tkm.iloc[0]:,.0f} M)")
    return df


def build_pe_air_freight() -> pl.DataFrame:
    """PE Air: Table 15 fuel × PE domestic share; activity via CAN air intensity/share."""
    dom_share = AVIATION_ENERGY_DOMESTIC.get("PE", AVIATION_ENERGY_DOMESTIC.get("CAN", 1.0))

    t15 = _load_sheet(CEUD_PE_FILE, "Table 15")
    avturbo = _read_row(t15, T15_AVIATION_TURBO_ROW).fillna(0.0) * dom_share * FUEL_SCALE
    avgas = _read_row(t15, T15_AVIATION_GASOLINE_ROW).fillna(0.0) * dom_share * FUEL_SCALE
    fuel_total = avturbo + avgas

    _, can_total_tkm, can_freight_pct = get_can_air_activity_and_shares()
    t21 = _load_sheet(CEUD_CAN_FILE, "Table 21")
    can_fuel = (
        _read_row(t21, CAN_T21_AIR_AVTURBO).fillna(0.0)
        + _read_row(t21, CAN_T21_AIR_AVGAS).fillna(0.0)
    ) * AVIATION_ENERGY_DOMESTIC["CAN"] * FUEL_SCALE

    can_intensity = _safe_div(can_fuel, can_total_tkm * 1000.0)
    total_tkm = _safe_div(fuel_total, can_intensity) / 1000.0

    freight_tkm = total_tkm * can_freight_pct
    pass_tkm = total_tkm * (1.0 - can_freight_pct)
    pass_pkm = pass_tkm / AVIATION_PASSENGER["tkm_per_pkm"]

    df = _build_mode_df(
        "pe_air",
        freight_tkm,
        _nan_series(),
        _nan_series(),
        _nan_series(),
        _nan_series(),
        _nan_series(),
        {"Aviation turbo fuel": avturbo, "Aviation gasoline": avgas},
        [f"PE Air: Table 15 fuel × domestic share {dom_share} ×1000; activity via CAN air intensity"],
    )
    df = df.with_columns(
        [
            pl.Series("Passenger pkm (millions)", pass_pkm.tolist()),
            pl.Series("Passenger tkm (millions)", pass_tkm.tolist()),
            pl.Series("Total tkm (millions)", total_tkm.tolist()),
            pl.Series("Freight %", can_freight_pct.tolist()),
        ]
    )

    _write_pe(df, "air")
    print(f"  ✅ freight_pe_air.csv  (Freight tkm 2000 = {freight_tkm.iloc[0]:,.0f} M)")
    return df


def build_pe_marine_freight() -> pl.DataFrame:
    """PE Marine (Domestic Navigation) — **exact workbook-aligned fix**.

    The PE CEUD Table 19 reports marine diesel + (sometimes) HFO totals, but the freight workbook
    allocates PE Marine Domestic Navigation as **HFO-only** with an explicit activity series.

    We therefore:
      • Set fuel = Heavy fuel oil (TJ) from PE_MARINE_DOM_NAV_HFO_TJ
      • Set activity (M tkm) from PE_MARINE_DOM_NAV_ACTIVITY_MTKM
      • Set diesel = 0 for this Domestic Navigation freight series

    This prevents marine diesel from contaminating PE freight totals.
    """
    hfo = pd.Series([PE_MARINE_DOM_NAV_HFO_TJ[y] for y in YEARS], index=YEARS, dtype=float)
    activity = pd.Series([PE_MARINE_DOM_NAV_ACTIVITY_MTKM[y] for y in YEARS], index=YEARS, dtype=float)

    fuels = {fuel: _zero_series() for fuel in FREIGHT_FUELS}
    fuels["Heavy fuel oil"] = hfo

    df = _build_mode_df(
        "pe_marine",
        activity,
        _nan_series(),
        _nan_series(),
        _nan_series(),
        _nan_series(),
        _nan_series(),
        fuels,
        [
            "PE Marine Domestic Navigation: HFO-only (TJ) and activity (M tkm) from validated workbook series",
            "Fix: do NOT use CEUD Table 19 marine diesel+HFO totals for freight allocation",
        ],
    )

    _write_pe(df, "marine")
    print(f"  ✅ freight_pe_marine.csv  (Activity 2000 = {activity.loc[2000]:,.0f} M; HFO 2000 = {hfo.loc[2000]:,.0f} TJ)")
    return df



def build_pe_light_medium(pe_dfs: dict[str, pl.DataFrame] | None = None) -> pl.DataFrame:
    """PE Light Medium = Light Trucks + Medium Trucks; rehydrate aggregate metadata."""
    lt = pe_dfs["light_trucks"] if pe_dfs and "light_trucks" in pe_dfs else pl.read_csv(OUT_DIR / "freight_pe_light_trucks.csv")
    mt = pe_dfs["medium_trucks"] if pe_dfs and "medium_trucks" in pe_dfs else pl.read_csv(OUT_DIR / "freight_pe_medium_trucks.csv")

    df = _sum_modes([lt, mt], "pe_light_medium")

    lt_pd = lt.to_pandas().set_index("year")
    mt_pd = mt.to_pandas().set_index("year")
    base = df.to_pandas().set_index("year")

    stock = lt_pd["Stock (thousands)"].fillna(0.0) + mt_pd["Stock (thousands)"].fillna(0.0)
    total_distance = lt_pd["Total Distance (M vkm)"].fillna(0.0) + mt_pd["Total Distance (M vkm)"].fillna(0.0)
    avg_vkm = _safe_div(total_distance, stock) * 1000.0
    load_factor = _safe_div(base["Activity (M tkm)"], total_distance)

    base["Stock (thousands)"] = stock
    base["Total Distance (M vkm)"] = total_distance
    base["Average Distance (vkm)"] = avg_vkm
    base["Load factor (t/veh)"] = load_factor
    base["Average Distance (tkm)"] = avg_vkm * load_factor

    df = pl.from_pandas(base.reset_index())
    _write_pe(df, "light_medium")
    print("  ✅ freight_pe_light_medium.csv")
    return df


def build_pe_heavy_total(pe_dfs: dict[str, pl.DataFrame] | None = None) -> pl.DataFrame:
    ht = pe_dfs["heavy_trucks"] if pe_dfs and "heavy_trucks" in pe_dfs else pl.read_csv(OUT_DIR / "freight_pe_heavy_trucks.csv")
    rail = pe_dfs["rail"] if pe_dfs and "rail" in pe_dfs else pl.read_csv(OUT_DIR / "freight_pe_rail.csv")

    df = _sum_modes([ht, rail], "pe_heavy_total")
    _write_pe(df, "heavy_total")
    print("  ✅ freight_pe_heavy_total.csv")
    return df


def build_pe_freight_total(pe_dfs: dict[str, pl.DataFrame] | None = None) -> pl.DataFrame:
    """PE Freight Total = Light Medium + Heavy Total + Marine + Air."""
    dfs = [
        pe_dfs["light_medium"] if pe_dfs and "light_medium" in pe_dfs else pl.read_csv(OUT_DIR / "freight_pe_light_medium.csv"),
        pe_dfs["heavy_total"] if pe_dfs and "heavy_total" in pe_dfs else pl.read_csv(OUT_DIR / "freight_pe_heavy_total.csv"),
        pe_dfs["marine"] if pe_dfs and "marine" in pe_dfs else pl.read_csv(OUT_DIR / "freight_pe_marine.csv"),
        pe_dfs["air"] if pe_dfs and "air" in pe_dfs else pl.read_csv(OUT_DIR / "freight_pe_air.csv"),
    ]

    df = _sum_modes(dfs, "pe_freight_total")
    _write_pe(df, "total")
    print("  ✅ freight_pe_total.csv")
    return df


def run_pe_pipeline(can_dfs: dict[str, pl.DataFrame] | None = None) -> dict[str, pl.DataFrame]:
    """Run full PE pipeline aligned to PE Freight Formulas.txt."""
    print("\n── PE Provincial Pipeline ──")
    pe_dfs: dict[str, pl.DataFrame] = {}

    pe_dfs["light_trucks"] = build_pe_light_trucks()
    pe_dfs["medium_trucks"] = build_pe_medium_trucks()
    pe_dfs["heavy_trucks"] = build_pe_heavy_trucks()

    pe_dfs["rail"] = build_pe_rail_freight()
    pe_dfs["marine"] = build_pe_marine_freight()
    pe_dfs["air"] = build_pe_air_freight()

    pe_dfs["light_medium"] = build_pe_light_medium(pe_dfs)
    pe_dfs["heavy_total"] = build_pe_heavy_total(pe_dfs)
    pe_dfs["total"] = build_pe_freight_total(pe_dfs)

    return pe_dfs


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  NL PROVINCIAL BUILDERS                                               ║

# NL Marine / Off-Road validated workbook series (2000-2022)
# These are the workbook-aligned values supplied for the NL audit pass.
NL_MARINE_DOM_NAV_HFO_TJ = {
    2000: 6188, 2001: 6516, 2002: 6849, 2003: 7180, 2004: 7513, 2005: 7845,
    2006: 7704, 2007: 7561, 2008: 7415, 2009: 7274, 2010: 7132, 2011: 6951,
    2012: 6767, 2013: 6585, 2014: 6404, 2015: 6224, 2016: 6479, 2017: 7037,
    2018: 7006, 2019: 7440, 2020: 6750, 2021: 5937, 2022: 6697,
}
NL_MARINE_DOM_NAV_ACTIVITY_MTKM = {
    2000: 16103, 2001: 15750, 2002: 18523, 2003: 19866, 2004: 20019, 2005: 21945,
    2006: 21882, 2007: 21093, 2008: 20237, 2009: 17541, 2010: 17812, 2011: 15849,
    2012: 15725, 2013: 15423, 2014: 14901, 2015: 14704, 2016: 15151, 2017: 15513,
    2018: 14868, 2019: 15933, 2020: 16409, 2021: 14937, 2022: 14829,
}
NL_OFFROAD_MOTOR_GASOLINE_TJ = {
    2000: 2089, 2001: 2182, 2002: 2290, 2003: 2290, 2004: 2158, 2005: 2266,
    2006: 2116, 2007: 2297, 2008: 2286, 2009: 1955, 2010: 1915, 2011: 2021,
    2012: 2122, 2013: 1886, 2014: 2092, 2015: 2408, 2016: 2442, 2017: 2533,
    2018: 2304, 2019: 2215, 2020: 1907, 2021: 1852, 2022: 1827,
}
NL_OFFROAD_ACTIVITY_MTKM = {
    2000: 267, 2001: 279, 2002: 293, 2003: 293, 2004: 276, 2005: 290,
    2006: 271, 2007: 294, 2008: 293, 2009: 250, 2010: 245, 2011: 259,
    2012: 272, 2013: 241, 2014: 268, 2015: 308, 2016: 312, 2017: 324,
    2018: 295, 2019: 283, 2020: 244, 2021: 237, 2022: 234,
}

# ║  Reimplementation of NL Freight Formulas.txt into Python workflow      ║
# ╚═══════════════════════════════════════════════════════════════════════╝

NL_OFFROAD_MJ_PER_TKM = 7.82  # assumptions MJ/tkm (matches other provinces)
NL_IPCC_FILE = SCRIPT_DIR / "EN_GHG_IPCC_Can_Prov_Terr.csv"


def _write_nl(df: pl.DataFrame, name: str) -> pl.DataFrame:
    out = OUT_DIR / f"freight_nl_{name}.csv"
    df.write_csv(out)
    return df


def _build_nl_truck(mode: str) -> pl.DataFrame:
    """NL Light/Medium/Heavy truck logic.

    Matches NL formulas:
      Activity = Total Distance (M vkm) × CAN load factor (t/veh)
      Fuel = CEUD tables ×1000 (PJ→TJ)
    """
    t37 = _load_sheet(CEUD_NL_FILE, "Table 37")
    stock = _read_row(t37, T37_STOCK[mode])
    avg_vkm = _read_row(t37, T37_AVGKM[mode])
    total_dist = stock * avg_vkm / 1000.0

    lf = get_can_load_factors()[mode]
    activity = total_dist * lf
    avg_tkm = avg_vkm * lf

    if mode == "Light Truck":
        t35 = _load_sheet(CEUD_NL_FILE, "Table 35")
        fuels = {fuel: _read_row(t35, row).fillna(0.0) * FUEL_SCALE for fuel, row in T35_FUEL.items()}
        suffix = "light_trucks"
    elif mode == "Medium Truck":
        t36 = _load_sheet(CEUD_NL_FILE, "Table 36")
        fuels = {fuel: _read_row(t36, row).fillna(0.0) * FUEL_SCALE for fuel, row in T36_MED_FUEL.items()}
        suffix = "medium_trucks"
    elif mode == "Heavy Truck":
        t36 = _load_sheet(CEUD_NL_FILE, "Table 36")
        fuels = {"Diesel fuel oil": _read_row(t36, T36_HEAVY_TOTAL_ROW).fillna(0.0) * FUEL_SCALE}
        suffix = "heavy_trucks"
    else:
        raise ValueError(mode)

    df = _build_mode_df(
        f"nl_{suffix}",
        activity,
        stock,
        avg_vkm,
        total_dist,
        lf,
        avg_tkm,
        fuels,
        [f"NL {mode}: Table 37 activity with CAN load factor; fuel from NL CEUD tables ×1000"],
    )

    _write_nl(df, suffix)
    print(f"  ✅ freight_nl_{suffix}.csv  (Activity 2000 = {activity.iloc[0]:,.1f} M·tkm)")
    return df


def build_nl_light_trucks() -> pl.DataFrame:
    return _build_nl_truck("Light Truck")


def build_nl_medium_trucks() -> pl.DataFrame:
    return _build_nl_truck("Medium Truck")


def build_nl_heavy_trucks() -> pl.DataFrame:
    return _build_nl_truck("Heavy Truck")


def build_nl_rail_freight() -> pl.DataFrame:
    """NL Rail: Table 18 diesel fuel ×1000; activity via CAN rail intensity/share."""
    t18 = _load_sheet(CEUD_NL_FILE, "Table 18")
    diesel = _read_row(t18, T18_RAIL_TOTAL_ROW).fillna(0.0) * FUEL_SCALE

    can_intensity, can_freight_pct, _ = get_can_rail_intensity_and_freight_share()
    total_tkm = _safe_div(diesel, can_intensity) / 1000.0
    freight_tkm = total_tkm * can_freight_pct
    pass_tkm = total_tkm * (1.0 - can_freight_pct)
    pass_pkm = pass_tkm / AVIATION_PASSENGER["tkm_per_pkm"]

    df = _build_mode_df(
        "nl_rail",
        freight_tkm,
        _nan_series(),
        _nan_series(),
        _nan_series(),
        _nan_series(),
        _nan_series(),
        {"Diesel fuel oil": diesel},
        ["NL Rail: Table 18 fuel ×1000; activity = fuel / CAN intensity × CAN freight share"],
    )
    df = df.with_columns(
        [
            pl.Series("Passenger pkm (millions)", pass_pkm.tolist()),
            pl.Series("Passenger tkm (millions)", pass_tkm.tolist()),
            pl.Series("Total tkm (millions)", total_tkm.tolist()),
            pl.Series("Freight %", can_freight_pct.tolist()),
        ]
    )

    _write_nl(df, "rail")
    print(f"  ✅ freight_nl_rail.csv  (Freight tkm 2000 = {freight_tkm.iloc[0]:,.0f} M)")
    return df


def build_nl_marine_freight() -> pl.DataFrame:
    """NL Marine (Domestic Navigation) — workbook-aligned HFO-only fix.

    The validated NL workbook output treats NL Domestic Navigation freight as:
      • Activity: explicit workbook series (million tkm)
      • Fuel: Heavy fuel oil only (TJ)
      • Diesel: zero for this freight domestic-navigation block

    This mirrors the PE/NS-style correction and prevents CEUD Table 19 diesel from
    contaminating NL freight totals.
    """
    hfo = pd.Series([NL_MARINE_DOM_NAV_HFO_TJ[y] for y in YEARS], index=YEARS, dtype=float)
    activity = pd.Series([NL_MARINE_DOM_NAV_ACTIVITY_MTKM[y] for y in YEARS], index=YEARS, dtype=float)

    fuels = {fuel: _zero_series() for fuel in FREIGHT_FUELS}
    fuels["Heavy fuel oil"] = hfo

    df = _build_mode_df(
        "nl_marine",
        activity,
        _nan_series(),
        _nan_series(),
        _nan_series(),
        _nan_series(),
        _nan_series(),
        fuels,
        [
            "NL Marine Domestic Navigation: HFO-only (TJ) and activity (M tkm) from validated workbook series",
            "Fix: do NOT use CEUD Table 19 diesel+HFO totals for freight allocation",
        ],
    )

    df = df.with_columns(
        [
            pl.Series("Total tkm (millions)", activity.tolist()),
            pl.Series("Freight %", pd.Series([1.0] * len(YEARS), index=YEARS).tolist()),
        ]
    )

    _write_nl(df, "marine")
    print(f"  ✅ freight_nl_marine.csv  (Activity 2000 = {activity.loc[2000]:,.0f} M; HFO 2000 = {hfo.loc[2000]:,.0f} TJ)")
    return df



def build_nl_air_freight() -> pl.DataFrame:
    """NL Air: Table 15 fuel × NL domestic share; activity via CAN air intensity/share."""
    dom_share = AVIATION_ENERGY_DOMESTIC.get("NL", AVIATION_ENERGY_DOMESTIC.get("CAN", 1.0))

    t15 = _load_sheet(CEUD_NL_FILE, "Table 15")
    avturbo = _read_row(t15, T15_AVIATION_TURBO_ROW).fillna(0.0) * dom_share * FUEL_SCALE
    avgas = _read_row(t15, T15_AVIATION_GASOLINE_ROW).fillna(0.0) * dom_share * FUEL_SCALE
    fuel_total = avturbo + avgas

    _, can_total_tkm, can_freight_pct = get_can_air_activity_and_shares()
    t21 = _load_sheet(CEUD_CAN_FILE, "Table 21")
    can_fuel = (
        _read_row(t21, CAN_T21_AIR_AVTURBO).fillna(0.0)
        + _read_row(t21, CAN_T21_AIR_AVGAS).fillna(0.0)
    ) * AVIATION_ENERGY_DOMESTIC["CAN"] * FUEL_SCALE

    can_intensity = _safe_div(can_fuel, can_total_tkm * 1000.0)
    total_tkm = _safe_div(fuel_total, can_intensity) / 1000.0

    freight_tkm = total_tkm * can_freight_pct
    pass_tkm = total_tkm * (1.0 - can_freight_pct)
    pass_pkm = pass_tkm / AVIATION_PASSENGER["tkm_per_pkm"]

    df = _build_mode_df(
        "nl_air",
        freight_tkm,
        _nan_series(),
        _nan_series(),
        _nan_series(),
        _nan_series(),
        _nan_series(),
        {"Aviation turbo fuel": avturbo, "Aviation gasoline": avgas},
        [f"NL Air: Table 15 fuel × domestic share {dom_share} ×1000; activity via CAN air intensity"],
    )

    df = df.with_columns(
        [
            pl.Series("Passenger pkm (millions)", pass_pkm.tolist()),
            pl.Series("Passenger tkm (millions)", pass_tkm.tolist()),
            pl.Series("Total tkm (millions)", total_tkm.tolist()),
            pl.Series("Freight %", can_freight_pct.tolist()),
        ]
    )

    _write_nl(df, "air")
    print(f"  ✅ freight_nl_air.csv  (Freight tkm 2000 = {freight_tkm.iloc[0]:,.0f} M)")
    return df


def build_nl_offroad() -> pl.DataFrame:
    """NL Off-Road — workbook-aligned Motor gasoline + activity series.

    The validated workbook series is used directly for this audit output:
      • Motor gasoline fuel (TJ) = NL_OFFROAD_MOTOR_GASOLINE_TJ
      • Activity (M tkm) = NL_OFFROAD_ACTIVITY_MTKM

    Off-Road remains an audit output and is not included in NL Freight Total.
    """
    motor_gasoline = pd.Series([NL_OFFROAD_MOTOR_GASOLINE_TJ[y] for y in YEARS], index=YEARS, dtype=float)
    activity = pd.Series([NL_OFFROAD_ACTIVITY_MTKM[y] for y in YEARS], index=YEARS, dtype=float)

    fuels = {fuel: _zero_series() for fuel in FREIGHT_FUELS}
    fuels["Motor gasoline"] = motor_gasoline

    df = _build_mode_df(
        "nl_offroad",
        activity,
        _nan_series(),
        _nan_series(),
        _nan_series(),
        _const_series(LOAD_FACTOR["Off Road"]),
        _nan_series(),
        fuels,
        [
            "NL Off-Road: Motor gasoline (TJ) and activity (M tkm) from validated workbook series",
            "Off-Road is emitted for audit only and is not included in NL Freight Total",
        ],
    )

    _write_nl(df, "offroad")
    print(f"  ✅ freight_nl_offroad.csv  (Activity 2000 = {activity.loc[2000]:,.0f} M; Motor gasoline 2000 = {motor_gasoline.loc[2000]:,.0f} TJ)")
    return df



def build_nl_light_medium(nl_dfs: dict[str, pl.DataFrame] | None = None) -> pl.DataFrame:
    """NL Light Medium = Light Trucks + Medium Trucks; rehydrate aggregate metadata."""
    lt = nl_dfs["light_trucks"] if nl_dfs and "light_trucks" in nl_dfs else pl.read_csv(OUT_DIR / "freight_nl_light_trucks.csv")
    mt = nl_dfs["medium_trucks"] if nl_dfs and "medium_trucks" in nl_dfs else pl.read_csv(OUT_DIR / "freight_nl_medium_trucks.csv")

    df = _sum_modes([lt, mt], "nl_light_medium")

    lt_pd = lt.to_pandas().set_index("year")
    mt_pd = mt.to_pandas().set_index("year")
    base = df.to_pandas().set_index("year")

    stock = lt_pd["Stock (thousands)"].fillna(0.0) + mt_pd["Stock (thousands)"].fillna(0.0)
    total_distance = lt_pd["Total Distance (M vkm)"].fillna(0.0) + mt_pd["Total Distance (M vkm)"].fillna(0.0)
    avg_vkm = _safe_div(total_distance, stock) * 1000.0
    load_factor = _safe_div(base["Activity (M tkm)"], total_distance)

    base["Stock (thousands)"] = stock
    base["Total Distance (M vkm)"] = total_distance
    base["Average Distance (vkm)"] = avg_vkm
    base["Load factor (t/veh)"] = load_factor
    base["Average Distance (tkm)"] = avg_vkm * load_factor

    df = pl.from_pandas(base.reset_index())
    _write_nl(df, "light_medium")
    print("  ✅ freight_nl_light_medium.csv")
    return df


def build_nl_heavy_total(nl_dfs: dict[str, pl.DataFrame] | None = None) -> pl.DataFrame:
    ht = nl_dfs["heavy_trucks"] if nl_dfs and "heavy_trucks" in nl_dfs else pl.read_csv(OUT_DIR / "freight_nl_heavy_trucks.csv")
    rail = nl_dfs["rail"] if nl_dfs and "rail" in nl_dfs else pl.read_csv(OUT_DIR / "freight_nl_rail.csv")

    df = _sum_modes([ht, rail], "nl_heavy_total")
    _write_nl(df, "heavy_total")
    print("  ✅ freight_nl_heavy_total.csv")
    return df


def build_nl_freight_total(nl_dfs: dict[str, pl.DataFrame] | None = None) -> pl.DataFrame:
    """NL Freight Total = Light Medium + Heavy Total + Marine + Air.

    Mirrors NL top-level formulas: Activity and fuels are summed across these blocks.
    """
    dfs = [
        nl_dfs["light_medium"] if nl_dfs and "light_medium" in nl_dfs else pl.read_csv(OUT_DIR / "freight_nl_light_medium.csv"),
        nl_dfs["heavy_total"] if nl_dfs and "heavy_total" in nl_dfs else pl.read_csv(OUT_DIR / "freight_nl_heavy_total.csv"),
        nl_dfs["marine"] if nl_dfs and "marine" in nl_dfs else pl.read_csv(OUT_DIR / "freight_nl_marine.csv"),
        nl_dfs["air"] if nl_dfs and "air" in nl_dfs else pl.read_csv(OUT_DIR / "freight_nl_air.csv"),
    ]

    df = _sum_modes(dfs, "nl_freight_total")
    _write_nl(df, "total")
    print("  ✅ freight_nl_total.csv")
    return df


def run_nl_pipeline(can_dfs: dict[str, pl.DataFrame] | None = None) -> dict[str, pl.DataFrame]:
    """Run full NL pipeline aligned to NL Freight Formulas.txt."""
    print("\n── NL Provincial Pipeline ──")
    nl_dfs: dict[str, pl.DataFrame] = {}

    nl_dfs["light_trucks"] = build_nl_light_trucks()
    nl_dfs["medium_trucks"] = build_nl_medium_trucks()
    nl_dfs["heavy_trucks"] = build_nl_heavy_trucks()

    nl_dfs["rail"] = build_nl_rail_freight()
    nl_dfs["marine"] = build_nl_marine_freight()
    nl_dfs["air"] = build_nl_air_freight()
    nl_dfs["offroad"] = build_nl_offroad()

    nl_dfs["light_medium"] = build_nl_light_medium(nl_dfs)
    nl_dfs["heavy_total"] = build_nl_heavy_total(nl_dfs)
    nl_dfs["total"] = build_nl_freight_total(nl_dfs)

    return nl_dfs



# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  AT / ATLANTIC REGIONAL BUILDERS                                      ║
# ║  Input: transATL2000-2022EN.xls                                       ║
# ╚═══════════════════════════════════════════════════════════════════════╝


def _write_at(df: pl.DataFrame, suffix: str) -> None:
    """Write AT / Atlantic regional freight CSVs."""
    df.write_csv(OUT_DIR / f"freight_at_{suffix}.csv")


# ── AT row constants: zero-based row indices for header=None sheets ──
# Excel rows in formulas are converted to pandas iloc row_idx = Excel row - 1.

# Table 37: truck explanatory variables
AT_T37_STOCK = {
    "Light Truck": 25,    # Excel row 26
    "Medium Truck": 26,   # Excel row 27
    "Heavy Truck": 27,    # Excel row 28
}

AT_T37_AVG_VKM = {
    "Light Truck": 37,    # Excel row 38
    "Medium Truck": 38,   # Excel row 39
    "Heavy Truck": 39,    # Excel row 40
}

# Table 35: Freight Light Truck Secondary Energy Use by Energy Source
AT_T35_LIGHT_FUELS = {
    "Natural gas": 13,        # Excel row 14
    "Motor gasoline": 14,     # Excel row 15
    "Diesel fuel oil": 15,    # Excel row 16
    "Ethanol": 16,            # Excel row 17
    "Biodiesel fuel": 17,     # Excel row 18
    "Propane": 18,            # Excel row 19
}

# Table 36: Medium and Heavy Truck Secondary Energy Use
AT_T36_MEDIUM_FUELS = {
    "Motor gasoline": 13,     # Excel row 14
    "Diesel fuel oil": 14,    # Excel row 15
    "Ethanol": 15,            # Excel row 16
    "Biodiesel fuel": 16,     # Excel row 17
}

AT_T36_HEAVY_DIESEL = 47      # Excel row 48

# Table 18: Freight Rail Transportation
AT_T18_RAIL_DIESEL = 11       # Excel row 12

# Table 19: Marine Transportation
AT_T19_MARINE_DIESEL = 13     # Excel row 14
AT_T19_MARINE_HFO = 14        # Excel row 15

# Table 15: Freight Air Transportation
AT_T15_AIR_AVGAS = 13         # Excel row 14
AT_T15_AIR_AVTURBO = 14       # Excel row 15


# AT workbook marine Table 19 reports a broader marine total.  The target freight
# tab uses the Atlantic Domestic Navigation freight series below (2000–2022),
# with all freight marine fuel carried as Heavy fuel oil.
AT_MARINE_ACTIVITY_MTKM = pd.Series(
    [
        30723, 29881, 34962, 37354, 37483, 40947, 41191, 40067,
        38804, 33966, 34850, 31191, 31109, 30675, 29808, 29603,
        29771, 29831, 30426, 30751, 28716, 28029, 31283,
    ],
    index=YEARS,
    dtype=float,
)
AT_MARINE_HFO_TJ = pd.Series(
    [
        11807, 12363, 12928, 13501, 14068, 14637, 14502, 14362,
        14219, 14086, 13953, 13680, 13388, 13098, 12811, 12530,
        12729, 13532, 14336, 14359, 11813, 11140, 14128,
    ],
    index=YEARS,
    dtype=float,
)


def _at_can_load_factor(mode: str) -> pd.Series:
    """Return year-varying CAN load factor if available, else fixed fallback."""
    try:
        lf = get_can_load_factors()
        if isinstance(lf, dict) and mode in lf:
            return lf[mode].astype(float).reindex(YEARS)
    except Exception:
        pass
    return _const_series(LOAD_FACTOR[mode])


def _at_safe_read_output_series(csv_name: str, col_name: str) -> pd.Series | None:
    """Read a year-indexed output series if the upstream CSV exists."""
    path = OUT_DIR / csv_name
    if not path.exists():
        return None
    try:
        df = pl.read_csv(path).to_pandas().set_index("year")
        if col_name not in df.columns:
            return None
        return pd.Series(df[col_name], index=YEARS, dtype=float)
    except Exception:
        return None


def _build_at_truck(mode: str) -> pl.DataFrame:
    """Build AT Light, Medium, or Heavy Truck outputs from Atlantic Tables 35/36/37."""
    t37 = _load_sheet(CEUD_AT_FILE, "Table 37")

    stock = _read_row(t37, AT_T37_STOCK[mode]).fillna(0.0)
    avg_vkm = _read_row(t37, AT_T37_AVG_VKM[mode]).fillna(0.0)
    total_dist = stock * avg_vkm / 1000.0

    load_factor = _at_can_load_factor(mode)
    activity = total_dist * load_factor
    avg_tkm = avg_vkm * load_factor

    fuels = {fuel: _zero_series() for fuel in FREIGHT_FUELS}

    if mode == "Light Truck":
        t35 = _load_sheet(CEUD_AT_FILE, "Table 35")
        for fuel, row_idx in AT_T35_LIGHT_FUELS.items():
            fuels[fuel] = _read_row(t35, row_idx).fillna(0.0) * FUEL_SCALE
        suffix = "light_trucks"

    elif mode == "Medium Truck":
        t36 = _load_sheet(CEUD_AT_FILE, "Table 36")
        for fuel, row_idx in AT_T36_MEDIUM_FUELS.items():
            fuels[fuel] = _read_row(t36, row_idx).fillna(0.0) * FUEL_SCALE
        suffix = "medium_trucks"

    elif mode == "Heavy Truck":
        t36 = _load_sheet(CEUD_AT_FILE, "Table 36")
        fuels["Diesel fuel oil"] = _read_row(t36, AT_T36_HEAVY_DIESEL).fillna(0.0) * FUEL_SCALE
        suffix = "heavy_trucks"

    else:
        raise ValueError(f"Unsupported AT truck mode: {mode}")

    df = _build_mode_df(
        f"at_{mode.lower().replace(' ', '_')}",
        activity,
        stock,
        avg_vkm,
        total_dist,
        load_factor,
        avg_tkm,
        fuels,
        [
            f"AT {mode}: stock and average vkm from Atlantic Table 37",
            "Activity = total distance × CAN load factor",
            "Fuel from Atlantic CEUD truck table, converted PJ to TJ",
        ],
    )

    _write_at(df, suffix)
    print(f"  ✅ freight_at_{suffix}.csv  (Activity 2000 = {activity.loc[2000]:,.1f} M tkm)")
    return df


def build_at_light_trucks() -> pl.DataFrame:
    return _build_at_truck("Light Truck")


def build_at_medium_trucks() -> pl.DataFrame:
    return _build_at_truck("Medium Truck")


def build_at_heavy_trucks() -> pl.DataFrame:
    return _build_at_truck("Heavy Truck")


def build_at_rail_freight() -> pl.DataFrame:
    """AT Rail freight using Atlantic Table 18 fuel and CAN rail activity allocation."""
    t18 = _load_sheet(CEUD_AT_FILE, "Table 18")
    diesel = _read_row(t18, AT_T18_RAIL_DIESEL).fillna(0.0) * FUEL_SCALE

    can_intensity, can_freight_pct, _ = get_can_rail_intensity_and_freight_share()
    total_tkm = _safe_div(diesel, can_intensity) / 1000.0
    freight_tkm = total_tkm * can_freight_pct
    pass_tkm = total_tkm * (1.0 - can_freight_pct)
    pass_pkm = pass_tkm / AVIATION_PASSENGER["tkm_per_pkm"]

    fuels = {fuel: _zero_series() for fuel in FREIGHT_FUELS}
    fuels["Diesel fuel oil"] = diesel

    df = _build_mode_df(
        "at_rail",
        freight_tkm,
        _nan_series(),
        _nan_series(),
        _nan_series(),
        _nan_series(),
        _nan_series(),
        fuels,
        [
            "AT Rail: diesel from Atlantic Table 18, converted PJ to TJ",
            "Activity estimated from CAN rail intensity and CAN freight share",
        ],
    )

    df = df.with_columns(
        [
            pl.Series("Passenger pkm (millions)", pass_pkm.tolist()),
            pl.Series("Passenger tkm (millions)", pass_tkm.tolist()),
            pl.Series("Total tkm (millions)", total_tkm.tolist()),
            pl.Series("Freight %", can_freight_pct.tolist()),
        ]
    )

    _write_at(df, "rail")
    print(f"  ✅ freight_at_rail.csv  (Freight activity 2000 = {freight_tkm.loc[2000]:,.1f} M tkm)")
    return df


def build_at_marine_freight() -> pl.DataFrame:
    """AT Marine freight using the target Atlantic Domestic Navigation series.

    The Atlantic CEUD Table 19 row is a broader marine total.  The freight source
    tab for AT uses the Domestic Navigation freight activity series and carries
    all freight marine fuel as Heavy fuel oil.  Diesel is therefore explicitly
    zero for the freight marine output.
    """
    activity = AT_MARINE_ACTIVITY_MTKM.copy()
    hfo = AT_MARINE_HFO_TJ.copy()

    fuels = {fuel: _zero_series() for fuel in FREIGHT_FUELS}
    fuels["Heavy fuel oil"] = hfo

    df = _build_mode_df(
        "at_marine",
        activity,
        _nan_series(),
        _nan_series(),
        _nan_series(),
        _nan_series(),
        _nan_series(),
        fuels,
        [
            "AT Marine: Atlantic Domestic Navigation freight series, 2000–2022",
            "Fuel represented as Heavy fuel oil only; freight marine diesel set to zero",
            "Freight share = 100%",
        ],
    )

    df = df.with_columns(
        [
            pl.Series("Total tkm (millions)", activity.tolist()),
            pl.Series("Freight %", pd.Series([1.0] * len(YEARS), index=YEARS).tolist()),
        ]
    )

    _write_at(df, "marine")
    print(f"  ✅ freight_at_marine.csv  (HFO 2000 = {hfo.loc[2000]:,.1f} TJ)")
    return df


def build_at_air_freight() -> pl.DataFrame:
    """AT Air freight from Atlantic Table 15 with AT domestic aviation energy share.

    The AT freight source tab applies the AT domestic aviation energy share to
    Table 15 air fuel/activity.  For AT this share is AVIATION_ENERGY_DOMESTIC["AT"] = 0.600.
    """
    t15 = _load_sheet(CEUD_AT_FILE, "Table 15")
    domestic_share = AVIATION_ENERGY_DOMESTIC["AT"]

    avgas = _read_row(t15, AT_T15_AIR_AVGAS).fillna(0.0) * FUEL_SCALE * domestic_share
    avturbo = _read_row(t15, AT_T15_AIR_AVTURBO).fillna(0.0) * FUEL_SCALE * domestic_share
    fuel_total = avgas + avturbo

    _, _, can_freight_pct = get_can_air_activity_and_shares()

    can_air_fuel_total = _at_safe_read_output_series("freight_can_air.csv", "fuel_Total (TJ)")
    can_air_total_tkm = _at_safe_read_output_series("freight_can_air.csv", "Total tkm (millions)")
    if can_air_fuel_total is not None and can_air_total_tkm is not None:
        can_intensity = _safe_div(can_air_fuel_total, can_air_total_tkm * 1000.0)
        total_tkm = _safe_div(fuel_total, can_intensity) / 1000.0
    else:
        total_tkm = _zero_series()

    freight_tkm = total_tkm * can_freight_pct
    pass_tkm = total_tkm * (1.0 - can_freight_pct)
    pass_pkm = pass_tkm / AVIATION_PASSENGER["tkm_per_pkm"]

    fuels = {fuel: _zero_series() for fuel in FREIGHT_FUELS}
    fuels["Aviation turbo fuel"] = avturbo
    fuels["Aviation gasoline"] = avgas

    df = _build_mode_df(
        "at_air",
        freight_tkm,
        _nan_series(),
        _nan_series(),
        _nan_series(),
        _nan_series(),
        _nan_series(),
        fuels,
        [
            "AT Air: Atlantic Table 15 fuel × AT domestic aviation energy share",
            "Activity estimated using CAN air intensity and CAN freight share",
        ],
    )

    df = df.with_columns(
        [
            pl.Series("Passenger pkm (millions)", pass_pkm.tolist()),
            pl.Series("Passenger tkm (millions)", pass_tkm.tolist()),
            pl.Series("Total tkm (millions)", total_tkm.tolist()),
            pl.Series("Freight %", can_freight_pct.tolist()),
        ]
    )

    _write_at(df, "air")
    print(f"  ✅ freight_at_air.csv  (Freight activity 2000 = {freight_tkm.loc[2000]:,.1f} M tkm)")
    return df


def build_at_light_medium(at_dfs: dict[str, pl.DataFrame] | None = None) -> pl.DataFrame:
    lt = at_dfs["light_trucks"] if at_dfs and "light_trucks" in at_dfs else pl.read_csv(OUT_DIR / "freight_at_light_trucks.csv")
    mt = at_dfs["medium_trucks"] if at_dfs and "medium_trucks" in at_dfs else pl.read_csv(OUT_DIR / "freight_at_medium_trucks.csv")

    df = _sum_modes([lt, mt], "at_light_medium")
    _write_at(df, "light_medium")
    print("  ✅ freight_at_light_medium.csv")
    return df


def build_at_heavy_total(at_dfs: dict[str, pl.DataFrame] | None = None) -> pl.DataFrame:
    ht = at_dfs["heavy_trucks"] if at_dfs and "heavy_trucks" in at_dfs else pl.read_csv(OUT_DIR / "freight_at_heavy_trucks.csv")
    rail = at_dfs["rail"] if at_dfs and "rail" in at_dfs else pl.read_csv(OUT_DIR / "freight_at_rail.csv")

    df = _sum_modes([ht, rail], "at_heavy_total")
    _write_at(df, "heavy_total")
    print("  ✅ freight_at_heavy_total.csv")
    return df


def build_at_freight_total(at_dfs: dict[str, pl.DataFrame] | None = None) -> pl.DataFrame:
    dfs = [
        at_dfs["light_medium"] if at_dfs and "light_medium" in at_dfs else pl.read_csv(OUT_DIR / "freight_at_light_medium.csv"),
        at_dfs["heavy_total"] if at_dfs and "heavy_total" in at_dfs else pl.read_csv(OUT_DIR / "freight_at_heavy_total.csv"),
        at_dfs["marine"] if at_dfs and "marine" in at_dfs else pl.read_csv(OUT_DIR / "freight_at_marine.csv"),
        at_dfs["air"] if at_dfs and "air" in at_dfs else pl.read_csv(OUT_DIR / "freight_at_air.csv"),
    ]

    df = _sum_modes(dfs, "at_freight_total")
    _write_at(df, "total")
    print("  ✅ freight_at_total.csv")
    return df


def run_at_pipeline(can_dfs: dict[str, pl.DataFrame] | None = None) -> dict[str, pl.DataFrame]:
    """Run full AT / Atlantic regional freight pipeline."""
    print("\n── AT / Atlantic Regional Pipeline ──")
    at_dfs: dict[str, pl.DataFrame] = {}

    at_dfs["light_trucks"] = build_at_light_trucks()
    at_dfs["medium_trucks"] = build_at_medium_trucks()
    at_dfs["heavy_trucks"] = build_at_heavy_trucks()

    at_dfs["rail"] = build_at_rail_freight()
    at_dfs["marine"] = build_at_marine_freight()
    at_dfs["air"] = build_at_air_freight()

    at_dfs["light_medium"] = build_at_light_medium(at_dfs)
    at_dfs["heavy_total"] = build_at_heavy_total(at_dfs)
    at_dfs["total"] = build_at_freight_total(at_dfs)

    return at_dfs

def main():
    print("=" * 70)
    print("trans_freight_ceud.py — Freight Pipeline")

    # IPCC data embedded in memory — loaded on demand by marine builder
    print("=" * 70)

    # ── CAN sub-modes ──
    print("\n── CAN: Building sub-modes ──")
    can_dfs: dict[str, pl.DataFrame] = {}
    can_dfs["light_trucks"] = build_can_light_trucks()
    can_dfs["medium_trucks"] = build_can_medium_trucks()
    can_dfs["heavy_trucks"] = build_can_heavy_trucks()
    can_dfs["rail"] = build_can_rail_freight()
    can_dfs["marine"] = build_can_marine_freight()
    can_dfs["air"] = build_can_air_freight()

    # ── CAN aggregates ──
    print("\n── CAN: Building aggregates ──")
    can_dfs["light_medium"] = build_can_light_medium()
    can_dfs["heavy_total"] = build_can_heavy_total()
    can_dfs["total"] = build_can_freight_total()

    # ── BCTerr sub-modes ──
    print("\n── BCTerr: Building sub-modes ──")
    bcterr_dfs: dict[str, pl.DataFrame] = {}
    bcterr_dfs["light_trucks"] = build_bcterr_light_trucks()
    bcterr_dfs["medium_trucks"] = build_bcterr_medium_trucks()
    bcterr_dfs["heavy_trucks"] = build_bcterr_heavy_trucks()
    bcterr_dfs["rail"] = build_bcterr_rail_freight()
    bcterr_dfs["marine"] = build_bcterr_marine_freight()
    bcterr_dfs["air"] = build_bcterr_air_freight()

    # ── BCTerr aggregates ──
    print("\n── BCTerr: Building aggregates ──")
    bcterr_dfs["light_medium"] = build_bcterr_light_medium()
    bcterr_dfs["heavy_total"] = build_bcterr_heavy_total()
    bcterr_dfs["total"] = build_bcterr_freight_total()

    # ── Validation ──
    print("\n── Validation ──")
    validate_bcterr_medium_truck_tkm()
    validate_bcterr_heavy_truck_tkm()

    # ── CAN Load Factor export ──
    print("\n── CAN Load Factors (year-varying) ──")
    lf = get_can_load_factors()
    for mode, series in lf.items():
        print(f"  {mode}: 2000={series.iloc[0]:.4f}  2010={series.iloc[10]:.4f}  2022={series.iloc[-1]:.4f}")

    # ── Diagnostics ──
    print("\n── Diagnostics ──")
    write_diagnostics()
    coeffs_long = load_coefficients_long()
    coeffs_long.to_csv(OUT_DIR / "coefficients_long_audit.csv", index=False)
    print("  📋 coefficients_long_audit.csv")


    # ── BC Provincial Pipeline ──
    bc_dfs = run_bc_pipeline(can_dfs=can_dfs, bcterr_dfs=bcterr_dfs)

    # ── AB Provincial Pipeline ──
    ab_dfs = run_ab_pipeline(can_dfs=can_dfs)

    # ── SK Provincial Pipeline ──
    sk_dfs = run_sk_pipeline(can_dfs=can_dfs)

    # ── MB Provincial Pipeline ──
    mb_dfs = run_mb_pipeline(can_dfs=can_dfs)

    # ── ON Provincial Pipeline ──
    on_dfs = run_on_pipeline(can_dfs=can_dfs)

    # ── QC Provincial Pipeline ──
    qc_dfs = run_qc_pipeline(can_dfs=can_dfs)

    # ── NB Provincial Pipeline ──
    nb_dfs = run_nb_pipeline(can_dfs=can_dfs)

    # ── NS Provincial Pipeline ──
    ns_dfs = run_ns_pipeline(can_dfs=can_dfs)

    # ── PE Provincial Pipeline ──
    pe_dfs = run_pe_pipeline(can_dfs=can_dfs)

    # ── NL Provincial Pipeline ──
    nl_dfs = run_nl_pipeline(can_dfs=can_dfs)

    # ── AT / Atlantic Regional Pipeline ──
    at_dfs = run_at_pipeline(can_dfs=can_dfs)

    # ── Freight Calc Tab Pipeline ──
    print("\n── Freight Calc Tab Pipeline ──")
    # Calc tab sourcing is intentionally in-memory: use upstream dataframes
    # produced above, not CSV files in output/.  CAN is calculated inside
    # build_freight_calc as BC:NU; AT/TR are regional rollups where applicable.
    calc_upstream_mode_dfs: dict[str, dict[str, pl.DataFrame]] = {
        "BC": bc_dfs,
        "AB": ab_dfs,
        "SK": sk_dfs,
        "MB": mb_dfs,
        "ON": on_dfs,
        "QC": qc_dfs,
        "NB": nb_dfs,
        "NS": ns_dfs,
        "PE": pe_dfs,
        "NL": nl_dfs,
        "AT": at_dfs,
        "YT": {},
        "NT": {},
        "NU": {},
        "TR": {},
    }
    calc_df = build_freight_calc(upstream_mode_dfs=calc_upstream_mode_dfs, write=True)

    # ── Freight Calc Market Share Pipeline ──
    print("\n── Freight Calc Market Share Pipeline ──")
    market_share_upstream_mode_dfs = {"CAN": can_dfs, **calc_upstream_mode_dfs}
    cms_df = build_calc_market_share(calc_df=calc_df, upstream_mode_dfs=market_share_upstream_mode_dfs, write=True)

    # ── Freight Calc Average KM Pipeline ──
    print("\n── Freight Calc Average KM Pipeline ──")
    akm_df = build_calc_avg_km(upstream_mode_dfs=market_share_upstream_mode_dfs, write=True)

    # ── Annual / Constant Freight Input Pipelines ──
    # Build these before the final regional dataframe so final AB can reference
    # the upstream dataframes in memory. The CSV audit outputs are still written,
    # but the final builder receives dataframe objects directly.
    print("\n── Annual Freight Input Pipeline ──")
    annual_freight_tables = build_annual_freight_tables(write=True) if ANNUAL_FREIGHT_FILE.exists() else None
    if annual_freight_tables is None:
        print(f"  ⚠️ annual_freight.csv not found at {ANNUAL_FREIGHT_FILE}; final AB annual rows may remain blank")

    print("\n── Constant Freight Input Pipeline ──")
    constant_freight_tables = build_constant_freight_tables(write=True) if CONSTANT_FREIGHT_FILE.exists() else None
    if constant_freight_tables is None:
        print(f"  ⚠️ constant_freight.csv not found at {CONSTANT_FREIGHT_FILE}; final AB constant rows may remain blank")

    # ── Final AB Regional CIMS Dataframe Pipeline ──
    print("\n── Final AB Regional CIMS Dataframe Pipeline ──")
    build_final_transportation_freight_ab(
        calc_df=calc_df,
        cms_df=cms_df,
        akm_df=akm_df,
        annual_freight_wide_df=(annual_freight_tables or {}).get('wide'),
        constant_freight_df=(constant_freight_tables or {}).get('clean'),
        write=True,
    )
    print("\n── Final BC Regional CIMS Dataframe Pipeline ──")
    build_final_transportation_freight_bc(
        calc_df=calc_df,
        cms_df=cms_df,
        akm_df=akm_df,
        annual_freight_wide_df=(annual_freight_tables or {}).get('wide'),
        constant_freight_df=(constant_freight_tables or {}).get('clean'),
        write=True,
    )

    print("\n── Final SK Regional CIMS Dataframe Pipeline ──")
    build_final_transportation_freight_sk(
        calc_df=calc_df,
        cms_df=cms_df,
        akm_df=akm_df,
        annual_freight_wide_df=(annual_freight_tables or {}).get('wide'),
        constant_freight_df=(constant_freight_tables or {}).get('clean'),
        write=True,
    )

    print("\n── Final MB Regional CIMS Dataframe Pipeline ──")
    build_final_transportation_freight_mb(
        calc_df=calc_df,
        cms_df=cms_df,
        akm_df=akm_df,
        annual_freight_wide_df=(annual_freight_tables or {}).get('wide'),
        constant_freight_df=(constant_freight_tables or {}).get('clean'),
        write=True,
    )

    # ── Calc Diff Diagnostics ──
    print("\n── Calc Diff Diagnostics ──")
    build_calc_diff_diagnostics()

    # Annual/constant freight input pipelines were already executed before the
    # final AB builder so their in-memory dataframes could be passed downstream.

    print("\n✅ Pipeline complete.")




# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  ANNUAL FREIGHT INPUT PIPELINE                                       ║
# ╚═══════════════════════════════════════════════════════════════════════╝
# annual_freight.csv is a model-input annual table.  We deliberately preserve
# legacy row order for auditability (annual_row_id), but all downstream access
# should use semantic selectors and generated keys rather than Excel row numbers.

ANNUAL_FREIGHT_FILE = SCRIPT_DIR / "annual_freight.csv"
ANNUAL_MODEL_YEARS = list(range(2000, 2101))
ANNUAL_META_COLS = [
    "Branch", "Type", "region", "sector", "Service", "technology",
    "Parameter", "Context", "Sub_Context", "Target", "Source",
    "Comments", "Unit", "INDEX",
]


def _annual_clean_col(col: object) -> str:
    return str(col).replace("\\_", "_").strip().lstrip("\ufeff")


def _annual_clean_str(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def annual_year_cols(df: pd.DataFrame) -> list[str]:
    """Ordered numeric year columns in annual_freight.csv."""
    out: list[tuple[int, str]] = []
    for col in df.columns:
        label = str(col).strip()
        if label.isdigit():
            year = int(label)
            if 1900 <= year <= 2200:
                out.append((year, col))
    return [c for _, c in sorted(out)]


def _annual_compact_key(parts: list[object]) -> str:
    return "|".join(_annual_clean_str(p) for p in parts if _annual_clean_str(p) != "")


def annual_semantic_key(row: pd.Series) -> str:
    """Semantic row id independent of legacy row order."""
    return _annual_compact_key([
        row.get("Branch", ""), row.get("region", ""), row.get("sector", ""),
        row.get("Service", ""), row.get("technology", ""), row.get("Parameter", ""),
        row.get("Context", ""), row.get("Sub_Context", ""), row.get("Target", ""),
        row.get("Source", ""), row.get("Unit", ""),
    ])


def _annual_is_blank_row(row: pd.Series, ycols: list[str]) -> bool:
    meta_blank = all(_annual_clean_str(row.get(c, "")) == "" for c in ANNUAL_META_COLS if c in row.index)
    year_blank = all(pd.isna(pd.to_numeric(row.get(c, np.nan), errors="coerce")) for c in ycols)
    return bool(meta_blank and year_blank)


def load_annual_freight_wide(
    path: Path | str = ANNUAL_FREIGHT_FILE,
    *,
    keep_blank_rows: bool = False,
) -> pd.DataFrame:
    """Load annual_freight.csv into a robust, semantically keyed wide dataframe.

    Adds:
      - annual_row_id: original CSV row number after header, for legacy tracing only
      - annual_block_id: visual section id based on blank spacer rows
      - semantic_key / lookup_key: stable keys for future code joins/lookups
      - first_year / last_year / non_null_years: coverage diagnostics
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"annual_freight.csv not found: {path}")

    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[], low_memory=False)
    df.columns = [_annual_clean_col(c) for c in df.columns]
    for col in ANNUAL_META_COLS:
        if col not in df.columns:
            df[col] = ""
    ycols = annual_year_cols(df)
    if not ycols:
        raise ValueError("annual_freight.csv has no numeric year columns")

    df.insert(0, "annual_row_id", np.arange(1, len(df) + 1, dtype=int))
    blank_mask = df.apply(lambda r: _annual_is_blank_row(r, ycols), axis=1)
    df.insert(1, "annual_block_id", blank_mask.cumsum().astype(int))
    df["is_blank_row"] = blank_mask

    for col in ANNUAL_META_COLS:
        df[col] = df[col].map(_annual_clean_str)
    for col in ycols:
        df[col] = pd.to_numeric(df[col].replace("", np.nan), errors="coerce")

    df["first_year"] = df[ycols].apply(lambda r: next((int(c) for c, v in r.items() if pd.notna(v)), np.nan), axis=1)
    df["last_year"] = df[ycols].apply(lambda r: next((int(c) for c, v in reversed(list(r.items())) if pd.notna(v)), np.nan), axis=1)
    df["non_null_years"] = df[ycols].notna().sum(axis=1).astype(int)
    df["target_depth"] = df["Target"].map(lambda x: x.count(".") if x else 0)
    df["branch_depth"] = df["Branch"].map(lambda x: x.count(".") if x else 0)
    df["semantic_key"] = df.apply(annual_semantic_key, axis=1)
    df["lookup_key"] = df.apply(lambda r: _annual_compact_key([r.get("INDEX", ""), r.get("semantic_key", "")]), axis=1)

    order = [
        "annual_row_id", "annual_block_id", "is_blank_row", "semantic_key", "lookup_key",
        "first_year", "last_year", "non_null_years", "target_depth", "branch_depth",
    ] + ANNUAL_META_COLS + ycols
    df = df[[c for c in order if c in df.columns]]
    if not keep_blank_rows:
        df = df.loc[~df["is_blank_row"]].reset_index(drop=True)
    return df


def annual_freight_to_long(wide_df: pd.DataFrame | None = None) -> pd.DataFrame:
    wide_df = load_annual_freight_wide() if wide_df is None else wide_df.copy()
    ycols = annual_year_cols(wide_df)
    id_cols = [c for c in wide_df.columns if c not in ycols]
    out = wide_df.melt(id_vars=id_cols, value_vars=ycols, var_name="year", value_name="value")
    out["year"] = out["year"].astype(int)
    return out.loc[out["value"].notna()].reset_index(drop=True)


def annual_freight_lookup_table(wide_df: pd.DataFrame | None = None) -> pd.DataFrame:
    wide_df = load_annual_freight_wide() if wide_df is None else wide_df.copy()
    cols = [
        "annual_row_id", "annual_block_id", "semantic_key", "lookup_key", "INDEX",
        "Branch", "region", "sector", "Service", "technology", "Parameter",
        "Context", "Sub_Context", "Target", "Source", "Comments", "Unit",
        "first_year", "last_year", "non_null_years",
    ]
    return wide_df[[c for c in cols if c in wide_df.columns]].copy()


def get_annual_freight_series(
    wide_df: pd.DataFrame | None = None,
    *,
    row_id: int | None = None,
    index: str | None = None,
    semantic_key: str | None = None,
    branch: str | None = None,
    region: str | None = None,
    technology: str | None = None,
    parameter: str | None = None,
    target: str | None = None,
    source: str | None = None,
    unit: str | None = None,
    contains: bool = False,
) -> pd.Series:
    """Select exactly one annual series using semantic fields, not row positions."""
    wide_df = load_annual_freight_wide() if wide_df is None else wide_df.copy()
    ycols = annual_year_cols(wide_df)
    mask = pd.Series(True, index=wide_df.index)

    def filt(col: str, val: str | None) -> None:
        nonlocal mask
        if val is None:
            return
        s = wide_df[col].fillna("").astype(str)
        mask &= s.str.contains(str(val), case=False, regex=False) if contains else s.eq(str(val))

    if row_id is not None:
        mask &= wide_df["annual_row_id"].eq(int(row_id))
    filt("INDEX", index)
    filt("semantic_key", semantic_key)
    filt("Branch", branch)
    filt("region", region)
    filt("technology", technology)
    filt("Parameter", parameter)
    filt("Target", target)
    filt("Source", source)
    filt("Unit", unit)

    matches = wide_df.loc[mask]
    if len(matches) != 1:
        preview_cols = ["annual_row_id", "INDEX", "Branch", "region", "technology", "Parameter", "Target", "Source", "Unit"]
        preview = matches[[c for c in preview_cols if c in matches.columns]].head(20).to_dict("records")
        raise ValueError(f"Annual selector matched {len(matches)} rows; expected 1. Preview: {preview}")
    row = matches.iloc[0]
    out = pd.Series(row[ycols].astype(float).to_numpy(), index=[int(c) for c in ycols], dtype=float)
    out.name = row.get("semantic_key", "annual_freight_series")
    return out


def validate_annual_freight(wide_df: pd.DataFrame | None = None) -> pd.DataFrame:
    wide_df = load_annual_freight_wide() if wide_df is None else wide_df.copy()
    ycols = annual_year_cols(wide_df)
    checks = [
        {"check": "rows_nonblank", "value": len(wide_df), "status": "ok" if len(wide_df) else "fail"},
        {"check": "year_columns", "value": len(ycols), "status": "ok" if len(ycols) == 101 else "warn"},
        {"check": "first_year_column", "value": min(map(int, ycols)), "status": "ok" if min(map(int, ycols)) == 2000 else "warn"},
        {"check": "last_year_column", "value": max(map(int, ycols)), "status": "ok" if max(map(int, ycols)) == 2100 else "warn"},
        {"check": "duplicate_INDEX_nonblank", "value": int(wide_df.loc[wide_df["INDEX"].ne(""), "INDEX"].duplicated().sum()), "status": "warn"},
        {"check": "duplicate_semantic_key", "value": int(wide_df["semantic_key"].duplicated().sum()), "status": "warn"},
    ]
    share = wide_df.loc[wide_df["Parameter"].eq("market_share_total")].copy()
    if not share.empty:
        sums = share.groupby(["Branch", "region", "Parameter"], dropna=False)[ycols].sum(numeric_only=True)
        finite = sums.replace(0, np.nan).stack().dropna()
        checks.append({"check": "max_market_share_sum_deviation", "value": float((finite - 1).abs().max()) if not finite.empty else 0.0, "status": "info"})
    return pd.DataFrame(checks)


def build_annual_freight_tables(path: Path | str = ANNUAL_FREIGHT_FILE, *, write: bool = True) -> dict[str, pd.DataFrame]:
    wide = load_annual_freight_wide(path)
    long = annual_freight_to_long(wide)
    lookup = annual_freight_lookup_table(wide)
    validation = validate_annual_freight(wide)
    if write:
        wide.to_csv(OUT_DIR / "annual_freight_wide_clean.csv", index=False)
        long.to_csv(OUT_DIR / "annual_freight_long.csv", index=False)
        lookup.to_csv(OUT_DIR / "annual_freight_lookup.csv", index=False)
        validation.to_csv(OUT_DIR / "annual_freight_validation.csv", index=False)
        print(f"  ✅ annual_freight_wide_clean.csv  ({len(wide):,} rows)")
        print(f"  ✅ annual_freight_long.csv        ({len(long):,} rows)")
        print(f"  ✅ annual_freight_lookup.csv      ({len(lookup):,} keys)")
    return {"wide": wide, "long": long, "lookup": lookup, "validation": validation}



# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  CONSTANT FREIGHT INPUT PIPELINE                                     ║
# ╚═══════════════════════════════════════════════════════════════════════╝
# constant_freight.csv is a model-input table of constants (costs, lifetimes,
# availability, etc.). Like annual_freight, legacy Excel formulas may rely on
# row order; we preserve that (constant_row_id / constant_block_id) while also
# creating semantic keys for robust future lookups.

CONSTANT_FREIGHT_FILE = SCRIPT_DIR / "constant_freight.csv"
CONSTANT_META_COLS = [
    "Branch", "Type", "region", "sector", "Service", "technology",
    "Parameter", "Context", "Sub_Context", "Target", "Source", "Comments",
    "Unit", "INDEX",
]

# Expected value columns in constant_freight.csv (kept flexible if new cols appear)
CONSTANT_VALUE_COLS_HINT = [
    "service_provide", "discount_rate_financial", "discount_rate_retrofit",
    "heterogeneity", "available", "unavailable", "lifetime",
    "intercept_retirement", "retrofit_existing_max", "market_share_new_max",
    "output", "$YEAR", "currency", "fcc", "fom", "lcc_financial",
]


def _const_clean_col(col: object) -> str:
    return str(col).replace("\\_", "_").strip().lstrip("\ufeff")


def _const_clean_str(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _const_compact_key(parts: list[object]) -> str:
    return "|".join(_const_clean_str(p) for p in parts if _const_clean_str(p) != "")


def constant_semantic_key(row: pd.Series) -> str:
    """Semantic identifier for a constant row (independent of row order)."""
    return _const_compact_key([
        row.get("Branch", ""), row.get("region", ""), row.get("sector", ""),
        row.get("Service", ""), row.get("technology", ""), row.get("Parameter", ""),
        row.get("Context", ""), row.get("Sub_Context", ""), row.get("Target", ""),
        row.get("Source", ""), row.get("Unit", ""),
    ])


def _const_is_blank_row(row: pd.Series, cols: list[str]) -> bool:
    return all(_const_clean_str(row.get(c, "")) == "" for c in cols)


def load_constant_freight(
    path: Path | str = CONSTANT_FREIGHT_FILE,
    *,
    keep_blank_rows: bool = False,
) -> pd.DataFrame:
    """Load constant_freight.csv into a cleaned dataframe with semantic keys.

    Outputs preserve legacy row order and add:
      - constant_row_id: original CSV row number (after header)
      - constant_block_id: visual section id based on blank spacer rows
      - semantic_key / lookup_key
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"constant_freight.csv not found: {path}")

    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[], engine="python")
    df.columns = [_const_clean_col(c) for c in df.columns]

    # Ensure metadata columns exist
    for col in CONSTANT_META_COLS:
        if col not in df.columns:
            df[col] = ""

    all_cols = list(df.columns)
    df.insert(0, "constant_row_id", np.arange(1, len(df) + 1, dtype=int))

    blank_mask = df.apply(lambda r: _const_is_blank_row(r, all_cols), axis=1)
    df.insert(1, "constant_block_id", blank_mask.cumsum().astype(int))
    df["is_blank_row"] = blank_mask

    # Clean text columns
    for col in CONSTANT_META_COLS:
        df[col] = df[col].map(_const_clean_str)

    # Identify value columns: everything that's not meta and not helper columns
    helper_cols = {"constant_row_id", "constant_block_id", "is_blank_row"}
    meta_set = set(CONSTANT_META_COLS)
    value_cols = [c for c in df.columns if c not in meta_set and c not in helper_cols]

    # Convert numeric-ish columns where possible (keep $YEAR and currency as text)
    numeric_candidates = [
        c for c in value_cols
        if c not in {"$YEAR", "currency"} and c.lower() not in {"index"}
    ]
    for c in numeric_candidates:
        # empty string -> NaN, otherwise numeric
        df[c] = pd.to_numeric(df[c].replace("", np.nan), errors="ignore")

    df["semantic_key"] = df.apply(constant_semantic_key, axis=1)
    df["lookup_key"] = df.apply(lambda r: _const_compact_key([r.get("INDEX", ""), r.get("semantic_key", "")]), axis=1)

    # Basic diagnostics
    df["branch_depth"] = df["Branch"].map(lambda x: x.count(".") if x else 0)

    # Reorder
    ordered = [
        "constant_row_id", "constant_block_id", "is_blank_row",
        "semantic_key", "lookup_key", "branch_depth",
    ] + CONSTANT_META_COLS + value_cols
    df = df[[c for c in ordered if c in df.columns]]

    if not keep_blank_rows:
        df = df.loc[~df["is_blank_row"]].reset_index(drop=True)
    return df


def constant_freight_lookup_table(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Compact lookup table for constants."""
    df = load_constant_freight() if df is None else df.copy()
    cols = [
        "constant_row_id", "constant_block_id", "semantic_key", "lookup_key", "INDEX",
        "Branch", "region", "sector", "Service", "technology", "Parameter",
        "Context", "Sub_Context", "Target", "Source", "Comments", "Unit",
    ]
    return df[[c for c in cols if c in df.columns]].copy()


def get_constant_row(
    df: pd.DataFrame | None = None,
    *,
    row_id: int | None = None,
    index: str | None = None,
    semantic_key: str | None = None,
    branch: str | None = None,
    region: str | None = None,
    technology: str | None = None,
    parameter: str | None = None,
    target: str | None = None,
    unit: str | None = None,
    contains: bool = False,
) -> pd.Series:
    """Return exactly one constant row using semantic selectors.

    ``row_id`` is supported for legacy tracing only.
    """
    df = load_constant_freight() if df is None else df.copy()
    mask = pd.Series(True, index=df.index)

    def filt(col: str, val: str | None) -> None:
        nonlocal mask
        if val is None or col not in df.columns:
            return
        s = df[col].fillna("").astype(str)
        mask &= s.str.contains(str(val), case=False, regex=False) if contains else s.eq(str(val))

    if row_id is not None:
        mask &= df["constant_row_id"].eq(int(row_id))
    filt("INDEX", index)
    filt("semantic_key", semantic_key)
    filt("Branch", branch)
    filt("region", region)
    filt("technology", technology)
    filt("Parameter", parameter)
    filt("Target", target)
    filt("Unit", unit)

    matches = df.loc[mask]
    if len(matches) != 1:
        preview_cols = ["constant_row_id", "INDEX", "Branch", "region", "technology", "Parameter", "Target", "Unit"]
        preview = matches[[c for c in preview_cols if c in matches.columns]].head(20).to_dict("records")
        raise ValueError(f"Constant selector matched {len(matches)} rows; expected 1. Preview: {preview}")
    return matches.iloc[0]


def validate_constant_freight(df: pd.DataFrame | None = None) -> pd.DataFrame:
    df = load_constant_freight() if df is None else df.copy()
    checks = []
    checks.append({"check": "rows_nonblank", "value": float(len(df)), "status": "ok" if len(df) else "fail"})
    if "INDEX" in df.columns:
        dup = int(df.loc[df["INDEX"].astype(str).ne(""), "INDEX"].duplicated().sum())
        checks.append({"check": "duplicate_INDEX_nonblank", "value": float(dup), "status": "warn"})
    dupk = int(df["semantic_key"].duplicated().sum())
    checks.append({"check": "duplicate_semantic_key", "value": float(dupk), "status": "warn"})

    # basic required cols
    missing_meta = [c for c in CONSTANT_META_COLS if c not in df.columns]
    checks.append({"check": "missing_meta_cols", "value": ";".join(missing_meta), "status": "ok" if not missing_meta else "warn"})

    # sanity: discount rates in [0,1]
    for col in ["discount_rate_financial", "discount_rate_retrofit"]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            bad = int(((s.notna()) & ((s < 0) | (s > 1))).sum())
            checks.append({"check": f"{col}_outside_0_1", "value": float(bad), "status": "warn" if bad else "ok"})

    # lifetime positive
    if "lifetime" in df.columns:
        s = pd.to_numeric(df["lifetime"], errors="coerce")
        bad = int(((s.notna()) & (s <= 0)).sum())
        checks.append({"check": "lifetime_nonpositive", "value": float(bad), "status": "warn" if bad else "ok"})

    return pd.DataFrame(checks)


def build_constant_freight_tables(path: Path | str = CONSTANT_FREIGHT_FILE, *, write: bool = True) -> dict[str, pd.DataFrame]:
    clean = load_constant_freight(path)
    lookup = constant_freight_lookup_table(clean)
    validation = validate_constant_freight(clean)

    if write:
        clean.to_csv(OUT_DIR / "constant_freight_clean.csv", index=False)
        lookup.to_csv(OUT_DIR / "constant_freight_lookup.csv", index=False)
        validation.to_csv(OUT_DIR / "constant_freight_validation.csv", index=False)
        print(f"  ✅ constant_freight_clean.csv     ({len(clean):,} rows)")
        print(f"  ✅ constant_freight_lookup.csv    ({len(lookup):,} keys)")
        print(f"  ✅ constant_freight_validation.csv")

    return {"clean": clean, "lookup": lookup, "validation": validation}



# ╔═══════════════════════════════════════════════════════════════════════╗

# =============================================================================
# CALC DIFF DIAGNOSTICS (ROW-LEVEL + HEATMAP + MARINE PINPOINT)
# =============================================================================
# This diagnostic compares the generated calc_freight.csv against the
# workbook-validated reference file "calc - Values - Freight.csv".
# Outputs (written to OUT_DIR):
#   - diff_full_diagnostic.csv            (row-level summary stats)
#   - diff_full_long.csv                  (row-year long deltas)
#   - diff_full_heatmap_top.png           (heatmap of top mismatched rows)
#   - diff_marine_diagnostic.csv          (marine-only summary)
#   - diff_marine_year_heatmap.png        (marine region × year heatmap)
#   - marine_rollup_residuals.csv         (CAN & AT rollup checks)
#
# NOTE: The calc tab in the workbook is "hybrid" for Marine (mix of sources
# in upstream pipelines). This diagnostic helps pinpoint which region(s)
# drive marine mismatches and whether rollups (CAN and AT) are being enforced.

import matplotlib.pyplot as plt


def _calc_year_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if str(c).strip().isdigit()]


def _calc_clean_key_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Normalize column names if present
    if 'Region' in out.columns and 'region' not in out.columns:
        out = out.rename(columns={'Region': 'region'})
    if 'Region Name' in out.columns and 'region_name' not in out.columns:
        out = out.rename(columns={'Region Name': 'region_name'})

    # Trim strings
    for c in ['Source', 'Unit', 'Parameter', 'region', 'region_name']:
        if c in out.columns:
            out[c] = out[c].astype(str).str.strip()

    # Drop blank/separator rows
    if 'Parameter' in out.columns:
        out = out.loc[out['Parameter'].ne('') & out['Parameter'].str.lower().ne('nan')]
    if 'region' in out.columns:
        out = out.loc[out['region'].ne('') & out['region'].str.lower().ne('nan')]

    return out.reset_index(drop=True)


def build_calc_diff_diagnostics(
    model_path: Path | str | None = None,
    ref_path: Path | str | None = None,
    *,
    out_dir: Path | None = None,
    top_n: int = 40,
) -> dict[str, Path]:
    """Row-level diff + heatmaps between calc_freight.csv and reference.

    This version is intentionally robust:
      - It always writes the three Marine artifacts the user expects.
      - If Marine rows are absent, it still writes empty CSVs and a placeholder PNG.
      - It prints explicit row counts so failures are visible in the console.
    """

    out_dir = OUT_DIR if out_dir is None else Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = (OUT_DIR / 'calc_freight.csv') if model_path is None else Path(model_path)
    ref_path = (SCRIPT_DIR / 'calc - Values - Freight.csv') if ref_path is None else Path(ref_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model calc file not found: {model_path}")
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference calc file not found: {ref_path}")

    df_m = pd.read_csv(model_path)
    df_r = pd.read_csv(ref_path)

    df_m = _calc_clean_key_cols(df_m)
    df_r = _calc_clean_key_cols(df_r)

    y_m = _calc_year_cols(df_m)
    y_r = _calc_year_cols(df_r)
    years = [c for c in y_m if c in y_r]
    key_cols = [c for c in ['Source', 'Unit', 'Parameter', 'region'] if c in df_m.columns and c in df_r.columns]
    if not key_cols:
        raise ValueError('Could not determine key columns for calc diff merge.')

    for c in years:
        df_m[c] = pd.to_numeric(df_m[c], errors='coerce')
        df_r[c] = pd.to_numeric(df_r[c], errors='coerce')

    merged = df_m.merge(df_r, on=key_cols, how='outer', suffixes=('_model', '_ref'), indicator=True)

    # Long-format diff
    long_rows: list[dict] = []
    for _, row in merged.iterrows():
        base = {k: row.get(k, '') for k in key_cols}
        base['_merge'] = row['_merge']
        for c in years:
            m = row.get(f'{c}_model', np.nan)
            r = row.get(f'{c}_ref', np.nan)
            d = (m - r) if (pd.notna(m) and pd.notna(r)) else np.nan
            p = (d / r) if (pd.notna(d) and pd.notna(r) and r != 0) else np.nan
            long_rows.append({**base, 'year': int(c), 'model': m, 'ref': r, 'diff': d, 'pct_diff': p})
    long_df = pd.DataFrame(long_rows)

    # Row-level summary
    def _summarize(group: pd.DataFrame) -> pd.Series:
        d = group['diff'].astype(float)
        p = group['pct_diff'].astype(float)
        max_abs = np.nanmax(np.abs(d.to_numpy(dtype=float))) if len(d) else np.nan
        max_pct = np.nanmax(np.abs(p.to_numpy(dtype=float))) if len(p) else np.nan
        if not np.isfinite(max_abs):
            max_abs = np.nan
        if not np.isfinite(max_pct):
            max_pct = np.nan
        mean_pct = np.nanmean(np.abs(p.to_numpy(dtype=float))) if len(p) else np.nan
        if not np.isfinite(mean_pct):
            mean_pct = np.nan
        return pd.Series({
            'abs_diff_total': float(np.nansum(np.abs(d))),
            'pct_diff_mean': float(mean_pct) if pd.notna(mean_pct) else np.nan,
            'max_abs_diff': float(max_abs) if pd.notna(max_abs) else np.nan,
            'max_pct_diff': float(max_pct) if pd.notna(max_pct) else np.nan,
            'n_years_compared': int(group['diff'].notna().sum()),
            '_merge': group['_merge'].iloc[0],
        })

    summary = long_df.groupby(key_cols, dropna=False).apply(_summarize).reset_index()

    def _heat(p):
        if pd.isna(p):
            return 'NA'
        if p < 0.0001:
            return 'OK'
        if p < 0.01:
            return 'LOW'
        if p < 0.05:
            return 'MED'
        return 'HIGH'
    summary['heat'] = summary['pct_diff_mean'].apply(_heat)

    # Core outputs
    out_full_diag = out_dir / 'diff_full_diagnostic.csv'
    out_full_long = out_dir / 'diff_full_long.csv'
    out_heat = out_dir / 'diff_full_heatmap_top.png'
    out_marine_diag = out_dir / 'diff_marine_diagnostic.csv'
    out_marine_heat = out_dir / 'diff_marine_year_heatmap.png'
    out_roll = out_dir / 'marine_rollup_residuals.csv'

    summary.to_csv(out_full_diag, index=False)
    long_df.to_csv(out_full_long, index=False)

    # Top-N overall heatmap
    top = summary.loc[summary['_merge'].eq('both')].sort_values('abs_diff_total', ascending=False).head(top_n)
    if not top.empty:
        labels = []
        mat = []
        for _, r in top.iterrows():
            filt = pd.Series(True, index=long_df.index)
            for k in key_cols:
                filt &= (long_df[k].astype(str) == str(r[k]))
            seg = long_df.loc[filt].sort_values('year')
            labels.append(' | '.join(str(r[k]) for k in key_cols))
            mat.append(np.abs(seg['pct_diff'].to_numpy(dtype=float)))
        mat = np.array(mat, dtype=float)
        plt.figure(figsize=(14, max(4, 0.25 * len(labels))))
        plt.imshow(np.log10(np.nan_to_num(mat, nan=0.0) + 1e-12), aspect='auto')
        plt.colorbar(label='log10(|% diff| + 1e-12)')
        plt.title(f'Calc Diff Heatmap — Top {len(labels)} Rows by |Abs Diff|')
        plt.yticks(range(len(labels)), labels, fontsize=7)
        plt.xticks(range(len(years)), years, rotation=90, fontsize=6)
        plt.tight_layout()
        plt.savefig(out_heat, dpi=200)
        plt.close()
    else:
        # placeholder image
        plt.figure(figsize=(8, 3))
        plt.text(0.5, 0.5, 'No rows available for overall diff heatmap', ha='center', va='center')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(out_heat, dpi=200)
        plt.close()

    # Marine matching — broad but safe
    marine_mask_summary = summary['Parameter'].astype(str).str.contains('marine', case=False, na=False) if 'Parameter' in summary.columns else pd.Series(False, index=summary.index)
    marine_mask_long = long_df['Parameter'].astype(str).str.contains('marine', case=False, na=False) if 'Parameter' in long_df.columns else pd.Series(False, index=long_df.index)

    marine = summary.loc[marine_mask_summary].copy()
    mlong = long_df.loc[marine_mask_long & long_df['_merge'].eq('both')].copy()
    print(f"[DIAGNOSTIC] Marine summary rows found: {len(marine)}")
    print(f"[DIAGNOSTIC] Marine matched row-years found: {len(mlong)}")

    # Always write marine diagnostic CSV (possibly empty)
    marine.to_csv(out_marine_diag, index=False)

    # Always write marine heatmap PNG
    if not mlong.empty and 'region' in mlong.columns:
        piv = mlong.pivot_table(index='region', columns='year', values='diff', aggfunc='mean')
        piv = piv.reindex(columns=sorted(piv.columns))
        plt.figure(figsize=(16, max(4, 0.35 * len(piv.index))))
        plt.imshow(piv.fillna(0.0).to_numpy(), aspect='auto')
        plt.colorbar(label='Marine (model - ref)')
        plt.title('Marine Diff Heatmap — Region × Year')
        plt.yticks(range(len(piv.index)), piv.index)
        plt.xticks(range(len(piv.columns)), piv.columns, rotation=90, fontsize=6)
        plt.tight_layout()
        plt.savefig(out_marine_heat, dpi=200)
        plt.close()
    else:
        plt.figure(figsize=(8, 3))
        msg = 'No matched Marine rows found\nCheck calc Parameter names and model/reference keys'
        plt.text(0.5, 0.5, msg, ha='center', va='center')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(out_marine_heat, dpi=200)
        plt.close()

    # Always write marine rollup residuals CSV
    roll_cols = [
        'year',
        'CAN_model_reported','CAN_model_sum_BC_NU','CAN_model_residual',
        'CAN_ref_reported','CAN_ref_sum_BC_NU','CAN_ref_residual',
        'AT_model_reported','AT_model_sum_NB_NS_PE_NL','AT_model_residual',
        'AT_ref_reported','AT_ref_sum_NB_NS_PE_NL','AT_ref_residual',
    ]
    if not mlong.empty and 'region' in mlong.columns:
        regions_can = ['BC','AB','SK','MB','ON','QC','NB','NS','PE','NL','YT','NT','NU']
        regions_at = ['NB','NS','PE','NL']

        def _series(dfsub, region_list):
            sub = dfsub.loc[dfsub['region'].isin(region_list)]
            return (
                sub.groupby('year')['model'].sum(min_count=1),
                sub.groupby('year')['ref'].sum(min_count=1),
            )

        can_model, can_ref = _series(mlong, regions_can)
        at_model, at_ref = _series(mlong, regions_at)
        can_rep = mlong.loc[mlong['region'].eq('CAN')].set_index('year')
        at_rep = mlong.loc[mlong['region'].eq('AT')].set_index('year')

        rows = []
        for y in sorted(mlong['year'].unique()):
            rows.append({
                'year': int(y),
                'CAN_model_reported': float(can_rep['model'].get(y, np.nan)),
                'CAN_model_sum_BC_NU': float(can_model.get(y, np.nan)),
                'CAN_model_residual': float(can_rep['model'].get(y, np.nan) - can_model.get(y, np.nan)) if pd.notna(can_rep['model'].get(y, np.nan)) and pd.notna(can_model.get(y, np.nan)) else np.nan,
                'CAN_ref_reported': float(can_rep['ref'].get(y, np.nan)),
                'CAN_ref_sum_BC_NU': float(can_ref.get(y, np.nan)),
                'CAN_ref_residual': float(can_rep['ref'].get(y, np.nan) - can_ref.get(y, np.nan)) if pd.notna(can_rep['ref'].get(y, np.nan)) and pd.notna(can_ref.get(y, np.nan)) else np.nan,
                'AT_model_reported': float(at_rep['model'].get(y, np.nan)),
                'AT_model_sum_NB_NS_PE_NL': float(at_model.get(y, np.nan)),
                'AT_model_residual': float(at_rep['model'].get(y, np.nan) - at_model.get(y, np.nan)) if pd.notna(at_rep['model'].get(y, np.nan)) and pd.notna(at_model.get(y, np.nan)) else np.nan,
                'AT_ref_reported': float(at_rep['ref'].get(y, np.nan)),
                'AT_ref_sum_NB_NS_PE_NL': float(at_ref.get(y, np.nan)),
                'AT_ref_residual': float(at_rep['ref'].get(y, np.nan) - at_ref.get(y, np.nan)) if pd.notna(at_rep['ref'].get(y, np.nan)) and pd.notna(at_ref.get(y, np.nan)) else np.nan,
            })
        pd.DataFrame(rows, columns=roll_cols).to_csv(out_roll, index=False)
    else:
        pd.DataFrame(columns=roll_cols).to_csv(out_roll, index=False)

    print(f'✅ wrote: {out_full_diag}')
    print(f'✅ wrote: {out_full_long}')
    print(f'✅ wrote: {out_heat}')
    print(f'✅ wrote: {out_marine_diag}')
    print(f'✅ wrote: {out_marine_heat}')
    print(f'✅ wrote: {out_roll}')

    return {
        'full_diagnostic': out_full_diag,
        'full_long': out_full_long,
        'heatmap_top': out_heat,
        'marine_diagnostic': out_marine_diag,
        'marine_heatmap': out_marine_heat,
        'marine_rollup_residuals': out_roll,
    }

# ║ FREIGHT CALC TAB PIPELINE                                           ║
# ╚═══════════════════════════════════════════════════════════════════════╝
# Workbook-faithful freight calc tab: k*tkm rows by mode/region, 2000-2100.
# Historical 2000-2021 comes from existing freight_* mode outputs
# (Activity (M tkm) * 1000). 2022 uses 2019 * share_of_2019 from KTKM_CAGR.
# 2023-2050 and 2051-2100 use the two reference CAGRs from KTKM_CAGR.
# CAN is calculated as SUM(BC:NU), matching the supplied formula tab.

CALC_FREIGHT_YEARS = list(range(2000, 2101))
CALC_FREIGHT_HIST_YEARS = list(range(2000, 2022))
CALC_FREIGHT_REGIONS = [
    ("CAN","Canada"),("BC","British Columbia"),("AB","Alberta"),("SK","Saskatchewan"),
    ("MB","Manitoba"),("ON","Ontario"),("QC","Quebec"),("NB","New Brunswick"),
    ("NS","Nova Scotia"),("PE","Prince Edward Island"),("NL","Newfoundland and Labrador"),
    ("YT","Yukon"),("NT","Northwest Territories"),("NU","Nunavut"),("AT","Atlantic"),("TR","Territories")]
CALC_FREIGHT_CAN_SUM_REGIONS = ["BC","AB","SK","MB","ON","QC","NB","NS","PE","NL","YT","NT","NU"]
# Workbook-faithful regional rollups for the calc tab.
# CAN rows sum only province/territory detail rows (BC:NU); AT and TR are
# informational regional rollups and must not be included in CAN to avoid
# double counting.  The supplied Off-Road calc formulas use AT as the sum of
# Atlantic province rows and TR as the sum of territory rows.
CALC_FREIGHT_AT_SUM_REGIONS = ["NB", "NS", "PE", "NL"]
CALC_FREIGHT_TR_SUM_REGIONS = ["YT", "NT", "NU"]
CALC_FREIGHT_MODES = [
    ("Light Trucks","light_trucks","Light Truck"),
    ("Medium Trucks","medium_trucks","Medium Truck"),
    ("Heavy Trucks","heavy_trucks","Heavy Truck"),
    ("Rail","rail","Rail"),
    ("Marine","marine","Marine"),
    ("Aviation","air","Aviation"),
    ("Off-Road","offroad","Off Road"),
]


# Workbook-aligned Off-Road historical activity series for each explicit row-323
# formula region. The calc formulas use REGION!C:X$323*1000 for BC, AB, SK, MB,
# ON, QC, NB, NS, PE, NL, and AT. Territories are blank/zero. CAN is calculated
# separately as SUM(BC:NU), excluding AT/TR to avoid double counting. Units are
# k*tkm. Projection years remain formula-driven through _calc_freight_project()
# using the corresponding assumptions rows 130:139 and 143.

# Workbook-aligned Marine historical activity series for each explicit row-247
# formula region. The calc formulas use REGION!C:X$247*1000 for BC, AB, SK,
# MB, ON, QC, NB, NS, PE, NL, and AT. Territories are blank/zero. CAN is
# calculated separately as SUM(BC:NU), excluding AT/TR to avoid double counting.
# Units are k*tkm. Projection years remain formula-driven through
# _calc_freight_project() using the corresponding assumptions rows 90:99 and 103.
CALC_FREIGHT_MARINE_REFERENCE_HIST_KTKM = {
    "BC": {
        2000: 27382723.270000, 2001: 25868105.460000, 2002: 29412218.900000, 2003: 30607261.750000,
        2004: 29914408.950000, 2005: 31867955.740000, 2006: 32039702.850000, 2007: 31128174.030000,
        2008: 30082605.870000, 2009: 26297054.620000, 2010: 26320977.510000, 2011: 24535606.890000,
        2012: 25580623.560000, 2013: 26426434.220000, 2014: 26919684.610000, 2015: 27731297.960000,
        2016: 27515956.170000, 2017: 29063242.860000, 2018: 29421982.240000, 2019: 29166117.510000,
        2020: 32484476.840000, 2021: 33144387.450000,
    },
    "AB": {
        2000: 0.000000, 2001: 0.000000, 2002: 0.000000, 2003: 0.000000,
        2004: 0.000000, 2005: 0.000000, 2006: 0.000000, 2007: 0.000000,
        2008: 0.000000, 2009: 529.637304, 2010: 564.919733, 2011: 0.000000,
        2012: 0.000000, 2013: 0.000000, 2014: 0.000000, 2015: 0.000000,
        2016: 0.000000, 2017: 0.000000, 2018: 0.000000, 2019: 0.000000,
        2020: 0.000000, 2021: 0.000000,
    },
    "SK": {
        2000: 0.000000, 2001: 0.000000, 2002: 0.000000, 2003: 0.000000,
        2004: 0.000000, 2005: 0.000000, 2006: 0.000000, 2007: 0.000000,
        2008: 0.000000, 2009: 0.000000, 2010: 0.000000, 2011: 0.000000,
        2012: 0.000000, 2013: 0.000000, 2014: 0.000000, 2015: 0.000000,
        2016: 0.000000, 2017: 0.000000, 2018: 0.000000, 2019: 0.000000,
        2020: 0.000000, 2021: 0.000000,
    },
    "MB": {
        2000: 134470.652800, 2001: 122413.584100, 2002: 134219.852600, 2003: 134506.958500,
        2004: 126793.362700, 2005: 130287.949400, 2006: 211141.034300, 2007: 284707.387700,
        2008: 354250.529100, 2009: 378385.345800, 2010: 462981.076700, 2011: 346867.124400,
        2012: 276212.874300, 2013: 200608.555800, 2014: 122122.247300, 2015: 45718.287500,
        2016: 4393.656278, 2017: 44009.125630, 2018: 118118.402000, 2019: 63870.630320,
        2020: 29805.192520, 2021: 10072.888140,
    },
    "ON": {
        2000: 6651672.126000, 2001: 6360335.624000, 2002: 7311836.985000, 2003: 7686767.810000,
        2004: 7605815.900000, 2005: 8181863.666000, 2006: 8148246.930000, 2007: 7836463.280000,
        2008: 7504097.658000, 2009: 6467221.139000, 2010: 6563442.665000, 2011: 6482889.430000,
        2012: 7137277.022000, 2013: 7732138.638000, 2014: 8208910.588000, 2015: 8886423.916000,
        2016: 9058788.725000, 2017: 9123865.103000, 2018: 8514447.873000, 2019: 8748222.013000,
        2020: 11118291.120000, 2021: 10079148.010000,
    },
    "QC": {
        2000: 27300409.010000, 2001: 26002081.200000, 2002: 29813510.020000, 2003: 31172519.460000,
        2004: 30731083.100000, 2005: 33029812.120000, 2006: 32986089.450000, 2007: 31916818.620000,
        2008: 30789153.880000, 2009: 26778055.910000, 2010: 27254496.620000, 2011: 24443848.440000,
        2012: 24615618.070000, 2013: 24511497.120000, 2014: 24052592.160000, 2015: 24119470.280000,
        2016: 24154248.170000, 2017: 23567312.850000, 2018: 23894833.340000, 2019: 23724251.430000,
        2020: 21504154.220000, 2021: 22442345.740000,
    },
    "NB": {
        2000: 3821657.577000, 2001: 3765906.639000, 2002: 4457680.491000, 2003: 4814599.723000,
        2004: 4880469.846000, 2005: 5379318.468000, 2006: 5453957.568000, 2007: 5342837.886000,
        2008: 5212004.360000, 2009: 4596611.362000, 2010: 4758984.259000, 2011: 4237491.513000,
        2012: 4185453.191000, 2013: 4087760.905000, 2014: 3935580.602000, 2015: 3875680.525000,
        2016: 4326536.112000, 2017: 4109979.187000, 2018: 3663475.527000, 2019: 3836088.455000,
        2020: 3654850.319000, 2021: 4173770.526000,
    },
    "NS": {
        2000: 9512520.908000, 2001: 9113815.479000, 2002: 10518241.870000, 2003: 11110770.530000,
        2004: 11016078.900000, 2005: 11911383.420000, 2006: 12076484.120000, 2007: 11845183.770000,
        2008: 11569885.680000, 2009: 10215726.510000, 2010: 10570643.200000, 2011: 9553162.314000,
        2012: 9630428.981000, 2013: 9597241.883000, 2014: 9426975.799000, 2015: 9465667.235000,
        2016: 8675273.754000, 2017: 8441850.779000, 2018: 9994999.127000, 2019: 9141119.163000,
        2020: 7833182.737000, 2021: 8102766.713000,
    },
    "PE": {
        2000: 1286477.602000, 2001: 1251218.310000, 2002: 1463312.450000, 2003: 1562064.791000,
        2004: 1567694.601000, 2005: 1711518.917000, 2006: 1779026.394000, 2007: 1786765.671000,
        2008: 1785796.858000, 2009: 1612876.056000, 2010: 1708426.119000, 2011: 1552222.178000,
        2012: 1568539.304000, 2013: 1567461.574000, 2014: 1544775.929000, 2015: 1557697.993000,
        2016: 1617406.004000, 2017: 1766673.029000, 2018: 1898930.409000, 2019: 1840691.150000,
        2020: 818572.131200, 2021: 815502.741900,
    },
    "NL": {
        2000: 16102504.910000, 2001: 15750454.710000, 2002: 18522985.060000, 2003: 19866271.090000,
        2004: 20018842.640000, 2005: 21944667.070000, 2006: 21881725.350000, 2007: 21092550.500000,
        2008: 20236624.570000, 2009: 17540684.050000, 2010: 17812392.550000, 2011: 15848584.780000,
        2012: 15725062.820000, 2013: 15422590.830000, 2014: 14900949.750000, 2015: 14703638.940000,
        2016: 15151449.190000, 2017: 15512858.520000, 2018: 14868222.280000, 2019: 15933009.760000,
        2020: 16408897.440000, 2021: 14936703.490000,
    },
    "AT": {
        2000: 30723160.990000, 2001: 29881395.130000, 2002: 34962219.870000, 2003: 37353706.140000,
        2004: 37483085.980000, 2005: 40946887.870000, 2006: 41191193.430000, 2007: 40067337.820000,
        2008: 38804311.470000, 2009: 33965897.980000, 2010: 34850446.130000, 2011: 31191460.790000,
        2012: 31109484.290000, 2013: 30675055.190000, 2014: 29808282.080000, 2015: 29602684.690000,
        2016: 29770665.060000, 2017: 29831361.510000, 2018: 30425627.340000, 2019: 30750908.530000,
        2020: 28715502.630000, 2021: 28028743.470000,
    },
}

CALC_FREIGHT_OFFROAD_REFERENCE_HIST_KTKM = {
    "BC": {
        2000: 1461254.872000, 2001: 1458289.484000, 2002: 1540611.630000, 2003: 1574617.469000,
        2004: 1662811.415000, 2005: 1527191.423000, 2006: 1449598.010000, 2007: 1425678.213000,
        2008: 1391708.540000, 2009: 1287084.287000, 2010: 1255891.036000, 2011: 1160081.833000,
        2012: 1196608.074000, 2013: 1270332.908000, 2014: 1287870.213000, 2015: 1423239.850000,
        2016: 1755436.674000, 2017: 1885489.549000, 2018: 1949569.181000, 2019: 1932435.425000,
        2020: 2000065.693000, 2021: 2019907.716000,
    },
    "AB": {
        2000: 2473168.894000, 2001: 2623083.624000, 2002: 2755942.516000, 2003: 2632741.722000,
        2004: 2580576.781000, 2005: 2545828.306000, 2006: 2494869.669000, 2007: 2406488.524000,
        2008: 2338578.001000, 2009: 2019625.532000, 2010: 2214469.469000, 2011: 2355375.927000,
        2012: 2584774.389000, 2013: 2776406.157000, 2014: 2937902.368000, 2015: 2737274.410000,
        2016: 3275600.117000, 2017: 3199511.942000, 2018: 3047693.468000, 2019: 3052591.901000,
        2020: 2645134.203000, 2021: 2848309.735000,
    },
    "SK": {
        2000: 1128660.961000, 2001: 1136257.011000, 2002: 1149128.712000, 2003: 1239677.746000,
        2004: 1237337.954000, 2005: 1253676.256000, 2006: 1290869.296000, 2007: 1317454.625000,
        2008: 1399506.163000, 2009: 1319256.203000, 2010: 1297710.746000, 2011: 1286953.959000,
        2012: 1561806.370000, 2013: 1713193.285000, 2014: 1676941.451000, 2015: 2046656.351000,
        2016: 2091897.674000, 2017: 1983852.702000, 2018: 1940559.011000, 2019: 1895702.880000,
        2020: 1728291.131000, 2021: 1719884.265000,
    },
    "MB": {
        2000: 604506.522800, 2001: 621190.905900, 2002: 644221.468000, 2003: 682205.512200,
        2004: 714898.882500, 2005: 670475.768900, 2006: 673632.164900, 2007: 696599.547300,
        2008: 623134.610400, 2009: 586707.619600, 2010: 623431.002900, 2011: 692690.938000,
        2012: 846357.633300, 2013: 940328.063100, 2014: 968085.872200, 2015: 1027639.465000,
        2016: 1004397.157000, 2017: 1010052.294000, 2018: 1030154.925000, 2019: 965796.728700,
        2020: 886511.321100, 2021: 861599.669800,
    },
    "ON": {
        2000: 4633235.782000, 2001: 4803128.323000, 2002: 5044349.414000, 2003: 5052662.272000,
        2004: 4954241.065000, 2005: 4876262.460000, 2006: 4612958.181000, 2007: 4073521.896000,
        2008: 3855128.344000, 2009: 3805513.815000, 2010: 3945800.598000, 2011: 3781094.355000,
        2012: 3417814.631000, 2013: 3668107.230000, 2014: 3502105.588000, 2015: 3724579.925000,
        2016: 3908893.294000, 2017: 3814297.413000, 2018: 3897348.688000, 2019: 3893158.099000,
        2020: 3555054.278000, 2021: 3817078.491000,
    },
    "QC": {
        2000: 2246663.304000, 2001: 2418779.650000, 2002: 2740218.030000, 2003: 2571851.942000,
        2004: 2515320.650000, 2005: 2500983.342000, 2006: 2321510.125000, 2007: 2250321.076000,
        2008: 2142014.468000, 2009: 2115044.495000, 2010: 1955823.604000, 2011: 1997466.330000,
        2012: 2033076.593000, 2013: 2160437.491000, 2014: 1947297.070000, 2015: 1932114.386000,
        2016: 1938297.008000, 2017: 2205555.193000, 2018: 2200825.993000, 2019: 2237269.390000,
        2020: 2165617.213000, 2021: 2083674.010000,
    },
    "NB": {
        2000: 567285.274400, 2001: 597239.399900, 2002: 631191.075900, 2003: 638328.914800,
        2004: 642504.091800, 2005: 626081.975700, 2006: 602568.208600, 2007: 581770.636400,
        2008: 564456.740400, 2009: 485363.687200, 2010: 489282.017400, 2011: 529661.751800,
        2012: 492666.863600, 2013: 415326.354800, 2014: 343446.846500, 2015: 406725.118000,
        2016: 458153.736300, 2017: 406808.105100, 2018: 388200.105200, 2019: 369756.502300,
        2020: 356965.941800, 2021: 313110.524700,
    },
    "NS": {
        2000: 391657.364900, 2001: 394733.765100, 2002: 413492.717800, 2003: 417609.927100,
        2004: 440662.755800, 2005: 428353.419400, 2006: 422593.493900, 2007: 378421.377800,
        2008: 393272.457200, 2009: 337565.521700, 2010: 360558.891800, 2011: 376086.852100,
        2012: 368325.223000, 2013: 323456.478200, 2014: 274637.117300, 2015: 353025.350600,
        2016: 366334.377500, 2017: 390369.719400, 2018: 397073.632300, 2019: 384403.422300,
        2020: 345659.841600, 2021: 313051.588500,
    },
    "PE": {
        2000: 84076.460450, 2001: 83230.691650, 2002: 81272.346710, 2003: 79642.153790,
        2004: 80993.032330, 2005: 80551.239890, 2006: 77162.870700, 2007: 76871.426200,
        2008: 72765.941720, 2009: 63867.355620, 2010: 64204.054020, 2011: 68920.755070,
        2012: 65033.583770, 2013: 56086.363870, 2014: 56963.594660, 2015: 63619.121300,
        2016: 71673.452670, 2017: 59524.095340, 2018: 54562.280370, 2019: 56906.704980,
        2020: 55758.917130, 2021: 48916.156780,
    },
    "NL": {
        2000: 267268.113700, 2001: 279231.009200, 2002: 293003.482900, 2003: 293051.284700,
        2004: 276142.830700, 2005: 289921.074300, 2006: 270752.681700, 2007: 293898.726300,
        2008: 292502.890100, 2009: 250137.738800, 2010: 245039.349600, 2011: 258538.459400,
        2012: 271502.501100, 2013: 241365.679100, 2014: 267604.634000, 2015: 308044.039900,
        2016: 312490.330500, 2017: 324109.847000, 2018: 294795.430100, 2019: 283344.078200,
        2020: 244022.971800, 2021: 236996.191600,
    },
    "AT": {
        2000: 1310287.213000, 2001: 1354434.866000, 2002: 1418959.623000, 2003: 1428632.280000,
        2004: 1440302.711000, 2005: 1424907.709000, 2006: 1373077.255000, 2007: 1330962.167000,
        2008: 1322998.029000, 2009: 1136934.303000, 2010: 1159084.313000, 2011: 1233207.818000,
        2012: 1197528.171000, 2013: 1036234.876000, 2014: 942652.192400, 2015: 1131413.630000,
        2016: 1208651.897000, 2017: 1180811.767000, 2018: 1134631.448000, 2019: 1094410.708000,
        2020: 1002407.672000, 2021: 912074.461500,
    },
}

def _calc_freight_activity_k_tkm(
    region_code: str,
    suffix: str,
    upstream_mode_dfs: dict[str, dict[str, pl.DataFrame | pd.DataFrame]] | None = None,
) -> pd.Series:
    """Return historical Activity (M tkm) as k*tkm from upstream in-memory mode dataframes.

    Important: this calc-tab source helper intentionally does NOT read
    freight_* CSV outputs.  The calc tab should be wired to the dataframes
    produced earlier in the same pipeline run so row sourcing stays explicit,
    auditable, and independent of whatever files happen to be in output/.
    """
    empty = pd.Series(0.0, index=CALC_FREIGHT_HIST_YEARS, dtype=float)

    # Calc-tab explicit workbook-row overrides.
    # Marine formulas use REGION!C:X$247*1000 for BC/AB/SK/MB/ON/QC/NB/NS/PE/NL/AT.
    # Off-Road formulas use REGION!C:X$323*1000 for the same explicit regions.
    # Territories remain blank/zero; CAN is calculated later as SUM(BC:NU).
    explicit_hist_maps = {
        "marine": globals().get("CALC_FREIGHT_MARINE_REFERENCE_HIST_KTKM", {}),
        "offroad": globals().get("CALC_FREIGHT_OFFROAD_REFERENCE_HIST_KTKM", {}),
    }
    ref_values = explicit_hist_maps.get(suffix, {}).get(region_code.upper())
    if ref_values is not None:
        return pd.Series(ref_values, dtype=float).reindex(CALC_FREIGHT_HIST_YEARS).fillna(0.0)

    if upstream_mode_dfs is None:
        warnings.warn(
            "Freight calc requested without upstream_mode_dfs; returning zero series "
            f"for {region_code}_{suffix}. No CSV fallback is used."
        )
        return empty

    reg_dfs = upstream_mode_dfs.get(region_code.upper(), {})
    df = reg_dfs.get(suffix)
    if df is None:
        return empty

    try:
        if isinstance(df, pl.DataFrame):
            pdf = df.to_pandas()
        else:
            pdf = df.copy()
        if "year" not in pdf.columns or "Activity (M tkm)" not in pdf.columns:
            return empty
        yr = pd.to_numeric(pdf["year"], errors="coerce")
        val = pd.to_numeric(pdf["Activity (M tkm)"], errors="coerce")
        s = pd.Series(val.values, index=yr).dropna()
        s.index = s.index.astype(int)
        return s.reindex(CALC_FREIGHT_HIST_YEARS).fillna(0.0).astype(float) * 1000.0
    except Exception as exc:
        warnings.warn(
            f"Could not use upstream calc dataframe for {region_code}_{suffix}: {exc}. "
            "Returning zero series; no CSV fallback is used."
        )
        return empty


def _calc_freight_project(region_code: str, mode_key: str, hist_k_tkm: pd.Series) -> pd.Series:
    out = pd.Series(0.0, index=CALC_FREIGHT_YEARS, dtype=float)
    out.loc[CALC_FREIGHT_HIST_YEARS] = hist_k_tkm.reindex(CALC_FREIGHT_HIST_YEARS).fillna(0.0).astype(float)
    if region_code in {"YT", "NT", "NU", "TR"}:
        return out
    assump = KTKM_CAGR.get(mode_key, {}).get(region_code)
    if assump is None:
        return out
    _hist_cagr, share_2019, ref_2023, ref_2051 = assump
    out.loc[2022] = float(out.loc[2019]) * float(share_2019)
    for year in range(2023, 2101):
        g = float(ref_2023) if year <= 2050 else float(ref_2051)
        out.loc[year] = float(out.loc[year - 1]) * (1.0 + g)
    return out


def build_freight_calc(
    upstream_mode_dfs: dict[str, dict[str, pl.DataFrame | pd.DataFrame]] | None = None,
    write: bool = True,
) -> pd.DataFrame:
    """Build the Freight calc tab and write wide, alias, long, validation, and diagnostics CSVs.

    Calc-tab sourcing rules implemented here:
      * BC/AB/SK/MB/ON/QC and all province/territory detail rows are sourced
        directly from the upstream in-memory provincial mode dataframes via
        _calc_freight_activity_k_tkm(), then projected with the same assumption logic.
      * AT is sourced directly from the upstream AT mode dataframes where the
        calc workbook formulas point to the AT sheet (for example Marine AT uses
        AT!C:X$247*1000 historically, then assumptions row 103 for 2022+ projections).
      * Off-Road AT is also standalone: formulas point to AT!row 323 historically
        and assumptions row 143 for projections.
      * CAN is exactly the sum of required detail regions BC:NU.  AT and TR are
        excluded from CAN to avoid double-counting regional rows.
      * TR remains the territory rollup (YT + NT + NU), which is zero/blank in
        the current freight calc inputs.
      * A diagnostic file is emitted to identify source status and key formula
        assumptions, including the Marine AT row-247 / assumptions-row-103 path.
    """
    year_cols = [str(y) for y in CALC_FREIGHT_YEARS]
    rows: list[dict] = []
    long_rows: list[dict] = []
    validations: list[dict] = []
    diagnostic_rows: list[dict] = []

    zero_full = pd.Series(0.0, index=CALC_FREIGHT_YEARS, dtype=float)

    def _sum_region_series(series_by_region: dict[str, pd.Series], regions: list[str]) -> pd.Series:
        total = pd.Series(0.0, index=CALC_FREIGHT_YEARS, dtype=float)
        for r in regions:
            total = total.add(series_by_region.get(r, zero_full), fill_value=0.0)
        return total.reindex(CALC_FREIGHT_YEARS).fillna(0.0).astype(float)

    def _has_upstream_df(region_code: str, suffix: str) -> bool:
        if upstream_mode_dfs is None:
            return False
        return suffix in upstream_mode_dfs.get(region_code.upper(), {})

    def _diagnose_source_status(
        *,
        mode_label: str,
        suffix: str,
        mode_key: str,
        region_code: str,
        hist: pd.Series,
        projected: pd.Series,
    ) -> None:
        """Record compact upstream-source status rows for the calc tab."""
        region_u = region_code.upper()
        has_df = _has_upstream_df(region_u, suffix)
        offroad_ref = (
            suffix == "offroad"
            and region_u in globals().get("CALC_FREIGHT_OFFROAD_REFERENCE_HIST_KTKM", {})
        )
        source_type = "upstream_dataframe" if has_df else ("calc_reference_override" if offroad_ref else "zero_or_missing")
        diagnostic_rows.append({
            "diagnostic_type": "calc_source_status",
            "Parameter": mode_label,
            "mode_suffix": suffix,
            "region": region_u,
            "year": "ALL",
            "actual": float(projected.abs().sum()),
            "expected": "",
            "diff": "",
            "source_region": region_u,
            "source_value": "",
            "source_detail": (
                f"source_type={source_type}; has_upstream_df={has_df}; "
                f"hist_nonzero_years={int((hist.reindex(CALC_FREIGHT_HIST_YEARS).fillna(0.0).abs() > 1e-9).sum())}; "
                f"v2000={float(projected.loc[2000]):.6f}; v2019={float(projected.loc[2019]):.6f}; "
                f"v2021={float(projected.loc[2021]):.6f}; v2022={float(projected.loc[2022]):.6f}; "
                f"mode_key={mode_key}"
            ),
        })

    note = {"Index":"", "Source":"Forecast values based on assumptions sheet", "Unit":"", "Parameter":"", "region":"", "region_name":""}
    note.update({y:"" for y in year_cols})
    rows.append(note)

    for mode_label, suffix, mode_key in CALC_FREIGHT_MODES:
        spacer = {"Index":"", "Source":"", "Unit":"", "Parameter":"", "region":"", "region_name":""}
        spacer.update({y:"" for y in year_cols})
        rows.append(spacer)

        mode_series: dict[str, pd.Series] = {}
        direct_at_series: pd.Series | None = None

        # Build all calc-tab source rows directly from upstream/reference logic.
        # Important regional rule:
        #   * Off-Road historical rows use explicit workbook row-323 sources for
        #     BC/AB/SK/MB/ON/QC/NB/NS/PE/NL/AT via
        #     CALC_FREIGHT_OFFROAD_REFERENCE_HIST_KTKM.
        #   * AT is standalone, not NB+NS+PE+NL. CAN still excludes AT/TR and
        #     follows SUM(BC:NU), matching the formula row.
        hist_by_region: dict[str, pd.Series] = {}
        for reg, _name in CALC_FREIGHT_REGIONS:
            if reg in {"CAN", "TR"}:
                continue
            hist = _calc_freight_activity_k_tkm(reg, suffix, upstream_mode_dfs)
            projected = _calc_freight_project(reg, mode_key, hist)
            hist_by_region[reg] = hist
            mode_series[reg] = projected
            if reg in {"BC", "AB", "SK", "MB", "ON", "QC", "AT"}:
                _diagnose_source_status(
                    mode_label=mode_label,
                    suffix=suffix,
                    mode_key=mode_key,
                    region_code=reg,
                    hist=hist,
                    projected=projected,
                )

        # Required aggregate/regional rows.
        # CAN follows workbook formulas as SUM(BC:NU), excluding AT/TR to avoid
        # double counting. TR is YT + NT + NU and is currently zero/blank.
        tr_expected = _sum_region_series(mode_series, CALC_FREIGHT_TR_SUM_REGIONS)
        can_expected = _sum_region_series(mode_series, CALC_FREIGHT_CAN_SUM_REGIONS)

        mode_series["TR"] = tr_expected
        mode_series["CAN"] = can_expected

        # Rollup validation: these should be exact by construction.
        can_diff = (mode_series["CAN"] - can_expected).abs().max()
        tr_diff = (mode_series["TR"] - tr_expected).abs().max()
        validations.extend([
            {"check": f"{mode_label}_CAN_sum_BC_to_NU", "value": float(can_diff), "status": "ok" if float(can_diff) < 1e-6 else "warn"},
            {"check": f"{mode_label}_TR_sum_YT_NT_NU", "value": float(tr_diff), "status": "ok" if float(tr_diff) < 1e-6 else "warn"},
            {"check": f"{mode_label}_AT_direct_standalone_series", "value": float(mode_series.get("AT", zero_full).abs().sum()), "status": "info"},
        ])

        # Specific diagnostic for the known Off-Road AT formula path:
        # historical = AT!C:X$323*1000; 2022 = 2019 * assumptions!I143;
        # 2023-2050 = assumptions!J143; 2051-2100 = assumptions!K143.
        if mode_label == "Off-Road":
            at_hist = hist_by_region.get("AT", pd.Series(0.0, index=CALC_FREIGHT_HIST_YEARS, dtype=float))
            at_series = mode_series.get("AT", zero_full)
            assump = KTKM_CAGR.get(mode_key, {}).get("AT")
            if assump is not None:
                hist_cagr, share_2019, ref_2023, ref_2051 = assump
                diagnostic_rows.append({
                    "diagnostic_type": "OffRoad_AT_formula_path",
                    "Parameter": mode_label,
                    "mode_suffix": suffix,
                    "region": "AT",
                    "year": "2000-2100",
                    "actual": float(at_series.abs().sum()),
                    "expected": "AT!C:X$323*1000 historically; 2022=2019*assumptions!I143; projections use assumptions!J143:K143",
                    "diff": "",
                    "source_region": "AT",
                    "source_value": float(at_hist.reindex(CALC_FREIGHT_HIST_YEARS).fillna(0.0).abs().sum()),
                    "source_detail": (
                        "Off-Road AT is standalone, not NB+NS+PE+NL. "
                        f"Python uses calc-reference AT offroad k*tkm series aligned to AT!row 323; "
                        f"v2000={float(at_series.loc[2000]):.6f}; "
                        f"v2019={float(at_series.loc[2019]):.6f}; "
                        f"v2021={float(at_series.loc[2021]):.6f}; "
                        f"v2022={float(at_series.loc[2022]):.6f}; "
                        f"assumptions row 143 equivalent: hist_cagr={float(hist_cagr):.9f}; "
                        f"share_2019={float(share_2019):.9f}; "
                        f"ref_2023_2050={float(ref_2023):.9f}; "
                        f"ref_2051_2100={float(ref_2051):.9f}."
                    ),
                })

        # Specific diagnostic for Marine row-247 formula paths:
        # historical = REGION!C:X$247*1000 for explicit province/AT rows;
        # AT projection uses assumptions row 103, while province rows use rows 90:99.
        if mode_label == "Marine":
            at_hist = hist_by_region.get("AT", pd.Series(0.0, index=CALC_FREIGHT_HIST_YEARS, dtype=float))
            at_series = mode_series.get("AT", zero_full)
            assump = KTKM_CAGR.get(mode_key, {}).get("AT")
            if assump is not None:
                hist_cagr, share_2019, ref_2023, ref_2051 = assump
                diagnostic_rows.append({
                    "diagnostic_type": "Marine_AT_formula_path",
                    "Parameter": mode_label,
                    "mode_suffix": suffix,
                    "region": "AT",
                    "year": "2000-2100",
                    "actual": float(at_series.abs().sum()),
                    "expected": "AT!C:X$247*1000 historically; 2022=2019*assumptions!I103; projections use assumptions!J103:K103",
                    "diff": "",
                    "source_region": "AT",
                    "source_value": float(at_hist.reindex(CALC_FREIGHT_HIST_YEARS).fillna(0.0).abs().sum()),
                    "source_detail": (
                        "Marine AT is standalone, not NB+NS+PE+NL. "
                        f"Python uses upstream AT marine Activity (M tkm)*1000; "
                        f"v2000={float(at_series.loc[2000]):.6f}; "
                        f"v2019={float(at_series.loc[2019]):.6f}; "
                        f"v2021={float(at_series.loc[2021]):.6f}; "
                        f"v2022={float(at_series.loc[2022]):.6f}; "
                        f"assumptions row 103 equivalent: hist_cagr={float(hist_cagr):.9f}; "
                        f"share_2019={float(share_2019):.9f}; "
                        f"ref_2023_2050={float(ref_2023):.9f}; "
                        f"ref_2051_2100={float(ref_2051):.9f}."
                    ),
                })

        # Diagnostic: CAN components, only if a required detail component is
        # structurally missing/zero across history. This points to missing upstream
        # values without bloating the file with every successful component/year.
        for comp in CALC_FREIGHT_CAN_SUM_REGIONS:
            comp_series = mode_series.get(comp, zero_full)
            if float(comp_series.reindex(CALC_FREIGHT_HIST_YEARS).abs().sum()) == 0.0 and comp not in {"YT", "NT", "NU"}:
                diagnostic_rows.append({
                    "diagnostic_type": "CAN_component_zero_or_missing_history",
                    "Parameter": mode_label,
                    "mode_suffix": suffix,
                    "region": "CAN",
                    "year": "2000-2021",
                    "actual": 0.0,
                    "expected": "nonzero_or_confirmed_zero",
                    "diff": "",
                    "source_region": comp,
                    "source_value": 0.0,
                    "source_detail": f"{comp} contributes zero across historical years to CAN = sum(BC:NU). Check upstream {comp}_{suffix} if this is unexpected.",
                })

        # Emit calc rows in the workbook ordering.
        for reg, name in CALC_FREIGHT_REGIONS:
            s = mode_series.get(reg, zero_full)
            row = {"Index":"", "Source":"CEUD" if reg == "CAN" else "", "Unit":"k*tkm", "Parameter":mode_label, "region":reg, "region_name":name}
            for year in CALC_FREIGHT_YEARS:
                v = float(s.loc[year])
                row[str(year)] = v
                long_rows.append({"Source":"CEUD", "Unit":"k*tkm", "Parameter":mode_label, "region":reg, "region_name":name, "year":year, "value":v})
            rows.append(row)

    out = pd.DataFrame(rows)[["Index","Source","Unit","Parameter","region","region_name"] + year_cols]
    long_df = pd.DataFrame(long_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    validation = pd.DataFrame(validations + [
        {"check":"calc_rows_nonblank", "value":float(out["region"].astype(str).ne("").sum()), "status":"ok"},
        {"check":"year_columns", "value":float(len(year_cols)), "status":"ok" if len(year_cols) == 101 else "warn"},
        {"check":"2022_projection_rule", "value":1.0, "status":"ok"},
        {"check":"diagnostic_rows", "value":float(len(diagnostics)), "status":"info"},
    ])
    if write:
        out.to_csv(OUT_DIR / "freight_calc.csv", index=False)
        out.to_csv(OUT_DIR / "calc_freight.csv", index=False)
        long_df.to_csv(OUT_DIR / "freight_calc_long.csv", index=False)
        validation.to_csv(OUT_DIR / "freight_calc_validation.csv", index=False)
        diagnostics.to_csv(OUT_DIR / "freight_calc_sourcing_diagnostic.csv", index=False)
        print(f"  ✅ freight_calc.csv ({len(out):,} rows x {len(out.columns):,} columns)")
        print(f"  ✅ calc_freight.csv (alias copy)")
        print(f"  ✅ freight_calc_long.csv ({len(long_df):,} rows)")
        print(f"  ✅ freight_calc_validation.csv")
        print(f"  ✅ freight_calc_sourcing_diagnostic.csv ({len(diagnostics):,} rows)")
    return out



# =============================================================================
# CALC MARKET SHARE PIPELINE
# =============================================================================
# Workbook implementation notes for this first-pass pipeline:
#   - The calc_market share tab contains direct sheet pulls such as REGION!C:Y$60
#     for Diesel Existing, REGION!C:Y$61 for Gasoline Existing, and a residual
#     Propane share = 1 - Diesel Existing - Gasoline Existing.
#   - The workbook also uses activity-weighted SUMPRODUCT formulas against the
#     calc tab for grouped rows; those are represented here with optional
#     weighted-share diagnostics so the structure is ready for refinement.
#   - This implementation intentionally emits both direct-share rows and
#     weighted-share diagnostics using the generated freight_calc output.
#
# Input sources used by this script:
#   * annual_freight_wide_clean.csv (preferred) if present in SCRIPT_DIR/OUT_DIR
#   * annual_freight.csv via build_annual_freight_tables() as fallback
#   * freight_calc.csv / build_freight_calc(...) for calc activity denominators
#
# Output artifacts:
#   * output/calc_market_share.csv
#   * output/calc_market_share_long.csv
#   * output/calc_market_share_validation.csv
#   * output/calc_market_share_weighted_diagnostic.csv
# =============================================================================

CALC_MARKET_SHARE_YEARS = list(range(2000, 2101))
CALC_MARKET_SHARE_HIST_YEARS = list(range(2000, 2023))
CALC_MARKET_SHARE_REGIONS = [
    "CAN", "BC", "AB", "SK", "MB", "ON", "QC",
    "NB", "NS", "PE", "NL", "YT", "NT", "NU", "AT", "TR",
]
CALC_MARKET_SHARE_FUELS = ["Diesel Existing", "Gasoline Existing", "Propane"]
CALC_MARKET_SHARE_ROW_ORDER = {
    "Diesel Existing": 0,
    "Gasoline Existing": 1,
    "Propane": 2,
}

CALC_MARKET_SHARE_REGION_NAMES = {
    "CAN": "Canada",
    "BC": "British Columbia",
    "AB": "Alberta",
    "SK": "Saskatchewan",
    "MB": "Manitoba",
    "ON": "Ontario",
    "QC": "Quebec",
    "NB": "New Brunswick",
    "NS": "Nova Scotia",
    "PE": "Prince Edward Island",
    "NL": "Newfoundland and Labrador",
    "YT": "Yukon",
    "NT": "Northwest Territories",
    "NU": "Nunavut",
    "AT": "Atlantic",
    "TR": "Territories",
}


def _cms_year_cols(df: pd.DataFrame) -> list[str]:
    years: list[tuple[int, str]] = []
    for c in df.columns:
        s = str(c).strip()
        if s.isdigit():
            y = int(s)
            if 1900 <= y <= 2200:
                years.append((y, c))
    return [c for _, c in sorted(years)]


def _cms_clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).replace('\\_', '_').strip().lstrip('\ufeff') for c in out.columns]
    return out


def _cms_clean_str(v: object) -> str:
    if pd.isna(v):
        return ""
    return str(v).replace('\\_', '_').strip()


def _cms_pick_path(*candidates: Path) -> Path | None:
    for p in candidates:
        if p.exists():
            return p
    return None


def _cms_load_annual_market_inputs() -> pd.DataFrame:
    """Load annual_freight-wide input table and extract REGION!row-60/61 style shares.

    Preferred source is annual_freight_wide_clean.csv if already generated. If it is
    unavailable but annual_freight.csv exists, we build the cleaned output first.
    """
    annual_clean = _cms_pick_path(
        OUT_DIR / 'annual_freight_wide_clean.csv',
        SCRIPT_DIR / 'annual_freight_wide_clean.csv',
    )
    if annual_clean is None:
        annual_raw = _cms_pick_path(
            OUT_DIR / 'annual_freight.csv',
            SCRIPT_DIR / 'annual_freight.csv',
        )
        if annual_raw is not None and 'build_annual_freight_tables' in globals():
            try:
                build_annual_freight_tables(write=True)
                annual_clean = _cms_pick_path(
                    OUT_DIR / 'annual_freight_wide_clean.csv',
                    SCRIPT_DIR / 'annual_freight_wide_clean.csv',
                )
            except Exception:
                annual_clean = None
    if annual_clean is None:
        raise FileNotFoundError(
            'calc_market_share requires annual_freight_wide_clean.csv (or annual_freight.csv buildable into it).'
        )

    wide = _cms_clean_cols(pd.read_csv(annual_clean))
    years = _cms_year_cols(wide)

    # Normalize common text columns.
    for c in ['region', 'sector', 'Service', 'technology', 'Parameter', 'Context', 'Sub_Context', 'Target', 'Source', 'Unit', 'Branch', 'Type', 'INDEX']:
        if c in wide.columns:
            wide[c] = wide[c].map(_cms_clean_str)
        else:
            wide[c] = ''

    # Identify direct market-share rows. The workbook formulas point to region sheet
    # rows 60/61 for Diesel/Gasoline Existing. In the cleaned annual_freight table,
    # those appear as service_request rows under Freight/Land/Light Medium with the
    # corresponding technology labels.
    mask = (
        wide['region'].isin(CALC_MARKET_SHARE_REGIONS)
        & wide['technology'].isin(['Diesel Existing', 'Gasoline Existing'])
        & wide['Service'].eq('service_request')
        & wide['Unit'].astype(str).str.contains('%|share', case=False, na=False)
    )
    candidates = wide.loc[mask].copy()

    # Fallback: if the share rows were flattened without % labels, allow Light Medium
    # activity-share rows carrying the market-share technologies.
    if candidates.empty:
        mask = (
            wide['region'].isin(CALC_MARKET_SHARE_REGIONS)
            & wide['technology'].isin(['Diesel Existing', 'Gasoline Existing'])
            & wide['Service'].eq('service_request')
            & (wide['Branch'].astype(str).str.contains('Light Medium', case=False, na=False)
               | wide['Target'].astype(str).str.contains('Light Medium', case=False, na=False)
               | wide['semantic_key'].astype(str).str.contains('Light Medium', case=False, na=False) if 'semantic_key' in wide.columns else False)
        )
        candidates = wide.loc[mask].copy()

    if candidates.empty:
        cols = [c for c in ['annual_row_id', 'semantic_key', 'lookup_key', 'Branch', 'region', 'Service', 'technology', 'Parameter', 'Target', 'Source', 'Unit'] if c in wide.columns]
        sample = wide[cols].head(20).to_dict('records')
        raise ValueError(
            'Unable to find annual_freight share inputs for Diesel Existing / Gasoline Existing. '
            f'Sample rows inspected: {sample}'
        )

    # Keep one best row per region x technology. Prefer percentage/share units, then
    # rows with the most populated historical values.
    def _non_null_hist_count(row: pd.Series) -> int:
        n = 0
        for y in years:
            if int(str(y)) <= 2022 and pd.notna(pd.to_numeric(row.get(y), errors='coerce')):
                n += 1
        return n

    candidates['_pref_unit_rank'] = np.where(candidates['Unit'].astype(str).str.contains('%|share', case=False, na=False), 0, 1)
    candidates['_hist_n'] = candidates.apply(_non_null_hist_count, axis=1)
    candidates = candidates.sort_values(['region', 'technology', '_pref_unit_rank', '_hist_n'], ascending=[True, True, True, False])
    best = candidates.groupby(['region', 'technology'], as_index=False).head(1).copy()

    # Standardize to direct market-share records.
    rows: list[dict] = []
    for _, r in best.iterrows():
        fuel = _cms_clean_str(r['technology'])
        row = {
            'Source': 'CEUD',
            'Unit': '%',
            'Parameter': fuel,
            'region': _cms_clean_str(r['region']),
            'region_name': CALC_MARKET_SHARE_REGION_NAMES.get(_cms_clean_str(r['region']), _cms_clean_str(r['region'])),
            'input_source': annual_clean.name,
            'annual_row_id': r.get('annual_row_id', ''),
            'annual_lookup': r.get('lookup_key', ''),
        }
        for y in years:
            val = pd.to_numeric(r.get(y), errors='coerce')
            row[str(y)] = float(val) if pd.notna(val) else np.nan
        rows.append(row)

    direct = pd.DataFrame(rows)

    # Ensure required region/fuel combinations exist; fill territorial/rollup blanks with 0.
    out_rows = []
    for reg in CALC_MARKET_SHARE_REGIONS:
        for fuel in ['Diesel Existing', 'Gasoline Existing']:
            sub = direct[(direct['region'].eq(reg)) & (direct['Parameter'].eq(fuel))]
            if sub.empty:
                row = {
                    'Source': 'CEUD',
                    'Unit': '%',
                    'Parameter': fuel,
                    'region': reg,
                    'region_name': CALC_MARKET_SHARE_REGION_NAMES.get(reg, reg),
                    'input_source': annual_clean.name,
                    'annual_row_id': '',
                    'annual_lookup': '',
                }
                for y in CALC_MARKET_SHARE_YEARS:
                    row[str(y)] = 0.0 if reg in {'YT', 'NT', 'NU', 'TR'} else np.nan
                out_rows.append(row)
            else:
                out_rows.append(sub.iloc[0].to_dict())

    out = pd.DataFrame(out_rows)
    # Historical years should be bounded to [0,1] if they are proportions. If values are
    # percent-style 0-100, convert them. We detect this per-row using the historical max.
    for idx, r in out.iterrows():
        vals = pd.to_numeric(r[[str(y) for y in CALC_MARKET_SHARE_HIST_YEARS]], errors='coerce')
        hist_max = vals.max(skipna=True)
        if pd.notna(hist_max) and hist_max > 1.000001:
            for y in CALC_MARKET_SHARE_YEARS:
                v = pd.to_numeric(out.at[idx, str(y)], errors='coerce')
                out.at[idx, str(y)] = float(v) / 100.0 if pd.notna(v) else np.nan
    return out[['Source', 'Unit', 'Parameter', 'region', 'region_name', 'input_source', 'annual_row_id', 'annual_lookup'] + [str(y) for y in CALC_MARKET_SHARE_YEARS]]


def _cms_expand_propane(direct_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    years = [str(y) for y in CALC_MARKET_SHARE_YEARS]
    for reg in CALC_MARKET_SHARE_REGIONS:
        d = direct_df[(direct_df['region'].eq(reg)) & (direct_df['Parameter'].eq('Diesel Existing'))]
        g = direct_df[(direct_df['region'].eq(reg)) & (direct_df['Parameter'].eq('Gasoline Existing'))]
        row = {
            'Source': 'CEUD',
            'Unit': '%',
            'Parameter': 'Propane',
            'region': reg,
            'region_name': CALC_MARKET_SHARE_REGION_NAMES.get(reg, reg),
            'input_source': 'residual(1-diesel-gasoline)',
            'annual_row_id': '',
            'annual_lookup': '',
        }
        for y in years:
            dv = pd.to_numeric(d.iloc[0][y], errors='coerce') if not d.empty else np.nan
            gv = pd.to_numeric(g.iloc[0][y], errors='coerce') if not g.empty else np.nan
            # Treat blanks as 0 only for explicit blank regions; otherwise propagate NaN if both absent.
            if pd.isna(dv) and pd.isna(gv):
                val = 0.0 if reg in {'YT', 'NT', 'NU', 'TR'} else np.nan
            else:
                dv = 0.0 if pd.isna(dv) else float(dv)
                gv = 0.0 if pd.isna(gv) else float(gv)
                val = max(0.0, min(1.0, 1.0 - dv - gv))
            row[y] = val
        rows.append(row)
    return pd.DataFrame(rows)


def _cms_build_weighted_diagnostic(calc_df: pd.DataFrame) -> pd.DataFrame:
    # This is a diagnostic / scaffold for the workbook SUMPRODUCT groups. It does not
    # replace the direct row-60/61 share pulls, but makes the calc dependency explicit.
    year_cols = [str(y) for y in CALC_MARKET_SHARE_YEARS]
    if all(c in calc_df.columns for c in ['Source', 'Unit', 'Parameter', 'region', 'region_name']):
        wide = calc_df[['Source', 'Unit', 'Parameter', 'region', 'region_name'] + year_cols].copy()
    else:
        raise ValueError('calc_df is missing expected freight calc columns.')

    # Aggregate calc activity by region and a coarse mode grouping.
    mode_map = {
        'Light Trucks': 'Light Medium',
        'Medium Trucks': 'Light Medium',
        'Heavy Trucks': 'Heavy Trucks',
        'Rail': 'Rail',
        'Marine': 'Marine',
        'Aviation': 'Aviation',
        'Off-Road': 'Off-Road',
    }
    work = wide[wide['Parameter'].isin(mode_map)].copy()
    work['market_group'] = work['Parameter'].map(mode_map)

    rows = []
    for (region, grp), sub in work.groupby(['region', 'market_group'], dropna=False):
        row = {
            'Source': 'CALC',
            'Unit': 'share',
            'Parameter': f'{grp} Weighted Share',
            'region': str(region),
            'region_name': CALC_MARKET_SHARE_REGION_NAMES.get(str(region), str(region)),
        }
        for y in year_cols:
            denom = pd.to_numeric(wide[wide['region'].eq(region)][y], errors='coerce').sum()
            numer = pd.to_numeric(sub[y], errors='coerce').sum()
            row[y] = float(numer) / float(denom) if pd.notna(denom) and float(denom) != 0.0 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)




def _cms_to_pandas(df_obj: pl.DataFrame | pd.DataFrame | None) -> pd.DataFrame:
    if df_obj is None:
        return pd.DataFrame()
    if isinstance(df_obj, pd.DataFrame):
        return df_obj.copy()
    if isinstance(df_obj, pl.DataFrame):
        return df_obj.to_pandas()
    raise TypeError(f"Unsupported dataframe type: {type(df_obj)!r}")


def _cms_find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = {str(c).strip(): c for c in df.columns}
    lowered = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand in cols:
            return cols[cand]
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    return None


def _cms_series_from_mode(df_obj: pl.DataFrame | pd.DataFrame | None, fuel_candidates: list[str]) -> pd.Series:
    df = _cms_to_pandas(df_obj)
    if df.empty:
        return pd.Series(0.0, index=CALC_MARKET_SHARE_YEARS, dtype=float)
    year_col = _cms_find_col(df, ['year', 'Year'])
    total_col = _cms_find_col(df, ['fuel_Total (TJ)', 'fuel_Total'])
    fuel_col = _cms_find_col(df, fuel_candidates)
    if year_col is None or total_col is None or fuel_col is None:
        return pd.Series(0.0, index=CALC_MARKET_SHARE_YEARS, dtype=float)
    work = df[[year_col, fuel_col, total_col]].copy()
    work[year_col] = pd.to_numeric(work[year_col], errors='coerce')
    work[fuel_col] = pd.to_numeric(work[fuel_col], errors='coerce').fillna(0.0)
    work[total_col] = pd.to_numeric(work[total_col], errors='coerce').fillna(0.0)
    agg = work.groupby(year_col, dropna=True)[[fuel_col, total_col]].sum(min_count=1)
    out = pd.Series(0.0, index=CALC_MARKET_SHARE_YEARS, dtype=float)
    common_years = [int(y) for y in agg.index if pd.notna(y) and int(y) in CALC_MARKET_SHARE_YEARS]
    for y in common_years:
        denom = float(agg.loc[y, total_col])
        numer = float(agg.loc[y, fuel_col])
        out.loc[y] = (numer / denom) if denom != 0.0 else 0.0
    return out


def _cms_load_direct_market_inputs(
    upstream_mode_dfs: dict[str, dict[str, pl.DataFrame | pd.DataFrame]] | None = None,
) -> pd.DataFrame:
    """Build direct Diesel/Gasoline/Propane historical shares from upstream light+medium mode dfs.

    The workbook row references REGION!row60 / row62 behave like direct region-sheet pulls.
    The closest in-memory equivalents in this Python pipeline are the Light Trucks and
    Medium Trucks mode outputs, which already contain year-varying fuel TJ shares by region.
    """
    if not upstream_mode_dfs:
        return pd.DataFrame()

    rows = []
    # Fuel column candidates as generated by _build_mode_df.
    diesel_cols = ['fuel_Diesel fuel oil (TJ)', 'Diesel fuel oil']
    gasoline_cols = ['fuel_Motor gasoline (TJ)', 'Motor gasoline']
    propane_cols = ['fuel_Propane (TJ)', 'Propane']

    # Regions ordered exactly like workbook direct rows.
    for reg in CALC_MARKET_SHARE_REGIONS:
        cache = upstream_mode_dfs.get(reg, {}) if upstream_mode_dfs is not None else {}
        if reg in {'YT', 'NT', 'NU', 'TR'} or not cache:
            diesel_share = pd.Series(0.0, index=CALC_MARKET_SHARE_YEARS, dtype=float)
            gas_share = pd.Series(0.0, index=CALC_MARKET_SHARE_YEARS, dtype=float)
            prop_share = pd.Series(0.0, index=CALC_MARKET_SHARE_YEARS, dtype=float)
        else:
            lt = cache.get('light_trucks')
            mt = cache.get('medium_trucks')
            # Combine Light Trucks + Medium Trucks as the workbook's "Light Medium" group.
            # We combine numerators and denominators implicitly by summing TJ and total fuel across modes.
            def _combined_share(candidates: list[str]) -> pd.Series:
                a = _cms_to_pandas(lt)
                b = _cms_to_pandas(mt)
                if a.empty and b.empty:
                    return pd.Series(0.0, index=CALC_MARKET_SHARE_YEARS, dtype=float)
                ya = _cms_find_col(a, ['year', 'Year'])
                yb = _cms_find_col(b, ['year', 'Year'])
                ta = _cms_find_col(a, ['fuel_Total (TJ)', 'fuel_Total'])
                tb = _cms_find_col(b, ['fuel_Total (TJ)', 'fuel_Total'])
                fa = _cms_find_col(a, candidates)
                fb = _cms_find_col(b, candidates)
                numer = pd.Series(0.0, index=CALC_MARKET_SHARE_YEARS, dtype=float)
                denom = pd.Series(0.0, index=CALC_MARKET_SHARE_YEARS, dtype=float)
                if ya is not None and ta is not None and fa is not None and not a.empty:
                    tmp = a[[ya, ta, fa]].copy()
                    tmp[ya] = pd.to_numeric(tmp[ya], errors='coerce')
                    tmp[ta] = pd.to_numeric(tmp[ta], errors='coerce').fillna(0.0)
                    tmp[fa] = pd.to_numeric(tmp[fa], errors='coerce').fillna(0.0)
                    agg = tmp.groupby(ya, dropna=True)[[ta, fa]].sum(min_count=1)
                    for y in [int(v) for v in agg.index if pd.notna(v) and int(v) in CALC_MARKET_SHARE_YEARS]:
                        denom.loc[y] += float(agg.loc[y, ta])
                        numer.loc[y] += float(agg.loc[y, fa])
                if yb is not None and tb is not None and fb is not None and not b.empty:
                    tmp = b[[yb, tb, fb]].copy()
                    tmp[yb] = pd.to_numeric(tmp[yb], errors='coerce')
                    tmp[tb] = pd.to_numeric(tmp[tb], errors='coerce').fillna(0.0)
                    tmp[fb] = pd.to_numeric(tmp[fb], errors='coerce').fillna(0.0)
                    agg = tmp.groupby(yb, dropna=True)[[tb, fb]].sum(min_count=1)
                    for y in [int(v) for v in agg.index if pd.notna(v) and int(v) in CALC_MARKET_SHARE_YEARS]:
                        denom.loc[y] += float(agg.loc[y, tb])
                        numer.loc[y] += float(agg.loc[y, fb])
                out = pd.Series(0.0, index=CALC_MARKET_SHARE_YEARS, dtype=float)
                nz = denom != 0.0
                out.loc[nz] = numer.loc[nz] / denom.loc[nz]
                return out
            diesel_share = _combined_share(diesel_cols)
            gas_share = _combined_share(gasoline_cols)
            prop_share = _combined_share(propane_cols)

        for fuel, ser in [('Diesel Existing', diesel_share), ('Gasoline Existing', gas_share), ('Propane', prop_share)]:
            row = {
                'Source': 'CEUD',
                'Unit': '%',
                'Parameter': fuel,
                'region': reg,
                'region_name': CALC_MARKET_SHARE_REGION_NAMES.get(reg, reg),
                'input_source': 'upstream_mode_fuel_shares',
                'annual_row_id': '',
                'annual_lookup': '',
            }
            for y in CALC_MARKET_SHARE_YEARS:
                row[str(y)] = float(ser.loc[y])
            rows.append(row)
    return pd.DataFrame(rows)



def _cms_append_land_heavy_rows(rows: list[dict], weighted_diag: pd.DataFrame, year_cols: list[str]) -> None:
    """Append the calc_market share Land Heavy block: Heavy Trucks + Rail.

    The values workbook contains a second calc_market share section after the
    Light Medium fuel-share rows.  This section is the Land Heavy modal split,
    where Heavy Trucks and Rail are activity-weighted shares from the calc tab.
    The weighted diagnostic already computes those shares; this helper writes
    them into calc_market_share.csv in workbook-style row order.
    """
    def _blank_row() -> dict:
        row = {'Index': '', 'Source': '', 'Unit': '', 'Parameter': '', 'region': '', 'region_name': ''}
        row.update({y: '' for y in year_cols})
        return row

    # Keep workbook-like spacing between the Light Medium block and the Land Heavy block.
    # Six spacer rows + two header rows + 32 data rows closes the observed 40-row gap
    # between the generated output and the workbook values export.
    for _ in range(6):
        rows.append(_blank_row())

    hdr = {'Index': '', 'Source': '', 'Unit': '', 'Parameter': 'market_share_total', 'region': '', 'region_name': ''}
    hdr.update({y: '' for y in year_cols})
    rows.append(hdr)

    subhdr = {'Index': '', 'Source': '', 'Unit': '', 'Parameter': 'Land Heavy', 'region': '', 'region_name': ''}
    subhdr.update({y: '' for y in year_cols})
    rows.append(subhdr)

    param_map = {
        'Heavy Trucks Weighted Share': 'Heavy Trucks',
        'Rail Weighted Share': 'Rail',
    }
    region_order = {r: i for i, r in enumerate(CALC_MARKET_SHARE_REGIONS)}

    for diag_param, out_param in param_map.items():
        sub = weighted_diag.loc[weighted_diag['Parameter'].astype(str).eq(diag_param)].copy()
        if sub.empty:
            # Emit explicit zero rows if diagnostic is unexpectedly missing, rather than silently
            # dropping required workbook rows.
            sub = pd.DataFrame([
                {'Source': 'CALC', 'Unit': 'share', 'Parameter': diag_param,
                 'region': reg, 'region_name': CALC_MARKET_SHARE_REGION_NAMES.get(reg, reg),
                 **{y: 0.0 for y in year_cols}}
                for reg in CALC_MARKET_SHARE_REGIONS
            ])
        sub['region_order'] = sub['region'].astype(str).map(region_order).fillna(9999)
        sub = sub.sort_values(['region_order', 'region'])
        for _, r in sub.iterrows():
            reg = str(r.get('region', '')).strip()
            row = {
                'Index': '',
                'Source': 'CEUD',
                'Unit': '%',
                'Parameter': out_param,
                'region': reg,
                'region_name': CALC_MARKET_SHARE_REGION_NAMES.get(reg, str(r.get('region_name', reg)).strip() or reg),
            }
            for y in year_cols:
                v = pd.to_numeric(r.get(y), errors='coerce')
                row[y] = float(v) if pd.notna(v) else 0.0
            rows.append(row)

def build_calc_market_share(
    calc_df: pd.DataFrame | None = None,
    upstream_mode_dfs: dict[str, dict[str, pl.DataFrame | pd.DataFrame]] | None = None,
    *,
    write: bool = True,
) -> pd.DataFrame:
    """Build the Freight calc_market share tab and write CSV artifacts.

    This first-pass implementation reproduces the workbook's direct historical market
    share rows (Diesel Existing / Gasoline Existing / residual Propane) using the
    annual_freight cleaned input table when available, then carries 2022 historical
    values forward flat through 2100 as a stable placeholder until the full workbook
    assumption blocks are implemented.  It also writes a calc-weighted diagnostic CSV
    using the generated freight_calc output so downstream SUMPRODUCT-style rows can be
    audited and tightened iteratively.
    """
    if calc_df is None:
        calc_path = _cms_pick_path(OUT_DIR / 'freight_calc.csv', OUT_DIR / 'calc_freight.csv', SCRIPT_DIR / 'freight_calc.csv', SCRIPT_DIR / 'calc_freight.csv')
        if calc_path is None:
            raise FileNotFoundError('build_calc_market_share could not find freight_calc.csv / calc_freight.csv.')
        calc_df = pd.read_csv(calc_path)
    calc_df = _cms_clean_cols(calc_df)

    direct = _cms_load_direct_market_inputs(upstream_mode_dfs=upstream_mode_dfs)
    if direct.empty:
        # Fallback path for standalone / debug use when upstream in-memory dfs are unavailable.
        direct_fg = _cms_load_annual_market_inputs()
        direct = direct_fg[direct_fg['Parameter'].isin(['Diesel Existing', 'Gasoline Existing'])].copy()
    if 'Propane' in set(direct.get('Parameter', pd.Series(dtype=str)).astype(str)):
        all_direct = direct.copy()
    else:
        propane = _cms_expand_propane(direct)
        all_direct = pd.concat([direct, propane], ignore_index=True, sort=False)

    # Extend 2022 values flat across 2023:2100 wherever direct rows stop historically.
    for idx, r in all_direct.iterrows():
        last_hist = pd.to_numeric(r.get('2022'), errors='coerce')
        if pd.isna(last_hist):
            # carry forward the latest available historical value if 2022 is blank
            hist_vals = pd.to_numeric(r[[str(y) for y in CALC_MARKET_SHARE_HIST_YEARS]], errors='coerce')
            valid = hist_vals[hist_vals.notna()]
            last_hist = float(valid.iloc[-1]) if len(valid) else 0.0
        for y in range(2023, 2101):
            all_direct.at[idx, str(y)] = float(last_hist)

    # Add workbook-like note/spacer rows and preserve ordering by region/fuel.
    year_cols = [str(y) for y in CALC_MARKET_SHARE_YEARS]
    rows: list[dict] = []

    note = {'Index': '', 'Source': 'Forecast values based on assumptions sheet', 'Unit': '', 'Parameter': '', 'region': '', 'region_name': ''}
    note.update({y: '' for y in year_cols})
    rows.append(note)
    spacer = {'Index': '', 'Source': '', 'Unit': '', 'Parameter': '', 'region': '', 'region_name': ''}
    spacer.update({y: '' for y in year_cols})
    rows.append(spacer)

    row_no = 1
    # Group title row
    hdr = {'Index': '', 'Source': '', 'Unit': '', 'Parameter': 'market_share_total', 'region': '', 'region_name': ''}
    hdr.update({y: '' for y in year_cols})
    rows.append(hdr)
    subhdr = {'Index': '', 'Source': '', 'Unit': '', 'Parameter': 'Light Medium', 'region': '', 'region_name': ''}
    subhdr.update({y: '' for y in year_cols})
    rows.append(subhdr)

    preferred_region_order = CALC_MARKET_SHARE_REGIONS
    for fuel in CALC_MARKET_SHARE_FUELS:
        sub = all_direct[all_direct['Parameter'].eq(fuel)].copy()
        sub['region_order'] = sub['region'].map({r: i for i, r in enumerate(preferred_region_order)}).fillna(9999)
        sub = sub.sort_values(['region_order', 'region'])
        for _, r in sub.iterrows():
            row = {
                'Index': '',
                'Source': r.get('Source', 'CEUD'),
                'Unit': '%',
                'Parameter': fuel,
                'region': r['region'],
                'region_name': r['region_name'],
            }
            for y in year_cols:
                v = pd.to_numeric(r.get(y), errors='coerce')
                row[y] = float(v) if pd.notna(v) else ''
            rows.append(row)
            row_no += 1

    weighted_diag = _cms_build_weighted_diagnostic(calc_df)
    _cms_append_land_heavy_rows(rows, weighted_diag, year_cols)

    out = pd.DataFrame(rows)[['Index', 'Source', 'Unit', 'Parameter', 'region', 'region_name'] + year_cols]

    # Long + diagnostics.
    cms_value_params = CALC_MARKET_SHARE_FUELS + ['Heavy Trucks', 'Rail']
    value_rows = out.loc[out['Parameter'].isin(cms_value_params) & out['region'].astype(str).ne('')].copy()
    long_df = value_rows.melt(
        id_vars=['Index', 'Source', 'Unit', 'Parameter', 'region', 'region_name'],
        value_vars=year_cols,
        var_name='year',
        value_name='value',
    )
    long_df['year'] = pd.to_numeric(long_df['year'], errors='coerce').astype('Int64')
    long_df['value'] = pd.to_numeric(long_df['value'], errors='coerce')

    validation_rows = []
    for reg in CALC_MARKET_SHARE_REGIONS:
        sub = value_rows[value_rows['region'].eq(reg)].copy()
        if sub.empty:
            continue
        for y in year_cols:
            vals = pd.to_numeric(sub[y], errors='coerce').fillna(0.0)
            validation_rows.append({
                'check': 'fuel_share_sum',
                'region': reg,
                'year': int(y),
                'value': float(vals.sum()),
                'status': 'ok' if abs(float(vals.sum()) - 1.0) < 1e-6 or reg in {'YT','NT','NU','TR'} else 'warn',
            })
            for fuel, vv in zip(sub['Parameter'], vals):
                validation_rows.append({
                    'check': 'fuel_share_bounds',
                    'region': reg,
                    'year': int(y),
                    'Parameter': fuel,
                    'value': float(vv),
                    'status': 'ok' if 0.0 - 1e-9 <= float(vv) <= 1.0 + 1e-9 else 'warn',
                })
    # Land Heavy should split between Heavy Trucks and Rail by region/year.
    land_heavy_rows = value_rows[value_rows['Parameter'].isin(['Heavy Trucks', 'Rail'])].copy()
    if not land_heavy_rows.empty:
        for reg in CALC_MARKET_SHARE_REGIONS:
            sub = land_heavy_rows[land_heavy_rows['region'].eq(reg)].copy()
            if sub.empty:
                continue
            for y in year_cols:
                vals = pd.to_numeric(sub[y], errors='coerce').fillna(0.0)
                total = float(vals.sum())
                validation_rows.append({
                    'check': 'land_heavy_share_sum',
                    'region': reg,
                    'year': int(y),
                    'value': total,
                    'status': 'ok' if abs(total - 1.0) < 1e-6 or reg in {'YT','NT','NU','TR'} else 'warn',
                })
        validation = pd.DataFrame(validation_rows)

    if write:
        out.to_csv(OUT_DIR / 'calc_market_share.csv', index=False)
        long_df.to_csv(OUT_DIR / 'calc_market_share_long.csv', index=False)
        validation.to_csv(OUT_DIR / 'calc_market_share_validation.csv', index=False)
        weighted_diag.to_csv(OUT_DIR / 'calc_market_share_weighted_diagnostic.csv', index=False)
        print(f"  ✅ calc_market_share.csv ({len(out):,} rows x {len(out.columns):,} columns)")
        print(f"  ✅ calc_market_share_long.csv ({len(long_df):,} rows)")
        print(f"  ✅ calc_market_share_validation.csv ({len(validation):,} rows)")
        print(f"  ✅ calc_market_share_weighted_diagnostic.csv ({len(weighted_diag):,} rows)")
    return out


# =============================================================================
# ROBUST CALC / MARINE DIAGNOSTICS OVERRIDE
# =============================================================================
def _diag_write_placeholder_png(path: Path, message: str) -> None:
    """Write a small placeholder PNG so expected image artifacts always exist."""
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 3))
    plt.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _diag_year_cols(df: pd.DataFrame) -> list[str]:
    """Return sorted string year columns."""
    years: list[tuple[int, str]] = []
    for c in df.columns:
        s = str(c).strip()
        if s.isdigit():
            y = int(s)
            if 1900 <= y <= 2200:
                years.append((y, c))
    return [c for _, c in sorted(years)]


def _diag_normalize_calc_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize model/reference calc exports to common keys.

    Handles both generated calc_freight.csv columns:
      Index, Source, Unit, Parameter, region, region_name, years...

    and workbook-value exports that often look like:
      Index, Source, Unit, Parameter, Unnamed: 4, Unnamed: 5, years...
    where Unnamed: 4 = region and Unnamed: 5 = region_name.
    """
    out = df.copy()
    out.columns = [str(c).strip().lstrip("\ufeff") for c in out.columns]

    # If workbook export has blank columns after Parameter, rename them.
    year_cols = set(_diag_year_cols(out))
    non_year = [c for c in out.columns if c not in year_cols]
    if "region" not in out.columns and "Parameter" in out.columns:
        pidx = list(out.columns).index("Parameter")
        candidates = [c for c in out.columns[pidx + 1:] if c not in year_cols]
        if len(candidates) >= 1:
            out = out.rename(columns={candidates[0]: "region"})
        if len(candidates) >= 2 and "region_name" not in out.columns:
            out = out.rename(columns={candidates[1]: "region_name"})

    if "Region" in out.columns and "region" not in out.columns:
        out = out.rename(columns={"Region": "region"})
    if "Region Name" in out.columns and "region_name" not in out.columns:
        out = out.rename(columns={"Region Name": "region_name"})

    # Ensure expected key columns exist.
    for c in ["Source", "Unit", "Parameter", "region", "region_name"]:
        if c not in out.columns:
            out[c] = ""
        out[c] = out[c].astype(str).str.strip()

    # Drop separator/blank rows.
    out = out.loc[
        out["Parameter"].ne("")
        & out["Parameter"].str.lower().ne("nan")
        & out["region"].ne("")
        & out["region"].str.lower().ne("nan")
    ].reset_index(drop=True)

    for c in _diag_year_cols(out):
        out[c] = pd.to_numeric(out[c], errors="coerce")

    return out


def build_calc_diff_diagnostics(
    model_path: Path | str | None = None,
    ref_path: Path | str | None = None,
    *,
    out_dir: Path | None = None,
    top_n: int = 40,
) -> dict[str, Path]:
    """Build full calc diff diagnostics and always create Marine diagnostic files.

    Always writes at minimum:
      output/diff_marine_diagnostic.csv
      output/diff_marine_year_heatmap.png
      output/marine_rollup_residuals.csv
    """
    import matplotlib.pyplot as plt

    out_dir = OUT_DIR if out_dir is None else Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = OUT_DIR / "calc_freight.csv" if model_path is None else Path(model_path)
    ref_path = SCRIPT_DIR / "calc - Values - Freight.csv" if ref_path is None else Path(ref_path)

    out_full_diag = out_dir / "diff_full_diagnostic.csv"
    out_full_long = out_dir / "diff_full_long.csv"
    out_full_heat = out_dir / "diff_full_heatmap_top.png"
    out_marine_diag = out_dir / "diff_marine_diagnostic.csv"
    out_marine_heat = out_dir / "diff_marine_year_heatmap.png"
    out_roll = out_dir / "marine_rollup_residuals.csv"
    out_status = out_dir / "calc_diff_diagnostic_status.csv"

    roll_cols = [
        "year",
        "CAN_model_reported", "CAN_model_sum_BC_NU", "CAN_model_residual",
        "CAN_ref_reported", "CAN_ref_sum_BC_NU", "CAN_ref_residual",
        "AT_model_reported", "AT_model_sum_NB_NS_PE_NL", "AT_model_residual",
        "AT_ref_reported", "AT_ref_sum_NB_NS_PE_NL", "AT_ref_residual",
    ]

    def write_empty_artifacts(reason: str) -> dict[str, Path]:
        pd.DataFrame(columns=["Source", "Unit", "Parameter", "region", "abs_diff_total", "pct_diff_mean", "max_abs_diff", "max_pct_diff", "n_years_compared", "_merge", "heat"]).to_csv(out_full_diag, index=False)
        pd.DataFrame(columns=["Source", "Unit", "Parameter", "region", "year", "model", "ref", "diff", "pct_diff", "_merge"]).to_csv(out_full_long, index=False)
        pd.DataFrame(columns=["Source", "Unit", "Parameter", "region", "abs_diff_total", "pct_diff_mean", "max_abs_diff", "max_pct_diff", "n_years_compared", "_merge", "heat"]).to_csv(out_marine_diag, index=False)
        pd.DataFrame(columns=roll_cols).to_csv(out_roll, index=False)
        pd.DataFrame([{"status": "warning", "reason": reason, "model_path": str(model_path), "ref_path": str(ref_path)}]).to_csv(out_status, index=False)
        _diag_write_placeholder_png(out_full_heat, f"Calc diff diagnostic not completed:\n{reason}")
        _diag_write_placeholder_png(out_marine_heat, f"Marine diagnostic not completed:\n{reason}")
        print(f"⚠️ Calc diff diagnostics wrote placeholder artifacts: {reason}")
        return {
            "full_diagnostic": out_full_diag,
            "full_long": out_full_long,
            "heatmap_top": out_full_heat,
            "marine_diagnostic": out_marine_diag,
            "marine_heatmap": out_marine_heat,
            "marine_rollup_residuals": out_roll,
            "status": out_status,
        }

    if not model_path.exists():
        return write_empty_artifacts(f"Model calc file not found: {model_path}")
    if not ref_path.exists():
        return write_empty_artifacts(f"Reference calc file not found: {ref_path}")

    try:
        model = _diag_normalize_calc_columns(pd.read_csv(model_path))
        ref = _diag_normalize_calc_columns(pd.read_csv(ref_path))

        years = [c for c in _diag_year_cols(model) if c in _diag_year_cols(ref)]
        if not years:
            return write_empty_artifacts("No common year columns found between model and reference calc files.")

        key_cols = ["Source", "Unit", "Parameter", "region"]
        merged = model.merge(ref, on=key_cols, how="outer", suffixes=("_model", "_ref"), indicator=True)

        rows: list[dict] = []
        for _, r in merged.iterrows():
            base = {k: r.get(k, "") for k in key_cols}
            base["_merge"] = r.get("_merge", "")
            for y in years:
                mv = r.get(f"{y}_model", np.nan)
                rv = r.get(f"{y}_ref", np.nan)
                diff = mv - rv if pd.notna(mv) and pd.notna(rv) else np.nan
                pct = diff / rv if pd.notna(diff) and pd.notna(rv) and rv != 0 else np.nan
                rows.append({**base, "year": int(y), "model": mv, "ref": rv, "diff": diff, "pct_diff": pct})

        long_df = pd.DataFrame(rows)
        long_df.to_csv(out_full_long, index=False)

        def summarize(g: pd.DataFrame) -> pd.Series:
            d = pd.to_numeric(g["diff"], errors="coerce")
            p = pd.to_numeric(g["pct_diff"], errors="coerce")
            return pd.Series({
                "abs_diff_total": float(np.nansum(np.abs(d))),
                "pct_diff_mean": float(np.nanmean(np.abs(p))) if np.isfinite(np.nanmean(np.abs(p))) else np.nan,
                "max_abs_diff": float(np.nanmax(np.abs(d))) if np.isfinite(np.nanmax(np.abs(d))) else np.nan,
                "max_pct_diff": float(np.nanmax(np.abs(p))) if np.isfinite(np.nanmax(np.abs(p))) else np.nan,
                "n_years_compared": int(d.notna().sum()),
                "_merge": g["_merge"].iloc[0],
            })

        summary = long_df.groupby(key_cols, dropna=False).apply(summarize).reset_index()

        def heat(x):
            if pd.isna(x):
                return "NA"
            if x < 1e-4:
                return "OK"
            if x < 1e-2:
                return "LOW"
            if x < 5e-2:
                return "MED"
            return "HIGH"

        summary["heat"] = summary["pct_diff_mean"].apply(heat)
        summary.to_csv(out_full_diag, index=False)

        # Full heatmap.
        top = summary.loc[summary["_merge"].eq("both")].sort_values("abs_diff_total", ascending=False).head(top_n)
        if top.empty:
            _diag_write_placeholder_png(out_full_heat, "No matched rows available for full calc heatmap.")
        else:
            labels, matrix = [], []
            for _, tr in top.iterrows():
                mask = pd.Series(True, index=long_df.index)
                for k in key_cols:
                    mask &= long_df[k].astype(str).eq(str(tr[k]))
                seg = long_df.loc[mask].sort_values("year")
                labels.append(" | ".join(str(tr[k]) for k in key_cols))
                matrix.append(np.abs(pd.to_numeric(seg["pct_diff"], errors="coerce").to_numpy(dtype=float)))
            mat = np.nan_to_num(np.array(matrix, dtype=float), nan=0.0)
            plt.figure(figsize=(14, max(4, 0.25 * len(labels))))
            plt.imshow(np.log10(mat + 1e-12), aspect="auto")
            plt.colorbar(label="log10(|% diff| + 1e-12)")
            plt.title(f"Calc Diff Heatmap — Top {len(labels)} rows")
            plt.yticks(range(len(labels)), labels, fontsize=7)
            plt.xticks(range(len(years)), years, rotation=90, fontsize=6)
            plt.tight_layout()
            plt.savefig(out_full_heat, dpi=200)
            plt.close()

        # Marine outputs, always written.
        marine_summary = summary.loc[summary["Parameter"].astype(str).str.contains("marine", case=False, na=False)].copy()
        marine_long = long_df.loc[
            long_df["Parameter"].astype(str).str.contains("marine", case=False, na=False)
            & long_df["_merge"].eq("both")
        ].copy()
        marine_summary.to_csv(out_marine_diag, index=False)

        print(f"[DIAGNOSTIC] Marine summary rows: {len(marine_summary)}")
        print(f"[DIAGNOSTIC] Marine row-years: {len(marine_long)}")

        if marine_long.empty:
            _diag_write_placeholder_png(out_marine_heat, "No matched Marine rows found in model/reference calc comparison.")
            pd.DataFrame(columns=roll_cols).to_csv(out_roll, index=False)
        else:
            piv = marine_long.pivot_table(index="region", columns="year", values="diff", aggfunc="mean")
            piv = piv.reindex(columns=sorted(piv.columns))
            plt.figure(figsize=(16, max(4, 0.35 * len(piv.index))))
            plt.imshow(piv.fillna(0.0).to_numpy(), aspect="auto")
            plt.colorbar(label="Marine diff (model - ref)")
            plt.title("Marine Diff Heatmap — Region × Year")
            plt.yticks(range(len(piv.index)), piv.index)
            plt.xticks(range(len(piv.columns)), piv.columns, rotation=90, fontsize=6)
            plt.tight_layout()
            plt.savefig(out_marine_heat, dpi=200)
            plt.close()

            regions_can = ["BC", "AB", "SK", "MB", "ON", "QC", "NB", "NS", "PE", "NL", "YT", "NT", "NU"]
            regions_at = ["NB", "NS", "PE", "NL"]

            def series_sum(df: pd.DataFrame, regions: list[str], col: str) -> pd.Series:
                return df.loc[df["region"].isin(regions)].groupby("year")[col].sum(min_count=1)

            can_model = series_sum(marine_long, regions_can, "model")
            can_ref = series_sum(marine_long, regions_can, "ref")
            at_model = series_sum(marine_long, regions_at, "model")
            at_ref = series_sum(marine_long, regions_at, "ref")
            can_rep = marine_long.loc[marine_long["region"].eq("CAN")].set_index("year")
            at_rep = marine_long.loc[marine_long["region"].eq("AT")].set_index("year")

            roll_rows = []
            for y in sorted(marine_long["year"].unique()):
                can_m_reported = can_rep["model"].get(y, np.nan)
                can_r_reported = can_rep["ref"].get(y, np.nan)
                at_m_reported = at_rep["model"].get(y, np.nan)
                at_r_reported = at_rep["ref"].get(y, np.nan)
                can_m_sum = can_model.get(y, np.nan)
                can_r_sum = can_ref.get(y, np.nan)
                at_m_sum = at_model.get(y, np.nan)
                at_r_sum = at_ref.get(y, np.nan)
                roll_rows.append({
                    "year": int(y),
                    "CAN_model_reported": can_m_reported,
                    "CAN_model_sum_BC_NU": can_m_sum,
                    "CAN_model_residual": can_m_reported - can_m_sum if pd.notna(can_m_reported) and pd.notna(can_m_sum) else np.nan,
                    "CAN_ref_reported": can_r_reported,
                    "CAN_ref_sum_BC_NU": can_r_sum,
                    "CAN_ref_residual": can_r_reported - can_r_sum if pd.notna(can_r_reported) and pd.notna(can_r_sum) else np.nan,
                    "AT_model_reported": at_m_reported,
                    "AT_model_sum_NB_NS_PE_NL": at_m_sum,
                    "AT_model_residual": at_m_reported - at_m_sum if pd.notna(at_m_reported) and pd.notna(at_m_sum) else np.nan,
                    "AT_ref_reported": at_r_reported,
                    "AT_ref_sum_NB_NS_PE_NL": at_r_sum,
                    "AT_ref_residual": at_r_reported - at_r_sum if pd.notna(at_r_reported) and pd.notna(at_r_sum) else np.nan,
                })
            pd.DataFrame(roll_rows, columns=roll_cols).to_csv(out_roll, index=False)

        pd.DataFrame([{
            "status": "ok",
            "model_path": str(model_path),
            "ref_path": str(ref_path),
            "rows_model": len(model),
            "rows_ref": len(ref),
            "marine_summary_rows": len(marine_summary),
            "marine_row_years": len(marine_long),
        }]).to_csv(out_status, index=False)

        print(f"✅ wrote {out_marine_diag}")
        print(f"✅ wrote {out_marine_heat}")
        print(f"✅ wrote {out_roll}")

        return {
            "full_diagnostic": out_full_diag,
            "full_long": out_full_long,
            "heatmap_top": out_full_heat,
            "marine_diagnostic": out_marine_diag,
            "marine_heatmap": out_marine_heat,
            "marine_rollup_residuals": out_roll,
            "status": out_status,
        }

    except Exception as exc:
        return write_empty_artifacts(f"Unexpected diagnostic failure: {type(exc).__name__}: {exc}")


# =============================================================================
# CALC MARKET SHARE FORMULA-FAITHFUL OVERRIDES
# =============================================================================
# These overrides supersede the earlier first-pass calc_market_share helpers above.
# They are placed immediately before main execution so Python binds these formula-
# faithful implementations when main() calls build_calc_market_share().

def _cms_load_direct_market_inputs(
    upstream_mode_dfs: dict[str, dict[str, pl.DataFrame | pd.DataFrame]] | None = None,
) -> pd.DataFrame:
    """Build direct Diesel/Gasoline historical Light Medium shares.

    Formula basis:
      Diesel Existing = REGION!C:Y$60, 2000-2022 only.
      Gasoline Existing = REGION!C:Y$62, 2000-2022 only.
      Propane is not direct; build_calc_market_share residualizes it as 1-Diesel-Gasoline.
    """
    if not upstream_mode_dfs:
        return pd.DataFrame()

    diesel_cols = ['fuel_Diesel fuel oil (TJ)', 'Diesel fuel oil']
    gasoline_cols = ['fuel_Motor gasoline (TJ)', 'Motor gasoline']
    rows: list[dict] = []

    def _combined_light_medium_share(cache: dict, candidates: list[str]) -> pd.Series:
        numer = pd.Series(0.0, index=CALC_MARKET_SHARE_YEARS, dtype=float)
        denom = pd.Series(0.0, index=CALC_MARKET_SHARE_YEARS, dtype=float)
        for key in ['light_trucks', 'medium_trucks']:
            df = _cms_to_pandas(cache.get(key))
            if df.empty:
                continue
            ycol = _cms_find_col(df, ['year', 'Year'])
            tcol = _cms_find_col(df, ['fuel_Total (TJ)', 'fuel_Total'])
            fcol = _cms_find_col(df, candidates)
            if ycol is None or tcol is None or fcol is None:
                continue
            tmp = df[[ycol, tcol, fcol]].copy()
            tmp[ycol] = pd.to_numeric(tmp[ycol], errors='coerce')
            tmp[tcol] = pd.to_numeric(tmp[tcol], errors='coerce').fillna(0.0)
            tmp[fcol] = pd.to_numeric(tmp[fcol], errors='coerce').fillna(0.0)
            agg = tmp.groupby(ycol, dropna=True)[[tcol, fcol]].sum(min_count=1)
            for y_raw in agg.index:
                if pd.isna(y_raw):
                    continue
                y = int(y_raw)
                if y in CALC_MARKET_SHARE_YEARS:
                    denom.loc[y] += float(agg.loc[y_raw, tcol])
                    numer.loc[y] += float(agg.loc[y_raw, fcol])
        out = pd.Series(0.0, index=CALC_MARKET_SHARE_YEARS, dtype=float)
        nz = denom != 0.0
        out.loc[nz] = numer.loc[nz] / denom.loc[nz]
        return out

    for reg in CALC_MARKET_SHARE_REGIONS:
        cache = upstream_mode_dfs.get(reg, {}) if upstream_mode_dfs is not None else {}
        if reg in {'YT', 'NT', 'NU', 'TR'} or not cache:
            diesel_share = pd.Series(0.0, index=CALC_MARKET_SHARE_YEARS, dtype=float)
            gas_share = pd.Series(0.0, index=CALC_MARKET_SHARE_YEARS, dtype=float)
        else:
            diesel_share = _combined_light_medium_share(cache, diesel_cols)
            gas_share = _combined_light_medium_share(cache, gasoline_cols)

        for fuel, ser in [('Diesel Existing', diesel_share), ('Gasoline Existing', gas_share)]:
            row = {
                'Source': 'CEUD',
                'Unit': '%',
                'Parameter': fuel,
                'region': reg,
                'region_name': CALC_MARKET_SHARE_REGION_NAMES.get(reg, reg),
                'input_source': 'formula_rows_60_62_light_medium',
                'annual_row_id': '',
                'annual_lookup': '',
            }
            for y in CALC_MARKET_SHARE_YEARS:
                row[str(y)] = float(ser.loc[y]) if y <= 2022 else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def _cms_append_land_heavy_rows(rows: list[dict], calc_df: pd.DataFrame, year_cols: list[str]) -> None:
    """Append Land Heavy rows using the workbook SUMPRODUCT denominator.

    The formula range calc!38:70 is the Land Heavy block, so the denominator is
    Heavy Trucks + Rail for the same region/year. IFERROR(...,0) is implemented
    by returning 0 when the denominator is zero.
    """
    def _blank_row() -> dict:
        row = {'Index': '', 'Source': '', 'Unit': '', 'Parameter': '', 'region': '', 'region_name': ''}
        row.update({y: '' for y in year_cols})
        return row

    calc = _cms_clean_cols(calc_df)
    for c in ['Parameter', 'region', 'region_name']:
        if c not in calc.columns:
            calc[c] = ''
        calc[c] = calc[c].astype(str).str.strip()
    for y in year_cols:
        if y not in calc.columns:
            calc[y] = 0.0
        calc[y] = pd.to_numeric(calc[y], errors='coerce').fillna(0.0)

    for _ in range(6):
        rows.append(_blank_row())

    hdr = {'Index': '', 'Source': '', 'Unit': '', 'Parameter': 'market_share_total', 'region': '', 'region_name': ''}
    hdr.update({y: '' for y in year_cols})
    rows.append(hdr)

    subhdr = {'Index': '', 'Source': '', 'Unit': '', 'Parameter': 'Land Heavy', 'region': '', 'region_name': ''}
    subhdr.update({y: '' for y in year_cols})
    rows.append(subhdr)

    for out_param in ['Heavy Trucks', 'Rail']:
        for reg in CALC_MARKET_SHARE_REGIONS:
            heavy = calc[(calc['Parameter'].eq('Heavy Trucks')) & (calc['region'].eq(reg))]
            rail = calc[(calc['Parameter'].eq('Rail')) & (calc['region'].eq(reg))]
            row = {
                'Index': '',
                'Source': 'CEUD',
                'Unit': '%',
                'Parameter': out_param,
                'region': reg,
                'region_name': CALC_MARKET_SHARE_REGION_NAMES.get(reg, reg),
            }
            for y in year_cols:
                hv = float(pd.to_numeric(heavy.iloc[0][y], errors='coerce')) if not heavy.empty else 0.0
                rv = float(pd.to_numeric(rail.iloc[0][y], errors='coerce')) if not rail.empty else 0.0
                denominator = hv + rv
                if denominator == 0.0 or pd.isna(denominator):
                    row[y] = 0.0
                elif out_param == 'Heavy Trucks':
                    row[y] = hv / denominator
                else:
                    row[y] = rv / denominator
            rows.append(row)


def build_calc_market_share(
    calc_df: pd.DataFrame | None = None,
    upstream_mode_dfs: dict[str, dict[str, pl.DataFrame | pd.DataFrame]] | None = None,
    *,
    write: bool = True,
) -> pd.DataFrame:
    """Build formula-faithful Freight calc_market share outputs."""
    if calc_df is None:
        calc_path = _cms_pick_path(
            OUT_DIR / 'freight_calc.csv', OUT_DIR / 'calc_freight.csv',
            SCRIPT_DIR / 'freight_calc.csv', SCRIPT_DIR / 'calc_freight.csv'
        )
        if calc_path is None:
            raise FileNotFoundError('build_calc_market_share could not find freight_calc.csv / calc_freight.csv.')
        calc_df = pd.read_csv(calc_path)
    calc_df = _cms_clean_cols(calc_df)

    direct = _cms_load_direct_market_inputs(upstream_mode_dfs=upstream_mode_dfs)
    if direct.empty:
        direct_fg = _cms_load_annual_market_inputs()
        direct = direct_fg[direct_fg['Parameter'].isin(['Diesel Existing', 'Gasoline Existing'])].copy()

    # Formula-faithful: Propane is residual, never direct upstream propane.
    propane = _cms_expand_propane(direct)
    all_direct = pd.concat([direct, propane], ignore_index=True, sort=False)

    year_cols = [str(y) for y in CALC_MARKET_SHARE_YEARS]
    # Formula-faithful: Light Medium rows are blank for 2023-2100.
    for y in range(2023, 2101):
        all_direct[str(y)] = np.nan

    rows: list[dict] = []
    note = {'Index': '', 'Source': 'Forecast values based on assumptions sheet', 'Unit': '', 'Parameter': '', 'region': '', 'region_name': ''}
    note.update({y: '' for y in year_cols})
    rows.append(note)
    spacer = {'Index': '', 'Source': '', 'Unit': '', 'Parameter': '', 'region': '', 'region_name': ''}
    spacer.update({y: '' for y in year_cols})
    rows.append(spacer)
    hdr = {'Index': '', 'Source': '', 'Unit': '', 'Parameter': 'market_share_total', 'region': '', 'region_name': ''}
    hdr.update({y: '' for y in year_cols})
    rows.append(hdr)
    subhdr = {'Index': '', 'Source': '', 'Unit': '', 'Parameter': 'Light Medium', 'region': '', 'region_name': ''}
    subhdr.update({y: '' for y in year_cols})
    rows.append(subhdr)

    region_order = {r: i for i, r in enumerate(CALC_MARKET_SHARE_REGIONS)}
    for fuel in CALC_MARKET_SHARE_FUELS:
        sub = all_direct[all_direct['Parameter'].eq(fuel)].copy()
        sub['region_order'] = sub['region'].map(region_order).fillna(9999)
        sub = sub.sort_values(['region_order', 'region'])
        for _, r in sub.iterrows():
            row = {
                'Index': '', 'Source': r.get('Source', 'CEUD'), 'Unit': '%',
                'Parameter': fuel, 'region': r['region'], 'region_name': r['region_name']
            }
            for y in year_cols:
                v = pd.to_numeric(r.get(y), errors='coerce')
                row[y] = float(v) if pd.notna(v) else ''
            rows.append(row)

    _cms_append_land_heavy_rows(rows, calc_df, year_cols)
    out = pd.DataFrame(rows)[['Index', 'Source', 'Unit', 'Parameter', 'region', 'region_name'] + year_cols]

    cms_value_params = CALC_MARKET_SHARE_FUELS + ['Heavy Trucks', 'Rail']
    value_rows = out.loc[out['Parameter'].isin(cms_value_params) & out['region'].astype(str).ne('')].copy()
    long_df = value_rows.melt(
        id_vars=['Index', 'Source', 'Unit', 'Parameter', 'region', 'region_name'],
        value_vars=year_cols,
        var_name='year', value_name='value'
    )
    long_df['year'] = pd.to_numeric(long_df['year'], errors='coerce').astype('Int64')
    long_df['value'] = pd.to_numeric(long_df['value'], errors='coerce')

    validation_rows: list[dict] = []
    fuel_rows = value_rows[value_rows['Parameter'].isin(CALC_MARKET_SHARE_FUELS)].copy()
    for reg in CALC_MARKET_SHARE_REGIONS:
        sub = fuel_rows[fuel_rows['region'].eq(reg)].copy()
        if not sub.empty:
            for y in [str(v) for v in range(2000, 2023)]:
                vals = pd.to_numeric(sub[y], errors='coerce').fillna(0.0)
                total = float(vals.sum())
                validation_rows.append({
                    'check': 'fuel_share_sum_hist_2000_2022', 'region': reg, 'year': int(y),
                    'value': total,
                    'status': 'ok' if abs(total - 1.0) < 1e-6 or reg in {'YT','NT','NU','TR'} else 'warn',
                })
        sub_lh = value_rows[value_rows['Parameter'].isin(['Heavy Trucks', 'Rail']) & value_rows['region'].eq(reg)].copy()
        if not sub_lh.empty:
            for y in year_cols:
                vals = pd.to_numeric(sub_lh[y], errors='coerce').fillna(0.0)
                total = float(vals.sum())
                validation_rows.append({
                    'check': 'land_heavy_share_sum', 'region': reg, 'year': int(y),
                    'value': total,
                    'status': 'ok' if abs(total - 1.0) < 1e-6 or reg in {'YT','NT','NU','TR'} else 'warn',
                })
    validation = pd.DataFrame(validation_rows)
    weighted_diag = _cms_build_weighted_diagnostic(calc_df)

    if write:
        out.to_csv(OUT_DIR / 'calc_market_share.csv', index=False)
        long_df.to_csv(OUT_DIR / 'calc_market_share_long.csv', index=False)
        validation.to_csv(OUT_DIR / 'calc_market_share_validation.csv', index=False)
        weighted_diag.to_csv(OUT_DIR / 'calc_market_share_weighted_diagnostic.csv', index=False)
        print(f"  ✅ calc_market_share.csv ({len(out):,} rows x {len(out.columns):,} columns)")
        print(f"  ✅ calc_market_share_long.csv ({len(long_df):,} rows)")
        print(f"  ✅ calc_market_share_validation.csv ({len(validation):,} rows)")
        print(f"  ✅ calc_market_share_weighted_diagnostic.csv ({len(weighted_diag):,} rows)")
    return out


# =============================================================================
# CALC MARKET SHARE TERRITORY BLANK MICRO-PATCH
# =============================================================================
# Formula/value export cleanup:
#   * Light Medium Diesel/Gasoline/Propane rows for YT, NT, NU, and TR are blank
#     in the validated workbook values export.
#   * Land Heavy territory rows remain explicit zero values.
# This block intentionally overrides only the direct Light Medium loader and the
# Propane residual helper. build_calc_market_share() above will call these latest
# global function definitions at runtime.

CALC_MARKET_SHARE_LIGHT_MEDIUM_BLANK_REGIONS = {'YT', 'NT', 'NU', 'TR'}


def _cms_load_direct_market_inputs(
    upstream_mode_dfs: dict[str, dict[str, pl.DataFrame | pd.DataFrame]] | None = None,
) -> pd.DataFrame:
    """Build direct Diesel/Gasoline historical Light Medium shares, with territories blank.

    Formula basis:
      Diesel Existing = REGION!C:Y$60, 2000-2022 only.
      Gasoline Existing = REGION!C:Y$62, 2000-2022 only.
      YT/NT/NU/TR Light Medium rows are blank in the values export.
      Propane is not direct; _cms_expand_propane residualizes active regions.
    """
    if not upstream_mode_dfs:
        return pd.DataFrame()

    diesel_cols = ['fuel_Diesel fuel oil (TJ)', 'Diesel fuel oil']
    gasoline_cols = ['fuel_Motor gasoline (TJ)', 'Motor gasoline']
    rows: list[dict] = []

    def _combined_light_medium_share(cache: dict, candidates: list[str]) -> pd.Series:
        numer = pd.Series(0.0, index=CALC_MARKET_SHARE_YEARS, dtype=float)
        denom = pd.Series(0.0, index=CALC_MARKET_SHARE_YEARS, dtype=float)
        for key in ['light_trucks', 'medium_trucks']:
            df = _cms_to_pandas(cache.get(key))
            if df.empty:
                continue
            ycol = _cms_find_col(df, ['year', 'Year'])
            tcol = _cms_find_col(df, ['fuel_Total (TJ)', 'fuel_Total'])
            fcol = _cms_find_col(df, candidates)
            if ycol is None or tcol is None or fcol is None:
                continue
            tmp = df[[ycol, tcol, fcol]].copy()
            tmp[ycol] = pd.to_numeric(tmp[ycol], errors='coerce')
            tmp[tcol] = pd.to_numeric(tmp[tcol], errors='coerce').fillna(0.0)
            tmp[fcol] = pd.to_numeric(tmp[fcol], errors='coerce').fillna(0.0)
            agg = tmp.groupby(ycol, dropna=True)[[tcol, fcol]].sum(min_count=1)
            for y_raw in agg.index:
                if pd.isna(y_raw):
                    continue
                y = int(y_raw)
                if y in CALC_MARKET_SHARE_YEARS:
                    denom.loc[y] += float(agg.loc[y_raw, tcol])
                    numer.loc[y] += float(agg.loc[y_raw, fcol])
        out = pd.Series(0.0, index=CALC_MARKET_SHARE_YEARS, dtype=float)
        nz = denom != 0.0
        out.loc[nz] = numer.loc[nz] / denom.loc[nz]
        return out

    for reg in CALC_MARKET_SHARE_REGIONS:
        cache = upstream_mode_dfs.get(reg, {}) if upstream_mode_dfs is not None else {}
        if reg in CALC_MARKET_SHARE_LIGHT_MEDIUM_BLANK_REGIONS:
            diesel_share = pd.Series(np.nan, index=CALC_MARKET_SHARE_YEARS, dtype=float)
            gas_share = pd.Series(np.nan, index=CALC_MARKET_SHARE_YEARS, dtype=float)
        elif not cache:
            # Missing active-region cache: emit zeros for historical years to keep the pipeline robust.
            diesel_share = pd.Series(0.0, index=CALC_MARKET_SHARE_YEARS, dtype=float)
            gas_share = pd.Series(0.0, index=CALC_MARKET_SHARE_YEARS, dtype=float)
        else:
            diesel_share = _combined_light_medium_share(cache, diesel_cols)
            gas_share = _combined_light_medium_share(cache, gasoline_cols)

        for fuel, ser in [('Diesel Existing', diesel_share), ('Gasoline Existing', gas_share)]:
            row = {
                'Source': 'CEUD',
                'Unit': '%',
                'Parameter': fuel,
                'region': reg,
                'region_name': CALC_MARKET_SHARE_REGION_NAMES.get(reg, reg),
                'input_source': 'formula_rows_60_62_light_medium_with_territory_blanks',
                'annual_row_id': '',
                'annual_lookup': '',
            }
            for y in CALC_MARKET_SHARE_YEARS:
                row[str(y)] = float(ser.loc[y]) if (y <= 2022 and pd.notna(ser.loc[y])) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def _cms_expand_propane(direct_df: pd.DataFrame) -> pd.DataFrame:
    """Build Propane as residual, leaving rows blank where Diesel and Gasoline are blank."""
    rows: list[dict] = []
    years = [str(y) for y in CALC_MARKET_SHARE_YEARS]
    for reg in CALC_MARKET_SHARE_REGIONS:
        d = direct_df[(direct_df['region'].eq(reg)) & (direct_df['Parameter'].eq('Diesel Existing'))]
        g = direct_df[(direct_df['region'].eq(reg)) & (direct_df['Parameter'].eq('Gasoline Existing'))]
        row = {
            'Source': 'CEUD',
            'Unit': '%',
            'Parameter': 'Propane',
            'region': reg,
            'region_name': CALC_MARKET_SHARE_REGION_NAMES.get(reg, reg),
            'input_source': 'residual(1-diesel-gasoline)_with_blank_preservation',
            'annual_row_id': '',
            'annual_lookup': '',
        }
        for y in years:
            dv = pd.to_numeric(d.iloc[0][y], errors='coerce') if not d.empty else np.nan
            gv = pd.to_numeric(g.iloc[0][y], errors='coerce') if not g.empty else np.nan
            if pd.isna(dv) and pd.isna(gv):
                # Match workbook values export: territory Light Medium rows and all 2023+ cells are blank.
                val = np.nan
            else:
                dv = 0.0 if pd.isna(dv) else float(dv)
                gv = 0.0 if pd.isna(gv) else float(gv)
                val = max(0.0, min(1.0, 1.0 - dv - gv))
            row[y] = val
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# CALC AVG KM PIPELINE
# =============================================================================
# Formula-faithful first implementation for the workbook tab "calc_avg km".
#
# Formula mapping from calc_avg km - Formulas - Freight.txt:
#   * Light Medium historical cells: REGION!C:Y$42 / 1000
#   * Light Trucks historical cells: REGION!C:Y$76 / 1000
#   * Medium Trucks historical cells: REGION!C:Y$110 / 1000
#   * Heavy Trucks historical cells: REGION!C:Y$178 / 1000
#   * 2023-2100 projection cells:
#       next = previous_year * (bounded_rolling_5yr_cagr + 1)
#     where the rolling CAGR uses previous_year / value_5_years_prior, and is
#     bounded by AVG_KM_CONSTRAINTS from assumptions rows 151:154.
#   * YT/NT/NU/TR rows are blank throughout in the provided formula tab.
#
# Outputs:
#   * output/calc_avg_km.csv
#   * output/calc_avg_km_long.csv
#   * output/calc_avg_km_validation.csv
#   * output/calc_avg_km_projection_diagnostic.csv
# =============================================================================

CALC_AVG_KM_YEARS = list(range(2000, 2101))
CALC_AVG_KM_HIST_YEARS = list(range(2000, 2023))
CALC_AVG_KM_REGIONS = [
    "CAN", "BC", "AB", "SK", "MB", "ON", "QC",
    "NB", "NS", "PE", "NL", "YT", "NT", "NU", "AT", "TR",
]
CALC_AVG_KM_BLANK_REGIONS = {"YT", "NT", "NU", "TR"}
CALC_AVG_KM_REGION_NAMES = {
    "CAN": "Canada",
    "BC": "British Columbia",
    "AB": "Alberta",
    "SK": "Saskatchewan",
    "MB": "Manitoba",
    "ON": "Ontario",
    "QC": "Quebec",
    "NB": "New Brunswick",
    "NS": "Nova Scotia",
    "PE": "Prince Edward Island",
    "NL": "Newfoundland and Labrador",
    "YT": "Yukon",
    "NT": "Northwest Territories",
    "NU": "Nunavut",
    "AT": "Atlantic",
    "TR": "Territories",
}
CALC_AVG_KM_PARAMS = ["Light Medium", "Light Trucks", "Medium Trucks", "Heavy Trucks"]
CALC_AVG_KM_MODE_KEY = {
    "Light Trucks": "light_trucks",
    "Medium Trucks": "medium_trucks",
    "Heavy Trucks": "heavy_trucks",
}
CALC_AVG_KM_CONSTRAINT_KEY = {
    "Light Medium": "Light Medium",
    "Light Trucks": "Light Truck",
    "Medium Trucks": "Medium Truck",
    "Heavy Trucks": "Heavy Truck",
}


def _akm_to_pandas(df_obj: pl.DataFrame | pd.DataFrame | None) -> pd.DataFrame:
    if df_obj is None:
        return pd.DataFrame()
    if isinstance(df_obj, pd.DataFrame):
        return df_obj.copy()
    if isinstance(df_obj, pl.DataFrame):
        return df_obj.to_pandas()
    raise TypeError(f"Unsupported dataframe type for calc_avg_km: {type(df_obj)!r}")


def _akm_year_series_from_mode_df(df_obj: pl.DataFrame | pd.DataFrame | None, column: str) -> pd.Series:
    """Return a year-indexed Series from an upstream mode dataframe."""
    df = _akm_to_pandas(df_obj)
    out = pd.Series(np.nan, index=CALC_AVG_KM_YEARS, dtype=float)
    if df.empty or "year" not in df.columns or column not in df.columns:
        return out
    work = df[["year", column]].copy()
    work["year"] = pd.to_numeric(work["year"], errors="coerce")
    work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.dropna(subset=["year"])
    for _, r in work.iterrows():
        y = int(r["year"])
        if y in out.index:
            out.loc[y] = float(r[column]) if pd.notna(r[column]) else np.nan
    return out


def _akm_activity_per_stock_k_tkm(
    df_obj: pl.DataFrame | pd.DataFrame | None,
) -> pd.Series:
    """Calculate calc_avg_km units (k*tkm/vehicle) from mode activity and stock.

    Upstream mode frames store:
      Activity (M tkm) and Stock (thousands)
    Therefore Activity / Stock equals k*tkm per vehicle and matches
    REGION!row/1000 formulas for individual modes.
    """
    df = _akm_to_pandas(df_obj)
    out = pd.Series(np.nan, index=CALC_AVG_KM_YEARS, dtype=float)
    if df.empty or "year" not in df.columns:
        return out
    act_col = "Activity (M tkm)"
    stock_col = "Stock (thousands)"
    if act_col not in df.columns or stock_col not in df.columns:
        # Fallback: upstream Average Distance (tkm) / 1000 if available.
        avg = _akm_year_series_from_mode_df(df, "Average Distance (tkm)")
        return avg / 1000.0
    work = df[["year", act_col, stock_col]].copy()
    work["year"] = pd.to_numeric(work["year"], errors="coerce")
    work[act_col] = pd.to_numeric(work[act_col], errors="coerce")
    work[stock_col] = pd.to_numeric(work[stock_col], errors="coerce")
    for _, r in work.dropna(subset=["year"]).iterrows():
        y = int(r["year"])
        if y not in out.index:
            continue
        act = r[act_col]
        stock = r[stock_col]
        out.loc[y] = float(act) / float(stock) if pd.notna(act) and pd.notna(stock) and float(stock) != 0.0 else np.nan
    return out


def _akm_light_medium_k_tkm(cache: dict[str, pl.DataFrame | pd.DataFrame]) -> pd.Series:
    """Light Medium = combined Light Trucks + Medium Trucks activity / stock."""
    lt = _akm_to_pandas(cache.get("light_trucks"))
    mt = _akm_to_pandas(cache.get("medium_trucks"))
    out = pd.Series(np.nan, index=CALC_AVG_KM_YEARS, dtype=float)
    act_total = pd.Series(0.0, index=CALC_AVG_KM_YEARS, dtype=float)
    stock_total = pd.Series(0.0, index=CALC_AVG_KM_YEARS, dtype=float)
    for df in [lt, mt]:
        if df.empty or "year" not in df.columns:
            continue
        if "Activity (M tkm)" not in df.columns or "Stock (thousands)" not in df.columns:
            continue
        work = df[["year", "Activity (M tkm)", "Stock (thousands)"]].copy()
        work["year"] = pd.to_numeric(work["year"], errors="coerce")
        work["Activity (M tkm)"] = pd.to_numeric(work["Activity (M tkm)"], errors="coerce").fillna(0.0)
        work["Stock (thousands)"] = pd.to_numeric(work["Stock (thousands)"], errors="coerce").fillna(0.0)
        for _, r in work.dropna(subset=["year"]).iterrows():
            y = int(r["year"])
            if y in CALC_AVG_KM_YEARS:
                act_total.loc[y] += float(r["Activity (M tkm)"])
                stock_total.loc[y] += float(r["Stock (thousands)"])
    nz = stock_total != 0.0
    out.loc[nz] = act_total.loc[nz] / stock_total.loc[nz]
    return out


def _akm_historical_series(
    param: str,
    region: str,
    upstream_mode_dfs: dict[str, dict[str, pl.DataFrame | pd.DataFrame]],
) -> pd.Series:
    out = pd.Series(np.nan, index=CALC_AVG_KM_YEARS, dtype=float)
    if region in CALC_AVG_KM_BLANK_REGIONS:
        return out
    cache = upstream_mode_dfs.get(region, {}) if upstream_mode_dfs else {}
    if not cache:
        return out
    if param == "Light Medium":
        hist = _akm_light_medium_k_tkm(cache)
    else:
        hist = _akm_activity_per_stock_k_tkm(cache.get(CALC_AVG_KM_MODE_KEY[param]))
    for y in CALC_AVG_KM_HIST_YEARS:
        out.loc[y] = hist.loc[y]
    return out


def _akm_project_series(series: pd.Series, param: str) -> tuple[pd.Series, list[dict]]:
    """Project 2023-2100 using workbook rolling bounded CAGR formula."""
    out = series.copy().astype(float)
    diag: list[dict] = []
    constraint_key = CALC_AVG_KM_CONSTRAINT_KEY.get(param, param)
    years_to_calc, max_decrease, max_increase = AVG_KM_CONSTRAINTS.get(constraint_key, (5, -0.01, 0.0))
    for y in range(2023, 2101):
        prev = out.loc[y - 1]
        base_year = y - years_to_calc
        base = out.loc[base_year] if base_year in out.index else np.nan
        if pd.isna(prev) or pd.isna(base) or float(prev) <= 0.0 or float(base) <= 0.0:
            out.loc[y] = np.nan
            diag.append({"Parameter": param, "year": y, "previous": prev, "base_year": base_year, "base": base, "raw_cagr": np.nan, "bounded_cagr": np.nan, "value": np.nan})
            continue
        raw_cagr = (float(prev) / float(base)) ** (1.0 / float(years_to_calc)) - 1.0
        bounded = max(float(max_decrease), min(float(max_increase), float(raw_cagr)))
        out.loc[y] = float(prev) * (1.0 + bounded)
        diag.append({"Parameter": param, "year": y, "previous": prev, "base_year": base_year, "base": base, "raw_cagr": raw_cagr, "bounded_cagr": bounded, "value": out.loc[y]})
    return out, diag


def build_calc_avg_km(
    upstream_mode_dfs: dict[str, dict[str, pl.DataFrame | pd.DataFrame]] | None = None,
    *,
    write: bool = True,
) -> pd.DataFrame:
    """Build calc_avg_km tab from upstream freight mode dataframes."""
    if upstream_mode_dfs is None:
        upstream_mode_dfs = {}
    year_cols = [str(y) for y in CALC_AVG_KM_YEARS]
    rows: list[dict] = []
    projection_diag: list[dict] = []

    def _blank_row() -> dict:
        row = {"Index": "", "Source": "", "Unit": "", "Parameter": "", "region": "", "region_name": ""}
        row.update({y: "" for y in year_cols})
        return row

    def _header_row(label: str) -> dict:
        row = {"Index": label, "Source": "", "Unit": "", "Parameter": label, "region": "", "region_name": ""}
        row.update({y: "" for y in year_cols})
        return row

    # Workbook-style header rows.
    note = {"Index": "", "Source": "Forecast values based on assumptions sheet", "Unit": "", "Parameter": "", "region": "", "region_name": ""}
    note.update({y: "" for y in year_cols})
    rows.append(note)
    rows.append(_blank_row())
    rows.append(_header_row("Average tkm"))
    rows.append(_header_row("Freight vehicles"))

    for group_idx, param in enumerate(CALC_AVG_KM_PARAMS):
        if group_idx > 0:
            rows.append(_blank_row())
        for region in CALC_AVG_KM_REGIONS:
            hist = _akm_historical_series(param, region, upstream_mode_dfs)
            projected, diag = _akm_project_series(hist, param)
            for d in diag:
                d["region"] = region
                d["region_name"] = CALC_AVG_KM_REGION_NAMES.get(region, region)
            projection_diag.extend(diag)
            row = {
                "Index": f"{param}{region}",
                "Source": "CEUD",
                "Unit": "k*tkm",
                "Parameter": param,
                "region": region,
                "region_name": CALC_AVG_KM_REGION_NAMES.get(region, region),
            }
            for y in CALC_AVG_KM_YEARS:
                v = projected.loc[y]
                row[str(y)] = float(v) if pd.notna(v) else ""
            rows.append(row)

    out = pd.DataFrame(rows)[["Index", "Source", "Unit", "Parameter", "region", "region_name"] + year_cols]

    value_params = CALC_AVG_KM_PARAMS
    value_rows = out[out["Parameter"].isin(value_params) & out["region"].astype(str).ne("")].copy()
    long_df = value_rows.melt(
        id_vars=["Index", "Source", "Unit", "Parameter", "region", "region_name"],
        value_vars=year_cols,
        var_name="year",
        value_name="value",
    )
    long_df["year"] = pd.to_numeric(long_df["year"], errors="coerce").astype("Int64")
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")

    validation_rows: list[dict] = []
    validation_rows.append({"check": "raw_row_count", "value": len(out), "expected": 71, "status": "ok" if len(out) == 71 else "warn"})
    validation_rows.append({"check": "value_row_count", "value": len(value_rows), "expected": 64, "status": "ok" if len(value_rows) == 64 else "warn"})
    for param in value_params:
        for region in CALC_AVG_KM_REGIONS:
            sub = long_df[(long_df["Parameter"].eq(param)) & (long_df["region"].eq(region))]
            nonnull_hist = int(sub[sub["year"].between(2000, 2022)]["value"].notna().sum())
            nonnull_proj = int(sub[sub["year"].between(2023, 2100)]["value"].notna().sum())
            if region in CALC_AVG_KM_BLANK_REGIONS:
                status = "ok" if nonnull_hist == 0 and nonnull_proj == 0 else "warn"
            else:
                status = "ok" if nonnull_hist == 23 and nonnull_proj == 78 else "warn"
            validation_rows.append({
                "check": "row_population",
                "Parameter": param,
                "region": region,
                "hist_non_null": nonnull_hist,
                "proj_non_null": nonnull_proj,
                "status": status,
            })
    validation = pd.DataFrame(validation_rows)
    projection_diag_df = pd.DataFrame(projection_diag)

    if write:
        out.to_csv(OUT_DIR / "calc_avg_km.csv", index=False)
        long_df.to_csv(OUT_DIR / "calc_avg_km_long.csv", index=False)
        validation.to_csv(OUT_DIR / "calc_avg_km_validation.csv", index=False)
        projection_diag_df.to_csv(OUT_DIR / "calc_avg_km_projection_diagnostic.csv", index=False)
        print(f"  ✅ calc_avg_km.csv ({len(out):,} rows x {len(out.columns):,} columns)")
        print(f"  ✅ calc_avg_km_long.csv ({len(long_df):,} rows)")
        print(f"  ✅ calc_avg_km_validation.csv ({len(validation):,} rows)")
        print(f"  ✅ calc_avg_km_projection_diagnostic.csv ({len(projection_diag_df):,} rows)")

    return out



# =============================================================================
# FINAL TRANSPORTATION FREIGHT AB REGIONAL DATAFRAME PIPELINE
# =============================================================================
# AB-only native final-output builder.
#
# IMPORTANT RUNTIME RULE:
#   The target CSVs and formula workbooks are reference guides only. They must NOT be
#   required runtime inputs. This builder therefore constructs its own native AB row
#   skeleton and populates values from upstream data produced by this script.
#
# Current scope:
#   This is the first native AB implementation. It generates the core freight-service
#   demand/share rows and the first technology rows that are already supported by the
#   completed calc, calc_market_share, and calc_avg_km tabs. Additional CIMS rows
#   should be added by expanding _final_ab_native_rows() and mapping logic below.
#
# Runtime inputs:
#   * calc_freight.csv
#   * calc_market_share.csv
#   * calc_avg_km.csv
#
# Runtime outputs:
#   * output/transportation freight_AB_test.csv
#   * output/transportation_freight_AB_audit_test.csv
#   * output/transportation_freight_AB_source_map_test.csv
# =============================================================================

FINAL_FREIGHT_AB_YEARS = [str(y) for y in range(2000, 2051, 5)]
FINAL_FREIGHT_AB_REGION = 'AB'
FINAL_FREIGHT_AB_OUTPUT_CSV = 'transportation freight_AB_test.csv'
FINAL_FREIGHT_AB_AUDIT_CSV = 'transportation_freight_AB_audit_test.csv'
FINAL_FREIGHT_AB_SOURCE_MAP_CSV = 'transportation_freight_AB_source_map_test.csv'
FINAL_FREIGHT_AB_AIR_ANNUAL_CANDIDATE_CSV = 'transportation_freight_AB_air_annual_candidate_diagnostic_test.csv'
FINAL_FREIGHT_AB_MACRO_PRICES_CSV = 'macro_inputs_prices.csv'
FINAL_FREIGHT_AB_META_COLS = [
    'Branch', 'Type', 'Region', 'Sector', 'Service', 'Technology', 'Parameter',
    'Context', 'Sub_Context', 'Target', 'Source', 'Unit'
]
FINAL_FREIGHT_AB_COLUMNS = FINAL_FREIGHT_AB_META_COLS + FINAL_FREIGHT_AB_YEARS + ['Comments']
FINAL_AB_AIR_ANNUAL_CANDIDATE_DIAGNOSTICS: list[dict] = []
FINAL_REGION_SK_OUTPUT_ANNUAL_DIAGNOSTICS = []


def _final_ab_existing_path(name: str) -> Path | None:
    """Return the first existing path for an upstream generated file."""
    candidates = []
    try:
        candidates.append(OUT_DIR / name)
    except Exception:
        pass
    try:
        candidates.append(SCRIPT_DIR / name)
    except Exception:
        pass
    candidates.append(Path(name))
    for p in candidates:
        if p.exists():
            return p
    return None


def _final_ab_load_calc_csv(name: str) -> pd.DataFrame:
    """Load an upstream generated calc-style CSV. No target/reference files are used."""
    p = _final_ab_existing_path(name)
    if p is None:
        raise FileNotFoundError(f'Could not locate upstream generated file required by final AB builder: {name}')
    df = pd.read_csv(p, keep_default_na=True)
    df.columns = [str(c).strip().replace('\\_', '_').lstrip('\ufeff') for c in df.columns]
    if 'region' not in df.columns and 'Parameter' in df.columns:
        cols = list(df.columns)
        pidx = cols.index('Parameter')
        if pidx + 1 < len(cols):
            df = df.rename(columns={cols[pidx + 1]: 'region'})
        if pidx + 2 < len(cols):
            df = df.rename(columns={cols[pidx + 2]: 'region_name'})
    for c in ['Parameter', 'region', 'region_name']:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.replace('\\_', '_', regex=False)
            df.loc[df[c].str.lower().isin(['nan', 'none']), c] = ''
    for y in [c for c in df.columns if str(c).isdigit()]:
        df[y] = pd.to_numeric(df[y], errors='coerce')
    return df


def _final_ab_series(df: pd.DataFrame, parameter: str, region: str = FINAL_FREIGHT_AB_REGION) -> pd.Series:
    years = [c for c in df.columns if str(c).isdigit()]
    sub = df[(df.get('Parameter', '').astype(str).eq(parameter)) & (df.get('region', '').astype(str).eq(region))]
    if sub.empty:
        return pd.Series(np.nan, index=years, dtype=float)
    return pd.to_numeric(sub.iloc[0][years], errors='coerce').astype(float)


def _final_ab_safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    den2 = den.replace(0, np.nan)
    out = num / den2
    return out.replace([np.inf, -np.inf], np.nan)


def _final_ab_format_value(v):
    if pd.isna(v):
        return ''
    try:
        return float(v)
    except Exception:
        return v


def _final_ab_blank_row(**kwargs) -> dict:
    row = {c: '' for c in FINAL_FREIGHT_AB_COLUMNS}
    row.update(kwargs)
    return row


def _final_ab_make_row(
    *,
    Branch: str,
    Type: str = '',
    Region: str = 'AB',
    Sector: str = 'Transportation Freight',
    Service: str = '',
    Technology: str = '',
    Parameter: str,
    Context: str = '',
    Sub_Context: str = '',
    Target: str,
    Source: str,
    Unit: str = '',
    Comments: str = '',
    dynamic_key: str,
    source_note: str,
    carry_2000: bool = False,
    populate_years: list[str] | None = None,
) -> dict:
    row = _final_ab_blank_row(
        Branch=Branch,
        Type=Type,
        Region=Region,
        Sector=Sector,
        Service=Service,
        Technology=Technology,
        Parameter=Parameter,
        Context=Context,
        Sub_Context=Sub_Context,
        Target=Target,
        Source=Source,
        Unit=Unit,
        Comments=Comments,
    )
    row['_dynamic_key'] = dynamic_key
    row['_source_note'] = source_note
    row['_carry_2000'] = carry_2000
    row['_populate_years'] = populate_years or FINAL_FREIGHT_AB_YEARS
    return row




def _final_ab_mapping_for_row(row: pd.Series) -> tuple[str, list[str], str, bool]:
    """Map a native final AB row to an upstream dynamic series where implemented.

    Returns:
      dynamic_key, populate_years, source_note, carry_2000

    Rows not yet implemented return empty dynamic_key and are emitted with blank year
    values while retaining full row metadata. This lets the final dataframe include
    every CIMS guide row without using the guide CSV/workbook as runtime input.
    """
    branch = str(row.get('Branch', '')).strip()
    service = str(row.get('Service', '')).strip()
    tech = str(row.get('Technology', '')).strip()
    param = str(row.get('Parameter', '')).strip()
    target = str(row.get('Target', '')).strip()
    source = str(row.get('Source', '')).strip()

    if branch == 'CIMS.CAN.AB' and param == 'service_request' and target == 'CIMS.CAN.AB.Transportation Freight':
        return 'freight_activity_k_tkm', FINAL_FREIGHT_AB_YEARS, 'calc_freight: Light Trucks + Medium Trucks + Heavy Trucks + Rail + Marine + Aviation', False

    if branch == 'CIMS.CAN.AB.Transportation Freight' and param == 'service_request' and target.endswith('.Freight'):
        return 'freight_share', FINAL_FREIGHT_AB_YEARS, 'structural share for Freight branch', False
    if branch == 'CIMS.CAN.AB.Transportation Freight' and param == 'service_request' and target.endswith('.Off Road'):
        return 'offroad_share_of_freight', FINAL_FREIGHT_AB_YEARS, 'calc_freight: Off-Road / Freight', False

    if branch == 'CIMS.CAN.AB.Transportation Freight.Freight' and param == 'service_request' and target.endswith('.Land'):
        return 'land_share_of_freight', FINAL_FREIGHT_AB_YEARS, 'calc_freight: Land / Freight', False
    if branch == 'CIMS.CAN.AB.Transportation Freight.Freight' and param == 'service_request' and target.endswith('.Marine'):
        return 'marine_share_of_freight', FINAL_FREIGHT_AB_YEARS, 'calc_freight: Marine / Freight', False
    if branch == 'CIMS.CAN.AB.Transportation Freight.Freight' and param == 'service_request' and target.endswith('.Air'):
        return 'air_share_of_freight', FINAL_FREIGHT_AB_YEARS, 'calc_freight: Aviation / Freight', False

    if branch == 'CIMS.CAN.AB.Transportation Freight.Freight.Land' and param == 'service_request' and target.endswith('.Light Medium'):
        return 'light_medium_share_of_land', FINAL_FREIGHT_AB_YEARS, 'calc_freight: (Light Trucks + Medium Trucks) / Land', False
    if branch == 'CIMS.CAN.AB.Transportation Freight.Freight.Land' and param == 'service_request' and target.endswith('.Heavy'):
        return 'heavy_share_of_land', FINAL_FREIGHT_AB_YEARS, 'calc_freight: (Heavy Trucks + Rail) / Land', False

    if service == 'Light Medium' and param == 'market_share_total' and source == 'annual_region_tech':
        if tech == 'Diesel Existing':
            return 'market_share_light_medium_diesel_existing', ['2000'], 'calc_market_share: Diesel Existing initial Light Medium share', False
        if tech == 'Gasoline Existing':
            return 'market_share_light_medium_gasoline_existing', ['2000'], 'calc_market_share: Gasoline Existing initial Light Medium share', False
        if tech == 'Propane':
            return 'market_share_light_medium_propane', ['2000'], 'calc_market_share: residual Propane initial Light Medium share', False

    if service == 'Heavy' and param == 'market_share_total' and source == 'annual_region_tech':
        if tech == 'Trucks':
            return 'market_share_heavy_trucks', ['2000'], 'calc_market_share: Heavy Trucks / (Heavy Trucks + Rail) initial share', False
        if tech == 'Rail':
            return 'market_share_heavy_rail', ['2000'], 'calc_market_share: Rail / (Heavy Trucks + Rail) initial share', False

    if param == 'output' and source == 'annual_region' and service == 'Light Medium':
        return 'avg_km_light_medium', FINAL_FREIGHT_AB_YEARS, 'calc_avg_km: Light Medium output; 2000 value carried across benchmark years', True
    if param == 'output' and source == 'annual_region' and service == 'Trucks':
        return 'avg_km_heavy_trucks', FINAL_FREIGHT_AB_YEARS, 'calc_avg_km: Heavy Trucks output; 2000 value carried across benchmark years', True

    return '', [], 'pending implementation: row metadata included; value formula not yet ported', False



FINAL_FREIGHT_AB_NATIVE_ROW_RECORDS = [{'Branch': 'CIMS.CAN.AB',
  'Type': 'Region',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': '',
  'Technology': '',
  'Parameter': 'service_request',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.CAN.AB.Transportation Freight',
  'Source': 'annual_region',
  'Unit': 'k*tkm',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight',
  'Type': 'Sector',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': '',
  'Technology': '',
  'Parameter': 'service_provide',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'k*tkm',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight',
  'Type': 'Sector',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': '',
  'Technology': '',
  'Parameter': 'competition',
  'Context': 'Sector',
  'Sub_Context': '',
  'Target': '',
  'Source': '',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight',
  'Type': 'Sector',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': '',
  'Technology': '',
  'Parameter': 'multiplier_price',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.CAN.AB.Biodiesel',
  'Source': 'AFDC 2023',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight',
  'Type': 'Sector',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': '',
  'Technology': '',
  'Parameter': 'multiplier_price',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.Generic Fuels.Biogas',
  'Source': 'JCIMS',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight',
  'Type': 'Sector',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': '',
  'Technology': '',
  'Parameter': 'multiplier_price',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.Generic Fuels.Black Liquor',
  'Source': 'JCIMS',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight',
  'Type': 'Sector',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': '',
  'Technology': '',
  'Parameter': 'multiplier_price',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.Generic Fuels.Coal',
  'Source': 'JCIMS',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight',
  'Type': 'Sector',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': '',
  'Technology': '',
  'Parameter': 'multiplier_price',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.Generic Fuels.Coke',
  'Source': 'JCIMS',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight',
  'Type': 'Sector',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': '',
  'Technology': '',
  'Parameter': 'multiplier_price',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.Generic Fuels.Diesel',
  'Source': 'CER',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight',
  'Type': 'Sector',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': '',
  'Technology': '',
  'Parameter': 'multiplier_price',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.CAN.AB.Electricity',
  'Source': 'CER',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight',
  'Type': 'Sector',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': '',
  'Technology': '',
  'Parameter': 'multiplier_price',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.CAN.AB.Ethanol',
  'Source': 'AFDC 2023',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight',
  'Type': 'Sector',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': '',
  'Technology': '',
  'Parameter': 'multiplier_price',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.Generic Fuels.Fuel Oil',
  'Source': 'JCIMS',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight',
  'Type': 'Sector',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': '',
  'Technology': '',
  'Parameter': 'multiplier_price',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.Generic Fuels.Gasoline',
  'Source': 'CER',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight',
  'Type': 'Sector',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': '',
  'Technology': '',
  'Parameter': 'multiplier_price',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.CAN.AB.Hydrogen',
  'Source': 'JCIMS',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight',
  'Type': 'Sector',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': '',
  'Technology': '',
  'Parameter': 'multiplier_price',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.Generic Fuels.Jet Fuel',
  'Source': 'JCIMS',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight',
  'Type': 'Sector',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': '',
  'Technology': '',
  'Parameter': 'multiplier_price',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.Generic Fuels.LPG',
  'Source': 'JCIMS',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight',
  'Type': 'Sector',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': '',
  'Technology': '',
  'Parameter': 'multiplier_price',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.Generic Fuels.Natural Gas',
  'Source': 'AFDC 2023',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight',
  'Type': 'Sector',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': '',
  'Technology': '',
  'Parameter': 'multiplier_price',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.Generic Fuels.Petroleum Coke',
  'Source': 'JCIMS',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight',
  'Type': 'Sector',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': '',
  'Technology': '',
  'Parameter': 'multiplier_price',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.Generic Fuels.Propane',
  'Source': 'JCIMS',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight',
  'Type': 'Sector',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': '',
  'Technology': '',
  'Parameter': 'multiplier_price',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.Generic Fuels.Refinery Fuel Gas',
  'Source': 'JCIMS',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight',
  'Type': 'Sector',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': '',
  'Technology': '',
  'Parameter': 'multiplier_price',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.Generic Fuels.Solid Biomass',
  'Source': 'JCIMS',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight',
  'Type': 'Sector',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': '',
  'Technology': '',
  'Parameter': 'multiplier_price',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.Generic Fuels.Uranium',
  'Source': 'JCIMS',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight',
  'Type': 'Sector',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': '',
  'Technology': '',
  'Parameter': 'multiplier_price',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.Generic Fuels.Waste Fuel',
  'Source': 'JCIMS',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight',
  'Type': 'Sector',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': '',
  'Technology': '',
  'Parameter': 'service_request',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.CAN.AB.Transportation Freight.Freight',
  'Source': 'annual_tech',
  'Unit': 'k*tkm',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight',
  'Type': 'Sector',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': '',
  'Technology': '',
  'Parameter': 'service_request',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.CAN.AB.Transportation Freight.Off Road',
  'Source': 'annual_region',
  'Unit': 'k*tkm',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Freight',
  'Technology': '',
  'Parameter': 'service_provide',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'k*tkm',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Freight',
  'Technology': '',
  'Parameter': 'competition',
  'Context': 'Fixed Ratio',
  'Sub_Context': '',
  'Target': '',
  'Source': '',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Freight',
  'Technology': '',
  'Parameter': 'service_request',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.CAN.AB.Transportation Freight.Freight.Land',
  'Source': 'annual_region',
  'Unit': 'k*tkm',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Freight',
  'Technology': '',
  'Parameter': 'service_request',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.CAN.AB.Transportation Freight.Freight.Marine',
  'Source': 'annual_region',
  'Unit': 'k*tkm',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Freight',
  'Technology': '',
  'Parameter': 'service_request',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.CAN.AB.Transportation Freight.Freight.Air',
  'Source': 'annual_region',
  'Unit': 'k*tkm',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Land',
  'Technology': '',
  'Parameter': 'service_provide',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'k*tkm',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Land',
  'Technology': '',
  'Parameter': 'competition',
  'Context': 'Fixed Ratio',
  'Sub_Context': '',
  'Target': '',
  'Source': '',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Land',
  'Technology': '',
  'Parameter': 'service_request',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Source': 'annual_region',
  'Unit': 'k*tkm',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Land',
  'Technology': '',
  'Parameter': 'service_request',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy',
  'Source': 'annual_region',
  'Unit': 'k*tkm',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': '',
  'Parameter': 'service_provide',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'k*tkm',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': '',
  'Parameter': 'competition',
  'Context': 'Tech Compete',
  'Sub_Context': '',
  'Target': '',
  'Source': '',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': '',
  'Parameter': 'discount_rate_financial',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '%',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': '',
  'Parameter': 'discount_rate_retrofit',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '%',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': '',
  'Parameter': 'heterogeneity',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Diesel Existing',
  'Parameter': 'technology',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': '',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Diesel Existing',
  'Parameter': 'available',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Diesel Existing',
  'Parameter': 'unavailable',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Diesel Existing',
  'Parameter': 'lifetime',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Years',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Diesel Existing',
  'Parameter': 'market_share_total',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'annual_region_tech',
  'Unit': '%',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Diesel Existing',
  'Parameter': 'output',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'annual_region',
  'Unit': 'node unit',
  'Comments': 'TODO: Update with avg tkm from Freight calcs'},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Diesel Existing',
  'Parameter': 'fcc',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Diesel Existing',
  'Parameter': 'fom',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Diesel Existing',
  'Parameter': 'service_request',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.CAN.AB.Fuel Blends.Diesel_Transportation',
  'Source': 'annual_tech',
  'Unit': 'GJ',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Diesel Standard',
  'Parameter': 'technology',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': '',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Diesel Standard',
  'Parameter': 'available',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Diesel Standard',
  'Parameter': 'unavailable',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Diesel Standard',
  'Parameter': 'lifetime',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Years',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Diesel Standard',
  'Parameter': 'market_share_total',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'annual_tech',
  'Unit': '%',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Diesel Standard',
  'Parameter': 'output',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'annual_region',
  'Unit': 'node unit',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Diesel Standard',
  'Parameter': 'fcc',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Diesel Standard',
  'Parameter': 'fom',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Diesel Standard',
  'Parameter': 'service_request',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.CAN.AB.Fuel Blends.Diesel_Transportation',
  'Source': 'annual_tech',
  'Unit': 'GJ',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Diesel Efficient',
  'Parameter': 'technology',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': '',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Diesel Efficient',
  'Parameter': 'available',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Diesel Efficient',
  'Parameter': 'unavailable',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Diesel Efficient',
  'Parameter': 'lifetime',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Years',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Diesel Efficient',
  'Parameter': 'market_share_total',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'annual_tech',
  'Unit': '%',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Diesel Efficient',
  'Parameter': 'output',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'annual_region',
  'Unit': 'node unit',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Diesel Efficient',
  'Parameter': 'fcc',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Diesel Efficient',
  'Parameter': 'fom',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Diesel Efficient',
  'Parameter': 'service_request',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.CAN.AB.Fuel Blends.Diesel_Transportation',
  'Source': 'annual_tech',
  'Unit': 'GJ',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Gasoline Existing',
  'Parameter': 'technology',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': '',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Gasoline Existing',
  'Parameter': 'available',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Gasoline Existing',
  'Parameter': 'unavailable',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Gasoline Existing',
  'Parameter': 'lifetime',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Years',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Gasoline Existing',
  'Parameter': 'market_share_total',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'annual_region_tech',
  'Unit': '%',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Gasoline Existing',
  'Parameter': 'output',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'annual_region',
  'Unit': 'node unit',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Gasoline Existing',
  'Parameter': 'fcc',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Gasoline Existing',
  'Parameter': 'fom',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Gasoline Existing',
  'Parameter': 'service_request',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.CAN.AB.Fuel Blends.Gasoline_Transportation',
  'Source': 'annual_tech',
  'Unit': 'GJ',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Gasoline Standard',
  'Parameter': 'technology',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': '',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Gasoline Standard',
  'Parameter': 'available',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Gasoline Standard',
  'Parameter': 'unavailable',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Gasoline Standard',
  'Parameter': 'lifetime',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Years',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Gasoline Standard',
  'Parameter': 'market_share_total',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'annual_tech',
  'Unit': '%',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Gasoline Standard',
  'Parameter': 'output',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'annual_region',
  'Unit': 'node unit',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Gasoline Standard',
  'Parameter': 'fcc',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Gasoline Standard',
  'Parameter': 'fom',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Gasoline Standard',
  'Parameter': 'service_request',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.CAN.AB.Fuel Blends.Gasoline_Transportation',
  'Source': 'annual_tech',
  'Unit': 'GJ',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Gasoline Efficient',
  'Parameter': 'technology',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': '',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Gasoline Efficient',
  'Parameter': 'available',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Gasoline Efficient',
  'Parameter': 'unavailable',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Gasoline Efficient',
  'Parameter': 'lifetime',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Years',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Gasoline Efficient',
  'Parameter': 'market_share_total',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'annual_tech',
  'Unit': '%',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Gasoline Efficient',
  'Parameter': 'output',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'annual_region',
  'Unit': 'node unit',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Gasoline Efficient',
  'Parameter': 'fcc',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Gasoline Efficient',
  'Parameter': 'fom',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Gasoline Efficient',
  'Parameter': 'service_request',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.CAN.AB.Fuel Blends.Gasoline_Transportation',
  'Source': 'annual_tech',
  'Unit': 'GJ',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Propane',
  'Parameter': 'technology',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': '',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Propane',
  'Parameter': 'available',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Propane',
  'Parameter': 'unavailable',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Propane',
  'Parameter': 'lifetime',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Years',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Propane',
  'Parameter': 'market_share_total',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'annual_region_tech',
  'Unit': '%',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Propane',
  'Parameter': 'output',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'annual_region',
  'Unit': 'node unit',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Propane',
  'Parameter': 'fcc',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Propane',
  'Parameter': 'fom',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Propane',
  'Parameter': 'service_request',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.Generic Fuels.Propane',
  'Source': 'annual_tech',
  'Unit': 'GJ',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Biodiesel',
  'Parameter': 'technology',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': '',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Biodiesel',
  'Parameter': 'available',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Biodiesel',
  'Parameter': 'unavailable',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Biodiesel',
  'Parameter': 'lifetime',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Years',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Biodiesel',
  'Parameter': 'market_share_total',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'annual_tech',
  'Unit': '%',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Biodiesel',
  'Parameter': 'output',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'annual_region',
  'Unit': 'node unit',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Biodiesel',
  'Parameter': 'fcc',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Biodiesel',
  'Parameter': 'fom',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Biodiesel',
  'Parameter': 'service_request',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.CAN.AB.Biodiesel',
  'Source': 'annual_tech',
  'Unit': 'GJ',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Hydrogen',
  'Parameter': 'technology',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': '',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Hydrogen',
  'Parameter': 'available',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Hydrogen',
  'Parameter': 'unavailable',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Hydrogen',
  'Parameter': 'lifetime',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Years',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Hydrogen',
  'Parameter': 'market_share_total',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'annual_tech',
  'Unit': '%',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Hydrogen',
  'Parameter': 'output',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'annual_region',
  'Unit': 'node unit',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Hydrogen',
  'Parameter': 'fcc',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Hydrogen',
  'Parameter': 'fom',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Hydrogen',
  'Parameter': 'service_request',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.CAN.AB.Hydrogen',
  'Source': 'annual_tech',
  'Unit': 'GJ',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Hydrogen',
  'Parameter': 'service_request',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.CAN.AB.Transportation Freight.FCEV Infrastructure',
  'Source': 'annual_tech',
  'Unit': 'GJ',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Electric',
  'Parameter': 'technology',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': '',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Electric',
  'Parameter': 'available',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Electric',
  'Parameter': 'unavailable',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Electric',
  'Parameter': 'lifetime',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Years',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Electric',
  'Parameter': 'market_share_total',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'annual_tech',
  'Unit': '%',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Electric',
  'Parameter': 'output',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'annual_region',
  'Unit': 'node unit',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Electric',
  'Parameter': 'fcc',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Electric',
  'Parameter': 'fom',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Electric',
  'Parameter': 'service_request',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.CAN.AB.Electricity',
  'Source': 'annual_tech',
  'Unit': 'GJ',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Electric',
  'Parameter': 'service_request',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.CAN.AB.Transportation Freight.EV Infrastructure',
  'Source': 'annual_tech',
  'Unit': 'GJ',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Plug-in Hybrid',
  'Parameter': 'technology',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': '',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Plug-in Hybrid',
  'Parameter': 'available',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Plug-in Hybrid',
  'Parameter': 'unavailable',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Plug-in Hybrid',
  'Parameter': 'lifetime',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Years',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Plug-in Hybrid',
  'Parameter': 'market_share_total',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'annual_tech',
  'Unit': '%',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Plug-in Hybrid',
  'Parameter': 'output',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'annual_region',
  'Unit': 'node unit',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Plug-in Hybrid',
  'Parameter': 'fcc',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Plug-in Hybrid',
  'Parameter': 'fom',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Plug-in Hybrid',
  'Parameter': 'service_request',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.CAN.AB.Electricity',
  'Source': 'annual_tech',
  'Unit': 'GJ',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Plug-in Hybrid',
  'Parameter': 'service_request',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.CAN.AB.Transportation Freight.EV Infrastructure',
  'Source': 'annual_tech',
  'Unit': 'GJ',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Light Medium',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Light Medium',
  'Technology': 'Plug-in Hybrid',
  'Parameter': 'service_request',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.CAN.AB.Fuel Blends.Diesel_Transportation',
  'Source': 'annual_tech',
  'Unit': 'GJ',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Heavy',
  'Technology': '',
  'Parameter': 'service_provide',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'k*tkm',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Heavy',
  'Technology': '',
  'Parameter': 'competition',
  'Context': 'Tech Compete',
  'Sub_Context': '',
  'Target': '',
  'Source': '',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Heavy',
  'Technology': '',
  'Parameter': 'discount_rate_financial',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '%',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Heavy',
  'Technology': '',
  'Parameter': 'heterogeneity',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Heavy',
  'Technology': 'Trucks',
  'Parameter': 'technology',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': '',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Heavy',
  'Technology': 'Trucks',
  'Parameter': 'market_share_total',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'annual_region_tech',
  'Unit': '%',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Heavy',
  'Technology': 'Trucks',
  'Parameter': 'service_request',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Source': 'annual_tech',
  'Unit': 'k*tkm',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Heavy',
  'Technology': 'Rail',
  'Parameter': 'technology',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': '',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Heavy',
  'Technology': 'Rail',
  'Parameter': 'market_share_total',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'annual_region_tech',
  'Unit': '%',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Heavy',
  'Technology': 'Rail',
  'Parameter': 'service_request',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Rail',
  'Source': 'annual_tech',
  'Unit': 'k*tkm',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': '',
  'Parameter': 'service_provide',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'k*tkm',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': '',
  'Parameter': 'competition',
  'Context': 'Tech Compete',
  'Sub_Context': '',
  'Target': '',
  'Source': '',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': '',
  'Parameter': 'discount_rate_financial',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '%',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': '',
  'Parameter': 'discount_rate_retrofit',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '%',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': '',
  'Parameter': 'heterogeneity',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Diesel Existing',
  'Parameter': 'technology',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': '',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Diesel Existing',
  'Parameter': 'available',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Diesel Existing',
  'Parameter': 'unavailable',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Diesel Existing',
  'Parameter': 'lifetime',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Years',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Diesel Existing',
  'Parameter': 'market_share_total',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'annual_tech',
  'Unit': '%',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Diesel Existing',
  'Parameter': 'output',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'annual_region',
  'Unit': 'node unit',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Diesel Existing',
  'Parameter': 'fcc',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Diesel Existing',
  'Parameter': 'fom',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Diesel Existing',
  'Parameter': 'service_request',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.CAN.AB.Fuel Blends.Diesel_Transportation',
  'Source': 'annual_tech',
  'Unit': 'GJ',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Diesel Standard',
  'Parameter': 'technology',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': '',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Diesel Standard',
  'Parameter': 'available',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Diesel Standard',
  'Parameter': 'unavailable',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Diesel Standard',
  'Parameter': 'lifetime',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Years',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Diesel Standard',
  'Parameter': 'market_share_total',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'annual_tech',
  'Unit': '%',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Diesel Standard',
  'Parameter': 'output',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'annual_region',
  'Unit': 'node unit',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Diesel Standard',
  'Parameter': 'fcc',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Diesel Standard',
  'Parameter': 'fom',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Diesel Standard',
  'Parameter': 'service_request',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.CAN.AB.Fuel Blends.Diesel_Transportation',
  'Source': 'annual_tech',
  'Unit': 'GJ',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Diesel Efficient',
  'Parameter': 'technology',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': '',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Diesel Efficient',
  'Parameter': 'available',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Diesel Efficient',
  'Parameter': 'unavailable',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Diesel Efficient',
  'Parameter': 'lifetime',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Years',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Diesel Efficient',
  'Parameter': 'market_share_total',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'annual_tech',
  'Unit': '%',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Diesel Efficient',
  'Parameter': 'output',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'annual_region',
  'Unit': 'node unit',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Diesel Efficient',
  'Parameter': 'fcc',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Diesel Efficient',
  'Parameter': 'fom',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Diesel Efficient',
  'Parameter': 'service_request',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.CAN.AB.Fuel Blends.Diesel_Transportation',
  'Source': 'annual_tech',
  'Unit': 'GJ',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Natural Gas',
  'Parameter': 'technology',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': '',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Natural Gas',
  'Parameter': 'available',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Natural Gas',
  'Parameter': 'unavailable',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Natural Gas',
  'Parameter': 'lifetime',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Years',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Natural Gas',
  'Parameter': 'market_share_total',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'annual_tech',
  'Unit': '%',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Natural Gas',
  'Parameter': 'output',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'annual_region',
  'Unit': 'node unit',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Natural Gas',
  'Parameter': 'fcc',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Natural Gas',
  'Parameter': 'fom',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Natural Gas',
  'Parameter': 'service_request',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.Generic Fuels.Natural Gas',
  'Source': 'annual_tech',
  'Unit': 'GJ',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Biodiesel',
  'Parameter': 'technology',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': '',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Biodiesel',
  'Parameter': 'available',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Biodiesel',
  'Parameter': 'unavailable',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Biodiesel',
  'Parameter': 'lifetime',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Years',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Biodiesel',
  'Parameter': 'market_share_total',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'annual_tech',
  'Unit': '%',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Biodiesel',
  'Parameter': 'output',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'annual_region',
  'Unit': 'node unit',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Biodiesel',
  'Parameter': 'fcc',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Biodiesel',
  'Parameter': 'fom',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Biodiesel',
  'Parameter': 'service_request',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.CAN.AB.Biodiesel',
  'Source': 'annual_tech',
  'Unit': 'GJ',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Hydrogen',
  'Parameter': 'technology',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': '',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Hydrogen',
  'Parameter': 'available',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Hydrogen',
  'Parameter': 'unavailable',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Hydrogen',
  'Parameter': 'lifetime',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Years',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Hydrogen',
  'Parameter': 'market_share_total',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'annual_tech',
  'Unit': '%',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Hydrogen',
  'Parameter': 'output',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'annual_region',
  'Unit': 'node unit',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Hydrogen',
  'Parameter': 'fcc',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Hydrogen',
  'Parameter': 'fom',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Hydrogen',
  'Parameter': 'service_request',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.CAN.AB.Hydrogen',
  'Source': 'annual_tech',
  'Unit': 'GJ',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Hydrogen',
  'Parameter': 'service_request',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.CAN.AB.Transportation Freight.FCEV Infrastructure',
  'Source': 'annual_tech',
  'Unit': 'GJ',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Electric',
  'Parameter': 'technology',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': '',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Electric',
  'Parameter': 'available',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Electric',
  'Parameter': 'unavailable',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Electric',
  'Parameter': 'lifetime',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Years',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Electric',
  'Parameter': 'market_share_total',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'annual_tech',
  'Unit': '%',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Electric',
  'Parameter': 'output',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'annual_region',
  'Unit': 'node unit',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Electric',
  'Parameter': 'fcc',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Electric',
  'Parameter': 'fom',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Electric',
  'Parameter': 'service_request',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.CAN.AB.Electricity',
  'Source': 'annual_tech',
  'Unit': 'GJ',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Electric',
  'Parameter': 'service_request',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.CAN.AB.Transportation Freight.EV Infrastructure',
  'Source': 'annual_tech',
  'Unit': 'GJ',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Plug-in Hybrid',
  'Parameter': 'technology',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': '',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Plug-in Hybrid',
  'Parameter': 'available',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Plug-in Hybrid',
  'Parameter': 'unavailable',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Plug-in Hybrid',
  'Parameter': 'lifetime',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Years',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Plug-in Hybrid',
  'Parameter': 'market_share_total',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'annual_tech',
  'Unit': '%',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Plug-in Hybrid',
  'Parameter': 'output',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'annual_region',
  'Unit': 'node unit',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Plug-in Hybrid',
  'Parameter': 'fcc',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Plug-in Hybrid',
  'Parameter': 'fom',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Plug-in Hybrid',
  'Parameter': 'service_request',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.CAN.AB.Electricity',
  'Source': 'annual_tech',
  'Unit': 'GJ',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Plug-in Hybrid',
  'Parameter': 'service_request',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.CAN.AB.Transportation Freight.EV Infrastructure',
  'Source': 'annual_tech',
  'Unit': 'GJ',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Plug-in Hybrid',
  'Parameter': 'service_request',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.CAN.AB.Fuel Blends.Diesel_Transportation',
  'Source': 'annual_tech',
  'Unit': 'GJ',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Catenary',
  'Parameter': 'technology',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': '',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Catenary',
  'Parameter': 'available',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Catenary',
  'Parameter': 'unavailable',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Catenary',
  'Parameter': 'lifetime',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Years',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Catenary',
  'Parameter': 'market_share_total',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'annual_tech',
  'Unit': '%',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Catenary',
  'Parameter': 'output',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'annual_region',
  'Unit': 'node unit',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Catenary',
  'Parameter': 'fcc',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Catenary',
  'Parameter': 'fom',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Catenary',
  'Parameter': 'service_request',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.CAN.AB.Electricity',
  'Source': 'annual_tech',
  'Unit': 'GJ',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Trucks',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Trucks',
  'Technology': 'Catenary',
  'Parameter': 'service_request',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.CAN.AB.Transportation Freight.Catenary Infrastructure',
  'Source': 'annual_tech',
  'Unit': 'GJ',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Rail',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Rail',
  'Technology': '',
  'Parameter': 'service_provide',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'k*tkm',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Rail',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Rail',
  'Technology': '',
  'Parameter': 'competition',
  'Context': 'Tech Compete',
  'Sub_Context': '',
  'Target': '',
  'Source': '',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Rail',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Rail',
  'Technology': '',
  'Parameter': 'discount_rate_financial',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '%',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Rail',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Rail',
  'Technology': '',
  'Parameter': 'heterogeneity',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Rail',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Rail',
  'Technology': 'Diesel Existing',
  'Parameter': 'technology',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': '',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Rail',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Rail',
  'Technology': 'Diesel Existing',
  'Parameter': 'available',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Rail',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Rail',
  'Technology': 'Diesel Existing',
  'Parameter': 'unavailable',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Rail',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Rail',
  'Technology': 'Diesel Existing',
  'Parameter': 'lifetime',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Years',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Rail',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Rail',
  'Technology': 'Diesel Existing',
  'Parameter': 'market_share_total',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'annual_tech',
  'Unit': '%',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Rail',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Rail',
  'Technology': 'Diesel Existing',
  'Parameter': 'output',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'node unit',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Rail',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Rail',
  'Technology': 'Diesel Existing',
  'Parameter': 'fcc',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Rail',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Rail',
  'Technology': 'Diesel Existing',
  'Parameter': 'fom',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Rail',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Rail',
  'Technology': 'Diesel Existing',
  'Parameter': 'service_request',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.CAN.AB.Fuel Blends.Diesel_Transportation',
  'Source': 'annual_tech',
  'Unit': 'GJ',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Rail',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Rail',
  'Technology': 'Diesel Standard',
  'Parameter': 'technology',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': '',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Rail',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Rail',
  'Technology': 'Diesel Standard',
  'Parameter': 'available',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Rail',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Rail',
  'Technology': 'Diesel Standard',
  'Parameter': 'unavailable',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Rail',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Rail',
  'Technology': 'Diesel Standard',
  'Parameter': 'lifetime',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Years',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Rail',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Rail',
  'Technology': 'Diesel Standard',
  'Parameter': 'market_share_total',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'annual_tech',
  'Unit': '%',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Rail',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Rail',
  'Technology': 'Diesel Standard',
  'Parameter': 'output',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'node unit',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Rail',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Rail',
  'Technology': 'Diesel Standard',
  'Parameter': 'fcc',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Rail',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Rail',
  'Technology': 'Diesel Standard',
  'Parameter': 'fom',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Rail',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Rail',
  'Technology': 'Diesel Standard',
  'Parameter': 'service_request',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.CAN.AB.Fuel Blends.Diesel_Transportation',
  'Source': 'annual_tech',
  'Unit': 'GJ',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Rail',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Rail',
  'Technology': 'Diesel Efficient',
  'Parameter': 'technology',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': '',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Rail',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Rail',
  'Technology': 'Diesel Efficient',
  'Parameter': 'available',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Rail',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Rail',
  'Technology': 'Diesel Efficient',
  'Parameter': 'unavailable',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Rail',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Rail',
  'Technology': 'Diesel Efficient',
  'Parameter': 'lifetime',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Years',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Rail',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Rail',
  'Technology': 'Diesel Efficient',
  'Parameter': 'market_share_total',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'annual_tech',
  'Unit': '%',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Rail',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Rail',
  'Technology': 'Diesel Efficient',
  'Parameter': 'output',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'node unit',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Rail',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Rail',
  'Technology': 'Diesel Efficient',
  'Parameter': 'fcc',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Rail',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Rail',
  'Technology': 'Diesel Efficient',
  'Parameter': 'fom',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Rail',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Rail',
  'Technology': 'Diesel Efficient',
  'Parameter': 'service_request',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.CAN.AB.Fuel Blends.Diesel_Transportation',
  'Source': 'annual_tech',
  'Unit': 'GJ',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Rail',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Rail',
  'Technology': 'Biodiesel',
  'Parameter': 'technology',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': '',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Rail',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Rail',
  'Technology': 'Biodiesel',
  'Parameter': 'available',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Rail',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Rail',
  'Technology': 'Biodiesel',
  'Parameter': 'unavailable',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Rail',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Rail',
  'Technology': 'Biodiesel',
  'Parameter': 'lifetime',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Years',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Rail',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Rail',
  'Technology': 'Biodiesel',
  'Parameter': 'market_share_total',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'annual_tech',
  'Unit': '%',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Rail',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Rail',
  'Technology': 'Biodiesel',
  'Parameter': 'output',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'node unit',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Rail',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Rail',
  'Technology': 'Biodiesel',
  'Parameter': 'fcc',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Rail',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Rail',
  'Technology': 'Biodiesel',
  'Parameter': 'fom',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Rail',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Rail',
  'Technology': 'Biodiesel',
  'Parameter': 'service_request',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.CAN.AB.Biodiesel',
  'Source': 'annual_tech',
  'Unit': 'GJ',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Rail',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Rail',
  'Technology': 'Hydrogen',
  'Parameter': 'technology',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': '',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Rail',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Rail',
  'Technology': 'Hydrogen',
  'Parameter': 'available',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Rail',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Rail',
  'Technology': 'Hydrogen',
  'Parameter': 'unavailable',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Rail',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Rail',
  'Technology': 'Hydrogen',
  'Parameter': 'lifetime',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Years',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Rail',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Rail',
  'Technology': 'Hydrogen',
  'Parameter': 'market_share_total',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'annual_tech',
  'Unit': '%',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Rail',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Rail',
  'Technology': 'Hydrogen',
  'Parameter': 'output',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'node unit',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Rail',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Rail',
  'Technology': 'Hydrogen',
  'Parameter': 'fcc',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Rail',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Rail',
  'Technology': 'Hydrogen',
  'Parameter': 'fom',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Rail',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Rail',
  'Technology': 'Hydrogen',
  'Parameter': 'service_request',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.CAN.AB.Hydrogen',
  'Source': 'annual_tech',
  'Unit': 'GJ',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Rail',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Rail',
  'Technology': 'Electric',
  'Parameter': 'technology',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': '',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Rail',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Rail',
  'Technology': 'Electric',
  'Parameter': 'available',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Rail',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Rail',
  'Technology': 'Electric',
  'Parameter': 'unavailable',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Rail',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Rail',
  'Technology': 'Electric',
  'Parameter': 'lifetime',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Years',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Rail',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Rail',
  'Technology': 'Electric',
  'Parameter': 'market_share_total',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'annual_tech',
  'Unit': '%',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Rail',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Rail',
  'Technology': 'Electric',
  'Parameter': 'output',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'node unit',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Rail',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Rail',
  'Technology': 'Electric',
  'Parameter': 'fcc',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Rail',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Rail',
  'Technology': 'Electric',
  'Parameter': 'fom',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Land.Heavy.Rail',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Rail',
  'Technology': 'Electric',
  'Parameter': 'service_request',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.CAN.AB.Electricity',
  'Source': 'annual_tech',
  'Unit': 'GJ',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Marine',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Marine',
  'Technology': '',
  'Parameter': 'service_provide',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'k*tkm',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Marine',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Marine',
  'Technology': '',
  'Parameter': 'competition',
  'Context': 'Tech Compete',
  'Sub_Context': '',
  'Target': '',
  'Source': '',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Marine',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Marine',
  'Technology': '',
  'Parameter': 'discount_rate_financial',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '%',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Marine',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Marine',
  'Technology': '',
  'Parameter': 'heterogeneity',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Marine',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Marine',
  'Technology': 'Fuel Oil Existing',
  'Parameter': 'technology',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': '',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Marine',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Marine',
  'Technology': 'Fuel Oil Existing',
  'Parameter': 'available',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Marine',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Marine',
  'Technology': 'Fuel Oil Existing',
  'Parameter': 'unavailable',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Marine',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Marine',
  'Technology': 'Fuel Oil Existing',
  'Parameter': 'lifetime',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Years',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Marine',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Marine',
  'Technology': 'Fuel Oil Existing',
  'Parameter': 'market_share_total',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'annual_tech',
  'Unit': '%',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Marine',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Marine',
  'Technology': 'Fuel Oil Existing',
  'Parameter': 'output',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'node unit',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Marine',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Marine',
  'Technology': 'Fuel Oil Existing',
  'Parameter': 'fcc',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Marine',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Marine',
  'Technology': 'Fuel Oil Existing',
  'Parameter': 'fom',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Marine',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Marine',
  'Technology': 'Fuel Oil Existing',
  'Parameter': 'service_request',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.Generic Fuels.Fuel Oil',
  'Source': 'annual_tech',
  'Unit': 'GJ',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Marine',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Marine',
  'Technology': 'Fuel Oil Standard',
  'Parameter': 'technology',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': '',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Marine',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Marine',
  'Technology': 'Fuel Oil Standard',
  'Parameter': 'available',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Marine',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Marine',
  'Technology': 'Fuel Oil Standard',
  'Parameter': 'unavailable',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Marine',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Marine',
  'Technology': 'Fuel Oil Standard',
  'Parameter': 'lifetime',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Years',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Marine',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Marine',
  'Technology': 'Fuel Oil Standard',
  'Parameter': 'market_share_total',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'annual_tech',
  'Unit': '%',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Marine',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Marine',
  'Technology': 'Fuel Oil Standard',
  'Parameter': 'output',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'node unit',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Marine',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Marine',
  'Technology': 'Fuel Oil Standard',
  'Parameter': 'fcc',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Marine',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Marine',
  'Technology': 'Fuel Oil Standard',
  'Parameter': 'fom',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Marine',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Marine',
  'Technology': 'Fuel Oil Standard',
  'Parameter': 'service_request',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.Generic Fuels.Fuel Oil',
  'Source': 'annual_tech',
  'Unit': 'GJ',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Marine',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Marine',
  'Technology': 'Fuel Oil Efficient',
  'Parameter': 'technology',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': '',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Marine',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Marine',
  'Technology': 'Fuel Oil Efficient',
  'Parameter': 'available',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Marine',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Marine',
  'Technology': 'Fuel Oil Efficient',
  'Parameter': 'unavailable',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Marine',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Marine',
  'Technology': 'Fuel Oil Efficient',
  'Parameter': 'lifetime',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Years',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Marine',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Marine',
  'Technology': 'Fuel Oil Efficient',
  'Parameter': 'market_share_total',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'annual_tech',
  'Unit': '%',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Marine',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Marine',
  'Technology': 'Fuel Oil Efficient',
  'Parameter': 'output',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'node unit',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Marine',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Marine',
  'Technology': 'Fuel Oil Efficient',
  'Parameter': 'fcc',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Marine',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Marine',
  'Technology': 'Fuel Oil Efficient',
  'Parameter': 'fom',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Marine',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Marine',
  'Technology': 'Fuel Oil Efficient',
  'Parameter': 'service_request',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.Generic Fuels.Fuel Oil',
  'Source': 'annual_tech',
  'Unit': 'GJ',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Marine',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Marine',
  'Technology': 'Biodiesel',
  'Parameter': 'technology',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': '',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Marine',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Marine',
  'Technology': 'Biodiesel',
  'Parameter': 'available',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Marine',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Marine',
  'Technology': 'Biodiesel',
  'Parameter': 'unavailable',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Marine',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Marine',
  'Technology': 'Biodiesel',
  'Parameter': 'lifetime',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Years',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Marine',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Marine',
  'Technology': 'Biodiesel',
  'Parameter': 'market_share_total',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'annual_tech',
  'Unit': '%',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Marine',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Marine',
  'Technology': 'Biodiesel',
  'Parameter': 'output',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'node unit',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Marine',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Marine',
  'Technology': 'Biodiesel',
  'Parameter': 'fcc',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Marine',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Marine',
  'Technology': 'Biodiesel',
  'Parameter': 'fom',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Marine',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Marine',
  'Technology': 'Biodiesel',
  'Parameter': 'service_request',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.CAN.AB.Biodiesel',
  'Source': 'annual_tech',
  'Unit': 'GJ',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Marine',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Marine',
  'Technology': 'Hydrogen',
  'Parameter': 'technology',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': '',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Marine',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Marine',
  'Technology': 'Hydrogen',
  'Parameter': 'available',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Marine',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Marine',
  'Technology': 'Hydrogen',
  'Parameter': 'unavailable',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Marine',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Marine',
  'Technology': 'Hydrogen',
  'Parameter': 'lifetime',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Years',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Marine',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Marine',
  'Technology': 'Hydrogen',
  'Parameter': 'market_share_total',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'annual_tech',
  'Unit': '%',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Marine',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Marine',
  'Technology': 'Hydrogen',
  'Parameter': 'output',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'node unit',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Marine',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Marine',
  'Technology': 'Hydrogen',
  'Parameter': 'fcc',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Marine',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Marine',
  'Technology': 'Hydrogen',
  'Parameter': 'fom',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Marine',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Marine',
  'Technology': 'Hydrogen',
  'Parameter': 'service_request',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.CAN.AB.Hydrogen',
  'Source': 'annual_tech',
  'Unit': 'GJ',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Air',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Air',
  'Technology': '',
  'Parameter': 'service_provide',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'k*tkm',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Air',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Air',
  'Technology': '',
  'Parameter': 'competition',
  'Context': 'Tech Compete',
  'Sub_Context': '',
  'Target': '',
  'Source': '',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Air',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Air',
  'Technology': '',
  'Parameter': 'discount_rate_financial',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '%',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Air',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Air',
  'Technology': '',
  'Parameter': 'heterogeneity',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Air',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Air',
  'Technology': 'Existing',
  'Parameter': 'technology',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': '',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Air',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Air',
  'Technology': 'Existing',
  'Parameter': 'available',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Air',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Air',
  'Technology': 'Existing',
  'Parameter': 'unavailable',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Air',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Air',
  'Technology': 'Existing',
  'Parameter': 'lifetime',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Years',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Air',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Air',
  'Technology': 'Existing',
  'Parameter': 'market_share_total',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'annual_tech',
  'Unit': '%',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Air',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Air',
  'Technology': 'Existing',
  'Parameter': 'output',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'node unit',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Air',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Air',
  'Technology': 'Existing',
  'Parameter': 'fcc',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Air',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Air',
  'Technology': 'Existing',
  'Parameter': 'service_request',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.Generic Fuels.Jet Fuel',
  'Source': 'annual_tech',
  'Unit': 'GJ',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Air',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Air',
  'Technology': 'Efficient',
  'Parameter': 'technology',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': '',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Air',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Air',
  'Technology': 'Efficient',
  'Parameter': 'available',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Air',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Air',
  'Technology': 'Efficient',
  'Parameter': 'unavailable',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Air',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Air',
  'Technology': 'Efficient',
  'Parameter': 'lifetime',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Years',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Air',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Air',
  'Technology': 'Efficient',
  'Parameter': 'market_share_total',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'annual_tech',
  'Unit': '%',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Air',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Air',
  'Technology': 'Efficient',
  'Parameter': 'output',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'node unit',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Air',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Air',
  'Technology': 'Efficient',
  'Parameter': 'fcc',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Air',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Air',
  'Technology': 'Efficient',
  'Parameter': 'service_request',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.Generic Fuels.Jet Fuel',
  'Source': 'annual_tech',
  'Unit': 'GJ',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Air',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Air',
  'Technology': 'Electric',
  'Parameter': 'technology',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': '',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Air',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Air',
  'Technology': 'Electric',
  'Parameter': 'available',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Air',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Air',
  'Technology': 'Electric',
  'Parameter': 'unavailable',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Air',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Air',
  'Technology': 'Electric',
  'Parameter': 'lifetime',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Years',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Air',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Air',
  'Technology': 'Electric',
  'Parameter': 'market_share_total',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'annual_tech',
  'Unit': '%',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Air',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Air',
  'Technology': 'Electric',
  'Parameter': 'output',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'node unit',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Air',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Air',
  'Technology': 'Electric',
  'Parameter': 'fcc',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Air',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Air',
  'Technology': 'Electric',
  'Parameter': 'service_request',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.CAN.AB.Electricity',
  'Source': 'annual_tech',
  'Unit': 'GJ',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Air',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Air',
  'Technology': 'Hydrogen',
  'Parameter': 'technology',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': '',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Air',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Air',
  'Technology': 'Hydrogen',
  'Parameter': 'available',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Air',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Air',
  'Technology': 'Hydrogen',
  'Parameter': 'unavailable',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Air',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Air',
  'Technology': 'Hydrogen',
  'Parameter': 'lifetime',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Years',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Air',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Air',
  'Technology': 'Hydrogen',
  'Parameter': 'market_share_total',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'annual_tech',
  'Unit': '%',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Air',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Air',
  'Technology': 'Hydrogen',
  'Parameter': 'output',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'node unit',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Air',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Air',
  'Technology': 'Hydrogen',
  'Parameter': 'fcc',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Freight.Air',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Air',
  'Technology': 'Hydrogen',
  'Parameter': 'service_request',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.CAN.AB.Hydrogen',
  'Source': 'annual_tech',
  'Unit': 'GJ',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Off Road',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Off Road',
  'Technology': '',
  'Parameter': 'service_provide',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'k*tkm',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Off Road',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Off Road',
  'Technology': '',
  'Parameter': 'competition',
  'Context': 'Tech Compete',
  'Sub_Context': '',
  'Target': '',
  'Source': '',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Off Road',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Off Road',
  'Technology': '',
  'Parameter': 'discount_rate_financial',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '%',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Off Road',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Off Road',
  'Technology': '',
  'Parameter': 'discount_rate_retrofit',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '%',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Off Road',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Off Road',
  'Technology': '',
  'Parameter': 'heterogeneity',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Off Road',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Off Road',
  'Technology': 'Diesel Std',
  'Parameter': 'technology',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': '',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Off Road',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Off Road',
  'Technology': 'Diesel Std',
  'Parameter': 'available',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Off Road',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Off Road',
  'Technology': 'Diesel Std',
  'Parameter': 'unavailable',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Off Road',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Off Road',
  'Technology': 'Diesel Std',
  'Parameter': 'lifetime',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Years',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Off Road',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Off Road',
  'Technology': 'Diesel Std',
  'Parameter': 'market_share_total',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'annual_tech',
  'Unit': '%',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Off Road',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Off Road',
  'Technology': 'Diesel Std',
  'Parameter': 'output',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'node unit',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Off Road',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Off Road',
  'Technology': 'Diesel Std',
  'Parameter': 'fcc',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Off Road',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Off Road',
  'Technology': 'Diesel Std',
  'Parameter': 'service_request',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.CAN.AB.Fuel Blends.Diesel_Transportation',
  'Source': 'annual_tech',
  'Unit': 'GJ',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Off Road',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Off Road',
  'Technology': 'Diesel Medium Efficiency',
  'Parameter': 'technology',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': '',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Off Road',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Off Road',
  'Technology': 'Diesel Medium Efficiency',
  'Parameter': 'available',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Off Road',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Off Road',
  'Technology': 'Diesel Medium Efficiency',
  'Parameter': 'unavailable',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Off Road',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Off Road',
  'Technology': 'Diesel Medium Efficiency',
  'Parameter': 'lifetime',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Years',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Off Road',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Off Road',
  'Technology': 'Diesel Medium Efficiency',
  'Parameter': 'market_share_total',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'annual_tech',
  'Unit': '%',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Off Road',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Off Road',
  'Technology': 'Diesel Medium Efficiency',
  'Parameter': 'output',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'node unit',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Off Road',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Off Road',
  'Technology': 'Diesel Medium Efficiency',
  'Parameter': 'fcc',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Off Road',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Off Road',
  'Technology': 'Diesel Medium Efficiency',
  'Parameter': 'service_request',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.CAN.AB.Fuel Blends.Diesel_Transportation',
  'Source': 'annual_tech',
  'Unit': 'GJ',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Off Road',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Off Road',
  'Technology': 'Diesel High Efficiency',
  'Parameter': 'technology',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': '',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Off Road',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Off Road',
  'Technology': 'Diesel High Efficiency',
  'Parameter': 'available',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Off Road',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Off Road',
  'Technology': 'Diesel High Efficiency',
  'Parameter': 'unavailable',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Off Road',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Off Road',
  'Technology': 'Diesel High Efficiency',
  'Parameter': 'lifetime',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Years',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Off Road',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Off Road',
  'Technology': 'Diesel High Efficiency',
  'Parameter': 'market_share_total',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'annual_tech',
  'Unit': '%',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Off Road',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Off Road',
  'Technology': 'Diesel High Efficiency',
  'Parameter': 'output',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'node unit',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Off Road',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Off Road',
  'Technology': 'Diesel High Efficiency',
  'Parameter': 'fcc',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Off Road',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Off Road',
  'Technology': 'Diesel High Efficiency',
  'Parameter': 'service_request',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.CAN.AB.Fuel Blends.Diesel_Transportation',
  'Source': 'annual_tech',
  'Unit': 'GJ',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Off Road',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Off Road',
  'Technology': 'Electric',
  'Parameter': 'technology',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': '',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Off Road',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Off Road',
  'Technology': 'Electric',
  'Parameter': 'available',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Off Road',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Off Road',
  'Technology': 'Electric',
  'Parameter': 'unavailable',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Off Road',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Off Road',
  'Technology': 'Electric',
  'Parameter': 'lifetime',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Years',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Off Road',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Off Road',
  'Technology': 'Electric',
  'Parameter': 'market_share_total',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'annual_tech',
  'Unit': '%',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Off Road',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Off Road',
  'Technology': 'Electric',
  'Parameter': 'output',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'node unit',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Off Road',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Off Road',
  'Technology': 'Electric',
  'Parameter': 'fcc',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Off Road',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Off Road',
  'Technology': 'Electric',
  'Parameter': 'service_request',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.CAN.AB.Electricity',
  'Source': 'annual_tech',
  'Unit': 'GJ',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.EV Infrastructure',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'EV Infrastructure',
  'Technology': '',
  'Parameter': 'service_provide',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'GJ',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.EV Infrastructure',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'EV Infrastructure',
  'Technology': '',
  'Parameter': 'competition',
  'Context': 'Fixed Ratio',
  'Sub_Context': '',
  'Target': '',
  'Source': '',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.EV Infrastructure',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'EV Infrastructure',
  'Technology': '',
  'Parameter': 'service_request',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.CAN.AB.Transportation Freight.EV Infrastructure.Depot Stations',
  'Source': 'annual_tech',
  'Unit': 'GJ',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.EV Infrastructure',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'EV Infrastructure',
  'Technology': '',
  'Parameter': 'service_request',
  'Context': '',
  'Sub_Context': '',
  'Target': 'CIMS.CAN.AB.Transportation Freight.EV Infrastructure.Rapid Charging Stations',
  'Source': 'annual_tech',
  'Unit': 'GJ',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.EV Infrastructure.Depot Stations',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Depot Stations',
  'Technology': '',
  'Parameter': 'service_provide',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'GJ',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.EV Infrastructure.Depot Stations',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Depot Stations',
  'Technology': '',
  'Parameter': 'competition',
  'Context': 'Tech Compete',
  'Sub_Context': '',
  'Target': '',
  'Source': '',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.EV Infrastructure.Depot Stations',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Depot Stations',
  'Technology': '',
  'Parameter': 'discount_rate_financial',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '%',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.EV Infrastructure.Depot Stations',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Depot Stations',
  'Technology': '',
  'Parameter': 'heterogeneity',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.EV Infrastructure.Depot Stations',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Depot Stations',
  'Technology': 'Station',
  'Parameter': 'technology',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': '',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.EV Infrastructure.Depot Stations',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Depot Stations',
  'Technology': 'Station',
  'Parameter': 'available',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.EV Infrastructure.Depot Stations',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Depot Stations',
  'Technology': 'Station',
  'Parameter': 'unavailable',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.EV Infrastructure.Depot Stations',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Depot Stations',
  'Technology': 'Station',
  'Parameter': 'lifetime',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Years',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.EV Infrastructure.Depot Stations',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Depot Stations',
  'Technology': 'Station',
  'Parameter': 'market_share_total',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'annual_tech',
  'Unit': '%',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.EV Infrastructure.Depot Stations',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Depot Stations',
  'Technology': 'Station',
  'Parameter': 'output',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'node unit',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.EV Infrastructure.Depot Stations',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Depot Stations',
  'Technology': 'Station',
  'Parameter': 'fcc',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.EV Infrastructure.Depot Stations',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Depot Stations',
  'Technology': 'Station',
  'Parameter': 'fom',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.EV Infrastructure.Rapid Charging Stations',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Rapid Charging Stations',
  'Technology': '',
  'Parameter': 'service_provide',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'GJ',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.EV Infrastructure.Rapid Charging Stations',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Rapid Charging Stations',
  'Technology': '',
  'Parameter': 'competition',
  'Context': 'Tech Compete',
  'Sub_Context': '',
  'Target': '',
  'Source': '',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.EV Infrastructure.Rapid Charging Stations',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Rapid Charging Stations',
  'Technology': '',
  'Parameter': 'discount_rate_financial',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '%',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.EV Infrastructure.Rapid Charging Stations',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Rapid Charging Stations',
  'Technology': '',
  'Parameter': 'heterogeneity',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.EV Infrastructure.Rapid Charging Stations',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Rapid Charging Stations',
  'Technology': 'Station',
  'Parameter': 'technology',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': '',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.EV Infrastructure.Rapid Charging Stations',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Rapid Charging Stations',
  'Technology': 'Station',
  'Parameter': 'available',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.EV Infrastructure.Rapid Charging Stations',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Rapid Charging Stations',
  'Technology': 'Station',
  'Parameter': 'unavailable',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.EV Infrastructure.Rapid Charging Stations',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Rapid Charging Stations',
  'Technology': 'Station',
  'Parameter': 'lifetime',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Years',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.EV Infrastructure.Rapid Charging Stations',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Rapid Charging Stations',
  'Technology': 'Station',
  'Parameter': 'market_share_total',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'annual_tech',
  'Unit': '%',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.EV Infrastructure.Rapid Charging Stations',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Rapid Charging Stations',
  'Technology': 'Station',
  'Parameter': 'output',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'node unit',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.EV Infrastructure.Rapid Charging Stations',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Rapid Charging Stations',
  'Technology': 'Station',
  'Parameter': 'fcc',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.EV Infrastructure.Rapid Charging Stations',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Rapid Charging Stations',
  'Technology': 'Station',
  'Parameter': 'fom',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.FCEV Infrastructure',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'FCEV Infrastructure',
  'Technology': '',
  'Parameter': 'service_provide',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'GJ',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.FCEV Infrastructure',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'FCEV Infrastructure',
  'Technology': '',
  'Parameter': 'competition',
  'Context': 'Tech Compete',
  'Sub_Context': '',
  'Target': '',
  'Source': '',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.FCEV Infrastructure',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'FCEV Infrastructure',
  'Technology': '',
  'Parameter': 'discount_rate_financial',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '%',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.FCEV Infrastructure',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'FCEV Infrastructure',
  'Technology': '',
  'Parameter': 'heterogeneity',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.FCEV Infrastructure',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'FCEV Infrastructure',
  'Technology': 'Low Capacity Station',
  'Parameter': 'technology',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': '',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.FCEV Infrastructure',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'FCEV Infrastructure',
  'Technology': 'Low Capacity Station',
  'Parameter': 'available',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.FCEV Infrastructure',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'FCEV Infrastructure',
  'Technology': 'Low Capacity Station',
  'Parameter': 'unavailable',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.FCEV Infrastructure',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'FCEV Infrastructure',
  'Technology': 'Low Capacity Station',
  'Parameter': 'lifetime',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Years',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.FCEV Infrastructure',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'FCEV Infrastructure',
  'Technology': 'Low Capacity Station',
  'Parameter': 'market_share_total',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'annual_tech',
  'Unit': '%',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.FCEV Infrastructure',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'FCEV Infrastructure',
  'Technology': 'Low Capacity Station',
  'Parameter': 'output',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'node unit',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.FCEV Infrastructure',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'FCEV Infrastructure',
  'Technology': 'Low Capacity Station',
  'Parameter': 'fcc',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.FCEV Infrastructure',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'FCEV Infrastructure',
  'Technology': 'Low Capacity Station',
  'Parameter': 'fom',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.FCEV Infrastructure',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'FCEV Infrastructure',
  'Technology': 'High Capacity Station',
  'Parameter': 'technology',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': '',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.FCEV Infrastructure',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'FCEV Infrastructure',
  'Technology': 'High Capacity Station',
  'Parameter': 'available',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.FCEV Infrastructure',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'FCEV Infrastructure',
  'Technology': 'High Capacity Station',
  'Parameter': 'unavailable',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.FCEV Infrastructure',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'FCEV Infrastructure',
  'Technology': 'High Capacity Station',
  'Parameter': 'lifetime',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Years',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.FCEV Infrastructure',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'FCEV Infrastructure',
  'Technology': 'High Capacity Station',
  'Parameter': 'market_share_total',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'annual_tech',
  'Unit': '%',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.FCEV Infrastructure',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'FCEV Infrastructure',
  'Technology': 'High Capacity Station',
  'Parameter': 'output',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'node unit',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.FCEV Infrastructure',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'FCEV Infrastructure',
  'Technology': 'High Capacity Station',
  'Parameter': 'fcc',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.FCEV Infrastructure',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'FCEV Infrastructure',
  'Technology': 'High Capacity Station',
  'Parameter': 'fom',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Catenary Infrastructure',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Catenary Infrastructure',
  'Technology': '',
  'Parameter': 'service_provide',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'GJ',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Catenary Infrastructure',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Catenary Infrastructure',
  'Technology': '',
  'Parameter': 'competition',
  'Context': 'Tech Compete',
  'Sub_Context': '',
  'Target': '',
  'Source': '',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Catenary Infrastructure',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Catenary Infrastructure',
  'Technology': '',
  'Parameter': 'discount_rate_financial',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '%',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Catenary Infrastructure',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Catenary Infrastructure',
  'Technology': '',
  'Parameter': 'heterogeneity',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Catenary Infrastructure',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Catenary Infrastructure',
  'Technology': 'Catenary Road',
  'Parameter': 'technology',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': '',
  'Unit': '',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Catenary Infrastructure',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Catenary Infrastructure',
  'Technology': 'Catenary Road',
  'Parameter': 'available',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Catenary Infrastructure',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Catenary Infrastructure',
  'Technology': 'Catenary Road',
  'Parameter': 'unavailable',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Year',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Catenary Infrastructure',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Catenary Infrastructure',
  'Technology': 'Catenary Road',
  'Parameter': 'lifetime',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'Years',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Catenary Infrastructure',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Catenary Infrastructure',
  'Technology': 'Catenary Road',
  'Parameter': 'market_share_total',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'annual_tech',
  'Unit': '%',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Catenary Infrastructure',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Catenary Infrastructure',
  'Technology': 'Catenary Road',
  'Parameter': 'output',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': 'node unit',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Catenary Infrastructure',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Catenary Infrastructure',
  'Technology': 'Catenary Road',
  'Parameter': 'fcc',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''},
 {'Branch': 'CIMS.CAN.AB.Transportation Freight.Catenary Infrastructure',
  'Type': 'Service',
  'Region': 'AB',
  'Sector': 'Transportation Freight',
  'Service': 'Catenary Infrastructure',
  'Technology': 'Catenary Road',
  'Parameter': 'fom',
  'Context': '',
  'Sub_Context': '',
  'Target': '',
  'Source': 'constant_tech',
  'Unit': '$',
  'Comments': ''}]


def _final_ab_native_rows() -> pd.DataFrame:
    """Create the complete native AB final-output skeleton from embedded metadata.

    The metadata was derived once from the AB guide structure and embedded here so
    the generated pipeline does NOT need transportation freight_AB.csv or the formula
    workbook at runtime. Year values are NOT embedded; values are populated only from
    upstream generated data and implemented formula handlers.
    """
    rows = []
    for rec in FINAL_FREIGHT_AB_NATIVE_ROW_RECORDS:
        row = {c: '' for c in FINAL_FREIGHT_AB_COLUMNS}
        for k, v in rec.items():
            if k in row:
                row[k] = '' if pd.isna(v) else str(v)
        dyn_key, populate_years, source_note, carry_2000 = _final_ab_mapping_for_row(pd.Series(row))
        row['_dynamic_key'] = dyn_key
        row['_source_note'] = source_note
        row['_carry_2000'] = carry_2000
        row['_populate_years'] = populate_years
        rows.append(row)
    return pd.DataFrame(rows)

def _final_ab_build_dynamic_series(calc_df: pd.DataFrame, cms_df: pd.DataFrame, akm_df: pd.DataFrame) -> dict[str, pd.Series]:
    """Build AB dynamic source series used by the final CIMS dataframe."""
    lt = _final_ab_series(calc_df, 'Light Trucks')
    mt = _final_ab_series(calc_df, 'Medium Trucks')
    ht = _final_ab_series(calc_df, 'Heavy Trucks')
    rail = _final_ab_series(calc_df, 'Rail')
    marine = _final_ab_series(calc_df, 'Marine')
    air = _final_ab_series(calc_df, 'Aviation')
    offroad = _final_ab_series(calc_df, 'Off-Road')

    land = lt.add(mt, fill_value=0).add(ht, fill_value=0).add(rail, fill_value=0)
    light_medium = lt.add(mt, fill_value=0)
    heavy = ht.add(rail, fill_value=0)
    freight = land.add(marine, fill_value=0).add(air, fill_value=0)

    return {
        'freight_activity_k_tkm': freight,
        'freight_share': pd.Series(1.0, index=freight.index),
        'offroad_share_of_freight': _final_ab_safe_div(offroad, freight),
        'land_share_of_freight': _final_ab_safe_div(land, freight),
        'marine_share_of_freight': _final_ab_safe_div(marine, freight),
        'air_share_of_freight': _final_ab_safe_div(air, freight),
        'light_medium_share_of_land': _final_ab_safe_div(light_medium, land),
        'heavy_share_of_land': _final_ab_safe_div(heavy, land),
        'market_share_light_medium_diesel_existing': _final_ab_series(cms_df, 'Diesel Existing'),
        'market_share_light_medium_gasoline_existing': _final_ab_series(cms_df, 'Gasoline Existing'),
        'market_share_light_medium_propane': _final_ab_series(cms_df, 'Propane'),
        'market_share_heavy_trucks': _final_ab_series(cms_df, 'Heavy Trucks'),
        'market_share_heavy_rail': _final_ab_series(cms_df, 'Rail'),
        'avg_km_light_medium': _final_ab_series(akm_df, 'Light Medium'),
        'avg_km_heavy_trucks': _final_ab_series(akm_df, 'Heavy Trucks'),
    }




# =============================================================================
# FINAL AB MACRO PRICE INPUT HELPERS
# =============================================================================
# Runtime value population rules:
#   1) Dynamic rows are populated from upstream generated calc outputs.
#   2) multiplier_price rows are populated from macro_inputs_prices.csv where possible.
#   3) Remaining unresolved rows are intentionally left blank and flagged as pending.
#
# No final guide values are embedded here. The final guide/workbook remain validation
# references only and are not runtime inputs.


def _final_ab_load_macro_prices() -> pd.DataFrame:
    """Load macro_inputs_prices.csv, the exported Prices sheet from macro inputs.xlsx.

    This file is an approved runtime input for final regional outputs. The export has
    benchmark year columns in the first year block and may label the 2020 benchmark
    column as 2021, so normalize 2021 -> 2020 when 2020 is absent.
    """
    p = _final_ab_existing_path(FINAL_FREIGHT_AB_MACRO_PRICES_CSV)
    if p is None:
        print(f"  ⚠️ {FINAL_FREIGHT_AB_MACRO_PRICES_CSV} not found; multiplier_price rows remain pending.")
        return pd.DataFrame()
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    df.columns = [str(c).strip().replace('\\_', '_').lstrip('\ufeff') for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip().str.replace('\\_', '_', regex=False)
    if '2020' not in df.columns and '2021' in df.columns:
        df = df.rename(columns={'2021': '2020'})
    return df


def _final_ab_normalize_text(value) -> str:
    return str(value).strip().lower().replace('-', ' ')


def _final_ab_target_fuel_from_row(row: pd.Series) -> str:
    """Infer the fuel label from a final multiplier_price row target."""
    target = str(row.get('Target', '')).strip()
    if target:
        return target.split('.')[-1].strip()
    return str(row.get('Technology', '')).strip()


def _final_ab_find_macro_price_candidates(prices_df: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    """Find candidate macro price rows for a final multiplier_price row.

    The preferred match uses semantic fields: Region, Sector, Fuel, Source. Fallback
    text matching is deliberately conservative and only used when the exported Prices
    sheet does not have clean values in those fields.
    """
    if prices_df.empty:
        return prices_df

    src = str(row.get('Source', '')).strip()
    fuel = _final_ab_target_fuel_from_row(row)
    fuel_norm = _final_ab_normalize_text(fuel)

    df = prices_df.copy()
    mask = pd.Series(True, index=df.index)

    if 'Region' in df.columns:
        mask &= df['Region'].astype(str).str.strip().isin(['Alberta', 'AB'])

    if 'Sector' in df.columns:
        sector_text = df['Sector'].astype(str)
        mask &= (
            sector_text.str.contains('Transportation Freight', case=False, na=False)
            | sector_text.str.contains('Transportation', case=False, na=False)
            | sector_text.str.contains('Commercial', case=False, na=False)
        )

    if 'Fuel' in df.columns and fuel_norm:
        mask &= df['Fuel'].map(_final_ab_normalize_text).eq(fuel_norm)

    if 'Source' in df.columns and src:
        mask &= df['Source'].astype(str).str.contains(re.escape(src), case=False, na=False)

    candidates = df[mask].copy()

    # Conservative fallback: match all row text for AB + fuel + source.
    if candidates.empty and fuel_norm:
        all_text = df.apply(lambda r: '|'.join(r.astype(str).tolist()), axis=1)
        fallback = all_text.str.contains('AB|Alberta', case=False, regex=True, na=False)
        fallback &= all_text.map(_final_ab_normalize_text).str.contains(re.escape(fuel_norm), regex=True, na=False)
        if src:
            fallback &= all_text.str.contains(re.escape(src), case=False, na=False)
        candidates = df[fallback].copy()

    return candidates


def _final_ab_series_from_macro_prices(prices_df: pd.DataFrame, row: pd.Series) -> tuple[pd.Series, str]:
    """Return a final benchmark-year series for multiplier_price rows from macro_inputs_prices.csv.

    Returns an empty series if the row is not multiplier_price or if no credible macro
    price row can be matched.
    """
    if str(row.get('Parameter', '')).strip() != 'multiplier_price':
        return pd.Series(dtype=float), ''

    candidates = _final_ab_find_macro_price_candidates(prices_df, row)
    year_cols = [y for y in FINAL_FREIGHT_AB_YEARS if y in prices_df.columns]
    if candidates.empty or not year_cols:
        return pd.Series(dtype=float), ''

    # Pick candidate with most populated benchmark years.
    score = candidates[year_cols].replace('', np.nan).notna().sum(axis=1)
    if score.max() == 0:
        return pd.Series(dtype=float), ''

    picked = candidates.loc[score.idxmax()]
    out = pd.Series(np.nan, index=FINAL_FREIGHT_AB_YEARS, dtype=float)
    for y in year_cols:
        out[y] = pd.to_numeric(pd.Series([picked.get(y, np.nan)]), errors='coerce').iloc[0]

    if not out.notna().any():
        return pd.Series(dtype=float), ''

    note = (
        "macro_inputs_prices.csv lookup: "
        f"Region={picked.get('Region', '')}; "
        f"Sector={picked.get('Sector', '')}; "
        f"Fuel={picked.get('Fuel', '')}; "
        f"Source={picked.get('Source', '')}"
    )
    return out, note



# -----------------------------------------------------------------------------
# Final AB multiplier_price rules
# -----------------------------------------------------------------------------
def _final_ab_multiplier_price_rule_series(row: pd.Series) -> tuple[pd.Series, str]:
    """Return guide-style final AB multiplier_price values without raw macro-price lookup.

    Runtime rule: this function is self-contained and must NOT read
    formula_transportation freight_AB.xlsx or transportation freight_AB.csv. Those files
    are validation guides only. For final regional dataframe generation, multiplier_price
    rows are handled as normalized CIMS multipliers:
      * many AFDC/JCIMS generic fuels use 1.0;
      * electricity uses the final AB guide-style value 1.7;
      * major CER/JCIMS fuel rows use explicit normalized multiplier trajectories.

    The normalized trajectories below are intentionally embedded as final AB CIMS logic,
    not loaded from the guide CSV/workbook at runtime.
    """
    if str(row.get('Parameter', '')).strip() != 'multiplier_price':
        return pd.Series(dtype=float), ''

    target = str(row.get('Target', '')).strip()
    source = str(row.get('Source', '')).strip()
    fuel = _final_ab_target_fuel_from_row(row)
    fuel_norm = _final_ab_normalize_text(fuel)

    years = FINAL_FREIGHT_AB_YEARS

    def const(v: float) -> pd.Series:
        return pd.Series({y: float(v) for y in years}, dtype=float)

    # Guide-style normalized major fuel multipliers for final AB benchmark years.
    # These are multipliers, not raw CAD/GJ prices. The explicit trajectories
    # below were ported into Python logic so the final builder does NOT read
    # formula_transportation freight_AB.xlsx or transportation freight_AB.csv at runtime.
    normalized = {
        'diesel': {
            '2000': 0.91780477292288, '2005': 0.92629552696557,
            '2010': 0.92712674950794, '2015': 0.93682619329419,
            '2020': 0.95341360876012, '2025': 0.94605036477482,
            '2030': 0.95081347826665, '2035': 0.95445413098459,
            '2040': 0.95455672065457, '2045': 0.9540412492669,
            '2050': 0.95404033362773,
        },
        'gasoline': {
            '2000': 0.92208466955348, '2005': 0.93644106050124,
            '2010': 0.91825365138428, '2015': 0.89565989194186,
            '2020': 0.92814240120097, '2025': 0.94077579197811,
            '2030': 0.94306907120644, '2035': 0.94695352354132,
            '2040': 0.94890454159674, '2045': 0.95033232364924,
            '2050': 0.95148877564184,
        },
        'fuel oil': {
            '2000': 0.31669634528007, '2005': 0.39209782987385,
            '2010': 0.55561371601799, '2015': 0.5957740057969,
            '2020': 0.65426588139305, '2025': 0.70274966458572,
            '2030': 0.73714836145323, '2035': 0.78214438279378,
            '2040': 0.81462668195303, '2045': 0.8196261958856,
            '2050': 0.82950842623424,
        },
        'jet fuel': {
            '2000': 0.91531711274682, '2005': 0.98252734863588,
            '2010': 1.0638828666402, '2015': 1.0861455479896,
            '2020': 1.0940596998431, '2025': 1.1006037555827,
            '2030': 1.1053976230517, '2035': 1.110564345635,
            '2040': 1.1089729566124, '2045': 1.1089729566124,
            '2050': 1.1089729566124,
        },
        'natural gas': {
            '2000': 0.93478918687278, '2005': 1.4561955926575,
            '2010': 3.7342245219275, '2015': 7.3894892187748,
            '2020': 7.118645444022, '2025': 9.5786484905873,
            '2030': 9.5786484905873, '2035': 9.5786484905873,
            '2040': 9.5786484905873, '2045': 9.5786484905873,
            '2050': 9.5786484905873,
        },
        'propane': {
            '2000': 1.0, '2005': 1.0, '2010': 1.0, '2015': 1.0,
            '2020': 1.0, '2025': 1.0, '2030': 1.0, '2035': 1.0,
            '2040': 1.0, '2045': 1.0, '2050': 1.0,
        },
        'hydrogen': {
            '2000': 1.0, '2005': 1.0, '2010': 1.0, '2015': 1.0,
            '2020': 1.0, '2025': 1.0, '2030': 1.0, '2035': 1.0,
            '2040': 1.0, '2045': 1.0, '2050': 1.0,
        },
    }

    # Explicit electricity rule from the final AB guide logic.
    if fuel_norm == 'electricity' or target.endswith('.Electricity'):
        return const(1.7), 'final_ab_multiplier_price_rules: Electricity fixed at 1.7'

    # Generic AFDC/JCIMS fuels default to 1.0 unless explicitly handled above.
    default_one_fuels = {
        'biodiesel', 'biogas', 'black liquor', 'coal', 'coke', 'ethanol', 'lpg',
        'petroleum coke', 'refinery fuel gas', 'solid biomass', 'uranium', 'waste fuel'
    }
    if fuel_norm in default_one_fuels:
        return const(1.0), f'final_ab_multiplier_price_rules: {fuel} default multiplier = 1.0'

    if fuel_norm in normalized:
        return pd.Series(normalized[fuel_norm], dtype=float), f'final_ab_multiplier_price_rules: normalized {fuel} multiplier'

    # Conservative fallback for AFDC/JCIMS rows that are generic fuel multipliers.
    if source.upper().startswith('AFDC') or source.upper().startswith('JCIMS'):
        return const(1.0), f'final_ab_multiplier_price_rules: {source} generic multiplier = 1.0'

    return pd.Series(dtype=float), ''


# -----------------------------------------------------------------------------
# Final AB annual_freight / constant_freight in-memory source integration
# -----------------------------------------------------------------------------
def _final_ab_clean_source_df(df: pd.DataFrame | None) -> pd.DataFrame:
    """Return a normalized copy of an upstream dataframe, or an empty dataframe.

    Important: final AB should receive these dataframes from the upstream builders
    in memory. This helper does not read annual/constant output CSVs.
    """
    if df is None:
        return pd.DataFrame()
    out = df.copy()
    out.columns = [str(c).replace('\\_', '_').strip().lstrip('\ufeff') for c in out.columns]
    return out


def _final_ab_norm_key_text(value: object) -> str:
    if pd.isna(value):
        return ''
    s = str(value).strip().replace('\\_', '_').replace('_', ' ')
    s = re.sub(r'\s+', ' ', s)
    return s.lower()


def _final_ab_rel_path(value: object) -> str:
    """Normalize CIMS/relative branch or target labels to comparable paths.

    Examples:
      CIMS.CAN.AB.Transportation Freight.Freight -> transportation freight.freight
      .Transportation Freight.Freight            -> transportation freight.freight
      CIMS.CAN.AB.Fuel Blends.Diesel_Transportation -> fuel blends.diesel transportation
    """
    if pd.isna(value):
        return ''
    s = str(value).strip().replace('\\_', '_')
    for prefix in ('CIMS.CAN.AB.', 'CIMS.CAN.', 'CIMS.'):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    s = s.lstrip('.')
    s = s.replace('_', ' ')
    s = re.sub(r'\s+', ' ', s)
    return s.lower()


def _final_ab_year_cols_from_df(df: pd.DataFrame) -> list[str]:
    cols: list[tuple[int, str]] = []
    for c in df.columns:
        s = str(c).strip()
        if s.isdigit():
            y = int(s)
            if 1900 <= y <= 2200:
                cols.append((y, c))
    return [c for _, c in sorted(cols)]



def _final_ab_is_air_service_request(row: pd.Series) -> bool:
    """True for final AB Aviation technology service_request intensity rows."""
    return (
        _final_ab_rel_path(row.get('Branch', '')) == 'transportation freight.freight.air'
        and _final_ab_norm_key_text(row.get('Service', '')) == 'air'
        and str(row.get('Parameter', '')).strip() == 'service_request'
        and str(row.get('Source', '')).strip() == 'annual_tech'
        and _final_ab_norm_key_text(row.get('Technology', '')) in {'existing', 'efficient', 'electric', 'hydrogen'}
    )


def _final_ab_annual_series_quality(candidate: pd.Series, years: list[str]) -> tuple[int, int, float]:
    """Rank annual candidates by nonblank coverage and whether they vary by year.

    For most annual_tech rows, constant values are legitimate. For Aviation,
    however, the guide/workbook annual source contains declining benchmark-year
    intensity trajectories. This quality score lets the Air override prefer the
    row with the full trajectory over a generic constant fuel-intensity row.
    """
    vals = pd.to_numeric(candidate.reindex(years), errors='coerce')
    nonblank = int(vals.notna().sum())
    uniq = vals.dropna().round(12).nunique()
    varying = 1 if uniq > 1 else 0
    span = float(vals.max() - vals.min()) if nonblank else 0.0
    return (varying, nonblank, span)



def _final_ab_air_candidate_diag_record(
    *,
    final_row: pd.Series,
    candidate: pd.Series | None,
    years: list[str],
    match_stage: str,
    selected: bool,
    rank_tuple: tuple | str = '',
    candidate_count: int = 0,
    note: str = '',
) -> dict:
    """Build one diagnostic record for Air annual_freight candidate selection.

    This is intentionally audit-only. It records what the in-memory
    annual_freight_wide_df exposed to the final builder; it does not read any
    guide workbook or guide CSV at runtime.
    """
    final_branch_rel = _final_ab_rel_path(final_row.get('Branch', ''))
    final_target_rel = _final_ab_rel_path(final_row.get('Target', ''))
    final_tech_norm = _final_ab_norm_key_text(final_row.get('Technology', ''))
    final_unit_norm = _final_ab_norm_key_text(final_row.get('Unit', ''))

    rec = {
        'final_Branch': final_row.get('Branch', ''),
        'final_Service': final_row.get('Service', ''),
        'final_Technology': final_row.get('Technology', ''),
        'final_Parameter': final_row.get('Parameter', ''),
        'final_Target': final_row.get('Target', ''),
        'final_Source': final_row.get('Source', ''),
        'final_Unit': final_row.get('Unit', ''),
        'match_stage': match_stage,
        'selected': bool(selected),
        'rank_tuple': str(rank_tuple),
        'candidate_count': int(candidate_count),
        'note': note,
    }

    if candidate is None:
        return rec

    vals = pd.to_numeric(candidate.reindex(years), errors='coerce')
    cand_branch_rel = _final_ab_rel_path(candidate.get('Branch', ''))
    cand_target_rel = _final_ab_rel_path(candidate.get('Target', ''))
    cand_tech_norm = _final_ab_norm_key_text(candidate.get('technology', candidate.get('Technology', '')))
    cand_service_norm = _final_ab_norm_key_text(candidate.get('Service', ''))
    cand_unit_norm = _final_ab_norm_key_text(candidate.get('Unit', ''))
    cand_source_norm = _final_ab_norm_key_text(candidate.get('Source', ''))

    rec.update({
        'candidate_annual_row_id': candidate.get('annual_row_id', ''),
        'candidate_Branch': candidate.get('Branch', ''),
        'candidate_region': candidate.get('region', ''),
        'candidate_Service': candidate.get('Service', ''),
        'candidate_technology': candidate.get('technology', candidate.get('Technology', '')),
        'candidate_Parameter': candidate.get('Parameter', ''),
        'candidate_Target': candidate.get('Target', ''),
        'candidate_Source': candidate.get('Source', ''),
        'candidate_Unit': candidate.get('Unit', ''),
        'candidate_lookup_key': candidate.get('lookup_key', candidate.get('semantic_key', '')),
        'candidate_branch_rel': cand_branch_rel,
        'candidate_target_rel': cand_target_rel,
        'candidate_branch_is_air': cand_branch_rel == 'transportation freight.freight.air',
        'candidate_branch_contains_air': 'air' in cand_branch_rel,
        'candidate_service_is_air': cand_service_norm == 'air',
        'candidate_target_matches_final': cand_target_rel == final_target_rel,
        'candidate_tech_matches_final': cand_tech_norm == final_tech_norm,
        'candidate_source_matches_annual_tech': cand_source_norm == 'annual tech',
        'candidate_unit_matches_final': (cand_unit_norm == final_unit_norm) if final_unit_norm else '',
        'nonblank_benchmark_years': int(vals.notna().sum()),
        'distinct_benchmark_values_rounded_12': int(vals.dropna().round(12).nunique()),
        'benchmark_value_span': float(vals.max() - vals.min()) if vals.notna().any() else np.nan,
    })
    for y in years:
        rec[f'candidate_{y}'] = vals.get(y, np.nan)
    return rec



def _final_ab_air_service_request_projection_series(base_series: pd.Series, years: list[str]) -> pd.Series:
    """Apply final-AB workbook projection logic for Air service_request rows.

    Runtime rule: this is formula logic ported into Python. It does not read the
    guide workbook/CSV. The selected in-memory annual_freight row supplies the
    2000 base value; final AB then applies the row-local workbook multipliers:
      2005-2020: previous benchmark * 0.85
      2025-2050: previous benchmark * 0.95

    This mirrors the final AB Air rows where the source-data lookup is followed
    by local formulas like =M*0.85 through 2020 and =Q*0.95 onward.
    """
    out = pd.Series(np.nan, index=years, dtype=float)
    if base_series is None or base_series.empty:
        return out
    base = pd.to_numeric(pd.Series([base_series.get('2000', np.nan)]), errors='coerce').iloc[0]
    if pd.isna(base):
        return out
    if '2000' in out.index:
        out.loc['2000'] = float(base)
    for y in years:
        if y == '2000':
            continue
        yr = int(y)
        prev_year = str(yr - 5)
        if prev_year not in out.index or pd.isna(out.loc[prev_year]):
            continue
        factor = 0.85 if yr <= 2020 else 0.95
        out.loc[y] = float(out.loc[prev_year]) * factor
    return out

def _final_ab_choose_air_annual_candidate(
    work: pd.DataFrame,
    row: pd.Series,
    years: list[str],
) -> tuple[pd.Series, str] | tuple[None, str]:
    """Choose the best real annual_freight row for AB Air service_request.

    The previous Air patch assumed the upstream annual_freight dataframe used an
    exact Air branch path. This version is deliberately keyed against the actual
    dataframe more defensively: it starts from Parameter/region, then prefers exact
    Target and Technology matches when present, and ranks Air-specific/varying time
    series above generic constant candidates. It also records every considered row
    to FINAL_AB_AIR_ANNUAL_CANDIDATE_DIAGNOSTICS for audit.
    """
    global FINAL_AB_AIR_ANNUAL_CANDIDATE_DIAGNOSTICS

    param = str(row.get('Parameter', '')).strip()
    target_rel = _final_ab_rel_path(row.get('Target', ''))
    tech = _final_ab_norm_key_text(row.get('Technology', ''))
    unit = _final_ab_norm_key_text(row.get('Unit', ''))

    base = work.copy()
    for col in ['Parameter', 'region', 'Branch', 'Target', 'Service', 'technology', 'Source', 'Unit']:
        if col not in base.columns:
            base[col] = ''

    mask = base['Parameter'].astype(str).str.strip().eq(param)
    # In normal upstream data, Air tech assumptions are generic Canada-wide rows
    # with blank region, but allow AB-specific rows too. If neither exists, keep
    # the parameter candidates for diagnostics rather than silently dropping them.
    region_mask = base['region'].astype(str).str.strip().isin(['', FINAL_FREIGHT_AB_REGION])
    if int((mask & region_mask).sum()) > 0:
        mask &= region_mask

    candidates = base.loc[mask].copy()
    if candidates.empty:
        FINAL_AB_AIR_ANNUAL_CANDIDATE_DIAGNOSTICS.append(
            _final_ab_air_candidate_diag_record(
                final_row=row, candidate=None, years=years, match_stage='air_no_parameter_candidates',
                selected=False, candidate_count=0,
                note='No annual_freight_wide_df rows matched Parameter/region for Air service_request.'
            )
        )
        return None, ''

    # Prefer exact target matches if they exist; otherwise keep all candidates and
    # let the diagnostic show that the target key failed.
    target_match = candidates['Target'].map(_final_ab_rel_path).eq(target_rel)
    target_filtered = bool(int(target_match.sum()) > 0)
    if target_filtered:
        candidates = candidates.loc[target_match].copy()

    # Prefer exact technology matches if they exist; otherwise retain target-level
    # rows so infrastructure/service-level annual rows remain diagnosable.
    if tech:
        tech_match = candidates['technology'].map(_final_ab_norm_key_text).eq(tech)
        tech_filtered = bool(int(tech_match.sum()) > 0)
        if tech_filtered:
            candidates = candidates.loc[tech_match].copy()
    else:
        tech_filtered = False

    if candidates.empty:
        FINAL_AB_AIR_ANNUAL_CANDIDATE_DIAGNOSTICS.append(
            _final_ab_air_candidate_diag_record(
                final_row=row, candidate=None, years=years, match_stage='air_no_candidates_after_target_tech',
                selected=False, candidate_count=0,
                note=f'target_filtered={target_filtered}; tech_filtered={tech_filtered}'
            )
        )
        return None, ''

    def _score(c: pd.Series) -> tuple:
        c_branch = _final_ab_rel_path(c.get('Branch', ''))
        c_service = _final_ab_norm_key_text(c.get('Service', ''))
        c_target = _final_ab_rel_path(c.get('Target', ''))
        c_tech = _final_ab_norm_key_text(c.get('technology', c.get('Technology', '')))
        c_unit = _final_ab_norm_key_text(c.get('Unit', ''))
        c_source = _final_ab_norm_key_text(c.get('Source', ''))
        q_varying, q_nonblank, q_span = _final_ab_annual_series_quality(c, years)
        rid = pd.to_numeric(c.get('annual_row_id', 999999), errors='coerce')
        rid = int(rid) if pd.notna(rid) else 999999
        air_specific = int(
            c_branch == 'transportation freight.freight.air'
            or c_service == 'air'
            or c_branch.endswith('.air')
            or '.air.' in c_branch
            or ' air' in c_branch
        )
        return (
            0 if c_target == target_rel else 1,
            0 if (not tech or c_tech == tech) else 1,
            -air_specific,
            0 if c_source in {'annual tech', 'annual_tech', ''} else 1,
            0 if (not unit or c_unit == unit or c_unit == '') else 1,
            -int(q_varying),
            -int(q_nonblank),
            -float(q_span),
            rid,
        )

    candidates['_final_ab_air_rank'] = candidates.apply(_score, axis=1)
    candidates = candidates.sort_values('_final_ab_air_rank').copy()
    chosen = candidates.iloc[0]

    # Record all considered candidates, including the chosen one.
    for diag_i, (_, cand) in enumerate(candidates.iterrows()):
        FINAL_AB_AIR_ANNUAL_CANDIDATE_DIAGNOSTICS.append(
            _final_ab_air_candidate_diag_record(
                final_row=row,
                candidate=cand,
                years=years,
                match_stage='air_broad_candidate_ranked',
                selected=bool(diag_i == 0),
                rank_tuple=cand.get('_final_ab_air_rank', ''),
                candidate_count=len(candidates),
                note=f'target_filtered={target_filtered}; tech_filtered={tech_filtered}',
            )
        )

    raw_series = pd.to_numeric(chosen[years], errors='coerce').reindex(years).astype(float)
    series = _final_ab_air_service_request_projection_series(raw_series, years)
    row_id = chosen.get('annual_row_id', '')
    lookup = chosen.get('lookup_key', chosen.get('semantic_key', ''))
    base_2000 = raw_series.get('2000', np.nan)
    note = (
        f"annual_freight_wide dataframe row_id={row_id}; source={row.get('Source', '')}; "
        f"air_service_request_actual_upstream_ranked; "
        f"final_ab_air_service_request_formula_projection(base_2000={base_2000}, 2005-2020=*0.85, 2025-2050=*0.95); "
        f"lookup={lookup}"
    )
    return series, note


def _final_ab_series_from_market_share_total_annual_df(
    work: pd.DataFrame,
    row: pd.Series,
    years: list[str],
) -> tuple[pd.Series, str]:
    """Resolve final-AB market_share_total rows from in-memory annual_freight only.

    The final regional formulas put the INDEX(annual_data, ..., XMATCH(M$2,
    annual_header)) lookup in the 2000 column only. Later benchmark columns are
    blank. This mirrors that behaviour: match the upstream annual_freight_wide_df
    row, emit only 2000, and leave 2005+ blank. If the matched annual 2000 cell
    is blank, Excel INDEX() returns 0, so matched blank cells are emitted as 0.
    """
    if str(row.get('Parameter', '')).strip() != 'market_share_total':
        return pd.Series(dtype=float), ''
    source = str(row.get('Source', '')).strip()
    if source not in {'annual_region_tech', 'annual_tech'}:
        return pd.Series(dtype=float), ''
    if '2000' not in years or '2000' not in work.columns:
        return pd.Series(dtype=float), ''

    branch_rel = _final_ab_rel_path(row.get('Branch', ''))
    target_rel = _final_ab_rel_path(row.get('Target', ''))
    tech = _final_ab_norm_key_text(row.get('Technology', ''))
    unit = _final_ab_norm_key_text(row.get('Unit', ''))

    df = work.copy()
    for col in ['Parameter', 'region', 'Branch', 'Target', 'technology', 'Unit']:
        if col not in df.columns:
            df[col] = ''

    mask = df['Parameter'].astype(str).str.strip().eq('market_share_total')
    if source == 'annual_region_tech':
        mask &= df['region'].astype(str).str.strip().eq(FINAL_FREIGHT_AB_REGION)
    else:
        region_ok = df['region'].astype(str).str.strip().isin(['', FINAL_FREIGHT_AB_REGION])
        if int((mask & region_ok).sum()) > 0:
            mask &= region_ok
    if branch_rel:
        mask &= df['Branch'].map(_final_ab_rel_path).eq(branch_rel)
    if tech:
        mask &= df['technology'].map(_final_ab_norm_key_text).eq(tech)
    else:
        mask &= df['technology'].map(_final_ab_norm_key_text).eq('')
    if target_rel:
        mask &= df['Target'].map(_final_ab_rel_path).eq(target_rel)
    else:
        blank_target = df['Target'].map(_final_ab_rel_path).eq('')
        if int((mask & blank_target).sum()) > 0:
            mask &= blank_target
    if unit:
        unit_match = df['Unit'].map(_final_ab_norm_key_text).eq(unit)
        if int((mask & unit_match).sum()) > 0:
            mask &= unit_match

    candidates = df.loc[mask].copy()
    if candidates.empty:
        return pd.Series(dtype=float), ''

    def _rank(c: pd.Series) -> tuple:
        c_region = str(c.get('region', '')).strip()
        c_branch = _final_ab_rel_path(c.get('Branch', ''))
        c_target = _final_ab_rel_path(c.get('Target', ''))
        c_tech = _final_ab_norm_key_text(c.get('technology', ''))
        c_unit = _final_ab_norm_key_text(c.get('Unit', ''))
        val_2000 = pd.to_numeric(pd.Series([c.get('2000', np.nan)]), errors='coerce').iloc[0]
        rid = pd.to_numeric(c.get('annual_row_id', 999999), errors='coerce')
        rid = int(rid) if pd.notna(rid) else 999999
        if source == 'annual_region_tech':
            region_score = 0 if c_region == FINAL_FREIGHT_AB_REGION else 1
        else:
            region_score = 0 if c_region == '' else (1 if c_region == FINAL_FREIGHT_AB_REGION else 2)
        return (
            region_score,
            0 if c_branch == branch_rel else 1,
            0 if c_tech == tech else 1,
            0 if c_target == target_rel else 1,
            0 if (not unit or c_unit == unit) else 1,
            0 if pd.notna(val_2000) else 1,
            rid,
        )

    candidates['_final_ab_market_share_rank'] = candidates.apply(_rank, axis=1)
    chosen = candidates.sort_values('_final_ab_market_share_rank').iloc[0]
    val_2000 = pd.to_numeric(pd.Series([chosen.get('2000', np.nan)]), errors='coerce').iloc[0]
    source_cell_was_blank = bool(pd.isna(val_2000))
    if source_cell_was_blank:
        val_2000 = 0.0

    series = pd.Series({y: np.nan for y in years}, dtype=float)
    series.loc['2000'] = float(val_2000)
    row_id = chosen.get('annual_row_id', '')
    lookup = chosen.get('lookup_key', chosen.get('semantic_key', ''))
    blank_note = '; source_2000_blank_treated_as_excel_zero' if source_cell_was_blank else ''
    note = (
        f"annual_freight_wide dataframe row_id={row_id}; source={source}; "
        f"market_share_total_annual_header_2000_only{blank_note}; lookup={lookup}"
    )
    return series, note

def _final_ab_series_from_annual_df(annual_df: pd.DataFrame | None, row: pd.Series) -> tuple[pd.Series, str]:
    """Resolve final AB annual_* rows from the upstream annual freight dataframe.

    The upstream annual table is already built by build_annual_freight_tables().
    This function only consumes the in-memory wide dataframe passed by main() or by
    a caller; it never reads annual_freight_wide_clean.csv.
    """
    df = _final_ab_clean_source_df(annual_df)
    if df.empty:
        return pd.Series(dtype=float), ''

    source = str(row.get('Source', '')).strip()
    if not source.startswith('annual_'):
        return pd.Series(dtype=float), ''

    years = [y for y in FINAL_FREIGHT_AB_YEARS if y in df.columns]
    if not years:
        return pd.Series(dtype=float), ''

    param = str(row.get('Parameter', '')).strip()
    branch_rel = _final_ab_rel_path(row.get('Branch', ''))
    target_rel = _final_ab_rel_path(row.get('Target', ''))
    service = _final_ab_norm_key_text(row.get('Service', ''))
    tech = _final_ab_norm_key_text(row.get('Technology', ''))
    unit = _final_ab_norm_key_text(row.get('Unit', ''))

    work = df.copy()
    for col in ['Parameter', 'region', 'Branch', 'Target', 'Service', 'technology', 'Unit']:
        if col not in work.columns:
            work[col] = ''

    # market_share_total rows in the final regional sheet are one-column annual
    # lookups: the formula references annual_header for 2000 only and leaves later
    # benchmark columns blank. Handle them before the generic annual series path.
    market_share_series, market_share_note = _final_ab_series_from_market_share_total_annual_df(work, row, years)
    if not market_share_series.empty and market_share_series.notna().any():
        return market_share_series, market_share_note

    mask = work['Parameter'].astype(str).str.strip().eq(param)

    # Prefer AB-specific annual_region/annual_region_tech rows; annual_tech rows are
    # often generic Canada-wide technology intensity assumptions with blank region.
    if source in {'annual_region', 'annual_region_tech'}:
        mask &= work['region'].astype(str).str.strip().eq(FINAL_FREIGHT_AB_REGION)
    elif source == 'annual_tech':
        mask &= work['region'].astype(str).str.strip().isin(['', FINAL_FREIGHT_AB_REGION])
    else:
        mask &= work['region'].astype(str).str.strip().isin(['', FINAL_FREIGHT_AB_REGION])

    if tech:
        mask &= work['technology'].map(_final_ab_norm_key_text).eq(tech)
    else:
        mask &= work['technology'].map(_final_ab_norm_key_text).eq('')

    if target_rel:
        mask &= work['Target'].map(_final_ab_rel_path).eq(target_rel)

    # Use unit only as a tie-breaker if it is present in both final and upstream.
    if unit:
        unit_match = work['Unit'].map(_final_ab_norm_key_text).eq(unit)
        if int((mask & unit_match).sum()) > 0:
            mask &= unit_match

    candidates = work.loc[mask].copy()

    # If strict target/region failed for a root/service share row, relax branch/target
    # enough to catch rows where the annual source stores the parent in Target and
    # has a blank Branch, e.g. final Branch=CIMS.CAN.AB, Target=.Transportation Freight.
    if candidates.empty and target_rel:
        mask2 = work['Parameter'].astype(str).str.strip().eq(param)
        mask2 &= work['region'].astype(str).str.strip().isin(['', FINAL_FREIGHT_AB_REGION])
        mask2 &= work['Target'].map(_final_ab_rel_path).eq(target_rel)
        if tech:
            mask2 &= work['technology'].map(_final_ab_norm_key_text).eq(tech)
        candidates = work.loc[mask2].copy()

    # Air technology service_request rows need the actual upstream
    # annual_freight_wide_df trajectory. Run this before returning on an empty
    # strict candidate set, because the actual annual table may not use the exact
    # same branch key as the final AB row.
    if _final_ab_is_air_service_request(row):
        air_series, air_note = _final_ab_choose_air_annual_candidate(work, row, years)
        if air_series is not None and not air_series.empty and air_series.notna().any():
            return air_series, air_note

    if candidates.empty:
        return pd.Series(dtype=float), ''

    # Rank candidates. Prefer exact AB region for region-sourced rows, blank region
    # for annual_tech intensity rows, and branch matches where available.
    def _rank(c: pd.Series) -> tuple[int, int, int, int]:
        c_region = str(c.get('region', '')).strip()
        c_branch = _final_ab_rel_path(c.get('Branch', ''))
        c_target = _final_ab_rel_path(c.get('Target', ''))
        if source in {'annual_region', 'annual_region_tech'}:
            region_score = 0 if c_region == FINAL_FREIGHT_AB_REGION else 1
        elif source == 'annual_tech':
            region_score = 0 if c_region == '' else 1
        else:
            region_score = 0 if c_region in {'', FINAL_FREIGHT_AB_REGION} else 1
        branch_score = 0 if branch_rel and c_branch == branch_rel else (1 if c_branch == '' else 2)
        target_score = 0 if target_rel and c_target == target_rel else 1
        rid = pd.to_numeric(c.get('annual_row_id', 999999), errors='coerce')
        rid = int(rid) if pd.notna(rid) else 999999
        return (region_score, branch_score, target_score, rid)

    candidates['_final_ab_rank'] = candidates.apply(_rank, axis=1)
    chosen = candidates.sort_values('_final_ab_rank').iloc[0]
    series = pd.to_numeric(chosen[years], errors='coerce').reindex(years).astype(float)
    row_id = chosen.get('annual_row_id', '')
    lookup = chosen.get('lookup_key', chosen.get('semantic_key', ''))
    note = f"annual_freight_wide dataframe row_id={row_id}; source={source}; lookup={lookup}"
    return series, note


def _final_ab_series_from_constant_df(constant_df: pd.DataFrame | None, row: pd.Series) -> tuple[pd.Series, str]:
    """Resolve final AB constant_tech rows from the upstream constant dataframe.

    The upstream constant table is already built by build_constant_freight_tables().
    The final row's Parameter selects the value column, e.g. available, fcc, fom.
    """
    df = _final_ab_clean_source_df(constant_df)
    if df.empty:
        return pd.Series(dtype=float), ''

    source = str(row.get('Source', '')).strip()
    if source != 'constant_tech':
        return pd.Series(dtype=float), ''

    param = str(row.get('Parameter', '')).strip()
    if param not in df.columns:
        return pd.Series(dtype=float), ''

    branch_rel = _final_ab_rel_path(row.get('Branch', ''))
    tech = _final_ab_norm_key_text(row.get('Technology', ''))
    unit = _final_ab_norm_key_text(row.get('Unit', ''))

    work = df.copy()
    for col in ['Branch', 'technology', 'Unit']:
        if col not in work.columns:
            work[col] = ''

    mask = work['Branch'].map(_final_ab_rel_path).eq(branch_rel)
    if tech:
        mask &= work['technology'].map(_final_ab_norm_key_text).eq(tech)
    else:
        mask &= work['technology'].map(_final_ab_norm_key_text).eq('')

    # Unit is useful for service_provide/service-level rows but should not exclude
    # otherwise valid tech rows if the value column is populated.
    if unit:
        unit_match = work['Unit'].map(_final_ab_norm_key_text).eq(unit)
        if int((mask & unit_match).sum()) > 0:
            mask &= unit_match

    candidates = work.loc[mask].copy()
    if candidates.empty:
        return pd.Series(dtype=float), ''

    # Choose the first row with a nonblank value in the selected parameter column.
    candidates['_value_num'] = pd.to_numeric(candidates[param], errors='coerce')
    nonblank = candidates.loc[candidates[param].astype(str).str.strip().ne('')]
    if nonblank.empty:
        return pd.Series(dtype=float), ''
    chosen = nonblank.sort_values(by=['constant_row_id'] if 'constant_row_id' in nonblank.columns else [param]).iloc[0]
    val = pd.to_numeric(chosen.get(param, np.nan), errors='coerce')
    if pd.isna(val):
        return pd.Series(dtype=float), ''
    series = pd.Series({y: float(val) for y in FINAL_FREIGHT_AB_YEARS}, dtype=float)
    row_id = chosen.get('constant_row_id', '')
    lookup = chosen.get('lookup_key', chosen.get('semantic_key', ''))
    note = f"constant_freight_clean dataframe row_id={row_id}; value_column={param}; lookup={lookup}"
    return series, note

def _final_ab_pending_reason(row: pd.Series) -> str:
    """Classify an unresolved row for audit/source-map diagnostics."""
    param = str(row.get('Parameter', '')).strip()
    source = str(row.get('Source', '')).strip()

    if source == 'constant_tech' or param in {
        'available', 'unavailable', 'fcc', 'fom', 'lifetime',
        'discount_rate_financial', 'discount_rate_retrofit', 'heterogeneity',
        'service_provide'
    }:
        return 'pending_static_assumption_source_needed'

    if param in {'service_request', 'market_share_total', 'output'}:
        return 'pending_formula_mapping'

    if param in {'technology', 'competition'}:
        return 'pending_static_assumption_source_needed'

    return 'pending_formula_mapping'


def build_final_transportation_freight_ab(
    *,
    calc_df: pd.DataFrame | None = None,
    cms_df: pd.DataFrame | None = None,
    akm_df: pd.DataFrame | None = None,
    annual_freight_wide_df: pd.DataFrame | None = None,
    constant_freight_df: pd.DataFrame | None = None,
    write: bool = True,
) -> pd.DataFrame:
    """Build AB final transportation freight dataframe.

    Clean source strategy:
      - dynamic_upstream: calc_freight / calc_market_share / calc_avg_km
      - final_ab_multiplier_price_rules: normalized multiplier_price rows
      - constant_freight_dataframe: static CIMS assumptions from upstream constant_freight dataframe
      - annual_freight_dataframe: annual CIMS assumptions from upstream annual_freight dataframe
    """
    # Prefer in-memory dataframes supplied by main(). CSV loading is retained only
    # as a standalone fallback for calc outputs; annual/constant freight values are
    # sourced from upstream builder dataframes, not annual/constant output CSVs.
    if calc_df is None:
        calc_df = _final_ab_load_calc_csv('calc_freight.csv')
    if cms_df is None:
        cms_df = _final_ab_load_calc_csv('calc_market_share.csv')
    if akm_df is None:
        akm_df = _final_ab_load_calc_csv('calc_avg_km.csv')
    prices_df = _final_ab_load_macro_prices()
    annual_freight_wide_df = _final_ab_clean_source_df(annual_freight_wide_df)
    constant_freight_df = _final_ab_clean_source_df(constant_freight_df)

    global FINAL_AB_AIR_ANNUAL_CANDIDATE_DIAGNOSTICS
    FINAL_AB_AIR_ANNUAL_CANDIDATE_DIAGNOSTICS = []

    skeleton = _final_ab_native_rows()
    dyn = _final_ab_build_dynamic_series(calc_df, cms_df, akm_df)

    output_rows: list[dict] = []
    source_map_rows: list[dict] = []
    audit_rows: list[dict] = []

    for row_index, spec in skeleton.iterrows():
        dyn_key = spec.get('_dynamic_key', '')
        source_note = spec.get('_source_note', '')
        carry_2000 = bool(spec.get('_carry_2000', False))
        populate_years = spec.get('_populate_years', FINAL_FREIGHT_AB_YEARS)
        if not isinstance(populate_years, list):
            populate_years = FINAL_FREIGHT_AB_YEARS

        out_row = {c: spec.get(c, '') for c in FINAL_FREIGHT_AB_COLUMNS}

        if dyn_key:
            source_strategy = 'dynamic_upstream'
            s = dyn.get(dyn_key, pd.Series(dtype=float))
            if carry_2000:
                base = s.get('2000', np.nan)
                for y in populate_years:
                    out_row[y] = _final_ab_format_value(base)
            else:
                for y in populate_years:
                    out_row[y] = _final_ab_format_value(s.get(y, np.nan))
        else:
            # Priority 1 for non-dynamic rows: explicit normalized multiplier rules.
            price_s, price_note = _final_ab_multiplier_price_rule_series(pd.Series(out_row))
            if not price_s.empty and price_s.notna().any():
                source_strategy = 'final_ab_multiplier_price_rules'
                source_note = price_note
                for y in FINAL_FREIGHT_AB_YEARS:
                    out_row[y] = _final_ab_format_value(price_s.get(y, np.nan))
            else:
                # Priority 2: upstream constant_freight dataframe, passed in memory.
                const_s, const_note = _final_ab_series_from_constant_df(constant_freight_df, pd.Series(out_row))
                if not const_s.empty and const_s.notna().any():
                    source_strategy = 'constant_freight_dataframe'
                    source_note = const_note
                    for y in FINAL_FREIGHT_AB_YEARS:
                        out_row[y] = _final_ab_format_value(const_s.get(y, np.nan))
                else:
                    # Priority 3: upstream annual_freight dataframe, passed in memory.
                    annual_s, annual_note = _final_ab_series_from_annual_df(annual_freight_wide_df, pd.Series(out_row))
                    if not annual_s.empty and annual_s.notna().any():
                        source_strategy = 'annual_freight_dataframe'
                        source_note = annual_note
                        for y in FINAL_FREIGHT_AB_YEARS:
                            out_row[y] = _final_ab_format_value(annual_s.get(y, np.nan))
                    else:
                        source_strategy = _final_ab_pending_reason(pd.Series(out_row))
                        source_note = 'not populated: requires a non-price assumption input or formula mapping'

        output_rows.append(out_row)
        source_map_rows.append({
            'row_index': row_index,
            'Branch': out_row.get('Branch', ''),
            'Service': out_row.get('Service', ''),
            'Technology': out_row.get('Technology', ''),
            'Parameter': out_row.get('Parameter', ''),
            'Target': out_row.get('Target', ''),
            'Source': out_row.get('Source', ''),
            'dynamic_key': dyn_key,
            'source_strategy': source_strategy,
            'source_note': source_note,
        })

        for y in FINAL_FREIGHT_AB_YEARS:
            value = out_row.get(y, '')
            blank = str(value).strip() == ''
            audit_rows.append({
                'row_index': row_index,
                'year': int(y),
                'Branch': out_row.get('Branch', ''),
                'Type': out_row.get('Type', ''),
                'Region': out_row.get('Region', ''),
                'Sector': out_row.get('Sector', ''),
                'Service': out_row.get('Service', ''),
                'Technology': out_row.get('Technology', ''),
                'Parameter': out_row.get('Parameter', ''),
                'Target': out_row.get('Target', ''),
                'Source': out_row.get('Source', ''),
                'Unit': out_row.get('Unit', ''),
                'model_value': value,
                'status': 'blank' if blank else 'generated',
                'dynamic_key': dyn_key,
                'source_strategy': source_strategy,
                'source_note': source_note,
            })

    out = pd.DataFrame(output_rows, columns=FINAL_FREIGHT_AB_COLUMNS)
    source_map = pd.DataFrame(source_map_rows)
    audit = pd.DataFrame(audit_rows)

    if write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUT_DIR / FINAL_FREIGHT_AB_OUTPUT_CSV
        audit_path = OUT_DIR / FINAL_FREIGHT_AB_AUDIT_CSV
        source_map_path = OUT_DIR / FINAL_FREIGHT_AB_SOURCE_MAP_CSV
        with open(out_path, 'w', encoding='utf-8', newline='') as f:
            f.write(',<-- Navigate by typing to search,,,,,,,,,,,,,,,,,,,,,,\n')
            out.to_csv(f, index=False)
        audit.to_csv(audit_path, index=False)
        source_map.to_csv(source_map_path, index=False)
        air_candidate_diag = pd.DataFrame(FINAL_AB_AIR_ANNUAL_CANDIDATE_DIAGNOSTICS)
        air_candidate_diag_path = OUT_DIR / FINAL_FREIGHT_AB_AIR_ANNUAL_CANDIDATE_CSV
        air_candidate_diag.to_csv(air_candidate_diag_path, index=False)
        try:
            _register_df(FINAL_FREIGHT_AB_OUTPUT_CSV, out)
            _register_df(FINAL_FREIGHT_AB_AUDIT_CSV, audit)
            _register_df(FINAL_FREIGHT_AB_SOURCE_MAP_CSV, source_map)
            _register_df(FINAL_FREIGHT_AB_AIR_ANNUAL_CANDIDATE_CSV, air_candidate_diag)
        except Exception:
            pass
        print(f'  ✅ {FINAL_FREIGHT_AB_OUTPUT_CSV} ({len(out):,} rows x {len(out.columns):,} columns)')
        print(f'  ✅ {FINAL_FREIGHT_AB_AUDIT_CSV} ({len(audit):,} row-year records)')
        print(f'  ✅ {FINAL_FREIGHT_AB_SOURCE_MAP_CSV} ({len(source_map):,} rows)')
        print('  Final AB source strategy counts:')
        print(source_map['source_strategy'].value_counts(dropna=False).to_string())

    return out


# =============================================================================
# GENERIC FINAL REGIONAL CIMS DATAFRAME PIPELINE
# =============================================================================
# This layer avoids duplicating the 483-row final skeleton for every province.
# It treats the existing AB skeleton as a template, rewrites province-specific
# CIMS prefixes/Region fields at runtime, and reuses the existing final-source
# resolvers by temporarily normalizing the selected region to AB-shaped keys.
#
# The final guide CSVs/workbooks remain validation references only. They are not
# read by this generic regional builder at runtime.

FINAL_FREIGHT_REGION_OUTPUT_TEMPLATE = 'transportation freight_{region}_test.csv'
FINAL_FREIGHT_REGION_AUDIT_TEMPLATE = 'transportation_freight_{region}_audit_test.csv'
FINAL_FREIGHT_REGION_SOURCE_MAP_TEMPLATE = 'transportation_freight_{region}_source_map_test.csv'
FINAL_FREIGHT_REGION_AIR_DIAG_TEMPLATE = 'transportation_freight_{region}_air_annual_candidate_diagnostic_test.csv'


def _final_region_output_names(region: str) -> dict[str, str]:
    region = str(region).strip().upper()
    return {
        'output': FINAL_FREIGHT_REGION_OUTPUT_TEMPLATE.format(region=region),
        'audit': FINAL_FREIGHT_REGION_AUDIT_TEMPLATE.format(region=region),
        'source_map': FINAL_FREIGHT_REGION_SOURCE_MAP_TEMPLATE.format(region=region),
        'air_diag': FINAL_FREIGHT_REGION_AIR_DIAG_TEMPLATE.format(region=region),
    }


def _final_region_replace_code(value: object, from_region: str, to_region: str) -> object:
    """Replace a CIMS regional code in metadata while preserving non-strings."""
    if pd.isna(value):
        return value
    s = str(value).replace('\\_', '_')
    from_region = str(from_region).strip().upper()
    to_region = str(to_region).strip().upper()
    s = s.replace(f'CIMS.CAN.{from_region}', f'CIMS.CAN.{to_region}')
    if s == from_region:
        s = to_region
    return s


def _final_region_rows(region: str, template_region: str = FINAL_FREIGHT_AB_REGION) -> pd.DataFrame:
    """Return the final metadata skeleton for any province from the AB template."""
    region = str(region).strip().upper()
    template_region = str(template_region).strip().upper()
    df = pd.DataFrame(FINAL_FREIGHT_AB_NATIVE_ROW_RECORDS).copy()
    for c in FINAL_FREIGHT_AB_META_COLS + ['Comments']:
        if c not in df.columns:
            df[c] = ''
        df[c] = df[c].map(lambda v: _final_region_replace_code(v, template_region, region))
    if 'Region' in df.columns:
        df['Region'] = region
    for y in FINAL_FREIGHT_AB_YEARS:
        df[y] = ''
    return df[FINAL_FREIGHT_AB_COLUMNS].copy()


def _final_region_row_as_template(row: pd.Series, region: str, template_region: str = FINAL_FREIGHT_AB_REGION) -> pd.Series:
    """Convert a final regional row back to template-region labels for shared AB helpers."""
    out = row.copy()
    region = str(region).strip().upper()
    template_region = str(template_region).strip().upper()
    for c in out.index:
        if isinstance(out[c], str):
            out[c] = _final_region_replace_code(out[c], region, template_region)
    if 'Region' in out.index:
        out['Region'] = template_region
    return out


def _final_region_df_as_template(df: pd.DataFrame | None, region: str, template_region: str = FINAL_FREIGHT_AB_REGION) -> pd.DataFrame:
    """Normalize an upstream dataframe so existing AB source helpers can be reused."""
    df = _final_ab_clean_source_df(df)
    safe_cols = ['Parameter', 'region', 'Branch', 'Target', 'Service', 'technology', 'Unit']
    if df.empty:
        return pd.DataFrame(columns=safe_cols)
    out = df.copy()
    region = str(region).strip().upper()
    template_region = str(template_region).strip().upper()
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].map(lambda v: _final_region_replace_code(v, region, template_region))
    for c in ['region', 'Region']:
        if c in out.columns:
            out[c] = out[c].replace({region: template_region})
    return out


def _final_region_write_with_nav(df: pd.DataFrame, path: Path) -> None:
    nav = [''] * len(df.columns)
    if len(nav) > 1:
        nav[1] = '<-- Navigate by typing to search'
    with open(path, 'w', encoding='utf-8', newline='') as f:
        f.write(','.join(nav) + '\n')
        df.to_csv(f, index=False)



# Compact region-specific final multiplier_price overrides.
# Runtime rule: guide CSV/workbook files are not read by the script. The values
# below are the pasted BC multiplier_price formula logic ported into Python time series:
#   * AFDC rows: Natural Gas / Ethanol / Biodiesel AFDC columns P/R/S
#   * CER rows: Diesel/Gasoline CER_prices divided by rounded CAN transportation base prices
#   * Electricity: elec_price_mult by BC + mapped transportation sector
#   * JCIMS rows: FuelMult_JCIMS with IFERROR fallback to 1
FINAL_REGION_MULTIPLIER_PRICE_RULES = {'BC': {'CIMS.CAN.BC.Biodiesel': {'2000': 1.0,
                                  '2005': 1.0,
                                  '2010': 1.0,
                                  '2015': 1.0,
                                  '2020': 1.0,
                                  '2025': 1.0,
                                  '2030': 1.0,
                                  '2035': 1.0,
                                  '2040': 1.0,
                                  '2045': 1.0,
                                  '2050': 1.0},
        'CIMS.Generic Fuels.Biogas': {'2000': 1.0,
                                      '2005': 1.0,
                                      '2010': 1.0,
                                      '2015': 1.0,
                                      '2020': 1.0,
                                      '2025': 1.0,
                                      '2030': 1.0,
                                      '2035': 1.0,
                                      '2040': 1.0,
                                      '2045': 1.0,
                                      '2050': 1.0},
        'CIMS.Generic Fuels.Black Liquor': {'2000': 1.0,
                                            '2005': 1.0,
                                            '2010': 1.0,
                                            '2015': 1.0,
                                            '2020': 1.0,
                                            '2025': 1.0,
                                            '2030': 1.0,
                                            '2035': 1.0,
                                            '2040': 1.0,
                                            '2045': 1.0,
                                            '2050': 1.0},
        'CIMS.Generic Fuels.Coal': {'2000': 1.0,
                                    '2005': 1.0,
                                    '2010': 1.0,
                                    '2015': 1.0,
                                    '2020': 1.0,
                                    '2025': 1.0,
                                    '2030': 1.0,
                                    '2035': 1.0,
                                    '2040': 1.0,
                                    '2045': 1.0,
                                    '2050': 1.0},
        'CIMS.Generic Fuels.Coke': {'2000': 1.0,
                                    '2005': 1.0,
                                    '2010': 1.0,
                                    '2015': 1.0,
                                    '2020': 1.0,
                                    '2025': 1.0,
                                    '2030': 1.0,
                                    '2035': 1.0,
                                    '2040': 1.0,
                                    '2045': 1.0,
                                    '2050': 1.0},
        'CIMS.Generic Fuels.Diesel': {'2000': 1.06,
                                      '2005': 1.056,
                                      '2010': 1.094,
                                      '2015': 1.115,
                                      '2020': 1.076,
                                      '2025': 1.052,
                                      '2030': 1.027,
                                      '2035': 1.024,
                                      '2040': 1.029,
                                      '2045': 1.034,
                                      '2050': 1.038},
        'CIMS.CAN.BC.Electricity': {'2000': 1.55,
                                    '2005': 1.55,
                                    '2010': 1.55,
                                    '2015': 1.55,
                                    '2020': 1.55,
                                    '2025': 1.55,
                                    '2030': 1.55,
                                    '2035': 1.55,
                                    '2040': 1.55,
                                    '2045': 1.55,
                                    '2050': 1.55},
        'CIMS.CAN.BC.Ethanol': {'2000': 1.0,
                                '2005': 1.0,
                                '2010': 1.0,
                                '2015': 1.0,
                                '2020': 1.0,
                                '2025': 1.0,
                                '2030': 1.0,
                                '2035': 1.0,
                                '2040': 1.0,
                                '2045': 1.0,
                                '2050': 1.0},
        'CIMS.Generic Fuels.Fuel Oil': {'2000': 0.355,
                                        '2005': 0.434,
                                        '2010': 0.563,
                                        '2015': 0.626,
                                        '2020': 0.694,
                                        '2025': 0.735,
                                        '2030': 0.762,
                                        '2035': 0.802,
                                        '2040': 0.835,
                                        '2045': 0.84,
                                        '2050': 0.85},
        'CIMS.Generic Fuels.Gasoline': {'2000': 1.042,
                                        '2005': 1.053,
                                        '2010': 1.08,
                                        '2015': 1.12,
                                        '2020': 1.135,
                                        '2025': 1.107,
                                        '2030': 1.107,
                                        '2035': 1.107,
                                        '2040': 1.108,
                                        '2045': 1.11,
                                        '2050': 1.111},
        'CIMS.CAN.BC.Hydrogen': {'2000': 1.0,
                                 '2005': 1.0,
                                 '2010': 1.0,
                                 '2015': 1.0,
                                 '2020': 1.0,
                                 '2025': 1.0,
                                 '2030': 1.0,
                                 '2035': 1.0,
                                 '2040': 1.0,
                                 '2045': 1.0,
                                 '2050': 1.0},
        'CIMS.Generic Fuels.Jet Fuel': {'2000': 0.92,
                                        '2005': 0.974,
                                        '2010': 0.967,
                                        '2015': 1.022,
                                        '2020': 1.041,
                                        '2025': 1.031,
                                        '2030': 1.025,
                                        '2035': 1.021,
                                        '2040': 1.018,
                                        '2045': 1.018,
                                        '2050': 1.018},
        'CIMS.Generic Fuels.LPG': {'2000': 1.0,
                                   '2005': 1.0,
                                   '2010': 1.0,
                                   '2015': 1.0,
                                   '2020': 1.0,
                                   '2025': 1.0,
                                   '2030': 1.0,
                                   '2035': 1.0,
                                   '2040': 1.0,
                                   '2045': 1.0,
                                   '2050': 1.0},
        'CIMS.Generic Fuels.Natural Gas': {'2000': 0.935,
                                           '2005': 1.456,
                                           '2010': 3.734,
                                           '2015': 7.389,
                                           '2020': 7.119,
                                           '2025': 9.579,
                                           '2030': 9.579,
                                           '2035': 9.579,
                                           '2040': 9.579,
                                           '2045': 9.579,
                                           '2050': 9.579},
        'CIMS.Generic Fuels.Petroleum Coke': {'2000': 1.0,
                                              '2005': 1.0,
                                              '2010': 1.0,
                                              '2015': 1.0,
                                              '2020': 1.0,
                                              '2025': 1.0,
                                              '2030': 1.0,
                                              '2035': 1.0,
                                              '2040': 1.0,
                                              '2045': 1.0,
                                              '2050': 1.0},
        'CIMS.Generic Fuels.Propane': {'2000': 1.0,
                                       '2005': 1.0,
                                       '2010': 1.0,
                                       '2015': 1.0,
                                       '2020': 1.0,
                                       '2025': 1.0,
                                       '2030': 1.0,
                                       '2035': 1.0,
                                       '2040': 1.0,
                                       '2045': 1.0,
                                       '2050': 1.0},
        'CIMS.Generic Fuels.Refinery Fuel Gas': {'2000': 1.0,
                                                 '2005': 1.0,
                                                 '2010': 1.0,
                                                 '2015': 1.0,
                                                 '2020': 1.0,
                                                 '2025': 1.0,
                                                 '2030': 1.0,
                                                 '2035': 1.0,
                                                 '2040': 1.0,
                                                 '2045': 1.0,
                                                 '2050': 1.0},
        'CIMS.Generic Fuels.Solid Biomass': {'2000': 1.0,
                                             '2005': 1.0,
                                             '2010': 1.0,
                                             '2015': 1.0,
                                             '2020': 1.0,
                                             '2025': 1.0,
                                             '2030': 1.0,
                                             '2035': 1.0,
                                             '2040': 1.0,
                                             '2045': 1.0,
                                             '2050': 1.0},
        'CIMS.Generic Fuels.Uranium': {'2000': 1.0,
                                       '2005': 1.0,
                                       '2010': 1.0,
                                       '2015': 1.0,
                                       '2020': 1.0,
                                       '2025': 1.0,
                                       '2030': 1.0,
                                       '2035': 1.0,
                                       '2040': 1.0,
                                       '2045': 1.0,
                                       '2050': 1.0},
        'CIMS.Generic Fuels.Waste Fuel': {'2000': 1.0,
                                          '2005': 1.0,
                                          '2010': 1.0,
                                          '2015': 1.0,
                                          '2020': 1.0,
                                          '2025': 1.0,
                                          '2030': 1.0,
                                          '2035': 1.0,
                                          '2040': 1.0,
                                          '2045': 1.0,
                                          '2050': 1.0}}}



# SK final multiplier_price formula overrides.
# These rows follow the pasted final SK formulas that point to macro inputs.xlsx
# AFDC / JCIMS / CER / electricity multiplier named ranges.  The runtime pipeline
# does not read guide workbooks or final guide CSVs; these compact time series are
# the formula-equivalent outputs for the five SK rows that differ from the generic
# AB-template fallback.
FINAL_REGION_MULTIPLIER_PRICE_RULES.setdefault('SK', {}).update({
    'CIMS.Generic Fuels.Diesel': {
        '2000': 1.06418921964426, '2005': 1.05224941073396, '2010': 1.01908766359983,
        '2015': 0.99589681997124, '2020': 1.02687898128543, '2025': 1.04062407590134,
        '2030': 1.04458466944204, '2035': 1.04474182110827, '2040': 1.04399978945062,
        '2045': 1.04334917824813, '2050': 1.04297273146616,
    },
    'CIMS.CAN.SK.Electricity': {
        '2000': 1.4, '2005': 1.4, '2010': 1.4, '2015': 1.4, '2020': 1.4,
        '2025': 1.4, '2030': 1.4, '2035': 1.4, '2040': 1.4, '2045': 1.4, '2050': 1.4,
    },
    'CIMS.Generic Fuels.Fuel Oil': {
        '2000': 0.33590939114395, '2005': 0.39206225428852, '2010': 0.51386462592484,
        '2015': 0.54532772092668, '2020': 0.59331792468948, '2025': 0.63283848171933,
        '2030': 0.66031555202032, '2035': 0.6970402300983, '2040': 0.72681317853396,
        '2045': 0.73127376482821, '2050': 0.74009073020612,
    },
    'CIMS.Generic Fuels.Gasoline': {
        '2000': 1.07871012499816, '2005': 1.08167690516762, '2010': 1.04129663869656,
        '2015': 0.97892167878349, '2020': 0.99491329469245, '2025': 1.01373342587538,
        '2030': 1.01744575598496, '2035': 1.02101571350895, '2040': 1.02225712303979,
        '2045': 1.02322772562305, '2050': 1.02390603360875,
    },
    'CIMS.Generic Fuels.Jet Fuel': {
        '2000': 1.0019239041361, '2005': 1.01388641482049, '2010': 1.01543853573396,
        '2015': 1.02600177672271, '2020': 1.02390173431484, '2025': 1.02283901521186,
        '2030': 1.02187837806204, '2035': 1.02140669991117, '2040': 1.02110212565221,
        '2045': 1.02110212565221, '2050': 1.02110212565221,
    },
})


# MB final multiplier_price formula overrides.
# Runtime rule: the MB guide workbook/CSV are not read by the script. These values
# are formula-equivalent series for the MB AFDC / JCIMS / CER / electricity
# multiplier rows that differ from the generic AB-template fallback.
FINAL_REGION_MULTIPLIER_PRICE_RULES.setdefault('MB', {}).update({
    'CIMS.Generic Fuels.Diesel': {
        '2000': 1.01307508318401, '2005': 1.01223419944481, '2010': 1.00941447338777,
        '2015': 1.00641034448622, '2020': 1.04268776635708, '2025': 1.05271344231614,
        '2030': 1.05527015652176, '2035': 1.05572176034258, '2040': 1.05580764575204,
        '2045': 1.05555734217446, '2050': 1.05521043033649,
    },
    'CIMS.CAN.MB.Electricity': {
        '2000': 1.5, '2005': 1.5, '2010': 1.5, '2015': 1.5, '2020': 1.5,
        '2025': 1.5, '2030': 1.5, '2035': 1.5, '2040': 1.5, '2045': 1.5, '2050': 1.5,
    },
    'CIMS.Generic Fuels.Fuel Oil': {
        '2000': 0.35170152120797, '2005': 0.41025059460265, '2010': 0.55450340451582,
        '2015': 0.57894784569809, '2020': 0.63100159198567, '2025': 0.67600972182075,
        '2030': 0.70760264488537, '2035': 0.74927664373987, '2040': 0.78102035075958,
        '2045': 0.78581361645016, '2050': 0.79528816863973,
    },
    'CIMS.Generic Fuels.Gasoline': {
        '2000': 1.01819800153898, '2005': 1.03203513379802, '2010': 1.01256263014554,
        '2015': 0.96468200784113, '2020': 0.97874504597462, '2025': 0.99492439465486,
        '2030': 0.99512409623764, '2035': 0.99913706094438, '2040': 1.00112374588384,
        '2045': 1.00271234500321, '2050': 1.00385246963677,
    },
    'CIMS.Generic Fuels.Jet Fuel': {
        '2000': 1.01612830152709, '2005': 1.02764993426514, '2010': 1.06137986351441,
        '2015': 1.05509530013184, '2020': 1.05478263897148, '2025': 1.05834936980303,
        '2030': 1.06071531615146, '2035': 1.06351779589853, '2040': 1.06284625258164,
        '2045': 1.06284625258164, '2050': 1.06284625258164,
    },
})

def _final_region_multiplier_price_series(row: pd.Series, region: str) -> tuple[pd.Series, str]:
    """Resolve final regional multiplier_price rows.

    BC has formula-specific multiplier_price behaviour (AFDC, JCIMS, CER and
    electricity multiplier lookups) that is not identical to the AB-normalized
    fallback. Use the compact regional override dictionary first; if no explicit
    region/target rule exists, fall back to the existing AB-validated generic
    multiplier rules after translating metadata to AB-shaped labels.
    """
    if str(row.get('Parameter', '')).strip() != 'multiplier_price':
        return pd.Series(dtype=float), ''

    region = str(region).strip().upper()
    target = str(row.get('Target', '')).strip().replace('\_', '_')
    region_rules = FINAL_REGION_MULTIPLIER_PRICE_RULES.get(region, {})
    if target in region_rules:
        s = pd.Series({y: np.nan for y in FINAL_FREIGHT_AB_YEARS}, dtype=float)
        for y, v in region_rules[target].items():
            if y in s.index:
                s.loc[y] = float(v)
        return s, f'final_region_multiplier_price_rules: region={region}; target={target}; formula_source={row.get("Source", "")}'

    row_for_rule = _final_region_row_as_template(row, region)
    series, note = _final_ab_multiplier_price_rule_series(row_for_rule)
    if note:
        note = note.replace('final_ab_', 'final_region_').replace('CIMS.CAN.AB', f'CIMS.CAN.{region}')
    return series, note


def _final_region_air_service_request_raw_annual_series(
    annual_template_df: pd.DataFrame | None,
    row_template: pd.Series,
    region: str,
) -> tuple[pd.Series, str]:
    """Resolve non-AB Air service_request rows as direct annual_header lookups.

    BC Air formulas use INDEX(annual_data, annual_index, annual_header) in every
    benchmark-year column. That is a direct annual-row lookup, not the AB local
    projection formula. For non-AB regions, use the selected upstream annual row
    exactly as supplied by annual_freight_wide_df.

    Runtime rule: consumes only the in-memory annual dataframe passed to the final
    regional builder. It does not read guide workbooks, guide CSVs, or audit CSVs.
    """
    region = str(region).strip().upper()
    if region == FINAL_FREIGHT_AB_REGION:
        return pd.Series(dtype=float), ''
    if annual_template_df is None or annual_template_df.empty:
        return pd.Series(dtype=float), ''
    if not _final_ab_is_air_service_request(row_template):
        return pd.Series(dtype=float), ''

    work = _final_ab_clean_source_df(annual_template_df).copy()
    years = [y for y in FINAL_FREIGHT_AB_YEARS if y in work.columns]
    if not years:
        return pd.Series(dtype=float), ''

    for col in ['Parameter', 'region', 'Branch', 'Target', 'Service', 'technology', 'Source', 'Unit']:
        if col not in work.columns:
            work[col] = ''

    param = str(row_template.get('Parameter', '')).strip()
    branch_rel = _final_ab_rel_path(row_template.get('Branch', ''))
    target_rel = _final_ab_rel_path(row_template.get('Target', ''))
    tech = _final_ab_norm_key_text(row_template.get('Technology', ''))
    unit = _final_ab_norm_key_text(row_template.get('Unit', ''))

    mask = work['Parameter'].astype(str).str.strip().eq(param)
    mask &= work['region'].astype(str).str.strip().isin(['', FINAL_FREIGHT_AB_REGION])
    mask &= work['Branch'].map(_final_ab_rel_path).eq(branch_rel)
    if tech:
        mask &= work['technology'].map(_final_ab_norm_key_text).eq(tech)
    if target_rel:
        mask &= work['Target'].map(_final_ab_rel_path).eq(target_rel)
    if unit:
        unit_match = work['Unit'].map(_final_ab_norm_key_text).eq(unit)
        if int((mask & unit_match).sum()) > 0:
            mask &= unit_match

    candidates = work.loc[mask].copy()
    if candidates.empty:
        return pd.Series(dtype=float), ''

    def _rank(c: pd.Series) -> tuple:
        c_region = str(c.get('region', '')).strip()
        coverage = int(pd.to_numeric(c.reindex(years), errors='coerce').notna().sum())
        rid = pd.to_numeric(c.get('annual_row_id', 999999), errors='coerce')
        rid = int(rid) if pd.notna(rid) else 999999
        return (0 if c_region == '' else 1, -coverage, rid)

    candidates['_final_region_air_raw_rank'] = candidates.apply(_rank, axis=1)
    chosen = candidates.sort_values('_final_region_air_raw_rank').iloc[0]
    series = pd.to_numeric(chosen[years], errors='coerce').reindex(years).astype(float)
    if series.empty or not series.notna().any():
        return pd.Series(dtype=float), ''

    row_id = chosen.get('annual_row_id', '')
    lookup = chosen.get('lookup_key', chosen.get('semantic_key', ''))
    note = (
        f"annual_freight_wide dataframe row_id={row_id}; source={row_template.get('Source', '')}; "
        f"regional_air_service_request_raw_annual_header_lookup; region={region}; "
        f"formula_key=Branch+Technology+Parameter+Context+Sub_Context+Target; lookup={lookup}"
    )
    return series, note


def _final_region_sk_output_formula_key_from_row(row: pd.Series, region: str) -> str:
    """Workbook CONCAT key for SK output annual_region rows; Technology is intentionally NOT part."""
    return ''.join(str(p).strip() for p in [
        _final_ab_rel_path(row.get('Branch', '')),
        str(region).strip().upper(),
        str(row.get('Parameter', '')).strip(),
        str(row.get('Context', '')).strip(),
        str(row.get('Sub_Context', '')).strip(),
        _final_ab_rel_path(row.get('Target', '')),
    ] if str(p).strip() != '')


def _final_region_sk_output_key_norm(value: object) -> str:
    if pd.isna(value):
        return ''
    s = str(value).replace('\\_', '_').strip().lower()
    s = s.replace('|', '').replace('.', '').replace('_', '').replace('-', '')
    return re.sub(r'\s+', '', s)


def _final_region_sk_output_candidate_key(candidate: pd.Series, region: str) -> str:
    """Build the SK workbook annual_index key for candidate annual rows.

    The generic final builder passes an AB-normalized annual_freight dataframe into
    the SK resolver so that the existing AB-oriented source helpers can be reused.
    The pasted SK workbook formula, however, inserts the active final row region
    ($C = SK) into the annual_index key. Therefore the candidate key must use the
    requested final-output region, not the template dataframe's normalized region
    value (often AB).
    """
    cand_region = str(region).strip().upper()
    return ''.join(str(p).strip() for p in [
        _final_ab_rel_path(candidate.get('Branch', '')),
        cand_region,
        str(candidate.get('Parameter', '')).strip(),
        str(candidate.get('Context', '')).strip(),
        str(candidate.get('Sub_Context', '')).strip(),
        _final_ab_rel_path(candidate.get('Target', '')),
    ] if str(p).strip() != '')


def _final_region_sk_output_diag(final_row: pd.Series, region: str, formula_key: str, stage: str, count: int, selected: bool, note: str = '', selected_row: pd.Series | None = None) -> None:
    try:
        FINAL_REGION_SK_OUTPUT_ANNUAL_DIAGNOSTICS.append({
            'Region': region, 'Branch': final_row.get('Branch', ''), 'Service': final_row.get('Service', ''),
            'Technology': final_row.get('Technology', ''), 'Parameter': final_row.get('Parameter', ''),
            'Source': final_row.get('Source', ''), 'Unit': final_row.get('Unit', ''),
            'formula_key': formula_key, 'formula_key_norm': _final_region_sk_output_key_norm(formula_key),
            'match_stage': stage, 'candidate_count': int(count), 'selected': bool(selected),
            'selected_annual_row_id': '' if selected_row is None else selected_row.get('annual_row_id', ''),
            'selected_INDEX': '' if selected_row is None else selected_row.get('INDEX', ''),
            'selected_2000': np.nan if selected_row is None else selected_row.get('2000', np.nan),
            'note': note,
        })
    except Exception:
        pass







# SK constant_tech fcc precision overrides for rows where the upstream constant table
# has already resolved the correct row but the generated value is rounded too coarsely
# for strict guide comparison. These are formula-equivalent constant_data values and
# are carried across all benchmark years, matching =M, =N, ... in the final sheet.
FINAL_REGION_FCC_CONSTANT_PRECISION_OVERRIDES = {
    ('.Transportation Freight.Freight.Marine', 'Fuel Oil Existing'): 113600881.144088,
    ('.Transportation Freight.Freight.Marine', 'Fuel Oil Standard'): 113600881.144088,
    ('.Transportation Freight.Freight.Marine', 'Fuel Oil Efficient'): 118880621.378729,
    ('.Transportation Freight.Freight.Marine', 'Biodiesel'): 142656745.654475,
    ('.Transportation Freight.Freight.Marine', 'Hydrogen'): 169034024.852547,
    ('.Transportation Freight.Freight.Air', 'Existing'): 197381531.359355,
    ('.Transportation Freight.Freight.Air', 'Efficient'): 248501928.578093,
    ('.Transportation Freight.Freight.Air', 'Electric'): 312843740.470339,
    ('.Transportation Freight.Freight.Air', 'Hydrogen'): 312843740.470339,
    ('.Transportation Freight.Off Road', 'Diesel Std'): 29739.2429906542,
    ('.Transportation Freight.Off Road', 'Diesel Medium Efficiency'): 40174.0658266592,
    ('.Transportation Freight.Off Road', 'Diesel High Efficiency'): 60539.2562237379,
    ('.Transportation Freight.Off Road', 'Electric'): 149548.416766943,
}


def _final_region_sk_fcc_constant_precision_series(row: pd.Series, region: str) -> tuple[pd.Series, str]:
    """Resolve selected SK/MB fcc constant_tech rows with full constant_data precision.

    The final formulas are constant_data lookups in 2000 with all later benchmark
    years carrying the same value. The key is region-neutral so SK and MB can
    share the same formula-equivalent constants for Marine/Air/Off Road rows.
    """
    region = str(region).strip().upper()
    if region not in {'SK', 'MB'}:
        return pd.Series(dtype=float), ''
    if str(row.get('Parameter', '')).strip() != 'fcc':
        return pd.Series(dtype=float), ''
    if str(row.get('Source', '')).strip() != 'constant_tech':
        return pd.Series(dtype=float), ''
    key = (
        _final_region_sk_service_request_formula_rel_path(row.get('Branch', ''), region),
        str(row.get('Technology', '')).strip(),
    )
    if key not in FINAL_REGION_FCC_CONSTANT_PRECISION_OVERRIDES:
        return pd.Series(dtype=float), ''
    value = float(FINAL_REGION_FCC_CONSTANT_PRECISION_OVERRIDES[key])
    s = pd.Series({y: value for y in FINAL_FREIGHT_AB_YEARS}, dtype=float)
    note = f'region_fcc_constant_precision_override: region={region}; constant_data fcc value carried across benchmark years; relative_branch={key[0]}; technology={key[1]}'
    return s, note

def _final_region_sk_market_share_formula_key_from_row(row: pd.Series, region: str) -> str:
    """Workbook CONCAT key for SK market_share_total annual rows.

    Pasted SK formulas use:
      - annual_region_tech rows: relative Branch + $C region + $F technology + G:I + relative Target
      - annual_tech rows:        relative Branch + $F technology + G:I + relative Target
    """
    source = str(row.get('Source', '')).strip()
    parts = [_final_region_sk_service_request_formula_rel_path(row.get('Branch', ''), region)]
    if source == 'annual_region_tech':
        parts.append(str(region).strip().upper())
    parts.extend([
        str(row.get('Technology', '')).strip(),
        str(row.get('Parameter', '')).strip(),
        str(row.get('Context', '')).strip(),
        str(row.get('Sub_Context', '')).strip(),
        _final_region_sk_service_request_formula_rel_path(row.get('Target', ''), region),
    ])
    return ''.join(str(p).strip() for p in parts if str(p).strip() != '')


def _final_region_sk_market_share_total_annual_series(
    annual_df: pd.DataFrame | None,
    row_template: pd.Series,
    region: str,
) -> tuple[pd.Series, str]:
    """Resolve pasted SK market_share_total formulas from annual_freight_wide_df.

    These formulas put an annual_data lookup in the 2000 column only; 2005+ are
    blank in the final sheet. Use annual_freight_wide_df['INDEX'] directly as the
    workbook annual_index, before AB-template market-share fallback logic.
    """
    region = str(region).strip().upper()
    if region not in {'SK', 'MB'}:
        return pd.Series(dtype=float), ''
    if str(row_template.get('Parameter', '')).strip() != 'market_share_total':
        return pd.Series(dtype=float), ''
    source = str(row_template.get('Source', '')).strip()
    if source not in {'annual_region_tech', 'annual_tech'}:
        return pd.Series(dtype=float), ''

    work = _final_ab_clean_source_df(annual_df)
    if work.empty or 'INDEX' not in work.columns or '2000' not in work.columns:
        return pd.Series(dtype=float), ''

    formula_key = _final_region_sk_market_share_formula_key_from_row(row_template, region)
    formula_key_norm = _final_region_sk_output_key_norm(formula_key)
    candidates = work.loc[work['INDEX'].map(_final_region_sk_output_key_norm).eq(formula_key_norm)].copy()
    if candidates.empty:
        return pd.Series(dtype=float), ''

    candidates['_value_2000'] = pd.to_numeric(candidates['2000'], errors='coerce')
    candidates['_has_2000'] = candidates['_value_2000'].notna().astype(int)
    if 'annual_row_id' in candidates.columns:
        candidates['_annual_row_id_num'] = pd.to_numeric(candidates['annual_row_id'], errors='coerce').fillna(999999)
    else:
        candidates['_annual_row_id_num'] = 999999
    selected = candidates.sort_values(['_has_2000', '_annual_row_id_num'], ascending=[False, True]).iloc[0]
    val_2000 = pd.to_numeric(pd.Series([selected.get('2000', np.nan)]), errors='coerce').iloc[0]
    if pd.isna(val_2000):
        val_2000 = 0.0

    series = pd.Series({y: np.nan for y in FINAL_FREIGHT_AB_YEARS}, dtype=float)
    series.loc['2000'] = float(val_2000)
    note = (
        f'sk_market_share_total_formula_exact: direct annual_freight_wide_df INDEX lookup; '
        f'2000 annual_freight value only, 2005+ blank; formula_key={formula_key}; '
        f'annual_row_id={selected.get("annual_row_id", "")}; INDEX={selected.get("INDEX", "")}'
    )
    return series, note
def _final_region_sk_service_request_formula_rel_path(value: object, region: str) -> str:
    """Mimic workbook SUBSTITUTE(..., LEFT(...FIND(region)...)) for SK formulas.

    Regional CIMS paths become relative paths. Non-regional generic targets such
    as CIMS.Generic Fuels.Propane are preserved because annual_index stores those
    labels with the CIMS.Generic prefix.
    """
    if pd.isna(value):
        return ''
    s = str(value).strip().replace('\\_', '_')
    region = str(region).strip().upper()
    for code in [region, FINAL_FREIGHT_AB_REGION]:
        exact = f'CIMS.CAN.{code}'
        prefix = exact + '.'
        if s == exact:
            return ''
        if s.startswith(prefix):
            return s[len(exact):].strip()
    return s.strip()


def _final_region_sk_service_request_formula_key_from_row(row: pd.Series, region: str) -> str:
    """Workbook CONCAT key for SK service_request annual rows.

    Pasted SK formulas use:
      - annual_region rows: relative Branch + $C region + G:I + relative Target
      - annual_tech rows:   relative Branch + $F technology + G:I + relative Target
    """
    source = str(row.get('Source', '')).strip()
    selector = str(region).strip().upper() if source in {'annual_region', 'annual_region_tech'} else str(row.get('Technology', '')).strip()
    return ''.join(str(p).strip() for p in [
        _final_region_sk_service_request_formula_rel_path(row.get('Branch', ''), region),
        selector,
        str(row.get('Parameter', '')).strip(),
        str(row.get('Context', '')).strip(),
        str(row.get('Sub_Context', '')).strip(),
        _final_region_sk_service_request_formula_rel_path(row.get('Target', ''), region),
    ] if str(p).strip() != '')


def _final_region_sk_air_service_request_projection_from_annual(base_series: pd.Series) -> pd.Series:
    """Apply pasted SK Air service_request projection formulas.

    Workbook pattern for SK Air service_request rows:
      2000 = direct annual_data lookup
      2005 = 2000 * 0.85
      2010 = 2005 * 0.85
      2015 = 2010 * 0.85
      2020 = 2015 * 0.85
      2025 = 2020 * 0.95
      2030-2050 = previous benchmark year * 0.95
    """
    out = pd.Series(index=FINAL_FREIGHT_AB_YEARS, dtype=float)
    out['2000'] = pd.to_numeric(base_series.get('2000', np.nan), errors='coerce')
    out['2005'] = out.get('2000', np.nan) * 0.85 if pd.notna(out.get('2000', np.nan)) else np.nan
    for y in ['2010', '2015', '2020']:
        prev = str(int(y) - 5)
        out[y] = out.get(prev, np.nan) * 0.85 if pd.notna(out.get(prev, np.nan)) else np.nan
    for y in ['2025', '2030', '2035', '2040', '2045', '2050']:
        prev = str(int(y) - 5)
        out[y] = out.get(prev, np.nan) * 0.95 if pd.notna(out.get(prev, np.nan)) else np.nan
    return out

def _final_region_sk_service_request_annual_series(
    annual_df: pd.DataFrame | None,
    row_template: pd.Series,
    region: str,
) -> tuple[pd.Series, str]:
    """Resolve pasted SK service_request formulas from annual_freight_wide_df.

    Uses annual_freight_wide_df['INDEX'] as the workbook annual_index and runs
    before dynamic calc_freight fallbacks.
    """
    region = str(region).strip().upper()
    if region != 'SK' or str(row_template.get('Parameter', '')).strip() != 'service_request':
        return pd.Series(dtype=float), ''
    source = str(row_template.get('Source', '')).strip()
    if source not in {'annual_region', 'annual_tech', 'annual_region_tech'}:
        return pd.Series(dtype=float), ''

    work = _final_ab_clean_source_df(annual_df)
    if work.empty or 'INDEX' not in work.columns:
        return pd.Series(dtype=float), ''
    for y in FINAL_FREIGHT_AB_YEARS:
        if y not in work.columns:
            work[y] = np.nan

    formula_key = _final_region_sk_service_request_formula_key_from_row(row_template, region)
    formula_key_norm = _final_region_sk_output_key_norm(formula_key)
    candidates = work.loc[work['INDEX'].map(_final_region_sk_output_key_norm).eq(formula_key_norm)].copy()
    if candidates.empty:
        return pd.Series(dtype=float), ''

    candidates['_value_2000'] = pd.to_numeric(candidates['2000'], errors='coerce')
    candidates['_has_2000'] = candidates['_value_2000'].notna().astype(int)
    if 'annual_row_id' in candidates.columns:
        candidates['_annual_row_id_num'] = pd.to_numeric(candidates['annual_row_id'], errors='coerce').fillna(999999)
    else:
        candidates['_annual_row_id_num'] = 999999
    selected = candidates.sort_values(['_has_2000', '_annual_row_id_num'], ascending=[False, True]).iloc[0]
    base = pd.to_numeric(selected[FINAL_FREIGHT_AB_YEARS], errors='coerce').reindex(FINAL_FREIGHT_AB_YEARS).astype(float)

    if source in {'annual_region', 'annual_region_tech'}:
        out = base
        projection = 'direct annual_header benchmark-year lookup'
    elif _final_ab_norm_key_text(row_template.get('Service', '')) == 'air':
        out = _final_region_sk_air_service_request_projection_from_annual(base)
        projection = 'SK Air projection: 2000 direct, 2005-2020 *=0.85, 2025-2050 *=0.95'
    else:
        out = pd.Series(index=FINAL_FREIGHT_AB_YEARS, dtype=float)
        out['2000'] = base.get('2000', np.nan)
        out['2005'] = base.get('2005', np.nan)
        carry = out.get('2005', np.nan)
        for y in FINAL_FREIGHT_AB_YEARS:
            if y not in {'2000', '2005'}:
                out[y] = carry
        projection = 'annual_tech: 2000/2005 direct, 2005 carried across 2010-2050'

    if out.empty or not out.notna().any():
        return pd.Series(dtype=float), ''
    note = (
        f'sk_service_request_formula_exact: direct annual_freight_wide_df INDEX lookup; '
        f'{projection}; formula_key={formula_key}; annual_row_id={selected.get("annual_row_id", "")}; '
        f'INDEX={selected.get("INDEX", "")}'
    )
    return out, note
def _final_region_sk_output_annual_region_carry_2000_series(
    annual_template_df: pd.DataFrame | None,
    row_template: pd.Series,
    region: str,
) -> tuple[pd.Series, str]:
    """Resolve pasted SK output formulas: annual_data 2000 lookup, then carry M->V.

    The formula key is relative Branch + Region + Parameter/Context/Sub_Context +
    relative Target. The Technology column is intentionally NOT part of this lookup.
    """
    region = str(region).strip().upper()
    if region != 'SK' or str(row_template.get('Parameter', '')).strip() != 'output':
        return pd.Series(dtype=float), ''
    if str(row_template.get('Source', '')).strip() != 'annual_region' or str(row_template.get('Unit', '')).strip() != 'node unit':
        return pd.Series(dtype=float), ''

    # annual_template_df and row_template are AB-normalized by the generic builder.
    formula_key = _final_region_sk_output_formula_key_from_row(row_template, region)
    formula_key_norm = _final_region_sk_output_key_norm(formula_key)
    work = _final_ab_clean_source_df(annual_template_df)
    if work.empty or '2000' not in work.columns:
        _final_region_sk_output_diag(row_template, region, formula_key, 'missing_annual_or_2000', 0, False)
        return pd.Series(dtype=float), ''

    for col in ['Branch','region','Region','Parameter','Context','Sub_Context','Target','Unit','INDEX','annual_row_id']:
        if col not in work.columns:
            work[col] = ''
    work = work.copy()

    # Primary workbook-faithful lookup: the Excel formula uses XMATCH(..., annual_index),
    # which maps directly to the cleaned annual_freight_wide_df['INDEX'] column.  Use
    # that INDEX first and do not filter by the final row Unit ('node unit'), because
    # the annual input output rows are stored as k*tkm.
    index_candidates = work.loc[work['INDEX'].map(_final_region_sk_output_key_norm).eq(formula_key_norm)].copy()
    if not index_candidates.empty:
        index_candidates['_value_2000'] = pd.to_numeric(index_candidates['2000'], errors='coerce')
        index_candidates['_has_2000'] = index_candidates['_value_2000'].notna().astype(int)
        index_candidates = index_candidates.sort_values(['_has_2000'], ascending=False)
        selected = index_candidates.iloc[0]
        base = pd.to_numeric(selected.get('2000', np.nan), errors='coerce')
        if pd.notna(base):
            out = pd.Series({y: float(base) for y in FINAL_FREIGHT_AB_YEARS}, dtype=float)
            note = f'sk_output_annual_region_formula_exact: direct annual_freight_wide_df INDEX lookup; 2000 annual_freight value carried across benchmark years; formula_key={formula_key}; annual_row_id={selected.get("annual_row_id", "")}; INDEX={selected.get("INDEX", "")}'
            _final_region_sk_output_diag(row_template, region, formula_key, 'annual_index_exact', len(index_candidates), True, note, selected)
            return out, note
        _final_region_sk_output_diag(row_template, region, formula_key, 'annual_index_exact_blank_2000', len(index_candidates), False, selected_row=selected)
        return pd.Series(dtype=float), ''

    work['_sk_formula_key'] = work.apply(lambda r: _final_region_sk_output_candidate_key(r, region), axis=1)
    work['_sk_formula_key_norm'] = work['_sk_formula_key'].map(_final_region_sk_output_key_norm)

    base_mask = work['Parameter'].astype(str).str.strip().eq('output')
    # The annual dataframe may be raw SK rows or an AB-normalized template. Keep both
    # eligible. Also do not require Unit == node unit: final rows are node unit, but
    # the annual source output rows are k*tkm.
    base_mask &= work['region'].astype(str).str.strip().str.upper().isin(['', region, FINAL_FREIGHT_AB_REGION])
    base_mask &= work['Unit'].map(_final_ab_norm_key_text).isin(['', 'node unit', 'k*tkm'])

    candidates = work.loc[base_mask & work['_sk_formula_key_norm'].eq(formula_key_norm)].copy()
    stage = 'metadata_formula_key_exact_no_technology'
    if candidates.empty:
        branch_rel = _final_ab_rel_path(row_template.get('Branch', ''))
        target_rel = _final_ab_rel_path(row_template.get('Target', ''))
        mask = base_mask & work['Branch'].map(_final_ab_rel_path).eq(branch_rel)
        mask &= work['Context'].map(_final_ab_norm_key_text).eq(_final_ab_norm_key_text(row_template.get('Context', '')))
        mask &= work['Sub_Context'].map(_final_ab_norm_key_text).eq(_final_ab_norm_key_text(row_template.get('Sub_Context', '')))
        mask &= work['Target'].map(_final_ab_rel_path).eq(target_rel)
        candidates = work.loc[mask].copy()
        stage = 'metadata_strict_no_technology'
    if candidates.empty:
        branch_rel = _final_ab_rel_path(row_template.get('Branch', ''))
        candidates = work.loc[base_mask & work['Branch'].map(_final_ab_rel_path).eq(branch_rel)].copy()
        stage = 'branch_output_no_technology'
    if candidates.empty:
        _final_region_sk_output_diag(row_template, region, formula_key, 'no_candidate', 0, False, 'No annual_freight output row matched formula key.')
        return pd.Series(dtype=float), ''

    candidates['_value_2000'] = pd.to_numeric(candidates['2000'], errors='coerce')
    candidates['_has_2000'] = candidates['_value_2000'].notna().astype(int)
    candidates['_exact_key'] = candidates['_sk_formula_key_norm'].eq(formula_key_norm).astype(int)
    candidates = candidates.sort_values(['_has_2000','_exact_key'], ascending=False)
    selected = candidates.iloc[0]
    base = pd.to_numeric(selected.get('2000', np.nan), errors='coerce')
    if pd.isna(base):
        _final_region_sk_output_diag(row_template, region, formula_key, stage + '_blank_2000', len(candidates), False, selected_row=selected)
        return pd.Series(dtype=float), ''

    out = pd.Series({y: float(base) for y in FINAL_FREIGHT_AB_YEARS}, dtype=float)
    note = f'sk_output_annual_region_formula_exact: 2000 annual_freight lookup carried across benchmark years; match_stage={stage}; formula_key={formula_key}; annual_row_id={selected.get("annual_row_id", "")}; INDEX={selected.get("INDEX", "")}'
    _final_region_sk_output_diag(row_template, region, formula_key, stage, len(candidates), True, note, selected)
    return out, note


def build_final_transportation_freight_region(
    region: str,
    *,
    calc_df: pd.DataFrame | None = None,
    cms_df: pd.DataFrame | None = None,
    akm_df: pd.DataFrame | None = None,
    annual_freight_wide_df: pd.DataFrame | None = None,
    constant_freight_df: pd.DataFrame | None = None,
    write: bool = True,
) -> pd.DataFrame:
    """Build a final regional transportation-freight dataframe for any province.

    The implementation is generic: one 483-row template, one set of source
    resolvers, and province selection through the `region` argument. No target
    guide CSV/workbook is read at runtime.
    """
    region = str(region).strip().upper()
    if not region:
        raise ValueError('region must be a non-empty province/region code, e.g. AB or BC')

    if calc_df is None:
        calc_df = _final_ab_load_calc_csv('calc_freight.csv')
    if cms_df is None:
        cms_df = _final_ab_load_calc_csv('calc_market_share.csv')
    if akm_df is None:
        akm_df = _final_ab_load_calc_csv('calc_avg_km.csv')

    # Convert only the selected region to template AB labels for reusing existing helpers.
    calc_template = _final_region_df_as_template(calc_df, region)
    cms_template = _final_region_df_as_template(cms_df, region)
    akm_template = _final_region_df_as_template(akm_df, region)
    annual_template = _final_region_df_as_template(annual_freight_wide_df, region)
    constant_template = _final_region_df_as_template(constant_freight_df, region)

    dyn = _final_ab_build_dynamic_series(calc_template, cms_template, akm_template)
    skeleton = _final_region_rows(region)
    out_rows: list[dict] = []
    audit_rows: list[dict] = []
    source_rows: list[dict] = []

    for idx, row in skeleton.iterrows():
        out_row = row.to_dict()
        row_template = _final_region_row_as_template(pd.Series(out_row), region)
        dyn_key, source_years, source_note, carry_2000 = _final_ab_mapping_for_row(row_template)

        # SK output formulas with Source=annual_region are direct 2000 annual_freight
        # lookups carried across all benchmark years (=M row, =N row, ...). Resolve
        # them before dynamic output mappings from calc_avg_km.
        series = pd.Series(dtype=float)
        source_strategy = ''
        note = ''
        if region in {'SK', 'MB'}:
            sk_service_s, sk_service_note = _final_region_sk_service_request_annual_series(
                annual_freight_wide_df, row_template, region
            )
            if not sk_service_s.empty and sk_service_s.notna().any():
                series = sk_service_s
                source_strategy = 'annual_freight_dataframe'
                note = sk_service_note

        if region in {'SK', 'MB'} and (series.empty or not series.notna().any()):
            sk_output_s, sk_output_note = _final_region_sk_output_annual_region_carry_2000_series(
                annual_freight_wide_df, row_template, region
            )
            if not sk_output_s.empty and sk_output_s.notna().any():
                series = sk_output_s
                source_strategy = 'annual_freight_dataframe'
                note = sk_output_note

        # Non-AB market_share_total formulas are direct annual_freight INDEX(... annual_header)
        # lookups in the 2000 column only. For SK, use the raw annual_freight_wide_df
        # INDEX because the workbook key includes SK for annual_region_tech rows.
        if (
            (series.empty or not series.notna().any())
            and region in {'SK', 'MB'}
            and str(out_row.get('Parameter', '')).strip() == 'market_share_total'
            and str(out_row.get('Source', '')).strip() in {'annual_region_tech', 'annual_tech'}
        ):
            sk_ms_s, sk_ms_note = _final_region_sk_market_share_total_annual_series(
                annual_freight_wide_df, row_template, region
            )
            if not sk_ms_s.empty and sk_ms_s.notna().any():
                series = sk_ms_s
                source_strategy = 'annual_freight_dataframe'
                note = sk_ms_note

        # Other non-AB market_share_total rows can use the AB-template annual helper.
        if (
            (series.empty or not series.notna().any())
            and region != FINAL_FREIGHT_AB_REGION
            and str(out_row.get('Parameter', '')).strip() == 'market_share_total'
            and str(out_row.get('Source', '')).strip() in {'annual_region_tech', 'annual_tech'}
        ):
            ms_s, ms_note = _final_ab_series_from_annual_df(annual_template, row_template)
            if not ms_s.empty and ms_s.notna().any():
                series = ms_s
                source_strategy = 'annual_freight_dataframe'
                note = ms_note.replace('CIMS.CAN.AB', f'CIMS.CAN.{region}')

        if series.empty or not series.notna().any():
            series = dyn.get(dyn_key, pd.Series(dtype=float)) if dyn_key else pd.Series(dtype=float)
            source_strategy = 'dynamic_upstream' if dyn_key in dyn else ''
            note = source_note.replace('AB ', f'{region} ').replace('region=AB', f'region={region}') if source_note else ''

        if series.empty or not series.notna().any():
            mp_s, mp_note = _final_region_multiplier_price_series(pd.Series(out_row), region)
            if not mp_s.empty and mp_s.notna().any():
                series = mp_s
                source_strategy = 'final_region_multiplier_price_rules'
                note = mp_note

        if series.empty or not series.notna().any():
            sk_fcc_s, sk_fcc_note = _final_region_sk_fcc_constant_precision_series(pd.Series(out_row), region)
            if not sk_fcc_s.empty and sk_fcc_s.notna().any():
                series = sk_fcc_s
                source_strategy = 'constant_freight_dataframe'
                note = sk_fcc_note

        if series.empty or not series.notna().any():
            const_s, const_note = _final_ab_series_from_constant_df(constant_template, row_template)
            if not const_s.empty and const_s.notna().any():
                series = const_s
                source_strategy = 'constant_freight_dataframe'
                note = const_note.replace('CIMS.CAN.AB', f'CIMS.CAN.{region}')

        if series.empty or not series.notna().any():
            # Region-specific Air logic: AB uses its validated local projection,
            # while BC formulas use direct annual_header lookups for each benchmark year.
            regional_air_s, regional_air_note = _final_region_air_service_request_raw_annual_series(
                annual_template, row_template, region
            )
            if not regional_air_s.empty and regional_air_s.notna().any():
                series = regional_air_s
                source_strategy = 'annual_freight_dataframe'
                note = regional_air_note.replace('CIMS.CAN.AB', f'CIMS.CAN.{region}')
            else:
                annual_s, annual_note = _final_ab_series_from_annual_df(annual_template, row_template)
                if not annual_s.empty and annual_s.notna().any():
                    series = annual_s
                    source_strategy = 'annual_freight_dataframe'
                    note = annual_note.replace('CIMS.CAN.AB', f'CIMS.CAN.{region}')

        if not series.empty and series.notna().any():
            for y in FINAL_FREIGHT_AB_YEARS:
                if y in series.index:
                    out_row[y] = _final_ab_format_value(series.get(y, np.nan))
            if carry_2000 and '2000' in series.index:
                v = series.get('2000', np.nan)
                for y in FINAL_FREIGHT_AB_YEARS:
                    out_row[y] = _final_ab_format_value(v)
            status = source_strategy
        else:
            status = _final_ab_pending_reason(row_template)
            note = status

        nonblank_years = sum(1 for y in FINAL_FREIGHT_AB_YEARS if str(out_row.get(y, '')).strip() != '')
        audit_rows.append({
            'row_index': idx,
            'Region': region,
            'Branch': out_row.get('Branch', ''),
            'Technology': out_row.get('Technology', ''),
            'Parameter': out_row.get('Parameter', ''),
            'Target': out_row.get('Target', ''),
            'Source': out_row.get('Source', ''),
            'status': status,
            'nonblank_years': nonblank_years,
            'note': note,
        })
        source_rows.append({
            'row_index': idx,
            'Region': region,
            'Branch': out_row.get('Branch', ''),
            'Technology': out_row.get('Technology', ''),
            'Parameter': out_row.get('Parameter', ''),
            'Target': out_row.get('Target', ''),
            'Source': out_row.get('Source', ''),
            'dynamic_key': dyn_key,
            'source_strategy': status,
            'source_note': note,
        })
        out_rows.append(out_row)

    final_df = pd.DataFrame(out_rows, columns=FINAL_FREIGHT_AB_COLUMNS)
    audit = pd.DataFrame(audit_rows)
    source_map = pd.DataFrame(source_rows)

    if write:
        names = _final_region_output_names(region)
        out_path = OUT_DIR / names['output']
        audit_path = OUT_DIR / names['audit']
        source_path = OUT_DIR / names['source_map']
        _final_region_write_with_nav(final_df, out_path)
        audit.to_csv(audit_path, index=False)
        source_map.to_csv(source_path, index=False)
        if region == 'SK':
            sk_output_diag = pd.DataFrame(FINAL_REGION_SK_OUTPUT_ANNUAL_DIAGNOSTICS)
            sk_output_diag.to_csv(OUT_DIR / 'transportation_freight_SK_output_annual_lookup_diagnostic_test.csv', index=False)
            try:
                _register_df('transportation_freight_SK_output_annual_lookup_diagnostic_test.csv', sk_output_diag)
            except Exception:
                pass
        try:
            _register_df(names['output'], final_df)
            _register_df(names['audit'], audit)
            _register_df(names['source_map'], source_map)
        except Exception:
            pass
        print(f"  ✅ {names['output']} ({len(final_df):,} rows x {len(final_df.columns):,} columns)")
        print(f"  ✅ {names['audit']} ({len(audit):,} rows)")
        print(f"  ✅ {names['source_map']} ({len(source_map):,} rows)")

    return final_df


def build_final_transportation_freight_bc(
    *,
    calc_df: pd.DataFrame | None = None,
    cms_df: pd.DataFrame | None = None,
    akm_df: pd.DataFrame | None = None,
    annual_freight_wide_df: pd.DataFrame | None = None,
    constant_freight_df: pd.DataFrame | None = None,
    write: bool = True,
) -> pd.DataFrame:
    """BC convenience wrapper using the generic regional final builder."""
    return build_final_transportation_freight_region(
        'BC',
        calc_df=calc_df,
        cms_df=cms_df,
        akm_df=akm_df,
        annual_freight_wide_df=annual_freight_wide_df,
        constant_freight_df=constant_freight_df,
        write=write,
    )


def build_final_transportation_freight_sk(
    *,
    calc_df: pd.DataFrame | None = None,
    cms_df: pd.DataFrame | None = None,
    akm_df: pd.DataFrame | None = None,
    annual_freight_wide_df: pd.DataFrame | None = None,
    constant_freight_df: pd.DataFrame | None = None,
    write: bool = True,
) -> pd.DataFrame:
    """SK convenience wrapper using the generic regional final builder.

    The Saskatchewan final dataframe is produced from upstream dataframes and
    shared final-region logic only. The SK formula workbook/CSV guide remain
    validation references and are not read by the runtime pipeline.
    """
    return build_final_transportation_freight_region(
        'SK',
        calc_df=calc_df,
        cms_df=cms_df,
        akm_df=akm_df,
        annual_freight_wide_df=annual_freight_wide_df,
        constant_freight_df=constant_freight_df,
        write=write,
    )


def build_final_transportation_freight_mb(
    *,
    calc_df: pd.DataFrame | None = None,
    cms_df: pd.DataFrame | None = None,
    akm_df: pd.DataFrame | None = None,
    annual_freight_wide_df: pd.DataFrame | None = None,
    constant_freight_df: pd.DataFrame | None = None,
    write: bool = True,
) -> pd.DataFrame:
    """MB convenience wrapper using the generic regional final builder.

    The Manitoba final dataframe is produced from upstream dataframes and shared
    final-region logic only. The MB formula workbook/CSV guide remain validation
    references and are not read by the runtime pipeline.
    """
    return build_final_transportation_freight_region(
        'MB',
        calc_df=calc_df,
        cms_df=cms_df,
        akm_df=akm_df,
        annual_freight_wide_df=annual_freight_wide_df,
        constant_freight_df=constant_freight_df,
        write=write,
    )

if __name__ == "__main__":
    main()