"""
nir_to_cims.py
──────────────

  Stage 1 – Solve suppressed ('x') GHG values in the NIR.
            ECCC suppresses cells where disclosure would identify individual
            respondents, primarily NWT, Nunavut, and Nova Scotia for certain
            sectors and years. The solver uses Canada-level totals and known
            regional values to algebraically derive each suppressed cell,
            working through a defined sequence to avoid circular dependencies.
            An anchor year (2004) and a co-suppression year (2005, where NS,
            NWT, and Nu are all suppressed simultaneously) require special
            solve orders; 2006+ use a simplified path.
            Only the NIR CSV and mapping CSV are needed as inputs.

  Stage 2 – Map solved NIR rows to CIMS branch nodes using NIR_to_CIMS_map.csv.

  Stage 3 – Roll up ancestor totals using add_cims_totals.

Suppression solver is configured for the 2026 NIR pattern. Update the
config block below if a future NIR has different suppressed values.
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import polars as pl
sys.path.append(r"C:\cims\data\pipeline\utils")
from add_cims_totals import add_totals

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG — Input and output files
# ═══════════════════════════════════════════════════════════════════════════════

# Update folder with new nir when released each year
NIR_CSV  = r"C:\cims\data\raw_data\eccc\nir\GHG_Econ_Can_Prov_Terr.csv"
# Manual map - should stay the same unless there are changes to nir or cims naming
MAP_CSV  = r"C:\cims\data\mappings_conversions\NIR_to_CIMS_map.csv"
# Script outputs a version of the nir with the suppressed values solved
OUT_NIR  = r"C:\cims\data\processed_data\eccc\NIR_Suppressed_Solved.csv"
# Main output - NIR emissions mapped to cims nodes
OUT_CIMS = r"C:\cims\data\processed_data\eccc\NIR_to_CIMS.csv"

# Anchor year: last year where NWT and Nu Construction (idx=47) are unsuppressed.
# Used to compute the NWT/Nu split ratio. Move earlier if this year becomes suppressed.
ANCHOR_YEAR = 2004

# The one year where NS, NWT, and Nu are all suppressed simultaneously.
# Requires a special solve order to avoid circular dependencies.
CO_SUPPRESSION_YEAR = 2005

# Years where only NWT and Nu are suppressed (simpler two-region solve).
# Update the upper bound when a new NIR year is added.
SIMPLE_SUPPRESSION_YEARS = range(2006, 2025)

# Suppressed (region, index) pairs the solver handles, and the year they begin.
# These are the specific rows the solver knows how to estimate — there may be
# other suppressed values in the NIR not listed here (e.g. years prior to 2000)
# The pre-flight check will flag any relevant 'x' values outside this list.
KNOWN_SUPPRESSED = {
    ("Northwest Territories", 16): 2005,
    ("Northwest Territories", 45): 2005,
    ("Northwest Territories", 47): 2005,
    ("Nunavut",               16): 2005,
    ("Nunavut",               33): 2005,
    ("Nunavut",               34): 2005,
    ("Nunavut",               45): 2005,
    ("Nunavut",               47): 2005,
    ("Nova Scotia",           33): 2005,
    ("Nova Scotia",           34): 2005,
    ("Nova Scotia",           45): 2005,
    ("Nova Scotia",           47): 2005,
}

# Regions excluded from CIMS output (pre-split combined region, no year >= 2000 data)
SKIP_REGIONS = {"Northwest Territories and Nunavut"}

# Set False to exclude Canada-level sector rows from the CIMS output.
# Canada rows are national aggregates — summing them alongside provincial rows
# double-counts. The CIMS.CAN inventory total is always included regardless.
INCLUDE_CANADA_SECTORS = True

# All gas columns in the NIR (including CO2eq used for solving but not output)
GAS_COLS = [
    "CO2 (kt)", "CH4 (kt)", "N2O (kt)",
    "HFCs (kt CO2e)", "PFCs (kt CO2e)", "SF6 (kt)", "NF3 (kt)",
    "Total (kt CO2e)",
]

# Gas column names mapped to their output labels in the long format
GAS_LABELS = {
    "CO2 (kt)":       "CO2",
    "CH4 (kt)":       "CH4",
    "N2O (kt)":       "N2O",
    "HFCs (kt CO2e)": "HFCs",
    "PFCs (kt CO2e)": "PFCs",
    "SF6 (kt)":       "SF6",
    "NF3 (kt)":       "NF3",
}

# Resolve all paths and confirm output folders exist
NIR_CSV  = Path(NIR_CSV)
MAP_CSV  = Path(MAP_CSV)
OUT_NIR  = Path(OUT_NIR)
OUT_CIMS = Path(OUT_CIMS)
OUT_NIR.parent.mkdir(parents=True, exist_ok=True)
OUT_CIMS.parent.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════

# Read NIR as strings so 'x' suppression markers are preserved
raw = pl.read_csv(str(NIR_CSV), infer_schema_length=0)

# Mutable lookup: (year, region, index) -> row dict
data = {}
row_order = []
for row in raw.to_dicts():
    k = (int(row["Year"]), row["Region"], int(row["Index"]))
    data[k] = row
    row_order.append(k)

# Load mapping CSV
with open(str(MAP_CSV), encoding="utf-8-sig") as f:
    mapping_rows = list(csv.DictReader(f))

mapped_nir_keys = set(
    (m["Source"].strip(), m["Sector"].strip(),
     m["Sub-sector"].strip(), m["Sub-sub-sector"].strip())
    for m in mapping_rows
)

# Derive known (non-suppressed) provinces from the data
all_regions = raw["Region"].unique().to_list()
known_provinces = [
    r for r in all_regions
    if r not in ("Canada", "Northwest Territories", "Nunavut",
                 "Nova Scotia", "Northwest Territories and Nunavut")
]

# 2005 Buildings residual includes NWT since NWT_33 is not suppressed
known_provinces_with_nwt = known_provinces + ["Northwest Territories"]

# ═══════════════════════════════════════════════════════════════════════════════
# PRE-FLIGHT CHECKS
# WARNING = outputs may be wrong or incomplete, review before use
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("PRE-FLIGHT CHECKS")
print("=" * 60)

check_warnings = []

# ── Check 1: New suppressed values not handled by the solver ──────────────────
new_suppressed = []
for (yr, reg, idx), row in data.items():
    if not any(row.get(g) == "x" for g in GAS_COLS):
        continue
    known_start = KNOWN_SUPPRESSED.get((reg, idx))
    if known_start is None or yr > max(SIMPLE_SUPPRESSION_YEARS):
        new_suppressed.append((yr, reg, idx, row.get("Source", "")))

if new_suppressed:
    check_warnings.append(
        "NEW SUPPRESSED VALUES — solver does not handle these:\n" +
        "\n".join(f"    {yr} | {reg:28s} | idx={idx} | {src}"
                  for yr, reg, idx, src in sorted(new_suppressed))
    )

# ── Check 2: Anchor year integrity ────────────────────────────────────────────
# If anchor year Construction data is suppressed, all solved values will be wrong.
anchor_broken = []
for region in ("Northwest Territories", "Nunavut"):
    row = data.get((ANCHOR_YEAR, region, 47), {})
    if any(row.get(g) == "x" for g in GAS_COLS):
        anchor_broken.append(region)
    elif not any(row.get(g) and row.get(g) != "x" for g in GAS_COLS):
        anchor_broken.append(f"{region} (row missing entirely)")

if anchor_broken:
    check_warnings.append(
        f"ANCHOR YEAR {ANCHOR_YEAR} CONSTRUCTION DATA IS SUPPRESSED OR MISSING —\n"
        f"    all solved values will be wrong:\n" +
        "\n".join(f"    {r}" for r in anchor_broken) +
        f"\n    Move anchor to an earlier unsuppressed year and update ANCHOR_YEAR."
    )

# ── Check 3: NIR rows with non-zero emissions not covered by the mapping ───────
all_nir_keys_in_data = set(
    ((row.get("Source") or "").strip(), (row.get("Sector") or "").strip(),
     (row.get("Sub-sector") or "").strip(), (row.get("Sub-sub-sector") or "").strip())
    for row in data.values()
)

unmapped_nonzero = {}
for (yr, reg, idx), row in data.items():
    if yr < 2000 or reg in SKIP_REGIONS or idx == 0:
        continue
    src  = (row.get("Source")        or "").strip()
    sec  = (row.get("Sector")        or "").strip()
    sub  = (row.get("Sub-sector")    or "").strip()
    ssub = (row.get("Sub-sub-sector") or "").strip()
    nir_key = (src, sec, sub, ssub)

    if nir_key in mapped_nir_keys:
        continue

    # Skip if covered by a mapped parent (less-specific key in mapping)
    if any(k in mapped_nir_keys
           for k in [(src, sec, sub, ""), (src, sec, "", ""), (src, "", "", "")]
           if k != nir_key):
        continue

    # Skip if this row IS a parent of mapped rows (more specific keys exist)
    if any(k[0] == src and len([x for x in k if x]) > len([x for x in nir_key if x])
           for k in mapped_nir_keys if k != nir_key and k[0] == src):
        continue

    val = float(row["CO2 (kt)"]) if row.get("CO2 (kt)") not in ("x", None, "") else 0.0
    if val == 0.0:
        continue

    if nir_key not in unmapped_nonzero:
        unmapped_nonzero[nir_key] = [0.0, 0]
    unmapped_nonzero[nir_key][0] += val
    unmapped_nonzero[nir_key][1] += 1

if unmapped_nonzero:
    top = sorted(unmapped_nonzero.items(), key=lambda x: -x[1][0])
    lines = [
        f"    D={k[0][:28]:28s} E={k[1][:22]:22s} F={k[2][:22]:22s} G={k[3][:18]:18s}"
        f"  total_CO2={v[0]:10.1f} kt  ({v[1]} row-years)"
        for k, v in top[:15]
    ]
    if len(top) > 15:
        lines.append(f"    ... and {len(top) - 15} more")
    check_warnings.append(
        f"NIR ROWS WITH NON-ZERO EMISSIONS NOT IN MAPPING ({len(unmapped_nonzero)} unique):\n" +
        "\n".join(lines)
    )

# ── Check 4: Mapping entries with no matching NIR rows ────────────────────────
missing_from_nir = [k for k in mapped_nir_keys if k not in all_nir_keys_in_data]
if missing_from_nir:
    check_warnings.append(
        f"MAPPING ENTRIES WITH NO MATCHING NIR ROWS — possible renamed sector ({len(missing_from_nir)}):\n" +
        "\n".join(
            f"    D={k[0][:28]:28s} E={k[1][:22]:22s} F={k[2][:22]:22s} G={k[3]}"
            for k in sorted(missing_from_nir)
        )
    )

# ── Print pre-flight results ───────────────────────────────────────────────────
if not check_warnings:
    print("  ✓ All checks passed — no issues detected.\n")
else:
    print(f"  {'!'*58}")
    print(f"  {'*** ' + str(len(check_warnings)) + ' WARNING(S) — REVIEW BEFORE USING OUTPUTS ***':^58}")
    print(f"  {'!'*58}")
    for i, msg in enumerate(check_warnings, 1):
        print(f"\n  [{i}] ⚠  {msg}")
    print()
    print("  ⚠  Continuing — review warnings above before using outputs.")
    print()

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — Solve suppressed values
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("STAGE 1: Solving suppressed values")
print("=" * 60)

# ── Helper functions ──────────────────────────────────────────────────────────

def to_float(v):
    """Convert NIR cell value to float; returns 0.0 for suppressed or missing."""
    return 0.0 if (v is None or v == "x") else float(v)

def get(year, region, index, gas):
    """Return numeric gas value for a NIR row; 0.0 for missing or suppressed."""
    row = data.get((year, region, index))
    return 0.0 if row is None else to_float(row.get(gas))

def set_val(year, region, index, gas, value):
    """Write a solved value back. Clamped to 0 — negative values are arithmetic
    artifacts from the residual algebra on near-zero trace gases."""
    k = (year, region, index)
    if k not in data:
        raise KeyError(f"Row not found: {k}")
    data[k][gas] = str(max(0.0, value))

def sum_regions(year, index, gas, regions):
    """Sum a gas value across a list of regions for a given year and index."""
    return sum(get(year, r, index, gas) for r in regions)

def sum_toplevel(year, region, gas, exclude_idxs):
    """Sum all top-level sector rows (empty Sector column) for a region/year,
    skipping specified index numbers. Treats suppressed 'x' as 0."""
    total = 0.0
    for (yr, reg, idx), row in data.items():
        if yr != year or reg != region or idx in exclude_idxs:
            continue
        if (row.get("Sector") or "").strip():
            continue    # child row, not top-level
        v = row.get(gas)
        if v and v != "x":
            total += float(v)
    return total

def is_inventory_total(branch):
    """Returns True for CIMS.CAN and CIMS.CAN.[region] — inventory total rows."""
    return len(branch.split(".")) <= 3

# ── Anchor ratios: NWT and Nu share of combined Construction (idx=47) ──────────
print(f"Computing {ANCHOR_YEAR} Construction ratios (NWT vs Nu)...")
ratios_NWT = {}
ratios_Nu  = {}
for gas in GAS_COLS:
    nwt   = get(ANCHOR_YEAR, "Northwest Territories", 47, gas)
    nu    = get(ANCHOR_YEAR, "Nunavut",               47, gas)
    total = nwt + nu
    ratios_NWT[gas] = (nwt / total) if total > 0 else 0.5
    ratios_Nu[gas]  = (nu  / total) if total > 0 else 0.5

# ── Combined anchor-year NWT+Nu LMC (idx=45) ─────────────────────────────────
# Anchoring NS LMC to the prior year's combined NWT+Nu value breaks the
# circularity: NS_45 is solved without needing the suppressed NWT/Nu values.
combined_lmc_anchor = {
    gas: get(ANCHOR_YEAR, "Northwest Territories", 45, gas)
       + get(ANCHOR_YEAR, "Nunavut",               45, gas)
    for gas in GAS_COLS
}

# ── Solve co-suppression year ─────────────────────────────────────────────────
# Suppressed: NWT idx=16,45,47 | Nu idx=16,33,34,45,47 | NS idx=33,34,45,47
# Steps are ordered so each depends only on already-known values.
print(f"Solving {CO_SUPPRESSION_YEAR} (co-suppression year)...")
for gas in GAS_COLS:

    # 1. NS LMC (idx=45): residual after known provinces, minus anchor-year NWT+Nu
    k45   = get(CO_SUPPRESSION_YEAR, "Canada", 45, gas) - sum_regions(CO_SUPPRESSION_YEAR, 45, gas, known_provinces)
    ns_45 = k45 - combined_lmc_anchor[gas]

    # 2. NS Construction (idx=47): bottom-up from LMC minus known children
    ns_47 = (ns_45
             - get(CO_SUPPRESSION_YEAR, "Nova Scotia", 46, gas)
             - get(CO_SUPPRESSION_YEAR, "Nova Scotia", 48, gas))

    # 3. NWT Construction (idx=47): NWT ratio × remaining pool after NS
    k47    = get(CO_SUPPRESSION_YEAR, "Canada", 47, gas) - sum_regions(CO_SUPPRESSION_YEAR, 47, gas, known_provinces)
    nwt_47 = ratios_NWT[gas] * (k47 - ns_47)

    # 4. NS Buildings (idx=33): top-down from NS total
    ns_33 = (get(CO_SUPPRESSION_YEAR, "Nova Scotia", 0, gas)
             - sum_toplevel(CO_SUPPRESSION_YEAR, "Nova Scotia", gas, exclude_idxs={0, 33, 45})
             - ns_45)

    # 5. Nu Buildings (idx=33): residual after known provinces + NWT + NS
    k33   = get(CO_SUPPRESSION_YEAR, "Canada", 33, gas) - sum_regions(CO_SUPPRESSION_YEAR, 33, gas, known_provinces_with_nwt)
    nu_33 = k33 - ns_33

    # 6. Nu Construction (idx=47): Nu ratio × same pool as NWT
    #    (The Excel template's "Nu Buildings" filter term always evaluates to 0
    #    due to a Sector column mismatch, so it uses the same k47 pool as NWT.)
    nu_47 = ratios_Nu[gas] * (k47 - ns_47)

    # 7 & 8. LMC parent (idx=45): bottom-up from children for NWT and Nu
    nwt_45 = (get(CO_SUPPRESSION_YEAR, "Northwest Territories", 46, gas)
              + nwt_47
              + get(CO_SUPPRESSION_YEAR, "Northwest Territories", 48, gas))
    nu_45  = (get(CO_SUPPRESSION_YEAR, "Nunavut", 46, gas)
              + nu_47
              + get(CO_SUPPRESSION_YEAR, "Nunavut", 48, gas))

    # 9. NWT Electricity (idx=16): top-down from NWT total
    nwt_16 = (get(CO_SUPPRESSION_YEAR, "Northwest Territories", 0, gas)
              - sum_toplevel(CO_SUPPRESSION_YEAR, "Northwest Territories", gas, exclude_idxs={0, 16, 45})
              - nwt_45)

    # 10. Nu Electricity (idx=16): top-down; also subtract Nu Buildings (suppressed)
    nu_16 = (get(CO_SUPPRESSION_YEAR, "Nunavut", 0, gas)
             - sum_toplevel(CO_SUPPRESSION_YEAR, "Nunavut", gas, exclude_idxs={0, 16, 33, 45})
             - nu_45
             - nu_33)

    # 11 & 12. Buildings > Service Industry (idx=34): Buildings minus Residential (idx=35)
    ns_34 = ns_33 - get(CO_SUPPRESSION_YEAR, "Nova Scotia", 35, gas)
    nu_34 = nu_33 - get(CO_SUPPRESSION_YEAR, "Nunavut",    35, gas)

    set_val(CO_SUPPRESSION_YEAR, "Northwest Territories", 16, gas, nwt_16)
    set_val(CO_SUPPRESSION_YEAR, "Northwest Territories", 45, gas, nwt_45)
    set_val(CO_SUPPRESSION_YEAR, "Northwest Territories", 47, gas, nwt_47)
    set_val(CO_SUPPRESSION_YEAR, "Nova Scotia",           33, gas, ns_33)
    set_val(CO_SUPPRESSION_YEAR, "Nova Scotia",           34, gas, ns_34)
    set_val(CO_SUPPRESSION_YEAR, "Nova Scotia",           45, gas, ns_45)
    set_val(CO_SUPPRESSION_YEAR, "Nova Scotia",           47, gas, ns_47)
    set_val(CO_SUPPRESSION_YEAR, "Nunavut",               16, gas, nu_16)
    set_val(CO_SUPPRESSION_YEAR, "Nunavut",               33, gas, nu_33)
    set_val(CO_SUPPRESSION_YEAR, "Nunavut",               34, gas, nu_34)
    set_val(CO_SUPPRESSION_YEAR, "Nunavut",               45, gas, nu_45)
    set_val(CO_SUPPRESSION_YEAR, "Nunavut",               47, gas, nu_47)

# ── Solve simple suppression years (2006+) ────────────────────────────────────
# NS no longer suppressed — included in known provinces for Construction pool.
print(f"Solving {SIMPLE_SUPPRESSION_YEARS.start}–{SIMPLE_SUPPRESSION_YEARS.stop - 1}...")
for year in SIMPLE_SUPPRESSION_YEARS:
    for gas in GAS_COLS:
        k47    = get(year, "Canada", 47, gas) - sum_regions(year, 47, gas, known_provinces + ["Nova Scotia"])
        nwt_47 = ratios_NWT[gas] * k47
        nu_47  = ratios_Nu[gas]  * k47

        nwt_45 = (get(year, "Northwest Territories", 46, gas)
                  + nwt_47
                  + get(year, "Northwest Territories", 48, gas))
        nu_45  = (get(year, "Nunavut", 46, gas)
                  + nu_47
                  + get(year, "Nunavut", 48, gas))

        nwt_16 = (get(year, "Northwest Territories", 0, gas)
                  - sum_toplevel(year, "Northwest Territories", gas, exclude_idxs={0, 16, 45})
                  - nwt_45)
        nu_16  = (get(year, "Nunavut", 0, gas)
                  - sum_toplevel(year, "Nunavut", gas, exclude_idxs={0, 16, 45})
                  - nu_45)

        set_val(year, "Northwest Territories", 16, gas, nwt_16)
        set_val(year, "Northwest Territories", 45, gas, nwt_45)
        set_val(year, "Northwest Territories", 47, gas, nwt_47)
        set_val(year, "Nunavut",               16, gas, nu_16)
        set_val(year, "Nunavut",               45, gas, nu_45)
        set_val(year, "Nunavut",               47, gas, nu_47)

print("Suppressed values solved.\n")

# ── Write Stage 1 output ──────────────────────────────────────────────────────
print(f"Writing solved NIR → {OUT_NIR}")
headers = list(raw.columns)
with open(str(OUT_NIR), "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    for key in row_order:
        row = data[key]
        out_row = []
        for h in headers:
            v = row.get(h)
            if v is None:
                out_row.append("")
            else:
                try:
                    out_row.append(float(v))
                except (ValueError, TypeError):
                    out_row.append(v)
        writer.writerow(out_row)

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — Map NIR rows to CIMS branch nodes
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("STAGE 2: Mapping NIR → CIMS branches")
print("=" * 60)

# Build lookup: (Source, Sector, Sub-sector, Sub-sub-sector) -> CIMS node suffix
nir_to_cims = {
    (m["Source"].strip(), m["Sector"].strip(),
     m["Sub-sector"].strip(), m["Sub-sub-sector"].strip()): m["CIMS nodes"].strip()
    for m in mapping_rows
}

# Identify CIMS parent nodes — skipped when their children are present
all_cims_nodes = set(nir_to_cims.values())
cims_parents = {
    node for node in all_cims_nodes
    if any(other != node and other.startswith(node + ".") for other in all_cims_nodes)
}
if cims_parents:
    print(f"  CIMS parent nodes (skipped when children present): {sorted(cims_parents)}")

# ── Match NIR rows to CIMS nodes ─────────────────────────────────────────────
print("  Building CIMS output rows...")

matched          = []   # (region, year, cims_node, row)
inventory_totals = []   # idx=0 rows → CIMS.CAN or CIMS.CAN.[region]

for (year, region, index), row in data.items():
    if year < 2000 or region in SKIP_REGIONS:
        continue

    if index == 0:
        cims_branch = "CIMS.CAN" if region == "Canada" else f"CIMS.CAN.{region}"
        inventory_totals.append((region, year, cims_branch, row))
        continue

    if not INCLUDE_CANADA_SECTORS and region == "Canada":
        continue

    nir_key = (
        (row.get("Source")        or "").strip(),
        (row.get("Sector")        or "").strip(),
        (row.get("Sub-sector")    or "").strip(),
        (row.get("Sub-sub-sector") or "").strip(),
    )
    if nir_key in nir_to_cims:
        matched.append((region, year, nir_to_cims[nir_key], row))

# ── Build output rows ─────────────────────────────────────────────────────────
by_region_year = defaultdict(list)
for region, year, cims_node, row in matched:
    by_region_year[(region, year)].append((cims_node, row))

output_rows = []

for region, year, cims_branch, row in inventory_totals:
    out = {"CIMS Branch": cims_branch, "Year": year}
    for gas in GAS_LABELS:
        out[gas] = to_float(row.get(gas))
    output_rows.append(out)

for (region, year), entries in sorted(by_region_year.items()):
    nodes_present = {node for node, _ in entries}
    for cims_node, row in entries:
        if cims_node in cims_parents:
            if any(o != cims_node and o.startswith(cims_node + ".") for o in nodes_present):
                continue
        cims_branch = f"CIMS.CAN.{region}.{cims_node.lstrip('.')}"
        out = {"CIMS Branch": cims_branch, "Year": year}
        for gas in GAS_LABELS:
            out[gas] = to_float(row.get(gas))
        output_rows.append(out)

print(f"  Total CIMS output rows: {len(output_rows)}")

# ── Balance check: CIMS sector sums must match NIR inventory totals ───────────
print("\n  Validating totals balance (all gases)...")
cims_sums = defaultdict(lambda: defaultdict(float))
for (region, year), entries in by_region_year.items():
    nodes_present = {node for node, _ in entries}
    for cims_node, row in entries:
        if cims_node in cims_parents:
            if any(o != cims_node and o.startswith(cims_node + ".") for o in nodes_present):
                continue
        for gas in GAS_LABELS:
            cims_sums[(region, year)][gas] += to_float(row.get(gas))

balance_mismatches = []
for (region, year), gas_totals in cims_sums.items():
    for gas in GAS_LABELS:
        nir_total  = to_float(data.get((year, region, 0), {}).get(gas))
        cims_total = gas_totals.get(gas, 0.0)
        diff       = abs(cims_total - nir_total)
        if diff > 0.5:
            balance_mismatches.append((region, year, gas, nir_total, cims_total, diff))

if balance_mismatches:
    print(f"\n  {'!'*58}")
    print(f"  {'*** BALANCE WARNING — OUTPUTS MAY BE INCOMPLETE ***':^58}")
    print(f"  {'!'*58}")
    print(f"\n  CIMS sector sums do not match NIR totals ({len(balance_mismatches)} gas×region×year):")
    for reg, yr, gas, nir, cims, diff in sorted(balance_mismatches)[:30]:
        print(f"    {yr} {reg:28s} {gas:20s} NIR={nir:.2f}  CIMS={cims:.2f}  diff={diff:.4f}")
    if len(balance_mismatches) > 30:
        print(f"    ... and {len(balance_mismatches) - 30} more")
    print(f"\n  This usually means the mapping is missing NIR rows with non-zero emissions.")
else:
    print("  ✓ All region+year totals balance across all gases.")

# ── Build long format as Polars DataFrame ────────────────────────────────────
long_rows = []
for out in output_rows:
    branch = out["CIMS Branch"]
    parts  = branch.split(".")
    region = parts[2] if len(parts) >= 3 else "Canada"
    for gas, label in GAS_LABELS.items():
        val = out.get(gas)
        if val is not None:
            long_rows.append({
                "CIMS Branch":         branch,
                "Region":              region,
                "Year":                out["Year"],
                "Emissions Type (kt)": label,
                "Value":               val,
            })

df_long = pl.DataFrame(long_rows, schema={
    "CIMS Branch":         pl.Utf8,
    "Region":              pl.Utf8,
    "Year":                pl.Int64,
    "Emissions Type (kt)": pl.Utf8,
    "Value":               pl.Float64,
})

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — Roll up CIMS ancestor totals
# Calls add_totals() from add_cims_totals.py once per emissions type so gases
# are never summed together.
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("STAGE 3: Rolling up CIMS ancestor totals")
print("=" * 60)

COL_BRANCH = "CIMS Branch"
COL_REGION = "Region"
COL_YEAR   = "Year"
COL_ETYPE  = "Emissions Type (kt)"
COL_VALUE  = "Value"

# Separate inventory total rows from sector leaves.
# Inventory totals must not be passed to add_totals — they would be summed with
# the ancestor rollup of the same branch and double the value.
df_inventory = df_long.filter(pl.col(COL_BRANCH).map_elements(is_inventory_total, return_dtype=pl.Boolean))
df_sectors   = df_long.filter(~pl.col(COL_BRANCH).map_elements(is_inventory_total, return_dtype=pl.Boolean))

# Run add_totals separately for each emissions type so gases stay separate.
# add_cims_totals is a pandas-based util — convert to pandas for the call only.
totals_by_type = []
for etype in df_sectors[COL_ETYPE].unique().to_list():
    df_in = (
        df_sectors
        .filter(pl.col(COL_ETYPE) == etype)
        .select([COL_BRANCH, COL_REGION, COL_YEAR, COL_VALUE])
        .rename({COL_BRANCH: "CIMS_Branch"})
        .to_pandas()
    )
    df_result = add_totals(df_in)
    result_pl = (
        pl.from_pandas(df_result)
        .rename({"CIMS_Branch": COL_BRANCH})
        .with_columns(pl.lit(etype).alias(COL_ETYPE))
    )
    totals_by_type.append(result_pl)

df_totals = pl.concat(totals_by_type)
df_totals = df_totals.select([COL_BRANCH, COL_REGION, COL_YEAR, COL_ETYPE, COL_VALUE])

# Drop ancestor rows that match inventory total branches and add official ones back
df_totals = df_totals.filter(~pl.col(COL_BRANCH).map_elements(is_inventory_total, return_dtype=pl.Boolean))
df_totals = pl.concat([df_inventory, df_totals])
df_totals = df_totals.sort([COL_REGION, COL_YEAR, COL_BRANCH, COL_ETYPE])

# ── Write output ──────────────────────────────────────────────────────────────
print(f"\nWriting NIR to CIMS → {OUT_CIMS}")
df_totals.write_csv(str(OUT_CIMS))
print(f"  Leaf rows:  {len(df_long):,}")
print(f"  Total rows: {len(df_totals):,}")

# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("Done.")
print(f"  Solved NIR:  {OUT_NIR}")
print(f"  CIMS output: {OUT_CIMS}")
