"""
add_cims_totals.py — CIMS Branch Totals Utility
================================================

Reads a leaf-level CIMS branch CSV and produces a separate output CSV that
includes both the original leaf rows and rolled-up totals for every ancestor.

Reads from INPUT_CSV (leaves only, written by the pipeline).
Writes to OUTPUT_CSV (leaves + ancestors — never overwrites INPUT_CSV).

A branch can appear as both a direct mapped value (e.g. Chemical Products
for Stationary/Transport) and a parent of deeper leaves (e.g.
Chemical Products.Steam). Both contributions are summed correctly.

Usage
-----
    python add_cims_totals.py

    Or import into another script:

        from add_cims_totals import add_totals
        df_with_totals = add_totals(df_leaves)
"""

import pandas as pd


# ── Configuration ─────────────────────────────────────────────────────────────

# Pipeline writes leaves here — add_totals reads from this, never writes to it
INPUT_CSV  = "C:/cims/data/processed_data/eccc/nir/nir_walkthrough_cims_leaves.csv"
# add_totals writes the full output (leaves + ancestors) here
OUTPUT_CSV = "C:/cims/data/processed_data/eccc/nir/nir_walkthrough_cims.csv"

COL_BRANCH = "CIMS_Branch"
COL_REGION = "Region"
COL_YEAR   = "Year"
COL_VALUE  = "Value"


# ── Core function (importable) ────────────────────────────────────────────────

def add_totals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame of leaf-level CIMS rows, return a new DataFrame that
    also includes rolled-up totals for every ancestor branch.

    Every row in df is treated as a true leaf. For correct results, always
    pass the raw pipeline output — never pass the output of a previous
    add_totals call.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at least CIMS_Branch, Region, Year, Value columns.
        Each row is a leaf-level mapped value from the NIR pipeline.

    Returns
    -------
    pd.DataFrame
        Leaf rows plus ancestor rows, sorted by (Region, Year, CIMS_Branch).
        Branches that appear as both a leaf and a parent of other leaves will
        have a single combined row whose value is the sum of the leaf
        contribution and all descendant leaf contributions.
    """

    # ── Step 1: collapse any duplicate (Branch, Region, Year) in the input ────
    leaves = (
        df.groupby([COL_BRANCH, COL_REGION, COL_YEAR], as_index=False)[COL_VALUE]
        .sum()
    )
    leaves[COL_VALUE] = leaves[COL_VALUE].round(4)

    # ── Step 2: for every leaf, emit a record for each ancestor ───────────────
    ancestor_records = []
    for _, row in leaves.iterrows():
        parts = row[COL_BRANCH].split(".")
        for depth in range(3, len(parts)):
            ancestor_records.append({
                COL_BRANCH: ".".join(parts[:depth]),
                COL_REGION: row[COL_REGION],
                COL_YEAR:   row[COL_YEAR],
                COL_VALUE:  row[COL_VALUE],
            })

    # ── Step 3: sum ancestor groups ───────────────────────────────────────────
    if ancestor_records:
        df_ancestors = (
            pd.DataFrame(ancestor_records)
            .groupby([COL_BRANCH, COL_REGION, COL_YEAR], as_index=False)[COL_VALUE]
            .sum()
        )
        df_ancestors[COL_VALUE] = df_ancestors[COL_VALUE].round(4)
    else:
        df_ancestors = pd.DataFrame(columns=[COL_BRANCH, COL_REGION, COL_YEAR, COL_VALUE])

    # ── Step 4: merge leaves and ancestors ────────────────────────────────────
    # A branch like Chemical Products may appear as both a leaf (8.0) and an
    # ancestor row (2.0 from Chemical Products.Steam). Combine them with groupby.
    df_out = (
        pd.concat([leaves, df_ancestors], ignore_index=True)
        .groupby([COL_BRANCH, COL_REGION, COL_YEAR], as_index=False)[COL_VALUE]
        .sum()
    )
    df_out[COL_VALUE] = df_out[COL_VALUE].round(4)
    df_out = df_out.sort_values([COL_REGION, COL_YEAR, COL_BRANCH]).reset_index(drop=True)

    return df_out


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print(f"Loading leaves from {INPUT_CSV} ...")
    df_in = pd.read_csv(INPUT_CSV)
    print(f"  {len(df_in):,} rows, {df_in[COL_BRANCH].nunique():,} unique branches.")

    print("\nComputing ancestor totals ...")
    df_out = add_totals(df_in)

    all_out = set(df_out[COL_BRANCH])
    n_leaves    = sum(1 for b in all_out if not any(o.startswith(b + ".") for o in all_out))
    n_ancestors = len(df_out) - n_leaves
    print(f"  Leaf rows:     {n_leaves:,}")
    print(f"  Ancestor rows: {n_ancestors:,}")
    print(f"  Total rows:    {len(df_out):,}")

    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅  Saved: {OUTPUT_CSV}")
