"""
Drop rows for any group (typically Region) whose Value is zero — or
entirely missing — across every row in the group, i.e. there is no real
activity to request.

Meant to be applied to activity data already filtered down to a single
variable/series (one `oil.filter(pl.col('Variable') == sub)` call, say)
before it's built into service_request rows, so a region with no activity
for that series never produces a row pointing at a branch it doesn't have.

Mirrors the ad hoc zero-drop block originally written in
coal_mining/model_inputs.py's _build_met_finishing_rows.
"""

import polars as pl


def drop_zero_activity_regions(
    df: pl.DataFrame,
    region_col: str = 'Region',
    value_col: str = 'Value',
) -> pl.DataFrame:
    """
    Drop every row for any region_col group whose value_col is zero (or
    null/blank) across the whole group.

    `df` should already be scoped to a single activity series (e.g. one
    Variable's rows) — grouping by region alone across multiple series
    would otherwise conflate unrelated values.
    """
    if df.is_empty():
        return df
    nonzero_regions = (
        df.with_columns(pl.col(value_col).cast(pl.Float64, strict=False).alias('_num'))
        .group_by(region_col)
        .agg(pl.col('_num').abs().max().fill_null(0.0).alias('_max_val'))
        .filter(pl.col('_max_val') > 0)
        .select(region_col)
    )
    return df.join(nonzero_regions, on=region_col, how='inner')
