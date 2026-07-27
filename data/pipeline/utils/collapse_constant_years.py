"""
Collapse rows whose Value is constant across every populated Year within a
group down to a single row with a blank Year.

This mirrors the constant-value collapsing that flatten_fixed_data.py applies
to the fixed-data CSVs, but operates on an already-assembled long-format
polars DataFrame (Branch/.../Year/Value), so it can be applied to rows
sourced from processed_data / pipeline-derived data as well, at the point
where each sector's model_inputs.py builds its final output frame.

Groups with genuinely varying numeric values are left untouched (they're a
real time series). Groups with varying non-numeric literals collapse to one
default row (the most common value, blank Year) plus explicit override rows
only for the years whose value differs from that default.

The input row order is preserved: a collapsed group is placed at the
position of its first original row, and untouched rows keep their original
position, so callers don't need to re-sort afterward.
"""
from collections import Counter

import polars as pl

_DEFAULT_EXCLUDED_COLS = {"Year", "Value", "_order"}
_POS_COL = "_collapse_pos"


def _collapse_literal_varying(
    df: pl.DataFrame,
    id_cols: list[str],
    year_col: str,
    value_col: str,
    schema_cols: list[str],
) -> pl.DataFrame:
    """Python fallback for the rare case of varying non-numeric literals."""
    df = df.sort(_POS_COL)
    out_cols = schema_cols + [_POS_COL]

    out_rows: list[dict] = []
    for _key, sub in df.group_by(id_cols, maintain_order=True):
        values = sub[value_col].to_list()
        counts = Counter(values)
        max_count = max(counts.values())
        default_value = next(v for v in values if counts[v] == max_count)

        base_row = sub.row(0, named=True)  # smallest _collapse_pos in the group
        base_row = dict(base_row)
        base_row[year_col] = ""
        base_row[value_col] = default_value
        out_rows.append({c: base_row[c] for c in out_cols})

        for row, value in zip(sub.iter_rows(named=True), values):
            if value != default_value:
                out_rows.append({c: row[c] for c in out_cols})

    if not out_rows:
        return df.select(out_cols).clear()
    return pl.DataFrame(out_rows)


def collapse_constant_years(
    df: pl.DataFrame,
    id_cols: list[str] | None = None,
    year_col: str = "Year",
    value_col: str = "Value",
    order_col: str | None = "_order",
) -> pl.DataFrame:
    """
    Collapse (id_cols)-groups whose Value never changes across their
    populated Year rows down to a single row with a blank Year.

    id_cols defaults to every column except Year, Value, and _order.
    Row order is preserved (a collapsed group surfaces at the position of
    its first original row); order_col, if present, is only used as a
    secondary tiebreak and is otherwise unnecessary — original physical
    row order is tracked internally regardless.
    """
    if df.height == 0:
        return df

    schema_cols = df.columns

    if id_cols is None:
        exclude = set(_DEFAULT_EXCLUDED_COLS)
        id_cols = [c for c in schema_cols if c not in exclude]

    # Track each row's original position so the final output can be
    # restored to (effectively) the input's row order.
    work = df.with_row_index(_POS_COL)
    if order_col is not None and order_col in schema_cols:
        work = work.sort(order_col).with_columns(
            pl.int_range(pl.len()).alias(_POS_COL)
        )

    # Rows with no Value at all never participate in collapsing.
    blank_mask = pl.col(value_col).is_null() | (pl.col(value_col).str.strip_chars() == "")
    blank_rows = work.filter(blank_mask).select(schema_cols + [_POS_COL])
    data_rows = work.filter(~blank_mask)

    if data_rows.height == 0:
        return df

    # Normalize for equality comparisons: numerically-equal values (e.g.
    # "30" and "30.0") should be treated as the same value, like
    # flatten_fixed_data.py's float-aware comparison.
    data_rows = data_rows.with_columns(
        pl.col(value_col).cast(pl.Float64, strict=False).alias("_num")
    ).with_columns(
        pl.when(pl.col("_num").is_not_null())
        .then(pl.col("_num").cast(pl.Utf8))
        .otherwise(pl.col(value_col))
        .alias("_cmp")
    )

    stats = data_rows.group_by(id_cols).agg([
        pl.col("_cmp").n_unique().alias("_n_unique"),
        pl.col(_POS_COL).min().alias("_grp_pos"),
    ])
    data_rows = data_rows.join(stats, on=id_cols, how="left", nulls_equal=True)

    constant = data_rows.filter(pl.col("_n_unique") == 1)
    varying = data_rows.filter(pl.col("_n_unique") > 1)

    frames = [blank_rows]

    if constant.height > 0:
        collapsed = (
            constant.sort(_POS_COL)
            .group_by(id_cols, maintain_order=True)
            .agg(
                [pl.first(c).alias(c) for c in schema_cols if c not in id_cols]
                + [pl.first("_grp_pos").alias(_POS_COL)]
            )
        )

        # market_share_total always carries an explicit Year (2000), even
        # when its group collapses to a single value -- unlike every other
        # Parameter, a blank Year here would be read as "constant across
        # all years" rather than "an initial condition at year 2000".
        if "Parameter" in schema_cols:
            collapsed = collapsed.with_columns(
                pl.when(pl.col("Parameter") == "market_share_total")
                .then(pl.col(year_col))
                .otherwise(pl.lit(""))
                .alias(year_col)
            )
        else:
            collapsed = collapsed.with_columns(pl.lit("").alias(year_col))

        collapsed = collapsed.select(schema_cols + [_POS_COL])

        # If Source is part of the grouping key, a series that's constant at
        # the SAME value across a Source change (e.g. historical "CER/Stats
        # Can" years vs. extrapolated "Assumptions" years) would otherwise
        # collapse to two identical-value blank-Year rows, one per Source.
        # Keep only the earliest (first source) in that case; a Source
        # change that coincides with a genuine value change still keeps its
        # own row, since those represent different facts.
        if "Source" in id_cols:
            dedup_cols = [c for c in id_cols if c != "Source"]
            collapsed = (
                collapsed.with_columns(
                    pl.col(value_col).cast(pl.Float64, strict=False).alias("_dedup_num")
                )
                .with_columns(
                    pl.when(pl.col("_dedup_num").is_not_null())
                    .then(pl.col("_dedup_num").cast(pl.Utf8))
                    .otherwise(pl.col(value_col))
                    .alias("_dedup_cmp")
                )
                .sort(_POS_COL)
                .group_by(dedup_cols + ["_dedup_cmp"], maintain_order=True)
                .agg([
                    pl.first(c).alias(c)
                    for c in (schema_cols + [_POS_COL])
                    if c not in dedup_cols
                ])
                .select(schema_cols + [_POS_COL])
            )

        frames.append(collapsed)

    if varying.height > 0:
        bad_counts = varying.group_by(id_cols).agg(
            pl.col("_num").null_count().alias("_n_bad")
        )
        varying = varying.join(bad_counts, on=id_cols, how="left", nulls_equal=True)

        numeric_varying = varying.filter(pl.col("_n_bad") == 0).select(schema_cols + [_POS_COL])
        literal_varying = varying.filter(pl.col("_n_bad") > 0).select(schema_cols + [_POS_COL])

        if numeric_varying.height > 0:
            frames.append(numeric_varying)
        if literal_varying.height > 0:
            frames.append(
                _collapse_literal_varying(literal_varying, id_cols, year_col, value_col, schema_cols)
            )

    result = pl.concat(frames, how="vertical_relaxed")
    return result.sort(_POS_COL).select(schema_cols)
