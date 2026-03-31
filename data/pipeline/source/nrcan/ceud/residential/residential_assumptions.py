import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import altair as alt
    import polars as pl
    import pandas as pd
    from pathlib import Path
    import sys

    project_root = Path(__file__).parent.parent.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from source.nrcan.ceud.residential.residential import (
        extract_all_provinces,
        load_projection_params,
        PROVINCES,
    )
    from utils.extensions.data_extensions import (
        extend_series_constant,
        extend_series_linear,
        extend_series_trend_decline,
    )


@app.cell
def _():
    mo.md("""
    # 📊 Residential Data Projections
    """)
    return


@app.cell
def _():
    mo.md("""
    ## 1. Select provinces
    """)
    return


@app.cell
def _():
    province_input = mo.ui.text(
        value='BC, AB, ON, MB',
        label="Provinces (comma-separated):",
        full_width=True,
    )
    province_input
    return (province_input,)


@app.cell
def _(province_input):
    def load():
        provs = [p.strip().upper() for p in province_input.value.split(',')
                 if p.strip().upper() in PROVINCES]
        if not provs:
            return pl.DataFrame(), [], mo.md("⚠️ No valid province codes entered.")
        try:
            # extract_all_provinces now returns dict[str, pl.DataFrame]
            raw = extract_all_provinces(provs)
            combined = pl.concat(list(raw.values()), how='diagonal_relaxed')
            return combined, list(raw.keys()), mo.md(f"✅ Loaded: {', '.join(raw.keys())}")
        except Exception as exc:
            return pl.DataFrame(), [], mo.md(f"❌ Error: {exc}")

    pipeline_df, provinces, load_status = load()
    load_status
    return pipeline_df, provinces


@app.cell
def _():
    mo.md("""
    ## 2. Select Data Series
    """)
    return


@app.cell
def _():
    series_selector = mo.ui.dropdown(
        options={
            'Housing Stock (households)':  'housing_thousand',
            'Building Shares (%)':         'building_shares',
            'Floor Space (m²/building)':   'floorspace_per_building',
            'Water Heating (GJ/GJ)':       'wh_lowmed',
            'Cooling (GJ/GJ)':             'cooling_share_data',
        },
        value='Housing Stock (households)',
        label="Data Series:",
    )
    series_selector
    return (series_selector,)


@app.cell
def _():
    mo.md("""
    ## 3. Projection Parameters

    - **Housing Stock**: Linear growth (CAGR per province)
    - **Building Shares & Floor Space**: Trend + decline
    - **Water Heating & Cooling**: Constant (holds 2022 value)
    """)
    return


@app.cell
def _():
    """Load existing projection parameters from the assumptions CSV."""
    def load_config():
        csv_path = Path(r'C:\cims\data\raw_data\assumptions\residential_assumptions.csv')
        try:
            params = load_projection_params(csv_path)
            return params, mo.md(f"✅ Loaded parameters from {csv_path}")
        except Exception as exc:
            return {}, mo.md(f"⚠️ Could not load parameters: {exc}")

    params_config, config_status = load_config()
    config_status
    return (params_config,)


@app.cell
def _(params_config, provinces, series_selector):
    def create_param_inputs():
        if series_selector.value in ['wh_lowmed', 'cooling_share_data']:
            return None, mo.md("*No parameters needed — constant method holds 2022 value.*")

        method = ('linear' if series_selector.value == 'housing_thousand'
                  else 'trend_decline')

        def _get(series_key, prov, param, default):
            try:
                cfg = params_config.get(series_key, {})
                if prov in cfg:
                    pc = cfg[prov]
                else:
                    return default
                if param == 'rate1' and 'periods' in pc:
                    return pc['periods'][0][2]
                if param == 'rate2' and 'periods' in pc:
                    return pc['periods'][1][2]
                if param == 'decline1' and 'decrease_periods' in pc:
                    return pc['decrease_periods'][0][2]
                if param == 'decline2' and 'decrease_periods' in pc:
                    return pc['decrease_periods'][1][2]
            except Exception:
                pass
            return default

        sk = series_selector.value

        if method == 'linear':
            form = mo.md(
                "**Linear Growth Parameters** (decimal, e.g. 0.01 = 1%)\n\n" +
                "\n".join(
                    f"**{p}:** {{rate1_{p}}} (2023-2050) | {{rate2_{p}}} (2051-2100)"
                    for p in provinces
                )
            ).batch(**{
                f'rate1_{p}': mo.ui.text(
                    value=str(_get(sk, p, 'rate1', 0.01)),
                    label=f'{p} Period 1 (2023-2050)',
                )
                for p in provinces
            } | {
                f'rate2_{p}': mo.ui.text(
                    value=str(_get(sk, p, 'rate2', 0.005)),
                    label=f'{p} Period 2 (2051-2100)',
                )
                for p in provinces
            }).form()
            return form, form

        else:  # trend_decline
            form = mo.md(
                "**Trend + Decline Parameters** (decimal, e.g. 0.05 = 5% decrease)\n\n" +
                "\n".join(
                    f"**{p}:** {{dec1_{p}}} (2031-2050) | {{dec2_{p}}} (2051-2100)"
                    for p in provinces
                )
            ).batch(**{
                f'dec1_{p}': mo.ui.text(
                    value=str(_get(sk, p, 'decline1', 0.05)),
                    label=f'{p} Period 1 Decline (2031-2050)',
                )
                for p in provinces
            } | {
                f'dec2_{p}': mo.ui.text(
                    value=str(_get(sk, p, 'decline2', 0.10)),
                    label=f'{p} Period 2 Decline (2051-2100)',
                )
                for p in provinces
            }).form()
            return form, form

    param_form, param_display = create_param_inputs()
    param_display
    return (param_form,)


@app.cell
def _():
    mo.md("""
    ## 4. Set parameters
    """)
    return


@app.cell
def _(param_form, pipeline_df, provinces, series_selector):
    """
    Pull historical series from the pipeline DataFrame, apply the chosen
    projection method, and return a combined long-format DataFrame with a
    'type' column ('Historical' or 'Projected').
    """
    form_value = param_form.value if param_form is not None and hasattr(param_form, 'value') else None

    def _get_series(prov: str, variable: str, category: str) -> pd.Series:
        """Extract a year-indexed float Series for one province / variable / category."""
        mask = (
            (pl.col('province') == prov) &
            (pl.col('variable') == variable)
        )
        if category:
            mask = mask & (pl.col('category') == category)
        subset = pipeline_df.filter(mask)
        # Use get_column/to_list on numeric columns only — avoids pyarrow dependency
        years  = subset.get_column('year').cast(pl.Int64).to_list()
        values = subset.get_column('value').cast(pl.Float64).to_list()
        return pd.Series(values, index=years, dtype=float).sort_index()

    def _project_series(prov: str, s: pd.Series) -> pd.Series:
        """Apply the selected projection method to a single series."""
        var = series_selector.value

        if var in ['wh_lowmed', 'cooling_share_data']:
            return extend_series_constant(s)

        if var == 'housing_thousand':
            if form_value:
                try:
                    r1 = float(form_value.get(f'rate1_{prov}', 0.01))
                    r2 = float(form_value.get(f'rate2_{prov}', 0.005))
                except Exception:
                    r1, r2 = 0.01, 0.005
            else:
                r1, r2 = 0.01, 0.005
            return extend_series_linear(s, periods=[(2023, 2051, r1), (2051, 2101, r2)])

        # building_shares or floorspace_per_building → trend_decline
        if form_value:
            try:
                d1 = float(form_value.get(f'dec1_{prov}', 0.05))
                d2 = float(form_value.get(f'dec2_{prov}', 0.10))
            except Exception:
                d1, d2 = 0.05, 0.10
        else:
            d1, d2 = 0.05, 0.10
        return extend_series_trend_decline(
            s,
            trend_start=2000, trend_end=2022, trend_period=(2023, 2031),
            decrease_periods=[(2031, 2051, d1), (2051, 2101, d2)],
        )

    def project_all() -> pd.DataFrame:
        """
        Return a tidy pandas DataFrame ready for Altair with columns:
        Year, Value, Province, Category, Type (Historical / Projected)
        """
        rows = []
        var = series_selector.value

        # Determine which categories to iterate over
        if var == 'building_shares':
            categories = ['Single Detached', 'Single Attached', 'Apartments', 'Mobile Homes']
        elif var == 'floorspace_per_building':
            categories = ['Single Detached', 'Single Attached', 'Apartments', 'Mobile Homes']
        elif var == 'wh_lowmed':
            categories = ['']   # scalar — stored with empty category
        elif var == 'cooling_share_data':
            categories = ['Room', 'Central']
        else:  # housing_thousand
            categories = ['']

        for prov in provinces:
            for cat in categories:
                hist = _get_series(prov, var, cat)
                if hist.dropna().empty:
                    continue

                # Special case: Apartments in building_shares = 1 - sum(others)
                if var == 'building_shares' and cat == 'Apartments':
                    projected_all = extend_series_trend_decline(
                        hist,
                        trend_start=2000, trend_end=2022, trend_period=(2023, 2031),
                        decrease_periods=[(2031, 2051, 0.05), (2051, 2101, 0.10)],
                    )
                    # We just chart what we have; the residual logic lives in
                    # apply_extensions inside residential.py
                    proj = projected_all[projected_all.index > int(hist.dropna().index.max())]
                else:
                    proj = _project_series(prov, hist)
                    proj = proj[proj.index > int(hist.dropna().index.max())]

                for yr, val in hist.dropna().items():
                    rows.append({'Year': int(yr), 'Value': float(val),
                                 'Province': prov, 'Category': cat or 'Total',
                                 'Type': 'Historical'})
                for yr, val in proj.dropna().items():
                    rows.append({'Year': int(yr), 'Value': float(val),
                                 'Province': prov, 'Category': cat or 'Total',
                                 'Type': 'Projected'})

        return pd.DataFrame(rows)

    projected_df = project_all()

    if projected_df.empty:
        proj_status = mo.md("⚠️ No data found for the selected series and provinces.")
    else:
        proj_status = mo.md(
            f"✅ Projected {projected_df['Province'].nunique()} province(s), "
            f"{projected_df['Category'].nunique()} category/categories."
        )
    proj_status
    return (projected_df,)


@app.cell
def _():
    mo.md("""
    ## 5. Visualisation
    """)
    return


@app.cell
def _(projected_df, series_selector):
    def build_chart():
        if projected_df is None or projected_df.empty:
            return mo.md("*No projected data — check province selection.*")

        var = series_selector.value
        series_labels = {
            'housing_thousand':     'Housing Stock (households)',
            'building_shares':      'Building Shares (fraction)',
            'floorspace_per_building': 'Floor Space (m²/building)',
            'wh_lowmed':            'Water Heating (GJ/GJ)',
            'cooling_share_data':   'Cooling (GJ/GJ)',
        }
        y_label = series_labels.get(var, 'Value')
        title   = f"Projections: {y_label}"

        # Vertical rule at 2022 (end of historical data)
        rule = (
            alt.Chart(pd.DataFrame({'Year': [2022]}))
            .mark_rule(color='gray', strokeDash=[3, 3])
            .encode(x='Year:Q')
        )

        if var in ['building_shares', 'floorspace_per_building', 'cooling_share_data']:
            # Colour = category (building type / cooling type)
            # Stroke dash = province
            color_enc = alt.Color('Category:N', legend=alt.Legend(title='Category'))
            dash_enc  = alt.StrokeDash('Province:N', legend=alt.Legend(title='Province'))

            base = alt.Chart(projected_df).encode(
                x=alt.X('Year:Q', scale=alt.Scale(domain=[2000, 2100]), title='Year'),
                y=alt.Y('Value:Q', title=y_label),
                color=color_enc,
                strokeDash=dash_enc,
                tooltip=['Year:Q', 'Value:Q', 'Province:N', 'Category:N', 'Type:N'],
            )
            hist_line = base.transform_filter(alt.datum.Type == 'Historical').mark_line(
                point=True, strokeWidth=2)
            proj_line = base.transform_filter(alt.datum.Type == 'Projected').mark_line(
                strokeWidth=2)

        else:
            # Colour = province
            base = alt.Chart(projected_df).encode(
                x=alt.X('Year:Q', scale=alt.Scale(domain=[2000, 2100]), title='Year'),
                y=alt.Y('Value:Q', title=y_label),
                color=alt.Color('Province:N', legend=alt.Legend(title='Province')),
                tooltip=['Year:Q', 'Value:Q', 'Province:N', 'Type:N'],
            )
            hist_line = base.transform_filter(alt.datum.Type == 'Historical').mark_line(
                point=True, strokeWidth=2)
            proj_line = base.transform_filter(alt.datum.Type == 'Projected').mark_line(
                strokeDash=[5, 5], strokeWidth=2)

        chart = (hist_line + proj_line + rule).properties(
            title=title, width=800, height=500,
        ).interactive()

        return chart

    output = build_chart()
    output
    return


if __name__ == "__main__":
    app.run()
