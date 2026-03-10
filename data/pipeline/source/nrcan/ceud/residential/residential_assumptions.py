import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import altair as alt
    import pandas as pd
    from pathlib import Path
    import sys

    project_root = Path(__file__).parent.parent.parent.parent.parent

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from source.nrcan.ceud.residential.residential import extract_all_provinces
    from utils.extensions.dict_data_extensions import extend_data_constant, extend_data_linear, extend_trend_decline


@app.cell
def _():
    mo.md("""
    # 📊 Residential Data Projections
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ## 1. Select provinces
    """)
    return


@app.cell
def _():
    province_input = mo.ui.text(
        value='BC, AB, ON, MB',
        label="Provinces (comma-separated):",
        full_width=True
    )
    province_input
    return (province_input,)


@app.cell
def _(province_input):
    def load():
        provs = [p.strip().upper() for p in province_input.value.split(',')]
        try:
            raw = extract_all_provinces(provs)
            data = {p: {int(k) if isinstance(k, str) and k.isdigit() else k: v for k, v in d.items()} for p, d in raw.items()}
            return data, list(data.keys()), mo.md(f"✅ Loaded: {', '.join(list(data.keys()))}")
        except Exception as e:
            return {}, [], mo.md(f"❌ Error: {e}")

    pipeline_results, provinces, load_status = load()
    load_status
    return pipeline_results, provinces


@app.cell
def _():
    mo.md(r"""
    ## Select Data Series
    """)
    return


@app.cell
def _():
    mo.md("## 2. Select Data Series")
    series_selector = mo.ui.dropdown(
        options={
            'Housing Stock (thousands)': 'housing_stock',
            'Building Shares (%)': 'building_shares',
            'Floor Space (m²/building)': 'floorspace_per_building',
            'Water Heating (GJ/GJ)': 'water_heating',
            'Cooling (GJ/GJ)': 'cooling'
        },
        value='Housing Stock (thousands)',
        label="Data Series:"
    )
    series_selector
    return (series_selector,)


@app.cell
def _():
    mo.md(r"""
    ## 3. Projection Parameters

    - **Housing Stock**: Uses linear growth
    - **Building Shares & Floorspace**: Use trend decline
    - **Water Heating & Cooling**: Use constant (hold 2022 values)
    """)
    return


@app.cell
def _():
    """Load existing projection parameters from assumptions CSV"""

    def load_config():
        csv_path = Path(r'C:\cims\data\raw_data\assumptions\residential_assumptions.csv')

        PROV_NAME_TO_CODE = {
            'British Columbia': 'BC', 'Alberta': 'AB', 'Saskatchewan': 'SK',
            'Manitoba': 'MB', 'Ontario': 'ON', 'Quebec': 'QC',
            'New Brunswick': 'NB', 'Nova Scotia': 'NS', 'Prince Edward Island': 'PE',
            'Newfoundland and Labrador': 'NL', 'Territories': 'TR',
        }

        try:
            df = pd.read_csv(csv_path, header=None, dtype=str)
        except FileNotFoundError:
            return {}, mo.md(f"ℹ️ Assumptions CSV not found at {csv_path}, using defaults")
        except Exception as e:
            return {}, mo.md(f"⚠️ Error reading CSV: {e}")

        def pct(row, col):
            try:
                v = str(df.iloc[row, col]).strip().replace('%', '')
                return float(v) / 100.0 if v else None
            except:
                return None

        params_config = {}

        # Housing stock — rows 9-23 (0-indexed), province in col 5, period1 in col 9, period2 in col 10
        params_config['housing_stock'] = {'method': 'linear'}
        for row_idx in range(9, 24):
            try:
                prov_name = str(df.iloc[row_idx, 5]).strip()
            except:
                continue
            code = PROV_NAME_TO_CODE.get(prov_name)
            if code is None:
                continue
            r1, r2 = pct(row_idx, 9), pct(row_idx, 10)
            if r1 is not None and r2 is not None:
                params_config['housing_stock'][code] = {'periods': [[2023, 2051, r1], [2051, 2101, r2]]}

        # Building shares — global decline values on row 31, cols 9 & 10
        bs_dec1 = abs(pct(31, 9) or -0.05)
        bs_dec2 = abs(pct(31, 10) or -0.10)
        params_config['building_shares'] = {
            'method': 'trend_decline', 'trend_start': 2000,
            'trend_end': 2022, 'trend_period': [2023, 2031],
        }
        for code in ['AB', 'BC', 'MB', 'NB', 'NL', 'NS', 'ON', 'PE', 'QC', 'SK', 'TR']:
            params_config['building_shares'][code] = {'decrease_periods': [[2031, 2051, bs_dec1], [2051, 2101, bs_dec2]]}

        # Floorspace — global decline values on row 38, cols 9 & 10
        fs_dec1 = abs(pct(38, 9) or -0.05)
        fs_dec2 = abs(pct(38, 10) or -0.10)
        params_config['floorspace_per_building'] = {
            'method': 'trend_decline', 'trend_start': 2000,
            'trend_end': 2022, 'trend_period': [2023, 2031],
        }
        for code in ['AB', 'BC', 'MB', 'NB', 'NL', 'NS', 'ON', 'PE', 'QC', 'SK', 'TR']:
            params_config['floorspace_per_building'][code] = {'decrease_periods': [[2031, 2051, fs_dec1], [2051, 2101, fs_dec2]]}

        return params_config, mo.md(f"✅ Loaded parameters from {csv_path}")

    params_config, config_status = load_config()
    config_status
    return (params_config,)


@app.cell
def _():
    mo.md(r"""
    ## 4. Set parameters
    """)
    return


@app.cell
def _(params_config, provinces, series_selector):
    def create_param_inputs():
        # Determine method based on series type
        # Housing stock: linear
        # Building shares & floorspace: trend_decline
        # Water heating & cooling: constant (no parameters)
        if series_selector.value == 'housing_stock':
            method = 'linear'
        elif series_selector.value in ['building_shares', 'floorspace_per_building']:
            method = 'trend_decline'
        else:  # water_heating, cooling
            return None, mo.md("*No parameters needed - using constant method (holds 2022 values)*")

        # Helper to get config value with fallback to default
        def get_config_value(series_key, prov, param_path, default):
            """Navigate nested config dict safely"""
            try:
                if series_key in params_config:
                    # First try province-specific value
                    if prov in params_config[series_key]:
                        prov_config = params_config[series_key][prov]
                    # Fall back to "default" key if it exists
                    elif 'default' in params_config[series_key]:
                        prov_config = params_config[series_key]['default']
                    else:
                        return default

                    # Navigate param_path (e.g., 'periods.0.2' for rate1)
                    if param_path == 'rate1' and 'periods' in prov_config:
                        return prov_config['periods'][0][2]
                    elif param_path == 'rate2' and 'periods' in prov_config:
                        return prov_config['periods'][1][2]
                    elif param_path == 'decline1' and 'decrease_periods' in prov_config:
                        return prov_config['decrease_periods'][0][2]
                    elif param_path == 'decline2' and 'decrease_periods' in prov_config:
                        return prov_config['decrease_periods'][1][2]
            except:
                pass
            return default

        # Determine series key for config lookup
        series_config_key = series_selector.value

        if method == 'linear':
            # For linear: need 2 growth rates per province
            form = mo.md(f"""
            **Linear Growth Parameters** (enter as decimal, e.g., 0.01 = 1%)

            {chr(10).join([f'**{prov}:**{{rate1_{prov}}} (2023-2050) | {{rate2_{prov}}} (2051-2100)' for prov in provinces])}
            """).batch(**{
                f'rate1_{prov}': mo.ui.text(
                    value=str(get_config_value(series_config_key, prov, 'rate1', 0.01)),
                    label=f'{prov} Period 1 (2023-2050)'
                ) 
                for prov in provinces
            } | {
                f'rate2_{prov}': mo.ui.text(
                    value=str(get_config_value(series_config_key, prov, 'rate2', 0.005)),
                    label=f'{prov} Period 2 (2051-2100)'
                ) 
                for prov in provinces
            }).form()

            return form, form

        elif method == 'trend_decline':
            # For trend_decline: need 2 decline percentages per province
            form = mo.md(f"""
            **Trend + Decline Parameters** (enter as decimal, e.g., 0.05 = 5% decrease)

            {chr(10).join([f'**{prov}:**{{dec1_{prov}}} (2031-2050) | {{dec2_{prov}}} (2051-2100)' for prov in provinces])}
            """).batch(**{
                f'dec1_{prov}': mo.ui.text(
                    value=str(get_config_value(series_config_key, prov, 'decline1', 0.05)),
                    label=f'{prov} Period 1 Decline (2031-2050)'
                ) 
                for prov in provinces
            } | {
                f'dec2_{prov}': mo.ui.text(
                    value=str(get_config_value(series_config_key, prov, 'decline2', 0.10)),
                    label=f'{prov} Period 2 Decline (2051-2100)'
                ) 
                for prov in provinces
            }).form()

            return form, form

    param_form, param_display = create_param_inputs()
    param_display
    return (param_form,)


@app.cell
def _(pipeline_results, provinces, series_selector):
    # Extract series data

    def extract():
        data = {}
        for prov in provinces:
            if prov not in pipeline_results:
                continue

            pdata = pipeline_results[prov]

            if series_selector.value == 'housing_stock':
                # Housing stock is total - no building type needed
                if 'housing_thousand' in pdata:
                    data[prov] = pdata['housing_thousand']

            elif series_selector.value == 'building_shares':
                # Building shares - extract ALL building types for later residual calculation
                if 'building_shares' in pdata:
                    # Store all building types for this province
                    if prov not in data:
                        data[prov] = {}
                    for btype in ['Single Detached', 'Single Attached', 'Apartments', 'Mobile Homes']:
                        if btype in pdata['building_shares']:
                            data[prov][btype] = pdata['building_shares'][btype]

            elif series_selector.value == 'floorspace_per_building':
                # Floorspace - extract ALL building types
                if 'floorspace_per_building' in pdata:
                    if prov not in data:
                        data[prov] = {}
                    for btype in ['Single Detached', 'Single Attached', 'Apartments', 'Mobile Homes']:
                        if btype in pdata['floorspace_per_building']:
                            data[prov][btype] = pdata['floorspace_per_building'][btype]

            elif series_selector.value == 'water_heating':
                # Water heating - extract lowmed and high density
                if prov not in data:
                    data[prov] = {}
                if 'wh_lowmed' in pdata:
                    data[prov]['Low/Med Density'] = pdata['wh_lowmed']
                if 'wh_high' in pdata:
                    data[prov]['High Density'] = pdata['wh_high']

            elif series_selector.value == 'cooling':
                # Cooling - extract room and central types
                if 'cooling_share_data' in pdata:
                    if prov not in data:
                        data[prov] = {}
                    # cooling_share_data contains different cooling types
                    cooling_data = pdata['cooling_share_data']
                    # Extract Room and Central if they exist
                    if 'Room' in cooling_data:
                        data[prov]['Room'] = cooling_data['Room']
                    if 'Central' in cooling_data:
                        data[prov]['Central'] = cooling_data['Central']

        return data

    series_data = extract()

    # Show status
    if series_data:
        if series_selector.value in ['building_shares', 'floorspace_per_building', 'water_heating', 'cooling']:
            status_msg = mo.md(f"✅ Extracted all types for {len(series_data)} provinces")
        else:
            status_msg = mo.md(f"✅ Extracted {series_selector.value} for {len(series_data)} provinces")
    else:
        status_msg = mo.md(f"⚠️ No data found for {series_selector.value}")

    status_msg
    return (series_data,)


@app.cell
def _(param_form, provinces, series_data, series_selector):
    # Apply projections
    # Access form value at cell level for marimo reactivity
    proj_form_value = param_form.value if hasattr(param_form, 'value') else None

    def project():
        if not series_data:
            return {}, mo.md("⚠️ No series data loaded")

        # Determine method based on series type
        if series_selector.value == 'housing_stock':
            method = 'linear'
        elif series_selector.value in ['building_shares', 'floorspace_per_building']:
            method = 'trend_decline'
        else:  # water_heating, cooling
            method = 'constant'

        # Special handling for building_shares (with residual) and floorspace_per_building (all decline)
        if series_selector.value in ['building_shares', 'floorspace_per_building']:
            result = {}

            for prov in provinces:
                if prov not in series_data:
                    continue

                prov_data = series_data[prov]
                result[prov] = {}

                # For building shares: project three types, calculate apartments as residual
                # For floorspace: project all four types independently
                if series_selector.value == 'building_shares':
                    building_types_to_project = ['Single Detached', 'Single Attached', 'Mobile Homes']
                else:
                    building_types_to_project = ['Single Detached', 'Single Attached', 'Apartments', 'Mobile Homes']

                for btype in building_types_to_project:
                    if btype not in prov_data:
                        continue

                    # Building shares and floorspace always use trend_decline
                    if proj_form_value is None:
                        result[prov][btype] = extend_trend_decline(prov_data[btype])
                    else:
                        try:
                            dec1 = float(proj_form_value.get(f'dec1_{prov}', '0.05'))
                            dec2 = float(proj_form_value.get(f'dec2_{prov}', '0.10'))
                        except:
                            dec1, dec2 = 0.05, 0.10
                        decrease_periods = [(2031, 2051, dec1), (2051, 2101, dec2)]
                        result[prov][btype] = extend_trend_decline(
                            prov_data[btype],
                            trend_start=2000,
                            trend_end=2022,
                            trend_period=(2023, 2031),
                            decrease_periods=decrease_periods
                        )

                # Only for building_shares: Calculate Apartments as residual (100 - sum of others)
                if series_selector.value == 'building_shares':
                    result[prov]['Apartments'] = {}
                    all_years = set()
                    for btype in building_types_to_project:
                        if btype in result[prov]:
                            all_years.update(result[prov][btype].keys())

                    for year in all_years:
                        total_others = sum(
                            result[prov][btype].get(year, 0) 
                            for btype in building_types_to_project 
                            if btype in result[prov]
                        )
                        result[prov]['Apartments'][year] = 100.0 - total_others

            if series_selector.value == 'building_shares':
                return result, mo.md("✅ Applied trend_decline (Apartments = 100 - others)")
            else:
                return result, mo.md("✅ Applied trend_decline (all building types decline independently)")

        # Special handling for water_heating (has density types)
        elif series_selector.value == 'water_heating':
            result = {}
            for prov in provinces:
                if prov not in series_data:
                    continue
                prov_data = series_data[prov]
                result[prov] = {}
                for density_type in ['Low/Med Density', 'High Density']:
                    if density_type in prov_data:
                        result[prov][density_type] = extend_data_constant(prov_data[density_type])
            return result, mo.md("✅ Applied constant (holds 2022 values)")

        # Special handling for cooling (has Room and Central types)
        elif series_selector.value == 'cooling':
            result = {}
            for prov in provinces:
                if prov not in series_data:
                    continue
                prov_data = series_data[prov]
                result[prov] = {}
                for cooling_type in ['Room', 'Central']:
                    if cooling_type in prov_data:
                        result[prov][cooling_type] = extend_data_constant(prov_data[cooling_type])
            return result, mo.md("✅ Applied constant (holds 2022 values)")

        # Regular handling for housing stock (uses linear)
        if proj_form_value is None:
            # Use defaults
            result = {}
            for prov, data in series_data.items():
                result[prov] = extend_data_linear(data, periods=[(2023, 2051, 0.01), (2051, 2101, 0.005)])
            return result, mo.md("ℹ️ Using linear projection with defaults")

        # Parse form values
        result = {}
        for prov in provinces:
            if prov not in series_data:
                continue

            try:
                rate1 = float(proj_form_value.get(f'rate1_{prov}', '0.01'))
                rate2 = float(proj_form_value.get(f'rate2_{prov}', '0.005'))
            except:
                rate1, rate2 = 0.01, 0.005

            periods = [(2023, 2051, rate1), (2051, 2101, rate2)]
            result[prov] = extend_data_linear(series_data[prov], periods=periods)

        return result, mo.md(f"✅ Applied linear growth rates")

    projected, proj_status = project()
    proj_status
    return (projected,)


@app.cell
def _():
    mo.md(r"""
    ## 5. Visualization
    """)
    return


@app.cell
def _(param_form, projected, provinces, series_selector):
    # Access form value at cell level for marimo reactivity
    viz_form_value = param_form.value if hasattr(param_form, 'value') else None

    def viz():
        if not projected or not provinces:
            return mo.md("*No projected data*")

        # Show current parameters
        info_lines = []

        # Define tooltip based on series type
        if series_selector.value in ['building_shares', 'floorspace_per_building']:
            tooltip = [
                alt.Tooltip('Year:Q', title='Year'),
                alt.Tooltip('Value:Q', title='Value', format='.3f'),
                alt.Tooltip('Province:N', title='Province'),
                alt.Tooltip('Building Type:N', title='Building Type'),
                alt.Tooltip('Type:N', title='Data Type')
            ]
        else:
            tooltip = [
                alt.Tooltip('Year:Q', title='Year'),
                alt.Tooltip('Value:Q', title='Value', format='.2f'),
                alt.Tooltip('Province:N', title='Province'),
                alt.Tooltip('Type:N', title='Data Type')
            ]

        info = mo.md("  \n".join(info_lines)) if info_lines else mo.md("")

        # Build chart data
        chart_data = []

        # Special handling for building_shares, floorspace, water_heating, and cooling - show all types
        if series_selector.value in ['building_shares', 'floorspace_per_building']:
            for prov in provinces:
                if prov in projected:
                    for btype in ['Single Detached', 'Single Attached', 'Apartments', 'Mobile Homes']:
                        if btype in projected[prov]:
                            for year, value in projected[prov][btype].items():
                                if value is not None:
                                    chart_data.append({
                                        'Year': int(year),
                                        'Value': float(value),
                                        'Province': prov,
                                        'Category': btype,
                                        'Type': 'Historical' if year <= 2022 else 'Projected'
                                    })

        elif series_selector.value == 'water_heating':
            for prov in provinces:
                if prov in projected:
                    for density_type in ['Low/Med Density', 'High Density']:
                        if density_type in projected[prov]:
                            for year, value in projected[prov][density_type].items():
                                if value is not None:
                                    chart_data.append({
                                        'Year': int(year),
                                        'Value': float(value),
                                        'Province': prov,
                                        'Category': density_type,
                                        'Type': 'Historical' if year <= 2022 else 'Projected'
                                    })

        elif series_selector.value == 'cooling':
            for prov in provinces:
                if prov in projected:
                    for cooling_type in ['Room', 'Central']:
                        if cooling_type in projected[prov]:
                            for year, value in projected[prov][cooling_type].items():
                                if value is not None:
                                    chart_data.append({
                                        'Year': int(year),
                                        'Value': float(value),
                                        'Province': prov,
                                        'Category': cooling_type,
                                        'Type': 'Historical' if year <= 2022 else 'Projected'
                                    })

        else:
            # Regular handling for housing stock
            for prov in provinces:
                if prov in projected:
                    for year, value in projected[prov].items():
                        if value is not None:
                            chart_data.append({
                                'Year': int(year),
                                'Value': float(value),
                                'Province': prov,
                                'Type': 'Historical' if year <= 2022 else 'Projected'
                            })

        if not chart_data:
            return mo.vstack([info, mo.md("*No chart data created*")])

        df = pd.DataFrame(chart_data)

        # Series labels
        series_labels = {
            'housing_stock': 'Housing Stock (thousands)',
            'building_shares': 'Building Shares (%)',
            'floorspace_per_building': 'Floor Space (m²/building)',
            'water_heating': 'Water Heating (GJ/GJ)',
            'cooling': 'Cooling (GJ/GJ)'
        }

        base_title = series_labels.get(series_selector.value, 'Value')

        # Create chart based on series type
        if series_selector.value in ['building_shares', 'floorspace_per_building', 'water_heating', 'cooling']:
            # For these: color by Category (building type, density, or cooling type), strokeDash by Province
            if series_selector.value == 'water_heating':
                chart_title = f'Projections: {base_title} - All Density Types'
                category_label = 'Density Type'
                color_scale = alt.Scale(
                    domain=['Low/Med Density', 'High Density'],
                    range=['#1f77b4', '#ff7f0e']  # blue, orange
                )
            elif series_selector.value == 'cooling':
                chart_title = f'Projections: {base_title} - All Cooling Types'
                category_label = 'Cooling Type'
                color_scale = alt.Scale(
                    domain=['Room', 'Central'],
                    range=['#1f77b4', '#ff7f0e']  # blue, orange
                )
            else:
                chart_title = f'Projections: {base_title} - All Building Types'
                category_label = 'Building Type'
                color_scale = alt.Scale(
                    domain=['Single Detached', 'Single Attached', 'Apartments', 'Mobile Homes'],
                    range=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # blue, orange, green, red
                )

            # Define distinct stroke dash patterns for provinces (up to 10 provinces)
            dash_scale = alt.Scale(
                domain=list(set(df['Province'])),
                range=[
                    [1, 0],           # solid
                    [8, 4],           # dashed
                    [2, 2],           # dotted
                    [8, 4, 2, 4],     # dash-dot
                    [12, 4, 2, 4],    # long dash-dot
                    [2, 2, 8, 2],     # dot-dash
                    [8, 4, 2, 4, 2, 4], # dash-dot-dot
                    [12, 2],          # long dash
                    [4, 4],           # medium dash
                    [12, 4, 4, 4]     # custom
                ]
            )

            # Define tooltip
            tooltip = [
                alt.Tooltip('Year:Q', title='Year'),
                alt.Tooltip('Value:Q', title='Value', format='.3f'),
                alt.Tooltip('Province:N', title='Province'),
                alt.Tooltip('Category:N', title=category_label),
                alt.Tooltip('Type:N', title='Data Type')
            ]

            base = alt.Chart(df).encode(
                x=alt.X('Year:Q', scale=alt.Scale(domain=[2000, 2100]), title='Year'),
                y=alt.Y('Value:Q', title=base_title),
                color=alt.Color('Category:N', scale=color_scale, legend=alt.Legend(title=category_label, orient='right')),
                strokeDash=alt.StrokeDash('Province:N', scale=dash_scale, legend=alt.Legend(
                    title='Province', 
                    orient='right', 
                    symbolType='stroke', 
                    symbolStrokeWidth=4,
                    symbolSize=400,
                    labelFontSize=12
                )),
                tooltip=tooltip
            )

            hist = base.transform_filter(alt.datum.Type == 'Historical').mark_line(point=True, strokeWidth=3)
            proj_line = base.transform_filter(alt.datum.Type == 'Projected').mark_line(strokeWidth=3)

        else:
            # Regular chart for housing stock and cooling
            chart_title = f'Projections: {base_title}'

            # Define tooltip
            tooltip = [
                alt.Tooltip('Year:Q', title='Year'),
                alt.Tooltip('Value:Q', title='Value', format='.3f'),
                alt.Tooltip('Province:N', title='Province'),
                alt.Tooltip('Type:N', title='Data Type')
            ]

            base = alt.Chart(df).encode(
                x=alt.X('Year:Q', scale=alt.Scale(domain=[2000, 2100]), title='Year'),
                y=alt.Y('Value:Q', title=base_title),
                color=alt.Color('Province:N', legend=alt.Legend(title='Province')),
                tooltip=tooltip
            )

            hist = base.transform_filter(alt.datum.Type == 'Historical').mark_line(point=True, strokeWidth=2)
            proj_line = base.transform_filter(alt.datum.Type == 'Projected').mark_line(strokeDash=[5,5], strokeWidth=2)

        rule = alt.Chart(pd.DataFrame({'Year': [2022]})).mark_rule(color='gray', strokeDash=[3,3]).encode(x='Year:Q')

        chart = (hist + proj_line + rule).properties(
            title=chart_title,
            width=800, 
            height=500
        ).interactive()

        return mo.vstack([info, chart])

    output = viz()
    output
    return


if __name__ == "__main__":
    app.run()
