import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium", app_title="Control Table")

with app.setup:
    import marimo as mo
    import pandas as pd
    import json
    import re
    from pathlib import Path

    MAPPINGS_PATH = Path('C:/cims/data/mappings_conversions')

    # -------------------------------------------------------------------------
    # CONTROLS — edit values here directly, or use the UI below and click Save.
    # This dict lives at module level so pipeline scripts can import it directly:
    #   from control import CONTROLS
    # -------------------------------------------------------------------------
    CONTROLS = {
        "currency_year_cims": 2025,
        "currency_year_jcims": 2005,
        "currency_year_cer_enduse": 2025,
        "currency_year_cer_macro": 2017,
        "currency_year_cer_benchmark": 2022,
        "currency_country_cer_benchmark": "USD",
        "cer_ef_reference_scenario": "Current Measures",
        "last_historical_year": {
            "CEUD": 2023,
        },
    }


@app.cell
def _():
    mo.md("""
    # Control
    Define important variables like currency year, currency country, and reference scenario.
    Values shown are loaded from the `CONTROLS` dict in the `with app.setup` block of this
    file. Adjust with the widgets below, then click **Save** to write them back.
    """)
    return


@app.cell
def _():
    currency_year_cims = mo.ui.number(
        value=CONTROLS["currency_year_cims"], label="Currency Year CIMS"
    )
    currency_year_jcims = mo.ui.number(
        value=CONTROLS["currency_year_jcims"], label="Currency Year JCIMS"
    )
    currency_year_cer_enduse = mo.ui.number(
        value=CONTROLS["currency_year_cer_enduse"], label="CER End Use Price Year"
    )
    currency_year_cer_macro = mo.ui.number(
        value=CONTROLS["currency_year_cer_macro"], label="CER Macro Indicator Year"
    )
    currency_year_cer_benchmark = mo.ui.number(
        value=CONTROLS["currency_year_cer_benchmark"], label="CER Benchmark Currency Year"
    )
    currency_country_cer_benchmark = mo.ui.dropdown(
        options=["USD", "CAD"],
        value=CONTROLS["currency_country_cer_benchmark"],
        label="CER Benchmark Currency Country",
    )
    cer_ef_reference_scenario = mo.ui.dropdown(
        options=["Current Measures", "Canada Net-zero"],
        value=CONTROLS["cer_ef_reference_scenario"],
        label="CER EF Reference Scenario",
    )
    last_hist_year_ceud = mo.ui.number(
        value=CONTROLS["last_historical_year"]["CEUD"], label="Last Historical Year — CEUD"
    )
    return (
        cer_ef_reference_scenario,
        currency_country_cer_benchmark,
        currency_year_cer_benchmark,
        currency_year_cer_enduse,
        currency_year_cer_macro,
        currency_year_cims,
        currency_year_jcims,
        last_hist_year_ceud,
    )


@app.cell
def _(
    cer_ef_reference_scenario,
    currency_country_cer_benchmark,
    currency_year_cer_benchmark,
    currency_year_cer_enduse,
    currency_year_cer_macro,
    currency_year_cims,
    currency_year_jcims,
    last_hist_year_ceud,
):
    mo.vstack([
        mo.md("## Parameters"),
        mo.hstack([
            mo.vstack([
                currency_year_cims,
                currency_year_jcims,
                currency_year_cer_enduse,
                currency_year_cer_macro,
                currency_year_cer_benchmark,
                currency_country_cer_benchmark,
            ]),
            mo.vstack([
                cer_ef_reference_scenario,
            ]),
        ], gap=4),
        mo.md("## Last Historical Year"),
        mo.hstack([
            last_hist_year_ceud,
        ]),
    ])
    return


@app.cell
def _(
    cer_ef_reference_scenario,
    currency_country_cer_benchmark,
    currency_year_cer_benchmark,
    currency_year_cer_enduse,
    currency_year_cer_macro,
    currency_year_cims,
    currency_year_jcims,
    last_hist_year_ceud,
):
    # Build updated CONTROLS dict from current widget values
    current = {
        "currency_year_cims": int(currency_year_cims.value),
        "currency_year_jcims": int(currency_year_jcims.value),
        "currency_year_cer_enduse": int(currency_year_cer_enduse.value),
        "currency_year_cer_macro": int(currency_year_cer_macro.value),
        "currency_year_cer_benchmark": int(currency_year_cer_benchmark.value),
        "currency_country_cer_benchmark": currency_country_cer_benchmark.value,
        "cer_ef_reference_scenario": cer_ef_reference_scenario.value,
        "last_historical_year": {
            "CEUD": int(last_hist_year_ceud.value),
        },
    }
    return (current,)


@app.cell
def _(current):
    mo.vstack([
        mo.md("## Current Values"),
        mo.md("```json\n" + json.dumps(current, indent=2) + "\n```"),
    ])
    return


@app.cell
def _():
    save_button = mo.ui.run_button(label="💾 Save to control.py")
    return (save_button,)


@app.cell
def _(current, save_button):
    def _save():
        this_file = Path(__file__)
        source = this_file.read_text(encoding='utf-8')
        new_dict = '    CONTROLS = ' + json.dumps(current, indent=4).replace('\n', '\n    ')
        updated = re.sub(
            r'    CONTROLS = \{[^}]*\}',
            new_dict,
            source,
            count=1,
            flags=re.DOTALL,
        )
        this_file.write_text(updated, encoding='utf-8')
        return mo.md(f"✅ Saved to `{this_file}`")

    if save_button.value:
        save_status = _save()
    else:
        save_status = mo.md("_Click Save to write current values back into this file._")

    mo.vstack([save_button, save_status])
    return


@app.cell
def _():
    mo.md("""
    # Mappings
    View the energy, region, sector, and conversion mapping tables.
    All files are read from `C:/cims/data/mappings_conversions/`.
    """)
    return


@app.cell
def _():
    energy_map_df  = pd.read_csv(MAPPINGS_PATH / 'energy_map.csv')
    region_map_df  = pd.read_csv(MAPPINGS_PATH / 'region_map.csv')
    sector_map_df  = pd.read_csv(MAPPINGS_PATH / 'sector_map.csv')
    conversions_df = pd.read_csv(
        MAPPINGS_PATH / 'energy_conversions.csv',
        header=None,
        names=['col_a', 'col_b', 'col_c', 'col_d'],
    )
    return conversions_df, energy_map_df, region_map_df, sector_map_df


@app.cell
def _(conversions_df, energy_map_df, region_map_df, sector_map_df):
    mo.vstack([
        mo.md("## Energy Map"),
        mo.ui.table(energy_map_df),
        mo.md("## Region Map"),
        mo.ui.table(region_map_df),
        mo.md("## Sector Map"),
        mo.ui.table(sector_map_df),
        mo.md("## Energy Conversions"),
        mo.ui.table(conversions_df.dropna(how='all')),
    ])
    return


@app.cell
def _(energy_map_df, region_map_df, sector_map_df):
    # Export mapping DataFrames for use by other scripts
    MAPPINGS = {
        'energy_map': energy_map_df,
        'region_map': region_map_df,
        'sector_map': sector_map_df,
    }
    return


if __name__ == "__main__":
    app.run()
