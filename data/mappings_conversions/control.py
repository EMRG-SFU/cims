import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium", app_title="Control Table")

with app.setup(hide_code=True):
    import marimo as mo
    import pandas as pd
    import json
    import re
    from pathlib import Path

    MAPPINGS_PATH = Path('C:/cims/data/mappings_conversions')

    # -------------------------------------------------------------------------
    # CONTROLS — edit values here directly, or use the UI below and click Save.
    # This dict lives at module level so pipeline scripts can import it directly:
    #
    # last_data_year: the last year of data that either a) is published, or
    # b) the pipeline aligns to. All data sources except the cer use the final
    # published source. The cer data the pipeline aligns to out to 2050 and
    # includes data like oil and gas prices and production.
    # Archive data is data used in the pipeline but not longer updated
    # -------------------------------------------------------------------------
    CONTROLS = {
        "currency_year_cims": 2025,
        "currency_year_jcims": 2005,
        "currency_year_cer_enduse": 2025,
        "currency_year_cer_macro": 2017,
        "currency_year_cer_benchmark": 2022,
        "currency_country_cer_benchmark": "USD",
        "cer_ef_reference_scenario": "Current Measures",
        "pipeline": {
            "data_start": 2000,
            "projection_end": 2100
        },
        "last_data_year": {
            "ceud": 2022,
            "nir": 2024,
            "cer": 2050,
            "ceedc": 2024,
            "stat_can_resd": 2024,
            "stat_can_population": 2026,
            "stat_can_ippi": 2026,
            "stat_can_rmpi": 2026,
            "stat_can_gross_output": 2022,
            "stat_can_gdp": 2025,
            "stat_can_coal": 2026,
            "stat_can_crude_new": 2026,
            "aer_st98": 2025,
            "afdc": 2025
        },
        "archived_data": {
            "stat_can_coal_archive": 2007,
            "stat_can_crude_old": 2016,
            "nir_crosswalk": 2020,
            "bc_coal": 2020,
            "jcims_prices": 2050,
            "nir_anchor_year": 2004,
            "nir_co_suppression_year": 2005
        }
    }


@app.cell(hide_code=True)
def _():
    save_button = mo.ui.run_button(label="💾 Save to control.py")
    c  = CONTROLS
    pl = c["pipeline"]
    ld = c["last_data_year"]
    ar = c["archived_data"]

    params_currency = mo.ui.dictionary({
        "currency_year_cims":             mo.ui.text(value=str(c["currency_year_cims"]),             label="Currency Year CIMS"),
        "currency_year_jcims":            mo.ui.text(value=str(c["currency_year_jcims"]),            label="Currency Year JCIMS"),
        "currency_year_cer_enduse":       mo.ui.text(value=str(c["currency_year_cer_enduse"]),       label="CER End Use Price Year"),
        "currency_year_cer_macro":        mo.ui.text(value=str(c["currency_year_cer_macro"]),        label="CER Macro Indicator Year"),
        "currency_year_cer_benchmark":    mo.ui.text(value=str(c["currency_year_cer_benchmark"]),    label="CER Benchmark Currency Year"),
        "currency_country_cer_benchmark": mo.ui.dropdown(options=["USD", "CAD"],                 value=c["currency_country_cer_benchmark"], label="CER Benchmark Currency Country"),
        "cer_ef_reference_scenario":      mo.ui.dropdown(options=["Current Measures", "Canada Net-zero"], value=c["cer_ef_reference_scenario"], label="CER EF Reference Scenario"),
    })
    params_pipeline = mo.ui.dictionary({
        "data_start":     mo.ui.text(value=str(pl["data_start"]),     label="Data Start Year"),
        "projection_end": mo.ui.text(value=str(pl["projection_end"]), label="Projection End Year"),
    })
    params_ldy = mo.ui.dictionary({
        "ceud":                  mo.ui.text(value=str(ld["ceud"]),                  label="CEUD (NRCan)"),
        "nir":                   mo.ui.text(value=str(ld["nir"]),                   label="NIR (ECCC)"),
        "cer":                   mo.ui.text(value=str(ld["cer"]),                   label="Canada's Energy Future (CER)"),
        "ceedc":                 mo.ui.text(value=str(ld["ceedc"]),                 label="CEEDC"),
        "stat_can_resd":         mo.ui.text(value=str(ld["stat_can_resd"]),         label="RESD — 25-10-0029-01"),
        "stat_can_population":   mo.ui.text(value=str(ld["stat_can_population"]),   label="Population — 17-10-0009-01"),
        "stat_can_ippi":         mo.ui.text(value=str(ld["stat_can_ippi"]),         label="IPPI — 18-10-0267-01"),
        "stat_can_rmpi":         mo.ui.text(value=str(ld["stat_can_rmpi"]),         label="RMPI — 18-10-0268-01"),
        "stat_can_gross_output": mo.ui.text(value=str(ld["stat_can_gross_output"]), label="Gross Output — 36-10-0488-01"),
        "stat_can_gdp":          mo.ui.text(value=str(ld["stat_can_gdp"]),          label="GDP Light Industrial — 36-10-0711-01"),
        "stat_can_coal":         mo.ui.text(value=str(ld["stat_can_coal"]),         label="Coal 2008–present — 25-10-0046-01"),
        "stat_can_crude_new":    mo.ui.text(value=str(ld["stat_can_crude_new"]),    label="Crude New — 25-10-0063-01"),
        "aer_st98":              mo.ui.text(value=str(ld["aer_st98"]),              label="AER ST98 (Alberta coal)"),
        "afdc":                  mo.ui.text(value=str(ld["afdc"]),                  label="AFDC Biofuel Prices"),
    })
    params_archived = mo.ui.dictionary({
        "stat_can_coal_archive":   mo.ui.text(value=str(ar["stat_can_coal_archive"]),   label="Coal pre-2008 — 25-10-0048-01"),
        "stat_can_crude_old":      mo.ui.text(value=str(ar["stat_can_crude_old"]),      label="Crude Old — 25-10-0014-01"),
        "nir_crosswalk":           mo.ui.text(value=str(ar["nir_crosswalk"]),           label="NIR Crosswalk (contact Brad)"),
        "bc_coal":                 mo.ui.text(value=str(ar["bc_coal"]),                 label="BC Coal (BC Ministry)"),
        "jcims_prices":            mo.ui.text(value=str(ar["jcims_prices"]),            label="JCIMS Prices (not updated)"),
        "nir_anchor_year":         mo.ui.text(value=str(ar["nir_anchor_year"]),         label="NIR Anchor Year"),
        "nir_co_suppression_year": mo.ui.text(value=str(ar["nir_co_suppression_year"]), label="NIR Co-Suppression Year"),
    })

    mo.vstack([
        mo.callout(
            mo.md("Adjust values below, then click **Save** to write them back to "
                  "`control.py`. The pipeline reads this file at runtime — no other edits needed."),
            kind="info",
        ),
        mo.md("## Parameters"),
        mo.hstack([
            mo.vstack([
                params_currency["currency_year_cims"],
                params_currency["currency_year_jcims"],
                params_currency["currency_year_cer_enduse"],
                params_currency["currency_year_cer_macro"],
                params_currency["currency_year_cer_benchmark"],
                params_currency["currency_country_cer_benchmark"],
            ]),
            mo.vstack([
                params_currency["cer_ef_reference_scenario"],
            ]),
        ], gap=4),
        mo.md("## Pipeline Settings"),
        mo.hstack([
            params_pipeline["data_start"],
            params_pipeline["projection_end"],
        ], gap=4),
        mo.md("## Last Data Year"),
        mo.hstack([
            mo.vstack([
                mo.md("**NRCan / ECCC / CER**"),
                params_ldy["ceud"],
                params_ldy["nir"],
                params_ldy["cer"],
                mo.md("**CEEDC**"),
                params_ldy["ceedc"],
                mo.md("**Provincial**"),
                params_ldy["aer_st98"],
                mo.md("**US**"),
                params_ldy["afdc"],
            ]),
            mo.vstack([
                mo.md("**Statistics Canada**"),
                params_ldy["stat_can_resd"],
                params_ldy["stat_can_population"],
                params_ldy["stat_can_ippi"],
                params_ldy["stat_can_rmpi"],
                params_ldy["stat_can_gross_output"],
                params_ldy["stat_can_gdp"],
                params_ldy["stat_can_coal"],
                params_ldy["stat_can_crude_new"],
            ]),
        ], gap=4),
        mo.accordion({
            "Archived Data — retired sources, click to expand": mo.hstack([
                mo.vstack([
                    mo.md("**Retired Sources**"),
                    params_archived["stat_can_coal_archive"],
                    params_archived["stat_can_crude_old"],
                    params_archived["nir_crosswalk"],
                    params_archived["bc_coal"],
                    params_archived["jcims_prices"],
                ]),
                mo.vstack([
                    mo.md("**NIR Suppression Solver**"),
                    params_archived["nir_anchor_year"],
                    params_archived["nir_co_suppression_year"],
                ]),
            ], gap=4),
        }),
        mo.hstack([save_button], justify="start"),
    ])
    return (
        params_archived,
        params_currency,
        params_ldy,
        params_pipeline,
        save_button,
    )


@app.cell(hide_code=True)
def _(params_archived, params_currency, params_ldy, params_pipeline):
    cv = params_currency.value
    pv = params_pipeline.value
    lv = params_ldy.value
    av = params_archived.value

    current = {
        "currency_year_cims":             int(cv["currency_year_cims"]),
        "currency_year_jcims":            int(cv["currency_year_jcims"]),
        "currency_year_cer_enduse":       int(cv["currency_year_cer_enduse"]),
        "currency_year_cer_macro":        int(cv["currency_year_cer_macro"]),
        "currency_year_cer_benchmark":    int(cv["currency_year_cer_benchmark"]),
        "currency_country_cer_benchmark": cv["currency_country_cer_benchmark"],
        "cer_ef_reference_scenario":      cv["cer_ef_reference_scenario"],
        "pipeline": {
            "data_start":     int(pv["data_start"]),
            "projection_end": int(pv["projection_end"]),
        },
        "last_data_year": {
            "ceud":                  int(lv["ceud"]),
            "nir":                   int(lv["nir"]),
            "cer":                   int(lv["cer"]),
            "ceedc":                 int(lv["ceedc"]),
            "stat_can_resd":         int(lv["stat_can_resd"]),
            "stat_can_population":   int(lv["stat_can_population"]),
            "stat_can_ippi":         int(lv["stat_can_ippi"]),
            "stat_can_rmpi":         int(lv["stat_can_rmpi"]),
            "stat_can_gross_output": int(lv["stat_can_gross_output"]),
            "stat_can_gdp":          int(lv["stat_can_gdp"]),
            "stat_can_coal":         int(lv["stat_can_coal"]),
            "stat_can_crude_new":    int(lv["stat_can_crude_new"]),
            "aer_st98":              int(lv["aer_st98"]),
            "afdc":                  int(lv["afdc"]),
        },
        "archived_data": {
            "stat_can_coal_archive":   int(av["stat_can_coal_archive"]),
            "stat_can_crude_old":      int(av["stat_can_crude_old"]),
            "nir_crosswalk":           int(av["nir_crosswalk"]),
            "bc_coal":                 int(av["bc_coal"]),
            "jcims_prices":            int(av["jcims_prices"]),
            "nir_anchor_year":         int(av["nir_anchor_year"]),
            "nir_co_suppression_year": int(av["nir_co_suppression_year"]),
        },
    }
    return (current,)


@app.cell(hide_code=True)
def _(current, save_button):
    mo.stop(not save_button.value)
    this_file = Path(__file__)
    source = this_file.read_text(encoding='utf-8')
    new_dict = '    CONTROLS = ' + json.dumps(current, indent=4).replace('\n', '\n    ')
    updated = re.sub(
        r'    CONTROLS = \{.*?\n    \}',
        new_dict,
        source,
        count=1,
        flags=re.DOTALL,
    )
    this_file.write_text(updated, encoding='utf-8')
    mo.callout(mo.md(f"✅ Saved to `{this_file}`"), kind="success")
    return


@app.cell(hide_code=True)
def _():
    energy_map_df  = pd.read_csv(MAPPINGS_PATH / 'energy_map.csv')
    region_map_df  = pd.read_csv(MAPPINGS_PATH / 'region_map.csv')
    sector_map_df  = pd.read_csv(MAPPINGS_PATH / 'sector_map.csv')
    conversions_df = pd.read_csv(
        MAPPINGS_PATH / 'energy_conversions.csv',
        header=None,
        names=['col_a', 'col_b', 'col_c', 'col_d'],
    )
    mo.accordion({
        "Mappings — energy, region, sector, conversion tables": mo.vstack([
            mo.md("## Energy Map"),         mo.ui.table(energy_map_df),
            mo.md("## Region Map"),         mo.ui.table(region_map_df),
            mo.md("## Sector Map"),         mo.ui.table(sector_map_df),
            mo.md("## Energy Conversions"), mo.ui.table(conversions_df.dropna(how='all')),
        ])
    })
    return


if __name__ == "__main__":
    app.run()
