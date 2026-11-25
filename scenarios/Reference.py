import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import CIMS
    from CIMS.data_processing.common import paths

    # The data root follows the imported code, not the current folder (common/paths.py).
    paths.announce()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Run all cells below to load, validate, and run the CIMS model
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Set required file paths, models, and sectors
    """)
    return


@app.function
def reset_model_files():
    ### Base model and standard files below are required for the model to run
    model_path = str(paths.MODEL)
    # Model files should be located at: data/model_inputs/

    ### Base model to start initialisation
    base_model = 'CIMS_base'

    ### Default values and parameter list files
    default_path = str(paths.DEFAULTS / 'default-parameter.csv')
    list_path = str(paths.DEFAULTS / 'default-list.csv')
    deflator_path = str(paths.PROCESSED / 'currency/currency-deflator.csv')
    exchange_path = str(paths.PROCESSED / 'currency/currency-exchange.csv')

    update_files = {}

    ### Required sector files (included in data/model_inputs/model/)
    sector_req = [
        'fuels',
        'transmission',
        'coal_mining',
        'natural_gas',
        'natural_gas_market', # must come after natural_gas: it overrides the regional production path
        'petroleum_crude',
        'petroleum_refining',
        'electricity',
        'biodiesel',
        'ethanol',
        'hydrogen',
        'mining',
        'industrial_minerals',
        'iron_and_steel',
        'metal_smelting',
        'chemical_products',
        'pulp_and_paper',
        'light_industrial',
        'construction',
        'residential',
        'commercial',
        'transportation_passenger',
        'transportation_freight',
        'waste',
        'agriculture',
        'forestry',
        ]

    ### Required model files (included in data/model_inputs/model/)
    model_req = [
        'dcc', # declining capital cost
        # 'dic', # declining intangible cost (neighbour effect)
        'fic', # fixed intangible cost; primarily used for calibration
        'market_share_limits', # use of limits should be minimised
        ]

    # Each entry below is (folder, [file/subfolder names]); merging just means
    # appending the list onto update_files[folder], creating it if needed.
    for _path, _files in [
        (model_path, sector_req),
        (model_path, model_req),
    ]:
        if _files:
            update_files.setdefault(_path, []).extend(_files)
    return (
        base_model,
        default_path,
        list_path,
        deflator_path,
        exchange_path,
        model_path,
        sector_req,
        update_files,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Adjust scenario parameters and optional update files
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Region, year, and sector selection
    """)
    return


@app.cell
def _():
    # Make sure list of model files is reset even if notebook kernel is not restarted
    (
        base_model,
        default_path,
        list_path,
        deflator_path,
        exchange_path,
        model_path,
        sector_req,
        update_files,
    ) = reset_model_files()

    # Target dollar year and currency for scenario
    target_units={
        "currency": "CAD",
        "dollar_year": 2020
    }

    # Only uncommented regions below will be run in the simulation
    region_list = [
        'CIMS', # Required
        'RoW', # Required
        'CAN', # Required
        'BC',
        'AB',
        'SK',
        'MB',
        'ON',
        'QC',
        'NB',
        'NL',
        'NS',
        'PE',
        'NT',
        'NU',
        'YT',
    ]

    # Only uncommented years below will be run in the simulation
    year_list = [
        ### Historical ###
        2000,
        2001,
        2002,
        2003,
        2004,
        2005,
        2006,
        2007,
        2008,
        2009,
        2010,
        2011,
        2012,
        2013,
        2014,
        2015,
        2016,
        2017,
        2018,
        2019,
        2020,
        2021,
        2022,
        2023,
        2024,
        ### Forecast ###
        # 2025,
        # 2030,
        # 2035,
        # 2040,
        # 2045,
        # 2050,
    ]

    # Only uncommented sectors below will be run in the simulation
    # Uncomment individual sectors to calibrate one at a time
    # Must use Exogenous prices (and optional exogenous demand) file below when calibrating
    sector_list = [
        'Coal Mining',
        'Natural Gas',
        # 'Natural Gas Market', # Routes each region's production through the national market; needs 'Natural Gas' and all regions.
                                # Off: regions supply themselves and any subset of regions can run. Prices come from Generic Fuels either way.
                                # Design note: docs/design/natural-gas-market.md.
        'Petroleum Crude', # Must also run 'Natural Gas' sector since shared fuel blending
        'Petroleum Refining',
        'Electricity',
        'Biodiesel',
        'Ethanol',
        'Hydrogen',
        'Mining',
        'Industrial Minerals',
        'Iron and Steel',
        'Metal Smelting',
        'Chemical Products',
        'Pulp and Paper',
        'Light Industrial',
        'Construction',
        'Residential',
        'Commercial',
        'Transportation Passenger',
        'Transportation Freight',
        'Waste',
        'Agriculture',
        'Forestry',
    ]
    return (
        base_model,
        default_path,
        deflator_path,
        exchange_path,
        list_path,
        model_path,
        region_list,
        sector_list,
        sector_req,
        target_units,
        update_files,
        year_list,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Update file specifications
    """)
    return


@app.cell
def _(calibration_data, calibration_path, model_path, update_files):
    ### Optional model folder files (included in data/model_inputs/model/)
    model_optional = [
        'exogenous_prices',  # needed for correct calibration of historical years
        'exogenous_demand',  # use this file when calibrating endogenous supply sectors
        ]


    ### Policy files: one folder per policy under data/model_inputs/policies/
    # (<policy>/<policy>_<region>.csv). Entries are applied in list order, so the
    # reference policies come first, then turn-off files, then scenario files.
    policy_path = str(paths.POLICIES)

    ### Reference scenario policies (always included as the base model
    # specification, but they can be excluded if appropriate for your scenario)
    ref_policies = [
    ### Economy
        'carbon-tax_ref',        # federal fuel charge, BC tax (both 0 from 2025), QC cap-and-trade
        'obps_federal',          # federal OBPS, May 2026 headline trajectory
    ### Coal Mining
    ### Natural Gas Production
    ### Petroleum Crude
    ### Mining
    ### Electricity
        'coal-phase-out',
        'nuclear-ban',
        'nuclear-decommission',
        'clean-electricity',
        'cer',
    ### Biodiesel
    ### Ethanol
    ### Hydrogen
    ### Petroleum Refining
    ### Industrial Minerals
    ### Iron and Steel
    ### Metal Smelting
    ### Chemical Products
    ### Pulp and Paper
    ### Light Industrial
    ### Residential
        'incandescent-phase-out',
    ### Commercial
    ### Transportation Personal
        'ldv-zev_federal',                     # include before provincial version
        'ldv-zev_provincial',
        'renewable-fuel-content_federal',      # include before provincial version
        'renewable-fuel-content_provincial',
        # 'cfr',                               # Clean Fuel Regulations
    ### Transportation Freight
    ### Waste
        'waste-methane-large-sites',
    ### Agriculture
        ]


    # These files turn off existing Reference policies starting in the first model year
    # Off files must be updated annually to reflect the last common year between a set of scenarios
    turn_off_policies = [
        ### Economy
        # 'carbon-tax_off',
        # 'obps_federal_off',
        ]


    ### Optional scenario policies
    scenario_policies = [
        ### Economy
        # 'carbon-tax_net-zero',   # 2026 on, 2025 CAD rising 10%/yr
        ]

    ### Calibration Counterfactual Files
    # calibration_path = str(paths.CALIBRATION_TARGETS)
    # calibration_data = [
    #     'agriculture',
    #     'biodiesel',
    #     'chemical_products',
    #     'coal_mining',
    #     'commercial',
    #     'construction',
    #     'electricity',
    #     'ethanol',
    #     'forestry',
    #     'fuels',
    #     'hydrogen',
    #     'industrial_minerals',
    #     'iron_and_steel',
    #     'light_industrial',
    #     'metal_smelting',
    #     'mining',
    #     'natural_gas',
    #     'petroleum_crude',
    #     'petroleum_refining',
    #     'pulp_and_paper',
    #     'residential',
    #     'transportation_freight',
    #     'transportation_passenger',
    #     'waste'
    # ]

    ### Calibration Outputs (fitted FICs / lifetimes from a prior calibration pass)
    # One subfolder per sector, each with its own fitted_fics/ and fitted_lifetimes/
    # (data/model_inputs/calibration_outputs/<sector>/fitted_fics, ...), so several
    # sectors' calibration outputs can be loaded together without overwriting each other.
    calibration_outputs_path = str(paths.CALIBRATION_OUTPUTS)
    calibration_output_sectors = [
        'transportation_passenger',
        'transportation_freight',
        'residential',
        'commercial',
        'fuels',
        'pulp_and_paper',
        'light_industrial',
        'chemical_products',
        'metal_smelting',
        'iron_and_steel',
    ]
    calibration_outputs = [
        f'{_sector}/{_kind}'
        for _sector in calibration_output_sectors
        for _kind in ('fitted_fics', 'fitted_lifetimes')
    ]

    # Each entry below is (folder, [file/subfolder names]); merging just means
    # appending the list onto update_files[folder], creating it if needed.
    for _path, _files in [
        (model_path, model_optional),
        (policy_path, ref_policies + turn_off_policies + scenario_policies),
        (calibration_path, calibration_data),
        (calibration_outputs_path, calibration_outputs),
    ]:
        if _files:
            update_files.setdefault(_path, []).extend(_files)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Scenario name
    """)
    return


@app.cell
def _():
    ### Scenario Name
    ### This will be the save location for results (i.e., results_dir/scenario_name/results_general.csv)
    scenario_name = 'reference'  # Set this to current scenario (e.g., "Reference", "Net Zero")
    return (scenario_name,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Instantiate the Model
    """)
    return


@app.cell
def _(
    base_model,
    default_path,
    deflator_path,
    exchange_path,
    list_path,
    model_path,
    region_list,
    sector_list,
    sector_req,
    target_units,
    update_files,
    year_list,
):
    model = CIMS.Model(
        model_path=model_path,
        base_model= base_model,
        region_list=region_list,
        update_files=update_files,
        year_list=year_list,
        sector_list=sector_list,    
        default_values_csv_path=default_path,
        list_csv_path=list_path,
        sector_folders=sector_req,
        verbose=False,
        target_units=target_units,
        deflator_path=deflator_path,
        exchange_path=exchange_path,
    )
    return (model,)


@app.cell
def _(model):
    #################### Validate Files ####################
    model.validate_files()
    return


@app.cell
def _(model):
    #################### Show validator warnings ####################
    # change warning type as needed to view indicated nodes/techs
    model.validator.warnings['inconsistent_tech_refs']
    return


@app.cell
def _(model):
    #################### Construct graph ####################
    model.construct_graph()
    return


@app.cell
def _(model):
    #################### Validate Graph ####################
    model.validate_graph()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Run scenario and output results
    """)
    return


@app.cell
def _(model, scenario_name):
    #################### Run scenario and output results ###########################
    model.run(equilibrium_threshold=0.05, max_iterations=10, show_warnings=False, print_eq=True)

    results_path = str(paths.RESULTS / scenario_name)
    return (results_path,)


@app.cell
def _(model, results_path):
    #################### Output Results ###########################
    results_df = CIMS.log_model(
       model=model,
       output_file = f"{results_path}/results_general.csv",
       parameter_file=str(paths.CONFIG / "results-general.txt"),
       ensure_dir=True
    )
    return


@app.cell
def _(model, results_path):
    #################### Export Tech Results ####################
    tech_results_df = CIMS.log_model(
       model=model,
       output_file=f'{results_path}/results_tech.csv',
       parameter_file=str(paths.CONFIG / "results-tech.txt"),
       ensure_dir=True
    )
    return


@app.cell
def _(model, results_path):
    import pickle
    import gzip

    with gzip.open(f"{results_path}/model.pkl", "wb") as f:
        pickle.dump(model, f)
    return


@app.cell
def _():
    # import pickle
    # import gzip

    # with gzip.open(f"{results_path}/model.pkl", "rb") as f:
    #     model = pickle.load(f)
    return


@app.cell
def _(model):
    #################### Print service request loops after running model ####################
    import pprint
    pprint.pp(model.loops)
    return


if __name__ == "__main__":
    app.run()
