import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import CIMS


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


@app.cell(hide_code=True)
def _():
    ### Base model and standard files below are required for the model to run
    model_path = 'data/model_inputs/model'
    # Model files should be located at: data/model_inputs/

    ### Base model to start initialisation
    base_model = 'CIMS_base'

    ### Default values and parameter list files
    default_path = 'data/model_inputs/defaults/defaults_Parameters.csv'
    list_path = 'data/model_inputs/defaults/defaults_Lists.csv'

    update_files = {}

    ### Required sector files (included in data/model_inputs/model/)
    sector_req = [
        'fuels',
        'transmission',
        'coal mining',
        'natural gas',
        'petroleum crude',
        'petroleum refining',
        'electricity',
        'biodiesel',
        'ethanol',
        'hydrogen',
        'mining',
        'industrial minerals',
        'iron and steel',
        'metal smelting',
        'chemical products',
        'pulp and paper',
        'light industrial',
        'construction',
        'residential',
        'commercial',
        'transportation personal',
        'transportation freight',
        'waste',
        'agriculture',
        'forestry',
        ]
    if sector_req:
        if model_path not in update_files:
            update_files[model_path] = []
        update_files[model_path].extend(sector_req)


    ### Required model files (included in data/model_inputs/model/)
    model_req = [
        'DCC', # declining capital cost
        'DIC', # declining intangible cost (neighbour effect)
        'FIC', # fixed intangible cost; primarily used for calibration
        'market share limits', # use of limits should be minimised
        ]
    if model_req:
        if model_path not in update_files:
            update_files[model_path] = []
        update_files[model_path].extend(model_req)
    return base_model, default_path, list_path, model_path, update_files


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Adjust scenario parameters and optional update files
    """)
    return


@app.cell
def _(model_path, update_files):
    # Only uncommented regions below will be run in the simulation
    region_list = [
        'CIMS', # Required
        'RoW', # Required
        'CAN', # Required
        'BC',
        # 'AB',
        # 'SK',
        # 'MB',
        # 'ON',
        # 'QC',
        # 'AT',
    ]

    # Only uncommented years below will be run in the simulation
    year_list = [
        ### Historical ###
        2000,
        2005,
        2010,
        2015,
        2020,
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
        'Coal Mining'
        # 'Natural Gas', # Must run all regions due to Natural Gas Market
        # 'Petroleum Crude', # Must also run 'Natural Gas' sector since shared fuel blending
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
        'Transportation Personal',
        'Transportation Freight',
        'Waste',
        'Agriculture',
        'Forestry',
    ]


    ### Optional model folder files (included in data/model_inputs/model/)
    model_optional = [
        'exogenous prices',  # needed for correct calibration of historical years
        # 'exogenous demand',  # use this file when calibrating endogenous supply sectors
        ]
    if model_optional:
        if model_path not in update_files:
            update_files[model_path] = []
        update_files[model_path].extend(model_optional)


    ### Reference scenario update files (these files should always be included as the base model specification, but they can be excluded if appropriate for your scenario)
    ref_path = 'data/model_inputs/policies/reference'
    ref_policies = [
        # 'coal mining',   # <-- add this
    ### Economy
        'Ref_carbon tax',
        'Ref_OBPS_Fed',
    ### Coal Mining
    ### Natural Gas Production
    ### Petroleum Crude
    ### Mining
    ### Electricity
        'Ref_coal phase out',
        'Ref_nuclear ban',
        'Ref_nuclear decommission',
        'Ref_clean electricity',
        'Ref_CER',
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
        'Ref_incandescent phase out',
    ### Commercial
    ### Transportation Personal
        'Ref_LDV ZEV_Federal', #include before Prov version
        'Ref_LDV ZEV_Prov',
        'Ref_renewable fuel content_Fed', #include before Prov version
        'Ref_renewable fuel content_Prov',
    ### Transportation Freight
    ### Waste
        'Ref_waste methane large sites',
    ### Agriculture
        ]
    if ref_policies:
        if ref_path not in update_files:
            update_files[ref_path] = []
        update_files[ref_path].extend(ref_policies)


    # These files turn off existing Reference policies starting in the first model year
    # Off files must be updated annually to reflect the last common year between a set of scenarios
    turn_off_path = 'data/model_inputs/policies/turn off'
    turn_off_policies = [
    ### Economy
        # 'Off_OBPS_Fed',
    ### Coal Mining
    ### Natural Gas Production
    ### Petroleum Crude
    ### Mining
    ### Electricity
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
    ### Commercial
    ### Transportation Personal
    ### Transportation Freight
    ### Waste
    ### Agriculture
        ]
    if turn_off_policies:
        if turn_off_path not in update_files:
            update_files[turn_off_path] = []
        update_files[turn_off_path].extend(turn_off_policies)


    ### Optional scenario update files
    scenario_path = 'data/model_inputs/policies/net zero'    # Change this path as necessary based on current scenario
    scenario_policies = [
    ### Economy
        # 'NZ_carbon tax',
    ### Coal Mining
    ### Natural Gas Production
        # 'NZ_natural gas',
    ### Petroleum Crude
        # 'NZ_petroleum crude',
    ### Mining
    ### Electricity
        # 'NZ_electricity',
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
    ### Commercial
    ### Transportation Personal
    ### Transportation Freight
    ### Waste
    ### Agriculture
        ]
    if scenario_policies:
        if scenario_path not in update_files:
            update_files[scenario_path] = []
        update_files[scenario_path].extend(scenario_policies)



    ### Scenario Name
    ### This will be the save location for results (i.e., results_dir/scenario_name/results_general.csv)
    scenario_name = 'Reference'  # Set this to current scenario (e.g., "Reference", "Net Zero")
    return region_list, scenario_name, sector_list, year_list


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
    list_path,
    model_path,
    region_list,
    sector_list,
    update_files,
    year_list,
):
    # Once currency_deflator.csv and currency_exchange.csv are available in
    # data/model_inputs/, uncomment target_units/deflator_path/exchange_path below
    # to normalize cost inputs to a single currency and dollar-year.
    model = CIMS.Model(
        model_path=model_path,
        base_model= base_model,
        region_list=region_list,
        update_files=update_files,
        year_list=year_list,
        sector_list=sector_list,
        default_values_csv_path=default_path,
        list_csv_path=list_path,
        # target_units={"currency": "CAD", "dollar_year": 2020},
        # deflator_path="data/model_inputs/currency_deflator.csv",
        # exchange_path="data/model_inputs/currency_exchange.csv",
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
    model.validator.warnings['unrequested_nodes']
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

    results_path = f'results/{scenario_name}'
    return (results_path,)


@app.cell
def _(model, results_path):
    #################### Output Results ###########################
    results_df = CIMS.log_model(
       model=model, 
       output_file = f"{results_path}/results_general.csv",
       parameter_file="results/results_general.txt",
       ensure_dir=True   
    )
    return


@app.cell
def _(model, results_path):
    #################### Export Tech Results ####################
    tech_results_df = CIMS.log_model(
       model=model, 
       output_file=f'{results_path}/results_tech.csv',
       parameter_file = "results/results_tech.txt",
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
