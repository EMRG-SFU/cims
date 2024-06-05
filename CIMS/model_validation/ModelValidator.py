import pandas as pd
import numpy as np
from ..reader import get_node_cols, _bool_as_string
import warnings
import os
from . import validation_checks as validate
from .validation_utils import get_providers, get_requested
from pathlib import Path


class ModelValidator:
    def __init__(self, infile, sheet_map, col_list, year_list, sector_list,
                 scenario_files=None, default_values=None, node_col="Branch",
                 target_col="Target", root_node="CIMS"):
        self.infile = infile
        self.scenario_files = scenario_files or []
        excel_engine_map = {'.xlsb': 'pyxlsb',
                            '.xlsm': 'xlrd'}
        self.excel_engine = excel_engine_map[Path(self.infile).suffix]

        self.default_param_df = self.get_default_df(default_values)

        self.sheet_map = sheet_map
        self.node_col = node_col
        self.target_col = target_col
        self.col_list = col_list
        self.year_list = [str(x) for x in year_list]
        self.sector_list = sector_list

        self.model_df = self._get_model_df()
        self.root = root_node

        self.warnings = {}
        self.verbose = False
        self.raise_warnings = False

        self.index2branch_map = self._create_index_to_branch_map()
        self.branch2node_index_map = self._create_branch_to_node_index_map()

    def _get_model_df(self, read_base_file=True, read_scenario_files=True):
        files_to_read = []
        if read_base_file:
            files_to_read.append(self.infile)
        if read_scenario_files:
            files_to_read += self.scenario_files

        # Read in list of sheets from 'Lists' sheet in model description
        appended_data = []
        for file in files_to_read:
            for sheet in self.sheet_map:
                try:
                    mixed_type_columns = ['Context']
                    sheet_df = pd.read_excel(
                        file,
                        sheet_name=sheet,
                        header=1,
                        converters={c:_bool_as_string for c in mixed_type_columns},
                        engine=self.excel_engine).replace(
                            {np.nan: None, 'False': False, 'True': True})
                    appended_data.append(sheet_df)

                except ValueError:
                    print(f"Warning: {sheet} not included in {file}. Sheet was not imported into model.")

        model_df = pd.concat(appended_data,
                             ignore_index=True)  # Add province sheets together and re-index
        model_df.index += 3  # Adjust index to correspond to Excel line numbers
        # (+1: 0 vs 1 origin, +1: header skip, +1: column headers)
        model_df.columns = [str(c) for c in
                            model_df.columns]  # Convert all column names to strings (years were ints)
        n_cols, y_cols = get_node_cols(model_df,
                                       self.node_col)  # Find columns, separated year cols from non-year cols
        n_cols = [n_col for n_col in n_cols if n_col in self.col_list]
        y_cols = [y_col for y_col in y_cols if y_col in self.year_list]
        all_cols = n_cols + y_cols

        mdf = model_df.loc[1:,
              all_cols]  # Create df, drop irrelevant columns & skip first, empty row
        mdf['Parameter'] = mdf['Parameter'].str.lower()

        return mdf

    def get_default_df(self, default_values):
        if default_values is None:
            return pd.DataFrame()

        # Read model_description from excel
        mixed_type_columns = ['Default value']
        df = pd.read_excel(
            default_values,
            header=0,
            converters={c: _bool_as_string for c in mixed_type_columns},
            engine=self.excel_engine).replace(
                {np.nan: None, 'False': False, 'True': True})

        # Remove empty rows
        df = df.dropna(axis=0, how="all")

        # Convert parameter strings to lower case
        df['Parameter'] = df['Parameter'].str.lower()

        # Return
        return df

    def _create_branch_to_node_index_map(self):
        branch_index = {b: i for i, b in self.model_df[self.node_col].drop_duplicates(keep='first').items()}
        index_to_node_index_map = {self.index2branch_map[i]: branch_index[self.index2branch_map[i]] for i in self.model_df.index}
        return index_to_node_index_map

    def _create_index_to_branch_map(self):
        return {i: self.model_df['Branch'].loc[i] for i in self.model_df.index}

    def _raise_concerns(self, concerns, concern_key, concern_desc):
        if len(concerns) <= 0:
            more_info = ""
        else:
            more_info = f"See ModelValidator.warnings['{concern_key}'] for more info."

        info_str = f"{len(concerns)} {concern_desc}. {more_info}"

        if self.verbose:
            print(info_str)
        if self.raise_warnings:
            warnings.warn(info_str)

    def _run_check(self, check_function, **kwargs):
        # Collect list
        concern_list, concern_key, concern_desc = check_function(**kwargs)

        # Raise Concerns
        self._raise_concerns(concern_list, concern_key, concern_desc)

        # Return list
        self.warnings[concern_key] = concern_list

    def validate(self, verbose=True, raise_warnings=False):
        self.verbose = verbose
        self.raise_warnings = raise_warnings

        providers = get_providers(self.model_df, self.node_col)
        requested = get_requested(self.model_df, self.target_col)

        self._run_check(validate.mismatched_node_names, validator=self, providers=providers)
        self._run_check(validate.unspecified_nodes, providers=providers, requested=requested)
        self._run_check(validate.unrequested_nodes, providers=providers, requested=requested, root_node=self.root)
        self._run_check(validate.nodes_no_provided_service, validator=self)
        self._run_check(validate.invalid_competition_type, df=self.model_df)
        self._run_check(validate.nodes_requesting_self, validator=self)
        self._run_check(validate.nodes_no_requested_service, validator=self)
        self._run_check(validate.nodes_with_zero_output, validator=self)
        self._run_check(validate.supply_nodes_no_lcc_or_price, validator=self)
        self._run_check(validate.techs_no_base_market_share, validator=self)
        self._run_check(validate.duplicate_service_requests, validator=self)
        self._run_check(validate.bad_service_req, validator=self)
        self._run_check(validate.tech_compete_nodes_no_techs, validator=self)
        self._run_check(validate.market_child_requested, validator=self)
        self._run_check(validate.revenue_recycling_at_techs, validator=self)
        self._run_check(validate.both_cop_p2000_defined, validator=self)
        self._run_check(validate.inconsistent_service_req_context, validator=self)
        self._run_check(validate.inconsistent_tech_refs, validator=self)
        self._run_check(validate.service_req_at_tech_node, validator=self)
        self._run_check(validate.missing_parameter_default, validator=self)
        self._run_check(validate.min_max_conflicts, validator=self)
        self._run_check(validate.new_nodes_in_scenario, validator=self)
        self._run_check(validate.new_techs_in_scenario, validator=self)
        self._run_check(validate.zero_requested_nodes, validator=self, providers=providers, root_node=self.root)
