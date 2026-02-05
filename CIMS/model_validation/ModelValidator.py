import pandas as pd
import polars as pl
import numpy as np
import textwrap
from typing import List
import time 

from .validation_utils import get_providers, get_requested
from ..utils.model_description import column_list as COL
from ..utils.model_description.query import get_node_cols

from .registry import REGISTRY, resolve_kwargs, Phase, Severity


class ModelValidator:
    def __init__(self, csv_file_paths, col_list, year_list, sector_list,
                 csv_update_file_paths=None, default_values_csv_path=None, node_col=COL.branch,
                 target_col=COL.target, root_node="CIMS", list_csv_path=None):

        self.csv_files = csv_file_paths
        self.scenario_files = csv_update_file_paths or []

        self.default_param_df = self.get_default_df(default_values_csv_path)
        self.competition_types = self._get_list(list_csv_path, column_identifier="Competition")

        self.node_col = node_col
        self.target_col = target_col
        self.col_list = col_list
        self.year_list = [str(x) for x in year_list]
        self.sector_list = sector_list

        self.model_df = self._get_model_df()
        self.root_node = root_node

        self.warnings = {}

        self.index2branch_map = self._create_index_to_branch_map()
        self.branch2node_index_map = self._create_branch_to_node_index_map()

        # Track validation runs and whether errors were detected.
        self.file_validation_ran = False
        self.file_validation_errors = False
        self.file_validation_warnings = False
        self.graph_validation_ran = False
        self.graph_validation_errors = False
        self.graph_validation_warnings = False

    def _get_model_df(self, read_base_file=True, read_scenario_files=True):
        files_to_read = []
        if read_base_file:
            for file in self.csv_files:
                files_to_read.append(file)
        if read_scenario_files:
            for file in self.scenario_files:
                files_to_read.append(file)

        appended_data = []
        for csv_file in files_to_read:
            try:
                mixed_type_columns = [COL.context]
                sheet_df = pl.read_csv(
                    csv_file,
                    skip_rows=1,
                    use_pyarrow=False,
                    infer_schema_length=0,
                    ).with_columns(pl.all().replace(
                        {np.nan: None}
                        )).to_pandas()
                appended_data.append(sheet_df)

            except ValueError:
                print(f"Warning: Unable to parse csv_path at {csv_file}. Skipping.")

        model_df = pd.concat(appended_data,
                             ignore_index=True)  # Add province sheets together and re-index
        
        # Filter sectors (if applicable)
        if self.sector_list:
            if None not in self.sector_list:
                self.sector_list.append(None)
            model_df = model_df[model_df[COL.sector].isin(self.sector_list)]

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
        mdf[COL.parameter] = mdf[COL.parameter].str.lower()

        return mdf

    def get_default_df(self, default_values_csv_path):
        if default_values_csv_path is None:
            return pd.DataFrame()

        # Read model_description from excel
        mixed_type_columns = [COL.default_value]
        df = pl.read_csv(
            default_values_csv_path,
            use_pyarrow=False,
            infer_schema_length=0,
            ).with_columns(pl.all().replace(
                {np.nan: None}
            )).to_pandas()

        # Remove empty rows
        df = df.dropna(axis=0, how="all")

        # Convert parameter strings to lower case
        df[COL.parameter] = df[COL.parameter].str.lower()

        # Return
        return df

    def _get_list(self, list_csv_path, column_identifier):
        if list_csv_path is None:
            return []

        # Read List File from CSV
        df = pl.read_csv(
            list_csv_path, 
            use_pyarrow=False,
            infer_schema_length=0).to_pandas()
        
        # Remove empty rows
        df = df.dropna(axis=0, how='all')

        # Extract inheritable parameters
        list_clean = df[column_identifier].str.lower()

        return list_clean    
    
    def _create_branch_to_node_index_map(self):
        branch_index = {b: i for i, b in self.model_df[self.node_col].drop_duplicates(keep='first').items()}
        index_to_node_index_map = {self.index2branch_map[i]: branch_index[self.index2branch_map[i]] for i in self.model_df.index}
        return index_to_node_index_map

    def _create_index_to_branch_map(self):
        return {i: self.model_df[COL.branch].loc[i] for i in self.model_df.index}

    def _raise_concerns(self, concerns: List[object], concern_key: str, concern_desc: str) -> bool:
        """
        Format and print a single validation result.

        Prints the number of issues found, a short description, and (if applicable)
        a pointer to `ModelValidator.warnings[...]`. Messages are wrapped with a
        hanging indent for readability. Nothing unless issues are detected. 
        Returns True when concerns are present.
        """
        if len(concerns) <= 0:
            more_info = ""
        else:
            more_info = f"See ModelValidator.warnings['{concern_key}'] for more info."

        info_str = f"{len(concerns):5} {concern_desc}. {more_info}"

        if len(concerns) > 0:
            wrapped_print = textwrap.fill(
                info_str, 
                width=100, 
                initial_indent="",
                subsequent_indent=" " * (5 + 1)
            )
            print(wrapped_print)
            return True
        return False

    def _run_check(self, check_function, **kwargs) -> bool:
        # Collect list
        concern_list, concern_desc = check_function(**kwargs)

        # Raise Concerns
        has_concerns = self._raise_concerns(
            concern_list, check_function.__name__, concern_desc
        )

        # Return list
        self.warnings[check_function.__name__] = concern_list
        return has_concerns

    def validate(self):
        """
        Backwards-compatible alias for validate_files().
        """
        return self.validate_files()

    def validate_files(self):
        """
        Run file validation checks using the central registry.
        """
        start = time.time()
        
        print("\n=== Validating model files ===")
        
        # Per-run context values that may be reused by multiple checks
        providers = get_providers(self.model_df, self.node_col)
        requested = get_requested(self.model_df, self.target_col)
        context = {
            'providers': providers, 
            'requested': requested
        }

        # ---- Errors ----
        print("\n-- Errors --")
        errors_found = False
        for name, spec in REGISTRY.iter(phase=Phase.FILE, severity=Severity.ERROR):
            kwargs = resolve_kwargs(self, spec.argmap, context)
            errors_found |= self._run_check(spec.fn, **kwargs)
        self.file_validation_errors = errors_found
        if not errors_found:
            print("No errors found!")
        
        # ---- Warnings ----
        print("\n-- Warnings --")
        warnings_found = False
        for name, spec in REGISTRY.iter(phase=Phase.FILE, severity=Severity.WARNING):
            kwargs = resolve_kwargs(self, spec.argmap, context)
            warnings_found |= self._run_check(spec.fn, **kwargs)
        self.file_validation_warnings = warnings_found
        if not warnings_found:
            print("No warnings found!")
        
        timing = f" (completed in {time.time() - start:.2f}s)"
        print(f"\n=== File validation complete{timing} ===")

        self.file_validation_ran = True
   
    def validate_graph(self):
        """
        Run graph-phase validation checks using the central registry.
        """
        start = time.time()
        
        print("\n=== Running graph-phase validation ===")

        context = {}
        # ---- Errors ----
        print("\n-- Errors --")
        errors_found = False
        for name, spec in REGISTRY.iter(phase=Phase.GRAPH, severity=Severity.ERROR):
            kwargs = resolve_kwargs(self, spec.argmap, context)
            errors_found |= self._run_check(spec.fn, **kwargs)
        self.graph_validation_errors = errors_found
        if not errors_found:
            print("No errors found!")
        
        # ---- Warnings ----
        print("\n-- Warnings --")
        warnings_found = False
        for name, spec in REGISTRY.iter(phase=Phase.GRAPH, severity=Severity.WARNING):
            kwargs = resolve_kwargs(self, spec.argmap, context)
            warnings_found |= self._run_check(spec.fn, **kwargs)
        self.graph_validation_warnings = warnings_found
        if not warnings_found:
            print("No warnings found!")    
        
        timing = f" (completed in {time.time() - start:.2f}s)"
        print(f"\n=== Completed graph-phase validation{timing} ===")

        self.graph_validation_ran = True
