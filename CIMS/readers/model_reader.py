import numpy as np
import pandas as pd
import polars as pl

from ..utils.model_description import column_list as COL
from ..utils.model_description.query import get_node_cols
from ..utils.parameter import list as PARAM
from ..utils.parameter.parse import infer_type, is_year


class ModelReader:
    def __init__(self, csv_file_paths, col_list, year_list, sector_list,
                 default_values_csv_path=None, node_col=COL.branch, root_node="CIMS", list_csv_path=None):

        if default_values_csv_path:
            self.default_values_csv = default_values_csv_path
        if list_csv_path:
            self.list_csv = list_csv_path

        self.csv_files = csv_file_paths
        self.node_col = node_col
        self.col_list = col_list
        self.year_list = [str(x) for x in year_list]
        self.sector_list = sector_list

        self.model_df = self._get_model_df()
        self.root = root_node

        self.node_dfs = {}
        self.tech_dfs = {}

    def _get_model_df(self):
        appended_data = []
        for csv_file in self.csv_files:
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

        model_df = pd.concat(appended_data, ignore_index=True)  # Add province sheets together and re-index
        model_df.index += 3  # Adjust index to correspond to Excel line numbers
        # (+1: 0 vs 1 origin, +1: header skip, +1: column headers)
        model_df.columns = [str(c) for c in
                            model_df.columns]  # Convert all column names to strings (years were ints)
        n_cols, y_cols = get_node_cols(model_df, self.node_col)  # Find columns, separated year cols from non-year cols
        n_cols = [n_col for n_col in n_cols if n_col in self.col_list]
        y_cols = [y_col for y_col in y_cols if y_col in self.year_list]
        all_cols = n_cols + y_cols

        mdf = model_df.loc[1:, all_cols]  # Create df, drop irrelevant columns & skip first, empty row

        return mdf

    def get_model_description(self, inplace=False):
        # ------------------------
        # Filter sectors for calibration (if applicable)
        # ------------------------
        if self.sector_list:
            if None not in self.sector_list:
                self.sector_list.append(None)
            self.model_df = self.model_df.apply(lambda row: row[self.model_df[COL.sector].isin(self.sector_list)])

        # ------------------------
        # Extract Node DFs
        # ------------------------
        self.model_df[COL.parameter] = self.model_df[COL.parameter].str.lower()
        node_dfs = {n: gb for n, gb in self.model_df.groupby(by=COL.branch)}

        # ------------------------
        # Extract Tech DFs
        # ------------------------
        # Extract tech dfs from node dfs and rewrite node df without techs
        tech_dfs = {}
        for node_name, node_df in node_dfs.items():
            if not all(node_df[COL.technology].isnull()):
                tech_dfs[node_name] = {t: gb for t, gb in node_df.groupby(by=COL.technology)}
                node_dfs[node_name] = node_df[node_df[COL.technology].isnull()]#.drop(columns=COL.technology)

                # Remove region and sector columns from tech dfs
                for t in tech_dfs[node_name]:
                    tech_dfs[node_name][t] = tech_dfs[node_name][t].drop(columns=[COL.region])
                    tech_dfs[node_name][t] = tech_dfs[node_name][t].drop(columns=[COL.sector])

        if inplace:
            self.node_dfs = node_dfs
            self.tech_dfs = tech_dfs

        return node_dfs, tech_dfs

    def get_years(self):
        cols = [y for y in self.model_df.columns if is_year(y)]
        return cols

    def get_default_params(self):
        # Read model_description from excel

        df = pl.read_csv(
            self.default_values_csv,
            use_pyarrow=False,
            infer_schema_length=0,
            ).with_columns(pl.all().replace(
                    {np.nan: None}
                )).to_pandas()
        # Remove empty rows
        df = df.dropna(axis=0, how="all")

        # Convert parameter strings to lower case
        df[COL.parameter] = df[COL.parameter].str.lower()

        # Default Parameters
        df_has_defaults = df[~df[COL.default_value].isna()]
        node_tech_defaults = {}
        for param, val in zip(df_has_defaults[COL.parameter],
                              df_has_defaults[COL.default_value]):
            if val.lower() == 'none':
                val = None
            node_tech_defaults[param] = infer_type(val)

        # Return
        return node_tech_defaults
        
    def get_inheritable_params(self):
        return self._get_list(column_identifier="Inheritable")
    
    def get_valid_competition_types(self):
        return self._get_list(column_identifier="Competition")

    def _get_list(self, column_identifier):
        # Read List File from CSV
        df = pl.read_csv(
            self.list_csv, 
            use_pyarrow=False,
            infer_schema_length=0).to_pandas()
        
        # Remove empty rows
        df.dropna(axis=0, how='all')

        # Extract inheritable parameters
        list_clean = df[column_identifier].str.lower()

        return list_clean

    def get_output_params(self):
        pass