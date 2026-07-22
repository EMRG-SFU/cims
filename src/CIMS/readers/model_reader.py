import numpy as np
import polars as pl

from ..utils.model_description import column_list as COL
from ..utils.parameter.parse import infer_type


class ModelReader:
    def __init__(self, model_df,
                 default_values_csv_path=None, node_col=COL.branch, root_node="CIMS", list_csv_path=None):

        if default_values_csv_path:
            self.default_values_csv = default_values_csv_path
        if list_csv_path:
            self.list_csv = list_csv_path

        self.node_col = node_col

        self.model_df = model_df.copy()
        self.root = root_node

        self.node_dfs = {}
        self.tech_dfs = {}

    def get_model_description(self, inplace=False):
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
        return sorted(self.model_df["Year"].dropna().unique().tolist())

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
        list_clean = df[column_identifier].str.lower().tolist()

        return list_clean

    def get_output_params(self):
        pass