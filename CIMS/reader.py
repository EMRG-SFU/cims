import numpy as np
import pandas as pd
import polars as pl
import re
from pathlib import Path

from . import utils

# ********************************
# Helper Functions
# ********************************
# DataFrame helpers
# def non_empty_rows(df, exclude_column="Branch"):
#     """Return bool array to flag rows as False that have only None or False values, ignoring exclude_column"""
#     return df.loc[:, df.columns != exclude_column].T.apply(any)


# column extraction helpers
def is_year(cn):
    re_year = re.compile(r'^[0-9]{4}$')
    """Check if input int or str is 4 digits [0-9] between begin ^ and end $ of string"""
    # unit test: assert is_year, 1900
    return bool(re_year.match(str(cn)))


def find_first(items, pred=bool, default=None):
    """Find first item for that pred is True"""
    return next(filter(pred, items), default)


def find_first_index(items, pred=bool):
    """Find index of first item for that pred is True"""
    return find_first(enumerate(items), lambda kcn: pred(kcn[1]))[0]


def get_node_cols(mdf, first_data_col_name="Branch"):
    """Returns list of column names after 'Branch' and a list of years that follow """
    node_col_idx = find_first_index(mdf.columns,
                                    lambda cn: first_data_col_name.lower() in cn.lower())

    relevant_columns = mdf.columns[node_col_idx:]

    year_or_not = list(map(is_year, relevant_columns))
    first_year_idx = find_first_index(year_or_not)
    last_year_idx = find_first_index(year_or_not[first_year_idx:],
                                     lambda b: not b) + first_year_idx
    # list(...)[a:][:b] extracts b elements starting at a
    year_cols = mdf.columns[node_col_idx:][first_year_idx:last_year_idx]
    node_cols = mdf.columns[node_col_idx:][:first_year_idx]
    return node_cols, year_cols


class ModelReader:
    def __init__(self, infile, csv_file_paths, col_list, year_list, sector_list,
                 default_values_csv_path=None, node_col="Branch", root_node="CIMS"):
        self.infile = infile

        if default_values_csv_path:
            self.default_values_csv = default_values_csv_path
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
                mixed_type_columns = ['Context']

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
                print(f"Warning: {sheet} not included in {self.infile}. Sheet was not imported into model.")

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
            self.model_df = self.model_df.apply(lambda row: row[self.model_df['Sector'].isin(self.sector_list)])

        # ------------------------
        # Extract Node DFs
        # ------------------------
        self.model_df['Parameter'] = self.model_df['Parameter'].str.lower()
        node_dfs = {n: gb for n, gb in self.model_df.groupby(by='Branch')}

        # ------------------------
        # Extract Tech DFs
        # ------------------------
        # Extract tech dfs from node dfs and rewrite node df without techs
        tech_dfs = {}
        for node_name, node_df in node_dfs.items():
            if not all(node_df['Technology'].isnull()):
                tech_dfs[node_name] = {t: gb for t, gb in node_df.groupby(by='Technology')}
                node_dfs[node_name] = node_df[node_df['Technology'].isnull()]#.drop(columns='Technology')

                # Remove region and sector columns from tech dfs
                for t in tech_dfs[node_name]:
                    tech_dfs[node_name][t] = tech_dfs[node_name][t].drop(columns=['Region'])
                    tech_dfs[node_name][t] = tech_dfs[node_name][t].drop(columns=['Sector'])

        if inplace:
            self.node_dfs = node_dfs
            self.tech_dfs = tech_dfs

        return node_dfs, tech_dfs

    def get_years(self):
        cols = [y for y in self.model_df.columns if is_year(y)]
        return cols

    def get_default_params(self):
        # Read model_description from excel
        mixed_type_columns = ['Default value']

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
        df['Parameter'] = df['Parameter'].str.lower()

        # Default Parameters
        df_has_defaults = df[~df['Default value'].isna()]
        node_tech_defaults = {}
        for param, val in zip(df_has_defaults['Parameter'],
                              df_has_defaults['Default value']):
            node_tech_defaults[param] = utils.infer_type(val)

        # Return
        return node_tech_defaults

def _bool_as_string(val):
    """ Convert bools to str, otherwise leave value as is

    This is done to differentiate between boolean and integer values.
    Otherwise, during pd.read_excel() parsing, a single representation of
    0/False and 1/True is chosen, depending on which value is encountered first
    within the column.
    """
    if isinstance(val, bool):
        return str(val)
    return val
