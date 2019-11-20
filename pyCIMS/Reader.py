import numpy as np
import pandas as pd

import pandas as pd
import re


# ********************************
# Helper Functions
# ********************************
# DataFrame helpers
def non_empty_rows(df, exclude_column="Node"):
    """Return bool array to flag rows as False that have only None or False values, ignoring exclude_column"""
    return df.loc[:, df.columns != exclude_column].T.apply(any)


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


def get_node_cols(mdf, first_data_col_name="Node", extra_cols=['Demand?']):
    """Returns list of column names after 'Node' and a list of years that follow """
    node_col_idx = find_first_index(mdf.columns,
                                    lambda cn: first_data_col_name.lower() in cn.lower())

    extra_columns = pd.Index([c for c in mdf.columns if c in extra_cols])
    relevant_columns = mdf.columns[node_col_idx:].append(extra_columns)

    relevant_columns = mdf.columns[node_col_idx:].append(extra_columns)
    year_or_not = list(map(is_year, relevant_columns))
    first_year_idx = find_first_index(year_or_not)
    last_year_idx = find_first_index(year_or_not[first_year_idx:],
                                     lambda b: not b) + first_year_idx
    # list(...)[a:][:b] extracts b elements starting at a
    year_cols = mdf.columns[node_col_idx:][first_year_idx:last_year_idx]
    node_cols = mdf.columns[node_col_idx:][:first_year_idx].append(extra_columns)
    return node_cols, year_cols


# ********************************
# Core Functions
# ********************************
def read_model_description(infile, model_sheet, node_col, extra_cols):
    # ------------------------
    # Read in the data
    # ------------------------
    mxl = pd.read_excel(infile, sheet_name=None, header=1)  # Read model_description from excel
    model_df = mxl[model_sheet].replace({pd.np.nan: None})  # Read the model sheet into a dataframe
    model_df.index += 3                                     # Adjust index to correspond to Excel line numbers
                                                            # (+1: 0 vs 1 origin, +1: header skip, +1: column headers)
    model_df.columns = [str(c) for c in model_df.columns]   # Convert all column names to strings (years were ints)
    n_cols, y_cols = get_node_cols(model_df, node_col, extra_cols)    # Find columns, separated year cols from non-year cols
    all_cols = np.concatenate((n_cols, y_cols))
    mdf = model_df.loc[1:, all_cols]                        # Create df, drop irrelevant columns & skip first, empty row

    # ------------------------
    # Extract Node DFs
    # ------------------------
    # determine, row ranges for each node def, based on non-empty Node field
    node_rows = mdf.Node[~mdf.Node.isnull()]  # does not work if node names have been filled in
    node_rows.index.name = "Row Number"
    last_row = mdf.index[-1]
    node_start_ends = zip(node_rows.index,
                          node_rows.index[1:].tolist() + [last_row])

    # extract Node DataFrames, at this point still including Technologies
    node_dfs = {}
    non_node_cols = mdf.columns != node_col
    for s, e in node_start_ends:
        node_df = mdf.loc[s + 1:e - 1]
        node_df = node_df.loc[non_empty_rows(node_df), non_node_cols]
        try:
            node_name = list(node_df[node_df['Parameter'] == 'Service provided']['Branch'])[0]
        except IndexError:
            continue
        node_dfs[node_name] = node_df

    # ------------------------
    # Extract Tech DFs
    # ------------------------
    # Extract tech dfs from node df's and rewrite node df without techs
    tech_dfs = {}
    for nn, ndf in node_dfs.items():
        if any(ndf.Parameter.isin(["Technology", "Service"])):  # Technologies can also be called Services
            tdfs = {}
            first_row, last_row = ndf.index[0], ndf.index[-1]
            tech_rows = ndf.loc[ndf.Parameter.isin(["Technology", "Service"])].index
            for trs, tre in zip(tech_rows, tech_rows[1:].tolist() + [last_row]):
                tech_df = ndf.loc[trs:tre - 1]
                tech_name = tech_df.iloc[0].Value
                tdfs[tech_name] = tech_df
            tech_dfs[nn] = tdfs
            node_dfs[nn] = ndf.loc[:tech_rows[0] - 1]

    return node_dfs, tech_dfs

