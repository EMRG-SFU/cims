import pandas as pd
import numpy as np
import re


# DataFrame helpers
def display_df(df):
    """Display None as blanks"""
    df = pd.DataFrame(df)
    if not df.empty:
        display(pd.DataFrame(df).replace({None: ''}))
        
def non_empty_rows(df, exclude_column="Node"):
    """Return bool array to flag rows as False that have only None or False values, ignoring exclude_column"""
    return df.loc[:, df.columns != exclude_column].T.apply(any)

# column extraction helpers
re_year = re.compile(r'^[0-9]{4}$')
def is_year(cn):
    """Check if input int or str is 4 digits [0-9] between begin ^ and end $ of string"""
    # unit test: assert is_year, 1900
    return bool(re_year.match(str(cn)))

def find_first(items, pred=bool, default=None):
    """Find first item for that pred is True"""
    return next(filter(pred, items), default)

def find_first_index(items, pred=bool):
    """Find index of first item for that pred is True"""
    return find_first(enumerate(items), lambda kcn: pred(kcn[1]))[0]

def get_node_cols(mdf, first_data_col_name="Node"):
    """Returns list of column names after 'Node' and a list of years that follow """
    node_col_idx = find_first_index(mdf.columns,
                                    lambda cn: first_data_col_name.lower() in str(cn).lower()) # ML added str to get ints
    relevant_columns = mdf.columns[node_col_idx:]
    year_or_not = list(map(is_year, relevant_columns))
    first_year_idx = find_first_index(year_or_not)
    last_year_idx = find_first_index(year_or_not[first_year_idx:],
                                     lambda b: not b) + first_year_idx
    # list(...)[a:][:b] extracts b elements starting at a
    year_cols = mdf.columns[node_col_idx:][first_year_idx:last_year_idx]
    return mdf.columns[node_col_idx:][:first_year_idx], year_cols