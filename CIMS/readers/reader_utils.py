import re 
from ..utils import model_columns as COL

def is_year(cn):
    re_year = re.compile(r'^\d{4}$')
    """
    Check if input int or str is 4 digits [0-9] between begin ^ and end $ of string
    """
    # unit test: assert is_year, 1900
    return bool(re_year.match(str(cn)))


def find_first(items, pred=bool, default=None):
    """
    Find first item for that pred is True
    """
    return next(filter(pred, items), default)


def find_first_index(items, pred=bool):
    """
    Find index of first item for that pred is True
    """
    return find_first(enumerate(items), lambda kcn: pred(kcn[1]))[0]


def get_node_cols(mdf, first_data_col_name=COL.branch):
    """
    Returns list of column names after `first_data_col_name` and a list of years that follow
    """
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

def _bool_as_string(val):
    """
    Convert bools to str, otherwise leave value as is

    This is done to differentiate between boolean and integer values.
    Otherwise, during pd.read_excel() parsing, a single representation of
    0/False and 1/True is chosen, depending on which value is encountered first
    within the column.
    """
    if isinstance(val, bool):
        return str(val)
    return val
