import warnings
from ..old_utils import is_year
from ..utils import parameters as PARAM
from ..utils import model_columns as COL

def get_providers(df, node_col):
    providers = df[df[COL.parameter] == PARAM.service_provided][node_col]
    return providers


def get_requested(df, target_col):
    requested = df[df[COL.parameter] == PARAM.service_requested][target_col]
    return requested


def get_year_cols(df):
    year_columns = [c for c in df.columns if is_year(c)]
    return year_columns


def get_nodes(df, node_col):
    nodes = df[node_col].dropna().drop_duplicates()
    return nodes
