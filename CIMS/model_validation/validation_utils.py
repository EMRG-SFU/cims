import warnings
from ..utils import is_year


def get_providers(df, node_col):
    providers = df[df['Parameter'] == 'service provided'][node_col]
    return providers


def get_requested(df, target_col):
    requested = df[df['Parameter'] == 'service requested'][target_col]
    return requested


def get_year_cols(df):
    year_columns = [c for c in df.columns if is_year(c)]
    return year_columns


def get_nodes(df, node_col):
    nodes = df[node_col].dropna().drop_duplicates()
    return nodes
