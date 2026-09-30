from ..utils.model_description import column_list as COL
from ..utils.parameter import list as PARAM
def get_providers(df, node_col):
    providers = df[df[COL.parameter] == PARAM.service_provide][node_col]
    return providers


def get_requested(df, target_col):
    requested = df[df[COL.parameter] == PARAM.service_request][target_col]
    return requested


def get_nodes(df, node_col):
    nodes = df[node_col].dropna().drop_duplicates()
    return nodes
