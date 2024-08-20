import pandas as pd
import numpy as np

from ..utils import is_year
from .validation_utils import get_providers, get_year_cols, get_nodes

COMP_TYPE = 'competition type'
SERV_REQUESTED = 'service requested'


def invalid_competition_type(df):
    """
    Find list of nodes with an invalid competition type.
    """
    valid_comp_types = [
        'Root',
        'Region',
        'Sector',
        'Tech Compete',
        'Node Tech Compete',
        'Fixed Ratio',
        'Supply - Fixed Price',
        'Supply - Cost Curve Annual',
        'Supply - Cost Curve Cumulative'
    ]

    invalid_rows = df[(df['Parameter'] == COMP_TYPE) &
                      (~df['Context'].isin(valid_comp_types))]
    invalid_nodes = list(zip(invalid_rows.index, invalid_rows['Branch']))

    concern_key, concern_desc = "invalid_competition_type", "nodes had invalid competition types"
    return invalid_nodes, concern_key, concern_desc


def unspecified_nodes(providers, requested):
    """
    Identify any nodes which are targets of other nodes, but have not been specified within
    the model description.
    """
    referenced_unspecified = [(i, v) for i, v in requested.items() if v not in providers.values]
    concern_key, concern_desc = 'unspecified_nodes', 'non-root nodes are never referenced'
    return referenced_unspecified, concern_key, concern_desc


def unreferenced_nodes(providers, requested, root_node):
    """
    Identify any non-root nodes which are specified in the model description but are never
    requested by other nodes.
    """
    specified_unreferenced = [(i, v) for i, v in providers.items() if
                              (v not in requested.values) and (v != root_node)]

    concern_key, concern_desc = 'unreferenced_nodes', 'non-root nodes are never referenced'
    return specified_unreferenced, concern_key, concern_desc


def mismatched_node_names(validator, providers):
    """
    Identify any nodes whose service provided name doesn't match the last component of their
    branch.
    """

    mismatched = []
    for i, branch in providers.items():
        branch_node_name = branch.split('.')[-1]
        service_name = validator.model_df['Context'].loc[i]
        node_index = validator.branch2node_index_map[branch]
        if branch_node_name != service_name:
            mismatched.append((node_index, branch_node_name, service_name))

    concern_key, concern_desc = 'mismatched_node_names', 'service name/branch mismatches'
    return mismatched, concern_key, concern_desc


def nodes_no_provided_service(validator):
    """
    Identify any nodes which are specified but do not provide a service.
    """
    providers = get_providers(validator.model_df, node_col=validator.node_col)
    nodes = get_nodes(validator.model_df, validator.node_col)
    nodes_no_service = [(i, n) for i, n in nodes.items() if n not in providers.values]

    concern_key, concern_desc = ('nodes_no_provided_service',
                                 "nodes were specified but don't provide a service")
    return nodes_no_service, concern_key, concern_desc


def nodes_requesting_self(validator):
    """
    Identifies any nodes which request services of themselves. Adds (index, Node Name) pairs
    to the warnings dictionary, under the key "nodes_requesting_self".
    """
    request_rows = validator.model_df[validator.model_df['Parameter'] == 'service requested']
    self_requests = request_rows[
        request_rows[validator.node_col] == request_rows[validator.target_col]]
    self_requesting = [(i, node) for i, node in
                       zip(self_requests[validator.node_col], self_requests.index)]

    concern_key, concern_desc = ('nodes_requesting_self',
                                 'nodes requested services of themselves')
    return self_requesting, concern_key, concern_desc


def nodes_no_requested_service(validator):
    """
    Identify nodes or technologies which have been specified in the model description but don't
    request services from other nodes.
    """
    nodes_techs_no_serv_req = []

    # Nodes (without techs)
    nodes = validator.model_df.groupby(validator.node_col)
    for node, df in nodes:
        if pd.isna(df['Technology']).all():
            if SERV_REQUESTED not in df['Parameter'].values:
                nodes_techs_no_serv_req.append((validator.branch2node_index_map[node], node, None))

    # Technologies
    node_techs = validator.model_df.groupby([validator.node_col, 'Technology'])
    for (node, tech), df in node_techs:
        if SERV_REQUESTED not in df['Parameter'].values:
            nodes_techs_no_serv_req.append((validator.branch2node_index_map[node], node, tech))

    nodes_techs_no_serv_req.sort(key=lambda x: x[0])

    concern_key, concern_desc = ('nodes_no_requested_service',
                                 "nodes or technologies don't request other services")
    return nodes_techs_no_serv_req, concern_key, concern_desc


def nodes_with_zero_output(validator):
    """
    Identify nodes or technologies where the "Output" line has been set to 0 for
    any of the year values in the model description.
    """
    year_cols = get_year_cols(validator.model_df)
    output = validator.model_df[validator.model_df['Parameter'] == 'output']
    zero_outputs = output[(output[year_cols] == 0).any(axis=1)]
    zero_output_nodes = list(zero_outputs['Branch'].items())

    concern_key, concern_desc = 'nodes_with_zero_output', 'nodes have 0 in the output line'
    return zero_output_nodes, concern_key, concern_desc


def supply_nodes_no_lcc_or_price(validator):
    """
    Identify supply nodes that have neither a 'lcc_financial' nor 'price' row specified in
    the base year.
    """
    supply_nodes = validator.model_df[validator.model_df['Parameter'] == 'is supply']['Branch']

    cost_df = validator.model_df[validator.model_df['Parameter'].isin(['lcc_financial', 'price'])]
    has_base_year_cost = cost_df[~cost_df['2000'].isna()]['Branch']
    no_prod_cost = [(validator.branch2node_index_map[f], f) for f in supply_nodes if
                    f not in has_base_year_cost.values]

    concern_key, concern_desc = ('supply_without_lcc_or_price',
                                 "supply nodes don't have a life cycle cost or price")
    return no_prod_cost, concern_key, concern_desc



def techs_no_base_market_share(validator):
    """
    Identify technologies which have a marketshare line, but do not have a base year market share.
    """
    # The model's DataFrame
    data = validator.model_df

    base_year = [c for c in data.columns if is_year(c)][0]
    base_year_market_shares = data[data['Parameter'] == 'market share']
    no_base_year_ms = base_year_market_shares[base_year_market_shares[base_year].isna()]

    techs_no_base_year_ms = []
    for idx in no_base_year_ms.index:
        techs_no_base_year_ms.append((idx,
                                      data.loc[idx, validator.node_col],
                                      data.loc[idx, 'Technology']))

    concern_key, concern_desc = ('techs_no_base_year_ms',
                                 'technologies are missing a base year market share')

    return techs_no_base_year_ms, concern_key, concern_desc


def duplicate_service_requests(validator):
    """
    Identify nodes and technologies which request the same service twice.
    """
    concern_key, concern_desc = ('duplicate_req',
                                 'nodes/technologies request from the same service more'
                                 ' than once')
    # The model's DataFrame
    data = validator.model_df

    serv_request = data[data['Parameter'] == SERV_REQUESTED]
    duplicated = serv_request[serv_request.duplicated(
        subset=[validator.node_col, 'Technology', validator.target_col],
        keep=False)]

    if len(duplicated) > 0:
        # Group & list rows (index) where duplicates exist
        duplicated_with_idx = duplicated.reset_index()
        duplicated_groups = duplicated_with_idx.groupby(
            [validator.node_col, 'Technology', validator.target_col],
            dropna=False)['index'].apply(list)
        duplicated_groups = duplicated_groups.reset_index()

        # Create our Warning information
        duplicate_req = list(zip(duplicated_groups['index'],
                                 duplicated_groups['Branch'],
                                 duplicated_groups['Technology']))
    else:
        duplicate_req = []

    return duplicate_req, concern_key, concern_desc


def bad_service_req(validator):
    """
    Identify nodes/technologies that have a service requested line, but where the values in these
    lines are either blank or exogenously specified as 0.
    """
    # The model's DataFrame
    data = validator.model_df

    # Filter to Only Include Service Requested
    services_req = data[data['Parameter'] == SERV_REQUESTED]

    # Select only the year columns
    year_cols = [c for c in services_req.columns if is_year(c)]
    year_values = services_req[year_cols]

    # Identify rows that have 0's or missing values
    row_has_bad_values = year_values.replace(0, np.nan).isna().all(axis=1)
    rows_with_bad_values = services_req[row_has_bad_values]

    # Create our Warning information
    bad_service_requests = list(zip(rows_with_bad_values.index, rows_with_bad_values[validator.node_col]))

    concern_key, concern_desc = ('bad_service_req',
                                 "nodes/technologies contain only 0's or are missing values"
                                 " in service request rows")

    return bad_service_requests, concern_key, concern_desc


def tech_compete_nodes_no_techs(validator):
    """
    Identify tech compete nodes that don't record any technologies in the "Technology" column.
    """
    # The model's DataFrame
    data = validator.model_df

    # Find all Tech Compete Nodes
    tech_compete_nodes = data[(data['Parameter'] == 'competition type') &
                              (data['Context'] == 'Tech Compete')][validator.node_col]

    # Find all Technology Header Rows
    techs = data[data['Parameter'] == 'technology']

    # Determine which Tech Compete nodes don't have a Technology header
    tc_nodes_no_techs = []
    for idx, node in tech_compete_nodes.items():
        if node not in techs[validator.node_col].values:
            tc_nodes_no_techs.append((idx, node))

    concern_key, concern_desc = ('tech_compete_nodes_no_techs',
                                 "tech compete nodes don't contain Technology or Service "
                                 "headings")

    return tc_nodes_no_techs, concern_key, concern_desc


def market_child_requested(validator):
    """
    Identify any requests made to children of Markets (doesn't include requests from the market
    itself)
    """
    data = validator.model_df

    # Find Markets
    markets = validator.model_df[(validator.model_df['Parameter'] == COMP_TYPE) &
                                 (validator.model_df['Context'] == 'Market')][validator.node_col]

    # Find Market Children
    all_service_req = data[data['Parameter'] == SERV_REQUESTED]
    market_children = [all_service_req[validator.target_col].loc[i]
                       for i, b in all_service_req[validator.node_col].items()
                       if b in markets.values]

    # Find Service Requests for Market Children
    market_children_requests = []
    for i, src, tgt in zip(all_service_req.index,
                           all_service_req[validator.node_col],
                           all_service_req[validator.target_col]):
        if (src not in markets.values) and (tgt in market_children):
                market_children_requests.append((i, src, tgt))

    concern_key, concern_desc = ('market_child_requested',
                                 "nodes/technologies requested services from nodes which are part of a market")

    return market_children_requests, concern_key, concern_desc


def revenue_recycling_at_techs(validator):
    """Revenue recycling should only happen at nodes, never at techs"""

    # The model's DataFrame
    data = validator.model_df

    # Find Recycled Revenues Rows
    rr_tech_df = data[(data['Parameter'] == 'recycled revenues') &
                                  (~data['Technology'].isna())]

    techs_recycling_revenues = []
    for idx in rr_tech_df.index:
        techs_recycling_revenues.append((
            idx,
            rr_tech_df.loc[idx, validator.node_col],
            rr_tech_df.loc[idx, 'Technology']
        ))

    concern_key, concern_desc = ('techs_revenue_recycling',
                                 'technologies are trying to recycle revenues')

    return techs_recycling_revenues, concern_key, concern_desc


def both_cop_p2000_defined(validator):
    """No node should have both COP & P2000 exogenously defined"""

    data = validator.model_df

    # Find all instances of COP & P2000 in the model description
    cop_p2000 = data[data['Parameter'].isin(['cop', 'p2000'])]

    # Only keep the rows that aren't completely None
    cop_p2000 = cop_p2000.dropna(how='all',
                                 subset=[c for c in cop_p2000.columns if is_year(c)])
    duplicated = cop_p2000[cop_p2000.duplicated(
                 subset=[validator.node_col],
                 keep='first')]

    nodes_with_cop_and_p2000 = []
    for i, node in duplicated[validator.node_col].items():
        nodes_with_cop_and_p2000.append((i, node))

    concern_key, concern_desc = ('nodes_with_cop_and_p2000',
                                 'node(s) have both COP & P2000 exogenously defined')
    return nodes_with_cop_and_p2000, concern_key, concern_desc
