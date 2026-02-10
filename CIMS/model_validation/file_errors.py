import pandas as pd
import numpy as np

from .validation_utils import get_providers, get_year_cols, get_nodes

from ..utils.model_description import column_list as COL
from ..utils.parameter import list as PARAM
from ..utils.parameter.parse import is_year


def invalid_competition_type(validator):
    """
    Find list of nodes with an invalid competition type.
    """
    df = validator.model_df
    valid_competition_list = validator.competition_types
    
    invalid_rows = df[(df[COL.parameter] == PARAM.competition_type) &
                      (~df[COL.context].str.lower().isin(valid_competition_list))]
    invalid_nodes = list(zip(invalid_rows.index, invalid_rows[COL.branch]))

    concern_desc = "nodes have an invalid 'Competition Type'"
    return invalid_nodes, concern_desc

def undefined_nodes(validator, providers, requested):
    """
    Identify any nodes which are targets of other nodes, but have not been specified within
    the model description.
    """
    referenced_unspecified = [(i, v) for i, v in requested.items() if v not in providers.values]
    concern_desc = "nodes are requested by other nodes without being defined in the model"

    return referenced_unspecified, concern_desc

def nodes_no_provided_service(validator):
    """
    Identify any nodes which are specified but do not provide a service.
    """
    # Only count node-level service_provide rows (exclude tech-level rows)
    providers = validator.model_df[
        (validator.model_df[COL.parameter] == PARAM.service_provide) &
        (validator.model_df[COL.technology].isna())
    ][validator.node_col]
    nodes = get_nodes(validator.model_df, validator.node_col)
    nodes_no_service = [(i, n) for i, n in nodes.items() if n not in providers.values]

    concern_desc = "nodes are specified but have no 'Service Provide'"
    return nodes_no_service, concern_desc

def nodes_requesting_self(validator):
    """
    Identifies any nodes which request services of themselves.
    """
    request_rows = validator.model_df[validator.model_df[COL.parameter] == PARAM.service_request]
    self_requests = request_rows[
        request_rows[validator.node_col] == request_rows[validator.target_col]]
    self_requesting = [(i, node) for i, node in
                       zip(self_requests[validator.node_col], self_requests.index)]

    concern_desc = "nodes have a Service requested of themselves"
    return self_requesting, concern_desc

def nodes_with_zero_output(validator):
    """
    Identify nodes or technologies where the "Output" line has been set to 0 for
    any of the year values in the model description.
    """
    year_cols = get_year_cols(validator.model_df)
    output = validator.model_df[validator.model_df[COL.parameter] == PARAM.output]
    zero_outputs = output[(output[year_cols] == 0).any(axis=1)]
    zero_output_nodes = list(zero_outputs[COL.branch].items())

    concern_desc = "nodes have an Output value of 0"
    return zero_output_nodes, concern_desc

def supply_without_lcc_or_price(validator):
    """
    Identify supply nodes (fixed price or cost curve) that have no price, 
    lcc financial, or cost curve price specified in the base year.
    """
    supply_nodes = validator.model_df[(validator.model_df[COL.parameter].str.contains(PARAM.is_supply, case=False, na=False)) &
                                       (validator.model_df[COL.context] is True)][COL.branch]

    cost_df = validator.model_df[validator.model_df[COL.parameter].isin([PARAM.lcc_financial, PARAM.price, PARAM.cost_curve_price])]
    has_base_year_cost = cost_df[~cost_df[PARAM.base_year].isna()][COL.branch]
    no_prod_cost = [(validator.branch2node_index_map[f], f) for f in supply_nodes if
                    f not in has_base_year_cost.values]

    concern_desc = "supply nodes (fixed price or cost curve) are missing a price"
    return no_prod_cost, concern_desc

def techs_no_base_market_share(validator):
    """
    Identify technologies which have a market share line, but do not have a base year market share.
    """
    # The model's DataFrame
    data = validator.model_df

    base_year = [c for c in data.columns if is_year(c)][0]
    base_year_market_shares = data[data[COL.parameter] == PARAM.market_share_new]
    no_base_year_ms = base_year_market_shares[base_year_market_shares[base_year].isna()]

    techs_no_base_year_ms = []
    for idx in no_base_year_ms.index:
        techs_no_base_year_ms.append((idx,
                                      data.loc[idx, validator.node_col],
                                      data.loc[idx, COL.technology]))

    concern_desc = "technologies are missing a base year Market share"

    return techs_no_base_year_ms, concern_desc

def tech_compete_nodes_no_techs(validator):
    """
    Identify tech compete nodes that don't contain any technologies in the COL.technology column.
    """
    # The model's DataFrame
    data = validator.model_df

    # Find all Tech Compete Nodes
    tech_compete_nodes = data[(data[COL.parameter].str.lower().str.contains(PARAM.competition_type)) &
                              (data[COL.context].str.lower().str.contains(PARAM.competition_compete))][validator.node_col]

    # Find all Technology Header Rows
    techs = data[data[COL.parameter] == COL.technology.lower()]

    # Determine which Tech Compete nodes don't have a Technology header
    tc_nodes_no_techs = []
    for idx, node in tech_compete_nodes.items():
        if node not in techs[validator.node_col].values:
            tc_nodes_no_techs.append((idx, node))

    concern_desc = "tech compete nodes contain no technologies"

    return tc_nodes_no_techs, concern_desc

def revenue_recycling_at_techs(validator):
    """
    Revenue recycling should only happen at nodes, never at techs
    """

    # The model's DataFrame
    data = validator.model_df

    # Find Recycled Revenues Rows
    rr_tech_df = data[(data[COL.parameter] == PARAM.revenue_recycled) &
                                  (~data[COL.technology].isna())]

    techs_recycling_revenues = []
    for idx in rr_tech_df.index:
        techs_recycling_revenues.append((
            idx,
            rr_tech_df.loc[idx, validator.node_col],
            rr_tech_df.loc[idx, COL.technology]
        ))

    concern_desc = "technologies have a Recycled revenues parameter (this \
        should only occur at nodes)"

    return techs_recycling_revenues, concern_desc

def both_cop_p2000_defined(validator):
    """
    No node should have both COP & P2000 exogenously defined
    """

    data = validator.model_df

    # Find all instances of COP & P2000 in the model description
    cop_p2000 = data[data[COL.parameter].isin([PARAM.cop, PARAM.p2000])]

    # Only keep the rows that aren't completely None
    cop_p2000 = cop_p2000.dropna(how='all',
                                 subset=[c for c in cop_p2000.columns if is_year(c)])
    duplicated = cop_p2000[cop_p2000.duplicated(
                 subset=[validator.node_col],
                 keep='first')]

    nodes_with_cop_and_p2000 = []
    for i, node in duplicated[validator.node_col].items():
        nodes_with_cop_and_p2000.append((i, node))

    concern_desc = "nodes have both COP & P2000 exogenously defined"
    return nodes_with_cop_and_p2000, concern_desc

def inconsistent_tech_refs(validator):
    """
    Identify nodes which include `Technology` column values and reference a
    technology which does not exist at that node.
    """
    # The model's DataFrame
    data = validator.model_df
    tech_data = data[~data[COL.technology].isna()]

    # Build a Node -> [Technologies] map
    tech_rows = tech_data[tech_data[COL.parameter]==COL.technology.lower()]
    node_tech_map = {}
    for node, tech_name in zip(tech_rows[validator.node_col], tech_rows[COL.technology]):
        if node not in node_tech_map:
            node_tech_map[node] = []
        node_tech_map[node].append(tech_name)

    # Find Unique Node/Branch + Technology rows
    inconsistent_tech_refs = []
    tech_other = tech_data[~tech_data.isin(tech_rows)].dropna(how='all')
    for idx, node, tech in zip(tech_other.index, tech_other[validator.node_col], tech_other[COL.technology]):
        try:
            if tech not in node_tech_map[node]:
                inconsistent_tech_refs.append((idx, node, tech))
        except KeyError:
            inconsistent_tech_refs.append((idx, node, tech))


    # Create Warning information
    concern_desc = "rows have inconsistent technology names"

    return inconsistent_tech_refs, concern_desc

def service_req_at_tech_node(validator):
    """
    Identify tech nodes where a service request is specified at the node level.

    The implication of this is that values such as emissions_rate_cumul_cost
    will be incorrect.
    """
    # The model's DataFrame
    data = validator.model_df

    # Find all Tech compete nodes
    tech_nodes = data[(data[COL.parameter]==PARAM.competition_type) &
                      (data[COL.context].str.lower().str.contains(PARAM.competition_compete))][validator.node_col].unique()


    # Find service request rows specified at the node level of a [node-]tech
    # compete node
    req_at_tech_node_rows = data[(data[COL.parameter] == PARAM.service_request) &
                                 (data[COL.technology].isna()) &
                                 (data[validator.node_col].isin(tech_nodes))]

    # Find Unique Node/Branch + Technology rows
    service_req_at_tech_node = []
    for idx, node in zip(req_at_tech_node_rows.index,
                               req_at_tech_node_rows[validator.node_col]):
        service_req_at_tech_node.append((idx, node))

    # Create Warning information
    concern_desc = "Tech Compete nodes have node-level Service requests (these \
        should only occur at the tech-level)"

    return service_req_at_tech_node, concern_desc

def min_max_conflicts(validator):
    """
    Identify technologies where the market share limits set conflict with one
    another. For example, max=0.5<min=0.7.
    """
    # The model's DataFrame
    data = validator.model_df

    # Min/Max Marketshare Limits
    ms_min_limits = data[data[COL.parameter] == PARAM.market_share_new_min]
    ms_max_limits = data[data[COL.parameter] == PARAM.market_share_new_max]

    # Build a Node -> [Technologies] map
    df = pd.merge(ms_min_limits, ms_max_limits,
                  how='inner', validate='many_to_many',
                  on=[COL.branch, COL.region, COL.sector, COL.technology],
                  suffixes=["_min", "_max"])

    issues = {}
    for y in get_year_cols(data):
        incongruent_nodes = df[df[f"{y}_min"] > df[f"{y}_max"]]
        for branch, tech in zip(incongruent_nodes[COL.branch], incongruent_nodes[COL.technology]):
            if (branch, tech) not in issues:
                issues[(branch, tech)] = []
            issues[(branch, tech)].append(y)


    # Find Unique Node/Branch + Technology rows
    min_max_conflicts = []
    for ((node, tech), years) in issues.items():
        min_max_conflicts.append((node, tech, years))

    # Create Warning information
    concern_desc = "technologies contain market share limits that  conflict \
        with one another (e.g., min > max)"

    return min_max_conflicts, concern_desc

def new_nodes_in_scenario(validator):
    """
    Identify new nodes included in the scenario models (i.e. were not in the
    base model) but which don't have a service provide parameter.
    """
    if validator.scenario_files:
        # The model dataframes
        base_data = validator._get_model_df(read_scenario_files=False)
        scenario_data = validator._get_model_df(read_base_file=False)

        # Find nodes from base and scenario files
        base_nodes = set(base_data[validator.node_col].dropna())
        scen_nodes = set(scenario_data[validator.node_col].dropna())
        declared_new_nodes = set(scenario_data[scenario_data[COL.parameter]==PARAM.service_provide]\
            [validator.node_col].dropna())

        # Find new nodes which haven't been declared without a service provide line
        new_nodes_in_scenario = list(scen_nodes\
                                    .difference(declared_new_nodes)\
                                    .difference(base_nodes))
    else:
        new_nodes_in_scenario = []

    # Create Warning information
    concern_desc = "nodes were included in scenario/model update files without \
        a Service provide parameter"

    return new_nodes_in_scenario, concern_desc

def new_techs_in_scenario(validator):
    """
    Identify new technologies included in the scenario models (i.e. were not in
    the base model) but which don't have a technology parameter.
    """
    if validator.scenario_files:
        # The model dataframes
        base_data = validator._get_model_df(read_scenario_files=False)
        scenario_data = validator._get_model_df(read_base_file=False)

        # Find nodes from base and scenario files
        base_techs = set([tuple(x) for x in
                        base_data[[validator.node_col, COL.technology]]\
                            .dropna().drop_duplicates().values])
        scen_techs = set([tuple(x) for x in
                        scenario_data[[validator.node_col, COL.technology]]\
                            .dropna().drop_duplicates().values])

        scen_declared_techs = scenario_data[scenario_data[COL.parameter]==COL.technology.lower()]
        declared_new_techs = set([tuple(x) for x in
                                scen_declared_techs[[validator.node_col, COL.technology]]\
                                    .dropna().drop_duplicates().values])

        # Find new nodes which haven't been declared without a service provide line
        new_techs_in_scenario = list(scen_techs\
                                    .difference(declared_new_techs)\
                                    .difference(base_techs))
    else:
        new_techs_in_scenario = []

    # Create Warning information
    concern_desc = "technologies were included in scenario/model update files \
        without a Technology parameter"

    return new_techs_in_scenario, concern_desc

def lcc_at_tech_node(validator):
    """
    Identify any tech-compete nodes where an LCC value has been set exogenously.
    """
    tech_nodes = validator.model_df[COL.branch][(validator.model_df[COL.parameter] == PARAM.competition_type) & (validator.model_df[COL.context].str.lower().str.contains(PARAM.competition_compete))]
    lcc_nodes = validator.model_df[COL.branch][
        validator.model_df[COL.technology].isna() &
        validator.model_df[COL.parameter].str.lower().str.contains('lcc')]
    
    lcc_at_tech_nodes = [(i, n) for i, n in lcc_nodes.items() if n in tech_nodes]

    concern_desc = "tech compete nodes have exogenously defined LCC values"

    return lcc_at_tech_nodes, concern_desc

def lcc_at_tech(validator):
    """
    Identify any technologies where an LCC value has been set exogenously.
    """
    techs = validator.model_df[[COL.branch, COL.technology]].drop_duplicates().dropna(how='any')
    
    lcc_techs = validator.model_df[COL.branch][
        ~validator.model_df[COL.technology].isna() &
        validator.model_df[COL.parameter].str.lower().str.contains('lcc')]
    
    lcc_at_techs = [(i, n) for i, n in lcc_techs.items() if n in techs]

    concern_desc = "technologies have exogenously defined LCC values"

    return lcc_at_techs, concern_desc

def base_year_market_share_not_one(validator):
    """
    Identifies branches where the sum of base year market shares 
    for competing technologies does not equal 1.
    """
    # Extract market share data from model
    model_df = validator.model_df
    market_share_df = model_df[model_df[COL.parameter] == PARAM.market_share_total]

    # Identify base year column
    base_year_col = next(col for col in market_share_df.columns if is_year(col))

    # Select relevant columns and ensure numeric market shares
    selected_cols = [COL.branch, COL.technology, base_year_col]
    market_share_df = market_share_df[selected_cols].copy()
    market_share_df[base_year_col] = pd.to_numeric(market_share_df[base_year_col], errors='coerce')

    # Sum market shares for each branch
    grouped = market_share_df.groupby(COL.branch)[base_year_col].sum().reset_index()

    # Identify branches where the market share sum is not ~1
    invalid = grouped[~np.isclose(grouped[base_year_col], 1.0, atol=0.001)]

    # Build result: list of (node index, branch name, summed market share)
    nodes_with_bad_shares = [
        (validator.branch2node_index_map[branch], branch, total_share)
        for branch, total_share in zip(invalid[COL.branch], invalid[base_year_col])
    ]

    concern_desc = "nodes whose base year market shares do not sum to 1"

    return nodes_with_bad_shares, concern_desc

def nodes_missing_service_provide(validator):
    """
    Identify nodes missing a Service Provide parameter.
    """
    data = validator.model_df

    # Only node-level rows (exclude tech-level rows)
    node_rows = data[data[COL.technology].isna()]

    # All nodes defined in the model
    nodes = set(data[validator.node_col].dropna().unique())

    # Nodes with Service Provide
    service_rows = node_rows[node_rows[COL.parameter] == PARAM.service_provide]
    nodes_with_service = set(service_rows[validator.node_col].dropna().unique())

    missing = []
    for node in sorted(nodes):
        if node not in nodes_with_service:
            missing.append((validator.branch2node_index_map[node], node))

    concern_desc = "nodes are missing a Service Provide parameter"
    return missing, concern_desc

def nodes_missing_competition(validator):
    """
    Identify nodes missing a Competition parameter row with a non-empty Context value.
    """
    data = validator.model_df

    # Only node-level rows (exclude tech-level rows)
    node_rows = data[data[COL.technology].isna()]

    # All nodes defined in the model
    nodes = set(data[validator.node_col].dropna().unique())

    # Nodes with non-empty Competition values (found in context column)
    competition_rows = node_rows[node_rows[COL.parameter] == PARAM.competition_type]
    context_str = competition_rows[COL.context].astype(str).str.strip()
    competition_rows = competition_rows[competition_rows[COL.context].notna() & context_str.ne("")]
    nodes_with_competition = set(competition_rows[validator.node_col].dropna().unique())

    missing = []
    for node in sorted(nodes):
        if node not in nodes_with_competition:
            missing.append((validator.branch2node_index_map[node], node))

    concern_desc = "nodes are missing a Competition parameter"
    return missing, concern_desc
