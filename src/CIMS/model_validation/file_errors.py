import pandas as pd
import numpy as np


from ..utils.model_description import column_list as COL
from ..utils.parameter import list as PARAM


def _base_year(validator):
    """Return the base year: min of year_list if provided, else min Year in model_df."""
    if validator.year_list:
        return min(validator.year_list, key=int)
    return str(validator.model_df["Year"].dropna().min())

def invalid_competition_type(validator):
    """
    Find nodes with an invalid or missing competition type.

    Requires a list CSV (validator.competition_types). Returns None to signal
    skipped when no list is provided — the valid types are unknown without it.
    """
    if not len(validator.competition_types):
        return None, "no list CSV provided"

    df = validator.model_df
    comp_rows = df[df[COL.parameter] == PARAM.competition_type]
    value_str = comp_rows[COL.value].astype(str).str.strip().str.lower()
    missing_or_invalid = (
        comp_rows[COL.value].isna()
        | value_str.eq("")
        | ~value_str.isin(validator.competition_types)
    )

    invalid_rows = comp_rows[missing_or_invalid]
    invalid_nodes = list(zip(invalid_rows.index, invalid_rows[COL.branch]))

    concern_desc = "nodes have an invalid or missing competition type"
    return invalid_nodes, concern_desc

def undefined_nodes(validator, providers, requested):
    """
    Identify any nodes which are targets of other nodes, but have not been specified within
    the model description.

    Returns a dict {undefined_node: [row_indexes]} so the caller can see both
    the number of unique undefined nodes and every row that references each one.
    The concern_desc reports both the unique node count (via len(concerns) in
    _raise_concerns) and the total number of service_request rows involved.
    """
    # Group row indexes by undefined target node
    undefined: dict = {}
    for i, v in requested.items():
        if v not in providers.values:
            undefined.setdefault(v, []).append(i)

    total_requests = sum(len(idxs) for idxs in undefined.values())
    concern_desc = (
        f"node(s) are referenced across {total_requests} service_request row(s) "
        "without being defined in the model"
    )

    return undefined, concern_desc


def nodes_requesting_self(validator):
    """
    Identifies any nodes which request services of themselves.
    """
    request_rows = validator.model_df[validator.model_df[COL.parameter] == PARAM.service_request]
    self_requests = request_rows[
        request_rows[validator.node_col] == request_rows[validator.target_col]]
    self_requesting = list(zip(self_requests.index, self_requests[validator.node_col]))

    concern_desc = "nodes have a Service requested of themselves"
    return self_requesting, concern_desc

def nodes_with_zero_output(validator):
    """
    Identify nodes or technologies where the "Output" line has been set to 0 for
    any of the year values in the model description.
    """
    output = validator.model_df[validator.model_df[COL.parameter] == PARAM.output]
    zero_outputs = output[pd.to_numeric(output["Value"], errors="coerce") == 0]

    zero_output_nodes = []
    for node, group in zero_outputs.groupby(COL.branch):
        years = sorted(group["Year"].dropna().tolist())
        zero_output_nodes.append((group.index[0], node, years))

    concern_desc = "nodes have an Output value of 0"
    return zero_output_nodes, concern_desc

def supply_without_lcc_or_price(validator):
    """
    Identify supply nodes (fixed price or cost curve) that have no price,
    lcc financial, or cost curve price specified in the base year.

    Aggregation nodes with competition = Sector are excluded: they derive
    their base-year price from children and do not need a direct price row.
    """
    # Supply nodes: is_supply parameter with value == "True"
    supply_nodes = validator.model_df[
        validator.model_df[COL.parameter].str.contains(PARAM.is_supply, case=False, na=False) &
        (validator.model_df[COL.value].str.lower() == "true")
    ][COL.branch]

    # Aggregation nodes (competition = Sector) derive price from children — skip them
    sector_nodes = validator.model_df[
        (validator.model_df[COL.parameter] == PARAM.competition_type) &
        (validator.model_df[COL.value].str.lower() == "sector")
    ][COL.branch]
    supply_nodes = supply_nodes[~supply_nodes.isin(sector_nodes)]

    # Cost rows with a value defined in the base year
    base_year = _base_year(validator)
    cost_df = validator.model_df[validator.model_df[COL.parameter].isin(
        [PARAM.lcc_financial, PARAM.price, PARAM.cost_curve_price])]
    has_base_year_cost = cost_df[
        (cost_df["Year"] == base_year) & cost_df["Value"].notna()
    ][COL.branch]

    no_prod_cost = [(validator.branch2node_index_map[f], f) for f in supply_nodes if
                    f not in has_base_year_cost.values]

    concern_desc = "supply nodes (fixed price or cost curve) are missing a price"
    return no_prod_cost, concern_desc

def techs_no_base_market_share(validator):
    """
    Identify technologies which have a market share line, but do not have a base year market share.
    """
    data = validator.model_df
    base_year = _base_year(validator)
    ms_rows = data[data[COL.parameter] == PARAM.market_share_new]

    # Unique (node, tech) pairs with any market share value defined
    unique_defined = ms_rows[ms_rows["Value"].notna()].drop_duplicates(subset=[validator.node_col, COL.technology])

    # (node, tech) pairs that have a value at the base year
    base_year_ms = ms_rows[(ms_rows["Year"] == base_year) & ms_rows["Value"].notna()]
    has_base_year_ms = set(zip(base_year_ms[validator.node_col], base_year_ms[COL.technology]))

    # Use pd.Series so an empty mask is treated as boolean indexing, not column selection
    missing_base_year = pd.Series(
        [(n, t) not in has_base_year_ms
         for n, t in zip(unique_defined[validator.node_col], unique_defined[COL.technology])],
        index=unique_defined.index,
        dtype=bool,
    )
    flagged = unique_defined[missing_base_year]
    techs_no_base_year_ms = list(zip(flagged.index, flagged[validator.node_col], flagged[COL.technology]))

    concern_desc = f"technologies are missing a base year ({base_year}) market share"
    return techs_no_base_year_ms, concern_desc

def tech_compete_nodes_no_techs(validator):
    """
    Identify tech compete nodes that don't contain any technologies in the COL.technology column.
    """
    # The model's DataFrame
    data = validator.model_df

    # Find all Tech Compete Nodes
    tech_compete_nodes = data[(data[COL.parameter].str.lower().str.contains(PARAM.competition_type)) &
                              (data[COL.value].str.lower().str.contains(PARAM.competition_compete))][validator.node_col]

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
    Identify technologies with a revenue_recycled parameter row.

    Revenue recycling must only be applied at the node level, never at a technology.
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
    No node should have both COP & P2000 exogenously defined.
    """
    data = validator.model_df

    # Rows where COP or P2000 has a value defined
    cop_p2000 = data[data[COL.parameter].isin([PARAM.cop, PARAM.p2000]) & data["Value"].notna()]

    # Nodes with each parameter defined
    has_cop   = set(cop_p2000[cop_p2000[COL.parameter] == PARAM.cop][validator.node_col])
    has_p2000 = set(cop_p2000[cop_p2000[COL.parameter] == PARAM.p2000][validator.node_col])

    # One representative row per node that has both defined (for the index)
    flagged = cop_p2000[cop_p2000[validator.node_col].isin(has_cop & has_p2000)] \
                       .drop_duplicates(subset=[validator.node_col])
    nodes_with_cop_and_p2000 = list(zip(flagged.index, flagged[validator.node_col]))

    concern_desc = "nodes have both COP & P2000 exogenously defined"
    return nodes_with_cop_and_p2000, concern_desc

def inconsistent_tech_refs(validator):
    """
    Identify data rows whose Technology value doesn't match any technology
    declared (via Parameter='technology') at that node.

    Returns a dict {(node, tech_name): [row_indexes]} so the caller can see
    the number of unique mismatches (dict keys) and every affected row for
    each one. Common cause: typos or capitalisation differences between the
    declaration row and data rows.
    """
    data = validator.model_df
    tech_data = data[data[COL.technology].notna() & (data[COL.technology].str.strip() != "")]

    # Build node → [declared tech names] from Parameter='technology' rows
    tech_rows = tech_data[tech_data[COL.parameter] == COL.technology.lower()]
    node_tech_map: dict = {}
    for node, tech_name in zip(tech_rows[validator.node_col], tech_rows[COL.technology]):
        node_tech_map.setdefault(node, []).append(tech_name)

    # Find data rows that reference an undeclared tech at their node.
    # Result: {node: {tech: [row_indexes]}} so callers can drill down
    # node → undeclared tech names → affected rows.
    tech_other = tech_data[~tech_data.index.isin(tech_rows.index)]
    mismatches: dict = {}
    for idx, node, tech in zip(tech_other.index, tech_other[validator.node_col], tech_other[COL.technology]):
        if tech not in node_tech_map.get(node, []):
            mismatches.setdefault(node, {}).setdefault(tech, []).append(idx)

    total_rows = sum(len(idxs) for techs in mismatches.values() for idxs in techs.values())
    concern_desc = (
        f"node(s) contain data rows for technology names missing from that node's "
        f"'technology' parameter declaration — {total_rows} data row(s) affected"
    )

    return mismatches, concern_desc

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
                      (data[COL.value].str.lower().str.contains(PARAM.competition_compete))][validator.node_col].unique()


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
    data = validator.model_df

    ms_min_limits = data[data[COL.parameter] == PARAM.market_share_new_min]
    ms_max_limits = data[data[COL.parameter] == PARAM.market_share_new_max]

    df = pd.merge(ms_min_limits, ms_max_limits,
                  how='inner', validate='many_to_many',
                  on=[COL.branch, COL.region, COL.sector, COL.technology, "Year"],
                  suffixes=["_min", "_max"])

    df["Value_min"] = pd.to_numeric(df["Value_min"], errors="coerce")
    df["Value_max"] = pd.to_numeric(df["Value_max"], errors="coerce")
    incongruent = df[df["Value_min"] > df["Value_max"]]

    issues = incongruent.groupby([COL.branch, COL.technology])["Year"].apply(list).reset_index()
    min_max_conflicts = list(zip(issues[COL.branch], issues[COL.technology], issues["Year"]))

    concern_desc = "technologies contain market share limits that conflict \
        with one another (e.g., min > max)"

    return min_max_conflicts, concern_desc


def new_techs_in_scenario(validator):
    """
    Identify new technologies included in the scenario models (i.e. were not in
    the base model) but which don't have a technology parameter.
    """
    if not validator._scenario_df.empty:
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
    tech_nodes = validator.model_df[COL.branch][(validator.model_df[COL.parameter] == PARAM.competition_type) & (validator.model_df[COL.value].str.lower().str.contains(PARAM.competition_compete))]
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
    lcc_tech_rows = validator.model_df[
        ~validator.model_df[COL.technology].isna() &
        validator.model_df[COL.parameter].str.lower().str.contains('lcc')
    ].drop_duplicates(subset=[COL.branch, COL.technology])

    lcc_at_techs = list(zip(lcc_tech_rows.index, lcc_tech_rows[COL.branch]))

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

    # Identify base year
    base_year = _base_year(validator)

    # Filter to base year rows and ensure numeric market shares
    market_share_df = market_share_df[market_share_df["Year"] == base_year].copy()
    market_share_df["Value"] = pd.to_numeric(market_share_df["Value"], errors='coerce')

    # Sum market shares for each branch
    grouped = market_share_df.groupby(COL.branch)["Value"].sum().reset_index()

    # Identify branches where the market share sum is not ~1
    invalid = grouped[~np.isclose(grouped["Value"], 1.0, atol=0.001)]

    # Build result: list of (node index, branch name, summed market share)
    nodes_with_bad_shares = [
        (validator.branch2node_index_map[branch], branch, total_share)
        for branch, total_share in zip(invalid[COL.branch], invalid["Value"])
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
    Identify nodes missing a Competition parameter row.
    """
    data = validator.model_df

    # Only node-level rows (exclude tech-level rows)
    node_rows = data[data[COL.technology].isna()]

    # All nodes defined in the model
    nodes = set(data[validator.node_col].dropna().unique())

    # Nodes with Competition (any context value, including blank)
    competition_rows = node_rows[node_rows[COL.parameter] == PARAM.competition_type]
    nodes_with_competition = set(competition_rows[validator.node_col].dropna().unique())

    missing = []
    for node in sorted(nodes):
        if node not in nodes_with_competition:
            missing.append((validator.branch2node_index_map[node], node))

    concern_desc = "nodes are missing a Competition parameter"
    return missing, concern_desc

def no_structural_parent_node_exists(validator):
    """
    Identify non-root nodes whose structural parent is missing.
    """
    data = validator.model_df

    # All defined branches in the model (base + scenario files)
    branch_set = set(data[COL.branch].dropna().unique())
    
    missing_parents: dict = {}
    for node in sorted(branch_set):
        if node == validator.root_node:
            continue
        parent = ".".join(node.split(".")[:-1])
        if not parent or parent not in branch_set:
            missing_parents.setdefault(parent, []).append(node)

    total_children = sum(len(c) for c in missing_parents.values())
    concern_desc = f"parent node(s) are not defined — {total_children} child node(s) affected"
    return missing_parents, concern_desc
