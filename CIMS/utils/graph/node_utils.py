from __future__ import annotations  # For Type Hinting

import copy

from CIMS import cost_curves
from ..model_description import column_list as COL
from ..parameter.parse import infer_type, is_year
from ..parameter import list as PARAM


def _get_current_year_dict(graph, node, year, tech=None):
    if year in graph.nodes[node]:
        node_year_dict = graph.nodes[node][year]

        if tech is not None:
            if PARAM.technologies not in node_year_dict:
                year_dict = {}
            elif tech not in node_year_dict[PARAM.technologies]:
                year_dict = {}
            else:
                year_dict = node_year_dict[PARAM.technologies][tech]
        else:
            year_dict = node_year_dict
    else:
        year_dict = {}

    return year_dict


def _allowable_update(existing_value, update_value):
    # If the value to update is None
        # if there exists a value -> don't update
        # if there doesn't exist a value -> update
    # If the value to update is "None" -> change value to None & update
    # Otherwise -> update as is
    if isinstance(update_value, dict):
        year_value = update_value[PARAM.year_value]
        has_existing_val = bool(existing_value)
    else:
        year_value = update_value
        has_existing_val = existing_value is not None

    if year_value is None:
        if has_existing_val:
            return False, None
        else:
            return True, update_value
    elif isinstance(year_value, str) and year_value.lower() == 'none':
        if isinstance(update_value, dict):
            update_value[PARAM.year_value] = None
            return True, update_value
        else:
            return True, None
    else:
        return True, update_value


def _get_existing_value(year_dict, param, context, sub_context, target):
    # If `target` is defined, we expect to find what we want nested under
    # year_dict[`param`][`target`]. If it's not defined, it will just be under `param`.
    if target:
        target_result = year_dict.get(param, {}).get(target, {})
    else:
        target_result = year_dict.get(param, {})

    # As above, if `context`/`sub_context` are defined, the `PARAM.year_value` will
    # be similarly nested.
    if context:
        if sub_context:
            existing_value = target_result.get(context, {}).get(sub_context, {}).get(PARAM.year_value)
        else:
            context_result = target_result.get(context, {})
            if isinstance(context_result, dict):
                existing_value = target_result.get(context, {}).get(PARAM.year_value)
            else:
                existing_value = target_result.get(PARAM.year_value)
    else:
        existing_value = target_result.get(PARAM.year_value)

    return existing_value


def _update_year_dict(existing_year_dict, update_data):
    year_dict = copy.deepcopy(existing_year_dict)

    for _, _, param, context, sub_context, target, unit, year_value in zip(*update_data):
        value_dict = {
            PARAM.context: context,
            PARAM.sub_context: sub_context,
            PARAM.target: target,
            PARAM.unit: unit,
            PARAM.year_value: infer_type(year_value),
            PARAM.param_source: 'model'
            }
        if param not in year_dict:
            year_dict[param] = {}

        # Determine whether update should proceed
        existing_value = _get_existing_value(year_dict, param, context, sub_context, target)
        update_ok, value_dict = _allowable_update(existing_value, value_dict)
        if not update_ok:
            continue

        if param == PARAM.service_requested:
            year_dict[param].update({target: value_dict})
        elif param == PARAM.price_multiplier:
            year_dict[param].update({target: value_dict})
        else:
            # If a Context value is present, there are 3 possibilities for what needs to happen
            if context:

                # 1. We place our value dictionary inside a nested dictionary keyed
                #    by context & sub-context.
                if sub_context:
                    if context not in year_dict[param]:
                        year_dict[param][context] = {}
                    year_dict[param][context][sub_context] = value_dict

                # 2. We place our value dictionary keyed only by context.
                elif (value_dict[PARAM.year_value] is not None) or \
                    ((param in existing_year_dict) and (PARAM.year_value not in existing_year_dict[param])):
                    year_dict[param][context] = value_dict

                # 3. We save context as the year_value, which will remain constant
                #    across all years.
                else:
                    value_dict[PARAM.year_value] = infer_type(context)
                    year_dict[param] = value_dict
            else:
                year_dict[param] = value_dict

    return year_dict


def _add_all_year_data(graph, current_node_df, current_node):
    # 6 For the remaining data, group by year
    years = [c for c in current_node_df.columns if is_year(c)]          # Get Year Columns
    non_years = [c for c in current_node_df.columns if not is_year(c)]  # Get Non-Year Columns
    non_year_data = [current_node_df[c] for c in non_years]
    for year in years:
        current_year_data = non_year_data + [current_node_df[year]]
        existing_year_dict = _get_current_year_dict(graph, current_node, year)
        updated_year_dict = _update_year_dict(existing_year_dict, current_year_data)
        graph.nodes[current_node][year] = updated_year_dict
        # _build_one_year_data(graph, current_year_data, year, current_node)

    return graph, current_node_df


def _build_cost_curve(graph, current_node_df, current_node):
    comp_type = graph.nodes[current_node][PARAM.competition_type]
    if comp_type in ['supply - cost curve annual', 'supply - cost curve cumulative']:
        cc_func = cost_curves.build_cost_curve_function(current_node_df)
        graph.nodes[current_node][PARAM.cost_curve_function] = cc_func

        # Get rid of cost curve rows
        cost_curve_params = [PARAM.cost_curve_quantity, PARAM.cost_curve_price]
        current_node_df = current_node_df[~current_node_df[COL.parameter].isin(cost_curve_params)]

    return graph, current_node_df


def _copy_graph_and_df(graph, dfs, node, tech=None):
    current_df = dfs[node]
    if tech:
        current_df = current_df[tech]
    current_df = copy.copy(current_df)
    graph = copy.copy(graph)
    return graph, current_df


def _init_node(graph, current_node_df, current_node):
    # 2 Add an empty node to the graph
    graph.add_node(current_node)

    # 2.1 Add index for use in the results viewer file
    if PARAM.tree_index not in graph.nodes[current_node]:
        graph.max_tree_index[0] = max(graph.max_tree_index[0], current_node_df.index[0].item())
        graph.nodes[current_node][PARAM.tree_index] = current_node_df.index[0].item() + graph.cur_tree_index[0]

    return graph, current_node_df


def _standardize_param_value(val):
    if isinstance(val, str):
        return val.lower()
    else:
        return val


def _add_node_constant(graph, node_df, node, parameter, required=False):
    parameter_list = list(node_df[node_df[COL.parameter] == parameter][COL.context])

    if len(set(parameter_list)) == 1:
        parameter_val = _standardize_param_value(parameter_list[0])
    elif len(set(parameter_list)) > 1:
        raise ValueError(f"{parameter} has too many values at {node}.")
    elif parameter in graph.nodes[node]:
        parameter_val = graph.nodes[node][parameter]
    elif required:
        raise ValueError(f"Required {parameter} value not found at {node}")
    else:
        parameter_val = None

    # Add constant to graph
    graph.nodes[node][parameter] = parameter_val

    return graph


def _set_node_constants(graph, current_node_df, current_node):
    # 3.1 is supply
    graph = _add_node_constant(graph, current_node_df, current_node, PARAM.is_supply)
    current_node_df = current_node_df[current_node_df[COL.parameter] != PARAM.is_supply]

    # 3.2 structural aggregation
    graph = _add_node_constant(graph, current_node_df, current_node, PARAM.structural_aggregation)
    current_node_df = current_node_df[current_node_df[COL.parameter] != PARAM.structural_aggregation]

    # 3.3 competition type
    graph = _add_node_constant(graph, current_node_df, current_node, PARAM.competition_type, required=True)
    current_node_df = current_node_df[current_node_df[COL.parameter] != PARAM.competition_type]

    return graph, current_node_df


def _set_node_region_sector(graph, current_node_df, current_node):
    # 4 Set node's region and sector
    region_list = []
    for item in current_node_df[COL.region]:
        if item not in region_list and item != None:
            region_list.append(item)
    try:
        graph.nodes[current_node][COL.region.lower()] = region_list[0]
    except IndexError:
        pass
    current_node_df = current_node_df.drop(columns=[COL.region])

    if graph.nodes[current_node][PARAM.competition_type] not in ['root', 'region']:
        sector_list = []
        for item in current_node_df[COL.sector]:
            if item not in sector_list and item != None:
                sector_list.append(item)
        try:
            graph.nodes[current_node][COL.sector.lower()] = sector_list[0]
        except IndexError:
            pass
    current_node_df = current_node_df.drop(columns=[COL.sector])

    return graph, current_node_df


def _add_node_data(graph, current_node, node_dfs):
    # args and kwargs for including new fields (e.g. LCC, Service Cost, etc)
    """ Add and populate a new node to `graph`

    Parameters
    ----------
    current_node : str
        The name of the node (branch notation) to add.

    Returns
    -------
    networkx.Graph
        `graph` with `node` added, along with its associated data.
    """
    # 1 Copy the current graph & the current node's dataframe
    graph, current_node_df = _copy_graph_and_df(graph, node_dfs, current_node)

    # 2 Add an empty node to the graph
    graph, current_node_df = _init_node(graph, current_node_df, current_node)

    # 3 Set node constants
    graph, current_node_df = _set_node_constants(graph, current_node_df, current_node)

    # 4 Set node's region and sector
    graph, current_node_df = _set_node_region_sector(graph, current_node_df, current_node)

    # 5 Find the cost curve function
    graph, current_node_df = _build_cost_curve(graph, current_node_df, current_node)

    # 6 For the remaining data, group by year.
    graph, current_node_df = _add_all_year_data(graph, current_node_df, current_node)

    # 7 Return the new graph
    return graph


def _add_all_year_data_for_tech(graph, current_tech_df, node, current_tech):
    years = [c for c in current_tech_df.columns if is_year(c)]             # Get Year Columns
    non_years = [c for c in current_tech_df.columns if not is_year(c)]     # Get Non-Year Columns

    non_year_data = [current_tech_df[c] for c in non_years]
    for year in years:
        year_data_to_update = non_year_data + [current_tech_df[year]]
        existing_year_dict = _get_current_year_dict(graph, node, year, tech=current_tech)
        updated_year_dict = _update_year_dict(existing_year_dict, year_data_to_update)

        # Add technologies key (to the node's data) if needed
        if PARAM.technologies not in graph.nodes[node][year].keys():
            graph.nodes[node][year][PARAM.technologies] = {}

        # Add the technology specific data for that year
        graph.nodes[node][year][PARAM.technologies][current_tech] = updated_year_dict

        # Add index for use in the results viewer file
        if PARAM.tree_index not in graph.nodes[node][year][PARAM.technologies][current_tech]:
            graph.nodes[node][year][PARAM.technologies][current_tech][PARAM.tree_index] = current_tech_df.index[0].item() + graph.cur_tree_index[0]

    return graph, current_tech_df


def _init_tech(graph, current_tech_df):
    graph.max_tree_index[0] = max(graph.max_tree_index[0], current_tech_df.index[0].item())
    current_tech_df = current_tech_df[current_tech_df[COL.parameter] != COL.technology.lower()]
    return graph, current_tech_df


def _add_tech_data(graph, node, tech_dfs, current_tech):
    """
    Add and populate a new technology to `node`'s data within`graph`
    Parameters
    ----------
    node : str
        The name of the node the new technology data will reside in.
    tech : str
        The name of the technology being added to the graph.
    Returns
    -------
    networkx.Graph
        `graph` with the data for `tech` contained within `node`'s node data

    """
    # 1 Copy the current graph & the current tech's dataframe
    graph, current_tech_df = _copy_graph_and_df(graph, tech_dfs, node, current_tech)

    # 2. Prepare to add the technology
    graph, current_tech_df = _init_tech(graph, current_tech_df)

    # 3 Group data by year & add to the tech's dictionary
    # NOTE: This is very similar to what we do for nodes (above). However, it differs because
    # we aren't using the value column (its redundant here). TODO: Check
    graph, current_tech_df = _add_all_year_data_for_tech(graph, current_tech_df, node, current_tech)

    # 4 Return the new graph
    return graph


def make_or_update_nodes(graph, node_dfs, tech_dfs):
    """
    Add nodes to `graph` using `node_dfs` and `tech_dfs`. If node already exists, update it.

    Returns
    -------
    networkx.Graph
        An updated graph that contains all nodes and technologies in node_dfs and tech_dfs.
    """
    # 1 Copy graph
    new_graph = copy.copy(graph)

    # 2 Add nodes to the graph
    # The strategy of trying to add a node to the graph, and then adding it to a "to add" list
    # if that doesn't work, deals with the possibility that nodes may be defined out of order.
    node_dfs_to_add = list(node_dfs.keys())
    while len(node_dfs_to_add) > 0:
        node_data = node_dfs_to_add.pop(0)
        try:
            new_graph = _add_node_data(graph, node_data, node_dfs)
        except KeyError:
            node_dfs_to_add.append(node_data)

    # 3 Add technologies to the graph
    for node in tech_dfs:
        # Add technologies key to node data
        for tech in tech_dfs[node]:
            new_graph = _add_tech_data(graph, node, tech_dfs, tech)

    # Return the graph
    return new_graph


