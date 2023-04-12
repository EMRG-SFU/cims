"""
Tax Foresight functionality. Used to calculate emissions cost for complete LCC.
"""
from __future__ import annotations  # For Type Hinting
import networkx
from numpy import linspace, mean

from . import graph_utils


def initialize_tax_foresight(model: 'pyCIMS.Model') -> None:
    """
    Update model to populate the tax_foresight parameter for all nodes within sectors where a tax
    foresight method has been specified.

    Parameters
    ----------
    model : The model to be updated.

    Returns
    -------
    Updates model to include tax_foresight values for all nodes in all specified sectors.

    """
    # Find all sectors in model
    all_sectors = [node for node, data in model.graph.nodes(data=True)
                   if 'sector' in data['competition type'].lower()]

    for year in model.years:
        # Find the tax foresight methods defined in a particular year
        foresight_dict = model.get_param('tax_foresight', 'pyCIMS', year, dict_expected=True)

        # Record the tax foresight methods for specified sectors
        if foresight_dict is not None:
            for sector in foresight_dict:
                for node in all_sectors:
                    if node.split('.')[-1] == sector:
                        model.graph.nodes[node][year]['tax_foresight'] = foresight_dict[sector]

        # Pass foresight methods down to all other nodes in a sector
        graph_utils.top_down_traversal(model.graph,
                                       _inherit_tax_foresight,
                                       year)


def _inherit_tax_foresight(graph: networkx.DiGraph, node: str, year: str) -> None:
    """
    Updates node's tax_foresight parameter, based on the tax_foresight value of its sector. For use
    with a top-down traversal function.
    Parameters
    ----------
    graph : The graph containing node & its parents.
    node : The node whose tax_foresight method will be updated.
    year : The year whose tax_foresight method will be updated.

    Returns
    -------
    If tax_foresight has been specified for node's sector than node's tax_foresight parameter
    will be updated to match.
    """
    if 'tax_foresight' not in graph.nodes[node][year]:
        parents = list(graph.predecessors(node))
        parent_dict = {}
        if len(parents) > 0:
            parent = parents[0]
            if 'tax_foresight' in graph.nodes[parent][year] and parent != 'pyCIMS':
                parent_dict = graph.nodes[parent][year]['tax_foresight']
        if parent_dict:
            graph.nodes[node][year]['tax_foresight'] = parent_dict


def discounted_foresight(model: 'pyCIMS.Model', node: str, year: str, tech: str or None, ghg: str,
                         emission_type: str) -> float:
    """
    Use the "Discounted Tax Foresight" method to calculates an expected tax value for a given
    node/tech, ghg, & emission_type in a specified year. This function is called from
    emissions.calc_emissions_cost() during the calculation of complete life cycle cost.

    Parameters
    ----------
    model : The model containing the information to calculate the expected tax.
    node : The node whose expected tax is being calculated.
    year : The year for which expected tax is being calculated.
    tech : The technology whose expected tax is being calculated.
    ghg : The greenhouse gas whose expected tax is being calculated.
    emission_type : The emission type whose expected tax is being calculated.

    Returns
    -------
    A value for expected tax, calculated using the discounted tax foresight method, that can be
    used to calculate emissions cost.
    """
    # Interpolate tax values for any new stock's lifetime
    future_tax_values = _tax_foresight_interpolation(
        model, node, year, tech, ghg, emission_type
    )

    # Calculate expected tax based on future tax values, discounting taxes in future years.
    lifetime = int(model.get_param('lifetime', node, year, tech=tech))
    r_k = model.get_param('discount rate_financial', node, year)

    expected_tax = sum(
        (tax / (1 + r_k) ** (n - int(year) + 1)
         for tax, n in zip(future_tax_values, range(int(year), int(year) + lifetime)))
    )

    expected_tax *= r_k / (1 - (1 + r_k) ** (-lifetime))

    return expected_tax


def average_foresight(model, node, year, tech, ghg, emission_type):
    """
    Use the "Average Tax Foresight" method to calculates an expected tax value for a given
    node/tech, ghg, & emission_type in a specified year. This function is called from
    emissions.calc_emissions_cost() during the calculation of complete life cycle cost.

    Parameters
    ----------
    model : The model containing the information to calculate the expected tax.
    node : The node whose expected tax is being calculated.
    year : The year for which expected tax is being calculated.
    tech : The technology whose expected tax is being calculated
    ghg : The greenhouse gas whose expected tax is being calculated
    emission_type : The emission type whose expected tax is being calculated

    Returns
    -------
    A value for expected tax, calculated using the average tax foresight method, that can be
    used to calculate emissions cost.
    """
    # Interpolate tax values for any new stock's lifetime
    future_tax_values = _tax_foresight_interpolation(
        model, node, year, tech, ghg, emission_type
    )

    # Calculate expected tax by finding the average value of future tax values
    expected_tax = mean(future_tax_values)

    return expected_tax


def _tax_foresight_interpolation(model, node, year, tech, ghg, emission_type):
    """
    Use interpolation to calculate the expected tax in each year of a node/tech's lifetime,
    based on the taxes defined in the model.

    Parameters
    ----------
    model : pyCIMS.Model
        The model to retrieve paramter values from.
    node : str
        The node whose expected tax cost is being found.
    year : str
        The year whose expected tax cost is being found.
    tech : str or None
        The technology whose expected tax cost is being found.
    ghg : str
        The greenhouse gas whose expected tax cost is being found.
    emission_type: str
        The emission type whose expected tax cost is being found.

    Returns
    -------
    List [float] :
        A list of floats, each representing the interpolated tax for a year of the
        node/tech's lifetime.

    """
    lifetime = int(model.get_param('lifetime', node, year, tech=tech))

    tax_vals = []
    for year_n in range(int(year), int(year) + lifetime, model.step):
        if str(year_n) <= max(model.years):
            cur_tax = model.get_param('tax', node, str(year_n),
                                      context=ghg, sub_context=emission_type)
        else:  # when current year is out of range
            cur_tax = model.get_param('tax', node, max(model.years),
                                      context=ghg, sub_context=emission_type)
        if str(year_n + model.step) <= max(model.years):
            next_tax = model.get_param('tax', node, str(year_n + model.step),
                                       context=ghg, sub_context=emission_type)
        else:  # when future year(s) are out of range
            next_tax = cur_tax
        tax_vals.extend(linspace(cur_tax, next_tax, model.step, endpoint=False))

    return tax_vals
