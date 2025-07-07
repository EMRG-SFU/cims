import warnings

from CIMS import declining_costs, lcc_calculation
from ..graph.query import parent_name
from . import list as PARAM

calculation_directory = {
    PARAM.capital_cost_declining: declining_costs.calc_declining_capital_cost,
    PARAM.capital_cost: lcc_calculation.calc_capital_cost,
    PARAM.crf: lcc_calculation.calc_crf,
    PARAM.financial_upfront_cost: lcc_calculation.calc_financial_upfront_cost,
    PARAM.competition_upfront_cost: lcc_calculation.calc_competition_upfront_cost,
    PARAM.dic: declining_costs.calc_declining_intangible_cost,
    PARAM.financial_annual_cost: lcc_calculation.calc_financial_annual_cost,
    PARAM.competition_annual_cost: lcc_calculation.calc_competition_annual_cost,
    PARAM.service_cost: lcc_calculation.calc_competition_annual_service_cost,
    PARAM.financial_service_cost: lcc_calculation.calc_financial_annual_service_cost,
    PARAM.emissions_cost: lcc_calculation.calc_competition_emissions_cost,
    PARAM.financial_emissions_cost: lcc_calculation.calc_financial_emissions_cost,
    PARAM.lcc_financial: lcc_calculation.calc_financial_lcc,
    PARAM.lcc_competition: lcc_calculation.calc_lcc_competition,
    PARAM.price: lcc_calculation.calc_price,
    PARAM.fixed_cost_rate: lcc_calculation.calc_fixed_cost_rate,
    PARAM.price_subsidy: lcc_calculation.calc_price_subsidy
}

def get_param(model, param, node, year=None, tech=None, context=None, sub_context=None,
              return_source=False, do_calc=False, check_exist=False, dict_expected=False):
    """
    Gets a parameter's value from the model, given a specific context (node, year, tech, context, sub-context),
    calculating the parameter's value if needed.

    This will not re-calculate the parameter's value, but will only retrieve
    values which are already stored in the model or obtained via inheritance, default values,
    or estimation using the previous year's value. If return_source is True, this function will
    also, return how this value was originally obtained (e.g. via calculation)

    Parameters
    ----------
    model : CIMS.Model
        The model containing the parameter value of interest.
    param : str
        The name of the parameter whose value is being retrieved.
    node : str
        The name of the node (branch notation) whose parameter you are interested in retrieving.
    year : str
        The year which you are interested in. `year` must be provided for all parameters stored at
        the technology level, even if the parameter doesn't change year to year.
    tech : str, optional
        The name of the technology you are interested in. `tech` is not required for parameters
        that are specified at the node level. `tech` is required to get any parameter that is
        stored within a technology.
    context : str, optional
        Used when there is context available in the node. Analogous to the `context` column in the model description
    sub_context : str, optional
        Must be used only if context is given. Analogous to the `subcontext` column in the model description
    return_source : bool, default=False
        Whether to return the method by which this value was originally obtained.
    do_calc : bool
        applies calculation formula to calculate the parameter
    check_exist : bool, default=False
        Whether to check that the parameter exists as is given the context (without
        calculation, inheritance, or checking past years)
    dict_expected : bool, default=False
        Used to disable the warning get_param is returning a dict. Get_param should normally return a single value
        (float, str, etc.). If the user knows it expects a dict, then this flag is used.

    Returns
    -------
    float
        The value of the specified `param` at `node`, given the context provided by `year` and
        `tech`.
    str
        If return_source is `True`, will return a string indicating how the parameter's value
        was originally obtained. Can be one of {model, initialization, inheritance, calculation,
        default, or previous_year}.
    """
    val = None
    is_exogenous = None

    # Get Parameter from Description
    # ******************************
    # If the parameter's value is in the model description for that node & year (if the year has
    # been defined), use it.
    data = model.graph.nodes[node]
    if year:
        data = data[year]
        if tech:  # assumption: any tech node always requires a year
            data = data[PARAM.technologies][tech]

    # Val can be the final return result (float, string, etc) or a dict, check for other params
    if param in data:
        val = data[param]
        if isinstance(val, dict):
            if context:
                try:
                    val = val[context]
                    if sub_context:
                        try:
                            val = val[sub_context]
                        except KeyError:
                            val = None
                except KeyError:
                    val = None
            elif None in val:
                val = val[None]

    # Grab the year_value in the dictionary if exists
    if isinstance(val, dict) and (PARAM.year_value in val):
        param_source = val[PARAM.param_source]
        is_exogenous = param_source in ['model', 'initialization']
        val = val[PARAM.year_value]

    # Raise warning if user isn't using get_param correctly
    if isinstance(val, dict) and not dict_expected:
        warning_message = \
            f"get_param() is returning a `dict`, considering using more parameters in get_param().\
                \nParameter: {param if param else ''}\
                \nNode: {node if node else ''}\
                \nYear: {year if year else ''}\
                \nContext: {context if context else ''}\
                \nSub-context: {sub_context if sub_context else ''}\
                \nTech: {tech if tech else ''}"

        warnings.warn(warning_message)

    if val is not None:
        if not do_calc:
            if return_source:
                return val, param_source
            return val
        elif is_exogenous:
            if return_source:
                return val, param_source
            else:
                return val

    # If check_exist is True, raise an Exception if val has not yet been returned, which means
    # the value at the current context could not be found as is.
    if check_exist:
        raise Exception

    param_source = None
    # Calculate Parameter Value
    # ******************************
    # If there is a calculation for the parameter & the arguments for that calculation are present
    # in the model description for that node & year, calculate the parameter value using this
    # calculation.
    if (param in calculation_directory) & do_calc:
        param_calculator = calculation_directory[param]
        val = param_calculator(model, node, year, tech)
        param_source = 'calculation'

    # Inherit Parameter Value
    # ******************************
    # If the value has been defined at a structural parent node for that year, use that value.
    if (param_source is None) and (param in model.inheritable_params):
        if tech:
            try:
                val, source = model.get_param(param, node, year=year, context=context,
                                              sub_context=sub_context, return_source=True)
                assert (source in ['inheritance', 'model', 'default'])
                assert (val is not None)
                param_source = source
            except AssertionError:
                pass
        else:
            # If the value has been defined at a structural ancestor, it should be here with
            # param_source == 'inheritance'
            try:
                val = model.graph.nodes[node][year][param]
                if context:
                    val = val[context]
                    if sub_context:
                        val = val[sub_context]
                if val[PARAM.param_source] == 'inheritance':
                    param_source = 'inheritance'
            except KeyError:
                pass

    # Use a Default Parameter Value
    # ******************************
    # If there is a default value defined, use this value
    if param_source is None:
        if param in model.node_tech_defaults:
            val = model.get_parameter_default(param)
            param_source = 'default'

    # Use Last Year's Value
    # ******************************
    # Otherwise, use the value from the previous year. (If no base year value, throw an error)
    if param_source is None:
        if year is not None:
            prev_year = str(int(year) - model.step)
            if int(prev_year) >= model.base_year:
                val = model.get_param(param, node,
                                      year=prev_year,
                                      context=context,
                                      sub_context=sub_context,
                                      tech=tech,
                                      dict_expected=dict_expected)
                param_source = 'previous_year'
            else:
                val = None
                param_source = None
        else:
            val = None
            param_source = None

    if return_source:
        return val, param_source
    else:
        return val

def is_param_exogenous(model, param, node, year, tech=None):
    """Checks if a parameter is exogenously defined"""
    _, source = model.get_param(param, node, year=year, tech=tech, return_source=True)
    ms_exogenous = source == 'model'
    return ms_exogenous

def _find_value_in_ancestors(graph, node, parameter, year=None):
    """
    Find a parameter's value at a given node or its structural ancestors.

    First attempts to locate a parameter at a given node. If the parameter does not exist at that
    node, a recursive call is made to find the parameter's value at `node`'s parent, if one exists.
    If no parent exists None is returned.

    Parameters
    ----------
    graph : networkx.Graph
        The graph where `node` resides.
    node : str
        The name of the node to begin our search from. Must be contained within
        `graph`. (e.g. `CIMS.Canada.Alberta`)
    parameter : str
        The name of the parameter whose value is being found. (e.g. 
        `Price Multiplier`)
    year : str, optional
        The year associated with sub-dictionary to search at `node`. Default is None, which implies
        that year sub-dictionaries should be searched. Instead, only search for `parameter` in
        `node`s top level data.

    Returns
    -------
    Any
        The value associated with `parameter` if a value can be found at `node` or one of its
        ancestors. Otherwise None
    """
    data = graph.nodes[node]
    parent = parent_name(node, return_empty=True)

    # Look at the Node/Year
    if year:
        year_data = graph.nodes[node][year]
        if parameter in year_data.keys():
            return year_data[parameter]

    # Look at the Node
    if parameter in data.keys():
        return data[parameter]

    # Look at the Parent
    if parent:
        return _find_value_in_ancestors(graph, parent, parameter, year)


def get_ghg_and_emissions(graph, year):
    """
    Return 2 lists consisting of all the GHGs (CO2, CH4, etc.) and all the emission types (Process, Fugitive, etc.)
    Return 1 dictionary containing the GHGs as keys and GWPs as values
    :param DiGraph graph: graph to search for all emissions
    :param str year: year to find emissions, will likely be base year
    :return: list of GHGs and a list of emission types
    """

    ghg = []
    emission_type = []
    gwp = {}
    for node, data in graph.nodes(data=True):

        # Emissions from a node with technologies
        if PARAM.technologies in data[year]:
            techs = data[year][PARAM.technologies]
            for tech in techs:
                tech_data = data[year][PARAM.technologies][tech]
                if PARAM.emissions in tech_data or PARAM.emissions_removal in tech_data:
                    if PARAM.emissions in tech_data:
                        ghg_list = data[year][PARAM.technologies][tech][PARAM.emissions]
                    else:
                        ghg_list = data[year][PARAM.technologies][tech][PARAM.emissions_removal]

                    node_ghg = [ghg for ghg in ghg_list]
                    node_emission_type = [emission_type for emission_record in ghg_list.values() for
                                          emission_type in emission_record]

                    ghg = list(set(ghg + node_ghg))
                    emission_type = list(set(emission_type + node_emission_type))

        # Emissions from a supply node
        elif PARAM.emissions in data[year] or PARAM.emissions_removal in data[year]:
            if PARAM.emissions in data[year]:
                ghg_dict = data[year][PARAM.emissions]
            else:
                ghg_dict = data[year][PARAM.emissions_removal]

            node_ghg = [ghg for ghg in ghg_dict.keys()]

            node_emission_type = [emission_type for emission_record in ghg_dict.values() for
                                  emission_type in emission_record]

            ghg = list(set(ghg + node_ghg))
            emission_type = list(set(emission_type + node_emission_type))

        #GWP from CIMS node
        if PARAM.emissions_gwp in data[year]:
            for ghg2 in data[year][PARAM.emissions_gwp]:
                gwp[ghg2] = data[year][PARAM.emissions_gwp][ghg2][PARAM.year_value]

    return ghg, emission_type, gwp
    
