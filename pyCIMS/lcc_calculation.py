"""
This module contains the functions for LCC Calculations for the pyCIMS model.
"""
import warnings
import math
from .emissions import calc_emissions_cost, calc_per_unit_emissions_cost
from . import utils
from .revenue_recycling import calc_recycled_revenues


def lcc_calculation(sub_graph, node, year, model):
    """
    Determines economic parameters for `node` in `year` and stores the values in the sub_graph
    at the appropriate node. Specifically,

    Determines the node's:
    * Total Life Cycle Cost (weighted using total market share across all technologies)
    * Sum of Life Cycle Costs raised to the negative variance

    Determines each of the node's technology's:
    * Service cost
    * CRF
    * Full capital cost
    * Life Cycle Cost

    Initializes new_market_share and total_market_share for technologies where market share
    is exogenously defined.

    Parameters
    ----------
    sub_graph : NetworkX.Graph
        The subgraph where parameters will be stored.

    node : str
        The name of the node whose parameters we are calculating.

    year : str
        The year for which we are calculating parameters.

    Returns
    -------
        None. Produces side effects of updating the node in sub_graph to have parameter values.
    """
    # Check if the node has an exogenously defined Life Cycle Cost
    if 'life cycle cost' in sub_graph.nodes[node][year]:
        lcc, lcc_source = model.get_param('life cycle cost', node, year, context=node.split('.')[-1],
                                          return_source=True)  # context is the fuel name
        if lcc_source == 'model':
            # Retrieve the aggregate emissions cost at the node/tech
            calc_per_unit_emissions_cost(model, node, year)
            return

    # Check if the node is a tech compete node:
    if model.get_param('competition type', node) in ['tech compete', 'node tech compete', 'market']:
        total_lcc_v = 0.0
        v = model.get_param('heterogeneity', node, year)

        # Get all of the technologies in the node
        node_techs = sub_graph.nodes[node][year]['technologies'].keys()

        # For every tech in the node, retrieve or compute required economic values
        for tech in node_techs:
            # Service Cost
            # ************
            annual_service_cost, sc_source = model.get_param('service cost', node, year, tech=tech,
                                                             return_source=True, do_calc=True)
            val_dict = {'year_value': annual_service_cost,
                        "branch": str(node),
                        'param_source': sc_source}
            model.set_param_internal(val_dict, 'service cost', node, year, tech)

            # CRF
            # ************
            crf, crf_source = model.get_param('crf', node, year, tech=tech,
                                              return_source=True, do_calc=True)
            val_dict = {'year_value': crf, 'param_source': crf_source}
            model.set_param_internal(val_dict, 'crf', node, year, tech)

            # LCC (financial)
            # ************
            lcc, lcc_source = model.get_param('life cycle cost', node, year, tech=tech,
                                              return_source=True, do_calc=True)
            val_dict = {'year_value': lcc, 'param_source': lcc_source}
            model.set_param_internal(val_dict, 'life cycle cost', node, year, tech)

            # Complete LCC
            # ************
            complete_lcc, complete_lcc_source = model.get_param('complete life cycle cost',
                                                                node, year, tech=tech,
                                                                return_source=True,
                                                                do_calc=True)
            val_dict = {'year_value': complete_lcc, 'param_source': complete_lcc_source}
            model.set_param_internal(val_dict, 'complete life cycle cost', node, year, tech)

            # If the technology is available in this year, add to the total LCC^-v value.
            first_year_avail = model.get_param('available', node, str(model.base_year), tech=tech)
            first_year_unavail = model.get_param('unavailable', node, str(model.base_year), tech=tech)
            if first_year_avail <= int(year) < first_year_unavail:
                # Life Cycle Cost ^ -v
                if lcc < 0.01:
                    # When lcc < 0.01, we will approximate it's weight using a TREND line
                    w1 = 0.1 ** (-1 * v)
                    w2 = 0.01 ** (-1 * v)
                    slope = (w2 - w1) / (0.01 - 0.1)
                    weight = slope * lcc + (w1 - slope * 0.1)
                else:
                    weight = lcc ** (-1 * v)

                total_lcc_v += weight

        # Set sum of Life Cycle Cost raised to negative variance
        val_dict = utils.create_value_dict(total_lcc_v, param_source='calculation')
        sub_graph.nodes[node][year]["total_lcc_v"] = val_dict

        # Weighted Life Cycle Cost
        # ************************
        weighted_lccs = 0
        # For every tech, use a exogenous or previously calculated market share to calculate Life
        # Cycle Cost
        for tech in node_techs:
            # Determine whether Market share is exogenous or not
            ms, ms_source = model.get_param('market share', node, year, tech=tech, return_source=True)
            ms_exogenous = ms_source == 'model'

            # Determine what market share to use for weighing Life Cycle Costs
            # If market share is exogenous, set new & total market share to exogenous value
            if ms_exogenous:
                val_dict = utils.create_value_dict(ms, param_source='calculation')
                model.set_param_internal(val_dict, 'new_market_share', node, year, tech)
                model.set_param_internal(val_dict, 'total_market_share', node, year, tech)

            market_share = model.get_param('total_market_share', node, year, tech=tech)

            # Weight Life Cycle Cost and Add to Node Total
            # ********************************************
            curr_lcc = model.get_param('life cycle cost', node, year, tech=tech)
            weighted_lccs += market_share * curr_lcc

        # Subtract Recycled Revenues
        recycled_revenues = calc_recycled_revenues(model, node, year)
        lcc = weighted_lccs - recycled_revenues

        service_name = node.split('.')[-1]
        sub_graph.nodes[node][year]['life cycle cost'] = {
            service_name: utils.create_value_dict(lcc, param_source='calculation')}

    elif 'cost curve' in model.get_param('competition type', node):
        lcc = calc_cost_curve_lcc(model, node, year)
        service_name = node.split('.')[-1]
        sub_graph.nodes[node][year]['life cycle cost'] = {
            service_name: utils.create_value_dict(lcc, param_source='cost curve function')}

    else:
        # When calculating a service cost for a technology or node using the "Fixed Ratio" decision
        # rule, multiply the Life Cycle Costs of the service required by its 'service requested'
        # line value. Sometimes, the Service Requested line values act as percent shares that add up
        # to 1 for a given fixed ratio decision node. Other times, they do not and the Service
        # Requested Line values sum to numbers greater or less than 1.
        service_cost, sc_source = model.get_param('service cost', node, year,
                                                  return_source=True, do_calc=True)
        recycled_revenues = calc_recycled_revenues(model, node, year)
        lcc = service_cost - recycled_revenues

        service_name = node.split('.')[-1]
        sub_graph.nodes[node][year]['life cycle cost'] = {
            service_name: utils.create_value_dict(lcc, param_source=sc_source)}


def calc_cost_curve_lcc(model: "pyCIMS.Model", node: str, year: str):
    """
    Calculate a node's LCC using its cost curve function (stored in the node level data).
    Depending on the node's competition type, annual or cumulative provided quantity values will be
    used in the call to the cost curve interpolation function.

    Note, cost curve LCC is only used for fuels, so this LCC is the same as the financial LCC we use
    for other nodes.

    Parameters
    ----------
    model : The model containing node.
    node : The name of the node for which LCC  will be calculated.
    year : The year to calculate LCC for.

    Returns
    -------
    float : LCC (financial) calculated from the node's cost curve function.
    """
    comp_type = model.get_param('competition type', node).str.lower()
    if comp_type == 'fuel - cost curve annual':
        min_year = year
        max_year = year
    elif comp_type == 'fuel - cost curve cumulative':
        min_year = model.base_year
        max_year = year
    else:
        raise ValueError("Unrecognized cost curve calculation competition type")

    quantity = calc_cost_curve_quantity(model, node, min_year, max_year)
    cc_func = model.get_param('cost_curve_function', node)
    lcc = float(cc_func(quantity))

    return lcc


def calc_cost_curve_quantity(model: "pyCIMS.Model", node: str, min_year: str, max_year: str):
    """
    Calculate the total quantity provided by node from min_year to max_year (inclusive).
    This serves as a helper function for calc_cost_curve_lcc.

    Parameters
    ----------
    model : The model containing node.
    node : The name of the node for which total quantity will be calculated.
    min_year : The first year to include in the sum of total quantities.
    max_year : The last year to include in the sum of total quantities.

    Returns
    -------
    float : Total quantity provided by node from min_year to max_year (inclusive).
    """
    total_quantity = 0
    for year in range(int(min_year), int(max_year) + 1, model.step):
        if 'provided_quantities' in model.graph.nodes[node][str(year)]:
            year_provided_quant = model.get_param('provided_quantities', node, str(year))
            total_quantity += year_provided_quant.get_total_quantity()
    return total_quantity


def calc_financial_lcc(model: "pyCIMS.Model", node: str, year: str, tech: str) -> float:
    """
    Calculate the Financial Life Cycle Cost (called 'life cycle cost' in the model & model
    description). This LCC does not contain intangible costs.

    Parameters
    ----------
    model : The model containing component parts of financial LCC.
    node : The name of the node for which financial LCC will be calculated.
    year : The year to calculate financial LCC for.
    tech : The technology to calculate financial LCC for.

    Returns
    -------
    float : The financial LCC, calculated as fLCC = fUC + fAC + SC + EC

    See Also
    --------
    calc_complete_lcc: Calculates complete LCC, which includes intangible costs.
    """
    upfront_cost = model.get_param('financial upfront cost', node, year, tech=tech, do_calc=True)
    annual_cost = model.get_param('financial annual cost', node, year, tech=tech, do_calc=True)
    annual_service_cost = model.get_param('service cost', node, year, tech=tech, do_calc=True)
    emissions_cost = calc_emissions_cost(model, node, year, tech, allow_foresight=False)
    recycled_revenues = calc_recycled_revenues(model, node, year, tech)
    lcc = upfront_cost + annual_cost + annual_service_cost + emissions_cost - recycled_revenues
    return lcc


def calc_complete_lcc(model: "pyCIMS.Model", node: str, year: str, tech: str) -> float:
    """
    Calculate Complete Life Cycle Cost. This LCC includes intangible costs.

    Parameters
    ----------
    model : The model containing component parts of complete LCC.
    node : The name of the node for which complete LCC will be calculated.
    year : The year to calculate complete LCC for.
    tech : The technology to calculate complete LCC for.

    Returns
    -------
    float : The complete LCC, calculated as cLCC = cUC + cAC + SC + EC

    See Also
    --------
    calc_financial_lcc: Calculates financial LCC, which does not include intangible costs.
    """
    complete_upfront_cost = model.get_param('complete upfront cost', node, year, tech=tech, do_calc=True)
    complete_annual_cost = model.get_param('complete annual cost', node, year, tech=tech, do_calc=True)
    annual_service_cost = model.get_param('service cost', node, year, tech=tech, do_calc=True)
    emissions_cost = calc_emissions_cost(model, node, year, tech, allow_foresight=True)

    complete_lcc = complete_upfront_cost + complete_annual_cost + annual_service_cost + \
                   emissions_cost

    return complete_lcc




def calc_complete_upfront_cost(model: 'pyCIMS.Model', node: str, year: str, tech: str) -> float:
    """
    Calculates complete upfront cost, which includes intangible costs.

    Parameters
    ----------
    model : The model containing component parts of complete upfront cost.
    node : The node to calculate complete upfront cost for.
    year : The year to calculate complete upfront cost for.
    tech : The technology to calculate complete upfront cost for.

    Returns
    -------
    float : The complete upfront cost, defined as
            cUC = (CC - AllocC, + UIC_fixed + UIC_declining) / output * CRF

    See Also
    --------
    calc_financial_upfront_cost
    """
    crf = model.get_param('crf', node, year, tech=tech, do_calc=True)
    capital_cost = model.get_param('capital cost', node, year, tech=tech, do_calc=True)
    subsidy = model.get_param('subsidy', node, year, tech=tech, do_calc=True)
    fixed_uic = model.get_param('uic_fixed', node, year, tech=tech, do_calc=True)
    declining_uic = model.get_param('uic_declining', node, year, tech=tech, do_calc=True)
    output = model.get_param('output', node, year, tech=tech)

    complete_uc = (capital_cost + subsidy + fixed_uic + declining_uic) / output * crf

    return complete_uc


def calc_financial_upfront_cost(model: 'pyCIMS.Model', node: str, year: str, tech: str) -> float:
    """
    Calculates financial upfront cost, which does not include intangible costs.

    Parameters
    ----------
    model : The model containing component parts of financial upfront cost.
    node : The node to calculate financial upfront cost for.
    year : The year to calculate financial upfront cost for.
    tech : The technology to calculate financial upfront cost for.

    Returns
    -------
    float : The financial upfront cost, defined as fUC = (CC - AllocC) / output * CRF

    See Also
    --------
    calc_complete_upfront_cost
    """
    crf = model.get_param('crf', node, year, tech=tech, do_calc=True)
    capital_cost = model.get_param('capital cost', node, year, tech=tech, do_calc=True)
    subsidy = model.get_param('subsidy', node, year, tech=tech)
    output = model.get_param('output', node, year, tech=tech)

    financial_uc = (capital_cost + subsidy) / output * crf

    return financial_uc


def calc_complete_annual_cost(model: 'pyCIMS.Model', node: str, year: str, tech: str) -> float:
    """
    Calculates complete annual cost, which includes intangible costs.

    Parameters
    ----------
    model : The model containing component parts of complete annual cost.
    node : The node to calculate complete annual cost for.
    year : The year to calculate complete annual cost for.
    tech : The technology to calculate complete annual cost for.

    Returns
    -------
    float : The complete annual cost, defined as cAC = (OM + AIC_fixed + AIC_declining) / output

    See Also
    --------
    calc_financial_annual_cost
    """
    operating_maintenance = model.get_param('operating and maintenance', node, year, tech=tech)
    fixed_aic = model.get_param('aic_fixed', node, year, tech=tech)
    declining_aic = model.get_param('aic_declining', node, year, tech=tech, do_calc=True)
    output = model.get_param('output', node, year, tech=tech)

    complete_ac = (operating_maintenance +
                   fixed_aic +
                   declining_aic) / output

    return complete_ac


def calc_financial_annual_cost(model: 'pyCIMS.Model', node: str, year: str, tech: str) -> float:
    """
    Calculates financial annual cost, which includes intangible costs.

    Parameters
    ----------
    model : The model containing component parts of financial annual cost.
    node : The node to calculate financial annual cost for.
    year : The year to calculate financial annual cost for.
    tech : The technology to calculate financial annual cost for.

    Returns
    -------
    float : The financial annual cost, defined as fAC = OM / output

    See Also
    --------
    calc_complete_annual_cost
    """
    operating_maintenance = model.get_param('operating and maintenance', node, year, tech=tech)
    output = model.get_param('output', node, year, tech=tech)

    financial_ac = operating_maintenance / output

    return financial_ac


def calc_capital_cost(model: 'pyCIMS.Model', node: str, year: str, tech: str) -> float:
    """
    Calculates capital cost.

    Parameters
    ----------
    model : The model containing component parts of capital cost.
    node : The node to calculate capital cost for.
    year : The year to calculate capital cost for.
    tech : The technology to calculate capital cost for.

    Returns
    -------
    float : Capital cost, defined as CC = max{CC_declining, CC_overnight * CC_declining_limit}.
    """
    dcc_class = model.get_param('dcc_class', node, year, tech=tech, context='context')

    if dcc_class is None or int(year) == int(model.base_year):
        capital_cost = model.get_param('capital cost_overnight', node, year, tech=tech)
    else:
        capital_cost = model.get_param('capital cost_declining', node, year, tech=tech, do_calc=True)

    return capital_cost


def calc_declining_cc(model: 'pyCIMS.Model', node: str, year: str, tech: str) -> float:
    """
    Calculates declining capital cost.

    Parameters
    ----------
    model : The model containing component parts of declining capital cost.
    node : The node to calculate declining capital cost for.
    year : The year to calculate declining capital cost for.
    tech : The technology to calculate declining capital cost for.

    Returns
    -------
    float : The declining capital cost.
    """
    dcc_class = model.get_param('dcc_class', node, year, tech=tech, context='context')
    dcc_class_techs = model.dcc_classes[dcc_class]

    cc_overnight = model.get_param('capital cost_overnight', node, year, tech=tech)
    cc_declining_limit = model.get_param('dcc_limit', node, year, tech=tech)

    proven_stock = model.get_param('dcc_proven stock', node, year, tech=tech)
    # For transportation, 'dcc_proven stock' already given in vkt, so no need to convert

    bs_sum = 0
    ns_sum = 0

    for node_k, tech_k in dcc_class_techs:
        # Need to convert stocks for transportation techs to common vkt unit
        unit_convert = model.get_param('load factor', node_k, str(model.base_year), tech=tech_k)
        if unit_convert is None:
            unit_convert = 1

        # Base Stock summed over all techs in DCC class (base year only)
        bs_k = model.get_param('base_stock', node_k, str(model.base_year), tech=tech_k)
        if bs_k is not None:
            bs_sum += bs_k / unit_convert

        # New Stock summed over all techs in DCC class and over all previous years (excluding base year)
        year_list = [str(x) for x in range(int(model.base_year) + int(model.step), int(year), int(model.step))]
        for j in year_list:
            ns_jk = model.get_param('new_stock', node_k, j, tech=tech_k)
            ns_sum += ns_jk / unit_convert

    # Capital cost adjusted for global R&D
    gcc_t = model.get_param('GCC_t', node, year, tech=tech, do_calc=True)

    # Calculate Declining Capital Cost
    if (bs_sum + ns_sum) < proven_stock:
        cc_declining = max(gcc_t, cc_overnight * cc_declining_limit)
    else:
        inner_sums = (bs_sum + ns_sum) / proven_stock
        progress_ratio = model.get_param('dcc_progress ratio', node, year, tech=tech)
        log_decline = gcc_t * (inner_sums ** math.log(progress_ratio, 2))
        
        cc_declining = max(log_decline, cc_overnight * cc_declining_limit)

    return cc_declining


def calc_gcc(model: 'pyCIMS.Model', node: str, year: str, tech: str) -> float:
    """
    Calculate GCC, which is the capital cost adjusted for global R&D.

    Parameters
    ----------
    model : The model containing component parts of GCC.
    node : The node to calculate GCC for.
    year : The year to calculate GCC for.
    tech : The technology to calculate GCC for.

    Returns
    -------
    float : The GCC. If year is the base year, GCC = CC_overnight. Otherwise,
            GCC_t = GCC_(t-5) * (1 - AEEI)^5.
    """
    if int(year) == int(model.base_year):
        gcc = model.get_param('capital cost_overnight', node, year, tech=tech)
    else:
        previous_year = str(int(year) - model.step)
        aeei = model.get_param('dcc_aeei', node, year, tech=tech)
        gcc = ((1 - aeei) ** model.step) * calc_gcc(model, node, previous_year, tech=tech)

    return gcc


def calc_declining_uic(model: 'pyCIMS.Model', node: str, year: str, tech: str) -> float:
    """
    Calculate Upfront Declining Intangible Cost (UIC_declining).

    Parameters
    ----------
    model : The model containing component parts of declining UIC.
    node : The node to calculate declining UIC for.
    year : The year to calculate declining UIC for.
    tech : The technology to calculate declining UIC for.

    Returns
    -------
    float : The declining UIC.
    """
    # Retrieve Exogenous Terms from Model Description
    initial_uic = model.get_param('uic_declining_initial', node, year, tech=tech)
    rate_constant = model.get_param('uic_declining_rate', node, year, tech=tech)
    shape_constant = model.get_param('uic_declining_shape', node, year, tech=tech)

    # Calculate Declining UIC
    if int(year) == int(model.base_year):
        return_uic = initial_uic
    else:
        prev_year = str(int(year) - model.step)
        prev_nms = model.get_param('new_market_share', node, prev_year, tech=tech)

        try:
            denominator = 1 + shape_constant * math.exp(rate_constant * prev_nms)
        except OverflowError as overflow:
            print(node, year, shape_constant, rate_constant, prev_nms)
            raise overflow

        prev_uic_declining = calc_declining_uic(model, node, prev_year, tech)
        uic_declining = min(prev_uic_declining, initial_uic / denominator)

        return_uic = uic_declining

    return return_uic


def calc_declining_aic(model: 'pyCIMS.Model', node: str, year: str, tech: str) -> float:
    """
    Calculate Annual Declining Intangible Cost (declining AIC).

    Parameters
    ----------
    model : The model containing component parts of declining AIC.
    node : The node to calculate declining AIC for.
    year : The year to calculate declining AIC for.
    tech : The technology to calculate declining AIC for.

    Returns
    -------
    float : The declining AIC.
    """
    # Retrieve Exogenous Terms from Model Description
    initial_aic = model.get_param('aic_declining_initial', node, year, tech=tech)
    rate_constant = model.get_param('aic_declining_rate', node, year, tech=tech)
    shape_constant = model.get_param('aic_declining_shape', node, year, tech=tech)

    # Calculate Declining AIC
    if int(year) == int(model.base_year):
        return_val = initial_aic
    else:
        prev_year = str(int(year) - model.step)
        prev_nms = model.get_param('new_market_share', node, prev_year, tech=tech)

        try:
            denominator = 1 + shape_constant * math.exp(rate_constant * prev_nms)
        except OverflowError as overflow:
            print(node, year, shape_constant, rate_constant, prev_nms)
            raise overflow

        prev_aic_declining = calc_declining_aic(model, node, prev_year, tech)
        aic_declining = min(prev_aic_declining, initial_aic / denominator)

        return_val = aic_declining

    return return_val


def calc_crf(model: 'pyCIMS.Model', node: str, year: str, tech: str) -> float:
    """
    Calculate the Capital Recovery Factor (CRF).

    Parameters
    ----------
    model : The model containing component parts of CRF.
    node : The node to calculate CRF for.
    year : The year to calculate CRF for.
    tech : The technology to calculate CRF for.

    Returns
    -------
    float : The CRF, defined as CRF = discount_rate / (1-(1+discount_rate)^-lifespan)
    """

    finance_discount = model.get_param('discount rate_financial', node, year, tech=tech)
    payback_period = model.get_param('capital recovery', node, year, tech=tech)

    if finance_discount == 0:
        warnings.warn('Discount rate_financial has value of 0 at {} -- {}'.format(node, tech))
        finance_discount = model.get_tech_parameter_default('discount rate_financial')

    crf = finance_discount / (1 - (1 + finance_discount) ** (-1.0 * payback_period))

    return crf


def calc_annual_service_cost(model: 'pyCIMS.Model', node: str, year: str,
                             tech: str = None) -> float:
    """
    Find the service cost associated with a given technology.

    For each service being requested:
        i) If the service is a fuel, find the fuel price (Life Cycle Cost) and add it to the
           service cost.
       ii) Otherwise, use the service's financial Life Cycle Cost (already calculated).

    Parameters
    ----------
    model : The model containing component parts of service cost.
    node : The node to calculate service cost for.
    year : The year to calculate service cost for.
    tech : The technology to calculate service cost for.

    Returns
    -------
    float : The service cost, defined as SC = SUM_over_services(fLCC * request_amount)
    """

    def do_sc_calculation(service_requested):
        service_requested_value = service_requested['year_value']
        service_cost = 0
        if service_requested['branch'] in model.fuels:
            fuel_branch = service_requested['branch']

            if 'life cycle cost' in model.graph.nodes[fuel_branch][year]:
                fuel_name = list(model.graph.nodes[fuel_branch][year]['life cycle cost'].keys())[0]
                service_requested_lcc = \
                    model.graph.nodes[fuel_branch][year]['life cycle cost'][fuel_name]['year_value']
            else:
                service_requested_lcc = model.get_node_parameter_default('life cycle cost',
                                                                         'sector')
            try:
                fuel_name = fuel_branch.split('.')[-1]
                price_multiplier = model.graph.nodes[node][year]['price multiplier'][fuel_name][
                    'year_value']
            except KeyError:
                price_multiplier = 1
            service_requested_lcc *= price_multiplier
        else:
            service_requested_branch = service_requested['branch']
            if 'life cycle cost' in model.graph.nodes[service_requested_branch][year]:
                service_name = service_requested_branch.split('.')[-1]
                service_requested_lcc = \
                    model.graph.nodes[service_requested_branch][year]['life cycle cost'][
                        service_name]['year_value']
            else:
                # Encountering a non-visited node
                service_requested_lcc = 1

        service_cost += service_requested_lcc * service_requested_value

        return service_cost

    total_service_cost = 0
    graph = model.graph

    if tech is not None:
        data = graph.nodes[node][year]['technologies'][tech]
    else:
        data = graph.nodes[node][year]

    if 'service requested' in data:
        service_req = data['service requested']
        for req in service_req.values():
            total_service_cost += do_sc_calculation(req)

    return total_service_cost
