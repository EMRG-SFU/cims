"""
This module contains the functions for LCC Calculations for the CIMS model.
"""
import warnings

from .emissions import calc_complete_emissions_cost, calc_financial_emissions_cost, \
    calc_cumul_emissions_cost_rate
from . import utils
from .revenue_recycling import calc_recycled_revenues
from .cost_curves import calc_cost_curve_lcc
from .vintage_weighting import calculate_vintage_weighted_parameter


def lcc_calculation(sub_graph, node, year, model, **kwargs):
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
    if 'lcc_financial' in sub_graph.nodes[node][year]:
        lcc, lcc_source = model.get_param('lcc_financial', node, year,
                                          context=node.split('.')[-1],
                                          return_source=True)  # context is the fuel name
        if lcc_source == 'model':
            # Retrieve the aggregate emissions cost at the node/tech
            calc_cumul_emissions_cost_rate(model, node, year)

            # Calculate Price
            price, price_source = model.get_param('price', node, year, return_source=True,
                                                  do_calc=True)
            val_dict = {'year_value': price, 'param_source': price_source}
            model.set_param_internal(val_dict, 'price', node, year)

            return

    # Check if the node is a tech compete node:
    if model.get_param('competition type', node) in ['tech compete', 'node tech compete', 'market']:
        total_lcc_v = 0.0
        v = model.get_param('heterogeneity', node, year)

        # Get all the technologies in the node
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
            # TODO: Change to Price, knowing that internally the fLCC will be calculated.
            lcc, lcc_source = model.get_param('lcc_financial', node, year, tech=tech,
                                              return_source=True, do_calc=True)
            val_dict = {'year_value': lcc, 'param_source': lcc_source}
            model.set_param_internal(val_dict, 'lcc_financial', node, year, tech)

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
            first_year_unavail = model.get_param('unavailable', node, str(model.base_year),
                                                 tech=tech)
            if first_year_avail <= int(year) < first_year_unavail:
                # Life Cycle Cost ^ -v
                if lcc < 0.01:
                    # When lcc < 0.01, we will approximate its weight using a TREND line
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
            ms, ms_source = model.get_param('market share', node, year, tech=tech,
                                            return_source=True)
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
            curr_lcc = model.get_param('lcc_financial', node, year, tech=tech)
            weighted_lccs += market_share * curr_lcc

        # Maintain LCC for nodes where all techs have zero stock (and therefore no market share)
        # This issue affects endogenous fuels that are not used until later years (like hydrogen)
        if node in model.fuels:
            if weighted_lccs == 0 and int(year) != model.base_year:
                prev_year = str(int(year) - model.step)
                weighted_lccs = model.get_param('lcc_financial', node, prev_year,
                                                context=node.split('.')[-1])

        # Subtract Recycled Revenues
        recycled_revenues = calc_recycled_revenues(model, node, year)
        lcc = weighted_lccs - recycled_revenues

        # Check that stock isn't 0 (GL Issue #110)
        pq, src = model.get_param('provided_quantities', node, year, return_source=True)
        if utils.prev_stock_existed(model, node, year) and (pq is not None) and (
                src == 'calculation') and (pq.get_total_quantity() <= 0):
            lcc = 0

        service_name = node.split('.')[-1]
        sub_graph.nodes[node][year]['lcc_financial'] = {
            service_name: utils.create_value_dict(lcc, param_source='calculation')}

    elif 'cost curve' in model.get_param('competition type', node):
        lcc = calc_cost_curve_lcc(model, node, year,
                                  cost_curve_min_max=kwargs.get('cost_curve_min_max', None))
        service_name = node.split('.')[-1]
        sub_graph.nodes[node][year]['lcc_financial'] = {
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
        fixed_cost_rate = model.get_param('fixed cost rate', node, year, do_calc=True)
        lcc = service_cost + fixed_cost_rate - recycled_revenues

        pq, src = model.get_param('provided_quantities', node, year, return_source=True)
        if utils.prev_stock_existed(model, node, year) and (pq is not None) and (
                src == 'calculation') and (pq.get_total_quantity() <= 0):
            lcc = 0

        service_name = node.split('.')[-1]
        sub_graph.nodes[node][year]['lcc_financial'] = {
            service_name: utils.create_value_dict(lcc, param_source=sc_source)}

    # fLCC -> Price
    price, price_source = model.get_param('price', node, year, return_source=True, do_calc=True)
    val_dict = {'year_value': price, 'param_source': price_source}
    model.set_param_internal(val_dict, 'price', node, year)


def calc_financial_lcc(model: "CIMS.Model", node: str, year: str, tech: str) -> float:
    """
    Calculate the Financial Life Cycle Cost (called 'lcc_financial' in the model & model
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

    # Calculate the LCC of any new stock

    # Upfront Cost - vintage-weight full term
    new_upfront_cost, uc_src = model.get_param('financial upfront cost', node, year, tech=tech,
                                               do_calc=True, return_source=True)
    model.set_param_internal(param='new_stock_financial_upfront_cost', node=node, year=year,
                             tech=tech,
                             val=utils.create_value_dict(new_upfront_cost, param_source=uc_src))
    upfront_cost = calculate_vintage_weighted_parameter('new_stock_financial_upfront_cost', model,
                                                        node, year, tech)

    # Annual Cost - vintage-weight full term
    new_annual_cost, ac_src = model.get_param('financial annual cost', node, year, tech=tech,
                                              do_calc=True, return_source=True)
    model.set_param_internal(utils.create_value_dict(new_annual_cost, param_source=ac_src),
                             'new_stock_financial_annual_cost', node, year, tech=tech)
    annual_cost = calculate_vintage_weighted_parameter('new_stock_financial_annual_cost', model,
                                                       node, year, tech)

    # Annual Service Cost - vintage weight the service requested ratios
    annual_service_cost = model.get_param('financial service cost', node, year, tech=tech,
                                          do_calc=True)

    # Fixed Cost Rate -- No Vintage Weighting
    fixed_cost_rate = model.get_param('fixed cost rate', node, year, tech=tech, do_calc=True)

    # Emissions Cost -- Vintage-weight the emissions ratios, but leave the cost/emission the same
    emissions_cost = calc_financial_emissions_cost(model, node, year, tech, allow_foresight=False)

    # Recycled Revenues -- TODO: vintage weighting
    recycled_revenues = calc_recycled_revenues(model, node, year, tech)

    # Add it all together
    fLCC = upfront_cost + annual_cost + annual_service_cost + fixed_cost_rate + emissions_cost - \
           recycled_revenues

    return fLCC


def calc_complete_lcc(model: "CIMS.Model", node: str, year: str, tech: str) -> float:
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
    complete_upfront_cost = model.get_param('complete upfront cost', node, year, tech=tech,
                                            do_calc=True)
    complete_annual_cost = model.get_param('complete annual cost', node, year, tech=tech,
                                           do_calc=True)
    annual_service_cost = model.get_param('service cost', node, year, tech=tech, do_calc=True)
    fixed_cost_rate = model.get_param('fixed cost rate', node, year, tech=tech, do_calc=True)
    emissions_cost = calc_complete_emissions_cost(model, node, year, tech, allow_foresight=True)

    complete_lcc = complete_upfront_cost + complete_annual_cost + annual_service_cost + \
                   fixed_cost_rate + emissions_cost

    return complete_lcc


def calc_complete_upfront_cost(model: 'CIMS.Model', node: str, year: str, tech: str) -> float:
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
    output = model.get_param('output', node, year, tech=tech)

    complete_uc = (capital_cost + subsidy) / output * crf

    return complete_uc


def calc_financial_upfront_cost(model: 'CIMS.Model', node: str, year: str, tech: str) -> float:
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


def calc_complete_annual_cost(model: 'CIMS.Model', node: str, year: str, tech: str) -> float:
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
    operating_maintenance = model.get_param('fom', node, year, tech=tech)
    fixed_intangible_cost = model.get_param('fic', node, year, tech=tech)
    declining_intangible_cost = model.get_param('dic', node, year, tech=tech, do_calc=True)
    output = model.get_param('output', node, year, tech=tech)

    complete_ac = (
                              operating_maintenance + fixed_intangible_cost + declining_intangible_cost) / output

    return complete_ac


def calc_financial_annual_cost(model: 'CIMS.Model', node: str, year: str, tech: str) -> float:
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
    operating_maintenance = model.get_param('fom', node, year, tech=tech)
    output = model.get_param('output', node, year, tech=tech)

    financial_ac = operating_maintenance / output

    return financial_ac


def calc_capital_cost(model: 'CIMS.Model', node: str, year: str, tech: str) -> float:
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
    float : Capital cost, defined as CC = max{CC_declining, cc_fixed * CC_declining_limit}.
    """
    dcc_class = model.get_param('dcc_class', node, year, tech=tech, context='context')

    if dcc_class is None:
        capital_cost = model.get_param('fcc', node, year, tech=tech)
    else:
        cc_declining = model.get_param('capital cost_declining', node, year, tech=tech,
                                       do_calc=True)
        cc_fixed = model.get_param('fcc', node, year, tech=tech)
        cc_declining_limit = model.get_param('dcc_limit', node, year, tech=tech)

        capital_cost = max(cc_declining, cc_fixed * cc_declining_limit)

    return capital_cost


def calc_crf(model: 'CIMS.Model', node: str, year: str, tech: str) -> float:
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

    # Check if the node has an exogenously defined capital recovery, if not use lifetime
    try:
        payback_period = model.get_param('capital recovery', node, year, tech=tech,
                                         check_exist=True)
    except:
        payback_period = model.get_param('lifetime', node, year, tech=tech)

    finance_discount = model.get_param('discount rate_financial', node, year, tech=tech)
    if finance_discount == 0:
        warnings.warn('Discount rate_financial has value of 0 at {} -- {}'.format(node, tech))
        finance_discount = model.get_parameter_default('discount rate_financial')

    crf = finance_discount / (1 - (1 + finance_discount) ** (-1.0 * payback_period))

    return crf


def calc_financial_annual_service_cost(model: 'CIMS.Model', node: str, year: str,
                                       tech: str = None) -> float:
    """
    """

    def do_sc_calculation(service_requested):
        service_requested_value = calculate_vintage_weighted_parameter('service requested',
                                                                       model, node, year,
                                                                       tech, serv)
        service_cost = 0

        if service_requested['branch'] in model.fuels:
            fuel_branch = service_requested['branch']

            fuel_price = model.get_param('price', fuel_branch, year, do_calc=True)

            # Price Multiplier
            try:
                fuel_name = fuel_branch.split('.')[-1]
                price_multiplier = model.graph.nodes[node][year]['price multiplier'][fuel_name][
                    'year_value']
            except KeyError:
                price_multiplier = 1
            service_requested_price = fuel_price * price_multiplier

        else:
            service_requested_branch = service_requested['branch']
            if 'price' in model.graph.nodes[service_requested_branch][year]:
                service_requested_price = model.get_param('price', service_requested_branch, year)
            else:
                # Encountering a non-visited node
                service_requested_price = 1

        service_cost += service_requested_price * service_requested_value
        return service_cost

    total_service_cost = 0
    graph = model.graph

    if tech is not None:
        data = graph.nodes[node][year]['technologies'][tech]
    else:
        data = graph.nodes[node][year]

    if 'service requested' in data:
        service_req = data['service requested']
        for serv, req in service_req.items():
            total_service_cost += do_sc_calculation(req)

    return total_service_cost


def calc_complete_annual_service_cost(model: 'CIMS.Model', node: str, year: str,
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
            fuel_name = fuel_branch.split('.')[-1]

            fuel_price = model.get_param('price', fuel_branch, year, do_calc=True)

            # Price Multiplier
            try:
                fuel_name = fuel_branch.split('.')[-1]
                price_multiplier = model.graph.nodes[node][year]['price multiplier'][fuel_name][
                    'year_value']
            except KeyError:
                price_multiplier = 1

            service_requested_price = fuel_price * price_multiplier

        else:
            service_requested_branch = service_requested['branch']
            if 'price' in model.graph.nodes[service_requested_branch][year]:
                service_requested_price = model.get_param('price', service_requested_branch, year)
            else:
                # Encountering a non-visited node
                service_requested_price = 1

        service_cost += service_requested_price * service_requested_value
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


def calc_price_subsidy(model: 'CIMS.Model', node: str, year: str, tech=None):
    """
    Calculates the price_subsidy for a node in a given year.
    Price subsidy is the sum of benchmark * tax across all ghg/emission type combinations.

    Parameters
    ----------
    model : CIMS.Model
        The model of interest.
    node : str
        The node (in branch form) whose price subsidy is being calculated.
    year : str
        The year for which price subsidy is being calculated
    tech : str, optional
        This parameter only exists to enable this function to work with the general get_param()
        function. If tech is provided, an error will be raised.

    Returns
    -------
    float :
        The price subsidy for a particular node in a given year.
    """
    if tech is not None:
        raise ValueError('Cannot calculate price subsidy for a technology.')

    price_subsidy = 0
    benchmark = model.get_param('benchmark', node, year, dict_expected=True)

    if type(benchmark) is dict:
        tax = model.get_param('tax', node, year, dict_expected=True)
        for ghg in benchmark:
            for emission_type in benchmark[ghg]:
                try:
                    tax_value = tax[ghg][emission_type]['year_value']
                except KeyError:
                    tax_value = 0

                benchmark_value = benchmark[ghg][emission_type]['year_value']
                price_subsidy += benchmark_value * tax_value

    return price_subsidy


def calc_price(model, node, year, tech=None):
    """
    Calculates the Price of a node or technology.

    When a COP or P2000 is exogenously defined for a node, price is calculated based off of the
    node's fLCC, COP, P2000, and Additional costs. Otherwise, the node's price will be its fLCC.

    A technology's price will always be its fLCC.

    Parameters
    ----------
    model : CIMS.Model
        The model to retrieve data from & save the calculated price to.
    node : str
        The node (in branch form) whose price is being calculated.
    year : str
        The year for which we are calculating price.
    tech : str, optional
        If specified, the technology whose price is being retrieved.

    Returns
    -------
    None

    This function has the side effect of setting the node/year's "price" and "non-energy cost" (if
    not exogenously defined) parameters in the model. If calculating price for a base year,
    P2000, and COP are also set (if they weren't exogenously defined).
    """
    service = node.split('.')[-1]
    fLCC = model.get_param('lcc_financial', node, year, tech=tech, context=service)

    if tech is not None:
        price = fLCC
        model.set_param_internal(utils.create_value_dict(price, param_source='calculation'),
                                 'price', node, year, tech=tech)
        return price

    base_year = str(model.base_year)
    p2000, p2000_source = model.get_param('p2000', node, base_year, return_source=True)
    p2000_exogenous = (p2000 is not None) & (p2000_source == 'model')
    cop, cop_source = model.get_param('cop', node, base_year, return_source=True)
    cop_exogenous = (cop is not None) & (cop_source == 'model')

    if year == base_year:
        if p2000_exogenous:
            p2000, p2000_source = max([(p2000, 'model'), (fLCC + 0.01, 'calculation')],
                                      key=lambda x: x[0])
            cop, cop_source = fLCC / p2000, 'calculation'
            price = p2000
            non_energy_cost = price - fLCC
        elif cop_exogenous:
            cop, cop_source = min([(cop, 'model'), (fLCC / (fLCC + 0.01), 'calculation')],
                                  key=lambda x: x[0])
            cop = max(0.01, cop)
            p2000, p2000_source = fLCC / cop, 'calculation'
            price = p2000
            non_energy_cost = price - fLCC
        else:
            p2000, p2000_source = 0, 'calculation'
            cop, cop_source = 0, 'calculation'
            price = fLCC
            non_energy_cost = 0

        # Set parameters
        model.set_param_internal(utils.create_value_dict(p2000, param_source=p2000_source), 'p2000',
                                 node, year)
        model.set_param_internal(utils.create_value_dict(cop, param_source=cop_source), 'cop', node,
                                 year)
        model.set_param_internal(
            utils.create_value_dict(non_energy_cost, param_source='calculation'),
            'non-energy cost', node, year)

    else:
        # Find Base Year Values
        non_energy_cost_2000 = model.get_param('non-energy cost', node, base_year)
        fLCC_2000 = model.get_param('lcc_financial', node, base_year, context=service)

        # Current Year Values
        fLCC = model.get_param('lcc_financial', node, year, context=service)
        non_energy_cost = model.get_param('non-energy cost', node, year)

        # Calculate Price
        if p2000_exogenous or cop_exogenous:
            price = p2000 * (
                    fLCC / fLCC_2000 * cop + non_energy_cost / non_energy_cost_2000 * (1 - cop))
        else:
            non_energy_cost = 0
            model.set_param_internal(
                utils.create_value_dict(non_energy_cost, param_source='calculation'),
                'non-energy cost', node, year)
            price = fLCC

    # Set parameters
    price_subsidy = model.get_param('price_subsidy', node, year, do_calc=True)
    model.set_param_internal(utils.create_value_dict(price_subsidy, param_source='calculation'),
                             'price_subsidy', node, year)

    price = price - price_subsidy
    model.set_param_internal(utils.create_value_dict(price, param_source='calculation'),
                             'price', node, year)

    return price


def calc_fixed_cost_rate(model, node, year, tech=None):
    """
    Calculate the fixed cost rate for a node or technology. This is the total fixed cost (which is
    provided exogenously) divided by the total quantity being provided by the node or technology.

    Note: This parameter is not weighted by stock vintage.

    Parameters
    ----------
    model : CIMS.Model
        The model to retrieve data from & save the result to.
    node : str
        The node (in branch form) whose fixed cost rate is being calculated.
    year : str
        The year in which to calculated the fixed cost rate.
    tech : str, optional
        If specified, the technology whose fixed cost rate is being calculated.

    Returns
    -------
    float:
        The fixed cost rate calculated for the node or technology. If total fixed cost isn't
        specified in the model, the fixed cost rate returned is 0. Additionally, if a total fixed
        cost is specified for the node/tech in the model, the model will be updated with the
        calculated fixed cost rate.

    """
    total_fixed_cost = model.get_param('total fixed cost', node, year, tech=tech)
    if total_fixed_cost is not None:
        if tech:
            warnings.warn(
                "This function was not intended for use with technologies. While we won't stop"
                "you from using it for that purpose, you are doing so at your own risk. You"
                "may want to consider re-organizing your model to avoid this case.")

        if int(year) == model.base_year:
            fixed_cost_rate = 0
            fixed_cost_rate_dict = utils.create_value_dict(year_val=fixed_cost_rate,
                                                           param_source='default')
        else:
            prov_quant_object, src = model.get_param('provided_quantities', node, year,
                                                     return_source=True)
            prov_quant = prov_quant_object.get_total_quantity()

            if tech and prov_quant != 0:
                total_market_share = model.get_param('total_market_share', node, year, tech=tech)
                prov_quant = prov_quant * total_market_share

            if src == 'initialization':
                prev_year = str(int(year) - model.step)
                prov_quant = model.get_param('provided_quantities', node,
                                             prev_year).get_total_quantity()
                if prov_quant > 0:
                    fixed_cost_rate = total_fixed_cost / prov_quant
                else:
                    fixed_cost_rate = 0
                fixed_cost_rate_dict = utils.create_value_dict(year_val=fixed_cost_rate,
                                                               param_source='calculation')
            else:
                prov_quant = prov_quant_object.get_total_quantity()

                if tech and prov_quant != 0:
                    total_market_share = model.get_param('total_market_share', node, year, tech=tech)
                    prov_quant = prov_quant * total_market_share


                else:
                    fixed_cost_rate = total_fixed_cost / prov_quant
                    fixed_cost_rate_dict = utils.create_value_dict(year_val=fixed_cost_rate,
                                                                   param_source='calculation')

        model.set_param_internal(fixed_cost_rate_dict, 'fixed cost rate', node, year, tech=tech)

    else:
        fixed_cost_rate = 0

    return fixed_cost_rate
