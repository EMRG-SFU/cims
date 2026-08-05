import warnings
from collections.abc import Iterable

 
from CIMS.emissions import calc_competition_emissions_cost, calc_financial_emissions_cost, calc_emissions_rate_cumul_cost
from CIMS.revenue_recycling import calc_recycled_revenues
from CIMS.cost_curves import calc_cost_curve_lcc
from CIMS.vintage_weighting import calculate_vintage_weighted_parameter
from CIMS.utils import general_utils
from CIMS.utils.parameter import construction, list as PARAM


def lcc_calculation_faster(sub_graph, node, year, model, **kwargs):
    """
    Determines economic parameters for `node` in `year` and stores the values in the sub_graph
    at the appropriate node. Specifically,

    Determines the node's:
    * Total Lifecycle Cost (weighted using total market share across all technologies)
    * Sum of Lifecycle Costs raised to the negative variance

    Determines each of the node's technology's:
    * Service cost
    * CRF
    * Full capital cost
    * Lifecycle Cost

    Initializes market_share_new and market_share_total for technologies where market share
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
    # Check if the node has an exogenously defined Lifecycle Cost
    if PARAM.lcc_financial in sub_graph.nodes[node][year]:
        lcc, lcc_source = model.get_param(PARAM.lcc_financial, node, year, return_source=True)
        if lcc_source == 'model':
            # Retrieve the aggregate emissions cost at the node/tech
            calc_emissions_rate_cumul_cost(model, node, year)

            # Calculate Price
            price, price_source = model.get_param(PARAM.price, node, year, return_source=True,
                                                  do_calc=True)
            val_dict = {PARAM.year_value: price, PARAM.param_source: price_source}
            model.set_param_internal(val_dict, PARAM.price, node, year)

            return

    # Check if the node is a tech compete node:
    if model.get_param(PARAM.competition_type, node) == PARAM.competition_compete:
        #v = model.get_param(PARAM.heterogeneity, node, year)

        # Get all the technologies in the node
        node_techs = sub_graph.nodes[node][year][PARAM.technologies].keys()

        # For every tech in the node, retrieve or compute required economic values
        for tech in node_techs:
            # Service Cost
            # ************
            #annual_service_cost, sc_source = model.get_param(PARAM.service_cost, node, year, tech=tech,
            #                                                 return_source=True, do_calc=True)
            #val_dict = {PARAM.year_value: annual_service_cost,
            #            PARAM.param_source: sc_source}
            #model.set_param_internal(val_dict, PARAM.service_cost, node, year, tech)

            ## CRF
            ## ************
            #crf, crf_source = model.get_param(PARAM.crf, node, year, tech=tech,
            #                                  return_source=True, do_calc=True)
            #val_dict = {PARAM.year_value: crf, PARAM.param_source: crf_source}
            #model.set_param_internal(val_dict, PARAM.crf, node, year, tech)

            ## LCC (financial)
            ## ************
            ## TODO: Change to Price, knowing that internally the fLCC will be calculated.
            #lcc, lcc_source = model.get_param(PARAM.lcc_financial, node, year, tech=tech, return_source=True, do_calc=True)
            #val_dict = {PARAM.year_value: lcc, PARAM.param_source: lcc_source}
            #model.set_param_internal(val_dict, PARAM.lcc_financial, node, year, tech)

            # Competition LCC
            # ************
            lcc_competition, lcc_competition_source = model.get_param(PARAM.lcc_competition,
                                                                node, year, tech=tech,
                                                                return_source=True,
                                                                do_calc=True)
            val_dict = {PARAM.year_value: lcc_competition, PARAM.param_source: lcc_competition_source}
            model.set_param_internal(val_dict, PARAM.lcc_competition, node, year, tech)

        # Weighted Life Cycle Cost
        # ************************
        weighted_lccs = 0
        # For every tech, use an exogenous or previously calculated total market share to calculate Lifeycle Cost
        for tech in node_techs:
            ms_total = model.get_param(PARAM.market_share_total, node, year, tech=tech)

            # Weight Lifecycle Cost and Add to Node Total
            # ********************************************
            curr_lcc = model.get_param(PARAM.lcc_financial, node, year, tech=tech)
            weighted_lccs += ms_total * curr_lcc

        # Maintain LCC for nodes where all techs have zero stock (and therefore no market share)
        # This issue affects endogenous supply_nodes that are not used until later years (like hydrogen) and some sub-trees of demand_nodes
        if weighted_lccs == 0 and int(year) != model.base_year:
            prev_year = str(int(year) - model.step)
            weighted_lccs = model.get_param(PARAM.lcc_financial, node, prev_year)

        # Subtract Recycled Revenues
        revenue_recycled = calc_recycled_revenues(model, node, year)
        lcc = weighted_lccs - revenue_recycled

        # Check that stock isn't 0 (GL Issue #110)
        pq, src = model.get_param(PARAM.provided_quantities, node, year, return_source=True)
        if general_utils.prev_stock_existed(model, node, year) and (pq is not None) and (
                src == 'calculation') and (pq.sum_provided_by_total() <= 0):
            lcc = 0

        sub_graph.nodes[node][year][PARAM.lcc_financial] = construction.create_value_dict(lcc, param_source='calculation')

    elif 'cost curve' in model.get_param(PARAM.competition_type, node):
        lcc = calc_cost_curve_lcc(model, node, year, cost_curve_min_max=kwargs.get('cost_curve_min_max', None))
        sub_graph.nodes[node][year][PARAM.lcc_financial] = construction.create_value_dict(lcc, param_source='cost curve function')

    else:
        # When calculating a service cost for a technology or node using the Fixed Ratio decision
        # rule, multiply the Lifecycle Costs of the service required by its PARAM.service_request
        # line value. Sometimes, the Service Requested line values act as percent shares that add up
        # to 1 for a given fixed ratio decision node. Other times, they do not and the Service
        # Requested Line values sum to numbers greater or less than 1.
        service_cost, sc_source = model.get_param(PARAM.service_cost, node, year,
                                                  return_source=True, do_calc=True)
        revenue_recycled = calc_recycled_revenues(model, node, year)
        fixed_cost_rate = model.get_param(PARAM.fixed_cost_rate, node, year, do_calc=True)
        lcc = service_cost + fixed_cost_rate - revenue_recycled

        pq, src = model.get_param(PARAM.provided_quantities, node, year, return_source=True)
        if general_utils.prev_stock_existed(model, node, year) and (pq is not None) and (src == 'calculation') and (pq.sum_provided_by_total() <= 0):
            lcc = 0

        sub_graph.nodes[node][year][PARAM.lcc_financial] = construction.create_value_dict(lcc, param_source=sc_source)

    # fLCC -> Price
    price, price_source = model.get_param(PARAM.price, node, year, return_source=True, do_calc=True)
    val_dict = {PARAM.year_value: price, PARAM.param_source: price_source}
    model.set_param_internal(val_dict, PARAM.price, node, year)
