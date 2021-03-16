import warnings
from . import utils
import math


# TODO: See if I can get rid of this function or replace with a Model.set_param() method
def add_tech_param(g, node, year, tech, param, value=0.0, source=None, unit=None):
    """
    Include a new set of parameter: {values} as a nested dict

    """
    g.nodes[node][year]["technologies"][tech].update({str(param): {"year_value": value,
                                                                   "branch": str(node),
                                                                   "source": source,
                                                                   "unit": unit}})


def lcc_calculation(sub_graph, node, year, model, show_warnings=False):
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
    if 'Life Cycle Cost' in sub_graph.nodes[node][year]:
        pass
    # Check if the node is a tech compete node:
    elif model.get_param('competition type', node) == "tech compete":
        total_lcc_v = 0.0
        v = model.get_param('Heterogeneity', node, year)

        # Get all of the technologies in the node
        node_techs = sub_graph.nodes[node][year]["technologies"].keys()

        # For every tech in the node, retrieve or compute required economic values
        for tech in node_techs:
            # calculate_tech_econ_values(model.graph, node, tech, year)

            # If the technology is available in this year, go through it
            first_year_available = model.get_param('Available', node, str(model.base_year), tech)
            first_year_unavailable = model.get_param('Unavailable', node, str(model.base_year), tech)
            if first_year_available <= int(year) < first_year_unavailable:
                # Service Cost
                # ************
                annual_service_cost = model.get_param('Service cost', node, year, tech)
                # TODO: replace with a Model.set_param() function (similar to Model.get_param())
                tech_data = model.graph.nodes[node][year]["technologies"][tech]
                if 'Service cost' in tech_data:
                    tech_data['Service cost']['year_value'] = annual_service_cost
                else:
                    add_tech_param(model.graph, node, year, tech, 'Service cost', annual_service_cost)

                # CRF
                # ************
                crf = model.get_param("CRF", node, year, tech)
                # TODO: replace with a Model.set_param() function (similar to Model.get_param())
                tech_data = model.graph.nodes[node][year]["technologies"][tech]
                if 'CRF' in tech_data:
                    tech_data['CRF']['year_value'] = crf
                else:
                    add_tech_param(model.graph, node, year, tech, 'CRF', crf)

                # LCC
                # ************
                lcc = model.get_param('Life Cycle Cost', node, year, tech)
                # TODO: replace with a Model.set_param() function (similar to Model.get_param())
                tech_data = model.graph.nodes[node][year]["technologies"][tech]
                if 'Life Cycle Cost' in tech_data:
                    tech_data['Life Cycle Cost']['year_value'] = lcc
                else:
                    add_tech_param(model.graph, node, year, tech, 'Life Cycle Cost', lcc)

                # Life Cycle Cost ^ -v
                # ********************
                if round(lcc, 20) == 0:
                    if show_warnings:
                        warnings.warn('Life Cycle Cost has value of 0 at {} -- {}'.format(node, tech))
                    lcc = 0.0001

                if lcc < 0:
                    if show_warnings:
                        warnings.warn('Life Cycle Cost has negative value at {} -- {}'.format(node, tech))
                    lcc = 0.0001

                try:
                    lcc_neg_v = lcc ** (-1.0 * v)
                    total_lcc_v += lcc_neg_v
                except OverflowError as e:
                    raise e

        # Set sum of Life Cycle Cost raised to negative variance
        sub_graph.nodes[node][year]["total_lcc_v"] = total_lcc_v

        # Weighted Life Cycle Cost
        # ************************
        weighted_lccs = 0
        # For every tech, use a exogenous or previously calculated market share to calculate Life
        # Cycle Cost
        for tech in node_techs:
            # Determine whether Market share is exogenous or not
            exo_market_share = sub_graph.nodes[node][year]['technologies'][tech]['Market share']['year_value']
            exogenous = exo_market_share is not None
            sub_graph.nodes[node][year]['technologies'][tech]['Market share']['exogenous'] = exogenous

            # Determine what market share to use for weighing Life Cycle Costs
            # If market share is exogenous, set new & total market share to exogenous value
            if exogenous:
                sub_graph.nodes[node][year]['technologies'][tech]['new_market_share'] = exo_market_share
                sub_graph.nodes[node][year]['technologies'][tech]['total_market_share'] = exo_market_share

            market_share = model.get_param('total_market_share', node, year, tech)

            # Weight Life Cycle Cost and Add to Node Total
            # ********************************************
            # find the years where the tech is available
            first_year_available = model.get_param('Available', node, str(model.base_year), tech)
            first_year_unavailable = model.get_param('Unavailable', node, str(model.base_year), tech)
            if first_year_available <= int(year) < first_year_unavailable:
                curr_lcc = model.get_param('Life Cycle Cost', node, year, tech)
                weighted_lccs += market_share * curr_lcc

        fuel_name = node.split('.')[-1]
        sub_graph.nodes[node][year]["Life Cycle Cost"] = {fuel_name: utils.create_value_dict(weighted_lccs)}

    else:
        # When calculating a service cost for a technology or node using the "Fixed Ratio" decision
        # rule, multiply the Life Cycle Costs of the service required by its "Service Requested"
        # line value. Sometimes, the Service Requested line values act as percent shares that add up
        # to 1 for a given fixed ratio decision node. Other times, they do not and the Service
        # Requested Line values sum to numbers greater or less than 1.
        service_cost = model.get_param('Service cost', node, year)
        # Is service cost just the cost of these nodes?
        fuel_name = node.split('.')[-1]
        sub_graph.nodes[node][year]["Life Cycle Cost"] = {fuel_name: utils.create_value_dict(service_cost)}


def calc_lcc(model, node, year, tech):
    upfront_cost = model.get_param('Upfront cost', node, year, tech)
    annual_cost = model.get_param('Annual cost', node, year, tech)
    annual_service_cost = model.get_param('Service cost', node, year, tech)
    lcc = upfront_cost + annual_cost + annual_service_cost
    return lcc


def calc_upfront_cost(model, node, year, tech):
    crf = model.get_param("CRF", node, year, tech)
    capital_cost = model.get_param('Capital cost', node, year, tech)
    fixed_uic = model.get_param('Upfront intangible cost_fixed', node, year, tech)
    declining_uic = calc_declining_uic(model, node, year, tech)
    output = model.get_param('Output', node, year, tech)

    uc = (capital_cost +
          fixed_uic +
          declining_uic)/output * crf

    return uc


def calc_annual_cost(model, node, year, tech):
    output = model.get_param('Output', node, year, tech)
    operating_maintenance_cost = model.get_param('Operating and maintenance cost', node, year, tech)
    fixed_aic = model.get_param('Annual intangible cost_fixed', node, year, tech)
    declining_aic = model.get_param('Annual intangible cost_declining', node, year, tech)

    ac = (operating_maintenance_cost +
          fixed_aic +
          declining_aic) / output
    return ac


def calc_capital_cost(model, node, year, tech):
    cc_overnight = model.get_param('Capital cost_overnight', node, year, tech)
    declining_cc_limit = model.get_param('Capital cost_declining_limit', node, year, tech)
    declining_cc = model.get_param("Capital cost_declining", node, year, tech)

    if declining_cc is None:
        cc = cc_overnight
    else:
        cc = max(declining_cc, cc_overnight*declining_cc_limit)
    return cc


def calc_declining_cc(model, node, year, tech):
    dcc_class = model.get_param('Capital cost_declining_Class', node, year, tech, sub_param='value')

    if dcc_class is None:
        cc_declining = None

    else:
        # Progress Ratio
        progress_ratio = model.get_param('Capital cost_declining_Progress Ratio', node, year, tech)
        gcc_t = model.get_param('GCC_t', node, year, tech)  # capital cost adjusted for cumulative stock in all other countries

        # Cumulative New Stock summed over all techs in DCC Class
        dcc_class_techs = techs_in_dcc_class(model, dcc_class, year)
        cns_sum = 0
        for node_k, tech_k in dcc_class_techs:
            cns_k = model.get_param('Capital cost_declining_cumulative new stock', node_k, year, tech_k)
            cns_sum += cns_k

        # New Stock summed over all techs in DCC class and over all previous years
        # (excluding base year)
        dcc_class_techs = techs_in_dcc_class(model, dcc_class, year)
        ns_sum = 0
        for node_k, tech_k in dcc_class_techs:
            year_list = [str(x) for x in range(int(model.base_year)+int(model.step), int(year), int(model.step))]
            for j in year_list:
                ns_jk = model.get_param('new_stock', node_k, j, tech_k)
                ns_sum += ns_jk

        # Calculate Declining Capital Cost
        inner_sums = (cns_sum + ns_sum) / cns_sum
        cc_declining = gcc_t * (inner_sums ** math.log(progress_ratio, 2))

    return cc_declining


def calc_gcc(model, node, year, tech):
    previous_year = str(int(year) - model.step)
    if previous_year in model.years:
        aeei = model.get_param('Capital cost_declining_AEEI', node, year, tech)
        gcc = ((1 - aeei) ** model.step) * \
              calc_gcc(model, node, previous_year, tech)
    else:
        cc_overnight = model.get_param('Capital cost_overnight', node, year, tech)
        gcc = cc_overnight

    return gcc


def calc_declining_uic(model, node, year, tech):
    # Retrieve Exogenous Terms from Model Description
    initial_uic = model.get_param('Upfront intangible cost_declining_initial', node, year, tech)
    rate_constant = model.get_param('Upfront intangible cost_declining_rate', node, year, tech)
    shape_constant = model.get_param('Upfront intangible cost_declining_shape', node, year, tech)

    # Calculate Declining UIC
    if int(year) == int(model.base_year):
        return_uic = initial_uic
    else:
        prev_year = str(int(year) - model.step)
        prev_nms = model.get_param('new_market_share', node, prev_year, tech)

        denominator = 1 + shape_constant * math.exp(rate_constant * prev_nms)
        uic_declining = initial_uic / denominator
        return_uic = uic_declining

    return return_uic


def calc_declining_aic(sub_graph, node, tech, year, model):
    # Retrieve Exogenous Terms from Model Description
    initial_aic = model.get_param('Annual intangible cost_declining_initial', node, year, tech)
    rate_constant = model.get_param('Annual intangible cost_declining_rate', node, year, tech)
    shape_constant = model.get_param('Annual intangible cost_declining_shape', node, year, tech)

    # Calculate Declining AIC
    if int(year) == int(model.base_year):
        return_val = initial_aic
    else:
        prev_year = str(int(year) - model.step)
        prev_nms = model.get_param('new_market_share', node, prev_year, tech)

        denominator = 1 + shape_constant * math.exp(rate_constant * prev_nms)
        aic_declining = initial_aic / denominator
        return_val = aic_declining

    return return_val


def calc_declining_aic(model, node, year, tech):
    # Retrieve Exogenous Terms from Model Description
    initial_aic = model.get_param('Annual intangible cost_declining_initial', node, year, tech)
    rate_constant = model.get_param('Annual intangible cost_declining_rate', node, year, tech)
    shape_constant = model.get_param('Annual intangible cost_declining_shape', node, year, tech)

    # Calculate Declining AIC
    if int(year) == int(model.base_year):
        return_val = initial_aic
    else:
        prev_year = str(int(year) - model.step)
        prev_nms = model.get_param('new_market_share', node, prev_year, tech)

        denominator = 1 + shape_constant * math.exp(rate_constant * prev_nms)
        aic_declining = initial_aic / denominator
        return_val = aic_declining

    return return_val


def calc_crf(model, node, year, tech):
    finance_discount = model.get_param('Discount rate_Financial', node, year, tech)
    lifespan = model.get_param('Lifetime', node, year, tech)

    if finance_discount == 0:
        warnings.warn('Discount rate_Financial has value of 0 at {} -- {}'.format(node, tech))
        finance_discount = model.get_tech_parameter_default('Discount rate_Financial')

    crf = finance_discount / (1 - (1 + finance_discount) ** (-1.0 * lifespan))

    return crf


def calc_annual_service_cost(model, node, year, tech=None):
    """
    Find the service cost associated with a given technology.

    1. For each service being requested:
            i) If the service is a fuel, find the fuel price (Life Cycle Cost) and add it to the
               service cost. If the fuel doesn't have a fuel price,
           ii) Otherwise, use the service's Life Cycle Cost which was calculated already.
    2. Return the service cost (currently assumes that there can only be one
    """
    def do_sc_calculation(service_requested):
        service_requested_value = service_requested['year_value']
        service_cost = 0
        if service_requested['branch'] in model.fuels:
            fuel_branch = service_requested['branch']

            if 'Life Cycle Cost' in model.graph.nodes[fuel_branch][year]:
                fuel_name = list(model.graph.nodes[fuel_branch][year]['Life Cycle Cost'].keys())[0]
                service_requested_lcc = model.graph.nodes[fuel_branch][year]['Life Cycle Cost'][fuel_name]['year_value']
            else:
                service_requested_lcc = model.get_node_parameter_default('Life Cycle Cost', 'sector')
            try:
                price_multiplier = model.graph.nodes[node][year]['Price Multiplier'][fuel_name]['year_value']
            except KeyError:
                price_multiplier = 1
            service_requested_lcc *= price_multiplier
        else:
            service_requested_branch = service_requested['branch']
            if 'Life Cycle Cost' in model.graph.nodes[service_requested_branch][year]:
                service_requested_lcc = model.get_param('Life Cycle Cost', service_requested_branch, year)
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

    if 'Service requested' in data:
        service_req = data['Service requested']
        if isinstance(service_req, dict):
            if 'year_value' in service_req:
                total_service_cost += do_sc_calculation(service_req)
            else:
                for req in service_req.values():
                    total_service_cost += do_sc_calculation(req)
        elif isinstance(service_req, list):
            for req in service_req:
                total_service_cost += do_sc_calculation(req)
        else:
            print(f"type for service requested? {type(service_req)}")

    return total_service_cost


def techs_in_dcc_class(model, dcc_class, year):
    tech_list = []
    nodes = model.graph.nodes
    for node in nodes:
        if 'technologies' in nodes[node][year]:
            for tech in nodes[node][year]['technologies']:
                dccc = model.get_param('Capital cost_declining_Class', node, year, tech, sub_param='value')
                if dccc == dcc_class:
                    tech_list.append((node, tech))
    return tech_list
