import warnings
from . import utils
from . import econ
import math


def calculate_tech_econ_values(graph, node, tech, year):
    # Initialize Values that need to be calculated
    tech_param_names = ["Service cost", "CRF", "Life Cycle Cost", "Full capital cost"]
    tech_param_values = [0.0, 0.0, 0.0, 0.0]
    for param_name, param_val in zip(tech_param_names, tech_param_values):
        present_tech_params = graph.nodes[node][year]['technologies'][tech].keys()
        if param_name not in present_tech_params:
            add_tech_param(graph, node, year, tech, param_name, param_val)

    # Initialize values not provided exogenously
    def_param_names = ['Operating and maintenance cost', 'Output']
    def_param_values = [0, 1]
    for param_name, param_val in zip(def_param_names, def_param_values):
        if graph.nodes[node][year]["technologies"][tech][param_name]["year_value"] is None:
            graph.nodes[node][year]["technologies"][tech][param_name]["year_value"] = param_val


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
    elif sub_graph.nodes[node]["competition type"] == "tech compete":
        total_lcc_v = 0.0
        v = model.get_param('Heterogeneity', node, year)
        # Get all of the technologies in the node
        node_techs = sub_graph.nodes[node][year]["technologies"].keys()

        # For every tech in the node, retrieve or compute required economic values
        for tech in node_techs:
            calculate_tech_econ_values(model.graph, node, tech, year)

            # If the technology is available in this year, go through it
            # (range is [lower year, upper year + 1] to work with range function
            lower_year, upper_year = utils.range_available(model.graph, node, tech)

            if int(year) in range(lower_year, upper_year):
                # Service Cost
                # ************
                annual_service_cost = econ.get_technology_service_cost(sub_graph,
                                                                       node,
                                                                       year,
                                                                       tech,
                                                                       model)
                sub_graph.nodes[node][year]["technologies"][tech]["Service cost"]["year_value"] = annual_service_cost

                # CRF
                # ************
                crf = econ.get_crf(sub_graph,
                                   node,
                                   year,
                                   tech,
                                   model)
                sub_graph.nodes[node][year]["technologies"][tech]["CRF"]["year_value"] = crf

                # Capital Cost
                # ************
                tech_data = sub_graph.nodes[node][year]['technologies'][tech]

                # Find overnight capital cost
                cc_overnight = tech_data['Capital cost_overnight']['year_value']
                if cc_overnight is None:
                    cc_overnight = model.get_tech_parameter_default('Capital cost_overnight')

                # Find declining limit
                declining_cc_limit = tech_data['Capital cost_declining_limit']['year_value']
                if declining_cc_limit is None:
                    declining_cc_limit = model.get_tech_parameter_default('Capital cost_declining_limit')

                # Find Declining Capital Cost
                declining_cc = calc_declining_cc(sub_graph,
                                                 node,
                                                 year,
                                                 tech,
                                                 model)

                cap_cost = calc_capital_cost(declining_cc, cc_overnight, declining_cc_limit)

                # Life Cycle Cost
                # *****************
                fixed_uic = sub_graph.nodes[node][year]['technologies'][tech]['Upfront intangible cost_fixed']['year_value']
                if fixed_uic is None:
                    fixed_uic = model.get_tech_parameter_default('Upfront intangible cost_fixed')

                declining_uic = calc_declining_uic(sub_graph,
                                                   node,
                                                   tech,
                                                   year,
                                                   model)

                output = sub_graph.nodes[node][year]['technologies'][tech]['Output']['year_value']

                upfront_cost = calc_upfront_cost(cap_cost,
                                                 fixed_uic,
                                                 declining_uic,
                                                 output,
                                                 crf)

                operating_maintenance_cost = sub_graph.nodes[node][year]['technologies'][tech]['Operating and maintenance cost']['year_value']

                fixed_aic = sub_graph.nodes[node][year]['technologies'][tech]['Annual intangible cost_fixed']['year_value']
                if fixed_aic is None:
                    fixed_aic = model.get_tech_parameter_default('Annual intangible cost_fixed')

                declining_aic = calc_declining_aic(sub_graph,
                                                   node,
                                                   tech,
                                                   year,
                                                   model)

                annual_cost = calc_annual_cost(operating_maintenance_cost,
                                               fixed_aic,
                                               declining_aic,
                                               output)

                lcc = calc_lcc(upfront_cost, annual_cost, annual_service_cost)

                sub_graph.nodes[node][year]["technologies"][tech]["Life Cycle Cost"]["year_value"] = lcc

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
                sub_graph.nodes[node][year]['technologies'][tech][
                    'new_market_share'] = exo_market_share
                sub_graph.nodes[node][year]['technologies'][tech][
                    'total_market_share'] = exo_market_share
                market_share = exo_market_share

            # If market share is not exogenous, but was calculated in a previous iteration for
            # this year, use total market share for calculating Life Cycle Cost
            elif 'total_market_share' in sub_graph.nodes[node][year]['technologies'][tech].keys():
                market_share = sub_graph.nodes[node][year]['technologies'][tech]['total_market_share']

            # If market share is not exogenous and hasn't been calculated in a previous year, use
            # the total market share calculated in the previous year for calculating Life Cycle Cost
            else:
                previous_year = str(int(year) - model.step)
                market_share = sub_graph.nodes[node][previous_year]['technologies'][tech]['total_market_share']

            if market_share is None:
                if show_warnings:
                    warnings.warn("Market Share is NONE!!")

            # Weight Life Cycle Cost and Add to Node Total
            # ********************************************
            # find the years where the tech is available
            low, up = utils.range_available(sub_graph, node, tech)
            if int(year) in range(low, up):
                curr_lcc = sub_graph.nodes[node][year]["technologies"][tech]["Life Cycle Cost"]["year_value"]
                weighted_lccs += market_share * curr_lcc

        fuel_name = node.split('.')[-1]
        sub_graph.nodes[node][year]["Life Cycle Cost"] = {fuel_name: utils.create_value_dict(weighted_lccs)}

    elif ('technologies' in sub_graph.nodes[node].keys()) and (sub_graph.nodes[node]['competition_type']=='Fixed Ratio'):
        print("{} is fixed ratio w/ techs".format(node))
    else:
        # When calculating a service cost for a technology or node using the "Fixed Ratio" decision
        # rule, multiply the Life Cycle Costs of the service required by its "Service Requested"
        # line value. Sometimes, the Service Requested line values act as percent shares that add up
        # to 1 for a given fixed ratio decision node. Other times, they do not and the Service
        # Requested Line values sum to numbers greater or less than 1.
        service_cost = econ.get_node_service_cost(sub_graph,
                                                  model.graph,
                                                  node,
                                                  year,
                                                  model.fuels)

        # Is service cost just the cost of these nodes?
        fuel_name = node.split('.')[-1]
        sub_graph.nodes[node][year]["Life Cycle Cost"] = {fuel_name: utils.create_value_dict(service_cost)}


def calc_lcc(upfront_cost, annual_cost, annual_service_cost):
    lcc = upfront_cost + annual_cost + annual_service_cost
    return lcc


def calc_upfront_cost(capital_cost, fixed_uic, declining_uic, output, crf):
    uc = (capital_cost +
          fixed_uic +
          declining_uic)/output * crf
    return uc


def calc_annual_cost(operating_maintenance_cost, fixed_aic, declining_aic, output):
    ac = (operating_maintenance_cost +
          fixed_aic +
          declining_aic) / output
    return ac


def calc_capital_cost(declining_cc, overnight_cc, declining_limit=1):
    if declining_cc is None:
        cc = overnight_cc
    else:
        cc = max(declining_cc, overnight_cc*declining_limit)
    return cc


def calc_declining_cc(sub_graph, node, year, tech, model):
    tech_data = sub_graph.nodes[node][year]['technologies'][tech]
    dcc_class = tech_data['Capital cost_declining_Class']['value']

    if dcc_class is None:
        cc_declining = None

    else:
        # Progress Ratio
        progress_ratio = tech_data['Capital cost_declining_Progress Ratio']['year_value']
        if progress_ratio is None:
            progress_ratio = model.get_tech_parameter_default('Capital cost_declining_Progress Ratio')

        # GCC_t
        aeei = tech_data['Capital cost_declining_AEEI']['year_value']
        if aeei is None:
            aeei = model.get_tech_parameter_default('Capital cost_declining_AEEI')
        gcc_t = calc_gcc(sub_graph, node, tech, year, aeei, model)

        # Cumulative New Stock summed over all techs in DCC Class
        dcc_class_techs = techs_in_dcc_class(sub_graph, dcc_class, year)
        cns_sum = 0
        for node_k, tech_k in dcc_class_techs:
            cns_k = sub_graph.nodes[node_k][year]['technologies'][tech_k]['Capital cost_declining_cumulative new stock']['year_value']
            if cns_k is None:
                cns_k = model.get_tech_parameter_default('Capital cost_declining_cumulative new stock')
            cns_sum += cns_k

        # New Stock summed over all techs in DCC class and over all previous years
        # (excluding base year)
        dcc_class_techs = techs_in_dcc_class(sub_graph, dcc_class, year)
        ns_sum = 0
        for node_k, tech_k in dcc_class_techs:
            year_list = [str(x) for x in range(int(model.base_year)+int(model.step), int(year), int(model.step))]
            for j in year_list:
                ns_jk = sub_graph.nodes[node_k][j]['technologies'][tech_k]['new_stock']
                ns_sum += ns_jk

        # Calculate Declining Capital Cost
        inner_sums = (cns_sum + ns_sum) / cns_sum
        cc_declining = gcc_t * (inner_sums ** math.log(progress_ratio, 2))

    return cc_declining


def calc_gcc(sub_graph, node, tech, year, aeei, model):
    previous_year = str(int(year) - model.step)
    if previous_year in sub_graph.nodes[node]:
        gcc = ((1 - aeei) ** model.step) * \
              calc_gcc(sub_graph, node, tech, previous_year, aeei, model)
    else:
        cc_overnight = sub_graph.nodes[node][year]['technologies'][tech]['Capital cost_overnight']['year_value']
        if cc_overnight is None:
            cc_overnight = model.get_tech_parameter_default('Capital cost_overnight')
        gcc = cc_overnight

    return gcc


def calc_declining_uic(sub_graph, node, tech, year, model):
    # Retrieve Exogenous Terms from Model Description
    tech_data = sub_graph.nodes[node][year]['technologies'][tech]

    initial_uic = tech_data['Upfront intangible cost_declining_initial']['year_value']
    if initial_uic is None:
        initial_uic = model.get_tech_parameter_default('Upfront intangible cost_declining_initial')

    rate_constant = tech_data['Upfront intangible cost_declining_rate']['year_value']
    if rate_constant is None:
        rate_constant = model.get_tech_parameter_default('Upfront intangible cost_declining_rate')

    shape_constant = tech_data['Upfront intangible cost_declining_shape']['year_value']
    if shape_constant is None:
        shape_constant = model.get_tech_parameter_default('Upfront intangible cost_declining_shape')

    # Calculate Declining UIC
    if int(year) == int(model.base_year):
        return_uic = initial_uic
    else:
        prev_year = str(int(year) - model.step)
        prev_nms = sub_graph.nodes[node][prev_year]['technologies'][tech]['new_market_share']
        denominator = 1 + shape_constant * math.exp(rate_constant * prev_nms)
        uic_declining = initial_uic / denominator
        return_uic = uic_declining

    return return_uic


def calc_declining_aic(sub_graph, node, tech, year, model):
    # Retrieve Exogenous Terms from Model Description
    tech_data = sub_graph.nodes[node][year]['technologies'][tech]
    initial_aic = tech_data['Annual intangible cost_declining_initial']['year_value']
    if initial_aic is None:
        initial_aic = model.get_tech_parameter_default('Annual intangible cost_declining_initial')
    rate_constant = tech_data['Annual intangible cost_declining_rate']['year_value']
    if rate_constant is None:
        rate_constant = model.get_tech_parameter_default('Annual intangible cost_declining_rate')
    shape_constant = tech_data['Annual intangible cost_declining_shape']['year_value']
    if shape_constant is None:
        shape_constant = model.get_tech_parameter_default('Annual intangible cost_declining_shape')

    # Calculate Declining AIC
    if int(year) == int(model.base_year):
        return_val = initial_aic
    else:
        prev_year = str(int(year) - model.step)
        prev_nms = sub_graph.nodes[node][prev_year]['technologies'][tech]['new_market_share']
        denominator = 1 + shape_constant * math.exp(rate_constant * prev_nms)
        aic_declining = initial_aic / denominator
        return_val = aic_declining

    return return_val


def techs_in_dcc_class(graph, dcc_class, year):
    tech_list = []
    for node in graph.nodes:
        if 'technologies' in graph.nodes[node][year]:
            for tech in graph.nodes[node][year]['technologies']:
                try:
                    dccc = graph.nodes[node][year]['technologies'][tech]['Capital cost_declining_Class']['value']
                except KeyError:
                    dccc = None

                if dccc == dcc_class:
                    tech_list.append((node, tech))
    return tech_list