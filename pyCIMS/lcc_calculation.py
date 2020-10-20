# All code for the
import warnings
from . import utils
from . import econ


def calculate_tech_econ_values(graph, node, tech, year):
    # Initialize Values that need to be calculated
    tech_param_names = ["Service cost", "CRF", "LCC", "Full capital cost"]
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


def lcc_calculation(sub_graph, node, year, year_step, full_graph, fuels, show_warnings=False):
    """
    Determines economic parameters for `node` in `year` and stores the values in the sub_graph
    at the appropriate node. Specifically,

    Determines the node's:
    * Total LCC (weighted using total market share across all technologies)
    * Sum of LCCs raised to the negative variance

    Determines each of the node's technology's:
    * Service cost
    * CRF
    * Full capital cost
    * LCC

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
    print('\tcalculating LCC for {}'.format(node))
    total_lcc_v = 0.0
    v = econ.get_heterogeneity(sub_graph, node, year)

    # Check if the node is a tech compete node:
    if sub_graph.nodes[node]["competition type"] == "tech compete":
        node_techs = sub_graph.nodes[node][year]["technologies"].keys()

        # For every tech in the node, retrieve or compute required economic values
        for tech in node_techs:
            # calculate_tech_econ_values(sub_graph, node, tech, year)
            calculate_tech_econ_values(full_graph, node, tech, year)

            # If the technology is available in this year, go through it
            # (range is [lower year, upper year + 1] to work with range function
            # low, up = utils.range_available(sub_graph, node, tech)
            low, up = utils.range_available(full_graph, node, tech)

            if int(year) in range(low, up):
                # Service Cost
                # ************
                annual_service_cost = econ.get_technology_service_cost(sub_graph,
                                                                       full_graph,
                                                                       node,
                                                                       year,
                                                                       tech,
                                                                       fuels)
                sub_graph.nodes[node][year]["technologies"][tech]["Service cost"]["year_value"] = annual_service_cost

                # CRF
                # ************
                crf = econ.get_crf(sub_graph,
                                   node,
                                   year,
                                   tech,
                                   finance_base=0.1,
                                   life_base=10.0)
                sub_graph.nodes[node][year]["technologies"][tech]["CRF"]["year_value"] = crf

                # Full Capital Cost
                # *****************
                # will have more terms later
                full_cc = econ.get_capcost(sub_graph,
                                           node,
                                           year,
                                           tech,
                                           crf)
                sub_graph.nodes[node][year]["technologies"][tech]["Full capital cost"][
                    "year_value"] = full_cc

                cc_declining = 0    # Temporary, TODO: remove this line after testing completed
                cc_dec_limit = 1    # Temporary, TODO: remove this line after testing completed
                cc_overnight = sub_graph.nodes[node][year]['technologies'][tech]['Capital cost_overnight']['year_value'] # TODO: Implement defaults
                if cc_overnight is None:
                    cc_overnight = 0
                capital_cost = calc_capital_cost(cc_declining, cc_overnight, cc_dec_limit)

                # LCC
                # *****************
                declining_upfront_intangible_cost = 0   # Temporary, TODO: remove this line after testing completed
                fixed_upfront_intangible_cost = sub_graph.nodes[node][year]['technologies'][tech]['Upfront intangible cost_fixed']['year_value'] # TODO: implement defaults
                if fixed_upfront_intangible_cost is None:
                    fixed_upfront_intangible_cost = 0
                output = sub_graph.nodes[node][year]['technologies'][tech]['Output']['year_value']
                upfront_cost = calc_upfront_cost(capital_cost,
                                                 fixed_upfront_intangible_cost,
                                                 declining_upfront_intangible_cost,
                                                 output,
                                                 crf)

                operating_maintenance_cost = sub_graph.nodes[node][year]['technologies'][tech]['Operating and maintenance cost']['year_value']
                fixed_annual_intangible_cost = sub_graph.nodes[node][year]['technologies'][tech]['Annual intangible cost_fixed']['year_value']

                if fixed_annual_intangible_cost is None:
                    fixed_annual_intangible_cost = 0
                declining_annual_intangible_cost = 0    # Temporary, TODO: remove this line after testing completed
                output = sub_graph.nodes[node][year]['technologies'][tech]['Output']['year_value']
                annual_cost = calc_annual_cost(operating_maintenance_cost,
                                               fixed_annual_intangible_cost,
                                               declining_annual_intangible_cost,
                                               output)

                lcc = calc_lcc(upfront_cost, annual_cost, annual_service_cost)

                sub_graph.nodes[node][year]["technologies"][tech]["LCC"]["year_value"] = lcc

                # LCC ^ -v
                # *****************
                if round(lcc, 20) == 0:
                    if show_warnings:
                        warnings.warn('LCC has value of 0 at {} -- {}'.format(node, tech))
                    lcc = 0.0001

                if lcc < 0:
                    if show_warnings:
                        warnings.warn('LCC has negative value at {} -- {}'.format(node, tech))
                    lcc = 0.0001

                try:
                    lcc_neg_v = lcc ** (-1.0 * v)
                    total_lcc_v += lcc_neg_v
                except OverflowError as e:
                    raise e

        # Set sum of LCC raised to negative variance
        sub_graph.nodes[node][year]["total_lcc_v"] = total_lcc_v

        # Weighted LCC
        # *************
        weighted_lccs = 0
        # For every tech, use a exogenous or previously calculated market share to calculate LCC
        for tech in node_techs:
            # Determine whether Market share is exogenous or not
            exo_market_share = sub_graph.nodes[node][year]['technologies'][tech]['Market share']['year_value']
            if exo_market_share is not None:
                exogenous = True
            else:
                exogenous = False
            sub_graph.nodes[node][year]['technologies'][tech]['Market share']['exogenous'] = exogenous

            # Determine what market share to use for weighing LCCs
            # If market share is exogenous, set new & total market share to exogenous value
            if exogenous:
                sub_graph.nodes[node][year]['technologies'][tech][
                    'new_market_share'] = exo_market_share
                sub_graph.nodes[node][year]['technologies'][tech][
                    'total_market_share'] = exo_market_share
                market_share = exo_market_share

            # If market share is not exogenous, but was calculated in a previous iteration for
            # this year, use total market share for calculating LCC
            elif 'total_market_share' in sub_graph.nodes[node][year]['technologies'][tech].keys():
                market_share = sub_graph.nodes[node][year]['technologies'][tech]['total_market_share']

            # If market share is not exogenous and hasn't been calculated in a previous year,
            # use the total market share calculated in the previous year for calculating LCC
            else:
                previous_year = str(int(year) - year_step)
                market_share = sub_graph.nodes[node][previous_year]['technologies'][tech]['total_market_share']

            if market_share is None:
                if show_warnings:
                    warnings.warn("Market Share is NONE!!")

            # Weight LCC and Add to Node Total
            # ********************************
            # find the years where the tech is available
            low, up = utils.range_available(sub_graph, node, tech)
            if int(year) in range(low, up):
                curr_lcc = sub_graph.nodes[node][year]["technologies"][tech]["LCC"]["year_value"]
                weighted_lccs += market_share * curr_lcc

        fuel_name = node.split('.')[-1]
        sub_graph.nodes[node][year]["Life Cycle Cost"] = {fuel_name: utils.create_value_dict(weighted_lccs)}

    elif ('technologies' in sub_graph.nodes[node].keys()) and (sub_graph.nodes[node]['competition_type']=='Fixed Ratio'):
        print("{} is fixed ratio w/ techs".format(node))
    else:
        # When calculating a service cost for a technology or node using the "Fixed Ratio" decision
        # rule, multiply the LCCs of the service required by its "Service Requested" line value.
        # Sometimes, the Service Requested line values act as percent shares that add up to 1 for a
        # given fixed ratio decision node. Other times, they do not and the Service Requested Line
        # values sum to numbers greater or less than 1.
        service_cost = econ.get_node_service_cost(sub_graph,
                                                  full_graph,
                                                  node,
                                                  year,
                                                  fuels)

        # Is service cost just the cost of these nodes?
        fuel_name = node.split('.')[-1]
        sub_graph.nodes[node][year]["Life Cycle Cost"] = {fuel_name: utils.create_value_dict(service_cost)}


def calc_lcc(upfront_cost, annual_cost, annual_service_cost):
    lcc = upfront_cost + annual_cost + annual_service_cost
    return lcc


def calc_upfront_cost(capital_cost, fixed_upfront_intangible_cost, declining_upfront_intangible_cost, output, crf):
    uc = (capital_cost + fixed_upfront_intangible_cost + declining_upfront_intangible_cost)/output * crf
    return uc


def calc_annual_cost(operating_maintenance_cost, fixed_annual_intangible_cost, declining_annual_intangible_cost, output):
    ac = (operating_maintenance_cost + fixed_annual_intangible_cost + declining_annual_intangible_cost) / output
    return ac


def calc_capital_cost(declining_cc, overnight_cc, declining_limit=1):
    cc = max(declining_cc, overnight_cc*declining_limit)
    return cc


def calc_declining_cc(gcc, new_stock, cumul_new_stock, progress_ratio):
    pass


def calc_upfront_declining_intangible_cost(initial_upfront_intangible_cost, rate_constant, shape_constant, percent_new_market_share):
    rate_constant = 0
    pass


def annual_declining_intangible_cost(inition_annual_intangible_cost, rate_constant, shape_constant, percent_new_market_share):
    rate_constant = 0
    pass



