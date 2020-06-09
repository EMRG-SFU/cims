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
    def_param_names = ['Operating cost', 'Output']
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


def find_initial_market_share():
    pass


def lcc_calculation(sub_graph, node, year, year_step, full_graph, fuels):
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
    total_lcc_v = 0.0

    # Check if the node is a tech compete node.
    if sub_graph.nodes[node]["competition type"] == "tech compete":
        # Find the v parameter
        v = econ.get_heterogeneity(sub_graph, node, year)
        node_techs = sub_graph.nodes[node][year]["technologies"].keys()

        # For every tech in the node, retrieve or compute required economic values
        for tech in node_techs:
            calculate_tech_econ_values(sub_graph, node, tech, year)

            # If the technology is available in this year, go through it
            # (range is [lower year, upper year + 1] to work with range function
            low, up = utils.range_available(sub_graph, node, tech)

            if int(year) in range(low, up):
                # Service Cost
                # ************
                service_cost = econ.get_technology_service_cost(sub_graph,
                                                                full_graph,
                                                                node,
                                                                year,
                                                                tech,
                                                                fuels)
                sub_graph.nodes[node][year]["technologies"][tech]["Service cost"][
                    "year_value"] = service_cost

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

                # LCC
                # *****************
                operating_cost = \
                sub_graph.nodes[node][year]["technologies"][tech]["Operating cost"]["year_value"]
                lcc = service_cost + operating_cost + full_cc
                sub_graph.nodes[node][year]["technologies"][tech]["LCC"]["year_value"] = lcc

                # LCC ^ -v
                # *****************
                lcc_neg_v = lcc ** (-1.0 * v)
                total_lcc_v += lcc_neg_v

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
                warnings.warn("Market Share is NONE!!")

            # Weight LCC and Add to Node Total
            # ********************************
            # find the years where the tech is available
            low, up = utils.range_available(sub_graph, node, tech)
            if int(year) in range(low, up):
                curr_lcc = sub_graph.nodes[node][year]["technologies"][tech]["LCC"]["year_value"]
                weighted_lccs += market_share * curr_lcc

        sub_graph.nodes[node][year]["total lcc"] = weighted_lccs
