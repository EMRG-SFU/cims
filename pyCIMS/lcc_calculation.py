import warnings
from . import utils
import math
from . import graph_utils
from copy import deepcopy


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
        lcc, lcc_source = model.get_param('Life Cycle Cost', node, year, return_source=True)
        if lcc_source == 'model':
            return

    # Check if the node is a tech compete node:
    if model.get_param('competition type', node) in ["tech compete"]:
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
                annual_service_cost, sc_source = model.get_or_calc_param('Service cost',
                                                                         node, year, tech,
                                                                         return_source=True)

                val_dict = {'year_value': annual_service_cost, "branch": str(node), 'param_source': sc_source}
                model.set_param_internal(val_dict, 'Service cost', node, year, tech)

                # CRF
                # ************
                crf, crf_source = model.get_or_calc_param("CRF",
                                                          node, year, tech,
                                                          return_source=True)
                val_dict = {'year_value': crf, 'param_source': crf_source}
                model.set_param_internal(val_dict, 'CRF', node, year, tech)

                # LCC
                # ************
                lcc, lcc_source = model.get_or_calc_param('Life Cycle Cost',
                                                          node, year, tech,
                                                          return_source=True)
                val_dict = {'year_value': lcc, 'param_source': lcc_source}
                model.set_param_internal(val_dict, 'Life Cycle Cost', node, year, tech)

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
            ms, ms_source = model.get_param('Market share',
                                            node, year, tech,
                                            return_source=True)
            ms_exogenous = ms_source == 'model'

            # Determine what market share to use for weighing Life Cycle Costs
            # If market share is exogenous, set new & total market share to exogenous value
            if ms_exogenous:
                val_dict = utils.create_value_dict(ms, param_source='calculation')
                model.set_param_internal(val_dict, 'new_market_share', node, year, tech)
                model.set_param_internal(val_dict, 'total_market_share', node, year, tech)

            market_share = model.get_param('total_market_share', node, year, tech)

            # Weight Life Cycle Cost and Add to Node Total
            # ********************************************
            # find the years where the tech is available
            first_year_available = model.get_param('Available',
                                                   node, str(model.base_year), tech)
            first_year_unavailable = model.get_param('Unavailable',
                                                     node, str(model.base_year), tech)
            if first_year_available <= int(year) < first_year_unavailable:
                curr_lcc = model.get_param('Life Cycle Cost', node, year, tech)
                weighted_lccs += market_share * curr_lcc

        fuel_name = node.split('.')[-1]
        if "Life Cycle Cost" in sub_graph.nodes[node][year]:
            sub_graph.nodes[node][year]["Life Cycle Cost"].update(
                {fuel_name: utils.create_value_dict(weighted_lccs, param_source='calculation')})
        else:
            sub_graph.nodes[node][year]["Life Cycle Cost"] = {
                fuel_name: utils.create_value_dict(weighted_lccs, param_source='calculation')}

    else:
        # When calculating a service cost for a technology or node using the "Fixed Ratio" decision
        # rule, multiply the Life Cycle Costs of the service required by its "Service Requested"
        # line value. Sometimes, the Service Requested line values act as percent shares that add up
        # to 1 for a given fixed ratio decision node. Other times, they do not and the Service
        # Requested Line values sum to numbers greater or less than 1.
        service_cost, sc_source = model.get_or_calc_param('Service cost', node, year, return_source=True)
        # Is service cost just the cost of these nodes?
        fuel_name = node.split('.')[-1]

        if "Life Cycle Cost" in sub_graph.nodes[node][year]:
            sub_graph.nodes[node][year]["Life Cycle Cost"].update(
                {fuel_name: utils.create_value_dict(service_cost, param_source=sc_source)})
        else:
            sub_graph.nodes[node][year]["Life Cycle Cost"] = {
                fuel_name: utils.create_value_dict(service_cost, param_source=sc_source)}


def calc_lcc(model, node, year, tech):
    upfront_cost = model.get_or_calc_param('Upfront cost', node, year, tech)
    annual_cost = model.get_or_calc_param('Annual cost', node, year, tech)
    annual_service_cost = model.get_or_calc_param('Service cost', node, year, tech)
    emissions_cost = model.get_or_calc_param('Emissions cost', node, year, tech)
    lcc = upfront_cost + annual_cost + annual_service_cost + emissions_cost
    return lcc


def emissions_cost(model, node, year, tech):
    """
    Returns the emission cost at that node for the following parameters. First calculate total emissions, gross
    emissions, captured emissions, net emissions, and then the final emission cost. Returns the sum of all emission
    costs. To see how the calculation works,
    see the file 'Emissions_tax_example.xlsx': https://gitlab.rcg.sfu.ca/mlachain/pycims_prototype/-/issues/22#note_6489

    :param pyCIMS.model.Model model:
    :param str node: The name of the node whose parameters we are calculating.
    :param str year: The year for which we are calculating parameters.
    :param str tech: The tech from the node we're doing the calculation on
    :return: the total emission cost (float)
    """

    fuels = graph_utils.get_fuels(model.graph)
    if 'Tax' not in model.graph.nodes[node][year] or node in fuels:
        # No emissions cost
        return 0

    # Initialize all taxes and emission removal rates to 0
    tax_rates = {}

    if tech == 'Extraction of coal Biodiesel':
        rashid = 1

    for fuel_name in ['CO2', 'CH4', 'N2O']:
        tax_rates[fuel_name] = {}
        for sub_name in ['Process', 'Fugitive', 'Combustion']:
            tax_rates[fuel_name][sub_name] = 0

    removal_rates = deepcopy(tax_rates)

    # Grab correct tax values
    all_taxes = model.get_param('Tax', node, year)  # returns a dict
    for tax, tax_dict in all_taxes.items():
        # example of item in tax_rates -> 'CO2': {'Combustion': 5}
        tax_rates[tax][tax_dict['sub_param']] = tax_dict['year_value']

    # EMISSIONS tech level
    total_emissions = {}
    total = 0
    if 'Emissions' in model.graph.nodes[node][year]['technologies'][tech]:
        total_emissions[tech] = {}
        data = model.graph.nodes[node][year]['technologies'][tech]['Emissions']

        # If only 1 emission, put it in a list
        if isinstance(data, dict):
            data = [data]

        for fuel_info in data:
            if fuel_info['value'] not in total_emissions[tech]:
                total_emissions[tech][fuel_info['value']] = {}
            total_emissions[tech][fuel_info['value']][fuel_info['sub_param']] = fuel_info['year_value']

    # EMISSIONS REMOVAL tech level
    if 'Emissions removal' in model.graph.nodes[node][year]:
        removal_dict = model.graph.nodes[node][year]['Emissions removal']
        for removal_name, removal_data in removal_dict.items():
            removal_rates[removal_name][removal_data['sub_param']] = removal_data['year_value']

    # Check all services requested for
    # TODO: Update get_param() so that this if statement is fixed
    if 'Service requested' in model.graph.nodes[node][year]['technologies'][tech]:
        data = model.graph.nodes[node][year]['technologies'][tech]['Service requested']

        if isinstance(data, dict):
            # Wrap the single request in a list to work with below code
            data = [data]

        # EMISSIONS child level
        for child_info in data:
            req_val = child_info['year_value']
            child_node = child_info['branch']
            if 'Emissions' in model.graph.nodes[child_node][year] and child_node in fuels:
                fuel_emissions = model.get_param('Emissions', child_node, year)
                total_emissions[child_node] = {}
                for fuel_name, fuel_data in fuel_emissions.items():
                    if fuel_name not in total_emissions[child_node]:
                        total_emissions[child_node][fuel_name] = {}
                    total_emissions[child_node][fuel_name][fuel_data['sub_param']] = fuel_data['year_value'] * req_val

    gross_emissions = deepcopy(total_emissions)

    if 'Service requested' in model.graph.nodes[node][year]['technologies'][tech]:
        data = model.graph.nodes[node][year]['technologies'][tech]['Service requested']

        if isinstance(data, dict):
            data = [data]

        for child_info in data:
            req_val = child_info['year_value']
            child_node = child_info['branch']

            # GROSS EMISSIONS
            if 'Gross Emissions' in model.graph.nodes[child_node][year] and child_node in fuels:
                gross_dict = model.graph.nodes[child_node][year]['Gross Emissions']
                for gross_name, gross_data in gross_dict.items():
                    gross_emissions[child_node][gross_name][gross_data['sub_param']] = gross_data['year_value'] * req_val

            # EMISSIONS REMOVAL child level
            if 'technologies' in model.graph.nodes[child_node][year]:
                child_techs = model.graph.nodes[child_node][year]['technologies']
                for _, tech_data in child_techs.items():
                    if 'Emissions removal' in tech_data:
                        removal_dict = tech_data['Emissions removal']
                        removal_rates[removal_dict['value']][removal_dict['sub_param']] = removal_dict['year_value']

    # CAPTURED EMISSIONS
    captured_emissions = deepcopy(gross_emissions)
    for node_name in captured_emissions:
        for fuel_name in captured_emissions[node_name]:
            for sub_name in captured_emissions[node_name][fuel_name]:
                em_removed = removal_rates[fuel_name][sub_name]
                captured_emissions[node_name][fuel_name][sub_name] *= em_removed

    # NET EMISSIONS
    net_emissions = deepcopy(total_emissions)
    for node_name in net_emissions:
        for fuel_name in net_emissions[node_name]:
            for sub_name in net_emissions[node_name][fuel_name]:
                net_emissions[node_name][fuel_name][sub_name] -= captured_emissions[node_name][fuel_name][sub_name]

    # EMISSIONS COST
    emissions_cost = deepcopy(net_emissions)
    for node_name in emissions_cost:
        for fuel_name in emissions_cost[node_name]:
            for sub_name in emissions_cost[node_name][fuel_name]:
                tax = tax_rates[fuel_name][sub_name]
                emissions_cost[node_name][fuel_name][sub_name] *= tax

    # Add everything in nested dictionary together
    for node_name in emissions_cost:
        for fuel_name in emissions_cost[node_name]:
            for _, cost in emissions_cost[node_name][fuel_name].items():
                total += cost

    return total


def calc_upfront_cost(model, node, year, tech):
    crf = model.get_or_calc_param("CRF", node, year, tech)
    capital_cost = model.get_or_calc_param('Capital cost', node, year, tech)
    fixed_uic = model.get_or_calc_param('Upfront intangible cost_fixed', node, year, tech)
    declining_uic = calc_declining_uic(model, node, year, tech)
    output = model.get_param('Output', node, year, tech)

    uc = (capital_cost +
          fixed_uic +
          declining_uic) / output * crf

    return uc


def calc_annual_cost(model, node, year, tech):
    output = model.get_param('Output', node, year, tech)
    operating_maintenance_cost = model.get_param('Operating and maintenance cost', node, year, tech)
    fixed_aic = model.get_param('Annual intangible cost_fixed', node, year, tech)
    declining_aic = model.get_or_calc_param('Annual intangible cost_declining', node, year, tech)

    ac = (operating_maintenance_cost +
          fixed_aic +
          declining_aic) / output
    return ac


def calc_capital_cost(model, node, year, tech):
    cc_overnight = model.get_param('Capital cost_overnight', node, year, tech)
    declining_cc_limit = model.get_param('Capital cost_declining_limit', node, year, tech)
    declining_cc = model.get_or_calc_param("Capital cost_declining", node, year, tech)

    if declining_cc is None:
        cc = cc_overnight
    else:
        cc = max(declining_cc, cc_overnight * declining_cc_limit)
    return cc


def calc_declining_cc(model, node, year, tech):
    dcc_class = model.get_param('Capital cost_declining_Class', node, year, tech, sub_param='value')

    if dcc_class is None:
        cc_declining = None

    else:
        # Progress Ratio
        progress_ratio = model.get_param('Capital cost_declining_Progress Ratio', node, year, tech)
        gcc_t = model.get_or_calc_param('GCC_t', node, year,
                                        tech)  # capital cost adjusted for cumulative stock in all other countries

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
            year_list = [str(x) for x in range(int(model.base_year) + int(model.step), int(year), int(model.step))]
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

        try:
            denominator = 1 + shape_constant * math.exp(rate_constant * prev_nms)
        except OverflowError:
            print(node, year, shape_constant, rate_constant, prev_nms)
            raise OverflowError
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
                service_requested_lcc = model.get_node_parameter_default('Life Cycle Cost',
                                                                         'sector')
            try:
                fuel_name = fuel_branch.split('.')[-1]
                price_multiplier = model.graph.nodes[node][year]['Price Multiplier'][fuel_name]['year_value']
            except KeyError:
                price_multiplier = 1
            service_requested_lcc *= price_multiplier
        else:
            service_requested_branch = service_requested['branch']
            if 'Life Cycle Cost' in model.graph.nodes[service_requested_branch][year]:
                service_name = service_requested_branch.split('.')[-1]
                service_requested_lcc = \
                    model.graph.nodes[service_requested_branch][year]['Life Cycle Cost'][service_name]['year_value']
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
