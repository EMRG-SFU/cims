import copy
from . import utils
from . import graph_utils
import warnings


def get_heterogeneity(g, node, year):
    try:
        v = g.nodes[node][year]["Heterogeneity"]["v"]["year_value"]
    except KeyError:
        v = 10  # default val
    if v is None:
        v = 10
    return v


def get_provided(g, node, year, parent_provide):
    node_name = node
    provided = copy.copy(g.nodes[node][year]["Service provided"])
    service_name = list(provided.keys())[0]
    service = provided[service_name]['branch']
    if service == node_name:
        parent = graph_utils.get_parent(g, node, year)
        parent_request = parent["Service requested"][service_name]
        request_units = utils.split_unit(parent_request['unit'])
        service_unit = provided[service_name]['unit']

        parent_req_val = parent_request['year_value']
        parent_provide_unit = parent_provide[year][graph_utils.parent_name(node)][request_units[-1]]
        # provide_val = parent_request['year_value'] * \
        # parent_provide[year][graph_utils.parent_name(node)][request_units[-1]]
        provide_val = parent_req_val / parent_provide_unit

        g.nodes[node][year]["Service provided"][service_name]["year_value"] = provide_val
        return service_unit, provide_val


def lastyear_tech(g, node, year, tech, param, step=5):
    """
    Check if there's a value filled out within a tech, if not, take last years value
    """
    value = g.nodes[node][year]["technologies"][tech][param]["year_value"]
    if value is None:
        if year == "2000":
            raise ValueError(f"No initial value for node {node}, tech {tech}, parameter {param}")
        else:
            last_year = str(int(year) - step)
            value = g.nodes[node][last_year]["technologies"][tech][param]["year_value"]
    return value


def lastyear_fuel(g, node, year, fuel, param, step=5):
    """
    In an operation related to fuel, check if there's a value filled out, if not, take last years value
    """
    value = g.nodes[node][year][param][fuel]["year_value"]

    if value is None:
        if year == "2000":
            raise ValueError(f"No initial value for node {node}, fuel {fuel}, parameter {param}")
        else:
            last_year = str(int(year) - step)
            value = g.nodes[node][last_year][param][fuel]["year_value"]

    return value


def get_technology_service_cost(sub_graph, full_graph, node, year, tech, fuels):
    """
    Find the service cost associated with a given technology.

    1. For each service being requested:
            i) If the service is a fuel, find the fuel price (Life Cycle Cost) and add it to the
               service cost. If the fuel doesn't have a fuel price,
           ii) Otherwise, use the service's Life Cycle Cost which was calculated already.
    2. Return the service cost (currently assumes that there can only be one TODO: VERIFY / FIX THIS)
    """
    def do_sc_calculation(service_requested):
        service_requested_value = service_requested['year_value']
        service_cost = 0
        if service_requested['branch'] in fuels:
            fuel_branch = service_requested['branch']

            if 'Life Cycle Cost' in full_graph.nodes[fuel_branch][year]:
                fuel_name = list(full_graph.nodes[fuel_branch][year]['Life Cycle Cost'].keys())[0]
                service_requested_lcc = full_graph.nodes[fuel_branch][year]['Life Cycle Cost'][fuel_name]['year_value']
            else:
                service_requested_lcc = 1  # TODO: Properly implement defaults
                # full_graph.nodes[fuel_branch][year]['Life Cycle Cost'] = {fuel_name: utils.create_value_dict(1)}

            # fuel_name = list(full_graph.nodes[fuel_branch][year]['Life Cycle Cost'].keys())[0]
            # fuel_price = full_graph.nodes[fuel_branch][year]['Life Cycle Cost'][fuel_name]['year_value']
            # service_cost = fuel_price * service_requested_value
        else:
            # service_requested_value = service_requested['year_value']
            service_requested_branch = service_requested['branch']
            # TODO: Add Some Reasonable Default/Behaviour for when we have broken a loop & need to
            #  grab the lcc (currently, the LCC isn't known)
            if 'Life Cycle Cost' in full_graph.nodes[service_requested_branch][year]:
                service_name = service_requested_branch.split('.')[-1]
                service_requested_lcc = full_graph.nodes[service_requested_branch][year]['Life Cycle Cost'][service_name]['year_value']

            else:
                service_requested_lcc = 1 # TODO: Properly implement defaults

        service_cost += service_requested_lcc * service_requested_value

        return service_cost

    total_tech_service_cost = 0

    if 'Service requested' in sub_graph.nodes[node][year]['technologies'][tech]:
        service_req = sub_graph.nodes[node][year]['technologies'][tech]['Service requested']
        if isinstance(service_req, dict):
            total_tech_service_cost += do_sc_calculation(service_req)
        elif isinstance(service_req, list):
            for req in service_req:
                total_tech_service_cost += do_sc_calculation(req)
        else:
            print(f"type for service requested? {type(service_req)}")

    return total_tech_service_cost


def get_node_service_cost(sub_graph, full_graph, node, year, fuels):
    """
    Find the service cost associated with a given node that doesn't include technologies.

    1. For each service being requested:
            i) If the service is a fuel, find the fuel price and add it to the service cost.
           ii) Otherwise, use the service's total lcc which was calculated already.
           # ii) Otherwise, use the service's children/techs to find the lcc at the node. Add this to the service cost.
    4. Return the service cost (currently assumes that there can only be one TODO: VERIFY / FIX THIS)
    """
    def do_sc_calculation(service_requested):
        # print('\t {}'.format(service_requested))
        service_requested_value = service_requested['year_value']
        service_cost = 0
        if service_requested['branch'] in fuels:
            fuel_branch = service_requested['branch']
            fuel_name = list(full_graph.nodes[node][year]['Life Cycle Cost'].keys())[0]
            fuel_price = full_graph.nodes[fuel_branch][year]['Life Cycle Cost'][fuel_name]['year_value']
            service_cost = fuel_price * service_requested_value
        else:
            service_requested_value = service_requested['year_value']
            service_requested_branch = service_requested['branch']
            service_name = service_requested_branch.split('.')[-1]
            service_requested_lcc = full_graph.nodes[service_requested_branch][year]['Life Cycle Cost'][service_name]['year_value']
            service_cost += service_requested_lcc * service_requested_value

        return service_cost

    total_node_service_cost = 0

    if 'Service requested' in sub_graph.nodes[node][year]:
        service_req = sub_graph.nodes[node][year]['Service requested']
        if 'year_value' in service_req:
            total_node_service_cost += do_sc_calculation(service_req)
        else:
            for req in service_req.values():
                total_node_service_cost += do_sc_calculation(req)

    return total_node_service_cost


def get_crf(g, node, year, tech, finance_base=0.1, life_base=10.0):
    # TODO: Determine why we do this instead of get CRF from the model description
    # TODO: Verify that this is doing the correct calculation
    finance_discount = g.nodes[node][year]["technologies"][tech]["Discount rate_Financial"]["year_value"]
    lifespan = g.nodes[node][year]["technologies"][tech]["Lifetime"]["year_value"]
    if finance_discount is None:
        finance_discount = finance_base
    if finance_discount == 0:
        warnings.warn('Discount rate_Financial has value of 0 at {} -- {}'.format(node, tech))
        finance_discount = finance_base

    if lifespan is None:
        lifespan = life_base

    crf = finance_discount / (1 - (1 + finance_discount) ** (-1.0 * lifespan))

    return crf


def get_capcost(g, node, year, tech, crf, default_full_cc=0.0):
    # TODO: ADD DEFAULTS & FULL CAP COST
    # TODO: Ensure this is the correct calculation
    tech_data = g.nodes[node][year]['technologies'][tech]

    full_cap_cost = tech_data["Full capital cost"]["year_value"]
    if full_cap_cost:
        pass
    else:
        # Try to Calculate
        overnight_cc = tech_data['Capital cost_overnight']['year_value']
        upfront_fixed = tech_data['Upfront intangible cost_fixed']['year_value']
        upfront_declining = 0  # TODO: Find & implement calculation for this

        cap_cost_components = [x if x else 0 for x in [overnight_cc, upfront_fixed, upfront_declining]]
        cap_cost = sum(cap_cost_components)

        output = tech_data['Output']['year_value']
        full_cap_cost = (cap_cost / output) * crf

    return full_cap_cost