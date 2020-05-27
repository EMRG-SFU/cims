import copy
import math
from . import utils
from . import graph_utils


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


def get_technology_service_cost(g, full_graph, node, year, tech, fuels):
    """
    Find the service cost associated with a given technology.

    1. For each service being requested:
            i) If the service is a fuel, find the fuel price and add it to the service cost.
           ii) Otherwise, use the service's total lcc which was calculated already.
           # ii) Otherwise, use the service's children/techs to find the lcc at the node. Add this to the service cost.
    4. Return the service cost (currently assumes that there can only be one TODO: VERIFY / FIX THIS)
    """

    def do_sc_calculation(service_requested):
        service_requested_value = service_requested['year_value']
        service_cost = 0
        if service_requested['branch'] in fuels:
            fuel_branch = service_requested['branch']
            fuel_name = service_requested['branch'].split('.')[-1]

            fuel_price = full_graph.nodes[fuel_branch][year]['Production Cost'][fuel_name]['year_value']
            service_cost = fuel_price * service_requested_value
        else:
            service_requested_value = service_requested['year_value']
            service_requested_branch = service_requested['branch']
            service_requested_lcc = g.nodes[service_requested_branch][year]['total lcc']
            service_cost += service_requested_lcc * service_requested_value

        return service_cost

    total_tech_service_cost = 0

    if 'Service requested' in g.nodes[node][year]['technologies'][tech]:
        service_req = g.nodes[node][year]['technologies'][tech]['Service requested']
        if isinstance(service_req, dict):
            total_tech_service_cost += do_sc_calculation(service_req)
        elif isinstance(service_req, list):
            for req in service_req:
                total_tech_service_cost += do_sc_calculation(req)
        else:
            print(f"type for service requested? {type(service_req)}")

    return total_tech_service_cost


def get_crf(g, node, year, tech, finance_base=0.1, life_base=10.0):
    # TODO: Determine why we do this instead of get CRF from the model description
    # TODO: Verify that this is doing the correct calculation
    finance_discount = g.nodes[node][year]["technologies"][tech]["Discount rate_Financial"]["year_value"]
    lifespan = g.nodes[node][year]["technologies"][tech]["Lifetime"]["year_value"]
    if finance_discount is None:
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
    if full_cap_cost is None:
        # Try to Calculate
        overnight_cc = tech_data['Overnight capital cost']['year_value']
        upfront_fixed = tech_data['Upfront fixed intangible cost']['year_value']
        upfront_declining = 0  # TODO: Find & implement calculation for this
        if None in [overnight_cc, upfront_fixed, upfront_declining]:
            full_cap_cost = default_full_cc
        else:
            cap_cost = overnight_cc + upfront_fixed + upfront_declining
            output = tech['Output']['year_value']
            full_cap_cost = (cap_cost / output) * crf

    return full_cap_cost


def calculate_stock_retirement(type, starting_stock, starting_year, years, intercept=11.513):
    def base_stock_retirement(base_stock, initial_year, current_year, lifespan):
        unretired_base_stock = base_stock * (1 - (current_year - initial_year)/lifespan)
        return unretired_base_stock

    def purchased_stock_retirement(purchased_stock, purchased_year, current_year, lifespan):
        exponent = (intercept / lifespan) * ((current_year - purchased_year) - lifespan)
        unretired_purchased_stock = purchased_stock / (1 + math.exp(exponent))
        return unretired_purchased_stock

    stock_by_year = {}

    tech_lifespan = 10  # TODO: FIND THIS

    for year in years:
        if type == 'base':
            remaining_stock = base_stock_retirement(starting_stock, starting_year, year, tech_lifespan)
        elif type == 'purchased':
            remaining_stock = purchased_stock_retirement(starting_stock, starting_year, year, tech_lifespan)

        stock_by_year[year] = remaining_stock

    return stock_by_year
