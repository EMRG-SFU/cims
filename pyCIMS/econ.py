import warnings
from utils import get_param


def get_technology_service_cost(sub_graph, node, year, tech, model):
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

        else:
            service_requested_branch = service_requested['branch']
            if 'Life Cycle Cost' in model.graph.nodes[service_requested_branch][year]:
                service_name = service_requested_branch.split('.')[-1]
                service_requested_lcc = model.graph.nodes[service_requested_branch][year]['Life Cycle Cost'][service_name]['year_value']
                # NEW
                new_sr_lcc = model.get_param('Life Cycle Cost', service_requested_branch, year)
                assert(new_sr_lcc == service_requested_lcc)
                # NEW
            else:
                # Encountering a non-visited node
                service_requested_lcc = 1

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
           # ii) Otherwise, use the service's children/techs to find the lcc at the node. Add this
           to the service cost.
    4. Return the service cost (currently assumes that there can only be one
    """
    def do_sc_calculation(service_requested):
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


def get_crf(g, node, year, tech, model):
    finance_discount = model.get_param('Discount rate_Financial', node, year, tech)
    lifespan = model.get_param('Lifetime', node, year, tech)

    if finance_discount == 0:
        warnings.warn('Discount rate_Financial has value of 0 at {} -- {}'.format(node, tech))
        finance_discount = model.get_tech_parameter_default('Discount rate_Financial')

    crf = finance_discount / (1 - (1 + finance_discount) ** (-1.0 * lifespan))

    return crf
