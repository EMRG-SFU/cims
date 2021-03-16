import warnings


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
            fuel_lcc = full_graph.nodes[fuel_branch][year]['Life Cycle Cost'][fuel_name]['year_value']
            price_multiplier = full_graph.nodes[node][year]['Price Multiplier'][fuel_name]['year_value']
            fuel_price = fuel_lcc * price_multiplier
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

