import copy
import utils
import graph_utils


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

        provide_val = parent_request['year_value'] * parent_provide[year][graph_utils.parent_name(node)][request_units[-1]]
        g.nodes[node][year]["Service provided"][service_name]["year_value"] = provide_val
        return service_unit, provide_val


def lastyear_tech(g, node, year, tech, param, step=5):
    """
    Check if there's a value filled out within a tech, if not, take last years value
    """
    value = g.nodes[node][year]["technologies"][tech][param]["year_value"]
    if value == None:
        if year == "2000":
            raise ValueError(f"No initial value for node {node}, tech {tech}, parameter {param}")
        else:
            last_year = str( int(year) - step )
            value = g.nodes[node][last_year]["technologies"][tech][param]["year_value"]
    return value


def lastyear_fuel(g, node, year, fuel, param, step=5):
    """
    In an operation related to fuel, check if there's a value filled out, if not, take last years value
    """
    value = g.nodes[node][year][param][fuel]["year_value"]

    if value == None:
        if year == "2000":
            raise ValueError(f"No initial value for node {node}, fuel {fuel}, parameter {param}")
        else:
            last_year = str( int(year) - step )
            value = g.nodes[node][last_year][param][fuel]["year_value"]

    return value


def get_service_cost(g, node, year, tech, fuels, prices):
    """
    1. Initialize service cost to an empty list
    2. Find all the children of the node. Check to make sure child is a list (hopefully of children)
    3. Try to calculate the service cost
        (A) find the service(s) being requested
        (B) If 1 service is being requested:
            i) If the service is a fuel, grab the price of the fuel. Use it to calculate the service cost and append it.
           ii) Otherwise, find the lcc of the requesting service and append it.
        (C) If multiple services are being requested:
            i) essentially, execute step B for every service
    4. Return the first service cost (currently assumes that there can only be one TODO: FIX THIS)

    """
    # print("-- Service Cost: {} --".format(node))
    # print(prices[year])
    service_cost = []

    child = graph_utils.child_name(g, node, return_empty=True)

    utils.check_type(child, list)

    try:
        service_req = g.nodes[node][year]["technologies"][tech]["Service requested"]
        # sometimes more than one thing requested, that would make results into a list
        # (eg [{year_value:... , branch:...electricity, ...},
        #      {year_value:... , branch:...furnace, ...}}])
        if isinstance(service_req, dict):
            if service_req['branch'] in fuels:
                service_req_val = service_req["year_value"]
                # price_tech = prices[year][get_name(service_req['branch'])]["year_value"]
                price_tech = prices[service_req['branch']]["year_value"]  # JA
                service_cost.append(price_tech * service_req_val)

            else:
                service_req_val = service_req["year_value"]
                if isinstance(child, list):
                    # THIS should be a list with 1 element, but check to make sure it's correct
                    # IF IT'S POSSIBLE to have 2 children here, the graph would get overwritten
                    # But I don't know how to deal with 2 service costs for a single tech
                    for c in child:
                        # print(f"node within: {get_name(node)}")
                        # print(f"child within: {c}")
                        child_lccs = g.nodes[c][year]["total lcc"]
                        # print(f"child_lccs: {child_lccs}")
                        service_cost.append(child_lccs * service_req_val)
                # else check if child is a string

        elif isinstance(service_req, list):
            for reqs in service_req:
                if reqs['branch'] in fuels:
                    service_req_val = reqs["year_value"]
                    # TODO: Make sure the price multipliers are being applied
                    # service_cost.append(prices[year][get_name(reqs['branch'])]["year_value"] * service_req_val)
                    service_cost.append(prices[reqs['branch']]["year_value"] * service_req_val) #JA

                else:
                    # TODO check reqs that it's not overwriting values below
                    service_req_val = reqs["year_value"]
                    for c in child:
                        child_lccs = g.nodes[c][year]["total lcc"]
                        service_cost.append(child_lccs * service_req_val)

        else:
            print(f"type for service requested? {type(service_req)}")

    except KeyError:
        service_cost.append(0)

    # Just returning a single value before I know whether 2 service costs is possible
    return service_cost[0]


def get_crf(g, node, year, tech, finance_base=0.1, life_base=10.0):

    finance_discount = g.nodes[node][year]["technologies"][tech]["Discount rate_Financial"]["year_value"]
    lifespan = g.nodes[node][year]["technologies"][tech]["Lifetime"]["year_value"]

    if finance_discount == None:
        finance_discount = finance_base

    if lifespan == None:
        lifespan = life_base


    crf = finance_discount/(1 - (1 + finance_discount)**(-1.0*lifespan))

    return crf


def get_capcost(g, node, year, tech, crf, default_cc=0.0):
    cap_cost = g.nodes[node][year]["technologies"][tech]["Capital cost"]["year_value"]
    if cap_cost == None:
        cap_cost = default_cc

    output = g.nodes[node][year]["technologies"][tech]["Output"]["year_value"]
    full_cap_cost = (cap_cost/output)*crf

    return full_cap_cost