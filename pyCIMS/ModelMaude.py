import networkx as nx
import copy as copy
import re
import random
from pprint import pprint
import traceback
import sys
import logging


# Configure logging and set flag to raise exceptions
logging.raiseExceptions = True
logger = logging.getLogger(__name__)
logger.info('Start')


def find_value(graph, node, parameter, year=None):
    """
    Find a parameter's value at a given node or its structural ancestors.

    First attempts to locate a parameter at a given node. If the parameter does not exist at that node, a recursive call
    is made to find the parameter's value at `node`'s parent, if one exists. If no parent exists None is returned.
    Parameters
    ----------
    graph : networkx.Graph
        The graph where `node` resides.
    node : str
        The name of the node to begin our search from. Must be contained within `graph`. (e.g. 'pyCIMS.Canada.Alberta')
    parameter : str
        The name of the parameter whose value is being found. (e.g. 'Sector type')
    year : str, optional
        The year associated with sub-dictionary to search at `node`. Default is None, which implies that year
        sub-dictionaries should be searched. Instead, only search for `parameter` in `node`s top level data.

    Returns
    -------
    Any
        The value associated with `parameter` if a value can be found at `node` or one of its ancestors. Otherwise None
    """
    data = graph.nodes[node]
    parent = '.'.join(node.split('.')[:-1])

    # Look at the Node/Year
    if year:
        year_data = graph.nodes[node][year]
        if parameter in year_data.keys():
            return year_data[parameter]

    # Look at the Node
    if parameter in data.keys():
        return data[parameter]

    # Look at the Parent
    if parent:
        return find_value(graph, parent, parameter, year)

    # Worst Case, return None
    return None

def get_name(branch_name):
    '''
    Fetch name of the object of interest from a branch name

    Parameters
    ----------
    :branch_name: str, name of branch

    Returns
    -------
    :name: str, name of last word to the right in branch name
    '''
    name = '.'.join(branch_name.split('.')[-1:])
    return name




def check_type(variable, datatype, node="unnamed", passing=False):
    try:
        if not isinstance(variable, datatype):
            raise TypeError
    except TypeError:
        logger.error(f"Error in node {get_name(node)}, \nData should be a {datatype} but is a {type(variable)}\n",
                      exc_info=False)
        if passing:
            pass
        else:
            raise


def get_crf(g, node, tech, year):
    r = g.nodes[node][year]["technologies"][tech]["Discount rate_Financial"]["year_value"]
    # ML! Set to 2000 but check if viable
    N = g.nodes[node]["2000"]["technologies"][tech]["Lifetime"]["year_value"]
    if r == None:
        last_year = str(int(year)- 5)
        r = g.nodes[node][last_year]["technologies"][tech]["Discount rate_Financial"]["year_value"]
    if N == None:
        last_year = str(int(year)- 5)
        N = g.nodes[node][last_year]["technologies"][tech]["Lifetime"]["year_value"]

    crf = r/(1 - (1 + r)**(-1.0 * N))
    return crf



def get_output():
    '''
    temporary function to set output
    '''
    output = random.randint(100, 1000)
    return output



def get_capcost(g, node, tech, year):
    output = g.nodes[node][year]["technologies"][tech]["Output"]["year_value"]
    overnight_cc = g.nodes[node][year]["technologies"][tech]["Capital cost"]["year_value"]
    crf = g.nodes[node][year]["technologies"][tech]["CRF"]["year_value"]

    cap_cost = (overnight_cc/output)*crf
    return cap_cost



def get_lcc(g, node, tech, year):
    tech_vals = g.nodes[node][year]["technologies"][tech]
    cap_cost = tech_vals["Capital cost"]["year_value"]
    service_cost = tech_vals["Service cost"]["year_value"]
    op_cost = tech_vals["Operating cost"]["year_value"]
    lcc = cap_cost + service_cost + op_cost
    return lcc


def get_marketshare(g, node, tech, year, v=10):
    techs = find_value(g, node, 'technologies', year)
    sum_lcc = 0
    for alltech in techs.keys():
        lcc = g.nodes[node][year]["technologies"][alltech]["LCC"]["year_value"]
        print(f"lcc: {lcc}")
        print(alltech)

        if lcc == None:
            year = str( int(year) - 5)
            lcc = g.nodes[node][year]["technologies"][alltech]["LCC"]["year_value"]
        if lcc:
            lcc_v = lcc**(-1.0*v)
        else:
            lcc_v = 0.0

        sum_lcc += lcc_v
        print(f"sum lcc: {sum_lcc}")
    tech_lcc = g.nodes[node][year]["technologies"][alltech]["LCC"]["year_value"]
    if tech_lcc == 0.0 or tech_lcc == None:
        tech_lcc_v = 0.0
    else:
        tech_lcc_v = tech_lcc**(-1.0*v)
    if sum_lcc > 1e-16:
        market_share = tech_lcc/sum_lcc
    elif sum_lcc < 0.0:
        print("weird things in marketshare")
        market_share = 0.0
    else:
        # temporary fix on float point err
        sum_lcc = 1e-10
        market_share = tech_lcc/sum_lcc
    return market_share







class Model:
    """
    Relevant dataframes and associated information taken from the model description provided in `reader`. Also includes
    methods needed for building and running the Model.

    Parameters
    ----------
    reader : pyCIMS.Reader
        The Reader set up to ingest the description (excel file) for our model.

    Attributes
    ----------
    graph : networkx.DiGraph
        Model Graph populated using the `build_graph` method. Model services are nodes in `graph`, with data contained
        within an associated dictionary. Structural and Request/Provide relationships are edges in the `graph`.

    node_dfs : dict {str: pandas.DataFrame}
        Node names (branch form) are the keys in the dictionary. Associated DataFrames (specified in the excel model
        description) are the values. DataFrames do not include 'Technology' or 'Service' information for a node.

    tech_dfs : dict {str: dict {str: pandas.DataFrame}}
        Technology & service information from the excel model description. Node names (branch form) are keys in
        `tech_dfs` to sub-dictionaries. These sub-dictionaries have technology/service names as keys and pandas
         DataFrames as values. These DataFrames contain information from the excel model description.

    fuels : list [str]
        List of supply-side sector nodes (fuels, etc) requested by the demand side of the Model Graph.  Populated using
        the `build_graph` method.

    years : list [str or int]
        List of the years for which the model will be run.

    # TODO: Fill this in once we figure out how we want to do results
    results :

    """

    def __init__(self, reader):
        self.graph = nx.DiGraph()
        self.node_dfs, self.tech_dfs = reader.get_model_description()
        self.fuels = []
        self.years = reader.get_years()
        self.results = {}   # TODO: POPULATE THIS

    def build_graph(self):
        """ Populates self.graph with nodes & edges and sets self.fuels to a list of fuel nodes.

        Returns
        -------
        None

        """
        def is_year(cn: str or int) -> bool:
            """ Determines whether `cn` is a year

            Parameters
            ----------
            cn : int or str
                The value to check to determine if it is a year.

            Returns
            -------
            bool
                True if `cn` is made entirely of digits [0-9] and is 4 characters in length. False otherwise.

            Examples
            --------
            >>> is_year(1900)
            True

            >>> is_year('2010')
            True
            """
            re_year = re.compile(r'^[0-9]{4}$')
            return bool(re_year.match(str(cn)))

        def make_nodes():
            """
            Add nodes to `self.graph` using `self.node_dfs` and `self.tech_dfs`.

            Returns
            -------
            networkx.Graph
                An updated self.graph that contains all nodes and technologies in self.node_dfs and self.tech_dfs.
            """
            def add_node_data(current_node):
                """ Add and populate a new node to `self.graph`

                Parameters
                ----------
                current_node : str
                    The name of the node (branch format) to add.

                Returns
                -------
                networkx.Graph
                    `self.graph` with `node` added, along with its associated data.
                """
                # 1 Copy the current graph & the current node's dataframe
                current_node_df = copy.copy(self.node_dfs[current_node])
                graph = copy.copy(self.graph)

                # 2 Add an empty node to the graph
                graph.add_node(current_node)

                # 3 Find node type (supply, demand, or standard)
                typ = list(current_node_df[current_node_df['Parameter'] == 'Sector type']['Value'])
                if len(typ) > 0:
                    graph.nodes[current_node]['type'] = typ[0].lower()
                else:
                    # If type isn't in the node's df, try to find it in the ancestors
                    val = find_value(graph, current_node, 'type')
                    graph.nodes[current_node]['type'] = val if val else 'standard'
                # Drop Demand row
                current_node_df = current_node_df[current_node_df['Parameter'] != 'Sector type']

                # 4 Find node's competition type. (If there is one)
                comp_list = list(current_node_df[current_node_df['Parameter'] == 'Competition type']['Value'])
                if len(set(comp_list)) == 1:
                    comp_type = comp_list[0]
                    graph.nodes[current_node]['competition_type'] = comp_type.lower()
                elif len(set(comp_list)) > 1:
                    print("TOO MANY COMPETITION TYPES")
                # Get rid of competition type row
                current_node_df = current_node_df[current_node_df['Parameter'] != 'Competition type']

                # 5 For the remaining data, group by year.
                years = [c for c in current_node_df.columns if is_year(c)]          # Get Year Columns
                non_years = [c for c in current_node_df.columns if not is_year(c)]  # Get Non-Year Columns

                for y in years:
                    year_df = current_node_df[non_years + [y]]
                    year_dict = {}
                    for param, src, branch, unit, val, year_value in zip(*[year_df[c] for c in year_df.columns]):
                        if param in year_dict.keys():
                            pass
                        else:
                            year_dict[param] = {}

                        dct = {'source': src,
                               'branch': branch,
                               'unit': unit,
                               'year_value': year_value}

                        year_dict[param][val] = dct
                    # Add data to node
                    graph.nodes[current_node][y] = year_dict

                # 6 Return the new graph
                return graph

            def add_tech_data(node, tech):
                """
                Add and populate a new technology to `node`'s data within`self.graph`
                Parameters
                ----------
                node : str
                    The name of the node the new technology data will reside in.
                tech : str
                    The name of the technology being added to the graph.
                Returns
                -------
                networkx.Graph
                    `self.graph` with the data for `tech` contained within `node`'s node data

                """
                # 1 Copy the current graph & the current tech's dataframe
                t_df = copy.copy(self.tech_dfs[node][tech])
                graph = copy.copy(self.graph)

                # 2 Remove the row that indicates this is a service or technology.
                t_df = t_df[~t_df['Parameter'].isin(['Service', 'Technology'])]

                # 3 Group data by year & add to the tech's dictionary
                # NOTE: This is very similar to what we do for nodes (above). However, it differs because here we aren't
                # using the value column (its redundant here).
                years = [c for c in t_df.columns if is_year(c)]             # Get Year Columns
                non_years = [c for c in t_df.columns if not is_year(c)]     # Get Non-Year Columns

                for y in years:
                    year_df = t_df[non_years + [y]]
                    year_dict = {}

                    for parameter, source, branch, unit, value, year_value in zip(
                            *[year_df[c] for c in year_df.columns]):
                        dct = {'source': source,
                               'branch': branch,
                               'unit': unit,
                               'year_value': year_value}

                        if parameter in year_dict.keys():
                            if type(year_dict[parameter]) is list:
                                year_dict[parameter] = year_dict[parameter].append(dct)
                            else:
                                year_dict[parameter] = [year_dict[parameter], dct]
                        else:
                            year_dict[parameter] = dct

                    # Add technologies key (to the node's data) if needed
                    if 'technologies' not in graph.nodes[node][y].keys():
                        graph.nodes[node][y]['technologies'] = {}

                    # Add the technology specific data for that year
                    graph.nodes[node][y]['technologies'][tech] = year_dict

                # 4 Return the new graph
                return graph

            # 1 Copy graph
            new_graph = copy.copy(self.graph)

            # 2 Add nodes to the graph
            for n in self.node_dfs.keys():
                new_graph = add_node_data(n)

            # 3 Add technologies to the graph
            for node in self.tech_dfs:
                # Add technologies key to node data
                for tech in self.tech_dfs[node]:
                    new_graph = add_tech_data(node, tech)

            # Return the graph
            return new_graph

        def make_edges():
            """
            Add edges to `self.graph` using information in `self.node_dfs` and `self.tech_dfs`.

            Returns
            -------
            networkx.Graph
                An updated `self.graph` that contains all edges defined in `self.node_dfs` and `self.tech_dfs`.

            """
            def add_edges(node, df):
                """ Add edges associated with `node` to `self.graph` based on data in `df`.

                Edges are added to the graph based on: (1) if a node is requesting a service provided by
                another node or (2) the relationships implicit in the branch structure used to identify a node. When an
                edge is added to the graph, we also store the edge type ('request_provide', 'structure') in the edge
                attributes. An edge may have more than one type.

                Parameters
                ----------
                node : str
                    The name of the node we are creating edges for. Should already be a node within self.graph.

                df : pandas.DataFrame
                    The DataFrame we will use to create edges for `node`.

                Returns
                -------
                networkx.Graph
                    An updated version of self.graph with edges associated with `node` added to the graph.
                """
                # 1 Copy the graph
                graph = copy.copy(self.graph)

                # 2 Find edges based on requester/provider relationships
                #   These are the edges that exist because one node requests a service to another node
                # Find all nodes node is requesting services from
                providers = df[df['Parameter'] == 'Service requested']['Branch'].unique()
                rp_edges = [(node, p) for p in providers]
                graph.add_edges_from(rp_edges)

                # Add them to the graph
                for e in rp_edges:
                    try:
                        types = graph.edges[e]['type']
                        if 'request_provide' not in types:
                            graph.edges[e]['type'] += ['request_provide']
                    except KeyError:
                        graph.edges[e]['type'] = ['request_provide']

                # 3 Find edge based on branch structure.
                #   e.g. If our node was pyCIMS.Canada.Alberta.Residential we create an edge Alberta->Residential
                # Find the node's parent
                parent = '.'.join(node.split('.')[:-1])
                if parent:
                    s_edge = (parent, node)
                    # print(s_edge)
                    graph.add_edge(s_edge[0], s_edge[1])
                    # Add the edges type
                    try:
                        types = graph.edges[s_edge]['type']
                        if 'structure' not in types:
                            graph.edges[s_edge]['type'] += ['structure']
                    except KeyError:
                        graph.edges[s_edge]['type'] = ['structure']

                # 4 Return resulting graph
                return graph

            graph = self.graph

            for node in self.node_dfs:
                graph = add_edges(node, self.node_dfs[node])

            for node in self.tech_dfs:
                for tech in self.tech_dfs[node]:
                    graph = add_edges(node, self.tech_dfs[node][tech])

            return graph

        def get_fuels():
            """ Find the names of nodes supplying fuel.

            Currently, this is any node which (1) provides a service whose unit is GJ and (2) is a supply node.
            TODO: Update this once node type has been added to model description. Fuels will be any sector level
                  supply nodes specified within the graph.
            Returns
            -------
            list of str
                A list containing the names of nodes which supply fuels.
            """

            fuels = []
            for n, d in self.graph.nodes(data=True):
                is_supply = d['type'] == 'supply'
                # GJ trick should change later (might not be GJ) - ML!
                prov_gj = any([data['unit'] == 'GJ' for service, data in d[self.years[0]]['Service provided'].items()])
                if is_supply & prov_gj:
                    fuels += [n]
            return fuels

        self.graph = make_nodes()
        self.graph = make_edges()
        self.fuels = get_fuels()

    def run(self, equilibrium_threshold=0.05):
        """
        Runs the entire model, progressing year-by-year until an equilibrium has been reached for each year.

        # TODO: add max_iterations parameter

        Parameters
        ----------
        equilibrium_threshold : float, optional
            The largest relative difference between prices allowed for an equilibrium to be reached. Must be between
            [0, 1]. Relative difference is calculated as the absolute difference between two prices, divided by the
            first price.

        Returns
        -------
        None
            Nothing is returned, but `self.graph` will be updated with the resulting prices, quantities, etc calculated
            for each year.

        """
        def run_year(year):
            """
            Run the model for the given `year` until an equilibrium in fuel prices is reached between iterations.

            The model will be run for the given year, alternating between demand and supply calculations until an
            equilibrium in prices between iterations is reached.

            Parameters
            ----------
            year : int or str
                The year to run the model for.

            Returns
            -------
            None
                Nothing is returned, but `self.graph` will be updated with the resulting prices, quantities, etc.
            """
            def traverse_graph(sub_graph, node_process_func):
                """
                Visit each node in `sub_graph` applying `node_process_func` to each node as its visited.

                A node is only visited once its parents have been visited. In the case of a loop (where every node has
                at least one parent who hasn't been visited) the node closest to the `sub_graph` root will be visited
                and processed using the values held over from the last iteration.

                Parameters
                ----------
                sub_graph : networkx.DiGraph
                    The graph to be traversed.

                node_process_func : function (nx.DiGraph, str) -> None
                    The function to be applied to each node in `sub_graph`. Doesn't return anything but should have an
                    effect on the node data within `sub_graph`.

                Returns
                -------
                None

                """
                # Find the root of the sub-graph
                root = [n for n, d in sub_graph.in_degree() if d == 0][0]

                # Find the distance from the root to each node in the sub-graph
                dist_from_root = nx.single_source_shortest_path_length(sub_graph, root)

                # Start the traversal
                sg_cur = copy.deepcopy(sub_graph)
                visited = []

                while len(sg_cur.nodes) > 0:
                    active_front = [n for n, d in sg_cur.in_degree if d == 0]

                    if len(active_front) > 0:
                        # Choose a node on the active front
                        n_cur = active_front[0]
                        # Process that node in the sub-graph
                        node_process_func(sub_graph, n_cur)
                    else:
                        # Resolve a loop
                        candidates = {n: dist_from_root[n] for n in sg_cur}
                        n_cur = min(candidates, key=lambda x: candidates[x])
                        # Process chosen node in the sub-graph, using estimated values from their parents
                        node_process_func(sub_graph, n_cur, with_estimates=True)

                    visited.append(n_cur)
                    sg_cur.remove_node(n_cur)

            def depth_first_post(sub_graph, node_process_func, prices):
                """
                Visit each node in `sub_graph` applying `node_process_func` to each node as its visited.

                A node is only visited once its parents have been visited. In the case of a loop (where every node has
                at least one parent who hasn't been visited) the node closest to the `sub_graph` root will be visited
                and processed using the values held over from the last iteration.

                Parameters
                ----------
                sub_graph : networkx.DiGraph
                    The graph to be traversed.

                node_process_func : function (nx.DiGraph, str) -> None
                    The function to be applied to each node in `sub_graph`. Doesn't return anything but should have an
                    effect on the node data within `sub_graph`.

                Returns
                -------
                None

                """

                # Find the root of the sub-graph
                root = [n for n, d in sub_graph.in_degree() if d == 0][0]

                # Find the distance from the root to each node in the sub-graph
                dist_from_root = nx.single_source_shortest_path_length(sub_graph, root)

                original_leaves = [n for n, d in sub_graph.out_degree if d == 0]

                for node_name in sub_graph:
                    if node_name in original_leaves:
                        sub_graph.nodes[node_name]["is_leaf"] = True
                    else:
                        sub_graph.nodes[node_name]["is_leaf"] = False


                # Start the traversal
                sg_cur = copy.deepcopy(sub_graph)
                visited = []

                while len(sg_cur.nodes) > 0:
                    active_front = [n for n, d in sg_cur.out_degree if d == 0]

                    if len(active_front) > 0:
                        # Choose a node on the active front
                        n_cur = active_front[0]
                        # Process that node in the sub-graph
                        node_process_func(sub_graph, n_cur, prices)
                        # print(n_cur)
                    else:
                        # Resolve a loop
                        # ML! is this to deal with thigns like furnace? Not sure it's needed
                        candidates = {n: dist_from_root[n] for n in sg_cur}
                        n_cur = min(candidates, key=lambda x: candidates[x])
                        # Process chosen node in the sub-graph, using estimated values from their parents
                        node_process_func(sub_graph, n_cur, state_dict, with_estimates=True)
                        print(f"loop: {n_cur}")
                    visited.append(n_cur)
                    sg_cur.remove_node(n_cur)

            def get_subgraph(node_types):
                """
                Find the sub-graph of `self.graph` that only includes nodes whose type is in `node_types`.

                Parameters
                ----------
                node_types : list of str
                    A list of node types ('standard', 'supply', or 'demand') to include in the returned sub-graph.

                Returns
                -------
                networkx.Graph
                    The returned graph is a sub-graph of `self.graph`. A node is only included if its type is one
                    of `node_types`. A edge is only included if it connects two nodes found in the returned graph.
                """
                nodes = [n for n, a in self.graph.nodes(data=True) if a['type'] in node_types]
                sub_g = self.graph.subgraph(nodes).copy()
                return sub_g

            def calc_demand(prices = None):
                """
                Using `self.graph` calculate the services requested by the demand side of the tree in a given iteration.

                For now, this means getting the quantity of fuel being requested by demand-side nodes.

                Parameters
                ----------
                prices : dict of {str: float}
                    Fuel prices to use for estimates or initialization. (Likely were produced in the last iteration)

                Returns
                -------
                dict of {str: float}
                    A dictionary containing the fuels (keys) requested by the demand side of the tree, along with the
                    quantity of those fuels being requested.

                """


                def get_prices(sub_graph, prices):
                    # fetch starting prices if needed (make more general later)
                    if prices is None:
                        prices = find_value(sub_graph, 'pyCIMS.Canada.Alberta', 'Price', '2000')
                    return prices


                def calculate_service_cost(sub_graph, node, prices):
                    service_cost = {get_name(f): 0.0 for f in self.fuels}

                    # ML: to change once we fix the node types
                    if get_name(node) not in ["pyCIMS", "Canada", "Alberta", "Residential", "Buildings"]:

                        # make this outside function later
                        for key in sub_graph.nodes[node][year]["technologies"].keys():

                            sub_graph.nodes[node][year]["technologies"][key].update({"Service cost": {"year_value": 0.0}})
                            sub_graph.nodes[node][year]["technologies"][key].update({"Output": {"year_value": 0.0}})
                            sub_graph.nodes[node][year]["technologies"][key].update({"CRF": {"year_value": 0.0}})
                            sub_graph.nodes[node][year]["technologies"][key].update({"LCC": {"year_value": 0.0}})
                            sub_graph.nodes[node][year]["technologies"][key].update({"Full capital cost": {"year_value": 0.0}})

                            if sub_graph.nodes[node][year]["technologies"][key]["Operating cost"]["year_value"] == None:
                                sub_graph.nodes[node][year]["technologies"][key]["Operating cost"]["year_value"] = 0.0

                        try:
                            v = sub_graph.nodes[node][year]["Heterogeneity"]["v"]["year_value"]
                        except:
                            v = 10 # default val
                        techdata = find_value(sub_graph, node, 'technologies', year)
                        techs = copy.deepcopy(techdata)
                        check_type(techs, dict, node=node, passing=True)
                        for key, val in techs.items():
                            try:
                                branch_req = val['Service requested']['branch']
                                fuel_req = get_name(branch_req)
                                req_val = val['Service requested']['year_value']
                            except TypeError as err:
                                # ML: Sometimes this is a list (??)
                                if isinstance(val['Service requested'], list):
                                    # ML! I think this only happens when trying to get child of a leaf
                                    break
                                else:
                                    # raise
                                    break

                            if sub_graph.nodes[node]["is_leaf"]:
                                # print(f"leaf name {get_name(node)}")
                                fuel_req = get_name(branch_req)
                                if fuel_req in prices.keys():
                                    # ML! This will be summed when tech in more than 1 node
                                    fuel_price = prices[fuel_req]['year_value']
                                    sc = fuel_price * req_val
                                    service_cost[fuel_req] = sc
                                    s_cost = {"Service cost": {"year_value": sc}}

                                    sub_graph.nodes[node][year]["technologies"][key].update(s_cost)

                                    crf = {"CRF": {"year_value": get_crf(sub_graph, node, key, year)}}
                                    sub_graph.nodes[node][year]["technologies"][key].update(crf)

                                    output = {"Output": {"year_value": get_output()}}
                                    sub_graph.nodes[node][year]["technologies"][key].update(output)

                                    cap_cost = {"Full capital cost": {"year_value": get_capcost(sub_graph, node, key, year)}}
                                    sub_graph.nodes[node][year]["technologies"][key].update(cap_cost)

                                    lcc = {"LCC": {"year_value": get_lcc(sub_graph, node, key, year)}}
                                    sub_graph.nodes[node][year]["technologies"][key].update(lcc)

                                    market_share = {"Market share": {"year_value": get_marketshare(sub_graph, node, key, year, v)}}
                                    sub_graph.nodes[node][year]["technologies"][key].update(market_share)

                            else:
                                pass


                g_demand = get_subgraph(['demand', 'standard'])
                prices = get_prices(g_demand, prices)
                depth_first_post(g_demand, calculate_service_cost, prices)





            def calc_supply(demand):
                def supply_process_func(node, with_estimates=False):
                    pass

                g_supply = get_subgraph(['supply', 'standard'])


                # 3 Find the prices by fuel (Note: Unsure. This might actually happen in the previous step...)
                fuel_prices = {}

                return fuel_prices

            def equilibrium_check(dict1, dict2, threshold):
                """
                Determines whether an equilibrium has been reached between dict1 and dict2.

                Parameters
                ----------
                dict1 : dict of {str: float}
                    A dictionary mapping fuels to their prices. Keys must match keys in `dict2`.
                dict2 : dict of {str: float}
                    A dictionary mapping fuels to their prices. Keys must match keys in `dict1`.
                threshold : float, optional
                    The largest relative difference between prices allowed for an equilibrium to be reached. Must be
                    between [0, 1]. Relative difference is calculated as the absolute difference between two prices,
                    divided by the first price.

                Returns
                -------
                bool
                    True if all fuels have a relative difference <= `threshold` when comparing prices in `dict1` to
                    prices in `dict2`. False otherwise.

                """
                for fuel in dict1:
                    abs_diff = abs(dict1[fuel] - dict2[fuel])
                    rel_diff = abs_diff / dict1[fuel]
                    if rel_diff > threshold:
                        return False

                return True

            prev_prices = None
            equilibrium = False
            while not equilibrium:
                curr_demand = calc_demand(prev_prices)
                curr_prices = calc_supply(curr_demand)
                # equilibrium = equilibrium_check(prev_prices, curr_prices, equilibrium_threshold)
                equilibrium = True

                # prev_prices = curr_prices

            # TODO: Add finishing procedures. Ex. Storing resulting prices and demands

        for y in self.years:
            print(f"year: {y}")
            run_year(y)

    def search_nodes(self, search_term):
        """
        Search `graph.self` to find the nodes which contain `search_term` in the node name's final component. Not case
        sensitive.

        Parameters
        ----------
        search_term : str
            The search term.

        Returns
        -------
        list [str]
            A list of node names (branch format) whose last component contains `search_term`.
        """
        def search(name):
            components = name.split('.')
            last_comp = components[-1]
            return search_term.lower() in last_comp.lower()

        return [n for n in self.graph.nodes if search(n)]

    def aggregate(self, sub_graph, agg_key, agg_func=sum):
        """
        Retrieves values to aggregate from each node in the `sub_graph` using `agg_key`. Then applies the `agg_func` to
        these values to get a final aggregation.
        Parameters
        ----------
        sub_graph : networkx.Graph
            The graph to be aggregated over.

        agg_key : list [str]
            The key list needed to access the values for aggregation.

        agg_func : function (any) -> any
            The aggregation function to apply over the collection of values retreived using `agg_key` from all nodes in
            the `sub_graph`.

        Returns
        -------
        any
            The result from applying `agg_func` to the list of aggregated values retrieved from `sub_graph` using
            `agg_key`.
        """

        def get_nested_values(dict, key_list):
            """Retrieves value from the nested dictionary `dict` using the `key_list`.

            Parameters
            ----------
            dict : dict {str : dict or str}
                A nested dictionary, with at least `len(key_list)` levels of nesting.

            key_list : list [str]
                A list of keys used to access a value in dict. Where elements of `key_list` can be denoted as k0...kn,
                k0 must be a key in `dict`, k1 must be a key in `dict[k0]`, ... , kn must be a key in
                `dict[k0][k1]...[kn-1]`.

            Returns
            -------
            any
                Returns the value found by using the keys found in `key_list` to access a value within the nested
                dictionary `dict`. For example, if `key_list` = [x, y, z] then this function will retrieve the value
                found at `dict[x][y][z]`
            """
            value = dict
            for k in key_list:
                value = value[k]
            return value


        values_by_node = [get_nested_values(data, agg_key) for name, data in sub_graph.nodes(data=True)]

        all_values = [(k, v) for values in values_by_node for k, v in values.items()]

        # Make value lists, separated by key
        value_lists = {}
        for k, v in all_values:
            try:
                value_lists[k].append(v)
            except KeyError:
                value_lists[k] = [v]

        # Aggregate each list by applying agg_function
        aggregates = {k: agg_func(v) for k, v in value_lists.items()}

        return aggregates
