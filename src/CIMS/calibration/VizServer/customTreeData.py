
import VizServer.utility_functions as UF
import networkx as nx

def addTech(g_in):
    """
    # Try build on the above, and get the names of the nested technologies added to the graph as a different sort of node, then see if 
    # this is easy to visualize with the d3 viewer.

    """
    g = g_in.copy()
    edgeCount = 0
    outList = []
    nn = [a for a in g.nodes()]
    for n in nn:
        if 'technologies' in [a for a in g.nodes.get(n)['2000'].keys()]:
            techList = [a for a in g.nodes.get(n)['2000']['technologies'].keys()]
            outList = outList + techList
            for tt in techList:
                # Existing edge type is represented as list, as currently edges can be both `structural` AND `request_provide`
                g.add_node(n+"."+tt, type='tech', parentNode=n, techName=tt)
                g.add_edge(n, n+"."+tt, edge=['tech'])
                edgeCount += 1
    return((g, edgeCount, outList))


def get_calData_info(G, nodeName):
    """
    `G` is graph and `nodeName` is for node dict needed. Search here for calibration data -- this will be
    `calibration_quantity_requested` and `calibration_emissions_by_type` which will be node params
    (under the year keys), and `calibration_market_share_total` within the all the techs at each year. 
    """

    hasCalQuant    = any(['calibration_quantity_requested' in a for a in list(UF.listYearNodeParams_intersect(G, nodeName))])
    hasCalEmission = any(['calibration_emissions_by_type' in a for a in list(UF.listYearNodeParams_intersect(G, nodeName))])
    hasCalMS       = any(['calibration_market_share_total' in a for a in list(UF.list_yearly_techParams_intersect(G, nodeName))])
    return(
            {
                'hasCalQuant': hasCalQuant,
                'hasCalEmission': hasCalEmission,
                'hasCalMS': hasCalMS
            }
    )

def custom_tree_data(G, root, ident="id", children="children"):
    """ 

    # Custom tree_data function, because the default one from networkx.readwrite is
    # unable to put any additional information into the tree data structure aside
    # from the names of the nodes. I want to be able to store the fact that some
    # nodes are tech nodes, so they can easily be identified (coloured, or
    # something) in the d3 viz.
    This is based on the `tree_data` function in networkx.readwrite. It can provide/inse
        rt extra information into the JSON structure, because the build-in version only has the names.

    """
    if G.number_of_nodes() != G.number_of_edges() + 1:
        print(f"Number of nodes: {G.number_of_nodes()}, and edges: {G.number_of_edges()}")
        raise TypeError("G is not a tree.")
    if not G.is_directed():
        raise TypeError("G is not directed.")
    if not nx.is_weakly_connected(G):
        raise TypeError("G is not weakly connected.")

    if ident == children:
        raise nx.NetworkXError("The values for `id` and `children` must be different.")

    def add_children(n, G):
        nbrs = G[n]
        if len(nbrs) == 0:
            return []
        children_ = []
        for child in nbrs:
            edgeType = G.edges.get((n,child))['edge']
            #d = {**G.nodes[child], ident: child}
            # ::TODO:: This is where the emissions/quantities values should be for each node. Get them computed from the datastructure inside
            # this node, and then add them to the `d` dict below.
            nodeInfo = G.nodes()[child]
            d = {ident: child, 'isTechNode': 'tech' in edgeType, 'isReqProv': 'request_provide' in edgeType}
            d.update(get_calData_info(G, child))
            c = add_children(child, G)
            if c:
                d[children] = c
            children_.append(d)
        return children_

    #return {**G.nodes[root], ident: root, children: add_children(root, G)}
    return {ident: root, children: add_children(root, G)}
