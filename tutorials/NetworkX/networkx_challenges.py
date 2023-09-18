import matplotlib.pyplot as plt
import networkx as nx
import CIMS
import copy
import pprint as pp


def draw_net(network):
    plt.close()
    nx.draw(network,
            layout=nx.drawing.spring_layout(network),
            with_labels=True,
            font_color='black',
            font_size=8,
            node_color='seagreen')

    plt.show()


# *****************************************************************************
#  1. Creating a network
# *****************************************************************************
"""
Write code for each of the steps below to create a network based on the one
shown in this link: https://faculty.ucr.edu/~hanneman/nettext/Figure7_11.jpg
"""
# A. Create an empty Directional network

# B. Add nodes to the graph

# C. Add edges to the graph

# D. Use the `draw_net` function (defined in top of file) to draw the network

# *****************************************************************************
# 2. Add data to nodes
# *****************************************************************************

# A. See what data is stored at node "A" in your network

# B. Node data is stored in a dictionary. Add the key value pair
# {'type': 'supply'} to node "A".

# C. Access node A's type to confirm it has been set to 'supply'

# D. Use a for loop to set every node in the graph to set a type for every node in
# the graph. Set type to "supply" for A, B, & C. Set type to "demand" for D, E,
# & F.


# *****************************************************************************
# 3. Working with edges
# *****************************************************************************

# A. Find the total number of edges in your network

# B. Compare the in-degree and out-degree for node C. Ensure you understand the
# difference between the two.

# C. Find all the children of node B.

# D. Remove the edge from B -> E from the network

# E. Use the `draw_net` function (defined in top of file) to draw the network and
# confirm the edge was removed


# *****************************************************************************
# 4. Starting to work with the model graph
# *****************************************************************************
# Run the code below to create a graph based on the model description. It will
# be saved as network.
file = 'CIMS_prototype/CIMS_model_description_20200221.xlsm'
my_reader = CIMS.Reader(infile=file, node_col='Node',
                          sheet_map={'model': 'Model',
                                     'incompatible': 'Incompatible',
                                     'default_tech': 'Technologies'})
my_model = CIMS.Model(my_reader)
my_model.build_graph()
network = my_model.graph

# A. Find all the keys for attributes at node "CIMS.Canada.Alberta"

# B. Check what CIMS.Canada.Alberta.Electricity's competition type is

# C. Check out the data stored within the CIMS.Canada.Alberta.Electricity node.
# See if you can find patterns regarding how data is stored within the year
# data dictionaries


# *****************************************************************************
#  5. Relationships in our model network
# *****************************************************************************
# A. Find the children for the CIMS.Canada.Alberta.Electricity node

# B. Of those children, only print the ones that are connected to
# CIMS.Canada.Alberta through an edge where the edge's "type" attribute
# includes 'request_provide'. Each of these children represent a node where the
# CIMS.Canada.Alberta node is requesting some service from them. Nodes can
# also be connected through "structural" edges.

# C. Find the data stored at the structural parent of 'CIMS.Canada.Alberta'
# There are a few ways to do this, the easiest is below.


# *****************************************************************************
#  6. Generate a Subgraph
# *****************************************************************************
# A. Gather a list of nodes where 'type' is either supply or standard.

# B. Create a subgraph which contains only nodes where type is either supply or
# standard. The network should only contain edges between these nodes.

# C. Use the draw_net to draw the supply graph


# *****************************************************************************
#  7. Understand a Traversal
# *****************************************************************************
# I've included the traverse graph function below (implemented in CIMS).
# However, I have not included the docstring. Its purpose is to visit every
# node in the graph in a particular order, applying a function
# (node_process_func) to each node.
def traverse_graph(sub_graph, node_process_func):
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
            node_process_func(sub_graph, n_cur)

        visited.append(n_cur)
        sg_cur.remove_node(n_cur)


# A. What does the top_down_traversal function take as input?


# B. What does the top_down_traversal function return as output?

# C. Without running the function, what are the first 4 nodes that will be visited
# if we were apply our supply subgraph to the top_down_traversal function?
