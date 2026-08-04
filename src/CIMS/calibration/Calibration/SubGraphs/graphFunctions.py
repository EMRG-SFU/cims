
import networkx as nx
import numpy as np
import json
#from customTreeData import custom_tree_data

def is_thing_in(edge, typeName):
    if any([typeName.lower() in a.lower() for a in edge]):
        return(True)
    else:
        return(False)

def is_anything_in(edge, typeNameList):
    if any([any([s.lower() in a.lower() for s in typeNameList]) for a in edge]):
        return(True)
    else:
        return(False)
        
# This one just EXCLUDES an edge
def is_thing_not_in(edge, typeName):
    return( not is_thing_in(edge,typeName) )

# True if the edge has just a single type of exactly `typeName`
def is_thing_only(edge, typeName):
    #from IPython import embed; embed(header="blaaah")
    if (len(edge) == 1) and (edge[0] == typeName):
        return(True)
    else:
        return(False)



def getSubgraph( model, rootName ):
    """
    This subgraph extraction function is a copy (modified copy, soon) of the code in `vizServer/src/server.py`. There we extract
    subgraphs as needed using nx, and send them via the Flask layer to the d3.js. We end up with everything we need *for display* over
    there, but I don't believe it will work for calibration, because things like the "Generic Fuels" nodes don't end up in the graph.

    Running the LCC computations, the stock allocation and retirement function, and then the aggregation traversal on the graph is 
    what's necessary, and when a needed node is missing it throws an error. These operations need to complete for calibration to work, but
    that they work is no guarantee of calibration working *correctly*. Still need to implement tests vs a full CIMS run on the full
    graph.

    We are NOT currently going to apply `addTech` to the graph object here. Not sure if doing that will affect calibration (probably not, as
    they're attached using different edge types that regular CIMS doesn't know anything about). We want to be able to use this function to
    generate CER's requested calibration subgraphs, which are sector-specific but include all regions.
    """

    # HELPER FUNCTIONS for filtering edges. These are called with `t` being what comes out of graph.edges.get( edge )['edge'], which
    # seems to be a list of all the different edge types that edge is.

    # Is the single string `s` contained in any of the strings in the list of strings `t`
    # Useful for filtering edges that MUST contain the string (edge type) `s`
    def is_thing_in(t, s):
        if any([s.lower() in a.lower() for a in t]):
            return(True)
        else:
            return(False)

    # Is at least one of the strings in `sList` contained in any of the strings in list of strings `t`
    # Useful for filtering out edges of different types, but that each must contain at least one of the strings.
    def is_anything_in(t, sList):
        if any([any([s.lower() in a.lower() for s in sList]) for a in t]):
            return(True)
        else:
            return(False)
            
    # This one just NOTs `is_thing_in
    def is_thing_not_in(t, s):
        return( not is_thing_in(t,s) )

    def getEdge( edgeName ):
        try:
            return model.graph.edges().get(edgeName)['edge']
        except Exception as ee:
            print(f"Failing edge extraction for {edgeName}.")
            print(f"Failing edge info dict looks like: {model.graph.edges().get(edgeName)}")
            raise


    # Extract some groups of edge types

    # The "new" graphs from data-processing branch and CER data only contain these two edge types.
    ee_all = [e for e in model.graph.edges() if is_anything_in(getEdge(e), ['structural', 'request_provide'])]

    allNodesSet = list(nx.dfs_preorder_nodes( model.graph.edge_subgraph( ee_all ), rootName))

    # First subgraph is where we just extract it from the full graph via the node set
    subg_1 = model.graph.subgraph( allNodesSet )
    # Second one is where we take the previous graph, and grab an edge_subgraph out of it, but using all the edges. This
    # should not filter anything and it should be the same size as the previous
    subg_2 = subg_1.edge_subgraph( ee_all )

    # This isn't really going to be useful, as you're just taking the edge subgraph from the full model graph, where
    # the preorder_nodes node filtering step hasn't been done. So this just essentially gives you the whole graph back.
    subg_3 = model.graph.edge_subgraph( ee_all )

    return {
            's1': subg_1,
            's2': subg_2,
            's3': subg_3
            }


#############################
############################
#############################
############################
#
#     This is the weirdly-behaving function from the vizServer.

def subGraphRootedAt( modelGraph, rootName ):

    #g = addTech(modelGraph)[0]
    g = modelGraph

    def tryThing(blah):
        #print(f"We are trying: {blah}")
        try:
            return( g.edges.get(blah)['edge'] )
        except Exception as e:
            print(f"Failing Inner bit e is: {blah}")
            print(f"Failing Inner bit bla is: {g.edges.get(blah)}")
            raise
    
    try:
        ee_all = [e for e in g.edges() if is_anything_in(tryThing(e), ['tech','structural','request_provide'])]
        ee_tree = [e for e in g.edges() if is_anything_in(g.edges.get(e)['edge'], ['tech', 'structural'])]
        ee_rp_only = [e for e in g.edges() if is_thing_in(g.edges.get(e)['edge'], 'request_provide') and is_thing_not_in(g.edges.get(e)['edge'], 'structural')]
    except Exception as e:
        print(f"The edge is: {e}")
        print(f"All edges: {list(g.edges)[0:100]}")
        print(f"The full edge contents is: {g.edges.get(e)}")
        raise

    # This one is a subgraph out of just the structural and tech edges.
    # This will not contain links to any out-of-sector/region nodes AT ALL, therefore none
    # of these nodes make it into the set for the `dfs_preorder_nodes`
    gsub_st = g.subgraph(list(nx.dfs_preorder_nodes( g.edge_subgraph(ee_tree), rootName)))

    # This one is a subgraph out of just the requst_provide edges.
    # Is over ALL the nodes.

    #gsub = g.subgraph(list(nx.dfs_preorder_nodes( g.edge_subgraph(ee_rp_only), rootName)))
    fromRoot = g.subgraph(list(nx.dfs_preorder_nodes(g.edge_subgraph(ee_all), rootName)))
    gsub = fromRoot.edge_subgraph(ee_rp_only)

    ss = {#'tree' : custom_tree_data(gsub_st.edge_subgraph(ee_tree), rootName),
                   'graph':{
                       'nodes':[{'name': str(x) , 'index':i} for i,x in enumerate(gsub.nodes())],
                       'links':[{'source':u[0], 'target':u[1]} for u in gsub.edges()]
                       },
          'gsub': gsub,
          'fromRoot': fromRoot,
          'gsub_st': gsub_st
          }
    return(ss)


#############################
############################
#############################
############################


def outputGraphRootedAt( model, rootName, trueRoot):
    """
    As above, but here `trueRoot` is the root of the FULL graph (so, `CIMS`), and `rootName` is where you want the
    subgraph to be rooted -- the full tree is included downstream of `rootName`, but we also maintain a single simple path
    back up to `trueRoot`. The model requires this to run properly.
    """
    #g = addTech(model.graph)[0]
    g = model.graph
    ee_all = [e for e in g.edges() if is_anything_in(g.edges.get(e)['edge'], ['tech','structural','request_provide'])]
    ee_tree = [e for e in g.edges() if is_anything_in(g.edges.get(e)['edge'], ['tech', 'structural'])]
    ee_rp_only = [e for e in g.edges() if is_thing_in(g.edges.get(e)['edge'], 'request_provide') and is_thing_not_in(g.edges.get(e)['edge'], 'structural')]
    ee_struct_only = [e for e in g.edges() if is_thing_in(g.edges.get(e)['edge'], 'structural')]

    #testWTF = g.edge_subgraph(ee_struct_only)
    #testWTF2 = g.subgraph(list(nx.dfs_preorder_nodes( g.edge_subgraph(ee_struct_only), rootName)))
    #from IPython import embed; embed(header="check graph")

    # Be careful here. If you filter down to a subgraph using `edge_subgraph`, then add custom nodes, and then re-call
    # `g.subgraph`, you now get back all those edges you filtered away in the first place. Need to call the second subgraph
    # on the already made `edge_subgraph`.


    # Straight-line simple subgraph connecting `rootName` and `trueRoot`.
    extraSubgraph = nx.subgraph(g, nx.shortest_path(g, trueRoot, rootName))
    # Just extract the nodes and edges (separately) of that
    extraNodes = [n for n in extraSubgraph.nodes()]
    extraEdges = [e for e in extraSubgraph.edges()]

    structEdgeSubgraph = g.edge_subgraph(ee_struct_only + extraEdges)

    # This one is a subgraph out of just the structural and tech edges, plus the "extra" edges that we obtained
    # from the shortest_path subgraph above.
    #gsub_all = g.subgraph(list(nx.dfs_preorder_nodes( g.edge_subgraph(ee_struct_only + extraEdges), rootName))+extraNodes)

    nrs = g.subgraph(list(nx.dfs_preorder_nodes(g.edge_subgraph(ee_struct_only), rootName)))
    nrs_structEdges = [e for e in nrs.edges() if is_thing_in(nrs.edges.get(e)['edge'], 'structural')]

    comboEdgeSubgraph = g.edge_subgraph(nrs_structEdges + extraEdges)



    #gsub_all = g.subgraph(list(nx.dfs_preorder_nodes( g.edge_subgraph(ee_struct_only), rootName)))

    #return(gsub_all)
    return(comboEdgeSubgraph)


def outputGraphRootedAt_subgraphOnly( model, rootName):
    """
    `rootName` is where you want the subgraph to be rooted -- the full tree is included downstream of `rootName`.
    
    """
    #g = addTech(model.graph)[0]
    g = model.graph
    ee_all = [e for e in g.edges() if is_anything_in(g.edges.get(e)['edge'], ['tech','structural','request_provide'])]
    ee_tree = [e for e in g.edges() if is_anything_in(g.edges.get(e)['edge'], ['tech', 'structural'])]
    ee_rp_only = [e for e in g.edges() if is_thing_in(g.edges.get(e)['edge'], 'request_provide') and is_thing_not_in(g.edges.get(e)['edge'], 'structural')]
    ee_struct_only = [e for e in g.edges() if is_thing_in(g.edges.get(e)['edge'], 'structural')]

    structEdgeSubgraph = g.edge_subgraph(ee_struct_only)

    # This one is a subgraph out of just the structural and tech edges, plus the "extra" edges that we obtained
    # from the shortest_path subgraph above.
    #gsub_all = g.subgraph(list(nx.dfs_preorder_nodes( g.edge_subgraph(ee_struct_only + extraEdges), rootName))+extraNodes)

    nrs = g.subgraph(list(nx.dfs_preorder_nodes(g.edge_subgraph(ee_struct_only), rootName)))
    nrs_structEdges = [e for e in nrs.edges() if is_thing_in(nrs.edges.get(e)['edge'], 'structural')]

    # This is returned, and has the same number of nodes as the `nrs` subgraph but around half the edges. This is the edge subgraph
    # created from the edges in `nrs`. So... I'm a bit confused as to why there are 
    retGraph = g.edge_subgraph(nrs_structEdges)



    #gsub_all = g.subgraph(list(nx.dfs_preorder_nodes( g.edge_subgraph(ee_struct_only), rootName)))

    #return(gsub_all)
    #return(retGraph)
    return({'nrs':nrs, 
            'nrs_structEdges':nrs_structEdges,
           'retGraph':retGraph})


#def write_graph_json( g, rootName, fname ):
#    """
#    `g`: the graph object to write out
#    `rootName`: the name of the root node. We seem to need this to be able to write out the tree properly. I think this needs
#                to be the ROOT root though (like 'CIMS').
#    `fname`: the filepath to write it out to.
#    """
#    with open(fname, 'w') as f:
#        json.dump({'tree': custom_tree_data(g, rootName),
#                   'graph':{
#                       'nodes':[{'name': str(x), 'index':i} for i,x in enumerate(g.nodes())],
#                       'links':[{'source':u[0], 'target':u[1]} for u in g.edges()]
#                       }
#                   }, f, indent=4,)


def write_graphOnly_json( g, rootName, fname ):
    """
    The above function actually writes the nodes/edges of the graph, and a tree structure for the hierarchy. This tree
    struct is messing up, so here we want to only write out the nodes/edges to have a look for debugging.

    `g`: the graph object to write out
    `rootName`: the name of the root node. We seem to need this to be able to write out the tree properly. I think this needs
                to be the ROOT root though (like 'CIMS').
    `fname`: the filepath to write it out to.
    """
    with open(fname, 'w') as f:
        json.dump({'graph':{
                       'nodes':[{'name': str(x), 'index':i} for i,x in enumerate(g.nodes())],
                       'links':[{'source':u[0], 'target':u[1]} for u in g.edges()]
                       }
                   }, f, indent=4,)



def getParamVal( g, nodeName, paramName, year):
    """
    Fetches the single specific value for the parameter at the given year, inside the node given by `nodeName`.
    ::TODO:: What to do about the cases (as below) where nodeName/paramName or nodeName/paramName/year do not resolve to a unique value?
             This occurs when there's additional nesting based on the `target` or `context`, for things like quantities and emissions.
    """

    raise NotImplemented("getParamVal hasn't been implemented yet.")

def setParamVal( g, nodeName, paramName, year):
    """
    Sets the single specific value for the parameter at the given year, inside the node given by `nodeName`.
    Year is optional (nodes can have params that are independent of time).
    ::TODO:: General fix up.
    ::TODO:: How are params nested under context and subcontext supposed to work in here? (Cases where node/param and maybe (node/param/year) resolve to 
             multiple values (a dict? What's the structure of this multiplicity?)
    """

    raise NotImplemented("getParamVal hasn't been implemented yet.")

# Below is how this function was used in the Calibration `Reference.py` module, for saving and visualizing
# cims subgraphs. `outputGraphRootedAt_orig` is long gone.
#
### try:
###     outputGraphRootedAt_orig(model, rootName='CIMS.CAN.BC.Coal Mining', outputFile='can.bc.coalMining.all.json')
###     outputGraphRootedAt_orig(model, rootName='CIMS.CAN.ON.Metal Smelting', outputFile='can.on.metalSmelting.all.json')
###     outputGraphRootedAt_orig(model, rootName='CIMS.CAN.BC.Construction', outputFile='can.bc.construction.all.json')
###     outputGraphRootedAt_orig(model, rootName='CIMS.CAN.ON', outputFile='can.on.all.json')
###     outputGraphRootedAt_orig(model, rootName='CIMS.CAN.ON.Electricity', outputFile='can.on.electricity.json')
### except Exception as e:
###     wtfE = e
###     print(f"Exception happened: {e}")
###     print("breaking...")
###     raise
