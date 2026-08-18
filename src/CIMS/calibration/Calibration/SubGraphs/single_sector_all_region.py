"""
Single_sector_all_region.py

This needs to extract a sector-specific subtree, for all regions, and then reassemble them all on a common root structure.
This root structure might need to mimic the pre-extraction graph structure (i.e. CIMS.CAN.ON.
"""


import networkx as nx
import numpy as np
import json
import re
import pickle
import gzip

from Calibration.SubGraphs.graphFunctions import getSubgraph

def get_all_region_names(model):
    """
    This function starts from the set of all node addresses, and looks at what the ??s in
    "CIMS.CAN.??." stand in for.
    """
    pattern = r'^CIMS\.CAN\.([A-Za-z]{2})\.'
    allNodeNames = list(model.graph.nodes())
    allMatchesPre = [re.match(pattern, a, flags=re.IGNORECASE) for a in allNodeNames]
    allMatches = sorted(list(set([a.groups()[0] for a in allMatchesPre if a is not None])))
    return allMatches

def get_all_sector_names(model):
    """
    This function starts from the set of all node addresses, matches
    "CIMS.CAN.??." and then extracts what's between that, and the next '.'.
    """
    pattern = r'^CIMS\.CAN\.([A-Za-z]{2})\.(.*?)\.'
    allNodeNames = list(model.graph.nodes())
    allMatchesPre = [re.match(pattern, a, flags=re.IGNORECASE) for a in allNodeNames]
    allMatches = sorted(list(set([a.groups()[1] for a in allMatchesPre if a is not None])))
    return allMatches

def get_single_sector_all_region(model, sectorName, graphTrueRoot = "CIMS"):
    """
    Use a regular expression match to find node addresses that end with `sectorName`. This set of nodes should correspond to each region's
    sector-specific subtree, for that sector. A subGraph is extracted at this root, from each region, and then these are all added
    to a commen root node added just for that purpose.

    Ok actually the artificial common root idea didn't work so well, because in order to do calibration functions on this graph, an updated
    dcc_classes thing is required, and to "fake" re-update this after the model graph is swapped out for a subgraph doesn't work unless ALL the nodes
    in the graph are "official" CIMS nodes. My fake root was throwing a "can't find key 2000" error, because of course there's no real info at
    this node.

    So for the fake root, just use the normal CIMS. CIMS.CAN, CIMS.CAN.?? etc structure. Easiest way to do that is to just figure out 
    what these nodes and edges are, maybe using `nx.shortest_path` as before, and explicitly add these in while processing each region/sector
    subgraph.
    """

    allNodeNames = list(model.graph.nodes())
    allSectorRoots = [a for a in allNodeNames if re.match(f".*?\\.{sectorName}$", a, flags=re.IGNORECASE)]
    allSectorSubgraphs = [nx.DiGraph(getSubgraph(model, a)['s1']) for a in allSectorRoots]

    rootGraph = nx.DiGraph()
    #rootGraph.add_node('root')

    for subGRootName,subG in zip(allSectorRoots, allSectorSubgraphs):
        # Add the sector subgraph nodes and edges
        rootGraph.add_nodes_from(subG.nodes(data=True))
        rootGraph.add_edges_from(subG.edges(data=True))
        # Figure out the path connecting the subgraph to the `graphTrueRoot`
        rootConnSubgraph = nx.subgraph(model.graph, nx.shortest_path(model.graph, graphTrueRoot, subGRootName))
        rootGraph.add_nodes_from(rootConnSubgraph.nodes(data=True))
        rootGraph.add_edges_from(rootConnSubgraph.edges(data=True))

    model_out = model
    model_out.graph = rootGraph
    model_out.dcc_classes = model_out._dcc_classes()
    return model_out

def write_single_sector_all_region_pickle(model, sectorName, output_filepath = None, graphTrueRoot = "CIMS"):
    """
    Extract and assemble the single-sector-multi-region calibration graph structure using the function above, and then
    write it out to a gzipped pickle.
    """
    if output_filepath is None:
        output_filepath = f"allRegions_{sectorName}.pkl"
    with gzip.open(output_filepath, 'wb') as _f:
        pickle.dump(
            get_single_sector_all_region(model, sectorName, graphTrueRoot),
            _f
        )
    print(f"Wrote all region graph for sector {sectorName} to file {output_filepath}.")


