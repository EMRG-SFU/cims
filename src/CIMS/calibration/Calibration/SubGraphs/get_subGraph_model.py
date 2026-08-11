
import networkx as nx
import numpy as np
import json

from Calibration.SubGraphs.graphFunctions import getSubgraph

def get_subGraph_model(model, nodeName):

    model_out = model
    subGraph = getSubgraph(model, nodeName)['s1']

    model_out.graph = nx.DiGraph(subGraph)
    model_out.dcc_classes = model_out._dcc_classes()
    return model_out


def write_subGraph_pickle(model, nodeName, output_filepath = "subGraphModel.pkl"):    
        with open(output_filepath, 'wb') as _f:
            pickle.dump(get_subGraph_model(model, nodeName), _f)

        print(f"Subgraph-containing model written to: {output_filepath}")
