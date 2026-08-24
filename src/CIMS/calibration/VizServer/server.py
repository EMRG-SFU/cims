#!/usr/bin/env python
# coding: utf-8

import os
import flask
from flask import Flask
from flask import session
from flask import current_app
from flask import request
from flask import make_response
from flask import jsonify

import time
import pickle
import gzip
import json
import sys
import networkx as nx

from threading import Thread
from .html_table_util import make_service_table_context,\
                            make_tech_table_context,\
                            make_tech_table_context2,\
                            make_fic_table_context,\
                            make_emissions_table_context,\
                            make_requestedQuantities_table_context

#import utility_functions as uf
from . import utility_functions as uf

from .customTreeData import addTech, custom_tree_data
import CIMS.lcc_calculation as LCC
from .plotting import plot_ms_for_node_stack, plot_ms_for_node_line


# Going to see if this will help the templates to get found when this business is all 
# installed as a package.
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
TEMPLATE_PATH = os.path.join(BASE_DIR, "templates")
STATIC_PATH = os.path.join(BASE_DIR, "static")

#app = Flask("modelAPI", template_folder="templates/")
app = Flask("modelAPI", static_folder=STATIC_PATH, template_folder=TEMPLATE_PATH)


#print(f"Opening graph pickle file for loading...")
#with open('data/savedModel_withCalibration_post.pickle', 'rb') as f:
##with open('data/model_postRun.graph.pickle', 'rb') as f:
#    wholeModel = pickle.load(f)
#    modelGraph = wholeModel.graph
#print(f"Finished loading graph.")
#
#print(f"Setting current_app.modelGraph context...")
#with app.app_context():
#    current_app.modelGraph = modelGraph
#    current_app.wholeModel = wholeModel
#    current_app.testList = []
#print("Done.")



print("Defining utility functions and routing methods for flask...")
def is_thing_in(t, s):
    if any([s.lower() in a.lower() for a in t]):
        return(True)
    else:
        return(False)

def is_anything_in(t, sList):
    if any([any([s.lower() in a.lower() for s in sList]) for a in t]):
        return(True)
    else:
        return(False)
        
# This one just EXCLUDES an edge
def is_thing_not_in(t, s):
    return( not is_thing_in(t,s) )

def getFICDict(model, nodeName):
    """
    `nodeName` should be the name of a service node. This functions gets the value of the FIC parameter for each of the technologies found at
    this service node, and returns the results in a dict keyed using the technology name.
    """
    allTechNames = uf.getAllTechNames(model.graph, nodeName)
    allYearVals = uf.getAllNodeYears(model.graph, nodeName, asStr=True)
    node_fics = {tn:{yv:model.get_param('fic', nodeName, year=yv, tech=tn) for yv in allYearVals} for tn in allTechNames}
    return(node_fics)

def updateModelWithFICs(model_in, node_name, newData, cims_funcs):
    """
    `newData` is a dually nested dict, with the outermost layer being the tech, and the innermost being the year (as a string). The values in here are
    also strings. We need to iterate through this thing, and issue a `set_param` call, properly addressed, for each of these values.
    """
    model = model_in
    print(f"processing new data: {newData}")
    for tech_key in newData.keys():
        for year_key in newData[tech_key].keys():
            v = float(newData[tech_key][year_key])
            cims_funcs.set_param_calibration.set_param_calibration(model, v, 'fic', node_name, year=year_key, tech=tech_key, save=False)

    return(model)

def rerunCIMSModel(model_in, node_name):
    """
    Same functionality as the objective functions in the parameter optimization project. Here simply re-run those objective
    function steps (LCC calculation, and stock_allocation_and_retirement) needed for updating the market share values.
    """
    model = model_in
    allYearVals = uf.getAllNodeYears(model.graph, node_name, asStr=True)
    for yy in allYearVals:
        print(f"Recalculating market shares at: {node_name} for year {yy}")
        LCC.lcc_calculation(model.graph, node=node_name, year=yy, model=model)
        model.stock_allocation_and_retirement(model.graph, node=node_name, year=yy)
        LCC.lcc_calculation(model.graph, node=node_name, year=yy, model=model)
    return(model)

def subGraphRootedAt( modelGraph, rootName ):

    g = addTech(modelGraph)[0]

    def tryThing(blah):
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

    ss = {'tree' : custom_tree_data(gsub_st.edge_subgraph(ee_tree), rootName),
                   'graph':{
                       'nodes':[{'name': str(x) , 'index':i} for i,x in enumerate(gsub.nodes())],
                       'links':[{'source':u[0], 'target':u[1]} for u in gsub.edges()]
                       }
                   }
    return(ss)




@app.route('/graphTest/')
def graphTest():
    with app.app_context():
        return([a for a in current_app.modelGraph.nodes()][0:10])

@app.route('/getAllNodeNames/')
def getAllNodeNames():
    with app.app_context():
        nodeNameList = sorted([a for a in current_app.modelGraph.nodes()])
        response = flask.jsonify({'nodeList': nodeNameList})
        response.headers.add('Access-Control-Allow-Origin', '*')
    return(response)

@app.route('/getAllParamNames/')
def getAllParamNames():
    with app.app_context():
        nodeNameList = [a for a in current_app.modelGraph.nodes()]
        nodeParamUnionList = uf.union_of_sublists(
            [uf.listYearNodeParams_union(
                current_app.modelGraph, 
                nVar
            ) for nVar in nodeNameList]
        )

        techParamUnionList = uf.union_of_sublists(
            [uf.union_of_sublists([uf.listTechParams_union(
                current_app.modelGraph,
                nVar,
                tVar
            ) for tVar in uf.getAllTechNames(current_app.modelGraph, nVar)])
              for nVar in nodeNameList]
        )

        response = flask.jsonify({'nodeParams': sorted(nodeParamUnionList),
                                  'techParams': sorted(techParamUnionList)})
        response.headers.add('Access-Control-Allow-Origin', '*')
    return(response)
        

@app.route('/getNode/<string:node_name>')
def getNode(node_name):
    with app.app_context():
        #pass
        retNode = current_app.modelGraph.nodes.get(f"{node_name}")
        response = flask.jsonify(retNode)
        response.headers.add('Access-Control-Allow-Origin', '*')
    return(response)

@app.route('/getSubgraph/<string:rootNode_name>')
def getSubgraph(rootNode_name):
    with app.app_context():
        info = subGraphRootedAt(current_app.modelGraph, rootNode_name)
        response = flask.jsonify(info)
        response.headers.add('Access-Control-Allow-Origin', '*')
    return(response)
    




# Store the node address when user clicks on it.

@app.route('/setNodeSelection/<string:node_name>')
def setNodeSelection(node_name):
    with app.app_context():
        current_app.vizVars['selectedNode'] = node_name
        ret = {'selectedNode': node_name, 'status': 'success'}
        response = flask.jsonify(ret)
        response.headers.add('Access-Control-Allow-Origin', '*')
    return(response)








# EMISSIONS TABLES

@app.route('/getEmissionsTable/<string:node_name>')
def getEmissionsTable(node_name):
    with app.app_context():
        retNode = current_app.modelGraph.nodes.get(f"{node_name}")
        response = flask.Response(flask.render_template('serviceTable.html', context=make_emissions_table_context(node_name, retNode)))
        response.headers.add('Access-Control-Allow-Origin','*')
    return(response)

# REQUESTED QUANTITIES TABLES

@app.route('/getRequestedQuantitiesTable/<string:node_name>')
def getRequestedQuantitiesTable(node_name):
    with app.app_context():
        retNode = current_app.modelGraph.nodes.get(f"{node_name}")
        response = flask.Response(flask.render_template('serviceTable.html', context=make_requestedQuantities_table_context(node_name, retNode)))
        response.headers.add('Access-Control-Allow-Origin','*')
    return(response)











@app.route('/getServiceTable/<string:node_name>')
def getServiceTable(node_name):
    with app.app_context():
        retNode = current_app.modelGraph.nodes.get(f"{node_name}")
        response = flask.Response(flask.render_template('serviceTable.html', context=make_service_table_context(node_name, retNode)))
        response.headers.add('Access-Control-Allow-Origin','*')
    return(response)

@app.route('/getServiceTable_filtParam/<string:node_name>')
def getServiceTable_filtParam(node_name):
    with app.app_context():
        raw = request.args.get('params','')
        filtParams = [w.strip() for w in raw.split(',') if w.strip()]
        retNode = current_app.modelGraph.nodes.get(f"{node_name}")
        response = flask.Response(flask.render_template('serviceTable.html', context=make_service_table_context(node_name, retNode, filtParams=filtParams)))
        response.headers.add('Access-Control-Allow-Origin','*')
    return(response)
    

@app.route('/getTechTable_old/<string:node_name>/<string:tech_name>')
def getTechTable_old(node_name, tech_name):
    with app.app_context():
        retNode = current_app.modelGraph.nodes.get(f"{node_name}")
        response = flask.Response(flask.render_template('techTable.html', context=make_tech_table_context(node_name, tech_name, retNode)))
        response.headers.add('Access-Control-Allow-Origin','*')
    return(response)

@app.route('/getTechTable/<string:node_name>')
def getTechTable(node_name):
    with app.app_context():
        retNode = current_app.modelGraph.nodes.get(f"{node_name}")
        print(f"Node name: {node_name}")
        response = flask.Response(flask.render_template('techTable.html', context=make_tech_table_context2(current_app.modelGraph, node_name)))
        response.headers.add('Access-Control-Allow-Origin','*')
    return(response)


@app.route('/getTechTable_filtParam/<string:node_name>')
def getTechTable_filtParam(node_name):
    with app.app_context():
        raw = request.args.get('params', '')
        filtParams = [w.strip() for w in raw.split(',') if w.strip()]
        retNode = current_app.modelGraph.nodes.get(f"{node_name}")
        print(f"Node name: {node_name}")
        response = flask.Response(flask.render_template('techTable.html', context=make_tech_table_context2(current_app.modelGraph, node_name, filtParams=filtParams)))
        response.headers.add('Access-Control-Allow-Origin','*')
    return(response)



@app.errorhandler(404)
def not_found(e):
    # This comes from the `request` object which is implicitly available because of the
    # function decorator used for this.
    attempted_path = request.path

    # Build the JSON payload (feel free to adjust the shape)
    payload = {
        "error": "Not Found",
        "message": f"The URL '{attempted_path}' does not exist on this server." 
    }

    # `make_response` lets us attach headers before returning
    resp = make_response(jsonify(payload), 404)

    # Add the CORS header (or any others you require)
    resp.headers["Access-Control-Allow-Origin"] = "*"
    # Optional: expose additional CORS directives if you need them
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,PUT,DELETE,OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"

    return resp



##############################################################################################
##########################         Calibration Functions         #############################
##############################################################################################

@app.route('/newMethodTest/')
def newMethodTest():
    with app.app_context():
        response = flask.jsonify({'result':42, 'status':'yes_its_true'})
        response.headers.add('Access-Control-Allow-Origin', '*')
    return(response)

@app.route('/getNodeFICsJSON/<string:node_name>')
def getNodeFICsJSON(node_name):
    with app.app_context():
        fic_dict = getFICDict(current_app.wholeModel, node_name)
        response = flask.jsonify(fic_dict)
        response.headers.add('Access-Control-Allow-Origin', '*')
    return(response)

@app.route('/getNodeFICsHTML/<string:node_name>')
def getNodeFICsHTML(node_name):
    with app.app_context():
        response = flask.Response(flask.render_template('ficTable.html', context=make_fic_table_context(current_app.wholeModel, node_name)))
        response.headers.add('Access-Control-Allow-Origin', '*')
    return(response)



def add_preflight_cors_headers(resp):
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    resp.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    # Optional: allow caching of the pre‑flight for 1 hour
    resp.headers['Access-Control-Max-Age'] = '3600'
    return resp

def add_preflight_cors_headers_getOnly(resp):
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['Access-Control-Allow-Methods'] = 'GET'
    resp.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    # Optional: allow caching of the pre‑flight for 1 hour
    resp.headers['Access-Control-Max-Age'] = '3600'
    return resp

# This needs to be a POST request, with the techName:fic pairs as a python dict in JSON. 
@app.route('/setNodeFICs/<string:node_name>', methods=['POST','OPTIONS'])
def setNodeFICs(node_name):
    if request.method == 'OPTIONS':
        return add_preflight_cors_headers(make_response('', 204))
    with app.app_context():
        newFICData = request.get_json()
        current_app.wholeModel = updateModelWithFICs(current_app.wholeModel, node_name, newFICData, cims_funcs = current_app.cims_funcs)
        response = flask.jsonify({'success':True})
        response.headers.add('Access-Control-Allow-Origin', '*')
    return(response)


@app.route('/rerunCIMS/<string:node_name>', methods=['GET'])
def rerunCIMS(node_name):
    with app.app_context():
        updated_model = rerunCIMSModel(current_app.wholeModel, node_name)
        current_app.wholeModel = updated_model
        response = flask.jsonify({'success':True})

    return(add_preflight_cors_headers_getOnly(response))


@app.route('/ms_plotting_stack/<string:node_name>', methods=['GET'])
def ms_plotting_stack(node_name):
    with app.app_context():
        print(f"Generating image for {node_name}.")
        data_uri = plot_ms_for_node_stack(current_app.wholeModel, node_name)
        print(f"Image generated, creating response.")
        #response = flask.Response(image_bytes, mimetype='image/png', headers={'Cache-Control':'no-cache'})
        response = flask.Response(flask.render_template('plotImage.html', context={'data_uri':data_uri}, headers={'Cache-Control':'no-cache'}))
    return(add_preflight_cors_headers_getOnly(response))

@app.route('/ms_plotting_line/<string:node_name>', methods=['GET'])
def ms_plotting_line(node_name):
    with app.app_context():
        print(f"Generating image for {node_name}.")
        data_uri = plot_ms_for_node_line(current_app.wholeModel, node_name)
        print(f"Image generated, creating response.")
        #response = flask.Response(image_bytes, mimetype='image/png', headers={'Cache-Control':'no-cache'})
        response = flask.Response(flask.render_template('plotImage.html', context={'data_uri':data_uri}, headers={'Cache-Control':'no-cache'}))
    return(add_preflight_cors_headers_getOnly(response))


# Here it is! This is the main function that serves up the page that contains the d3 viz.
@app.route('/', methods=['GET'])
def getVizPage():
    base_url = request.host_url
    return( flask.render_template('index_tree.html', base_url=base_url))

@app.route('/getNodeCalFitIdea/', methods=['GET'])
def getNodeCalFitIdea():
    return( flask.render_template('index_nodeCalFitIdea.html'))

@app.route('/getNodeCalFitIdea_2/', methods=['GET'])
def getNodeCalFitIdea_2():
    return( flask.render_template('index_nodeCalFitIdea_2.html'))
##############################################################################################
##############################################################################################
##############################################################################################
#
#  Just some early testing functions
#
@app.route('/incrList/')
def incrList():
    with app.app_context():
        if len(current_app.testList) < 1:
            newVal = 0
            current_app.testList.append(newVal)
        else:
            lastVal = current_app.testList[-1]
            newVal = lastVal + 1
            current_app.testList.append( newVal )
        print(f"testList is: {current_app.testList}")
    return(json.dumps(newVal))

@app.route('/decrList/')
def decrList():
    with app.app_context():
        if len(current_app.testList) < 1:
            decrVal = False
        else:
            decrVal = current_app.testList.pop()
        print(f"testList is: {current_app.testList}")
    return(json.dumps(decrVal))
##############################################################################################
##############################################################################################
##############################################################################################

def run_server(pickle_path, vizVars, PORT=None):
    """
    `pickle_path` is a filepath pointing to a pickled CIMS model object that this flask
        server instance is going to serve.
    """
    start = time.perf_counter()

    with open(pickle_path, 'rb') as f:
        first_two = f.read(2)

    # Try reading in as a gzipped file, falling back to normal pickle if this "magic byte" isn't set,
    # of if it is but the attempt throws an exception.
    if first_two == b'\x1f\x8b':
        try:
            with gzip.open(pickle_path, 'rb') as f:
                wholeModel = pickle.load(f)
                modelGraph = wholeModel.graph

        except (OSError, gzip.BadGzipFile, EOFError):
            with open(pickle_path, 'rb') as f:
                wholeModel = pickle.load(f)
                modelGraph = wholeModel.graph

    else:
        
        with open(pickle_path, 'rb') as f:
            wholeModel = pickle.load(f)
            modelGraph = wholeModel.graph
        
    with app.app_context():
        current_app.modelGraph = modelGraph
        current_app.wholeModel = wholeModel
        current_app.testList = []

        # This is usually supplied by the Marimo notebook, but we don't have one of those here,
        # so just kind of approximate it for now.
        current_app.vizVars = vizVars

    end = time.perf_counter()

    print(f"Loading model took {end - start:.4f} sec.")

    if PORT is not None:
        app.run(debug=True, use_reloader=False, port=PORT)
    else:
        app.run(debug=True, use_reloader=False)

def run_server_modelObject(model_object, 
                           cims_funcs,
                           vizVars,
                           PORT=None):
    """
    `model_object` here is a full CIMS model. This function is designed to be called right from the
        controlling Jupyter/Marimo notebook itself.
    """
    with app.app_context():
        current_app.cims_funcs = cims_funcs
        current_app.wholeModel = model_object
        current_app.modelGraph = model_object.graph
        current_app.testList = []
        current_app.vizVars = vizVars

    if PORT is not None:
        app.run(debug=False, use_reloader=False, port=PORT)
    else:
        app.run(debug=False, use_reloader=False)

#thread = Thread(target=run_server)
##thread.daemon = True
#thread.start()





