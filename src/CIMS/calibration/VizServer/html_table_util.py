import jinja2
import string
import os, os.path
import sys
import pickle
import polars as pl


from functools import reduce

from . import utility_functions as uf


def numFormat(x):
    """
    Turn numbers in these tables into formatted strings having a particular
    number (let's start with 2) decimal places.
    """
    return f"{x:.2f}"  

def make_emissions_table_context(node_name, nodeDict, emissions_key="emissions_total_cumul_net"):
    """
    `nodeDict` is the dict structure that's stored in the nodes of the model graph.
    This function makes a data structure suitable for use with jinja2 html table templates,
    but does not include details about the technologies at that node. Context for that is
    returned by a different function. This one does include the technology NAMES though.
    """

    # Any nodeDict key that can be parsed as an integer we treat as a year.
    def isInt(v):
        try:
            thing = int(v)
        except Exception as e:
            return(False)
        return(True)

    yearHeaders = [a for a in nodeDict.keys() if isInt(a)]
    otherHeaders = [a for a in nodeDict.keys() if not isInt(a)]

    # These are all the rowNames ever found in any year, as a set
    rowNames = set(reduce(lambda x,y: x+y, [[a for a in nodeDict[yy].keys()] for yy in yearHeaders]))

    # Here we're just interested in the `emissions_key` rowName (and this extra structure is the reason that Emissions (and Quantities)
    # always showed up as an error in the 
    allEmDict = {yy:nodeDict[yy][emissions_key]["year_value"].emissions for yy in yearHeaders}

    nodeYearEmissions_pre = [{'year':yk, 'emission':f"{k}_{kk}_{kkk}", 'value':numFormat(vvv['year_value'])} 
     for yk,emDict in allEmDict.items() 
     for k,v in emDict.items()
     for kk,vv in v.items()
     for kkk,vvv in vv.items()]

    nodeYearEmissions = pl.DataFrame(nodeYearEmissions_pre)
    print(nodeYearEmissions)
    nodeYearEmissions_pivot = nodeYearEmissions.pivot(on="year", values="value")

    return({'colNames' : [a for a in nodeYearEmissions_pivot.columns],
            'rows' : [r for r in nodeYearEmissions_pivot.iter_rows()],
            'nodeName': node_name
            })


def make_requestedQuantities_table_context(node_name, nodeDict, rqKey="quantity_requested"):

    # Any nodeDict key that can be parsed as an integer we treat as a year.
    def isInt(v):
        try:
            thing = int(v)
        except Exception as e:
            return(False)
        return(True)

    yearHeaders = [a for a in nodeDict.keys() if isInt(a)]
    otherHeaders = [a for a in nodeDict.keys() if not isInt(a)]

    allQDict = {yy:nodeDict[yy][rqKey]["year_value"].requested_quantities for yy in yearHeaders}

    nodeYearQuantities_pre = [{'year':yk, 'fuel':f"{k}_{kk}", 'value': numFormat(vv)} for yk,qDict in allQDict.items() for k,v in qDict.items() for kk,vv in v.items()]
    nodeYearQuantities = pl.DataFrame(nodeYearQuantities_pre)
    nodeYearQuantities_pivot = nodeYearQuantities.pivot(on="year", values="value")

    return({
        'colNames': [a for a in nodeYearQuantities_pivot.columns],
        'rows': [r for r in nodeYearQuantities_pivot.iter_rows()],
        'nodeName': node_name
        })



def make_service_table_context(node_name, nodeDict, filtParams=None):
    """
    `nodeDict` is the dict structure that's stored in the nodes of the model graph.
    This function makes a data structure suitable for use with jinja2 html table templates,
    but does not include details about the technologies at that node. Context for that is
    returned by a different function. This one does include the technology NAMES though.
    """

    # Any nodeDict key that can be parsed as an integer we treat as a year.
    def isInt(v):
        try:
            thing = int(v)
        except Exception as e:
            return(False)
        return(True)

    yearHeaders = [a for a in nodeDict.keys() if isInt(a)]
    otherHeaders = [a for a in nodeDict.keys() if not isInt(a)]

    rowNames = set(reduce(lambda x,y: x+y, [[a for a in nodeDict[yy].keys()] for yy in yearHeaders]))

    if filtParams is not None:
        rowNames = [a for a in sorted(rowNames) if a in filtParams]
    else:
        rowNames = sorted(rowNames)

    def getMaybe(ff):
        try:
            return(ff())
        except Exception as e:
            #return(None)
            return('Error')

    tbl = [[str(rr)]+[getMaybe(lambda: nodeDict[yy][rr]['year_value']) for yy in yearHeaders] for rr in rowNames]

    return({'colNames' : ['paramName'] + yearHeaders,
            'rows' : tbl,
            'nodeName': node_name
           })



def make_tech_table_context(node_name, tech_name, nodeDict):
    """
    """
    
    def isInt(v):
        try:
            thing = int(v)
        except Exception as e:
            return(False)
        return(True)

    yearHeaders = [a for a in nodeDict.keys() if isInt(a)]
    otherHeaders = [a for a in nodeDict.keys() if not isInt(a)]

    rowNames = set(reduce(lambda x,y: x+y, [[a for a in nodeDict[yy]['technologies'][tech_name].keys()] for yy in yearHeaders]))
    rowNames = sorted(rowNames)

    def getMaybe(ff):
        try:
            return(ff())
        except Exception as e:
            return(None)

    tbl = [[str(rr)] + [getMaybe(lambda: nodeDict[yy]['technologies'][tech_name][rr]['year_value']) for yy in yearHeaders] for rr in rowNames]

    return({'colNames' : ['paramName'] + yearHeaders,
            'rows' : tbl,
            'nodeName': node_name,
            'techName': tech_name
            })


def make_fic_table_context(model, node_name):
    """
    Get all the techs located at service node `node_name`, and return a table with the names of the
    techs as the columns, the years as the rows, and the value of the FIC as the entries.
    This function is more of a helper... here we assemble a convenient set of data structures that the flask.render_template
    function can easily use to build the abovementioned table. This is pretty much identical to how the other functions in this
    module work for making the service and tech tables.
    """
    def getMaybe(ff):
        try:
            return(ff())
        except Exception as e:
            #return(None)
            return(f"Error: {e}")
    graph = model.graph

    allTechNames = uf.getAllTechNames(graph, node_name)
    allYearVals = uf.getAllNodeYears(graph, node_name, asStr=True)

    rowNames = sorted(allTechNames)
    colNames = ['Technology'] + allYearVals
    tbl = [[str(rr)] + [getMaybe(lambda: model.get_param('fic', node_name, year=yv, tech=rr)) for yv in allYearVals] for rr in rowNames]

    return({'colNames': colNames,
            'rows': tbl,
            'nodeName': node_name
            })

def make_tech_table_context2(graph, node_name, filtParams=None):
    """
    This version takes in the whole networkx graph in `graph`, and uses the `tech_name` and associated `node_name`
    to get the tech information needed for rendering the template.
    """
    
    def isInt(v):
        try:
            thing = int(v)
        except Exception as e:
            return(False)
        return(True)

    # Temporarily `node_name is coming in as the full address of the tech, so strip off the last element (tech name), and 
    # get the corresponding service node with what's left.
    node_name2 = ".".join(node_name.split(".")[0:-1])
    # Likewise, tech_name is just the last part of the full node address currently.
    tech_name = node_name.split(".")[-1]

    nodeDict = graph.nodes.get(node_name2)

    yearHeaders = [a for a in nodeDict.keys() if isInt(a)]
    otherHeaders = [a for a in nodeDict.keys() if not isInt(a)]

    rowNames = set(reduce(lambda x,y: x+y, [[a for a in nodeDict[yy]['technologies'][tech_name].keys()] for yy in yearHeaders]))

    if filtParams is not None:
        rowNames = [a for a in sorted(rowNames) if a in filtParams]
    else:
        rowNames = sorted(rowNames)

    def getMaybe(ff):
        try:
            return(ff())
        except Exception as e:
            return(None)

    tbl = [[str(rr)] + [getMaybe(lambda: nodeDict[yy]['technologies'][tech_name][rr]['year_value']) for yy in yearHeaders] for rr in rowNames]

    return({'colNames' : ['paramName'] + yearHeaders,
            'rows' : tbl,
            'nodeName': node_name2,
            'techName': tech_name
            })



