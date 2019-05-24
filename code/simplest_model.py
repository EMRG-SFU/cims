%matplotlib inline

import matplotlib.pyplot as plt

import numpy as np
import scipy as sp
import pandas as pd

import os
import json
import cson
from excel2json import convert_from_file

# build graph visualization
import networkx as nx

# local modules
import helper.utils

# ML: will remove later: just an artifact of my computer setting
os.chdir("/Users/maudelachaine/Desktop/bigDataHub/pycims_prototype/code")


'''
Model specified in JSON/CSON file

Have default model, but changes that can be written by the program

Make modifications possible through command line
'''

# This is temporary, will generalize later when prototype works

years = [2005, 2010, 2015, 2020, 2025, 2030, 2035, 2040, 2045, 2050]


class ContainerCIMS:
    '''
    This is the big container that contains all the nodes and connections
    Outputs:
       - Annualized life cycle costs by sector by year by technology
       - Fuel use and emissions by sector by year by technology
       - Service and auxiliary use by sector by year by technology
       - Total stock use by sector by year by technology
       - New stock use by sector by year by technology
       - Retrofits by sector by year by technology
       - Technology cost changes due to declining capital cost function

    Include Supply and Demand Feedbacks later? (10 types of output, see p.14 table 4 in manual)
    '''
    def __init__(self, datapath='data/simplest_model.json',
                       save_dir='saved_models',
                       graph_viz=True,
                       is_notebook=False,
                       make_log=True,
                       *args, **kwargs):

        self.datapath = datapath
        self.save_dir = save_dir
        self.is_notebook = is_notebook
        self.graph_viz = graph_viz

        if make_log:
            # log for debugging: keep path for plotting, but saves a logfile as well
            self.logpath = utils.logged()

       # get data member
       self.read_input()

       if self.graph_viz:
           # initialize graph for visualization - ML: not sure if should make a viz method
           G = nx.MultiDiGraph()

###### LAST MODS TO HERE: building visualization (May 3rd 2019, 3pm) #####


       '''
       Build tree/run model here

       '''




    # parent class constructor (`has-a` relationship)
    Node.__init__(self)


    def read_input(self, printJSON=False):
        '''
        Read datafile
        The file must contain the model structure. This could be changed in a later iteration
        '''

        if self.datapath.endswith('.xls') or self.datapath.endswith('.xlsx'):
            print('converting excel file into json file...')
            convert(self.datapath)    # ML: check that new file is created at the same place, with the same name (except ext)
            rootname, _ = os.path.splitext(self.datapath)
            self.datapath = rootname + ".json"

        if self.datapath.endswith('.json'):
            with open(self.datapath) as json_file:
                data = json.load(json_file)

        elif self.datapath.endswith('.cson'):
            with open(self.datapath) as cson_file:
                data = json.load(cson_file)

        else:
            print('Data file must have .json, .cson, .xls or .xlsx extension - creating empty data model')
            data = {}

        if printJSON and self.is_notebook:
            print('Collapsable rendition of model:')
            utils.RenderJSON(data)

        self.data = data

    # def make_graph(self):
    #     '''
    #     Initialize/build graph to visualize model (partly or completely)
    #     '''












class Node:
    '''
    ML: Find a better way to go through graph
    '''
    def __init__(self):
        data = self.data

        # to gather the types of values
        types = []
        answer = []
        attribs = []

        count = 0
        for key, value in self.tree_walk(data):
            if key == 'attributes':
                attribs.append(key)
                attribs.append(value)
                break

            if count % 2 == 0:
                types.append(key)
            else:
                answer.append(key)
            count += 1
        # attributes
        self.attribs
        self.types
        self.answer


    def tree_walk(self):
        '''
        clunky way to iterate through nested/structured dict
        '''
        dictionary = self.data
        for key, value in dictionary.items():
            if isinstance(value, dict):
                yield (key, value)
                yield from tree_walk(value)
            else:
                yield (key, value)

    def is_leaf(self):
        '''
        Returns boolean: whether it is a leaf (no descendent)
        '''
        assert self.ntype == "leaf" or "parent" or "root",
                                       "ntype must be set to `leaf` or `parent` or `root`"
        if self.ntype == "leaf":
            return True
        else:
            return False



class NodeType(Node): # NodeType is-a Node
    def __init__(self):
        '''
        fetch the place in the graph we are at
        '''







class EndUse(NodeType):
    def __init__(self):
        '''
        End use is a different type of node that is attached to demand (and potentially supply)
        Accounting of input/output is necessary
        '''


class TechType:
    def __init__(self, region, sector, end_use, attr_dict, *args, **kwargs):
        '''
        Note:
        check that attr_dict should be actually a dict input of build a
        dict from the AttrType

        Important: find out if tech container change in function of all arguments; if not, remove argument
        ---
        Constructor Args:
         `region`: str: name of region, eg 'alberta'
         `sector`: str: name of sector, eg 'residential'
         `end_use`: str: name of end_use, eg 'heating'
         `attr_dict`: dict: dictionary with key: attribute type, value: array of values for each year

        Note: may end up using a single dict for all args
        '''
        self.region = region
        self.sector = sector
        self.end_use = end_use
        self.attr_dict = attr_dict




class AttrType:
    '''
    Builds a dictionary of attributes for feeding into tech type
    '''
    def __init__(self, *args, **kwargs):





class DemandAssess:
    def __init__(self, params, techtype, *args, **kwargs):
        '''
        params: nested dict: contains all parameters necessary for computations
        techtype: str: name of technology
        '''

        ### 1

        self.baseyear = params["baseyear"] # reference year - 2000 at the moment
        self.runyear = params["runyear"]   # current year in simulation
        self.lifespan = params[techtype]["lifespan"] # lifespan of tech
                                                     # from Technology Data Master:  Cost Data:  Column L
        self.basestock = params[techtype]["basestock"] # basestock of tech,
                                                        # fetched from Calibration File

        ### 2
        self.purchase_year = params[techtype]["purchase_newstock"] # year new stock was purchased at time t. (usually equal to t)


        ### 3
        self.newstock = params[techtype]["newstock"] # new stock purchased at time t, by tech.
                                                     # from output file BXXNEWMA or CXXNEWMA, where
                                                     # XX stands for the province’s initials

        ### Retrofit and New Stock Competition
        ### 4

        self.size = params[techtype]["size"]

        # annual life-cycle-cost of tech k in size s at time t  (from CIMS output files BXXLEVEL OR CXXLEVEL)
        self.life_cycle_cost_kst = params[techtype]["life_cycle_cost"][self.size][self.runyear]

        # variance parameter representing cost homogeneity, from Economic Data: Economic Description: Column D.
        # ML note: Brad commented 'elasticity' - is it equivalent?
        self.elasticity = params[techtype]["v"]

        # number of technologies competing to provide the same service as k
        self.n_compete = params[techtype]["n_compete"]



        # methods (calculations)
        def base_stock_retirement(self):
            '''
            equation 1
            ML: check that result is a scalar (and no matrix multiplication needed)
            '''
            basestock = self.basestock*(self.runyear - self.baseyear)/self.lifespan
            # ensure it's a non-negative value
            basestock_pos = np.max([0, basestock])

            return basestock_pos

        def bet(self):
            '''
            equation 3
            Note: where does the 11.513 come from
            '''
            bet = ((self.runyear - self.purchase_year)* -11.513)/self.lifespan
            return bet


        def new_stock_retirement(self):
            '''
            equation 2
            newstock minus newstock weighted by inverse logit
            ML: check that result is a scalar (and no matrix multiplication needed)
            '''
            bet = self.bet()
            newstock = self.newstock - self.newstock*(1 + np.exp(-11.513 - bet))**(-1)
            return newstock

        # retrofit / new stock competition
        def marketshare(self):
            '''
            Equation 4
            '''
            lcc_v = self.life_cycle_cost_kst[0]**(-self.elasticity)
            lcc_v_j = [self.life_cycle_cost_kst[j]**(-self.elasticity) for j in range(self.n_compete) + 1]
            sum_lcc_v = np.sum(lcc_v_j) # all k + j
            ms_kst = lcc_v/sum_lcc_v
            return ms_kst

        def weighted_marketshare(self): # remove
            '''
            For nodes that compete technologies that come in two different sizes, market shares are
            determined separately for each size. After the market shares for each size have
            been allocated, they are averaged using the Size and Capacity Utilization (SCU) weights
            (equation 5)
            '''

            return weighted_marketshare

        # COMMON COMPUTATIONS:

        # Capital Costs cc_k

        # Operating Costs om_k

        # Energy Costs ec_kt

        # Service Costs sc_kt

        # Retrofit

        #### Mods on tech choice equations:

        # Declining Capital Cost Function (and more ...)

        #### Expectations - implement

        #### Misc

        # real cost vs transfer (or subsidy) may alter costs - [not now]




class Compute:
    '''
    Wrapper that contains calculations
    '''
    def __init__(self, nodes, *args, **kwargs):
        self.nodes = nodes
        '''
        Use methods from DemandAssess objects
        '''

        def residential_energy(self):
            '''
            Define tree for residential energy flow model
            '''




class FlowModel:
    def __init__(self, sector, flow_model_type, *args, **kwargs):
        '''
        Use to build connections/tree
        '''
        self.sector = sector
        self.flow_model_type = flow_model_type

        '''
        define methods with connections:
         - primary process
         - intermediate process
         - process competition
        '''








'''
Classes below may or may not be necessary
'''

class SupplySide(TechType, *args, **kwargs):
    def __init__(self):








### TESTS ####
#
#
# def tree_walk(dictionary):
#     '''
#     clunky way to iterate through nested/structured dict
#     '''
#
#     for key, value in dictionary.items():
#         if isinstance(value, dict):
#             yield (key, value)
#             yield from tree_walk(value)
#         else:
#             yield (key, value)
#
#
#
# # to gather the types of values
# types = []
# answer = []
# attribs = []
#
# count = 0
# for key, value in tree_walk(d):
#     if key == 'attributes':
#         attribs.append(key)
#         attribs.append(value)
#         break
#
#     if count % 2 == 0:
#         types.append(key)
#     else:
#         answer.append(key)
#     count += 1
#
#
# types
#
# attribs[1]
#
# for k, v in attribs[1].items():
#     print(k)
#     print(v)
#     print('\n')
