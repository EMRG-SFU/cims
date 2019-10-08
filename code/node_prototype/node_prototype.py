%matplotlib inline

import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp

import os
import json
import cson

from collections import OrderedDict

# build graph visualization
import networkx as nx

# local modules
import helper.utils

# # ML: will remove later: just an artifact of my computer setting
# os.chdir(os.path.join(os.getcwd(), "code"))
#

'''
Make fake dataset
'''
years = np.arange(2000, 2020, 5, dtype = np.int64)

macro_indicators = {'population': np.array([2801421, 3223339, 3645257, 4067175], dtype = np.float64),
                    'gdp': np.array([211, 265, 290, 330], dtype = np.float64)}



electric = {"elec_tech1":
                {"lifespan": 10,
                 "basestock": [500],  # ML! check if we have all ts or just t=0
                 "purchase_year":[1995],  # ML! check if we have all ts or just t=0 (also: meaning?)
                 "newstock": [40], # new stock purchased at time t, by tech. (? all t's?)
                 "lcc": [500, 450, 500, 350],  # ML! what does this look like (2 kinds of LCC?)
                 "capital_costs": [200],  #ML! is this at each time step?
                 "operation_costs": [180],  #ML! is this at each time step?
                 "fuel_price": [500],
                 "fuel_reqs": [1000],
                 "service_price": [100],  # ML! exact meaning of `service` -- first input given or not?
                 "service_reqs": [200],
                 "efficiency_workfactor":[0.4]   # proportion?
                 },

            "elec_tech2":
                {"lifespan": 20,
                 "basestock": [430, 450, 480, 500],
                 "purchase_year":[2000],
                 "newstock": [25],
                 "lcc": [400, 450, 450, 430],
                 "capital_costs": [150],
                 "operation_costs": [180],
                 "fuel_price": [450],
                 "fuel_reqs": [1200],
                 "service_price": [80],
                 "service_reqs": [120],
                 "efficiency_workfactor":[0.6]
                 }
            }




class DemandAssess:
    def __init__(self, techtype,
                       macro_indicators,
                       years,
                       baseyear=2000,
                       elasticity=0.1):
        '''
        :params: nested dict: contains all parameters necessary for computations
        :techtype: str: name of technology
        '''
        # get exogenous prices and activity (ML! (list all factors/variables))
        self.population = macro_indicators['population']
        self.gdp = macro_indicators['gdp']

        # get tech info
        self.tech_type = tech_type

        self.baseyear = baseyear
        self.years = years
        self.elasticity = elasticity
        # number of technologies competing to provide the same service as k
        # ML! is this all techs or all other techs
        self.n_compete = len(techtype)

        # lists to unpack parameters
        self.lifespan = []
        self.basestock = []
        self.purchase_year = []
        self.newstock = []
        self.elasticity = []
        self.life_cycle_cost = []
        self.tech = []

        # containers for computations
        self.market_share = {}

        for key, val in self.techtype.items():
            self.lifespan.append(val["lifespan"]) # lifespan of tech
            self.basestock.append(val["basestock"]) # basestock of tech
            self.purchase_year.append(val["purchase_year"]) # year new stock was purchased at time t. (usually equal to t)
            self.newstock.append(val["newstock"]) # new stock purchased at time t, by tech.
            self.life_cycle_cost.append(val["lcc"]) # ML! ->>> are there 2 different LCC in docs? (p.16-17)
            self.tech.append(key)

            for num, year in enumerate(self.years):
                self.basestock.append(self.basestock_retirement(year, self.basestock[num],
                                                                      val["lifespan"]))

                self.newstock.append(self.new_stock_retirement(year, val["purchase_year"][num],
                                                                     val["lifespan"],
                                                                     self.newstock[num]))

        for tech_id in range(self.n_compete):
            self.market_share[self.tech[tech_id]] = self.marketshare(self.life_cycle_cost,
                                                                     self.elasticity,
                                                                     tech_idx)

       self.energy_cost = self.energy_costs()
       self.service_price = self.service_prices()
       self.service_cost = self.service_costs()

        '''
        NODE COMPUTATIONS
        '''
        def basestock_retirement(self, cur_year, basestock, lifespan, baseyear=2000):
            '''
            equation 1
            '''
            new_basestock = basestock*(cur_year - baseyear)/lifespan
            # ensure it's a non-negative value
            basestock_ret = np.max([0, new_basestock])
            return basestock_ret


        def bet(self, cur_year, purchase_year, lifespan):
            '''
            equation 3
            Note: where does the 11.513 come from
            '''
            bet = (-11.513 * (cur_year - purchase_year))/lifespan
            return bet


        def new_stock_retirement(self, cur_year,
                                       purchase_year,
                                       lifespan,
                                       newstock):
            '''
            equation 2
            newstock minus newstock weighted by inverse logit
            ML: check that result is a scalar (and no matrix multiplication needed)
            '''
            bet = self.bet( cur_year, purchase_year, lifespan)
            newstock = newstock - newstock*(1 + np.exp(-11.513 - bet))**(-1)
            return newstock

        # retrofit / new stock competition
        def marketshare(self, all_lcc, elasticity, tech_idx):
            '''
            Equation 4
            '''
            lcc_v = all_lcc[tech_idx]**(-self.elasticity)
            lcc_v_j = [all_lcc[n]**(-self.elasticity) for n in range(self.n_compete)]
            sum_lcc_v = np.sum(lcc_v_j) # all k + j
            ms_kst = lcc_v/sum_lcc_v
            return ms_kst

        # LCC equation here???
        # weighted cap costs


        def energy_costs(self):   # ML! sum over sub types of energy?
            total_cost = np.sum(self.fuel_price) # (I think) sum all sub types at time t
            energy_cost = total_cost*self.fuel_reqs
            return energy_cost

        def service_prices(self, last_timestep):
            past_marketshare = np.sum(self.marketshare[last_timestep])
            service_price = (past_marketshare * self.lcc)* self.workfactor # summed over lcc too?
            return service_price



        def service_costs(self):   # ML! sum over sub types of services?
            total_cost = np.sum(self.service_price) # (I think) sum all sub types at time t
            service_cost = total_cost*self.service_reqs
            return service_cost












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
