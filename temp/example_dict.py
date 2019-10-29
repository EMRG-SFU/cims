import numpy as np

# ******************************************
# Fixed Ratio Example
# ******************************************
r = {'residential': {'service_supply': {'branch': 'Canada.Alberta.Residential.Buildings.Shell.Space Heating',
                                        'unit': 'household',
                                        'value': 'residential'},
                     'competition_type': 'fixed_ratio',
                     'prices': {'diesel': {'unit': '2015$/GJ',
                                           'values': [19.0, 25.0, 29.0, 32.0, 32.6, 33.3, 34.0, 34.6, 35.3, 36.0, 36.8]},
                                'light_fuel_oil': {'unit': '2015$/GJ',
                                                   'values': [20.0, 26.0, 30.0, 33.0, 33.7, 34.3, 35.0, 35.7, 36.4,
                                                              37.2, 37.9]},
                                'natural_gas': {'unit': '2015$/GJ',
                                                'values': [11.52, 11.52, 11.52, 11.52, 11.52, 11.52, 11.52, 11.52,
                                                           11.52, 11.52, 11.52]},
                                'electricity': {'unit': '2015$/GJ',
                                                'values': 22},
                                'wood': {'unit': '2015$/GJ',
                                         'values': [12.0, 12.0, 12.0, 12.0, 12.2, 12.5, 12.7, 13.0, 13.2, 13.5, 13.8]},
                                'co2': {'unit': '2015$/tCO2',
                                        'values': [0.0, 0.0, 0.0, 0.0, 20.0, 50.0, 70.0, 70.0, 70.0, 70.0, 70.0]},
                                'ch4': {'unit': '2015$/tCO2',
                                        'values': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
                                'n2o': {'unit': '2015$/tCO2',
                                        'values': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
                                },
                     'service_demand': {'buildings': {'branch': 'Canada.Alberta.Residential.Buildings',
                                                      'unit': 'building/household',
                                                      'values': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]}
                                        }
                     }
     }

# ******************************************
# Fixed Market Shares Example
# ******************************************
m = {'buildings': {'service_supply': {},
                   'competition_type': 'fixed_market_shares',
                   'market_share': {'single_family_detached': {'unit': '%',
                                                               'values': [0.719, 0.720, 0.719, 0.719, 0.718, 0.718,
                                                                          0.717, 0.717, 0.716, 0.716, 0.715]},
                                    'single_family_attached': {'unit': '%',
                                                               'values': [0.100, 0.101, 0.102, 0.103, 0.104, 0.105,
                                                                          0.106, 0.107, 0.108, 0.109, 0.110]},
                                    'apartment': {'unit': '%',
                                                  'values': [0.140, 0.140, 0.139, 0.139, 0.138, 0.138, 0.137, 0.137,
                                                             0.136, 0.136, 0.135]},
                                    'mobile': {'unit': '%',
                                               'values': [0.041, 0.040, 0.040, 0.040, 0.040, 0.040, 0.040, 0.040, 0.040,
                                                          0.040, 0.040]}},
                   'service_demand': {'Canada.Alberta.Residential.Buildings.Shell': {'single_family_detached': {'unit': '%',
                                                                                                                'values': [138.52, 141.02, 143.95, 146.24, 146.68, 146.68, 146.68, 146.68, 146.68, 146.68, 146.68]},
                                                                                     'single_family_attached': {'unit': '%',
                                                                                                                'values': [107.37, 111.23, 115.91, 121.53, 122.61, 122.61, 122.61, 122.61, 122.61, 122.61, 122.61]},
                                                                                     'apartment': {'unit': '%',
                                                                                                   'values': [84.76, 88.69, 92.97, 97.18, 97.93, 97.93, 97.93, 97.93, 97.93, 97.93, 97.93]},
                                                                                     'mobile': {'unit': '%',
                                                                                                'values': [91.72, 92.26, 93.40, 94.69, 95.04, 95.04, 95.04, 95.04, 95.04, 95.04, 95.04]}},
                                      'Canada.Alberta.Residential.Buildings.Dishwashing': {'single_family_detached': {'unit': '%',
                                                                                                                      'values': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]},
                                                                                           'single_family_attached': {'unit': '%',
                                                                                                                      'values': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]},
                                                                                           'apartment': {'unit': '%',
                                                                                                         'values': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]},
                                                                                           'mobile': {'unit': '%',
                                                                                                      'values': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]}},
                                      'Canada.Alberta.Residential.Buildings.Clothes washing': {}
                                      }
                   }
     }

# ******************************************
# Winner Take All Example
# ******************************************
w = {}

# ******************************************
# Tech Compete Example
# ******************************************
t = {"space_heating": {'service_supply': {'branch': 'Canada.Alberta.Residential.Buildings.Shell.Space Heating',
                                          'value': 'GJ'},
                       'competition_type': 'tech_compete',
                       'heterogeneity': 'v',
                       'technology': {'electric_baseboard': {'Available': {'unit': 'Date',
                                                                           'values': 2000},
                                                             'Unavailable': {'unit': 'Date',
                                                                             'values': 2100},
                                                             'Lifetime': {'unit': 'Years',
                                                                          'values': 25},
                                                             'Financial discount rate': {'unit': '%',
                                                                                         'values': 0.25},
                                                             'Capital Cost': {'unit': '2015$/GJ',
                                                                              'values': [2655, 2655, 2655, 2655, 2655,
                                                                                         2655, 2655, 2655, 2655, 2655,
                                                                                         2655]},
                                                             'Operating Cost': {'unit': '2015$/GJ',
                                                                                'values': []},
                                                             'Intangible Cost': {'unit': '2015$/GJ',
                                                                                 'values': []},
                                                             'Service Cost': {'unit': '2015$/GJ',
                                                                              'branch': 'Canada.Alberta.Electricity',
                                                                              'values': 22},
                                                             'Market share total_Max': {'unit': '%',
                                                                                        'values': []},
                                                             'Market share total_Min': {'unit': '%',
                                                                                        'values': []},
                                                             'Market share new_Max': {'unit': '%',
                                                                                      'values': []},
                                                             'Market share new_Min': {'unit': '%',
                                                                                      'values': []},
                                                             'Service demand': {'unit': 'GJ used / GJ provided',
                                                                                'branch': 'Canada.Alberta.Electricity',
                                                                                'values': [1, 1, 1, 1, 1, 1, 1, 1,
                                                                                           1, 1, 1]},
                                                             'Market share': {'unit': '%',
                                                                              'values': 0.006}},
                                      'furnace': {'Available': {'unit': 'Date',
                                                                'values': 2000},
                                                  'Unavailable': {'unit': 'Date',
                                                                  'values': 2100},
                                                  'Service Cost': {'unit': '2015$/GJ',
                                                                   'values': 503.29},
                                                  'Market share total_Max': {'unit': '%',
                                                                             'values': []},
                                                  'Market share total_Min': {'unit': '%',
                                                                             'values': []},
                                                  'Market share new_Max': {'unit': '%',
                                                                           'values': []},
                                                  'Market share new_Min': {'unit': '%',
                                                                           'values': []},
                                                  'Service demand': {'unit': 'GJ used / GJ provided',
                                                                     'branch': 'Canada.Alberta.Residential.Buildings.Shell.Space heating.Furnace',
                                                                     'values': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]},
                                                  'Market share': {'unit': '%',
                                                                   'values': 0.994}}
                                      }
                       }
     }


