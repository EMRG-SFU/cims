import pandas as pd
import numpy as np
from .reader import get_node_cols
import warnings
import os


class ModelValidator:
    def __init__(self, xl_file, node_col='Node'):
        self.xl_file = xl_file
        self.node_col = node_col

        self.warnings = {}

        # Turn Excel file into Dataframe
        excel_engine_map = {'.xlsb': 'pyxlsb',
                            '.xlsm': 'xlrd'}
        excel_engine = excel_engine_map[os.path.splitext(self.xl_file)[1]]
        mxl = pd.read_excel(xl_file, sheet_name=None, header=1, engine=excel_engine)
        model_df = mxl['Model'].replace({np.nan: None})      # Read the model sheet into a dataframe
        model_df.index += 3                                     # Adjust index to correspond to Excel line numbers
                                                                # +1: 0 vs 1 origin, +1: header skip, +1: column headers
        model_df.columns = [str(c) for c in
                            model_df.columns]                   # Convert all column names to strings (years were ints)
        n_cols, y_cols = get_node_cols(model_df, 'Node')        # Find columns, separated year cols from non-year cols
        all_cols = np.concatenate((n_cols, y_cols))
        mdf = model_df.loc[1:, all_cols]  # Create df, drop irrelevant columns & skip first, empty row

        self.model_df = mdf
        self.index2node_map = self.model_df[self.node_col].ffill() 

    def find_roots(self):
        root_idx = self.model_df[(self.model_df['Parameter'] == 'Competition type') & 
                                 (self.model_df['Value'] == 'Root')].index
        root_nodes = set([self.index2node_map[ri] for ri in root_idx])

        return root_nodes 

    def validate(self, verbose=True, raise_warnings=False):
        def invalid_competition_type():
            valid_comp_type = ['Root',
                               'Region',
                               'Sector',
                               'Sector No Tech',
                               'Tech Compete',
                               'Fixed Ratio']
            
            invalid_nodes = []
            comp_types = self.model_df[self.model_df['Parameter'] == 'Competition type']
            for index, value in zip(comp_types.index, comp_types['Value']):
                if value not in valid_comp_type:
                    invalid_nodes.append((index, self.index2node_map[index]))
                
            if len(invalid_nodes) > 0:
                self.warnings['invalid_competition_type'] = invalid_nodes

            # Print Problems
            if verbose:
                more_info = "See ModelValidator.warnings['invalid_competition_type'] for more info"
                print("{} nodes had invalid competition types. {}".format(len(invalid_nodes),
                                                                                  more_info if len(invalid_nodes) else ""))
            # Raise Warnings
            if raise_warnings:
                more_info = "See ModelValidator.warnings['invalid_competition_type'] for more info"
                w = "{} nodes had invalid competition types. {}".format(len(invalid_nodes),
                                                                                  more_info if len(invalid_nodes) else "")
                warnings.warn(w)
        
        def unspecified_nodes(p, r):
            referenced_unspecified = [(i, v) for i, v in r.iteritems() if v not in p.values]

            # referenced_unspecified = set(r).difference(set(p))
            if len(referenced_unspecified) > 0:
                self.warnings['unspecified_nodes'] = referenced_unspecified

            # Print Problems
            if verbose:
                more_info = "See ModelValidator.warnings['unspecified_nodes'] for more info"
                print("{} references to unspecified nodes. {}".format(len(referenced_unspecified),
                                                                                           more_info if len(referenced_unspecified) else ""))
            # Raise Warnings
            if raise_warnings:
                more_info = "See ModelValidator.warnings['mismatched_node_names'] for more info"
                w = "{} references to unspecified nodes. {}".format(len(referenced_unspecified),
                                                                                         more_info if len(referenced_unspecified) else "")
                warnings.warn(w)

        def unreferenced_nodes(p, r, roots):
            specified_unreferenced = [(i, v) for i, v in p.iteritems() if (v not in r.values) and (v not in roots)]

            if len(specified_unreferenced) > 0:
                self.warnings['unreferenced_nodes'] = specified_unreferenced

            # Print Problems
            if verbose:
                more_info = "See ModelValidator.warnings['unreferenced_nodes'] for more info"
                print("{} non-root nodes are never referenced. {}".format(len(specified_unreferenced),
                                                                          more_info if len(specified_unreferenced) else ""))
            # Raise Warnings
            if raise_warnings:
                more_info = "See ModelValidator.warnings['unreferenced_nodes'] for more info"
                w = "{} non-root nodes are never referenced. {}".format(len(specified_unreferenced),
                                                                        more_info if len(specified_unreferenced) else "")
                warnings.warn(w)

        def mismatched_node_names(p):
            node_name_indexes = self.model_df['Node'].dropna()
            mismatched = []
            # Given an index where a service provided line lives find the name of the Node housing it.
            for i, x in p.iteritems():
                cand = node_name_indexes.loc[:i]
                node_name_index = cand.index.max()
                node_name = list(cand)[-1]
                if x is None: 
                    mismatched.append((node_name_index, node_name, None))
                else:
                    branch_node_name = x.split('.')[-1]
                    if node_name != branch_node_name:
                        mismatched.append((node_name_index, node_name, branch_node_name))

            if len(mismatched) > 0:
                self.warnings['mismatched_node_names'] = mismatched

            # Print Problems
            if verbose:
                more_info = "See ModelValidator.warnings['mismatched_node_names'] for more info"
                print("{} node name/branch mismatches. {}".format(len(mismatched),
                                                                  more_info if len(mismatched) else ""))
            # Raise Warnings
            if raise_warnings:
                more_info = "See ModelValidator.warnings['mismatched_node_names'] for more info"
                w = "{} node name/branch mismatches. {}".format(len(mismatched), more_info if len(mismatched) else "")
                warnings.warn(w)

        def nodes_no_provided_service(p):
            nodes = self.model_df[self.node_col].dropna()
            nodes_that_provide = [self.index2node_map[i] for i, v in p.iteritems()]
            nodes_no_service = [(i,n) for i, n in nodes.iteritems() if n not in nodes_that_provide] 

            if len(nodes_no_service) > 0:
                self.warnings['nodes_no_provided_service'] = nodes_no_service

            # Print Problems
            if verbose:
                more_info = "See ModelValidator.warnings['nodes_no_service'] for more info"
                print("{} nodes were specified but don't provide a service. {}".format(len(nodes_no_service),
                                                                                       more_info if len(nodes_no_service) else ""))
            # Raise Warnings
            if raise_warnings:
                more_info = "See ModelValidator.warnings['nodes_no_service'] for more info"
                w = "{} nodes were specified but don't provide a service. {}".format(len(nodes_no_service),
                                                                                     more_info if len(nodes_no_service) else "")
                warnings.warn(w)

        def nodes_no_requested_service(r):
            nodes = self.model_df["Node"].dropna()
            techs = self.model_df[self.model_df['Parameter'] == "Technology"]['Value']
            nodes_with_tech = list(set(self.index2node_map[techs.index]))
            nodes_without_tech = [n for i, n in nodes.iteritems() if n not in nodes_with_tech]

            nodes_that_request = [self.index2node_map[i] for i, v in r.iteritems()]
            nodes_no_service = [(i, n) for i, n in nodes.iteritems() if n not in nodes_that_request]

            nodes_or_techs_no_service = [(i, n) for i, n in nodes_no_service if
                                         n in nodes_without_tech]

            for n in nodes_with_tech:
                node = self.index2node_map[self.index2node_map == n]
                techs_within_node = pd.DataFrame(
                    [(i, v) for i, v in techs.iteritems() if i in node.index],
                    columns=['Index', 'Name'])
                techs_within_node = techs_within_node.append(
                    {'Index': node.index.max(), 'Name': None},
                    ignore_index=True)
                for i in range(techs_within_node.shape[0]):
                    if i == techs_within_node.shape[0] - 1:
                        break
                    else:
                        start_index = techs_within_node['Index'].loc[i]
                        end_index = techs_within_node['Index'].loc[i + 1]
                        tech_name = techs_within_node['Name'].loc[i]
                        if 'Service requested' not in list(
                                self.model_df['Parameter'].loc[start_index:end_index]):
                            nodes_or_techs_no_service.append(
                                (nodes[nodes == n].index[0], n, tech_name))

            if len(nodes_or_techs_no_service) > 0:
                self.warnings['nodes_no_requested_service'] = nodes_or_techs_no_service

            # Print Problems
            if verbose:
                more_info = "See ModelValidator.warnings['nodes_no_requested_service'] for more info"
                print("{} nodes or technologies don't request other services. {}".format(
                    len(nodes_or_techs_no_service),
                    more_info if len(nodes_or_techs_no_service) else ""))
            # Raise Warnings
            if raise_warnings:
                more_info = "See ModelValidator.warnings['nodes_no_requested_service'] for more info"
                w = "{} nodes or technologies don't request other services. {}".format(
                    len(nodes_or_techs_no_service),
                    more_info if len(nodes_or_techs_no_service) else "")
                warnings.warn(w)

        providers = self.model_df[self.model_df['Parameter'] == 'Service provided']['Branch']
        requested = self.model_df[self.model_df['Parameter'] == 'Service requested']['Branch']
        roots = self.find_roots()

        mismatched_node_names(providers)
        unspecified_nodes(providers, requested)
        unreferenced_nodes(providers, requested, roots)
        nodes_no_provided_service(providers)
        invalid_competition_type()
        nodes_no_requested_service(requested)