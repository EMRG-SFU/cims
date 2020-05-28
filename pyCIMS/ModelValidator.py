import pandas as pd
import numpy as np
from .Reader import get_node_cols
import warnings


class ModelValidator:
    def __init__(self, xl_file, node_col='Node'):
        self.xl_file = xl_file
        self.node_col = node_col

        self.warnings = {}

        # Turn Excel file into Dataframe
        mxl = pd.read_excel(xl_file, sheet_name=None, header=1)    # Read model_description from excel
        model_df = mxl['Model'].replace({pd.np.nan: None})      # Read the model sheet into a dataframe
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
        #TODO: Update this once the model description has been updated
        #root_idx = self.model_df[(self.model_df['Parameter'] == 'Node type') &
                                 #(self.model_df['Value'] == 'Root')].index
        
        root_idx = self.model_df[(self.model_df['Parameter'] == 'Competition type') &
                                 (self.model_df['Value'] == 'Root')].index
        root_nodes = set([self.index2node_map[ri] for ri in root_idx])

        return root_nodes 

    def validate(self, verbose=True, raise_warnings=False):
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
            nodes_no_service = [(i, n) for i, n in nodes.iteritems() if n not in nodes_that_provide]

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

        providers = self.model_df[self.model_df['Parameter'] == 'Service provided']['Branch']
        requested = self.model_df[self.model_df['Parameter'] == 'Service requested']['Branch']
        roots = self.find_roots()

        mismatched_node_names(providers)
        unspecified_nodes(providers, requested)
        unreferenced_nodes(providers, requested, roots)
        nodes_no_provided_service(providers)