# Goal of this module is to provide debugging tools for the model description spreadsheet.

# What kinds of things do we need to debug?
#   Nodes are linked to but don't appear as nodes in the model description --> DONE
#   Nodes that are specified in the model description but are not referenced by any other node --> DONE
#   Nodes whose name and branch (last component of branch) do not match --> DONE

# Where does this debugging happen?
#   Can it operate on the excel file itself? Does it need to use the node & tech DataFrame dictionaries?

import pandas as pd
import numpy as np
from Reader import get_node_cols
import warnings


class ModelValidator:
    def __init__(self, xl_file):
        self.xl_file = xl_file
        self.warnings = {}

        # Turn Excel file into Dataframe
        mxl = pd.read_excel(xl_file, sheet_name=None, header=1)    # Read model_description from excel
        model_df = mxl['Model'].replace({pd.np.nan: None})      # Read the model sheet into a dataframe
        model_df.index += 3                                     # Adjust index to correspond to Excel line numbers
                                                                # +1: 0 vs 1 origin, +1: header skip, +1: column headers
        model_df.columns = [str(c) for c in
                            model_df.columns]                   # Convert all column names to strings (years were ints)
        n_cols, y_cols = get_node_cols(model_df,
                                       'Node',
                                       ['Demand?'])             # Find columns, separated year cols from non-year cols
        all_cols = np.concatenate((n_cols, y_cols))
        mdf = model_df.loc[1:, all_cols]  # Create df, drop irrelevant columns & skip first, empty row

        self.model_df = mdf

    def validate(self, verbose=False):
        def unspecified_nodes(p, r):
            referenced_unspecified = set(r).difference(set(p))
            if len(referenced_unspecified) > 0:
                w = """{} nodes have been referenced but not specified in the model description: 
                       {}""".format(len(referenced_unspecified), referenced_unspecified)
                self.warnings['unspecified_nodes'] = referenced_unspecified
                if verbose:
                    warnings.warn(w)

        def unreferenced_nodes(p, r):
            specified_unreferenced = set(p).difference(set(r))
            if len(specified_unreferenced) > 0:
                w = "{} nodes have been specified in the model description but are not referenced by another node:{}"\
                    .format(len(specified_unreferenced), specified_unreferenced)
                self.warnings['unreferenced_nodes'] = specified_unreferenced
                if verbose:
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
                w = """{} nodes had a name mismatch with their branch: {}""".format(len(mismatched), mismatched)
                self.warnings['mismatched_node_names'] = mismatched
                if verbose:
                    warnings.warn(w)

        providers = self.model_df[self.model_df['Parameter'] == 'Service provided']['Branch']
        requested = self.model_df[self.model_df['Parameter'] == 'Service requested']['Branch']

        mismatched_node_names(providers)
        unspecified_nodes(providers, requested)
        unreferenced_nodes(providers, requested)


