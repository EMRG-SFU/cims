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
        self.index2node_index_map = self._create_index_to_node_index_map()

    def find_roots(self):
        root_idx = self.model_df[(self.model_df['Parameter'] == 'Competition type') &
                                 (self.model_df['Value'] == 'Root')].index
        root_nodes = set([self.index2node_map[ri] for ri in root_idx])

        return root_nodes

    def _create_index_to_node_index_map(self):
        # Record the index for every node name
        node_indexes = []
        for i, name in zip(self.model_df.index, self.model_df[self.node_col]):
            if name:
                node_indexes.append(i)
            else:
                node_indexes.append(None)

        # Forward fill to get map
        index_to_node_index_map = pd.Series(node_indexes, index=self.model_df.index).ffill()

        return index_to_node_index_map

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

        def nodes_requesting_self(p, r):
            """
            Identifies any nodes which request services of themselves. Adds (index, Node Name) pairs
            to the warnings dictionary, under the key "nodes_requesting_self".

            If `verbose` is `True` will print nodes which were identified . If `raise_warnings` is
            `True` will raise warnings to indicate which nodes were identified.

            Parameters
            ----------
            p : pd.Series
                A series of branch names for any line in the model description where "Service
                provided" is the parameter. Index of this series is the line number of the row where
                it appears in the model description.

            r : pd.Series
                A series of branch names for any line in the model description where "Service
                requested" is the parameter. Index of this series is the line number of the row
                where is appears in the model description.

            Returns
            -------
            None
            """
            # Use Index to Node Index map to determine node index for all service provided lines
            p_df = pd.DataFrame(index=p.index)
            p_df['branch'] = p
            p_df['node_index'] = [self.index2node_index_map[i] for i, branch in p.iteritems()]

            # Use Index to Node Index map to determine node index for all service requested lines
            r_df = pd.DataFrame(index=r.index)
            r_df['branch'] = r
            r_df['node_index'] = [self.index2node_index_map[i] for i, branch in r.iteritems()]

            # Merge
            merge_df = r_df.reset_index().merge(p_df.reset_index(),
                                                on='node_index',
                                                suffixes=('_requested', '_supplied'))

            # Filter where request and provide are equal
            conflicts = merge_df[merge_df['branch_requested'] == merge_df['branch_supplied']]

            # Get Index, Branch pairs for logging
            self_requesting = [(i, b) for i, b in zip(conflicts['index_requested'],
                                                      conflicts['branch_requested'])]

            # Add violating nodes to warnings dictionary
            if len(self_requesting) > 0:
                self.warnings['nodes_requesting_self'] = self_requesting

            # Print Problems
            if verbose:
                more_info = "See ModelValidator.warnings['nodes_requesting_self'] for more info"
                print("{} nodes requested services of themselves. "
                      "{}".format(len(self_requesting),
                                  more_info if len(self_requesting) else ""))
            # Raise Warnings
            if raise_warnings:
                more_info = "See ModelValidator.warnings['nodes_requesting_self'] for more info"
                w = "{} nodes requested services of themselves " \
                    "{}".format(len(nodes_requesting_self),
                                more_info if len(nodes_requesting_self) else "")
                warnings.warn(w)

        def nodes_no_requested_service(r):
            nodes = self.model_df["Node"].dropna()
            techs = self.model_df[self.model_df['Parameter'] == "Technology"]['Value']
            nodes_with_tech = list(set(self.index2node_map[techs.index]))
            nodes_without_tech = [n for i, n in nodes.iteritems() if n not in nodes_with_tech]

            nodes_that_request = [self.index2node_map[i] for i, v in r.iteritems()]
            nodes_no_service = [(i, n) for i, n in nodes.iteritems() if n not in nodes_that_request]

            nodes_or_techs_no_service = [(i, n) for i, n in nodes_no_service if n in nodes_without_tech]
           
            for n in nodes_with_tech:
                node = self.index2node_map[self.index2node_map == n]
                techs_within_node = pd.DataFrame([(i, v) for i, v in techs.iteritems() if i in node.index],
                                                columns=['Index','Name'])
                techs_within_node = techs_within_node.append({'Index': node.index.max(), 'Name': None},
                                                            ignore_index=True)
                for i in range(techs_within_node.shape[0]):
                    if i == techs_within_node.shape[0]-1:
                        break
                    else:
                        start_index = techs_within_node['Index'].loc[i]
                        end_index = techs_within_node['Index'].loc[i+1]
                        tech_name = techs_within_node['Name'].loc[i]
                        if 'Service requested' not in list(self.model_df['Parameter'].loc[start_index:end_index]):
                            nodes_or_techs_no_service.append((nodes[nodes == n].index[0], n, tech_name))
        
            if len(nodes_or_techs_no_service) > 0:
                self.warnings['nodes_no_requested_service'] = nodes_or_techs_no_service

            # Print Problems
            if verbose:
                more_info = "See ModelValidator.warnings['nodes_no_requested_service'] for more info"
                print("{} nodes or technologies don't request other services. {}".format(len(nodes_or_techs_no_service),
                                                                                       more_info if len(nodes_or_techs_no_service) else ""))
            # Raise Warnings
            if raise_warnings:
                more_info = "See ModelValidator.warnings['nodes_no_requested_service'] for more info"
                w = "{} nodes or technologies don't request other services. {}".format(len(nodes_or_techs_no_service),
                                                                                     more_info if len(nodes_or_techs_no_service) else "")
                warnings.warn(w)

        def discrepencies_in_model_and_tree():
            excel_engine_map = {'.xlsb': 'pyxlsb',
                                '.xlsm': 'xlrd'}
            excel_engine = excel_engine_map[os.path.splitext(self.xl_file)[1]]
            mxl_tree = pd.read_excel(self.xl_file, sheet_name='Tree', header=3, engine=excel_engine)

            tree_df = mxl_tree.replace({np.nan: None})
            tree_sheet = pd.Series(tree_df['Branch']).dropna().reset_index(drop=True).str.lower()

            d = self.model_df
            p = d[d['Parameter'] == 'Service provided']['Branch']
            model_sheet = pd.Series(p).reset_index(drop=True).str.lower()

            nodes_with_discrepencies = []
            for i, n in tree_sheet.iteritems():
                discrepancy = False
                if n not in list(model_sheet):
                    discrepancy = True
                    model_order = None
                else:
                    if i != model_sheet[model_sheet == n].index[0]:
                        discrepancy = True
                        model_order = model_sheet[model_sheet == n].index[0]
                if discrepancy:
                    nodes_with_discrepencies.append((i, model_order, n))

            if len(nodes_with_discrepencies) > 0:
                self.warnings['discrepencies_in_model_and_tree'] = nodes_with_discrepencies
                
            # Print Problems
            if verbose:
                more_info = "See ModelValidator.warnings['discrepencies_in_model_and_tree'] for " \
                            "more info."
                print("{} nodes have been defined in a different order between the model and tree "
                      "sheets. {}".format(len(nodes_with_discrepencies),
                                          more_info if len(nodes_with_discrepencies) else ""))
            # Raise Warnings
            if raise_warnings:
                more_info = "See ModelValidator.warnings['discrepencies_in_model_and_tree'] for " \
                            "more info."
                w = "{} nodes have been defined in a different order between the model and tree" \
                    "sheets. {}".format(len(nodes_with_discrepencies),
                                        more_info if len(nodes_with_discrepencies) else "")
                warnings.warn(w)  

        def nodes_with_zero_output():
            output = self.model_df[self.model_df['Parameter'] == 'Output'].iloc[:,7:18]
            zero_output_nodes = []
            for i in range(output.shape[0]):
                if output.iloc[i, 0:12].eq(0).any():
                    ind = output.index[i]
                    zero_output_nodes.append((ind, self.index2node_map[ind]))

            if len(zero_output_nodes) > 0:
                self.warnings['nodes_with_zero_output'] = zero_output_nodes

            # Print Problems
            if verbose:
                more_info = "See ModelValidator.warnings['nodes_with_zero_output'] for more info"
                print("{} nodes have 0 in the output line. {}".format(len(zero_output_nodes),
                                                                                       more_info if len(zero_output_nodes) else ""))
            # Raise Warnings
            if raise_warnings:
                more_info = "See ModelValidator.warnings['nodes_with_zero_output'] for more info"
                w = "{} nodes have 0 in the output line. {}".format(len(zero_output_nodes),
                                                                                     more_info if len(zero_output_nodes) else "")
                warnings.warn(w)

        def fuel_nodes_no_lcc():
            d = self.model_df[self.model_df['Parameter'] == 'Node type']['Value'].str.lower() == 'supply'
            supply_nodes = [self.index2node_map[i] for i, v in d.iteritems() if v]

            no_prod_cost = []
            for n in supply_nodes:
                node = self.index2node_map[self.index2node_map == n]
                dat = self.model_df.loc[node.index, :]
                s = dat[dat['Parameter'] == 'Competition type']['Value'].str.lower()
                if 'sector no tech' in s.to_string():
                    if 'Life Cycle Cost' not in list(dat['Parameter']):
                        no_prod_cost.append((node.index[0], n))
                    else:
                        prod_cost = dat[dat['Parameter'] == 'Life Cycle Cost'].iloc[:, 7:18]
                        if prod_cost.iloc[0].isnull().any():
                            no_prod_cost.append((node.index[0], n))
        
            if len(no_prod_cost) > 0:
                self.warnings['fuels_without_lcc'] = no_prod_cost

            # Print Problems
            if verbose:
                more_info = "See ModelValidator.warnings['fuels_without_lcc'] for more info"
                print("{} fuel nodes don't have an Life Cycle Cost. {}".format(len(no_prod_cost),
                                                                   more_info if len(no_prod_cost) else ""))
            # Raise Warnings
            if raise_warnings:
                more_info = "See ModelValidator.warnings['fuels_without_lcc'] for more info"
                w = "{} fuel nodes don't have an Life Cycle Cost. {}".format(len(no_prod_cost),
                                                                 more_info if len(no_prod_cost) else "")
                warnings.warn(w)  

        def nodes_no_capital_cost():
            nodes = self.model_df[self.model_df['Parameter'] == 'Service provided']['Branch']
            nodes.index = nodes.index-1
            end = pd.Series(['None'],index=[self.model_df.index.max()])
            nodes = nodes.append(end)
            no_cap_cost = []

            for i in range(nodes.shape[0]-1):
                node_index = nodes.index[i]
                node_name = nodes.iloc[i]
                start_index = nodes.index[i]
                end_index = nodes.index[i+1]
                dat = self.model_df.loc[start_index:end_index]
                tech_nodes = dat[dat['Parameter'] == 'Competition type']['Value'].str.lower() == 'tech compete'
                if tech_nodes.iloc[0]:
                    if 'Technology' not in list(dat['Parameter']):
                        if 'Capital cost_overnight' not in list(dat['Parameter']):
                            no_cap_cost.append((node_index,node_name))
                    else: 
                        techs = dat[dat['Parameter'] == 'Technology']['Value']
                        end = pd.Series(['None'],index=[dat.index.max()])
                        techs = techs.append(end)
                        for i in range(techs.shape[0]-1):
                            tech_name = techs.iloc[i]
                            start_index = techs.index[i]
                            end_index = techs.index[i+1]
                            if 'Capital cost_overnight' not in list(dat['Parameter'].loc[start_index:end_index]):
                                no_cap_cost.append((node_index,node_name,tech_name))

            if len(no_cap_cost) > 0:
                self.warnings['nodes_without_capital_cost'] = no_cap_cost

            # Print Problems
            if verbose:
                more_info = "See ModelValidator.warnings['nodes_without_capital_cost'] for more info"
                print("{} tech compete nodes don't have capital cost. {}".format(len(no_cap_cost),
                                                                   more_info if len(no_cap_cost) else ""))
            # Raise Warnings
            if raise_warnings:
                more_info = "See ModelValidator.warnings['nodes_without_capital_cost'] for more info"
                w = "{} tech compete nodes don't have capital cost. {}".format(len(no_cap_cost),
                                                                 more_info if len(no_cap_cost) else "")
                warnings.warn(w)  
                    
        
        providers = self.model_df[self.model_df['Parameter'] == 'Service provided']['Branch']
        requested = self.model_df[self.model_df['Parameter'] == 'Service requested']['Branch']
        roots = self.find_roots()

        mismatched_node_names(providers)
        unspecified_nodes(providers, requested)
        unreferenced_nodes(providers, requested, roots)
        nodes_no_provided_service(providers)
        invalid_competition_type()
        nodes_requesting_self(providers, requested)
        nodes_no_requested_service(requested)
        discrepencies_in_model_and_tree()
        nodes_with_zero_output()
        fuel_nodes_no_lcc()
        nodes_no_capital_cost()
