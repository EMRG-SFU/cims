import pandas as pd
import numpy as np
from .reader import get_node_cols
import warnings
import os
from .utils import is_year


class ModelValidator:
    def __init__(self, infile, sheet_map, node_col):
        self.infile = infile
        excel_engine_map = {'.xlsb': 'pyxlsb',
                            '.xlsm': 'xlrd'}
        self.excel_engine = excel_engine_map[os.path.splitext(self.infile)[1]]

        self.sheet_map = sheet_map
        self.node_col = node_col

        self.model_df = self._get_model_df()

        self.warnings = {}

        self.index2node_map = self.model_df[self.node_col].ffill()
        self.index2node_index_map = self._create_index_to_node_index_map()
        self.index2branch_map = self._create_index_to_branch_map()

    def _get_model_df(self):
        # Read in list of sheets from 'Lists' sheet in model description
        sheet_df = pd.read_excel(self.infile,
                                 sheet_name=self.sheet_map['model'],
                                 engine=self.excel_engine)

        # Remove nans from list
        sheet_list = [sheet for sheet in sheet_df['Model sheets'] if not pd.isna(sheet)]

        appended_data = []
        for sheet in sheet_list:
            sheet_df = pd.read_excel(self.infile,
                                     sheet_name=sheet,
                                     header=1,
                                     engine=self.excel_engine).replace({np.nan: None})
            appended_data.append(sheet_df)

        model_df = pd.concat(appended_data,
                             ignore_index=True)  # Add province sheets together and re-index
        model_df.index += 3  # Adjust index to correspond to Excel line numbers
        # (+1: 0 vs 1 origin, +1: header skip, +1: column headers)
        model_df.columns = [str(c) for c in
                            model_df.columns]  # Convert all column names to strings (years were ints)
        n_cols, y_cols = get_node_cols(model_df,
                                       self.node_col)  # Find columns, separated year cols from non-year cols
        all_cols = np.concatenate((n_cols, y_cols))
        mdf = model_df.loc[1:,
              all_cols]  # Create df, drop irrelevant columns & skip first, empty row
        mdf['Parameter'] = mdf['Parameter'].str.lower()

        return mdf

    def find_roots(self):
        root_idx = self.model_df[(self.model_df['Parameter'] == 'competition type') &
                                 (self.model_df['Context'] == 'Root')].index
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

    def _create_index_to_branch_map(self):
        all_services_provided = self.model_df[self.model_df['Parameter'] == 'service provided'][
            'Branch']
        node_index_2_branch_map = {int(self.index2node_index_map[i]): b for i, b in
                                   all_services_provided.items()}
        return {i: node_index_2_branch_map[ni] for i, ni in self.index2node_index_map.items()}

    def validate(self, verbose=True, raise_warnings=False):
        def invalid_competition_type():
            """
            Identify any nodes which use an invalid competition type.
    
            Parameters
            ----------
            None

            Returns
            -------
            None
            """
            valid_comp_type = ['Root',
                               'Region',
                               'Sector',
                               'Tech Compete',
                               'Node Tech Compete',
                               'Fixed Ratio',
                               'Market',
                               'Fuel - Fixed Price',
                               'Fuel - Cost Curve Annual',
                               'Fuel - Cost Curve Cumulative',
                               ]

            invalid_nodes = []
            comp_types = self.model_df[self.model_df['Parameter'] == 'competition type']
            for index, value in zip(comp_types.index, comp_types['Context']):
                if value not in valid_comp_type:
                    invalid_nodes.append((index, self.index2node_map[index]))

            if len(invalid_nodes) > 0:
                self.warnings['invalid_competition_type'] = invalid_nodes

            # Print Problems
            if verbose:
                more_info = "See ModelValidator.warnings['invalid_competition_type'] for more info"
                print("{} nodes had invalid competition types. {}".format(len(invalid_nodes),
                                                                          more_info if len(
                                                                              invalid_nodes) else ""))
            # Raise Warnings
            if raise_warnings:
                more_info = "See ModelValidator.warnings['invalid_competition_type'] for more info"
                w = "{} nodes had invalid competition types. {}".format(len(invalid_nodes),
                                                                        more_info if len(
                                                                            invalid_nodes) else "")
                warnings.warn(w)

        def unspecified_nodes(p, r):
            """
            Identify any nodes which are referenced in another node's 'service requested' row but
            the node has not been specified within the mdoel description. 

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
            referenced_unspecified = [(i, v) for i, v in r.iteritems() if v not in p.values]

            # referenced_unspecified = set(r).difference(set(p))
            if len(referenced_unspecified) > 0:
                self.warnings['unspecified_nodes'] = referenced_unspecified

            # Print Problems
            if verbose:
                more_info = "See ModelValidator.warnings['unspecified_nodes'] for more info"
                print("{} references to unspecified nodes. {}".format(len(referenced_unspecified),
                                                                      more_info if len(
                                                                          referenced_unspecified) else ""))
            # Raise Warnings
            if raise_warnings:
                more_info = "See ModelValidator.warnings['mismatched_node_names'] for more info"
                w = "{} references to unspecified nodes. {}".format(len(referenced_unspecified),
                                                                    more_info if len(
                                                                        referenced_unspecified) else "")
                warnings.warn(w)

        def unreferenced_nodes(p, r, roots):
            """
            Identify a non-root node which is specified in the model description but its services 
            are not requested by another node.

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
                
            roots: set
                   The root node in the model description.
                
            Returns
            -------
            None
            """
            specified_unreferenced = [(i, v) for i, v in p.iteritems() if
                                      (v not in r.values) and (v not in roots)]

            if len(specified_unreferenced) > 0:
                self.warnings['unreferenced_nodes'] = specified_unreferenced

            # Print Problems
            if verbose:
                more_info = "See ModelValidator.warnings['unreferenced_nodes'] for more info"
                print(
                    "{} non-root nodes are never referenced. {}".format(len(specified_unreferenced),
                                                                        more_info if len(
                                                                            specified_unreferenced) else ""))
            # Raise Warnings
            if raise_warnings:
                more_info = "See ModelValidator.warnings['unreferenced_nodes'] for more info"
                w = "{} non-root nodes are never referenced. {}".format(len(specified_unreferenced),
                                                                        more_info if len(
                                                                            specified_unreferenced) else "")
                warnings.warn(w)

        def mismatched_node_names(p):
            """
            Identify any nodes whose name (from the Node col) does not match the last component 
            of their branch. 
            
            Parameters
            ----------
            p : pd.Series
                A series of branch names for any line in the model description where "Service
                provided" is the parameter. Index of this series is the line number of the row where
                it appears in the model description.

            Returns
            -------
            None
            """
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
                                                                  more_info if len(
                                                                      mismatched) else ""))
            # Raise Warnings
            if raise_warnings:
                more_info = "See ModelValidator.warnings['mismatched_node_names'] for more info"
                w = "{} node name/branch mismatches. {}".format(len(mismatched), more_info if len(
                    mismatched) else "")
                warnings.warn(w)

        def nodes_no_provided_service(p):
            """
            Identify any nodes which are specified but does not provide a service. 
            Adds (index, Node Name) pairs to the warnings dictionary, 
            under the key "nodes_no_provided_service".

            Parameters
            ----------
            p : pd.Series
                A series of branch names for any line in the model description where "Service
                provided" is the parameter. Index of this series is the line number of the row where
                it appears in the model description.

            Returns
            -------
            None
            """
            nodes = self.model_df[self.node_col].dropna()
            nodes_that_provide = [self.index2node_map[i] for i, v in p.iteritems()]
            nodes_no_service = [(i, n) for i, n in nodes.iteritems() if n not in nodes_that_provide]

            if len(nodes_no_service) > 0:
                self.warnings['nodes_no_provided_service'] = nodes_no_service

            # Print Problems
            if verbose:
                more_info = "See ModelValidator.warnings['nodes_no_service'] for more info"
                print("{} nodes were specified but don't provide a service. {}".format(
                    len(nodes_no_service),
                    more_info if len(nodes_no_service) else ""))
            # Raise Warnings
            if raise_warnings:
                more_info = "See ModelValidator.warnings['nodes_no_service'] for more info"
                w = "{} nodes were specified but don't provide a service. {}".format(
                    len(nodes_no_service),
                    more_info if len(nodes_no_service) else "")
                warnings.warn(w)

        def nodes_requesting_self(p, r):
            """
            Identifies any nodes which request services of themselves. Adds (index, Node Name) pairs
            to the warnings dictionary, under the key "nodes_requesting_self".

            If `verbose` is `True` will print nodes whic`h were identified . If `raise_warnings` is
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

        def nodes_no_requested_service():
            """
            Identify nodes or technologies which have been specified in the model description but
            don't request services from other nodes.
            
            Parameters
            ----------
            None

            Returns
            -------
            None
            """
            # The model's DataFrame
            data = self.model_df

            # Add a Column w/ Technology Name
            node_names = data['Node']
            node_boundaries = node_names.apply(lambda x: '' if x is not None else x)
            techs = data[data['Parameter'] == 'technology']['Context']
            node_boundaries.update(techs)
            tech_names = node_boundaries.ffill()
            data['tech'] = tech_names

            # Forward Fill Node IDs
            data['node_id'] = [self.index2node_index_map[i] for i in data.index]
            data['node_id'] = data['node_id'].fillna(0).astype(int)
            data['node_id'] = data['node_id'].ffill()

            # Select the columns that will tell us things
            data_subset = data[['node_id', 'tech', 'Parameter']]
            group_cols = ['node_id', 'tech']
            grouped = data_subset.groupby(group_cols)['Parameter'].apply(list).reset_index()

            # Find techs with no service request
            technologies = grouped[grouped['tech'] != ""]
            serv_req_bool = technologies['Parameter'].apply(lambda x: 'service requested' in x)
            techs_no_serv_req = technologies[~serv_req_bool]

            # Find nodes with no service request
            nodes = grouped[grouped['tech'] == '']
            nodes_non_tech = nodes[~nodes['node_id'].isin(technologies['node_id'])]
            serv_req_bool = nodes_non_tech['Parameter'].apply(lambda x: 'service requested' in x)
            nodes_no_serv_req = nodes_non_tech[~serv_req_bool]

            # Combine Techs & Nodes
            nodes_and_techs = pd.concat([techs_no_serv_req, nodes_no_serv_req])

            # Create our Warning information
            node_names = [self.index2node_map[i] for i in nodes_and_techs['node_id']]
            nodes_techs_no_serv_req = list(zip(nodes_and_techs['node_id'],
                                               node_names,
                                               nodes_and_techs['tech']))

            if len(nodes_techs_no_serv_req) > 0:
                self.warnings['nodes_no_requested_service'] = nodes_techs_no_serv_req

            if verbose or raise_warnings:
                info = "{} nodes or technologies don't request other " \
                       "services.".format(len(nodes_techs_no_serv_req))
                more_info = "See ModelValidator.warnings['nodes_no_requested_service'] " \
                            "for more info."

                if verbose:
                    print("{} {}".format(info, more_info if len(nodes_techs_no_serv_req) else ""))
                if raise_warnings:
                    w = ("{} {}".format(info, more_info if len(nodes_techs_no_serv_req) else ""))
                    warnings.warn(w)

        # def discrepencies_in_model_and_tree():
        #     """
        #     Determine whether there are any discrepancies in the Model Description Excel file between
        #     the Model and Tree sheet, in terms of structure and order.
        #
        #     Parameters
        #     ----------
        #     None
        #
        #     Returns
        #     -------
        #     None
        #     """
        #     excel_engine_map = {'.xlsb': 'pyxlsb',
        #                         '.xlsm': 'xlrd'}
        #     excel_engine = excel_engine_map[os.path.splitext(self.xl_file)[1]]
        #     mxl_tree = pd.read_excel(self.xl_file, sheet_name='Tree', header=3, engine=excel_engine)
        #
        #     tree_df = mxl_tree.replace({np.nan: None})
        #     tree_sheet = pd.Series(tree_df['Branch']).dropna().reset_index(drop=True).str.lower()
        #
        #     d = self.model_df
        #     p = d[d['Parameter'] == 'service provided']['Branch']
        #     model_sheet = pd.Series(p).reset_index(drop=True).str.lower()
        #
        #     nodes_with_discrepencies = []
        #     for i, n in tree_sheet.iteritems():
        #         discrepancy = False
        #         if n not in list(model_sheet):
        #             discrepancy = True
        #             model_order = None
        #         else:
        #             if i != model_sheet[model_sheet == n].index[0]:
        #                 discrepancy = True
        #                 model_order = model_sheet[model_sheet == n].index[0]
        #         if discrepancy:
        #             nodes_with_discrepencies.append((i, model_order, n))
        #
        #     if len(nodes_with_discrepencies) > 0:
        #         self.warnings['discrepencies_in_model_and_tree'] = nodes_with_discrepencies
        #
        #     # Print Problems
        #     if verbose:
        #         more_info = "See ModelValidator.warnings['discrepencies_in_model_and_tree'] for " \
        #                     "more info."
        #         print("{} nodes have been defined in a different order between the model and tree "
        #               "sheets. {}".format(len(nodes_with_discrepencies),
        #                                   more_info if len(nodes_with_discrepencies) else ""))
        #     # Raise Warnings
        #     if raise_warnings:
        #         more_info = "See ModelValidator.warnings['discrepencies_in_model_and_tree'] for " \
        #                     "more info."
        #         w = "{} nodes have been defined in a different order between the model and tree" \
        #             "sheets. {}".format(len(nodes_with_discrepencies),
        #                                 more_info if len(nodes_with_discrepencies) else "")
        #         warnings.warn(w)

        def nodes_with_zero_output():
            """
            Identify nodes or technologies where the "Output" line has been set to 0 for 
            any of the year values in the model description.
            
            Parameters
            ----------
            None

            Returns
            -------
            None
            """
            output = self.model_df[self.model_df['Parameter'] == 'output'].iloc[:, 8:19]
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
                                                                      more_info if len(
                                                                          zero_output_nodes) else ""))
            # Raise Warnings
            if raise_warnings:
                more_info = "See ModelValidator.warnings['nodes_with_zero_output'] for more info"
                w = "{} nodes have 0 in the output line. {}".format(len(zero_output_nodes),
                                                                    more_info if len(
                                                                        zero_output_nodes) else "")
                warnings.warn(w)

        def fuel_nodes_no_lcc_or_price():
            """
            Identify fuel nodes that do not have neither a 'financial life cycle cost' or 'price'
            row specified in the base year.
            
            Parameters
            ----------
            None

            Returns
            -------
            None
            """
            d = self.model_df[self.model_df['Parameter'] == 'node type'][
                    'Context'].str.lower() == 'supply'
            supply_nodes = [self.index2node_map[i] for i, v in d.iteritems() if v]

            no_prod_cost = []
            for n in supply_nodes:
                node = self.index2node_map[self.index2node_map == n]
                dat = self.model_df.loc[node.index, :]
                s = dat[dat['Parameter'] == 'competition type']['Context'].str.lower()
                if 'fuel - fixed price' in s.to_string():
                    # Check whether fLCC is specified
                    if 'financial life cycle cost' not in dat['Parameter'].values:
                        fLCC_specified = False
                    else:
                        year_columns = [c for c in dat.columns if is_year(c)]
                        base_year = year_columns[0]
                        fLCC = \
                            dat[dat['Parameter'] == 'financial life cycle cost'][base_year].values[
                                0]
                        if fLCC is None:
                            fLCC_specified = False
                        else:
                            fLCC_specified = True

                    # Check whether price is specified
                    if 'price' not in dat['Parameter'].values:
                        price_specified = False
                    else:
                        year_columns = [c for c in dat.columns if is_year(c)]
                        base_year = year_columns[0]
                        price = dat[dat['Parameter'] == 'price'][base_year].values[0]
                        if price is None:
                            price_specified = False
                        else:
                            price_specified = True

                    # Check that one of fLCC or price is specified
                    if not (fLCC_specified or price_specified):
                        no_prod_cost.append((node.index[0], n))

            if len(no_prod_cost) > 0:
                self.warnings['fuels_without_lcc_or_price'] = no_prod_cost

            # Print Problems
            if verbose:
                more_info = "See ModelValidator.warnings['fuels_without_lcc_or_price'] for more info"
                print("{} fuel nodes don't have a life cycle cost or price. {}".format(
                    len(no_prod_cost), more_info if len(no_prod_cost) else ""))
            # Raise Warnings
            if raise_warnings:
                more_info = "See ModelValidator.warnings['fuels_without_lcc_or_price'] for more info"
                w = "{} fuel nodes don't have a life cycle cost or price. {}".format(
                    len(no_prod_cost), more_info if len(no_prod_cost) else "")
                warnings.warn(w)

        def nodes_no_capital_cost():
            """
            Identify tech compete nodes/technologies where the "Capital cost_overnight" row 
            hasn't been included in the model description. 
            
            Parameters
            ----------
            None

            Returns
            -------
            None
            """
            nodes = self.model_df[self.model_df['Parameter'] == 'service provided']['Branch']
            nodes.index = nodes.index - 1
            end = pd.Series(['None'], index=[self.model_df.index.max()])
            nodes = pd.concat([nodes, end])
            no_cap_cost = []

            for i in range(nodes.shape[0] - 1):
                node_index = nodes.index[i]
                node_name = nodes.iloc[i]
                start_index = nodes.index[i]
                end_index = nodes.index[i + 1]
                dat = self.model_df.loc[start_index:end_index]
                tech_nodes = dat[dat['Parameter'] == 'competition type'][
                                 'Context'].str.lower() == 'tech compete'
                if tech_nodes.iloc[0]:
                    if 'technology' not in list(dat['Parameter']):
                        if 'capital cost_overnight' not in list(dat['Parameter']):
                            no_cap_cost.append((node_index, node_name))
                    else:
                        techs = dat[dat['Parameter'] == 'technology']['Context']
                        end = pd.Series(['None'], index=[dat.index.max()])
                        techs = pd.concat([techs, end])
                        for i in range(techs.shape[0] - 1):
                            tech_name = techs.iloc[i]
                            start_index = techs.index[i]
                            end_index = techs.index[i + 1]
                            if 'capital cost_overnight' not in list(
                                    dat['Parameter'].loc[start_index:end_index]):
                                no_cap_cost.append((node_index, node_name, tech_name))

            if len(no_cap_cost) > 0:
                self.warnings['nodes_without_capital_cost'] = no_cap_cost

            # Print Problems
            if verbose:
                more_info = "See ModelValidator.warnings['nodes_without_capital_cost'] for more info"
                print("{} tech compete nodes don't have capital cost. {}".format(len(no_cap_cost),
                                                                                 more_info if len(
                                                                                     no_cap_cost) else ""))
            # Raise Warnings
            if raise_warnings:
                more_info = "See ModelValidator.warnings['nodes_without_capital_cost'] for more info"
                w = "{} tech compete nodes don't have capital cost. {}".format(len(no_cap_cost),
                                                                               more_info if len(
                                                                                   no_cap_cost) else "")
                warnings.warn(w)

        def nodes_bad_total_market_share(Precision=3):
            """
            Identify nodes with market shares not totally 100% in the Base Year.
            
            Parameters
            ----------
            Precision : int
                        Precision is the number of decimals we will round to when market shares are 
                        summed across nodes. For now, the default value is 3.

            Returns
            -------
            None
            """
            # The model's dataframe
            data = self.model_df

            # Find what column contains the base year values
            base_year_col = [[c for c in data.columns if is_year(c)][0]]

            # For each market share, find what node it belongs to (by index)
            market_shares = data[data['Parameter'] == 'market share'][base_year_col]
            market_shares['Node'] = [int(self.index2node_index_map[i]) for i in market_shares.index]

            # Sum the market shares for each node
            ms_totals = round(market_shares.groupby('Node').sum(), Precision)

            # Find which nodes have a bad total market share
            ms_total_not_100 = ms_totals[ms_totals[base_year_col[0]] != 1].reset_index()
            node_names = [self.index2node_map[i] for i in ms_total_not_100['Node']]
            nodes_with_bad_total_ms = list(zip(ms_total_not_100['Node'],
                                               node_names,
                                               ms_total_not_100[base_year_col[0]]))

            if len(nodes_with_bad_total_ms) > 0:
                self.warnings['nodes_with_bad_total_ms'] = nodes_with_bad_total_ms

            if verbose:
                info = "{} nodes have market shares that don't sum to 100%.".format(
                    len(nodes_with_bad_total_ms))
                more_info = "See ModelValidator.warnings['nodes_with_bad_total_ms'] for more info."
                print("{} {}".format(info,
                                     more_info if len(nodes_with_bad_total_ms) else ""))

            if raise_warnings:
                info = "{} nodes have market shares that don't sum to 100%.".format(
                    len(nodes_with_bad_total_ms))
                more_info = "See ModelValidator.warnings['nodes_with_bad_total_ms'] for more info."
                w = ("{} {}".format(info,
                                    more_info if len(nodes_with_bad_total_ms) else ""))
                warnings.warn(w)

        def techs_no_base_market_share():
            """
            Identify technologies without a base year market share.

            Parameters
            ----------
            None

            Returns
            -------
            None
            """
            # The model's DataFrame
            data = self.model_df

            # Add a Column w/ Technology Name
            techs = data[data['Parameter'] == 'technology']['Context']
            data['Tech'] = techs
            data['Tech'] = data['Tech'].ffill()

            # Find Market Share Rows
            market_shares = data[data['Parameter'] == 'market share']

            # Find Instances where there isn't a base year market share
            base_year_col = [c for c in data.columns if is_year(c)][0]
            ms_no_base_year = market_shares[market_shares[base_year_col].isna()]

            # Create our Warning information
            node_names = [self.index2node_map[i] for i in ms_no_base_year.index]
            techs_no_base_year_ms = list(zip(ms_no_base_year.index,
                                             node_names,
                                             ms_no_base_year['Tech']))

            if len(techs_no_base_year_ms) > 0:
                self.warnings['techs_no_base_year_ms'] = techs_no_base_year_ms

            if verbose:
                info = "{} technologies are missing a base year market share.".format(
                    len(techs_no_base_year_ms))
                more_info = "See ModelValidator.warnings['techs_no_base_year_ms'] for more info."
                print("{} {}".format(info,
                                     more_info if len(techs_no_base_year_ms) else ""))

            if raise_warnings:
                info = "{} technologies are missing a base year market share.".format(
                    len(techs_no_base_year_ms))
                more_info = "See ModelValidator.warnings['techs_no_base_year_ms'] for more info."
                w = ("{} {}".format(info,
                                    more_info if len(techs_no_base_year_ms) else ""))
                warnings.warn(w)

        def duplicate_service_requests():
            """
            Identify technologies which request the same service twice.
            
            Parameters
            ----------
            None

            Returns
            -------
            None
            """
            # The model's DataFrame
            data = self.model_df

            # Add a Column w/ Technology Name
            node_names = data['Node']
            node_boundaries = node_names.apply(lambda x: '' if x is not None else x)
            techs = data[data['Parameter'] == 'technology']['Context']
            node_boundaries.update(techs)
            tech_names = node_boundaries.ffill()
            data['tech'] = tech_names

            # Forward Fill Node IDs
            data['node_id'] = [self.index2node_index_map[i] for i in data.index]
            data['node_id'] = data['node_id'].fillna(0).astype(int)
            data['node_id'] = data['node_id'].ffill()

            # Filter to Service Request Rows only
            services_req = data[data['Parameter'] == 'service requested']

            # Select the columns that will tell us things
            req_info = services_req[['node_id', 'tech', 'Branch']]
            duplicated = req_info[req_info.duplicated(keep=False)]

            if len(duplicated) > 0:
                # Group & list rows (index) where duplicates exist
                duplicated_with_idx = duplicated.reset_index()
                duplicated_groups = duplicated_with_idx.groupby(['node_id', 'tech', 'Branch'])[
                    'index'].apply(list)
                duplicated_groups = duplicated_groups.reset_index()

                # Create our Warning information
                node_names = [self.index2node_map[i] for i in duplicated_groups['node_id']]
                duplicate_req = list(zip(duplicated_groups['index'],
                                         node_names,
                                         duplicated_groups['tech']))
            else:
                duplicate_req = []

            if len(duplicate_req) > 0:
                self.warnings['duplicate_req'] = duplicate_req

            if verbose:
                info = "{} nodes/technologies request from the same service more than once.".format(
                    len(duplicate_req))
                more_info = "See ModelValidator.warnings['duplicate_req'] for more info."
                print("{} {}".format(info,
                                     more_info if len(duplicate_req) else ""))

            if raise_warnings:
                info = "{} nodes/technologies request from the same service more than once.".format(
                    len(duplicate_req))
                more_info = "See ModelValidator.warnings['duplicate_req'] for more info."
                w = ("{} {}".format(info,
                                    more_info if len(duplicate_req) else ""))
                warnings.warn(w)

        def bad_service_req():
            """
            Identify nodes/technologies that have a service requested line, but where the values in
            this lines are either blank or exogenously specified as 0.
            
            Parameters
            ----------
            None

            Returns
            -------
            None
            """
            # The model's DataFrame
            data = self.model_df

            # Filter to Only Include Service Requested
            services_req = data[data['Parameter'] == 'service requested']

            # Select only the year columns
            year_cols = [c for c in services_req.columns if is_year(c)]
            year_values = services_req[year_cols]

            # Identify rows that have 0's or missing values
            rows_nan_zero = year_values.replace(0, np.nan).isna().sum(axis=1)
            row_has_bad_values = rows_nan_zero == len(year_cols)
            rows_with_bad_values = services_req[row_has_bad_values]

            # Create our Warning information
            node_names = [self.index2node_map[i] for i in rows_with_bad_values.index]
            bad_service_req = list(zip(rows_with_bad_values.index, node_names))

            if len(bad_service_req) > 0:
                self.warnings['bad_service_req'] = bad_service_req

            if verbose or raise_warnings:
                info = "{} nodes/technologies contain only 0's or are missing values in service " \
                       "request rows.".format(len(bad_service_req))
                more_info = "See ModelValidator.warnings['bad_service_req'] for more info."

                if verbose:
                    print("{} {}".format(info, more_info if len(bad_service_req) else ""))
                if raise_warnings:
                    w = ("{} {}".format(info, more_info if len(bad_service_req) else ""))
                    warnings.warn(w)

        def tech_compete_nodes_no_techs():
            """
            Identify tech compete nodes that don't contain 'technology' or 'service' headings --
            thereby appearing to not have a technology present.
                    
            Parameters
            ----------
            None

            Returns
            -------
            None
            """

            def check_for_header(lst):
                lower_lst = [str(p).lower() for p in lst]
                if ('technology' in lower_lst) or ('service' in lower_lst):
                    header_present = True
                else:
                    header_present = False
                return header_present

            # The model's DataFrame
            data = self.model_df

            # Forward Fill Node IDs
            data['node_id'] = [self.index2node_index_map[i] for i in data.index]
            data['node_id'] = data['node_id'].fillna(0).astype(int)
            data['node_id'] = data['node_id'].ffill()

            # Add a Column w/ Competition Type
            comp_types = data[data['Parameter'] == 'competition type'][['node_id', 'Context']]
            comp_types.columns = ['node_id', 'comp_type']
            data = data.merge(comp_types, on='node_id')

            # Only include Tech Compete Nodes
            tech_compete = data[data['comp_type'] == 'Tech Compete']

            # Find which columns are missing headers
            all_tc = tech_compete[['node_id', 'Parameter']]
            params = all_tc.groupby('node_id')['Parameter'].apply(list)
            missing_header = params[~params.apply(check_for_header)]

            # Create our Warning information
            node_names = [self.index2node_map[i] for i in missing_header.index]
            tc_nodes_no_techs = list(zip(missing_header.index,
                                         node_names))

            if len(tc_nodes_no_techs) > 0:
                self.warnings['tech_compete_nodes_no_techs'] = tc_nodes_no_techs

            if verbose or raise_warnings:
                info = "{} tech compete nodes don't contain Technology or Service " \
                       "headings".format(len(tc_nodes_no_techs))
                more_info = "See ModelValidator.warnings['tech_compete_nodes_no_techs'] for " \
                            "more info."

                if verbose:
                    print("{} {}".format(info, more_info if len(tc_nodes_no_techs) else ""))
                if raise_warnings:
                    w = ("{} {}".format(info, more_info if len(tc_nodes_no_techs) else ""))
                    warnings.warn(w)

        def market_child_requested():
            """
            Identify any requests for fuel which are made to nodes which are part of a Market.
            """
            data = self.model_df

            # Find Markets
            markets = self.model_df[(self.model_df['Parameter'] == 'competition type') &
                                    (self.model_df['Context'] == 'Market')]
            market_node_idxs = [self.index2node_index_map[i] for i in markets.index]

            # Find Nodes that are children of a market
            all_service_req = data[data['Parameter'] == 'service requested']['Branch']
            market_children = [b for i, b in all_service_req.items()
                               if self.index2node_index_map[i] in market_node_idxs]

            # Find Service Requests for Market Children
            market_children_requests = [(i, self.index2branch_map[i], b) for i, b in
                                        all_service_req.items()
                                        if (b in market_children) and
                                        (self.index2node_index_map[i] not in market_node_idxs)]

            if len(market_children_requests) > 0:
                self.warnings['market_child_requested'] = market_children_requests

            if verbose or raise_warnings:
                info = f"{len(market_children_requests)} nodes/technologies requested services " \
                       f"from nodes which are part of a market."
                more_info = "See ModelValidator.warnings['market_child_requested'] for more info."

                if verbose:
                    print(f"{info} {more_info if len(market_children_requests) else ''}")
                if raise_warnings:
                    w = f"{info} {more_info if len(market_children_requests) else ''}"
                    warnings.warn(w)

        def revenue_recycling_at_techs():
            """Revenue recycling should only happen at nodes, never at techs"""
            # The model's DataFrame
            data = self.model_df

            # Add a Column w/ Technology Name
            techs = data[data['Parameter'] == 'technology']['Context']
            data['Tech'] = techs
            # data['Tech'] = data['Tech'].ffill()

            # nodes_ffill = data['Node'].map(lambda x: None if pd.isna(x) else 'Node')
            # techs_ffill = data['Tech'].map(lambda x: None if pd.isna(x) else 'Tech')
            node_or_tech = []
            for n, t in zip(data['Node'], data['Tech']):
                if not pd.isna(t):
                    node_or_tech.append('Tech')
                elif not pd.isna(n):
                    node_or_tech.append('Node')
                else:
                    node_or_tech.append(None)

            data['Node_or_Tech'] = node_or_tech
            data['Node_or_Tech'] = data['Node_or_Tech'].ffill()

            # Find Recycled Revenues Rows
            recycled_revenues = data[data['Parameter'] == 'recycled revenues']

            # Find instances where recycled revenues is defined for a tech
            rr_at_tech = recycled_revenues[recycled_revenues['Node_or_Tech'] == 'Tech']

            # Create our Warning information
            node_names = [self.index2node_map[i] for i in rr_at_tech.index]
            tech_names = [data['Tech'].ffill().loc[i] for i in rr_at_tech.index]
            techs_recycling_revenues = list(zip(rr_at_tech.index, node_names, tech_names))

            if len(techs_recycling_revenues) > 0:
                self.warnings['techs_revenue_recycling'] = techs_recycling_revenues

            if verbose:
                info = "{} technologies are trying to recycle revenues.".format(
                    len(techs_recycling_revenues))
                more_info = "See ModelValidator.warnings['techs_revenue_recycling'] for more info."
                print("{} {}".format(info,
                                     more_info if len(techs_recycling_revenues) else ""))

            if raise_warnings:
                info = "{} technologies are trying to recycle revenues".format(
                    len(techs_recycling_revenues))
                more_info = "See ModelValidator.warnings['techs_revenue_recycling'] for more info."
                w = ("{} {}".format(info,
                                    more_info if len(techs_recycling_revenues) else ""))
                warnings.warn(w)

        def both_cop_p2000_defined():
            """No node should have both COP & P2000 exogenously defined"""

            data = self.model_df

            # Find all instances of COP & P2000 in the model description
            cop_p2000 = data[data['Parameter'].isin(['cop', 'p2000'])]

            # Only keep the ones that aren't None
            cop_p2000 = cop_p2000.dropna(how='all',
                                         subset=[c for c in cop_p2000.columns if is_year(c)])

            # Find what node each parameter is referencing
            node_indexes = pd.Series([self.index2node_index_map[i] for i in cop_p2000.index])
            node_indexes_with_both = node_indexes[node_indexes.duplicated()]

            # Create our Warning information
            branches = [self.index2branch_map[i] for i in node_indexes_with_both]
            nodes_with_cop_and_p2000 = list(zip(node_indexes_with_both, branches))

            if len(nodes_with_cop_and_p2000) > 0:
                self.warnings['nodes_with_cop_and_p2000'] = nodes_with_cop_and_p2000

            if verbose:
                info = "{} node(s) have both COP & P2000 exogenously defined.".format(
                    len(nodes_with_cop_and_p2000))
                more_info = "See ModelValidator.warnings['nodes_with_cop_and_p2000'] for more info."
                print("{} {}".format(info,
                                     more_info if len(nodes_with_cop_and_p2000) else ""))

            if raise_warnings:
                info = "{} node(s) have both COP & P2000 exogenously defined".format(
                    len(nodes_with_cop_and_p2000))
                more_info = "See ModelValidator.warnings['nodes_with_cop_and_p2000'] for more info."
                w = ("{} {}".format(info,
                                    more_info if len(nodes_with_cop_and_p2000) else ""))
                warnings.warn(w)

        providers = self.model_df[self.model_df['Parameter'] == 'service provided']['Branch']
        requested = self.model_df[self.model_df['Parameter'] == 'service requested']['Branch']
        roots = self.find_roots()

        mismatched_node_names(providers)
        unspecified_nodes(providers, requested)
        unreferenced_nodes(providers, requested, roots)
        nodes_no_provided_service(providers)
        invalid_competition_type()
        nodes_requesting_self(providers, requested)
        nodes_no_requested_service()
        # discrepencies_in_model_and_tree()
        nodes_with_zero_output()
        fuel_nodes_no_lcc_or_price()
        nodes_no_capital_cost()
        # nodes_bad_total_market_share()
        techs_no_base_market_share()
        duplicate_service_requests()
        bad_service_req()
        tech_compete_nodes_no_techs()
        market_child_requested()
        revenue_recycling_at_techs()
        both_cop_p2000_defined()
