#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
import re
from IPython.display import Image, display, Markdown, HTML
get_ipython().run_line_magic('matplotlib', 'inline')

from helper.direct_display import one_cell_toggle

# HTML version: jupyter nbconvert read_model.ipynb --to html --TemplateExporter.exclude_input=True


# In[10]:


# Make back into .py script

get_ipython().system('jupyter nbconvert --to script read_model.ipynb')


# In[13]:


infile = "../pyCIMS_model_description.xlsm"

NODE_COL = "Node"
MODEL_SHEET = "Model"
# TODO extract other string constants used below and place them in some constants here

mxl = pd.read_excel(infile, sheet_name=None, header=1)
mdf0 = mxl[MODEL_SHEET].replace({pd.np.nan: None})
mdf0.index += 3  # adjust index to correspond to Excel line numbers (+1: 0 vs 1 origin, +1: header skip, +1: column headers)

one_cell_toggle()


# In[15]:


display(Markdown("""
# pyCIMS model reader

The code here reads [{}]({}) and extracts the '{}' sheet into a Pandas DataFrame.
The dataframe is then split and the referenced parts are organized in dictionaries, see below.

Later, the individual node and tech frames are analyzed for service connectivity information, 
enabling the construction of a graph.
""".format(infile, infile, MODEL_SHEET)))
one_cell_toggle()


# In[16]:


# DataFrame helpers
def display_df(df):
    """Display None as blanks"""
    df = pd.DataFrame(df)
    if not df.empty:
        display(pd.DataFrame(df).replace({None: ''}))
def non_empty_rows(df, exclude_column="Node"):
    """Return bool array to flag rows as False that have only None or False values, ignoring exclude_column"""
    return df.loc[:, df.columns != exclude_column].T.apply(any)

# column extraction helpers
re_year = re.compile(r'^[0-9]{4}$')
def is_year(cn):
    """Check if input int or str is 4 digits [0-9] between begin ^ and end $ of string"""
    # unit test: assert is_year, 1900
    return bool(re_year.match(str(cn)))

def find_first(items, pred=bool, default=None):
    """Find first item for that pred is True"""
    return next(filter(pred, items), default)

def find_first_index(items, pred=bool):
    """Find index of first item for that pred is True"""
    return find_first(enumerate(items), lambda kcn: pred(kcn[1]))[0]

def get_node_cols(mdf, first_data_col_name="Node"):
    """Returns list of column names after 'Node' and a list of years that follow """
    node_col_idx = find_first_index(mdf.columns,
                                    lambda cn: first_data_col_name.lower() in cn.lower())
    relevant_columns = mdf.columns[node_col_idx:]
    year_or_not = list(map(is_year, relevant_columns))
    first_year_idx = find_first_index(year_or_not)
    last_year_idx = find_first_index(year_or_not[first_year_idx:],
                                     lambda b: not b) + first_year_idx
    # list(...)[a:][:b] extracts b elements starting at a
    year_cols = mdf.columns[node_col_idx:][first_year_idx:last_year_idx]
    return mdf.columns[node_col_idx:][:first_year_idx], year_cols


# In[17]:


node_cols, year_cols = get_node_cols(mdf0)
all_cols = np.concatenate((node_cols, year_cols))
mdf = mdf0.loc[1:,all_cols] # drop irrelevant columns and skip first, empty row


# In[6]:


# display_df(mdf)


# In[7]:


# determine, row ranges for each node def, based on non-empty Node field
node_rows = mdf.Node[~mdf.Node.isnull()] # does not work if node names have been filled in
node_rows.index.name = "Row Number"
last_row = mdf.index[-1]
node_start_ends = zip(node_rows.index,
                      node_rows.index[1:].tolist() + [last_row])


# In[8]:


# extract Node DataFrames, at this point still including Technologies
node_dfs = {}
non_node_cols = mdf.columns != NODE_COL
for s, e in node_start_ends:
    node_name = mdf.Node[s]
    node_df = mdf.loc[s+1:e-1]
    node_df = node_df.loc[non_empty_rows(node_df), non_node_cols]
    # mdf.loc[s+1:e-1, "Node"] = node_name
    node_dfs[node_name] = node_df
# len(node_dfs)


# In[9]:


## intermediate output for dev purposes
# for nn, ndf in node_dfs.items():
#     display(Markdown("Node: **{}**".format(nn)))
#     display_df(ndf)


# In[10]:


# Extract tech dfs from node df's and rewrite node df without techs
tech_dfs = {}
for nn, ndf in node_dfs.items():
    if any(ndf.Parameter == "Technology"):
        tdfs = {}
        first_row, last_row = ndf.index[0], ndf.index[-1]
        tech_rows = ndf.loc[ndf.Parameter == "Technology"].index
        for trs, tre in zip(tech_rows, tech_rows[1:].tolist()+[last_row]):
            tech_df = ndf.loc[trs:tre-1]
            tech_name = tech_df.iloc[0].Value
            tdfs[tech_name] = tech_df
        tech_dfs[nn] = tdfs
        node_dfs[nn] = ndf.loc[:tech_rows[0]-1]


# # Display Nodes and Technologies
# `node_dfs` contains a dictionary giving the DataFrames that hold the relevant rows for each node, without Technology info  
# `tech_dfs` contains a dictionary for each node naming the technologies and holding the relevant rows in a DF

# In[11]:


# display content of entire Model dataframe as separate df's
for nn, ndf in node_dfs.items():
    display(Markdown("Node: **{}**".format(nn)))
    display_df(ndf)
    if nn in tech_dfs:
        for tech_name, tdf in tech_dfs[nn].items():
            display(Markdown("Node / Technology: **{} / {}**".format(nn, tech_name)))
            display_df(tdf)


# # Display Service node paths and demand connections 

# In[12]:


## simple node connectivity output
# for nn, ndf in node_dfs.items():
#     supply_path = ndf.loc[ndf.Parameter.str.lower() == "service supply", "Branch"]
#     if not supply_path.empty:
#         display(Markdown("## Node: **{}**".format(nn)))
#         display(Markdown("Path: **{}**".format(supply_path.values[0])))
#         if any(ndf.Parameter.str.lower() == "service demand"):
#             display(Markdown("### Service demand connections"))
#             display_df(ndf.loc[ndf.Parameter.str.lower() == "service demand", ["Branch","Value"]])
#         if nn in tech_dfs:
#             for tech_name, tdf in tech_dfs[nn].items():
#                 display(Markdown("### Technology service demand connections"))
#                 display(Markdown("Node / Technology: **{} / {}**".format(nn, tech_name)))
#                 display_df(tdf.loc[tdf.Parameter.str.lower() == "service demand", ["Branch","Value"]])
#                 # display_df(tdf)


# In[13]:


nodes = set()
edges = set()

def process_connection(con_path, con_name, what=""):
    if con_name and con_name != con_path.split(".")[-1]:
        display(Markdown("   **{}** '{}': '{}'".format(what, con_name, con_path)))
    else:
        display(Markdown("   **{}**: '{}'".format(what, con_path)))

def process_node_service_demands(ndf, what="", separate_table_per_name=False, func=display_df):
    # instead of just displaying dfs below, we could create an edge for each row
    service_demand_idxs = ndf.Parameter.str.lower() == "service demand"
    if any(service_demand_idxs):
        selected_cols = ["Value", "Branch", "Unit"] + year_cols.tolist()
        service_demand_df = ndf.loc[service_demand_idxs]
        service_demand_names = service_demand_df.Value.unique()
        if any(service_demand_names): # named services
            if not separate_table_per_name:
                sd_df = (service_demand_df.sort_values("Value")[selected_cols]
                         .rename(columns={"Value": what if what else "Service Type"}))
                func(sd_df)
            else: # separate table for each demand name
                for name in service_demand_names:
                    sd_df = (service_demand_df.loc[service_demand_df.Value == name, selected_cols]
                             .rename(columns={"Value": what if what else "Service Type"}))
                    
                    func(sd_df)
        else: # components with unnamed paths only
            func(service_demand_df[["Branch"] + year_cols.tolist()].rename(columns={"Parameter":"Component"}))
#             for con_path in service_demand_df["Branch"].values:
#                 process_connection(con_path, None, what="Component")

def create_edges(df, node_path=None,
                 show_df=False, do_print=True): # use show_df=True for debugging
    if do_print:
        display(Markdown("### Node Path: {}".format(node_path)))
        print("at node '{}'".format(node_path))
    for index, rdf in df.iterrows():
        target_path = rdf["Branch"]
        tech_path = None
        if rdf.index[0] == "Branch":
            target_name = None
        else:
            target_name = rdf.iloc[0]
            if not target_path or target_name != target_path.split(".")[-1]:
                tech_path = ".".join([node_path, target_name])
                if do_print:
                    if target_path:
                        print("    via '{}'".format(tech_path))
                    else:
                        print("    connect '{}'".format(tech_path))
            else:
                if target_path:
                    target_name = None
        if do_print and target_path:
            print("        connect '{}' ".format(target_path))            
        if target_path:
            nodes.add(target_path)
            if tech_path:
                nodes.add(tech_path)
                edges.update([(node_path, tech_path), (tech_path, target_path)])
            else:
                edges.add((node_path, target_path))
        else:
            if tech_path:
                nodes.add(tech_path)
                edges.add((node_path, tech_path))
            else:
                print("WARNING: Missing target path and tech path in node '{}' tech '{}'".format(node_path, target_name))
                display_df(df)
    if show_df:
        display_df(df)

extract_nodes_and_edges = True
if not extract_nodes_and_edges:
    func = lambda df, node_path: display_df(df) # just display
    separate_tables = True
else:
    func = create_edges
    separate_tables = True
for nn, ndf in node_dfs.items():
    supply_path = ndf.loc[ndf.Parameter.str.lower() == "service supply", "Branch"]
    if not supply_path.empty:
        node_path = supply_path.values[0]
        process_node_service_demands(ndf, what="Service Type",
                                     separate_table_per_name=separate_tables,
                                     func=lambda df: func(df, node_path=node_path))
        if nn in tech_dfs:
            all_tech_dfs = pd.concat(tech_dfs[nn].values())
            process_node_service_demands(all_tech_dfs, what="Technology",
                                         separate_table_per_name=separate_tables,
                                         func=lambda df: func(df, node_path=node_path))


# ## Make graph and display it

# In[20]:


import networkx as nx
to_pdot = nx.drawing.nx_pydot.to_pydot
def view_pydot(pdot, width=800):
    plt = Image(pdot.create_png(),
               width=width, unconfined=True)
    display(plt)


# In[15]:


G = nx.Graph()
G.add_nodes_from(nodes)
G.add_edges_from(edges)


# In[16]:


## not very useful layout (don't use this)
#import matplotlib.pyplot as plt
#plt.rcParams['figure.figsize'] = [8, 6]
#nx.draw_networkx(G, font_size=10, labels=dict((nn, nn.split(".")[-1]) for nn in nodes))


# In[17]:


#with open("whole-graph.png","wb") as fh:
#    fh.write(to_pdot(G).create_png())

for k, C in enumerate(nx.connected_component_subgraphs(G)):
    pdot = to_pdot(C)
    display(Markdown("### Graph Component {}".format(k)))
    view_pydot(pdot, width=1600)


# In[ ]:




