from ..model_description import column_list as COL
# ==========================================
# Data Structure Constants
# ==========================================
context = COL.context.lower()
param_source= "param_source"
source= "source"
sub_context = COL.sub_context.lower()
target= COL.target.lower()
technologies = "technologies"
unit= COL.unit.lower()
year_value = "year_value"

# ==========================================
# General
# ==========================================
available = "available" # general
competition_type = "competition type" # general
edge_type = "type" # general
heterogeneity = 'heterogeneity' # general
is_supply = 'is supply' # general
lifetime = "lifetime"   # general
service_provided = "service provided" # general
service_requested = "service requested"
tree_index = 'tree index' # general
unavailable = "unavailable" # general
no_inheritance = "no_inheritance" # general

# ==========================================
# Life Cycle Cost
# ==========================================
benchmark = "benchmark" # lcc
capital_cost = "capital cost"   # lcc
capital_recovery = "capital recovery" # lcc
competition_annual_cost = "competition annual cost" # lcc
competition_upfront_cost = "competition upfront cost" # lcc
cop = "cop" # lcc
cost_curve_function = "cost_curve_function" # lcc
cost_curve_lcc_max = "cost_curve_lcc_max" # lcc
cost_curve_lcc_min = "cost_curve_lcc_min" # lcc
cost_curve_price = 'cost curve price' # lcc
cost_curve_quantity = 'cost curve quantity' # lcc
crf = "crf" # lcc
discount_rate_financial = "discount rate_financial" # lcc
discount_rate_retrofit ='discount rate_retrofit' # lcc
financial_annual_cost = "financial annual cost" # lcc
financial_emissions_cost = "financial emissions cost" # lcc
financial_service_cost = 'financial service cost' # lcc
financial_upfront_cost = "financial upfront cost" # lcc
fixed_cost_rate = 'fixed cost rate' # lcc 
fom = "fom" # lcc
fcc = "fcc" # lcc
lcc_competition = "lcc_competition" # lcc
lcc_financial = "lcc_financial" # lcc
new_stock_financial_annual_cost = "new_stock_financial_annual_cost" # lcc
new_stock_financial_upfront_cost = "new_stock_financial_upfront_cost" # lcc
non_energy_cost = "non-energy cost" # lcc
non_energy_cost_change = "non-energy cost change" # lcc
output = "output" # lcc
p2000 = "p2000" # lcc    
price = "price" # lcc
price_multiplier = "price multiplier" # lcc
price_subsidy = "price_subsidy"  # lcc
recycled_revenues = "recycled revenues" # lcc
revenue_recycle_rate = "revenue recycle rate"   # lcc
service_cost = "service cost" # lcc
subsidy = "subsidy" # lcc
tax = "tax" # lcc
tax_foresight = "tax_foresight" # lcc
total_fixed_cost = 'total fixed cost' # lcc  
total_lcc_v = "total_lcc_v" # lcc

# ==========================================
# Stock Allocation
# ==========================================
added_retrofit_stock = "added_retrofit_stock"   # stock allocation
adjustment_multiplier = 'adjustment_multiplier' # stock_allocation
base_stock = "base_stock"   # stock allocation
base_stock_remaining = "base_stock_remaining" # stock allocation
fic = "fic" # stock allocation
domestic_elasticity = "domestic elasticity" # macro-economics
export_benchmark = "export benchmark" # macro-economics
export_elasticity = 'export elasticity' # macro-economics
export_subsidy = "export subsidy" # macro-economics
global_price = "global price" # macro-economics
market_share = "market share"   # stock allocation
market_share_class = "market share_class" # stock allocation / market share limits
market_share_class_max = "market share_class_max" # stock allocation / market share limits
market_share_class_min = "market share_class_min" # stock allocation / market share limits
market_share_new_max = "market share new_max" # stock allocation
market_share_new_min = "market share new_min" # stock allocation
new_market_share = "new_market_share" # stock allocation
new_stock = "new_stock" # stock allocation
new_stock_remaining = "new_stock_remaining" # stock allocation
new_stock_remaining_pre_surplus = 'new_stock_remaining_pre_surplus' # stock allocation
ref_stock_exported = 'ref stock exported' # stock allocation /macro-economics
retirement_intercept = "retirement intercept" # stock_allocation
retrofit_existing_max = 'retrofit_existing_max' # stock_allocation
retrofit_existing_min = 'retrofit_existing_min' # stock_allocation
retrofit_heterogeneity = 'retrofit_heterogeneity' # stock_allocation
retrofit_new_max = "retrofit_new_max"    # stock allocation
retrofit_new_min = "retrofit_new_min"   # stock allocation
retrofit_stock = "retrofit_stock" # stock allocation
stock_demanded = "stock demanded" # stock allocation / macro-economics
stock_exported = "stock exported" # stock allocation / macro-economics
total_market_share = "total_market_share" # stock allocation
total_stock = "total_stock" # stock allocation

# ==========================================
# Aggregation
# ==========================================
aggregation_requested = "aggregation requested" # aggregation
aggregation_weight = "aggregation_weight" # aggregation
distributed_supply = "distributed_supply"   # aggregation
provided_quantities = "provided_quantities" # aggregation
requested_quantities = "requested_quantities" # aggregation
structural_aggregation = "structural_aggregation" # aggregation

# ==========================================
# Emissions
# ==========================================
aggregate_emissions_cost_rate = "aggregate_emissions_cost_rate" # emissions
avoided_emissions_rate = "avoided_emissions_rate"   # emissions
bio_emissions_rate = "bio_emissions_rate" # emissions
cumul_avoided_emissions_rate = "cumul_avoided_emissions_rate" # emissions
cumul_bio_emissions_rate = "cumul_bio_emissions_rate" # emissions
cumul_emissions_cost_rate = "cumul_emissions_cost_rate" # emissions
cumul_negative_emissions_rate = "cumul_negative_emissions_rate" # emissions
cumul_net_emissions_rate = "cumul_net_emissions_rate" # emissions
emissions = "emissions" # emissions
emissions_biomass = "emissions_biomass" # emissions
emissions_cost = "emissions cost" # emissions
emissions_cost_rate = "emissions_cost_rate" # emissions
emissions_gwp = 'emissions_gwp'  # emissions     
emissions_removal = "emissions_removal" # emissions     
negative_emissions_rate = "negative_emissions_rate" # emissions
net_emissions_rate = "net_emissions_rate" # emissions
total_cumul_avoided_emissions = "total_cumul_avoided_emissions" # emissions
total_cumul_bio_emissions = "total_cumul_bio_emissions" # emissions
total_cumul_emissions_cost = "total_cumul_emissions_cost" # emissions
total_cumul_negative_emissions = "total_cumul_negative_emissions" # emissions
total_cumul_net_emissions = "total_cumul_net_emissions" # emissions
total_direct_avoided_emissions = "total_direct_avoided_emissions" # emissions
total_direct_bio_emissions = "total_direct_bio_emissions" # emissions
total_direct_emissions_cost = "total_direct_emissions_cost" # emissions
total_direct_negative_emissions = "total_direct_negative_emissions" # emissions
total_direct_net_emissions = "total_direct_net_emissions" # emissions

# ==========================================
# Declining Costs
# ==========================================
capital_cost_declining = "capital cost_declining"  # declining cost
capital_cost_min = "capital_cost_min"   # declining cost
dcc_capacity_1 = "dcc_capacity_1"   # declining costs
dcc_capacity_2 = "dcc_capacity_2"   # declining costs
dcc_capacity_3 = "dcc_capacity_3"   # declining costs
dcc_class = "dcc_class" # declining costs
dcc_limit = "dcc_limit" # declining costs
dcc_min_learning = "dcc_min learning" # declining costs
dcc_progress_ratio_1 = "dcc_progress ratio_1" # declining costs
dcc_progress_ratio_2 = "dcc_progress ratio_2" # declining costs
dcc_progress_ratio_3 = "dcc_progress ratio_3" # declining costs
dic = "dic" # declining costs
dic_class = "dic_class" # declining costs
dic_initial = "dic_initial" # declining costs
dic_slope = "dic_slope" # declining costs
dic_min = "dic_min" # declining costs
dic_x50 = "dic_x50" # declining costs
load_factor = "load factor" # declining costs

