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
competition_type = "competition_type" # general
edge_type = "type" # general
heterogeneity = 'heterogeneity' # general
is_supply = 'is_supply' # general
lifetime = "lifetime"   # general
service_provided = "provide_service" # general
service_requested = "request_service"
tree_index = 'tree_index' # general
unavailable = "unavailable" # general
no_inheritance = "no_inheritance" # general

# ==========================================
# Life Cycle Cost
# ==========================================
benchmark = "benchmark" # lcc
capital_cost = "capital_cost"   # lcc
capital_recovery = "capital_recovery" # lcc
competition_annual_cost = "competition_annual_cost" # lcc
competition_upfront_cost = "competition_upfront_cost" # lcc
cop = "cop" # lcc
cost_curve_function = "cost_curve_function" # lcc
cost_curve_lcc_max = "cost_curve_lcc_max" # lcc
cost_curve_lcc_min = "cost_curve_lcc_min" # lcc
cost_curve_price = 'cost_curve_price' # lcc
cost_curve_quantity = 'cost_curve_quantity' # lcc
crf = "crf" # lcc
discount_rate_financial = "discount_rate_financial" # lcc
discount_rate_retrofit ='discount_rate_retrofit' # lcc
financial_annual_cost = "financial_annual_cost" # lcc
financial_emissions_cost = "financial_emissions_cost" # lcc
financial_service_cost = 'financial_service_cost' # lcc
financial_upfront_cost = "financial_upfront_cost" # lcc
fixed_cost_rate = 'fixed_cost_rate' # lcc 
fom = "fom" # lcc
fcc = "fcc" # lcc
lcc_competition = "lcc_competition" # lcc
lcc_financial = "lcc_financial" # lcc
new_stock_financial_annual_cost = "new_stock_financial_annual_cost" # lcc
new_stock_financial_upfront_cost = "new_stock_financial_upfront_cost" # lcc
non_energy_cost = "non_energy_cost" # lcc
non_energy_cost_change = "non_energy_cost_change" # lcc
output = "output" # lcc
p2000 = "p2000" # lcc    
price = "price" # lcc
price_multiplier = "multiplier_price" # lcc
price_subsidy = "price_subsidy"  # lcc
recycled_revenues = "recycled_revenues" # lcc
revenue_recycle_rate = "revenue_recycle_rate"   # lcc
service_cost = "service_cost" # lcc
subsidy = "subsidy" # lcc
tax = "tax" # lcc
tax_foresight = "tax_foresight" # lcc
total_fixed_cost = 'total_fixed_cost' # lcc  
total_lcc_v = "total_lcc_v" # lcc

# ==========================================
# Stock Allocation
# ==========================================
added_retrofit_stock = "stock_retrofit_added"   # stock allocation
adjustment_multiplier = 'adjustment_multiplier' # stock_allocation
base_stock = "base_stock"   # stock allocation
base_stock_remaining = "base_stock_remaining" # stock allocation
fic = "fic" # stock allocation
domestic_elasticity = "domestic_elasticity" # macro-economics
export_benchmark = "export_benchmark" # macro-economics
export_elasticity = 'export_elasticity' # macro-economics
export_subsidy = "export_subsidy" # macro-economics
global_price = "global_price" # macro-economics
market_share = "market_share_existing"   # stock allocation
market_share_class = "market_share_class" # stock allocation / market share limits
market_share_class_max = "market_share_class_max" # stock allocation / market share limits
market_share_class_min = "market_share_class_min" # stock allocation / market share limits
market_share_new_max = "market_share_new_max" # stock allocation
market_share_new_min = "market_share_new_min" # stock allocation
new_market_share = "new_market_share" # stock allocation
new_stock = "stock_new" # stock allocation
new_stock_remaining = "new_stock_remaining" # stock allocation
new_stock_remaining_pre_surplus = 'new_stock_remaining_pre_surplus' # stock allocation
ref_stock_exported = 'ref_stock_exported' # stock allocation /macro-economics
retirement_intercept = "intercept_retirement" # stock_allocation
retrofit_existing_max = 'retrofit_existing_max' # stock_allocation
retrofit_existing_min = 'retrofit_existing_min' # stock_allocation
retrofit_heterogeneity = 'retrofit_heterogeneity' # stock_allocation
retrofit_new_max = "retrofit_new_max"    # stock allocation
retrofit_new_min = "retrofit_new_min"   # stock allocation
retrofit_stock = "stock_retrofit" # stock allocation
stock_demanded = "stock_demanded" # stock allocation / macro-economics
stock_exported = "stock_exported" # stock allocation / macro-economics
total_market_share = "total_market_share" # stock allocation
total_stock = "stock_total" # stock allocation

# ==========================================
# Aggregation
# ==========================================
aggregation_requested = "request_aggregation" # aggregation
aggregation_weight = "aggregation_weight" # aggregation
distributed_supply = "quantities_distributed_supply"   # aggregation
provided_quantities = "quantities_provided" # aggregation
requested_quantities = "quantities_requested" # aggregation
structural_aggregation = "structural_aggregation" # aggregation

# ==========================================
# Emissions
# ==========================================
aggregate_emissions_cost_rate = "aggregate_emissions_cost_rate" # emissions
avoided_emissions_rate = "emissions_avoided_rate"   # emissions
bio_emissions_rate = "emissions_bio_rate" # emissions
cumul_avoided_emissions_rate = "cumul_avoided_emissions_rate" # emissions
cumul_bio_emissions_rate = "cumul_bio_emissions_rate" # emissions
cumul_emissions_cost_rate = "cumul_emissions_cost_rate" # emissions
cumul_negative_emissions_rate = "cumul_negative_emissions_rate" # emissions
cumul_net_emissions_rate = "cumul_net_emissions_rate" # emissions
emissions = "emissions" # emissions
emissions_biomass = "emissions_biomass" # emissions
emissions_cost = "emissions_cost" # emissions
emissions_cost_rate = "emissions_cost_rate" # emissions
emissions_gwp = 'emissions_gwp'  # emissions     
emissions_removal = "emissions_removal" # emissions     
negative_emissions_rate = "emissions_negative_rate" # emissions
net_emissions_rate = "emissions_net_rate" # emissions
total_cumul_avoided_emissions = "emissions_avoided_total_cumul" # emissions
total_cumul_bio_emissions = "emissions_bio_total_cumul" # emissions
total_cumul_emissions_cost = "emissions_cost_total_cumul" # emissions
total_cumul_negative_emissions = "emissions_negative_total_cumul" # emissions
total_cumul_net_emissions = "emissions_net_total_cumul" # emissions
total_direct_avoided_emissions = "total_direct_avoided_emissions" # emissions
total_direct_bio_emissions = "total_direct_bio_emissions" # emissions
total_direct_emissions_cost = "total_direct_emissions_cost" # emissions
total_direct_negative_emissions = "total_direct_negative_emissions" # emissions
total_direct_net_emissions = "total_direct_net_emissions" # emissions

# ==========================================
# Declining Costs
# ==========================================
capital_cost_declining = "capital_cost_declining"  # declining cost
capital_cost_min = "capital_cost_min"   # declining cost
dcc_capacity_1 = "dcc_capacity_1"   # declining costs
dcc_capacity_2 = "dcc_capacity_2"   # declining costs
dcc_capacity_3 = "dcc_capacity_3"   # declining costs
dcc_class = "dcc_class" # declining costs
dcc_limit = "dcc_limit" # declining costs
dcc_min_learning = "dcc_min_learning" # declining costs
dcc_progress_ratio_1 = "dcc_progress_ratio_1" # declining costs
dcc_progress_ratio_2 = "dcc_progress_ratio_2" # declining costs
dcc_progress_ratio_3 = "dcc_progress_ratio_3" # declining costs
dic = "dic" # declining costs
dic_class = "dic_class" # declining costs
dic_initial = "dic_initial" # declining costs
dic_slope = "dic_slope" # declining costs
dic_min = "dic_min" # declining costs
dic_x50 = "dic_x50" # declining costs
load_factor = "multiplier_load_factor" # declining costs

