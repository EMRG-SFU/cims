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
available = "available"
competition_type = "competition_type"
edge_type = "type"
heterogeneity = 'heterogeneity'
is_supply = 'is_supply'
lifetime = "lifetime"  
service_provided = "provide_service"
service_requested = "request_service"
tree_index = 'tree_index'
unavailable = "unavailable"
no_inheritance = "no_inheritance"

# ==========================================
# Life Cycle Cost
# ==========================================
benchmark = "benchmark"
capital_cost = "capital_cost"  
capital_recovery = "capital_recovery"
competition_annual_cost = "competition_annual_cost"
competition_upfront_cost = "competition_upfront_cost"
cop = "cop"
cost_curve_function = "cost_curve_function"
cost_curve_lcc_max = "cost_curve_lcc_max"
cost_curve_lcc_min = "cost_curve_lcc_min"
cost_curve_price = 'cost_curve_price'
cost_curve_quantity = 'cost_curve_quantity'
crf = "crf"
discount_rate_financial = "discount_rate_financial"
discount_rate_retrofit ='discount_rate_retrofit'
financial_annual_cost = "financial_annual_cost"
financial_emissions_cost = "financial_emissions_cost"
financial_service_cost = 'financial_service_cost'
financial_upfront_cost = "financial_upfront_cost"
fixed_cost_rate = 'fixed_cost_rate'
fom = "fom"
fcc = "fcc"
lcc_competition = "lcc_competition"
lcc_financial = "lcc_financial"
new_stock_financial_annual_cost = "new_stock_financial_annual_cost"
new_stock_financial_upfront_cost = "new_stock_financial_upfront_cost"
non_energy_cost = "non_energy_cost"
non_energy_cost_change = "non_energy_cost_change"
output = "output"
p2000 = "p2000"
price = "price"
price_multiplier = "multiplier_price"
price_subsidy = "price_subsidy" 
recycled_revenues = "recycled_revenues"
revenue_recycle_rate = "revenue_recycle_rate"  
service_cost = "service_cost"
subsidy = "subsidy"
tax = "tax"
tax_foresight = "tax_foresight"
total_fixed_cost = 'total_fixed_cost'
total_lcc_v = "total_lcc_v"

# ==========================================
# Stock Allocation
# ==========================================
added_retrofit_stock = "stock_retrofit_added"  
adjustment_multiplier = 'adjustment_multiplier'
base_stock = "base_stock"  
base_stock_remaining = "base_stock_remaining"
fic = "fic"
domestic_elasticity = "domestic_elasticity"
export_benchmark = "export_benchmark"
export_elasticity = 'export_elasticity'
export_subsidy = "export_subsidy"
global_price = "global_price"
market_share = "market_share_existing"  
market_share_class = "market_share_class"
market_share_class_max = "market_share_class_max"
market_share_class_min = "market_share_class_min"
market_share_new_max = "market_share_new_max"
market_share_new_min = "market_share_new_min"
new_market_share = "new_market_share"
new_stock = "stock_new"
new_stock_remaining = "new_stock_remaining"
new_stock_remaining_pre_surplus = 'new_stock_remaining_pre_surplus'
ref_stock_exported = 'ref_stock_exported'
retirement_intercept = "intercept_retirement"
retrofit_existing_max = 'retrofit_existing_max'
retrofit_existing_min = 'retrofit_existing_min'
retrofit_heterogeneity = 'retrofit_heterogeneity'
retrofit_new_max = "retrofit_new_max"   
retrofit_new_min = "retrofit_new_min"  
retrofit_stock = "stock_retrofit"
stock_demanded = "stock_demanded"
stock_exported = "stock_exported"
total_market_share = "total_market_share"
total_stock = "stock_total"

# ==========================================
# Aggregation
# ==========================================
aggregation_requested = "request_aggregation"
aggregation_weight = "aggregation_weight"
distributed_supply = "quantities_distributed_supply"  
provided_quantities = "quantities_provided"
requested_quantities = "quantities_requested"
structural_aggregation = "structural_aggregation"

# ==========================================
# Emissions
# ==========================================
aggregate_emissions_cost_rate = "aggregate_emissions_cost_rate"
avoided_emissions_rate = "emissions_avoided_rate"  
bio_emissions_rate = "emissions_bio_rate"
cumul_avoided_emissions_rate = "cumul_avoided_emissions_rate"
cumul_bio_emissions_rate = "cumul_bio_emissions_rate"
cumul_emissions_cost_rate = "cumul_emissions_cost_rate"
cumul_negative_emissions_rate = "cumul_negative_emissions_rate"
cumul_net_emissions_rate = "cumul_net_emissions_rate"
emissions = "emissions"
emissions_biomass = "emissions_biomass"
emissions_cost = "emissions_cost"
emissions_cost_rate = "emissions_cost_rate"
emissions_gwp = 'emissions_gwp' 
emissions_removal = "emissions_removal"
negative_emissions_rate = "emissions_negative_rate"
net_emissions_rate = "emissions_net_rate"
total_cumul_avoided_emissions = "emissions_avoided_total_cumul"
total_cumul_bio_emissions = "emissions_bio_total_cumul"
total_cumul_emissions_cost = "emissions_cost_total_cumul"
total_cumul_negative_emissions = "emissions_negative_total_cumul"
total_cumul_net_emissions = "emissions_net_total_cumul"
total_direct_avoided_emissions = "total_direct_avoided_emissions"
total_direct_bio_emissions = "total_direct_bio_emissions"
total_direct_emissions_cost = "total_direct_emissions_cost"
total_direct_negative_emissions = "total_direct_negative_emissions"
total_direct_net_emissions = "total_direct_net_emissions"

# ==========================================
# Declining Costs
# ==========================================
capital_cost_declining = "capital_cost_declining" 
capital_cost_min = "capital_cost_min"  
dcc_capacity_1 = "dcc_capacity_1"  
dcc_capacity_2 = "dcc_capacity_2"  
dcc_capacity_3 = "dcc_capacity_3"  
dcc_class = "dcc_class"
dcc_limit = "dcc_limit"
dcc_min_learning = "dcc_min_learning"
dcc_progress_ratio_1 = "dcc_progress_ratio_1"
dcc_progress_ratio_2 = "dcc_progress_ratio_2"
dcc_progress_ratio_3 = "dcc_progress_ratio_3"
dic = "dic"
dic_class = "dic_class"
dic_initial = "dic_initial"
dic_slope = "dic_slope"
dic_min = "dic_min"
dic_x50 = "dic_x50"
load_factor = "multiplier_load_factor"

