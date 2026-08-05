from ..model_description import column_list as COL
# ==========================================
# Data Structure Constants
# ==========================================
context = COL.context.lower()
param_source= "param_source"
sub_context = COL.sub_context.lower()
target= COL.target.lower()
technologies = "technologies"
unit= COL.unit.lower()
year_value = "year_value"

# ==========================================
# General
# ==========================================
available = "available"
base_year = "2000"
competition_type = "competition"
competition_compete = "tech compete"
edge_type = "edge"
heterogeneity = "heterogeneity"
inheritance = "inheritance"
is_supply = "is_supply"
lifetime = "lifetime"
service_request = "service_request"
tree_index = "tree_index"
unavailable = "unavailable"

# ==========================================
# Lifecycle Cost
# ==========================================
benchmark = "benchmark"
capital_cost = "capital_cost"
capital_recovery = "capital_recovery"
competition_cost_annual = "competition_cost_annual"
competition_cost_upfront = "competition_cost_upfront"
cop = "cop"
cost_curve_function = "cost_curve_function"
cost_curve_lcc_max = "cost_curve_lcc_max"
cost_curve_lcc_min = "cost_curve_lcc_min"
cost_curve_price = "cost_curve_price"
cost_curve_quantity = "cost_curve_quantity"
crf = "crf"
discount_rate_financial = "discount_rate_financial"
discount_rate_retrofit ="discount_rate_retrofit"
emissions_cost = "emissions_cost" # TODO: rename this to `competition_cost_emissions` and clarify throughout code
fcc = "fcc"
financial_cost_annual = "financial_cost_annual"
financial_cost_emissions = "financial_cost_emissions"
financial_cost_service = "financial_cost_service"
financial_cost_upfront = "financial_cost_upfront"
fixed_cost_rate = "fixed_cost_rate"
fixed_cost_total = "fixed_cost_total"
fom = "fom"
lcc_competition = "lcc_competition"
lcc_financial = "lcc_financial"
multiplier_price = "multiplier_price"
non_energy_cost = "non_energy_cost"
non_energy_cost_change = "non_energy_cost_change"
output = "output"
p2000 = "p2000"
price = "price"
price_subsidy = "price_subsidy" 
revenue_recycled = "revenue_recycled"
revenue_recycle_rate = "revenue_recycle_rate"
service_cost = "service_cost"
stock_new_financial_cost_annual = "stock_new_financial_cost_annual"
stock_new_financial_cost_upfront = "stock_new_financial_cost_upfront"
subsidy = "subsidy"
tax = "tax"
tax_foresight = "tax_foresight"
total_lcc_v = "total_lcc_v"

# ==========================================
# Stock Allocation
# ==========================================
domestic_elasticity = "domestic_elasticity"
export_benchmark = "export_benchmark"
export_elasticity = "export_elasticity"
export_subsidy = "export_subsidy"
fic = "fic"
global_price = "global_price"
intercept_retirement = "intercept_retirement"
market_share_class = "market_share_class"
market_share_class_max = "market_share_class_max"
market_share_class_min = "market_share_class_min"
market_share_new = "market_share_new"
market_share_new_max = "market_share_new_max"
market_share_new_min = "market_share_new_min"
market_share_total = "market_share_total"
retrofit_existing_max = "retrofit_existing_max"
retrofit_existing_min = "retrofit_existing_min"
retrofit_heterogeneity = "retrofit_heterogeneity"
retrofit_new_max = "retrofit_new_max" 
retrofit_new_min = "retrofit_new_min"
stock_base = "stock_base"
stock_base_remaining = "stock_base_remaining"
stock_demanded = "stock_demanded"
stock_exported = "stock_exported"
stock_new = "stock_new"
stock_new_remaining = "stock_new_remaining"
stock_new_remaining_pre_surplus = "stock_new_remaining_pre_surplus"
stock_ref_exported = "stock_ref_exported"
stock_retrofit = "stock_retrofit"
stock_retrofit_added = "stock_retrofit_added"
stock_surplus_adjustment = "stock_surplus_adjustment"
stock_total = "stock_total"

# ==========================================
# Aggregation
# ==========================================
aggregate_weight = "aggregate_weight"
aggregate_structural = "aggregate_structural" #TODO: should this be replaced by manual/auto `quantity_aggregate` use?
distributed_supply = "quantity_distributed"
provided_quantities = "quantity_provided"
requested_quantities = "quantity_requested"
quantity_aggregate = "quantity_aggregate"

# ==========================================
# Emissions
# ==========================================
emissions = "emissions"
emissions_biomass = "emissions_biomass"
emissions_gwp = "emissions_gwp" 
emissions_rate_cumul_avoided = "emissions_rate_cumul_avoided"
emissions_rate_cumul_bio = "emissions_rate_cumul_bio"
emissions_rate_cumul_cost = "emissions_rate_cumul_cost"
emissions_rate_cumul_negative = "emissions_rate_cumul_negative"
emissions_rate_cumul_net = "emissions_rate_cumul_net"
emissions_rate_direct_avoided = "emissions_rate_direct_avoided"
emissions_rate_direct_bio = "emissions_rate_direct_bio"
emissions_rate_direct_cost = "emissions_rate_direct_cost"
emissions_rate_direct_negative = "emissions_rate_direct_negative"
emissions_rate_direct_net = "emissions_rate_direct_net"
emissions_removal = "emissions_removal"
emissions_total_cumul_avoided = "emissions_total_cumul_avoided"
emissions_total_cumul_bio = "emissions_total_cumul_bio"
emissions_total_cumul_cost = "emissions_total_cumul_cost"
emissions_total_cumul_negative = "emissions_total_cumul_negative"
emissions_total_cumul_net = "emissions_total_cumul_net"
emissions_total_direct_avoided = "emissions_total_direct_avoided"
emissions_total_direct_bio = "emissions_total_direct_bio"
emissions_total_direct_cost = "emissions_total_direct_cost"
emissions_total_direct_negative = "emissions_total_direct_negative"
emissions_total_direct_net = "emissions_total_direct_net"

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
dcc_learning_min = "dcc_learning_min"
dcc_progress_ratio_1 = "dcc_progress_ratio_1"
dcc_progress_ratio_2 = "dcc_progress_ratio_2"
dcc_progress_ratio_3 = "dcc_progress_ratio_3"
dic = "dic"
dic_class = "dic_class"
dic_initial = "dic_initial"
dic_slope = "dic_slope"
dic_min = "dic_min"
dic_x50 = "dic_x50"
multiplier_load_factor = "multiplier_load_factor"

