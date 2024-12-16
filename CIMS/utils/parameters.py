from .import model_columns as COL

# ------------ Data Structure Constants ------------ #
technologies = "technologies"
year_value = "year_value"
context = COL.context.lower()
sub_context = COL.sub_context.lower()
target= COL.target.lower()
source= "source"
unit= COL.unit.lower()
param_source= "param_source"

# ------------ Basic ------------ #
competition_type = "competition type"
is_supply = 'is supply'
available = "available"
unavailable = "unavailable"
heterogeneity = 'heterogeneity'
lifetime = "lifetime"
output = "output"

# ------------ Emissions ------------ #
emissions_gwp = 'emissions_gwp'
emissions_cost_rate = "emissions_cost_rate"
cumul_emissions_cost_rate = "cumul_emissions_cost_rate"
emissions = "emissions"
emissions_removal = "emissions_removal"
emissions_biomass = "emissions_biomass"
emissions_removal = "emissions_removal"
emissions_cost = "emissions cost"
net_emissions_rate = "net_emissions_rate"
avoided_emissions_rate = "avoided_emissions_rate"
negative_emissions_rate = "negative_emissions_rate"
bio_emissions_rate = "bio_emissions_rate"
cumul_emissions_cost_rate = "cumul_emissions_cost_rate"
cumul_net_emissions_rate = "cumul_net_emissions_rate"
cumul_bio_emissions_rate = "cumul_bio_emissions_rate"
cumul_avoided_emissions_rate = "cumul_avoided_emissions_rate"
cumul_negative_emissions_rate = "cumul_negative_emissions_rate"
aggregate_emissions_cost_rate = "aggregate_emissions_cost_rate"
total_direct_net_emissions = "total_direct_net_emissions"
total_direct_avoided_emissions = "total_direct_avoided_emissions"
total_direct_negative_emissions = "total_direct_negative_emissions"
total_direct_bio_emissions = "total_direct_bio_emissions"
total_direct_emissions_cost = "total_direct_emissions_cost"
total_cumul_net_emissions = "total_cumul_net_emissions"
total_cumul_avoided_emissions = "total_cumul_avoided_emissions"
total_cumul_negative_emissions = "total_cumul_negative_emissions"
total_cumul_bio_emissions = "total_cumul_bio_emissions"
total_cumul_emissions_cost = "total_cumul_emissions_cost"

# ------------ Declining Capital Cost ------------ #
dcc_class = "dcc_class"
dcc_capacity_1 = "dcc_capacity_1"
dcc_capacity_2 = "dcc_capacity_2"
dcc_capacity_3 = "dcc_capacity_3"
dcc_progress_ratio_1 = "dcc_progress ratio_1"
dcc_progress_ratio_2 = "dcc_progress ratio_2"
dcc_progress_ratio_3 = "dcc_progress ratio_3"
dcc_limit = "dcc_limit"
dcc_min_learning = "dcc_min learning"

# ------------ Declining Intangible Cost ------------ #
dic = "dic"
dic_class = "dic_class"
dic_initial = "dic_initial"
dic_slope = "dic_slope"
dic_x50 = "dic_x50"
dic_min = "dic_min"

# ------------ Stock ------------ #
base_stock = "base_stock"
new_stock = "new_stock"
total_stock = "total_stock"
new_stock_remaining = "new_stock_remaining"
base_stock_remaining = "base_stock_remaining"
added_retrofit_stock = "added_retrofit_stock"
discount_rate_retrofit ='discount rate_retrofit'
retrofit_existing_min = 'retrofit_existing_min'
retrofit_existing_max = 'retrofit_existing_max'
retrofit_heterogeneity = 'retrofit_heterogeneity'
new_stock_remaining_pre_surplus = 'new_stock_remaining_pre_surplus'

# ------------ Market Shares ------------ #
new_market_share = "new_market_share"
total_market_share = "total_market_share"
market_share = "market share"
market_share_new_min = "market share new_min"
market_share_new_max = "market share new_max"

# ------------ Financial Costs ------------ #
lcc_financial = "lcc_financial"
financial_emissions_cost = "financial emissions cost"
financial_upfront_cost = "financial upfront cost"
new_stock_financial_upfront_cost = "new_stock_financial_upfront_cost"
financial_annual_cost = "financial annual cost"
new_stock_financial_annual_cost = "new_stock_financial_annual_cost"
financial_service_cost = 'financial service cost'

# ------------ Competition Costs ------------ #
competition_upfront_cost = "competition upfront cost"
competition_annual_cost = "competition annual cost"
competition_annual_cost = "competition annual cost"
lcc_competition = "lcc_competition"

# ------------ Cost Curves ------------ #
cost_curve_function = "cost_curve_function"
cost_curve_quantity = 'cost curve quantity'
cost_curve_price = 'cost curve price'
cost_curve_lcc_min = "cost_curve_lcc_min"
cost_curve_lcc_max = "cost_curve_lcc_max"

# ------------ Taxes ------------ #
tax_foresight = "tax_foresight"
tax = "tax"

# ------------ Edge Parameters ------------ #
aggregation_weight = "aggregation_weight"
edge_type = "type"

# ------------ Node-Node Relationships ------------ #
provided_quantities = "provided_quantities"
service_requested = "service requested"
aggregation_requested = "aggregation requested"
structural_aggregation = "structural_aggregation"
requested_quantities = "requested_quantities"
# ------------ Other ------------ #
adjustment_multiplier = 'adjustment_multiplier' # stock_allocation
benchmark = "benchmark"
capital_cost = "capital cost"
capital_cost_declining = "capital cost_declining"
capital_cost_min = "capital_cost_min"
capital_recovery = "capital recovery"
cop = "cop"
crf = "crf"
discount_rate_financial = "discount rate_financial"
distributed_supply = "distributed_supply"
domestic_elasticity = "domestic elasticity" # macro-economics
export_elasticity = 'export elasticity' # macro-economics
export_subsidy = "export subsidy" # macro-economics
export_benchmark = "export benchmark" # macro-economics
fcc = "fcc"
fic = "fic"
fixed_cost_rate = 'fixed cost rate' 
fom = "fom"
global_price = "global price" # macro-economics
load_factor = "load factor"
market_share_class = "market share_class" # market share limits
market_share_class_min = "market share_class_min" # market share limits
market_share_class_max = "market share_class_max" # market share limits
non_energy_cost = "non-energy cost"
non_energy_cost_change = "non-energy cost change"
p2000 = "p2000"
price = "price"
price_multiplier = "price multiplier"
price_subsidy = "price_subsidy"
ref_stock_exported = 'ref stock exported' # macro-economics
revenue_recycle_rate = "revenue recycle rate"
recycled_revenues = "recycled revenues"
retirement_intercept = "retirement intercept" # stock_allocation
retrofit_new_min = "retrofit_new_min"
retrofit_new_max = "retrofit_new_max"
retrofit_stock = "retrofit_stock" # stock allocation
service_cost = "service cost"
service_provided = "service provided"
stock_exported = "stock exported" # macro-economics
stock_demanded = "stock demanded" # macro-economics
subsidy = "subsidy"
total_fixed_cost = 'total fixed cost'
total_lcc_v = "total_lcc_v"
tree_index = 'tree index'

