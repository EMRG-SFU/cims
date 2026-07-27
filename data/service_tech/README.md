# Entry fields
context / sub_context / target: include if applicable
value_type: -- choose one
    - fixed (default if blank): value entered directly in yaml
    - csv: point to file in data/processed/; parameter, context, sub_context, target, by_year, and by_region will be used to index the value
    - calculation: point to marimo notebook where calculation process is documented, but enter calculated value directly in yaml
    - ref: point to proxy technology and parameter (e.g., "electricity_baseboard.service_request.target[".electricity"]"); ".value" is implied
    - formula: include reference parameters and formula for Python to resolve (e.g., "electricity_baseboard.service_request.target[".electricity"] / 2.5")
by_year:
    - true: use annual value-year pairs included in csv file
    - false (default if blank): use constant time-independent value
by_region:
    - true: use value-region pairs included in csv file
    - false (default if blank): use region-independent value
source: include source of data, assumption, or reason for use of proxy technology
note: include if applicable
currency: should be in the format YEAR_3-CHAR-DENOMINATION (e.g., "2015_USD"); fcc and fom need to be in this currency

# Notes
--> if no value is entered, the default parameter value will be used in the simulation
--> tech unit should match the service_provide unit of the parent branch node
--> output is in the same unit as the tech unit
--> repeat parameters service_request, emissions, emissions_biomass, emissions_removal as needed
--> for service_request note if units change from tech unit