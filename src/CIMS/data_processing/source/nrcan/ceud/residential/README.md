# Residential Sector Notes

## CEUD

1. Cooling data for the territories missing in 2002 and 2014, filled in with previous years data
2. Manitoba vintage bucket for last historical period is labeled 2022 instead of 2021
3. Will need to split out territorial data into each territory when RESD data is ready

## Space-heating intensity (`residential_heating_intensity.py`)

Replaces the REM699 workbook (`AB_CIMS_Input_Res` rows 363-378) behind the Vintage
`Reference` → `Heating` `service_request` values in `raw_data/fixed_data/residential`.
Output: `processed_data/nrcan/ceud/residential_heating_intensity.csv` (GJ/m2).

1. Space-heating fuel by building type x system: stock (Tables 22-25) x provincial energy
   per unit (Table 8), balanced by IPF to Table 6 (type) and Table 8 (system) totals.
2. Heat = fuel x Table 26 efficiency; dual systems 80 % first-named fuel / 20 % second.
3. Vintage shape within each type from Table 33 (GOTR per m2), scaled to the type's heat.
4. CIMS bins aggregated floor-weighted; High = apartments, LowMed = SFD + SFA + MOB.
5. Back-fill leading years, 2021-2035 / >2035 = 0.75 / 0.75^2 x 2001-2020 (JCIMS),
   constant after the last CEUD year, TR applied to YT/NT/NU. No weather normalisation.
   BC Marine/Cold split is left to the sector module.

`Options` switches restore the workbook logic; `validation/check_workbook_port.py`
reproduces the workbook's rows 365-373 (2000-2020) exactly from its own inputs.
Note: the 2021_after (MB 2022_after) vintage's heat is not in any historical bin
(up to ~7 % of heat in 2023); those bins use the JCIMS ratios instead.
