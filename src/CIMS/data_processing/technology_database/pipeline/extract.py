"""Layer 1 -> layer 2: read every extraction and return its facts, as reported.

Each source's own technology name is kept exactly as the extraction wrote it
(`source-technology`). Nothing is renamed, grouped or canonicalized here: which
archetype a source's technology is an example of is decided in one place,
records/source-example.csv, and applied by examples.py.

An EXAMPLE is one technology as one source presents it, in one of that source's
own contexts (its process or table band). Its id is

    <source-id>:<source-technology>[@<process>]

where source-id is the document the extraction was made from (UKERC for the
UKERC tables, even where a value is attributed to an original UKERC cites).

Returns facts, flows and constraints in the long schema, plus per-example
attributes (description, readiness, state of development) that sources report
as parameters but that describe the example rather than quantify it.
"""
from __future__ import annotations

import re
from collections import defaultdict

from .common import RAW, REFERENCE, currency_year, read

# UK TIMES (via Bataille) prices its "improvement technologies" and add-ons as
# the cost of adding them to an existing plant: cost-basis incremental.
UKTM_ADD_ON = re.compile(r"^(Improvement technology \d+ for existing|[^.]*\badd-on for existing)")

# --- the extractions, in load order (record ids are numbered in this order) --
TEMPLATE_FILES = [
    ("gov-usa/epa/epa_2007_biomass-chp-technologies-catalog.csv", "epa_2007_001"),
    ("gov-usa/epa/epa_2017_chp-technologies-catalog.csv", "epa_2017_001"),
    ("gov-usa/lbnl/lbnl_2021_electrification-of-boilers.csv", "lbnl_2021_001"),
    ("iea/etsap/etsap_2010_industrial-combustion-boilers.csv", "etsap_2010_001"),
]
UKERC = ("other-reports/ukerc_2013_usable-energy-database", "ukerc_2013_001")
BATAILLE = ("other-reports/bataille_2017_heavy-industry-database", "bataille_2017_001")
IPCC = ("peer-reviewed/ipcc_2022_ar6-wgiii-chapter11", "ipcc_2022_001")
IEA_GUIDE = ("iea/iea_2024_clean-tech-guide", "iea_2024_001")
ETSAP_BRIEFS = [
    ("iea/etsap/etsap_2010_iron-and-steel", "etsap_2010_002"),
    ("iea/etsap/etsap_2010_cement-production", "etsap_2010_003"),
    ("iea/etsap/etsap_2015_pulp-and-paper", "etsap_2015_001"),
    ("iea/etsap/etsap_2012_aluminium-production", "etsap_2012_001"),
    ("iea/etsap/etsap_2013_heat-pumps", "etsap_2013_002"),
    # extracted 2026-09-23/24 from the ETSAP brief assessment
    ("iea/etsap/etsap_2010_combined-heat-and-power", "etsap_2010_004"),
    ("iea/etsap/etsap_2010_biomass-for-heat-and-power", "etsap_2010_005"),
    ("iea/etsap/etsap_2013_stationary-fuel-cells", "etsap_2013_004"),
    ("iea/etsap/etsap_2010_syngas-from-coal", "etsap_2010_011"),
    ("iea/etsap/etsap_2014_coal-mining-and-logistics", "etsap_2014_002"),
    ("iea/etsap/etsap_2013_biogas-and-bio-syngas", "etsap_2013_012"),
    # the remaining partial briefs, extracted 2026-09-24
    ("iea/etsap/etsap_2010_geothermal-heat-and-power", "etsap_2010_006"),
    ("iea/etsap/etsap_2010_gas-fired-power", "etsap_2010_008"),
    ("iea/etsap/etsap_2013_concentrating-solar-power", "etsap_2013_001"),
    ("iea/etsap/etsap_2013_thermal-energy-storage", "etsap_2013_005"),
    ("iea/etsap/etsap_2010_co2-capture-and-storage", "etsap_2010_007"),
    ("iea/etsap/etsap_2014_hydrogen-production-and-distribution", "etsap_2014_003"),
    ("iea/etsap/etsap_2013_bio-methanol", "etsap_2013_007"),
    ("iea/etsap/etsap_2010_liquid-fuels-from-coal-and-gas", "etsap_2010_012"),
    ("iea/etsap/etsap_2010_unconventional-oil-and-gas", "etsap_2010_010"),
    ("iea/etsap/etsap_2011_oil-and-gas-logistics", "etsap_2011_001"),
    ("iea/etsap/etsap_2014_oil-refineries", "etsap_2014_001"),
    ("iea/etsap/etsap_2013_renewable-desalination", "etsap_2013_008"),
    ("iea/etsap/etsap_2013_bio-ethylene", "etsap_2013_009"),
    ("iea/etsap/etsap_2013_biomass-production-and-logistics", "etsap_2013_010"),
    ("iea/etsap/etsap_2013_liquid-biofuels", "etsap_2013_011"),
]
# other extractions in the ETSAP-brief shape (process/technology rows, _nodes, _flows),
# loaded by the same reader: (stem, source id, region of the data)
BRIEF_SHAPE = [
    ("gov-usa/doe/bandwidth-studies/doe_2015_iron-and-steel-bandwidth-study", "doe_2015_001", "US"),
    ("gov-usa/doe/bandwidth-studies/doe_2015_petroleum-refining-bandwidth-study", "doe_2015_002", "US"),
    ("gov-usa/doe/bandwidth-studies/doe_2015_pulp-and-paper-bandwidth-study", "doe_2015_003", "US"),
    ("gov-usa/doe/bandwidth-studies/doe_2015_chemicals-bandwidth-study", "doe_2015_004", "US"),
    ("gov-usa/doe/bandwidth-studies/doe_2017_aluminum-bandwidth-study", "doe_2017_001", "US"),
    ("gov-usa/doe/bandwidth-studies/doe_2017_cement-bandwidth-study", "doe_2017_002", "US"),
    ("gov-usa/doe/bandwidth-studies/doe_2017_food-and-beverage-bandwidth-study", "doe_2017_003", "US"),
    ("gov-usa/doe/doe_2022_industrial-decarbonization-roadmap", "doe_2022_001", "US"),
    ("gov-usa/epa/epa_2022_waste-heat-to-power", "epa_2022_001", "US"),
    ("gov-usa/netl/netl_2022_industrial-co2-capture-retrofit-database", "netl_2022_001", "US"),
    ("iea/iea_2018_future-of-petrochemicals", "iea_2018_001", "global"),
    ("iea/iea_2019_future-of-hydrogen", "iea_2019_002", "global"),
    ("iea/iea_2021_ammonia-technology-roadmap", "iea_2021_001", "global"),
    ("iea/philibert_2017_renewable-energy-for-industry_iea", "philibert_2017_001", "global"),
    ("other-reports/lehne-preston_2018_making-concrete-change_chatham-house", "lehne-preston_2018_001", "global"),
    ("other-reports/caelp-e3_2024_decarbonizing-industrial-heat", "caelp_2024_001", "US"),
    ("other-reports/rtc_2023_renewable-thermal-vision", "rtc_2023_001", "US"),
    ("other-reports/van-der-meijden_2011_bio-methane-from-wood_ecn", "van-der-meijden_2011_001", "global"),
    ("peer-reviewed/amin_2011_methane-catalytic-cracking-review_ijhe", "amin_2011_001", "global"),
    ("peer-reviewed/armijo-philibert_2020_green-hydrogen-ammonia-chile-argentina_ijhe", "armijo-philibert_2020_001", "global"),
    ("peer-reviewed/fasihi_2019_direct-air-capture-techno-economics_jclepro", "fasihi_2019_001", "global"),
    ("peer-reviewed/habert_2020_cement-concrete-decarbonization_nat-rev-earth-environ", "habert_2020_001", "global"),
    ("peer-reviewed/leeson_2017_industrial-ccs-techno-economics_ijggc", "leeson_2017_001", "global"),
    ("peer-reviewed/moore_2017_thermal-hydrogen_ijhe", "moore_2017_001", "global"),
    ("peer-reviewed/roemer_2026_mining-energy-decarbonization-review_energies", "roemer_2026_001", "global"),
    ("peer-reviewed/vogl_2018_hydrogen-direct-reduction-steelmaking_jclepro", "vogl_2018_001", "global"),
    ("gov-dk/dea/dea_2026_technology-data-industrial-process-heat", "dea_2026_001", "Denmark"),
    ("peer-reviewed/humbert_2024_electrowinning-iron-economics_jsm", "humbert_2024_001", "global"),
    ("peer-reviewed/siyambalapitiya_2026_induction-hydrogen-billet-reheating_jsm", "siyambalapitiya_2026_001", "Australia"),
    ("peer-reviewed/svensson_2021_electric-plasma-lime-calcination_fclim", "svensson_2021_001", "Sweden"),
    ("peer-reviewed/kuparinen_2017_lime-kiln-renewable-fuels_bioresources", "kuparinen_2017_001", "South America"),
    ("peer-reviewed/kiss_2012_energy-efficient-distillation_energy", "kiss_2012_001", "global"),
    ("peer-reviewed/smith_2020_haber-bosch-carbon-free-ammonia_ees", "smith_2020_001", "global"),
    ("peer-reviewed/layritz_2021_steam-cracker-electrification-ocm_apenergy", "layritz_2021_001", "EU"),
    ("gov-eu/jrc/brinkmann_2014_chlor-alkali-bref_jrc", "brinkmann_2014_001", "EU"),
    ("gov-eu/jrc/jrc_2007_lvic-s-bref", "jrc_2007_001", "EU"),
    ("gov-usa/epa/epa_2010_nitric-acid-ghg-technologies", "epa_2010_001", "US"),
    ("other-reports/rissman_2022_low-temperature-industrial-heat_energy-innovation", "rissman_2022_001", "US"),
    ("gov-nl/pbl/van-geene_2024_midden-database-v0-4a_pbl", "van-geene_2024_001", "Netherlands"),
    ("peer-reviewed/uekert_2023_closed-loop-plastics-recycling_acs-sce", "uekert_2023_001", "US"),
    ("peer-reviewed/onarheim_2017_ccs-pulp-paper-part1-performance_ijggc", "onarheim_2017_001", "Finland"),
    ("peer-reviewed/onarheim_2017_ccs-pulp-paper-part2-economics_ijggc", "onarheim_2017_002", "Finland"),
    ("gov-dk/dea/dea_2024_technology-data-carbon-capture-transport-storage", "dea_2024_002", "Denmark"),
    ("gov-dk/dea/dea_2026_technology-data-renewable-fuels", "dea_2026_003", "Denmark"),
    ("gov-dk/dea/dea_2026_technology-data-energy-storage", "dea_2026_004", "Denmark"),
    ("other-reports/mpp_2022_steel-capex-opex-per-technology", "mpp_2022_001", "global"),
    ("other-reports/rissman_2023_industrial-thermal-batteries_energy-innovation", "rissman_2023_001", "US"),
    ("peer-reviewed/zuberi_2022_heat-pumps-us-food-manufacturing_ecm", "zuberi_2022_001", "US"),
    ("peer-reviewed/zuberi_2023_heat-pumps-us-textile-paper-automotive_energy-efficiency", "zuberi_2023_001", "US"),
    ("peer-reviewed/kong_2016_emerging-energy-efficiency-pulp-paper_jclepro", "kong_2016_001", "global"),
    ("peer-reviewed/fleiter_2012_german-pulp-paper-saving-potentials_energy", "fleiter_2012_001", "Germany"),
    ("peer-reviewed/lipiainen_2022_pulp-paper-energy-transition-finland-sweden_spc", "lipiainen_2022_001", "Finland and Sweden"),
    ("peer-reviewed/hasanbeigi_2014_alternative-emerging-ironmaking_rser", "hasanbeigi_2014_001", "global"),
    ("gov-usa/lbnl/hasanbeigi_2013_emerging-steel-technologies_lbnl", "hasanbeigi_2013_001", "global"),
    ("peer-reviewed/ren_2006_olefins-steam-cracking-alternatives_energy", "ren_2006_001", "global"),
    ("gov-usa/lbnl/neelis_2008_petrochemical-energy-guide_lbnl", "neelis_2008_001", "US"),
    ("other-reports/ieaghg_2018_co2-capture-cement-iron-steel_tr03", "ieaghg_2018_001", "global"),
    ("gov-usa/netl/hughes_2023_cost-co2-capture-industrial-sources_netl", "hughes_2023_001", "US"),
    ("gov-nl/pbl/van-geene_2025_nobian-salt-chlor-alkali-update_pbl", "van-geene_2025_001", "Netherlands"),
    ("gov-nl/pbl/negri_2021_polyolefins-decarbonisation_pbl", "negri_2021_001", "Netherlands"),
    ("gov-nl/pbl/haque_2023_energy-efficiency-options_pbl", "haque_2023_001", "Netherlands"),
    ("gov-usa/lbnl/worrell_2010_energy-efficiency-iron-steel_lbnl", "worrell_2010_001", "US"),
    ("gov-usa/epa/epa_2010_pulp-paper-ghg-technologies", "epa_2010_002", "US"),
    ("gov-usa/epa/epa_2012_iron-steel-ghg-technologies", "epa_2012_001", "US"),
    ("iea/iea_2020_iron-steel-technology-roadmap_iea", "iea_2020_002", "global"),
    ("other-reports/larson_2006_gasification-biorefining-kraft-pulp-paper_princeton", "larson_2006_001", "US"),
    ("gov-usa/lbnl/khrushch_1999_chp-carbon-reduction-chemicals-pulp-paper_lbnl", "khrushch_1999_001", "US"),
    ("other-reports/app_2010_soact-steelmaking-handbook_lbnl", "app_2010_001", "global"),
    ("gov-usa/doe/doe_2023_liftoff-industrial-decarbonization", "doe_2023_001", "US"),
    ("gov-usa/doe/doe_2025_liftoff-chemicals-refining", "doe_2025_001", "US"),
    ("gov-eu/jrc/boulamanti_2017_chemicals-prospective-scenarios_jrc", "boulamanti_2017_001", "EU"),
    ("gov-usa/lbnl/kramer_2009_energy-efficiency-pulp-paper_lbnl", "kramer_2009_001", "US"),
    ("gov-eu/jrc/falcke_2017_lvoc-bref_jrc", "falcke_2017_001", "EU"),
    ("gov-eu/jrc/jrc_2022_ferrous-metals-processing-bref", "jrc_2022_001", "EU"),
    ("gov-eu/jrc/remus_2013_iron-steel-bref_jrc", "remus_2013_001", "EU"),
    ("gov-eu/jrc/suhr_2015_pulp-paper-bref_jrc", "suhr_2015_001", "EU"),
    ("gov-eu/jrc/jrc_2007_lvic-aaf-bref", "jrc_2007_002", "EU"),
    ("gov-eu/jrc/jrc_2007_polymers-bref", "jrc_2007_003", "EU"),
    ("gov-usa/doe/doe_2023_liftoff-low-carbon-cement", "doe_2023_002", "US"),
    ("gov-eu/jrc/schorcht_2013_cement-lime-magnesia-bref_jrc", "schorcht_2013_001", "EU"),
    ("other-reports/csi-ecra_2017_cement-technology-papers", "csi-ecra_2017_001", "global"),
    ("iea/iea_2018_cement-technology-roadmap_iea", "iea_2018_002", "global"),
    ("peer-reviewed/voldsund_2019_cement-co2-capture-comparison_energies", "voldsund_2019_001", "EU"),
    ("gov-usa/epa/worrell_2013_energy-efficiency-cement_energystar", "worrell_2013_001", "US"),
    ("gov-usa/lbnl/kermeli_2011_energy-efficiency-concrete_lbnl", "kermeli_2011_001", "US"),
    ("gov-eu/jrc/scalet_2013_glass-bref_jrc", "scalet_2013_001", "EU"),
    ("gov-usa/lbnl/worrell_2008_energy-efficiency-glass_lbnl", "worrell_2008_001", "US"),
    ("gov-eu/jrc/jrc_2007_ceramics-bref", "jrc_2007_004", "EU"),
    ("gov-eu/jrc/barthe_2015_refining-bref_jrc", "barthe_2015_001", "EU"),
    ("gov-usa/epa/worrell_2015_energy-efficiency-petroleum-refineries_energystar", "worrell_2015_001", "US"),
    ("gov-usa/epa/kermeli_2016_energy-efficiency-metal-casting_energystar", "kermeli_2016_001", "US"),
    ("gov-eu/jrc/aries_2024_smitheries-foundries-bref_jrc", "aries_2024_001", "EU"),
    ("gov-eu/jrc/cusano_2017_non-ferrous-metals-bref_jrc", "cusano_2017_001", "EU"),
    ("gov-eu/jrc/lecomte_2017_large-combustion-plants-bref_jrc", "lecomte_2017_001", "EU"),
    ("gov-eu/jrc/giner-santonja_2019_food-drink-milk-bref_jrc", "giner-santonja_2019_001", "EU"),
    ("gov-usa/lbnl/masanet_2012_energy-efficiency-baking_lbnl", "masanet_2012_001", "US"),
    ("gov-usa/epa/stuckrath_2022_energy-efficiency-distilleries_energystar", "stuckrath_2022_001", "US"),
    ("gov-usa/lbnl/galitsky_2003_energy-efficiency-breweries_lbnl", "galitsky_2003_001", "US"),
    ("gov-usa/lbnl/galitsky_2003_energy-efficiency-corn-wet-milling_lbnl", "galitsky_2003_002", "US"),
    ("gov-usa/lbnl/galitsky_2008_energy-efficiency-pharmaceutical_lbnl", "galitsky_2008_001", "US"),
    ("gov-eu/jrc/brinkmann_2016_common-waste-water-gas-bref_jrc", "brinkmann_2016_001", "EU"),
    ("gov-eu/jrc/daginnus_2023_common-waste-gas-chemical-bref_jrc", "daginnus_2023_001", "EU"),
    ("gov-eu/jrc/stubdrup_2016_wood-based-panels-bref_jrc", "stubdrup_2016_001", "EU"),
    ("gov-eu/jrc/neuwahl_2019_waste-incineration-bref_jrc", "neuwahl_2019_001", "EU"),
    ("gov-eu/jrc/pinasseau_2018_waste-treatment-bref_jrc", "pinasseau_2018_001", "EU"),
    ("gov-dk/dea/dea_2026_technology-data-electricity-district-heat", "dea_2026_006", "Denmark"),
    ("gov-dk/dea/dea_2026_technology-data-heating-installations", "dea_2026_007", "Denmark"),
    ("gov-dk/dea/dea_2026_technology-review-non-road-machinery", "dea_2026_005", "Denmark"),
    ("gov-nl/pbl/joshi_2022_midden-onshore-gas-oil_pbl", "joshi_2022_001", "Netherlands"),
    ("other-reports/ager_2023_hthp-paper-drying_kth", "ager_2023_001", "Sweden"),
    ("other-reports/agora_2021_breakthrough-strategies-climate-neutral-industry", "agora_2021_001", "EU"),
    ("peer-reviewed/allanore_2015_molten-oxide-electrolytes_jes", "allanore_2015_001", "global"),
    ("peer-reviewed/arens_2017_low-carbon-iron-steel-germany_jclepro", "arens_2017_001", "Germany"),
    ("peer-reviewed/bhaskar_2020_hydrogen-direct-reduction_energies", "bhaskar_2020_001", "EU"),
    ("other-reports/bhp_2023_electric-smelting-furnace", "bhp_2023_001", "Australia"),
    ("peer-reviewed/cancio-diaz_2017_limestone-calcined-clay-cement_deveng", "cancio-diaz_2017_001", "Cuba"),
    ("peer-reviewed/ciambellotti_2024_hthp-tissue-paper_energies", "ciambellotti_2024_001", "EU"),
    ("peer-reviewed/devlin_2023_green-hydrogen-steel-opportunities_ncomms", "devlin_2023_001", "global"),
    ("other-reports/draxler_2021_green-steel-technology-assessment_gsfe", "draxler_2021_001", "EU"),
    ("gov-usa/epa/natural-gas-star/epa_2006_lessons-learned_natural-gas-star", "epa_2006_001", "US"),
    ("gov-usa/epa/natural-gas-star/epa_2011_pro-fact-sheets_natural-gas-star", "epa_2011_001", "US"),
    ("gov-usa/epa/epa_2013_chp-boiler-replacement-fact-sheet", "epa_2013_001", "US"),
    ("other-reports/fpinnovations_2026_kraft-pulp-decarbonization-canada", "fpinnovations_2026_001", "Canada"),
    ("gov-can/nrcan/francis_2002_energy-benchmarking-pulp-paper_nrcan", "francis_2002_001", "Canada"),
    ("other-reports/golder_2021_canadian-steel-energy-ghg-roadmap_cspa", "golder_2021_001", "Canada"),
    ("peer-reviewed/hamaguchi_2012_biofuels-kraft-pulp-mills_energies", "hamaguchi_2012_001", "global"),
    ("iea/iea_2013_chemical-catalytic-processes-roadmap_iea", "iea_2013_001", "global"),
    ("other-reports/ieaghg_2013_co2-capture-integrated-steel-mill", "ieaghg_2013_003", "EU"),
    ("other-reports/ieaghg_2016_ccs-retrofit-pulp-board-mills", "ieaghg_2016_001", "EU"),
    ("other-reports/kearns_2021_ccs-technology-readiness-costs_gccsi", "kearns_2021_001", "US"),
    ("peer-reviewed/kim_2022_iron-steel-decarbonization-review_erss", "kim_2022_001", "global"),
    ("other-reports/kiss_2017_heat-pumps-chemical-process-industry_crc", "kiss_2017_001", "EU"),
    ("peer-reviewed/kuparinen_2019_biomass-ccu-kraft-pulp_mitig", "kuparinen_2019_001", "global"),
    ("other-reports/material-economics_2019_industrial-transformation-2050", "material-economics_2019_001", "EU"),
    ("peer-reviewed/moya_2013_eu-iron-steel-efficiency-payback_jclepro", "moya_2013_001", "EU"),
    ("peer-reviewed/mynko_2023_steam-cracking-electrification_jclepro", "mynko_2023_001", "EU"),
    ("gov-can/nrcan/nrcan_2008_benchmarking-canadian-pulp-paper-mills", "nrcan_2008_001", "Canada"),
    ("peer-reviewed/otto_2017_power-to-steel_energies", "otto_2017_001", "Germany"),
    ("peer-reviewed/owttrim_2022_pulp-paper-energy-saving-bandwidths_ecm", "owttrim_2022_001", "Canada"),
    ("peer-reviewed/pardo_2013_eu-iron-steel-prospective-scenarios_energy", "pardo_2013_001", "EU"),
    ("peer-reviewed/perez-ramirez_2003_n2o-nitric-acid_apcatb", "perez-ramirez_2003_001", "global"),
    ("other-reports/provides_2018_deep-eutectic-solvents-paper", "provides_2018_001", "EU"),
    ("peer-reviewed/rechberger_2020_hydrogen-direct-reduction_steel-research", "rechberger_2020_001", "EU"),
    ("peer-reviewed/riedewald_2021_mixed-plastic-pyrolysis-molten-metal_wm", "riedewald_2021_001", "Belgium"),
    ("peer-reviewed/rouwenhorst_2019_islanded-ammonia-power_rser", "rouwenhorst_2019_001", "global"),
    ("peer-reviewed/sabah_2026_electric-smelting-furnace-modelling_jsm", "sabah_2026_001", "global"),
    ("peer-reviewed/sanchez-berriel_2016_limestone-calcined-clay-cement-cuba_jclepro", "sanchez-berriel_2016_001", "Cuba"),
    ("peer-reviewed/saygin_2011_chemical-best-practice-technology_energy", "saygin_2011_001", "global"),
    ("peer-reviewed/schneider_2015_cement-process-technology_ccr", "schneider_2015_001", "global"),
    ("peer-reviewed/singh_2021_enzymatic-pet-recycling_joule", "singh_2021_001", "US"),
    ("peer-reviewed/solis_2020_chemical-recycling-household-plastics_wm", "solis_2020_001", "global"),
    ("other-reports/spees_2023_thermal-batteries_brattle", "spees_2023_001", "US"),
    ("peer-reviewed/talaei_2018_chemical-sector-efficiency-ghg_energy", "talaei_2018_001", "Canada"),
    ("peer-reviewed/talaei_2020_canada-iron-steel-efficiency_energy-efficiency", "talaei_2020_001", "Canada"),
    ("peer-reviewed/toktarova_2020_swedish-steel-pathways_energies", "toktarova_2020_001", "Sweden"),
    ("other-reports/vakkilainen_2006_kraft-recovery-boilers-high-dry-solids", "vakkilainen_2006_001", "global"),
    ("peer-reviewed/arpagaus_2018_high-temperature-heat-pumps-review_energy", "arpagaus_2018_001", "EU"),
    ("other-reports/deloitte_2022_alumina-refining-roadmap_arena", "deloitte_2022_001", "Australia"),
    ("gov-usa/eia/eia_2024_capital-cost-performance-utility-scale-generating-technologies", "eia_2024_001", "US"),
    ("other-reports/element-energy_2018_industrial-fuel-switching", "element-energy_2018_001", "UK"),
    ("other-reports/element-energy_2020_deep-decarbonisation-pathways-uk-industry", "element-energy_2020_001", "UK"),
    ("peer-reviewed/gardarsdottir_2019_cement-co2-capture-cost-analysis_energies", "gardarsdottir_2019_001", "EU"),
    ("peer-reviewed/haas_2006_biodiesel-production-cost-model_biortech", "haas_2006_001", "US"),
    ("gov-usa/nrel/humbird_2011_cellulosic-ethanol-dilute-acid-design_nrel", "humbird_2011_001", "US"),
    ("other-reports/iea-hpt_2023_annex-58-task-1-high-temperature-heat-pump-technologies_task-report", "iea-hpt_2023_001", "global"),
    ("iea/iea_2025_global-hydrogen-review_iea", "iea_2025_001", "global"),
    ("other-reports/ieaghg_2017_smr-merchant-hydrogen-ccs", "ieaghg_2017_001", "Netherlands"),
    ("other-reports/ieaghg_2017_refinery-capture-retrofit-recap", "ieaghg_2017_002", "Netherlands"),
    ("other-reports/irena_2020_green-hydrogen-cost-reduction", "irena_2020_001", "global"),
    ("gov-usa/nrel/jadun_2017_end-use-electric-technology-projections_nrel", "jadun_2017_001", "US"),
    ("gov-usa/pnnl/jones_2013_fast-pyrolysis-hydrotreating-design_pnnl", "jones_2013_001", "US"),
    ("gov-eu/jrc/jrc_2009_energy-efficiency-bref", "jrc_2009_001", "EU"),
    ("peer-reviewed/kwiatkowski_2006_corn-dry-grind-ethanol-cost-model_indcrop", "kwiatkowski_2006_001", "US"),
    ("peer-reviewed/madeddu_2020_power-to-heat-european-industry_erl", "madeddu_2020_001", "EU"),
    ("peer-reviewed/mantei_2021_oxymethylene-ether-co2-h2-tea_sef", "mantei_2021_001", "Germany"),
    ("gov-usa/lbnl/martin_2000_emerging-energy-efficient-industrial-technologies_lbnl", "martin_2000_001", "US"),
    ("gov-usa/nrel/mcmillan_2021_solar-industrial-process-heat-us_nrel", "mcmillan_2021_001", "US"),
    ("other-reports/mpp_2023_net-zero-aluminium-strategy", "mpp_2023_001", "global"),
    ("gov-usa/netl/netl_2022_fossil-hydrogen-production-comparison", "netl_2022_002", "US"),
    ("gov-usa/nrel/nrel_2025_annual-technology-baseline-electricity", "nrel_2025_001", "US"),
    ("peer-reviewed/rehfeldt_2018_bottom-up-heating-cooling-demand-european-industry_energy-efficiency", "rehfeldt_2018_001", "EU"),
    ("other-reports/rehfeldt_2024_direct-electrification-industrial-process-heat_agora", "rehfeldt_2024_001", "EU"),
    ("peer-reviewed/sanchez_2018_ccs-us-biorefineries_pnas", "sanchez_2018_001", "US"),
    ("other-reports/saw_2023_cst-bayer-alumina_adelaide", "saw_2023_001", "Australia"),
    ("peer-reviewed/schemme_2020_h2-based-synthetic-fuels_ijhe", "schemme_2020_001", "Germany"),
    ("peer-reviewed/schoeneberger_2022_electrification-us-industrial-boilers_appenergy", "schoeneberger_2022_001", "US"),
    ("peer-reviewed/simoni_2022_decarbonising-lime-industry_rser", "simoni_2022_001", "EU"),
    ("peer-reviewed/tao_2017_ethanol-to-jet-blendstocks-tea_greenchem", "tao_2017_001", "US"),
    ("other-reports/vattenfall_2018_cemzero-electrified-cement", "vattenfall_2018_001", "Sweden"),
    ("peer-reviewed/west_2008_four-biodiesel-processes-hysys_biortech", "west_2008_001", "Canada"),
    ("other-reports/wsp_2015_uk-roadmap-cement", "wsp_2015_001", "UK"),
    ("other-reports/wsp_2015_uk-roadmap-ceramic", "wsp_2015_002", "UK"),
    ("other-reports/wsp_2015_uk-roadmap-glass", "wsp_2015_003", "UK"),
    ("other-reports/wsp_2015_uk-roadmap-chemicals", "wsp_2015_004", "UK"),
    ("other-reports/wsp_2015_uk-roadmap-petroleum-refining", "wsp_2015_005", "UK"),
    ("other-reports/wsp_2015_uk-roadmap-food-and-beverage", "wsp_2015_006", "UK"),
    ("other-reports/wsp_2015_uk-roadmap-iron-steel", "wsp_2015_007", "UK"),
    ("other-reports/wsp_2015_uk-roadmap-pulp-paper", "wsp_2015_008", "UK"),
    ("peer-reviewed/zier_2021_decarbonization-options-glass-industry_ecmx", "zier_2021_001", "EU"),
    ("gov-usa/inl/abou-jaoude_2024_advanced-reactor-cost-meta-analysis", "abou-jaoude_2024_001", "US"),
    ("peer-reviewed/ahluwalia_2023_fuel-cell-hybrid-mining-truck_energies", "ahluwalia_2023_001", "global"),
    ("other-reports/alcoa_2024_mvr-alumina-refining-close-out_arena", "alcoa_2024_001", "Australia"),
    ("peer-reviewed/almeida-benalcazar_2022_gas-fermentation-ethanol-decarbonization_compchemeng", "almeida-benalcazar_2022_001", "EU"),
    ("gov-usa/anl/anl_2025_hdsam-v5-5", "anl_2025_001", "US"),
    ("peer-reviewed/ballantyne_2014_comminution-benchmarking-copper-gold_mineng", "ballantyne_2014_001", "Australia"),
    ("gov-can/bc-hydro/bc-hydro_2004_integrated-electricity-plan_app-d", "bc-hydro_2004_001", "Canada (BC)"),
    ("gov-usa/doe/bcs_2007_mining-energy-bandwidth-study", "bcs_2007_001", "US"),
    ("other-reports/blair_2000_alberta-polystyrene-production-options", "blair_2000_001", "Canada (Alberta)"),
    ("other-reports/brandt_2022_opgee-v3-0b-methodology_stanford", "brandt_2022_001", "global"),
    ("other-reports/brown_2020_advanced-biofuels-potential-for-cost-reduction_ieabioenergy", "brown_2020_001", "global"),
    ("other-reports/ceri_2017_oil-sands-supply-costs-development-projects_study-163", "ceri_2017_001", "Canada"),
    ("other-reports/ceri_2017_oil-sands-processes-technologies-efficiencies_study-164", "ceri_2017_002", "Canada"),
    ("other-reports/concawe_2019_co2-reduction-technologies-eu-refining_concawe", "concawe_2019_001", "EU"),
    ("peer-reviewed/coursol_2015_energy-consumption-copper-smelting_jom", "coursol_2015_001", "global"),
    ("peer-reviewed/de-beer_1998_future-technologies-energy-efficient-iron-steel_aree", "de-beer_1998_001", "global"),
    ("gov-usa/doe/doe_2000_energy-environmental-profile-us-iron-steel", "doe_2000_001", "US"),
    ("gov-usa/doe/doe_2002_energy-environmental-profile-us-mining_gold-silver", "doe_2002_001", "US"),
    ("gov-usa/doe/doe_2024_liftoff-clean-hydrogen-update", "doe_2024_001", "US"),
    ("gov-can/eccc/eccc_2025_upstream-oil-gas-methane-amendments-rias_canada-gazette", "eccc_2025_001", "Canada"),
    ("other-reports/efma_2000_bat-ammonium-nitrate-can-production", "efma_2000_001", "EU"),
    ("gov-usa/eia/eia_1996_aeo-1997-assumptions", "eia_1996_001", "US"),
    ("peer-reviewed/ellis_2020_electrochemical-synthesis-cement_pnas", "ellis_2020_001", "US"),
    ("gov-dk/ens/ens_2015_kortlaegning-energisparepotentialer-erhvervslivet", "ens_2015_001", "Denmark"),
    ("gov-dk/ens/ens_2015_kortlaegning-energiforbrug-virksomheder", "ens_2015_002", "Denmark"),
    ("gov-usa/epa/epa_1999_us-methane-emissions-1990-2020", "epa_1999_001", "US"),
    ("gov-usa/epa/epa_2001_pfc-emission-reduction-cost-aluminum-smelters", "epa_2001_001", "US"),
    ("gov-usa/epa/epa_2019_non-co2-mac-methodology", "epa_2019_001", "global"),
    ("gov-usa/epa/epa_2023_oil-gas-nsps-eg-climate-review-ria", "epa_2023_001", "US"),
    ("other-reports/esmap_2001_clean-coal-technologies-china-non-power_vol-2", "esmap_2001_001", "China"),
    ("peer-reviewed/geleynse_2018_alcohol-to-jet-pathway-tea_chemsuschem", "geleynse_2018_001", "US"),
    ("other-reports/gielen_1997_technology-characterisation-ceramic-inorganic-markal_ecn", "gielen_1997_001", "Western Europe"),
    ("peer-reviewed/gielen_2002_co2-iron-steel-japan-reduction-potentials_enpol", "gielen_2002_001", "Japan"),
    ("other-reports/hedman_2008_waste-energy-recovery-interstate-gas-pipelines_ingaa", "hedman_2008_001", "US"),
    ("peer-reviewed/hills_2017_leilac-low-cost-capture-cement-lime_energyprocedia", "hills_2017_001", "EU"),
    ("peer-reviewed/hornby-anderson_2002_ais-chemistry-eaf-steelmaking-economics", "hornby-anderson_2002_001", "US"),
    ("other-reports/icf_2015_methane-emission-reduction-opportunities-canada_edf", "icf_2015_001", "Canada"),
    ("iea/iea_2026_electrification-special-report_iea", "iea_2026_006", "global"),
    ("gov-usa/lbnl/interlab_1997_scenarios-us-carbon-reductions_ch04", "interlab_1997_001", "US"),
    ("gov-usa/doe/jacobs_2006_energy-bandwidth-study-pulp-paper_doe", "jacobs_2006_001", "US"),
    ("gov-eu/jrc/carlsson_2014_etri-energy-technology-reference-indicators_jrc", "jrc_2014_001", "EU"),
    ("other-reports/kcb_2017_study-tailings-management-technologies_mend", "kcb_2017_001", "Canada"),
    ("other-reports/keesom_2018_bitumen-partial-upgrading-whitepaper_ai", "keesom_2018_001", "Canada"),
    ("peer-reviewed/keith_2018_process-capturing-co2-atmosphere_joule", "keith_2018_001", "North America"),
    ("gov-usa/lbnl/kranz_2001_shortened-depreciation-chp-investment-costs_lbnl", "kranz_2001_001", "US"),
    ("peer-reviewed/kukkonen_2017_bev-energy-consumption-underground_acg", "kukkonen_2017_001", "Finland"),
    ("other-reports/lazard_2021_levelized-cost-of-hydrogen-v2", "lazard_2021_001", "US"),
    ("other-reports/lazard_2023_levelized-cost-of-energy-plus", "lazard_2023_001", "US"),
    ("other-reports/lazard_2024_levelized-cost-of-energy-plus", "lazard_2024_001", "US"),
    ("other-reports/leblanc_2014_trolley-assist-open-pit_bba", "leblanc_2014_001", "Canada"),
    ("peer-reviewed/lemar_2001_policies-promote-chp-us-industry_enpol", "lemar_2001_001", "US"),
    ("other-reports/mccann_2000_alberta-propylene-upgrading-prospects", "mccann_2000_001", "Canada (Alberta)"),
    ("other-reports/mueller_2013_corn-ethanol-emerging-plant-technologies_uic", "mueller_2013_001", "US"),
    ("gov-can/nccp/nccp_1999_aluminum-industry-options-paper", "nccp_1999_001", "Canada"),
    ("gov-can/nccp/nccp_1999_magnesium-industry-options-paper", "nccp_1999_002", "Canada"),
    ("gov-can/nccp/nccp_1999_minerals-metals-foundation-paper", "nccp_1999_003", "Canada"),
    ("gov-can/nccp/nccp_1999_minerals-metals-ghg-mitigation-opportunities", "nccp_1999_004", "Canada"),
    ("gov-can/nccp/nccp_1999_upstream-oil-gas-options-paper", "nccp_1999_005", "Canada"),
    ("gov-can/nccp/nccp_2000_iron-steel-plant-level-analysis-options", "nccp_2000_001", "Canada"),
    ("gov-usa/inl/nelson_2011_htgr-integration-industrial-processes", "nelson_2011_001", "US"),
    ("peer-reviewed/norgate_2010_mining-mineral-processing-energy-ghg_jclepro", "norgate_2010_001", "Australia"),
    ("gov-can/nrcan/nrcan_2001_energy-benchmark-cement-clinker_cipec", "nrcan_2001_001", "Canada"),
    ("gov-can/nrcan/nrcan_2005_underground-bulk-mines-energy-benchmark", "nrcan_2005_001", "Canada"),
    ("other-reports/pca_2002_us-canadian-labor-energy-input-survey-2000", "pca_2002_001", "US and Canada"),
    ("other-reports/pca_2003_canadian-portland-cement-tables-2001", "pca_2003_001", "Canada"),
    ("other-reports/pembina_2000_ghg-reduction-upstream-oil-gas-pipelines-bc", "pembina_2000_001", "Canada (BC)"),
    ("peer-reviewed/pfeifer_2002_thermodynamic-analysis-eaf-energy-efficiency", "pfeifer_2002_001", "Germany"),
    ("peer-reviewed/plaza_2020_ccus-cement-industry-review_energies", "plaza_2020_001", "global"),
    ("gov-usa/lbnl/price_2000_china-industrial-sector-international-context", "price_2000_001", "China"),
    ("peer-reviewed/raj_2016_techno-economic-lng-western-canada_seta", "raj_2016_001", "Canada"),
    ("other-reports/rightor_2020_beneficial-electrification-industry_aceee", "rightor_2020_001", "US"),
    ("other-reports/riotinto_2022_hydrogen-calcination-feasibility_arena", "riotinto_2022_001", "Australia"),
    ("peer-reviewed/rubini_2022_robust-regenerative-turbo-reactor_gpps", "rubini_2022_001", "global"),
    ("peer-reviewed/schlosser_2020_large-scale-heat-pumps-review_rser", "schlosser_2020_001", "EU"),
    ("gov-usa/lbnl/sezgen_1995_commercial-space-conditioning-technology-data_lbnl", "sezgen_1995_001", "US"),
    ("gov-usa/lbnl/sezgen_1995_water-heating-commercial-buildings-technology-data_lbnl", "sezgen_1995_002", "US"),
    ("other-reports/snclavalin_2000_zinc-plant-modernization-flin-flon", "snclavalin_2000_001", "Canada"),
    ("other-reports/songhurst_2018_lng-plant-cost-reduction-2014-18_oies", "songhurst_2018_001", "global"),
    ("peer-reviewed/thomson_2000_ghg-control-eaf-steelmaking", "thomson_2000_001", "Canada and UK"),
    ("peer-reviewed/tokac_2025_diesel-vs-electric-trucks-open-pit_jclepro", "tokac_2025_001", "Canada"),
    ("gov-usa/nrel/topolski_2022_hydrogen-blending-natural-gas-pipelines_nrel", "topolski_2022_001", "US"),
    ("other-reports/wade_2003_guide-decentralized-energy-technologies", "wade_2003_001", "global"),
    ("other-reports/wallace_2021_hydrogen-fuel-cell-mine-haul-trucks_ucalgary", "wallace_2021_001", "Canada"),
    ("peer-reviewed/wang_2013_hpgr-circuits-huckleberry-sabc_cmp", "wang_2013_001", "Canada"),
    ("gov-usa/nrel/warren_2021_lithium-extraction-geothermal-brines_nrel", "warren_2021_001", "US"),
    ("other-reports/world-bank_1996_china-efficient-industrial-boilers", "world-bank_1996_001", "China"),
    ("gov-usa/lbnl/worrell_2000_energy-use-intensity-us-chemical-industry_lbnl", "worrell_2000_001", "US"),
    ("peer-reviewed/zhang_2003_biodiesel-waste-cooking-oil-economics_biortech", "zhang_2003_001", "Canada"),
    ("peer-reviewed/zier_2023_german-glass-industry-decarbonization_ecmx", "zier_2023_001", "Germany"),
    ("peer-reviewed/zuhlsdorf_2019_heat-pump-process-heat-above-150c_ecmx", "zuhlsdorf_2019_001", "EU"),
    ("other-reports/aranda_2014_large-scale-biosng-economy_ecn", "aranda_2014_001", "global"),
    ("peer-reviewed/carbo_2011_biosng-beccs-avoidance-cost_energyprocedia", "carbo_2011_001", "Netherlands"),
    ("gov-can/eccc/eccc_2022_clean-fuel-regulations-rias_canada-gazette", "eccc_2022_001", "Canada"),
    ("peer-reviewed/habert_2010_cement-technology-factor-4_cemconres", "habert_2010_001", "France"),
    ("other-reports/hallbar_2017_bc-rng-resource-supply-potential", "hallbar_2017_001", "Canada (BC)"),
    ("peer-reviewed/handler_2016_gas-fermentation-ethanol-lca_iecr", "handler_2016_001", "US"),
    ("other-reports/ieaghg_2007_co2-capture-medium-scale-combustion", "ieaghg_2007_001", "Europe"),
    ("other-reports/irlam_2017_global-costs-ccs_gccsi", "irlam_2017_001", "US (with Canada column)"),
    ("peer-reviewed/kaiser_2017_refinery-complexity-review_petsci", "kaiser_2017_001", "US"),
    ("other-reports/larsen_2019_direct-air-capture-policy_rhodium", "larsen_2019_001", "US"),
    ("other-reports/liberty_2020_frac-fleets-tier4-diesel-dual-fuel-electric", "liberty_2020_001", "US"),
    ("other-reports/minea_2012_lumber-drying-heat-pumps_hpc-newsletter", "minea_2012_001", "Canada (Quebec)"),
    ("peer-reviewed/mishra_2026_iron-ore-pelletization-review_mmtb", "mishra_2026_001", "global"),
    ("peer-reviewed/robben_2019_sensor-based-ore-sorting_minerals", "robben_2019_001", "global"),
    ("other-reports/rtc_2022_electric-heat-pumps-brief", "rtc_2022_001", "US"),
    ("peer-reviewed/scrivener_2018_calcined-clay-limestone-cement-lc3_cemconres", "scrivener_2018_001", "Cuba and India"),
    ("peer-reviewed/smith_2001_air-separation-review_fuproc", "smith_2001_001", "US"),
    ("peer-reviewed/surup_2020_charcoal-reductant-ferroalloy-review_processes", "surup_2020_001", "Norway, Brazil, EU"),
    ("peer-reviewed/tomani_2010_lignoboost-process_cellchemtechnol", "tomani_2010_001", "Sweden"),
    ("peer-reviewed/toro_2021_solvent-bitumen-extraction-eseieh-tea_ualberta", "toro_2021_001", "Canada (Alberta)"),
    ("peer-reviewed/vera_2023_dle-environmental-impact-review_nree", "vera_2023_001", "Argentina and Chile"),
    ("peer-reviewed/wiencke_2018_molten-oxide-electrolysis-iron_jae", "wiencke_2018_001", "France"),
    ("peer-reviewed/kildahl_2023_bf-bof-thermochemical-sector-coupling_jcp", "kildahl_2023_001", "UK"),
    ("peer-reviewed/hu_2009_co2-corex-vs-blast-furnace_jes", "hu_2009_001", "China"),
    ("other-reports/bataille_2021_global-facility-level-net-zero-steel-pathways_iddri", "bataille_2021_001", "global"),
    ("gov-can/nrcan_2007_benchmarking-energy-intensity-canadian-steel-industry_cipec", "nrcan_2007_001", "Canada (international benchmarks)"),
    ("other-reports/fertilizer-canada_2023_ammonia-production-ghg-benchmarking", "fertilizer-canada_2023_001", "Canada"),
    ("gov-usa/doe/anl_2017_venting-flaring-ocs_bsee", "anl_2017_001", "US (Gulf of Mexico offshore)"),
    ("other-reports/dunsky_2023_canada-methane-abatement-opportunity_edf", "dunsky_2023_001", "Canada"),
    ("peer-reviewed/zhang_2020_natural-gas-liquefaction-processes-review_ate", "zhang_2020_001", "global"),
    ("peer-reviewed/pospisil_2019_lng-liquefaction-regasification-energy-demand_rser", "pospisil_2019_001", "global"),
    ("other-reports/clearstone_2018_equipment-component-fugitive-emission-factors-alberta_aer", "clearstone_2018_001", "Canada (Alberta)"),
    ("other-reports/shell_2024_quest-annual-summary-report-2023", "shell_2024_001", "Canada (Alberta)"),
    ("gov-can/intergroup_2017_natural-gas-fuel-mine-haul-trucks_nrcan", "intergroup_2017_001", "Canada"),
    ("gov-usa/doe/doe_2006_steam-tip-sheet-20-backpressure-turbogenerators", "doe_2006_001", "US"),
    ("gov-usa/doe/doe_2012_steam-tip-sheet-22-high-pressure-boilers-backpressure-turbine", "doe_2012_002", "US"),
    ("gov-usa/doe/doe_2020_program-record-20004-cost-electrolytic-hydrogen", "doe_2020_001", "US"),
    ("other-reports/gccsi_2021_blue-hydrogen", "gccsi_2021_001", "global"),
    ("other-reports/pembina_2021_carbon-intensity-blue-hydrogen", "pembina_2021_001", "Canada (Alberta and BC)"),
    ("gov-usa/nrel/nrel_2013_hydrogen-pathways-updated-cost-wtw", "nrel_2013_001", "US"),
    ("gov-usa/nrel/nrel_2009_levelized-cost-centralized-distributed-hydrogen-h2a-v2", "nrel_2009_001", "US"),
    ("peer-reviewed/worrell_2022_bottom-up-deep-decarbonization-us-manufacturing_jclp", "worrell_2022_001", "US"),
    ("other-reports/ceri_2018_oil-sands-supply-costs-2018-2038", "ceri_2018_001", "Canada (Alberta)"),
    # regulator facility statistics (intake 2026-10-21; graded sourced, sizes = median of existing facilities)
    ("gov-can/petrinex/petrinex_2026_upstream-facility-statistics_ab-sk", "petrinex_2026_001", "Canada (Alberta and Saskatchewan)"),
    ("gov-can/aer/st39/aer_2026_st39-oil-sands-mine-statistics", "aer_2026_001", "Canada (Alberta)"),
    ("gov-can/aer/st50/aer_2026_st50a-gas-plant-licensed-capacity", "aer_2026_003", "Canada (Alberta)"),
    ("gov-can/aer/st59/aer_2026_alberta-drilling-statistics", "aer_2026_005", "Canada (Alberta)"),
    ("gov-can/aer/st60b/aer_2025_st60b-pneumatic-venting", "aer_2025_001", "Canada (Alberta)"),
    ("gov-can/bcer/bcer_2026_gas-plant-and-well-statistics", "bcer_2026_001", "Canada (British Columbia)"),
    # AER Directive 054 in-situ performance presentations (intake 2026-10-16)
    ("gov-can/aer/in-situ-performance-presentations/canadian-natural_2024_kirby-se-sagd-pilot-performance-presentation_aer", "canadian-natural_2024_001", "Canada (Alberta)"),
    ("gov-can/aer/in-situ-performance-presentations/canadian-natural_2025_jackfish-in-situ-performance-presentation_aer", "canadian-natural_2025_001", "Canada (Alberta)"),
    ("gov-can/aer/in-situ-performance-presentations/canadian-natural_2025_kirby-in-situ-performance-presentation_aer", "canadian-natural_2025_002", "Canada (Alberta)"),
    ("gov-can/aer/in-situ-performance-presentations/canadian-natural_2025_primrose-wolf-lake-in-situ-performance-presentation_aer", "canadian-natural_2025_003", "Canada (Alberta)"),
    ("gov-can/aer/in-situ-performance-presentations/canadian-natural_2025_peace-river-in-situ-performance-presentation_aer", "canadian-natural_2025_004", "Canada (Alberta)"),
    ("gov-can/aer/in-situ-performance-presentations/cenovus_2025_christina-lake-in-situ-performance-presentation_aer", "cenovus_2025_001", "Canada (Alberta)"),
    ("gov-can/aer/in-situ-performance-presentations/cenovus_2025_foster-creek-in-situ-performance-presentation_aer", "cenovus_2025_002", "Canada (Alberta)"),
    ("gov-can/aer/in-situ-performance-presentations/conocophillips_2025_surmont-in-situ-performance-presentation_aer", "conocophillips_2025_001", "Canada (Alberta)"),
    ("gov-can/aer/in-situ-performance-presentations/imperial_2025_cold-lake-in-situ-performance-presentation_aer", "imperial_2025_001", "Canada (Alberta)"),
    ("gov-can/aer/in-situ-performance-presentations/meg_2025_christina-lake-in-situ-performance-presentation_aer", "meg_2025_001", "Canada (Alberta)"),
    ("gov-can/aer/in-situ-performance-presentations/suncor_2025_firebag-in-situ-performance-presentation_aer", "suncor_2025_001", "Canada (Alberta)"),
    ("gov-can/aer/in-situ-performance-presentations/suncor_2025_mackay-river-in-situ-performance-presentation_aer", "suncor_2025_002", "Canada (Alberta)"),
    ("gov-nl/gielen_1997_building-materials-co2", "gielen_1997_002", "Western Europe"),
    ("other-reports/joosten_1998_process-data-synthetic-organic-materials", "joosten_1998_001", "Western Europe"),
    ("other-reports/hekkert_1997_technology-characterisation-natural-organic-materials", "hekkert_1997_001", "Western Europe"),
    ("gov-nl/gielen_1997_basic-metal-industry-energy-use", "gielen_1997_003", "Netherlands"),
    ("gov-nl/gielen_1996_petrochemical-industry-energy-use", "gielen_1996_001", "Netherlands"),
    ("gov-usa/doe/doe_2001_onsite-power-opportunities-industrial-sector", "doe_2001_001", "US"),
    ("other-reports/ekbom_2003_black-liquor-gasification-motor-fuels", "ekbom_2003_001", "Sweden"),
    ("gov-nl/hugill_2001_eco-efficiency-methanol-routes", "hugill_2001_001", "Netherlands"),
    ("gov-eu/jrc/lindley_1997_chlor-alkali-mercury-cells-alternatives", "lindley_1997_001", "EU"),
    ("gov-usa/doe/doe_2000_chemical-industry-profile-ch5-fertilizers", "doe_2000_002", "US"),
    ("gov-usa/doe/doe_2000_chemical-industry-profile-ch6-chlor-alkali", "doe_2000_002", "US"),
    ("gov-can/nccp_1999_industrial-minerals-options-paper", "nccp_1999_006", "Canada"),
    ("gov-usa/lbnl/martin_1999_energy-efficiency-co2-reduction-us-cement_lbnl", "martin_1999_001", "US"),
    ("peer-reviewed/daniels_2002_transition-paths-co2-steel_thesis", "daniels_2002_001", "Western Europe"),
    ("gov-usa/lbnl/worrell_1999_energy-efficiency-co2-reduction-us-iron-steel_lbnl", "worrell_1999_001", "US"),
    ("other-reports/morrey_2002_new-smelter-capacity-options-costs", "morrey_2002_001", "global"),
    ("gov-can/neill-gunter_1999_cogeneration-opportunities-pulp-paper", "neill-gunter_1999_001", "Canada"),
    ("other-reports/clearstone_2004_uog-inventory-vol5-emission-factors", "clearstone_2004_001", "Canada"),
    ("gov-usa/doe/doe_1998_energy-environmental-profile-petroleum-refining", "doe_1998_001", "US"),
    ("gov-usa/lbnl/worrell_2004_profile-petroleum-refining-california_lbnl", "worrell_2004_002", "US (California)"),
    ("other-reports/purvin-gertz_2004_sulphur-removal-canadian-gasoline-distillate", "purvin-gertz_2004_001", "Canada"),
    ("gov-can/nccp_1999_cement-concrete-options-paper", "nccp_1999_007", "Canada"),
    ("gov-usa/nrel/h2a/nrel_2018_h2a-production-models-v3", "nrel_2018_003", "US"),
    ("other-reports/concawe_2019_refinery-2050-conceptual-assessment_concawe", "concawe_2019_002", "EU"),
    ("other-reports/ieaghg_2008_co2-capture-cement-industry", "ieaghg_2008_001", "UK (Europe); Asia sensitivity"),
    ("other-reports/mckane-hasanbeigi_2010_motor-systems-efficiency-supply-curves_unido", "mckane-hasanbeigi_2010_001", "US, Canada, EU (MEDIUM base case); Thailand, Vietnam, Brazil in _measures"),
    ("gov-can/nrcan/nrcan_2005_open-pit-mines-energy-benchmark", "nrcan_2005_002", "Canada"),
    ("gov-usa/nrel/nrel_2026_h2a-lite", "nrel_2026_001", "US"),
    ("other-reports/ren_2009_petrochemicals-energy-use-economics-innovation_thesis", "ren_2009_001", "global"),
    ("other-reports/element-energy_2021_n-zip-model-assumptions-log_ccc", "element-energy_2021_001", "UK"),
    ("iea/etsap/etsap_2012_space-heating-and-cooling", "etsap_2012_002", "Europe"),
    ("iea/etsap/etsap_2012_lighting", "etsap_2012_003", "OECD"),
    ("gov-usa/epa/epa_2010_wastewater-energy-conservation", "epa_2010_003", "US"),
    ("gov-usa/ornl/ornl_2004_guide-to-chp-boiler", "ornl_2004_001", "US"),
    ("gov-usa/nrel/nrel_2016_thermal-energy-use-in-us-industry", "nrel_2016_001", "US"),
    ("peer-reviewed/gary_2007_petroleum-refining-technology-economics_crc", "gary_2007_001", "US (Gulf Coast)"),
    ("peer-reviewed/abella_2012_prelim-refinery-energy-ghg_est", "abella_2012_001", "Canada and US (Canadian crudes)"),
    ("other-reports/gpsa_2004_engineering-data-book", "gpsa_2004_001", "US"),
    ("peer-reviewed/kidnay_2019_fundamentals-natural-gas-processing_crc", "kidnay_2019_001", "US"),
    ("peer-reviewed/schlesinger_2011_extractive-metallurgy-copper_elsevier", "schlesinger_2011_001", "global (incl. Canada)"),
    ("peer-reviewed/crundwell_2011_extractive-metallurgy-nickel-cobalt-pgm_elsevier", "crundwell_2011_001", "global (incl. Canada)"),
    ("gov-usa/epa/epa_2023_oil-gas-nsps-eg-final-tsd", "epa_2023_002", "US"),
    ("gov-usa/epa/epa_2022_oil-gas-nsps-eg-supplemental-tsd", "epa_2022_002", "US"),
    ("gov-usa/epa/epa_2021_oil-gas-nsps-eg-proposal-tsd", "epa_2021_001", "US"),
    ("other-reports/carbon-limits_2016_zero-emission-pneumatic-controllers-usa", "carbon-limits_2016_001", "US"),
    ("other-reports/carbon-limits_2021_zero-emission-pneumatic-controllers-usa-update", "carbon-limits_2021_001", "US"),
    ("iea/iea_2026_methane-tracker-abatement", "iea_2026_004", "Canada and US"),
    ("other-reports/ipieca/ipieca_2025_energy-efficiency-compendium", "ipieca_2025_001", "global"),
    ("gov-usa/doe/mccoy_2014_premium-efficiency-motor-selection-guide_doe", "mccoy_2014_001", "US"),
    ("gov-usa/doe/doe_2014_motor-drive-system-performance-sourcebook", "doe_2014_002", "US"),
    ("gov-usa/doe/doe_2012_tip-sheet-premium-efficiency-motors", "doe_2012_003", "US"),
    ("gov-usa/doe/doe_2012_tip-sheet-asd-part-load-efficiency", "doe_2012_004", "US"),
    ("gov-usa/doe/xenergy_1998_us-industrial-motor-systems-market-assessment", "xenergy_1998_001", "US"),
    ("gov-usa/lbnl/rao_2021_motor-system-market-assessment-vol1-installed-base_lbnl", "rao_2021_001", "US"),
    ("gov-usa/lbnl/rao_2022_motor-system-market-assessment-vol3-energy-saving_lbnl", "rao_2022_001", "US"),
    ("iea/iea_2009_cement-technology-roadmap_iea", "iea_2009_001", "global"),
    ("other-reports/stork_2014_lime-industry-roadmap_ecofys", "stork_2014_001", "EU"),
    ("peer-reviewed/habert_2020_cement-concrete-decarbonization-vor_nat-rev-earth-environ", "habert_2020_001", "global"),
    ("peer-reviewed/johansson_2012_heat-supply-co2-capture-process-industry_ijggc", "johansson_2012_001", "Sweden"),
    ("other-reports/mccollum_2006_co2-compression-transport-storage-cost-models_ucdavis", "mccollum_2006_001", "US"),
    ("peer-reviewed/emanuelsson_2022_ccs-cost-product-chain-cement-pulp_ghgt16", "emanuelsson_2022_001", "EU (Sweden)"),
    ("peer-reviewed/burns_2024_brevik-ccs-lessons-learned_ghgt17", "burns_2024_001", "Norway"),
    ("other-reports/ct_2010_iee-accelerator-industrial-bakery", "ct_2010_002", "UK"),
    ("other-reports/ct_2011_iee-accelerator-paper-sector", "ct_2011_001", "UK"),
    ("gov-usa/lbnl/martin_2000_pulp-paper-energy-efficiency-opportunities_lbnl", "martin_2000_002", "US"),
    ("other-reports/carbon-trust_2003_steam-generation-costs_ecg066", "carbon-trust_2003_001", "UK"),
    ("other-reports/alcoa_2022_mvr-evaporation-feasibility-study_arena", "alcoa_2022_001", "Australia"),
    ("other-reports/alcoa_2022_mvr-retrofit-commercialisation_arena", "alcoa_2022_002", "Australia"),
    ("other-reports/iai/iai_2026_aluminium-statistics", "iai_2026_001", "North America, World, Canada"),
    ("peer-reviewed/paraszczak_2014_electrification-loaders-trucks-underground_repqj", "paraszczak_2014_001", "Canada and Sweden"),
    ("peer-reviewed/drebenstedt_2014_mine-planning-equipment-selection-mpes2013_springer", "drebenstedt_2014_001", "Canada, Brazil, Italy"),
    ("peer-reviewed/sanchez_2018_ccs-us-biorefineries-si_pnas", "sanchez_2018_001", "US"),
    ("peer-reviewed/smith_2020_haber-bosch-carbon-free-ammonia-si_ees", "smith_2020_001", "global"),
    ("peer-reviewed/vogl_2018_hydrogen-direct-reduction-steelmaking-si_jclepro", "vogl_2018_001", "EU"),
    ("iea/iea_2013_chemical-catalytic-processes-roadmap-annexes_iea", "iea_2013_001", "global"),
    ("peer-reviewed/zhang_2003_biodiesel-waste-cooking-oil-process-design_biortech", "zhang_2003_002", "Canada"),
    ("peer-reviewed/collodi_2010_smr-hydrogen-co2-capture_cet", "collodi_2010_001", "EU"),
    ("other-reports/van-der-stricht_2020_steelanol-steel-offgas-fermentation-ethanol", "van-der-stricht_2020_001", "EU"),
    ("other-reports/ajsi_2016_technologies-customized-list-asean-steel", "ajsi_2016_001", "ASEAN and Japan"),
    ("other-reports/nedo_2008_japanese-technologies-energy-saving", "nedo_2008_001", "Japan"),
    ("other-reports/ricardo-aea_2013_decarbonisation-heat-industry-review_decc", "ricardo-aea_2013_001", "UK and EU"),
    ("other-reports/irena_2014_renewable-energy-manufacturing-remap2030", "irena_2014_001", "global"),
    ("iea/iea_2026_heat-pump-monitor", "iea_2026_003", "global"),
    ("iea/iea_2026_global-methane-tracker-documentation", "iea_2026_004", "global (US base costs)"),
    ("iea/iea_2026_methane-abatement-model-cost-assumptions", "iea_2026_005", "US (base values before country adjustment)"),
    ("other-reports/purola_2021_heat-integration-hydrocarbon-processing-facility_uspto", "purola_2021_001", "Finland (generic plant simulation)"),
    ("peer-reviewed/tijani_2022_review-electric-cracking-hydrocarbons_acssuschemeng", "tijani_2022_001", "global / Netherlands"),
    ("other-reports/abella_2022_prelim-v1-6_ucalgary", "abella_2022_001", "global average"),
    ("gov-usa/doe/doe_2006_pumping-system-performance-sourcebook", "doe_2006_002", "US"),
    ("gov-usa/doe/doe_2003_fan-system-performance-sourcebook", "doe_2003_002", "US"),
    ("gov-usa/doe/doe_2016_compressed-air-system-performance-sourcebook", "doe_2016_002", "US"),
    ("peer-reviewed/almeida-benalcazar_2022_gas-fermentation-ethanol-decarbonization-si_compchemeng", "almeida-benalcazar_2022_001", "generic / EU"),
    ("other-reports/van-der-stricht_2025_steelanol-bfg-ethanol-economics", "van-der-stricht_2025_001", "global / EU"),
    ("peer-reviewed/denis-ryan_2016_carbon-intensive-materials-decarbonizing-si_climpol", "denis-ryan_2016_001", "global"),
    ("other-reports/gcca/gcca_2024_gnr-cement-indicators", "gcca_2024_001", "Canada"),
    ("other-reports/ietd_2013_industrial-efficiency-technology-database_iip", "ietd_2013_001", "global (US, Japan, China, India, EU)"),
    ("peer-reviewed/meyers_1997_handbook-petroleum-refining-processes_mcgraw-hill", "meyers_1997_001", "US (Gulf Coast licensor estimates)"),
    ("peer-reviewed/jones_2006_handbook-petroleum-processing_springer", "jones_2006_001", "generic / US"),
    ("peer-reviewed/challis_2020_dairy-pasteurisation-energy-human-factors_thesis", "challis_2020_001", "UK"),
    ("peer-reviewed/challis_2017_human-factors-dairy-process-control_jclepro", "challis_2017_001", "UK"),
    ("other-reports/kulterer_2020_audit-guide-steam-system-optimisation_klimaaktiv", "kulterer_2020_001", "Austria / EU"),
    ("peer-reviewed/weidmann_2013_swiss-energy-system-transformation-strategies_eth-thesis", "weidmann_2013_001", "Switzerland"),
    ("peer-reviewed/rootzen_2017_cement-co2-abatement-cost-supply-chain_climate-policy", "rootzen_2017_001", "Nordic countries"),
    ("iea/iea_2025_renewables-for-industry-low-temperature-heat-electrification_iea", "iea_2025_003", "EU / China / ASEAN"),
    ("other-reports/jrc/jrc_2026_idees-2023-industry-eu27", "jrc_2026_001", "EU27"),
    ("gov-usa/eia/eia_2023_buildings-appliance-equipment-costs-efficiencies", "eia_2023_002", "United States"),
    ("gov-usa/eia/eia_2013_buildings-appliance-equipment-costs-efficiencies", "eia_2013_002", "United States"),
    ("peer-reviewed/sharp_2014_biomass-boiler-higher-steam-temperatures-costs_tappij", "sharp_2014_001", "US"),
    ("gov-usa/doe/doe_2024_iac-tipsheet-electric-boiler", "doe_2024_002", "US"),
    ("gov-usa/doe/doe_2015_qtr-process-heating-technology-assessment", "doe_2015_005", "US"),
    ("peer-reviewed/zuberi_2022_electrification-industrial-boilers-usa_enef", "zuberi_2022_002", "US"),
    ("other-reports/ldes-council_2022_net-zero-heat-long-duration-storage", "ldes-council_2022_001", "global"),
    ("iea/iea-shc_2024_ship-technology-position-paper", "iea-shc_2024_001", "global"),
    ("peer-reviewed/marzouk_2025_solar-heat-industrial-processes-review_solar", "marzouk_2025_001", "global"),
    ("iea/iea_2025_low-temperature-heat-electrification-commentary", "iea_2025_002", "EU"),
    ("other-reports/iea_2024_future-of-geothermal-energy", "iea_2024_002", "global"),
    ("peer-reviewed/johnson_2016_oil-sands-mining-fugitive-methane_atmosenv", "johnson_2016_001", "Canada (Alberta)"),
    ("other-reports/clearstone_2019_alberta-uog-methane-inventory-methodology_aer", "clearstone_2019_001", "Canada (Alberta)"),
    ("peer-reviewed/abbas-wan-daud_2010_hydrogen-methane-decomposition-review_ijhe", "abbas-wan-daud_2010_001", "global"),
    ("other-reports/aea_2010_abatement-major-emitting-industrial-sectors_ccc", "aea_2010_001", "UK"),
    ("other-reports/bazzanella-ausfelder_2017_low-carbon-energy-feedstock-european-chemical-industry_dechema", "bazzanella-ausfelder_2017_001", "EU"),
    ("peer-reviewed/bishop_2026_cascading-wood-use-beccs_commsenv", "bishop_2026_001", "EU"),
    ("other-reports/ce-delft_2010_technological-developments-europe-co2-efficient-manufacturing", "ce-delft_2010_001", "EU"),
    ("peer-reviewed/dalsgard_2000_process-integration-danish-food-industry_ijat", "dalsgard_2000_001", "DK"),
    ("peer-reviewed/de-luna_2019_renewable-electrosynthesis-vs-petrochemical-processes_science", "de-luna_2019_001", "global"),
    ("gov-uk/decc/decc_2013_digest-uk-energy-statistics_dukes", "decc_2013_001", "UK"),
    ("peer-reviewed/duxon_2007_inorganic-polymer-green-concrete_cemconres", "duxon_2007_001", "global"),
    ("gov-nz/eeca/eeca_2010_steam-hot-water-process-heating-best-practice-guide", "eeca_2010_001", "NZ"),
    ("gov-usa/lbnl/einstein_2001_steam-systems-industry-energy-efficiency_aceee", "einstein_2001_001", "US"),
    ("other-reports/faber_2019_sunrise-sustainable-ammonia-production_sunrise", "faber_2019_001", "EU"),
    ("other-reports/favier_2018_sustainable-future-european-cement-concrete_ethz", "favier_2018_001", "EU"),
    ("peer-reviewed/feliciano-bruzual-mathews_2013_bio-pci-charcoal-injection-economics_revmetal", "feliciano-bruzual-mathews_2013_001", "global"),
    ("peer-reviewed/fischedick_2014_innovative-steel-production-technologies_jclepro", "fischedick_2014_001", "DE"),
    ("peer-reviewed/fritzson-berntsson_2006_slaughter-meat-energy-efficiency_jfoodeng", "fritzson-berntsson_2006_002", "SE"),
    ("peer-reviewed/fritzson-berntsson_2006_slaughter-meat-process-integration_jfoodeng", "fritzson-berntsson_2006_003", "SE"),
    ("peer-reviewed/hita_2011_heat-recovery-food-drink-times_eceee", "hita_2011_001", "FR"),
    ("other-reports/hptstcj_2010_heat-pumps-food-beverage-survey", "hptstcj_2010_001", "JP"),
    ("peer-reviewed/huang_2025_biocarbon-direct-injection-blast-furnace_srin", "huang_2025_001", "CA"),
    ("iea/iea_2009_energy-technology-transitions-for-industry_iea", "iea_2009_002", "global"),
    ("iea/iea_2012_bioenergy-heat-power-roadmap_iea", "iea_2012_001", "global"),
    ("other-reports/ieaghg_2013_co2-capture-ironmaking-overview_tr3", "ieaghg_2013_002", "global"),
    ("peer-reviewed/khatir_2013_bread-baking-oven-energy-efficiency_apenergy", "khatir_2013_001", "UK"),
    ("other-reports/amf/matschegg_2023_sustainable-aviation-fuels_amf-task63", "matschegg_2023_001", "global"),
    ("peer-reviewed/motelica_2012_membrane-retrofit-paraffin-olefin-separation_iecr", "motelica_2012_001", "EU"),
    ("other-reports/nera-aea_2009_uk-supply-curve-renewable-heat_decc", "nera-aea_2009_001", "UK"),
    ("peer-reviewed/norgate-langberg_2009_charcoal-steelmaking-environmental-economic_isij", "norgate-langberg_2009_001", "AU"),
    ("peer-reviewed/norman_2013_industrial-energy-use-improvement-potential_bath", "norman_2013_001", "UK"),
    ("peer-reviewed/ploegmakers_2013_membrane-ethylene-ethane-hybrid-distillation_iecr", "ploegmakers_2013_001", "EU"),
    ("peer-reviewed/polusani_2025_lca-geopolymer-concrete-fly-ash-slag_iis", "polusani_2025_001", "IN"),
    ("peer-reviewed/premkumar-rangaiah_2009_retrofitting-columns-dividing-wall_cherd", "premkumar-rangaiah_2009_001", "global"),
    ("peer-reviewed/saygin_2012_industrial-energy-use-co2-efficiency-biomass-ccs_uu", "saygin_2012_001", "global; NL (Ch.5-6); OECD/non-OECD (Ch.7)"),
    ("other-reports/amf/schroder_2020_methanol-as-motor-fuel_amf-annex56", "schroder_2020_001", "global"),
    ("peer-reviewed/shafiei_2023_city-gate-water-bath-heater-optimization_jthermeng", "shafiei_2023_001", "IR"),
    ("peer-reviewed/sharp_1999_calcium-sulfoaluminate-cements_ice-themes-low-carbon-concrete", "sharp_1999_001", "global"),
    ("peer-reviewed/stengel_2009_lca-geopolymer-concrete_tum", "stengel_2009_001", "EU"),
    ("gov-usa/doe/stubbles_2000_energy-use-us-steel-industry_doe", "stubbles_2000_001", "US"),
    ("other-reports/van-den-brink_2002_combined-catalytic-nox-n2o-nitric-acid_ecn", "van-den-brink_2002_001", "EU"),
    ("peer-reviewed/walenta-gartner_2007_low-co2-belite-sulfoaluminate-ferrite-cement_iccc", "walenta-gartner_2011_001", "EU"),
    ("other-reports/bisaka_2016_prereduction-dc-smelting-ilmenite-pellets_saimm", "bisaka_2016_001", "South Africa"),
    ("peer-reviewed/song_2025_ilmenite-pellets-eaf-titania-slag_rineng", "song_2025_001", "China"),
    ("peer-reviewed/pistorius_2009_partial-slag-solidification-ilmenite-smelter_up", "pistorius_2009_001", "South Africa"),
]
# parameters that describe the example rather than quantify it
EXAMPLE_ATTRS = ("description", "trl", "state-of-development", "year-available")


def slug(x: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (x or "").lower()).strip("-")


def example_id(source_id: str, tech: str, process: str = "") -> str:
    return f"{source_id}:{tech}" + (f"@{slug(process)}" if process else "")



_DOCS = None


def held_document(stem: str) -> str | None:
    """The document file a source's extraction comes from, as the file manifest names
    it (so the build gives the same answer without primary-sources/ on the disk)."""
    global _DOCS
    if _DOCS is None:
        _DOCS = {}
        for r in read(RAW / "source-details_files.csv"):
            f = (r.get("file_on_disk") or "").strip()
            if f and not f.endswith(".csv"):
                folder = (r.get("file_path") or "").strip()
                k = f"{folder}/{f.rsplit('.', 1)[0]}" if folder else f.rsplit(".", 1)[0]
                _DOCS.setdefault(k, []).append(f)
    # several formats of one document (the MIDDEN .ods and .pdf): the first by name,
    # as the build read them from the disk before
    return min(_DOCS[stem]) if _DOCS.get(stem) else None

class Loader:
    def __init__(self):
        self.facts, self.flows, self.cons = [], [], []
        self.examples: dict[str, dict] = {}
        self.missing: list[str] = []
        self.trl_map = {(r["scale"], r["level"].strip().upper()): r["canonical-trl"]
                        for r in read(REFERENCE / "trl-scale-map.csv")}

    # -- examples ---------------------------------------------------------
    def example(self, source_id, tech, process="", node=None, extraction=""):
        eid = example_id(source_id, tech, process)
        ex = self.examples.get(eid)
        if ex is None:
            node = node or {}
            ex = self.examples[eid] = {
                "example-id": eid, "source-id": source_id, "source-technology": tech,
                "process": process, "sector-reported": node.get("sector", ""),
                "end-use-reported": node.get("end-use", ""),
                "step-reported": node.get("process-step", ""),
                "service-unit-reported": node.get("service-unit", ""),
                "extraction-file": extraction, "description": "", "trl": "",
                "trl-as-reported": "", "trl-scale": "", "state-of-development": "",
                "year-available": "", "capacity-basis": "", "capacity-unit": "",
                "note": ""}
        return ex

    def attr(self, ex, param, val, scale="trl-1-9"):
        """Record a descriptive parameter on the example. First value wins; a
        readiness level is mapped onto the common 1-9 ladder, carrying a range
        at its lower bound."""
        if param == "trl":
            if ex["trl"]:
                return
            v = (val or "").strip().upper().replace("–", "-")
            parts = [p.strip() for p in re.split(r"[-,]", v) if p.strip()]
            hits = [h for h in (self.trl_map.get((scale, p)) for p in parts) if h]
            if not hits:
                ex["note"] = (ex["note"] + " " if ex["note"] else "") + (
                    f"Readiness {val!r} on the {scale} scale is not in trl-scale-map.csv.")
                return
            ex["trl"], ex["trl-as-reported"], ex["trl-scale"] = min(hits, key=int), val, scale
        elif not ex[param]:
            ex[param] = val

    # -- facts ------------------------------------------------------------
    def fact(self, ex, **kw):
        row = {"record-id": f"tp-{len(self.facts) + 1:04d}", "example-id": ex["example-id"],
               "source-technology": ex["source-technology"]}
        row.update(kw)
        self.facts.append(row)

    # -- the source shapes ----------------------------------------------------
    def template_file(self, rel, source_id):
        """A single-file extraction in the template format: one process step,
        no node map. These four are boiler and CHP catalogues."""
        path = RAW / rel
        if not path.exists():
            self.missing.append(rel)
            return
        node = {"sector": "all-industry", "end-use": "steam-and-hot-water"}
        for row in read(path):
            tech = (row.get("technology") or "").strip()
            if not tech:
                continue
            ex = self.example(source_id, tech, row.get("process", ""), node, rel)
            param = (row.get("parameter") or "").strip()
            val = (row.get("value") or "").strip()
            if param in EXAMPLE_ATTRS:
                if val:
                    self.attr(ex, param, val)
                continue
            if not (val or row.get("formula")):
                continue
            note = (row.get("note") or "").strip()
            m = re.search(r"COST BASIS:\s*([a-z-]+)", note)
            cap = re.search(r"CAPACITY BASIS:\s*([a-z-]+)", note)
            unit = (row.get("unit") or "").strip()
            self.fact(ex, parameter=param, target=(row.get("target") or "").strip(),
                      unit=unit, value=val, **{"capacity-basis": cap.group(1) if cap else ""},
                      **{"value-min": (row.get("value-min") or "").strip(),
                         "value-max": (row.get("value-max") or "").strip(),
                         "uncertainty-type": "reported-range" if (row.get("value-min") or
                                                                  row.get("value-max")) else "",
                         "value-type": "derived" if row.get("formula") else "reported",
                         "formula": (row.get("formula") or "").strip(),
                         "cost-basis": m.group(1) if m else "", "currency": currency_year(unit),
                         "source-id": source_id, "source-file": path.name,
                         "source-locator": (row.get("reference") or "").strip(),
                         "extraction-file": rel, "via-source-id": "", "confidence": "",
                         "status": "draft", "note": note})

    def nodes(self, stem, source_id):
        path = RAW / f"{stem}_nodes.csv"
        if not path.exists():
            self.missing.append(f"{stem}_nodes.csv")
            return {}
        out = {}
        for r in read(path):
            out[(r["process"], r["technology"])] = r
            self.example(source_id, r["technology"], r["process"], r, f"{stem}.csv")
        return out

    def ukerc(self):
        stem, sid = UKERC
        nm = self.nodes(stem, sid)
        if not nm:
            return
        recon = UkercRecon()
        for n, row in enumerate(read(RAW / f"{stem}.csv"), 1):
            key = (row.get("process", ""), row["technology"])
            if key not in nm:
                continue
            orig = recon.credit("main", n, row)
            if orig is None:
                continue
            ex = self.example(sid, row["technology"], row.get("process", ""))
            param, val = row["parameter"], (row.get("value") or "").strip()
            if param in ("trl", "state-of-development", "year-available"):
                if val:
                    self.attr(ex, param, val)
                continue
            if not (val or row.get("formula")):
                continue
            unit = (row.get("unit") or "").strip()
            note = (row.get("note") or "").strip()
            cost = param.startswith(("capex", "opex"))
            # UKERC is a compilation: the value belongs to whoever it cites
            # (design rule: two-level attribution). The original is the source,
            # UKERC the via.
            self.fact(ex, parameter=param, target=(row.get("target") or "").strip(),
                      unit=unit, value=val, region="UK",
                      **{"value-min": (row.get("value-min") or "").strip(),
                         "value-max": (row.get("value-max") or "").strip(),
                         "uncertainty-type": "reported-range" if (row.get("value-min") or
                                                                  row.get("value-max")) else "",
                         "value-type": "derived" if row.get("formula") else "reported",
                         "formula": (row.get("formula") or "").strip(),
                         "cost-basis": "installed" if cost else "",
                         "capacity-basis": "product-output" if cost else "",
                         "currency": currency_year(unit),
                         "source-id": orig[0] if orig else sid,
                         "via-source-id": sid if orig else "",
                         "via-locator": (row.get("reference") or "").strip(),
                         "source-file": "ukerc_2013_usable-energy-database.xlsx",
                         "source-locator": (row.get("reference") or "").strip(),
                         "extraction-file": f"data/raw_data/technology_database/{stem}.csv",
                         "confidence": "", "status": "draft",
                         "note": note + (f"  Cited by UKERC alongside: {', '.join(orig[1:])}."
                                         if len(orig) > 1 else "")})
        techs = {t for (_, t) in nm}
        for i, row in enumerate(read(RAW / f"{stem}_flows.csv"), 1):
            if row["technology"] not in techs:
                continue
            orig = recon.credit("flows", i, row)
            if orig is None:
                continue
            ex = self.example_for(sid, row["technology"], nm)
            self.flows.append({
                "record-id": f"tf-{i:04d}", "example-id": ex["example-id"],
                "source-technology": row["technology"],
                "direction": row["direction"], "commodity": row["commodity"],
                "value": row["value"], "unit": row["unit"], "per-unit": row["per-unit"],
                "value-min": row["value-min"], "value-max": row["value-max"],
                "co-product": row["co-product"], "value-type": "reported",
                "formula": row["formula"], "region": "UK",
                "source-id": orig[0] if orig else sid,
                "via-source-id": sid if orig else "",
                "source-file": "ukerc_2013_usable-energy-database.xlsx",
                "source-locator": row["reference"], "status": "draft", "note": row["note"]})
        for i, row in enumerate(read(RAW / f"{stem}_constraints.csv"), 1):
            if row["technology"] not in techs:
                continue
            orig = recon.credit("constraints", i, row)
            if orig is None:
                continue
            ex = self.example_for(sid, row["technology"], nm)
            self.cons.append({
                "record-id": f"tc-{i:04d}", "example-id": ex["example-id"],
                "source-technology": row["technology"],
                "constraint": row["constraint"], "region": row["region"],
                "year": row["year"], "value": row["value"], "unit": row["unit"],
                "constraint-basis": row["constraint-basis"], "value-type": "reported",
                "source-id": orig[0] if orig else sid,
                "via-source-id": sid if orig else "",
                "source-file": "ukerc_2013_usable-energy-database.xlsx",
                "source-locator": row["reference"], "status": "draft", "note": row["note"]})

        self.missing.extend(recon.unreconciled())

    def example_for(self, sid, tech, nm, process=""):
        """The example a flow or constraint row belongs to. Those files carry
        no process column, so a technology listed under one process only is
        that example; one listed under several is keyed on its first."""
        if process:
            return self.example(sid, tech, process)
        procs = [p for (p, t) in nm if t == tech]
        return self.example(sid, tech, procs[0] if procs else "")

    def bataille(self):
        stem, sid = BATAILLE
        nm = self.nodes(stem, sid)
        for row in read(RAW / f"{stem}.csv") if nm else []:
            key = (row.get("process", ""), row["technology"])
            if key not in nm:
                continue
            ex = self.example(sid, row["technology"], row.get("process", ""))
            param, val = row["parameter"], (row.get("value") or "").strip()
            if param in ("trl", "state-of-development", "year-available"):
                if val:
                    self.attr(ex, param, val)
                continue
            if not val:
                continue
            unit = (row.get("unit") or "").strip()
            note = (row.get("note") or "").strip()
            cost = param.startswith(("capex", "opex"))
            if cost:
                ex["capacity-basis"] = ex["capacity-basis"] or (
                    "nameplate" if unit.endswith("/kW") else "annual-output")
                ex["capacity-unit"] = ex["capacity-unit"] or unit.split("/")[-1]
            self.fact(ex, parameter=param, target=(row.get("target") or "").strip(),
                      unit=unit, value=val, region="UK",
                      **{"value-min": (row.get("value-min") or "").strip(),
                         "value-max": (row.get("value-max") or "").strip(),
                         "uncertainty-type": "collapsed-rows" if row.get("value-min") else "",
                         "value-type": "reported", "formula": "",
                         "cost-basis": ("" if not cost else "incremental"
                                        if UKTM_ADD_ON.search(note) else "installed"),
                         "capacity-basis": "product-output" if cost else "",
                         "currency": currency_year(unit),
                         "source-id": sid, "via-source-id": "", "via-locator": "",
                         "source-file": "bataille_2017_heavy-industry-database.xlsx",
                         "source-locator": (row.get("reference") or "").strip(),
                         "extraction-file": f"data/raw_data/technology_database/{stem}.csv",
                         "confidence": "", "status": "review" if "REVIEW:" in note else "draft",
                         "note": note})

    def ipcc(self):
        stem, sid = IPCC
        nm = self.nodes(stem, sid)
        for row in read(RAW / f"{stem}.csv") if nm else []:
            key = (row.get("process", ""), row["technology"])
            if key not in nm:
                continue
            ex = self.example(sid, row["technology"], row.get("process", ""))
            param, val = row["parameter"], (row.get("value") or "").strip()
            # an empty readiness or state cell is kept as an unquantified row,
            # like any other IPCC cell without a number
            if param == "trl" and val:
                self.attr(ex, param, val, scale="nasa-1-9")
                continue
            if param in ("state-of-development", "year-available") and val:
                self.attr(ex, param, val)
                continue
            unit = (row.get("unit") or "").strip()
            orig = [x for x in (row.get("source-original") or "").split(";") if x]
            # a row with no number is kept: IPCC rates options whose cost cell
            # holds a condition ("Cost of H2"), and the rating is the finding
            quantified = val or row.get("value-min") or row.get("value-max")
            self.fact(ex, parameter=param, target=(row.get("target") or "").strip(),
                      unit=unit, value=val, region="global",
                      **{"value-min": (row.get("value-min") or "").strip(),
                         "value-max": (row.get("value-max") or "").strip(),
                         "uncertainty-type": "reported-range" if (row.get("value-min") or
                                                                  row.get("value-max")) else "",
                         "value-type": "reported", "formula": "", "cost-basis": "",
                         "capacity-basis": "", "currency": currency_year(unit),
                         "source-id": orig[0] if orig else sid,
                         "via-source-id": sid if orig else "",
                         "via-locator": (row.get("reference") or "").strip() if orig else "",
                         "source-file": f"{stem.split('/')[-1]}.pdf",
                         "source-locator": (row.get("reference") or "").strip(),
                         "extraction-file": f"data/raw_data/technology_database/{stem}.csv",
                         "confidence": "medium",
                         "status": "draft" if quantified else "unquantified",
                         "note": (row.get("note") or "").strip()
                                 + (f"  Cited by IPCC alongside: {', '.join(orig[1:])}."
                                    if len(orig) > 1 else "")})

    def iea_guide(self):
        """Readiness levels only, on the IEA ETP scale (which runs to 11)."""
        stem, sid = IEA_GUIDE
        nm = self.nodes(stem, sid)
        for row in read(RAW / f"{stem}.csv") if nm else []:
            key = (row.get("process", ""), row["technology"])
            val = (row.get("value") or "").strip()
            if key not in nm or not val or row["parameter"] != "trl":
                continue
            ex = self.example(sid, row["technology"], row.get("process", ""))
            self.attr(ex, "trl", val, scale="iea-etp")
            note = (row.get("note") or "").strip()
            if note and not ex["description"]:
                ex["description"] = note.split(".")[0]

    def etsap_brief(self, stem, sid, region="global"):
        nm = self.nodes(stem, sid)
        if not nm:
            return
        base = stem.split("/")[-1]
        src_file = held_document(stem) or base + ".pdf"
        for row in read(RAW / f"{stem}.csv"):
            key = (row.get("process", ""), row["technology"])
            if key not in nm:
                continue
            ex = self.example(sid, row["technology"], row.get("process", ""))
            param, val = (row.get("parameter") or "").strip(), (row.get("value") or "").strip()
            if param in EXAMPLE_ATTRS:
                if val:
                    self.attr(ex, param, val)
                continue
            # a range with no central value is kept (EAF dust at 1-780 g/t)
            if not (val or row.get("value-min") or row.get("value-max") or row.get("formula")):
                continue
            unit = (row.get("unit") or "").strip()
            note = (row.get("note") or "").strip()
            m = re.search(r"COST BASIS:\s*([a-z-]+)", note)
            cap = re.search(r"CAPACITY BASIS:\s*([a-z-]+)", note)
            self.fact(ex, parameter=param, target=(row.get("target") or "").strip(),
                      unit=unit, value=val, region=region,
                      **{"value-min": (row.get("value-min") or "").strip(),
                         "value-max": (row.get("value-max") or "").strip(),
                         "uncertainty-type": "reported-range" if (row.get("value-min") or
                                                                  row.get("value-max")) else "",
                         "value-type": "derived" if row.get("formula") else "reported",
                         "formula": (row.get("formula") or "").strip(),
                         "cost-basis": m.group(1) if m else "",
                         "capacity-basis": cap.group(1) if cap else
                         "product-output" if param.startswith(("capex", "opex")) else "",
                         "currency": currency_year(unit),
                         "source-id": sid, "via-source-id": "",
                         "source-file": src_file,
                         "source-locator": (row.get("reference") or "").strip(),
                         "extraction-file": f"data/raw_data/technology_database/{stem}.csv",
                         "confidence": "", "status": "draft", "note": note})
        fpath = RAW / f"{stem}_flows.csv"
        for row in read(fpath):
            key = (row.get("process", ""), row["technology"])
            if key not in nm:
                continue
            ex = self.example(sid, row["technology"], row.get("process", ""))
            self.flows.append({
                "record-id": f"tf-{len(self.flows) + 1:04d}", "example-id": ex["example-id"],
                "source-technology": row["technology"],
                "direction": row["direction"], "commodity": row["commodity"],
                "value": row["value"], "unit": row["unit"], "per-unit": row["per-unit"],
                "value-min": row["value-min"], "value-max": row["value-max"],
                "co-product": row["co-product"],
                "value-type": "derived" if row["formula"] else "reported",
                "formula": row["formula"], "region": region,
                "source-id": sid, "via-source-id": "", "source-file": src_file,
                "source-locator": row["reference"], "status": "draft", "note": row["note"]})


def load() -> Loader:
    L = Loader()
    for rel, sid in TEMPLATE_FILES:
        L.template_file(rel, sid)
    L.ukerc()
    L.bataille()
    L.ipcc()
    L.iea_guide()
    for stem, sid in ETSAP_BRIEFS:
        L.etsap_brief(stem, sid)
    for stem, sid, region in BRIEF_SHAPE:
        L.etsap_brief(stem, sid, region)
    return L


class UkercRecon:
    """UKERC rows whose original we hold (records/reference/ukerc-original-
    reconciliation.csv; user ruling 2026-10-03). When the original is held, its
    own figure is used and the UKERC row is dropped, unless UKERC states a
    reason for its different number (action `keep`). A row whose figure is not
    in the original at all is UKERC's own and is credited to UKERC (`ukerc-own`).
    `recredit:<id>` moves a
    row UKERC credited to the wrong document. A row crediting a held original
    that the file does not list is reported, so a newly held original gets
    reconciled before its UKERC rows are pooled beside it."""

    def __init__(self):
        self.rows = {(r["ukerc-file"], int(r["row"])): r
                     for r in read(REFERENCE / "ukerc-original-reconciliation.csv")}
        self.alias = {r["ukerc-source-id"]: r["held-source-id"] for r in self.rows.values()}
        self.held = {r["source_id"] for r in read(RAW / "source-details_files.csv")
                     if r.get("file_on_disk") and r.get("csv_data_extract")}
        self.held.discard(UKERC[1])
        self.loose = defaultdict(int)
        self.seen = set()

    def credit(self, part, n, row):
        """The originals the row is credited to, or None if it is dropped."""
        orig = [x for x in (row.get("source-original") or "").split(";") if x]
        r = self.rows.get((part, n))
        if r is None:
            if orig and orig[0] in self.held:
                self.loose[orig[0]] += 1
            return orig
        if r["technology"] != row["technology"] or (orig[:1] or [""])[0] != r["ukerc-source-id"]:
            raise SystemExit(f"ukerc-original-reconciliation.csv: {part} row {n} is "
                             f"{row['technology']} / {orig[:1]}, the file says {r['technology']} / "
                             f"{r['ukerc-source-id']}; the UKERC extraction changed, redo the rows")
        self.seen.add((part, n))
        act = r["action"]
        if act == "drop":
            return None
        if act == "ukerc-own":
            # not in the original at all: UKERC's own figure (its UK baseline,
            # stock constraints, TRLs), credited to UKERC (modeller 2026-10-03)
            return []
        if act.startswith("recredit:"):
            return [act.split(":", 1)[1]] + orig[1:]
        return [r["held-source-id"]] + orig[1:]

    def unreconciled(self):
        out = [f"UKERC {k} row {n} listed in ukerc-original-reconciliation.csv was not met"
               for (k, n) in sorted(set(self.rows) - self.seen)]
        out += [f"UKERC: {c} rows credit {s}, which is held, and are not reconciled "
                "(records/reference/ukerc-original-reconciliation.csv)"
                for s, c in sorted(self.loose.items())]
        return out
