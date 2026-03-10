"""
Extract residential calibration data and save to CIMS-formatted CSV files.

This script:
1. Extracts data for ALL provinces/territories
2. ALWAYS applies projections from residential_projection_params.json
3. Exports to CIMS-formatted CSV files

Output columns: Branch, Type, Region, Sector, Service, Technology, Parameter, 
                Context, Sub_Context, Target, Source, Unit, Year, Value

The following data is extracted:
- Housing stock (thousands of households)
- Building shares by type (Single Detached, Single Attached, Apartments, Mobile Homes)
- Floorspace per building
- Appliances per household
- Vintage bins (building age distributions) - LowMed and High density
- Heating technologies - Cold climate (all provinces), Marine climate (BC only)
  NOTE: BC exports BOTH Marine AND Cold climate heating data
- Cooling technologies (Room and Central)
- Water heating technologies

All percentage values are converted from 0-100 scale to 0-1 scale (fractions).

"""
# Robust path setup using __file__
import sys
from pathlib import Path

_current_file = Path(__file__)
_project_root = _current_file.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from source.nrcan.ceud.residential.residential import extract_all_data, PROVINCES
import pandas as pd
from pathlib import Path
import argparse


def format_to_cims(data, output_file, province_code):
    """
    Convert extracted data to CIMS-formatted CSV.
    
    Parameters
    ----------
    data : dict
        Extracted data from extract_all_data()
    output_file : str
        Path to output CSV file
    province_code : str
        Province code
        
    Returns
    -------
    str
        Path to the saved CSV file
    """
    # Retrieve extracted data
    housing_thousand = data.get('housing_thousand', {})
    building_shares = data.get('building_shares', {})
    floorspace_per_building = data.get('floorspace_per_building', {})
    appliances_per_household = data.get('appliances_per_household', {})
    vintage_bins_lowmed = data.get('vintage_bins_lowmed', [])
    vintage_bins_high = data.get('vintage_bins_high', [])
    cooling_share_data = data.get('cooling_share_data', {})
    wh_lowmed = data.get('wh_lowmed', {})
    wh_high = data.get('wh_high', {})
    wh_tech_lowmed = data.get('wh_tech_lowmed', {})
    wh_tech_high = data.get('wh_tech_high', {})
    
    # Check if BC for marine climate data
    is_bc = (province_code.upper() == 'BC')
    
    # Get heating data based on province
    if is_bc:
        # BC exports BOTH marine and cold climate
        heating_lowmed_marine = data.get('heating_lowmed_marine', {})
        heating_high_marine = data.get('heating_high_marine', {})
        heating_lowmed_cold = data.get('heating_lowmed_cold', {})
        heating_high_cold = data.get('heating_high_cold', {})
    else:
        # Other provinces only have cold
        heating_lowmed_cold = data.get('heating_lowmed_cold', {})
        heating_high_cold = data.get('heating_high_cold', {})
    
    rows = []
    
    def make_row(meta, year_dict, scale=1.0):
        """
        Helper to create rows with year-value pairs.
        
        Note: The extraction pipeline already converts percentages to fractions (0-1 scale),
        so most data uses scale=1.0. Only housing_thousand uses scale=1000.0 to convert
        from thousands to actual household count.
        """
        result = []
        for year, value in year_dict.items():
            if value is not None:
                row = {
                    'Branch': meta.get('Branch', ''),
                    'Type': meta.get('Type', ''),
                    'Region': province_code.upper(),
                    'Sector': 'Residential',
                    'Service': meta.get('Service', ''),
                    'Technology': meta.get('Technology', ''),
                    'Parameter': meta.get('Parameter', ''),
                    'Context': meta.get('Context', ''),
                    'Sub_Context': meta.get('Sub_Context', ''),
                    'Target': meta.get('Target', ''),
                    'Source': 'CEUD',
                    'Unit': meta.get('Unit', ''),
                    'Year': year,
                    'Value': value * scale
                }
                result.append(row)
        return result
    
    # ===== 1. HOUSING STOCK =====
    rows.extend(make_row(
        {'Branch': f'CIMS.CAN.{province_code.upper()}',
         'Type': 'Region',
         'Parameter': 'service_request',
         'Target': f'CIMS.CAN.{province_code.upper()}.Residential',
         'Unit': 'household'},
        housing_thousand, 1.0
    ))
    
    # ===== 2. APPLIANCES =====
    for appl_name, appl_data in appliances_per_household.items():
        if appl_data:
            rows.extend(make_row(
                {'Branch': f'CIMS.CAN.{province_code.upper()}.Residential.Dwellings',
                 'Type': 'Service',
                 'Service': 'Dwellings',
                 'Parameter': 'service_request',
                 'Target': f'CIMS.CAN.{province_code.upper()}.Residential.Dwellings.{appl_name}',
                 'Unit': 'unit/building'},
                appl_data, 1.0
            ))
    
    # ===== 3. BUILDING TYPES =====
    building_map = {
        "Apartments": ("Apartment", "High Density"),
        "Single Detached": ("Detached", "LowMed Density"),
        "Single Attached": ("Attached", "LowMed Density"),
        "Mobile Homes": ("Mobile", "LowMed Density"),
    }
    
    for source_key, (tech_name, density) in building_map.items():
        if source_key in building_shares:
            base_branch = f'CIMS.CAN.{province_code.upper()}.Residential.Dwellings.Building Type'
            
            # Building shares
            rows.extend(make_row(
                {'Branch': base_branch,
                 'Type': 'Service',
                 'Service': 'Building Type',
                 'Technology': tech_name,
                 'Parameter': 'market_share_total',
                 'Unit': '%'},
                building_shares[source_key], 1.0  # Already converted to 0-1 in extraction
            ))
            
            # Floorspace per building
            if source_key in floorspace_per_building:
                rows.extend(make_row(
                    {'Branch': base_branch,
                     'Type': 'Service',
                     'Service': 'Building Type',
                     'Technology': tech_name,
                     'Parameter': 'service_request',
                     'Target': f'{base_branch}.{density}',
                     'Unit': 'm2'},
                    floorspace_per_building[source_key], 1.0
                ))
    
    # ===== 4. COOLING - LowMed Density =====
    for ac_tech in cooling_share_data.keys():
        if ac_tech in cooling_share_data:
            rows.extend(make_row(
                {'Branch': f'CIMS.CAN.{province_code.upper()}.Residential.Dwellings.Building Type.LowMed Density.Cooling',
                 'Type': 'Service',
                 'Service': 'Cooling',
                 'Parameter': 'service_request',
                 'Target': f'CIMS.CAN.{province_code.upper()}.Residential.Dwellings.Building Type.LowMed Density.Cooling.{ac_tech}',
                 'Unit': 'GJ cooling/GJ cooling'},
                cooling_share_data[ac_tech], 1.0
            ))
    
    # ===== 5. COOLING - High Density =====
    for ac_tech in cooling_share_data.keys():
        if ac_tech in cooling_share_data:
            rows.extend(make_row(
                {'Branch': f'CIMS.CAN.{province_code.upper()}.Residential.Dwellings.Building Type.High Density.Cooling',
                 'Type': 'Service',
                 'Service': 'Cooling',
                 'Parameter': 'service_request',
                 'Target': f'CIMS.CAN.{province_code.upper()}.Residential.Dwellings.Building Type.High Density.Cooling.{ac_tech}',
                 'Unit': 'GJ cooling/GJ cooling'},
                cooling_share_data[ac_tech], 1.0
            ))
    
    # ===== 6. WATER HEATING SERVICE REQUEST - LowMed =====
    rows.extend(make_row(
        {'Branch': f'CIMS.CAN.{province_code.upper()}.Residential.Water Heating.LowMed Density',
         'Type': 'Service',
         'Service': 'Water Heating',
         'Parameter': 'service_request',
         'Target': f'CIMS.CAN.{province_code.upper()}.Residential.Water Heating.LowMed Density',
         'Unit': 'GJ water heating/GJ water heating'},
        wh_lowmed, 1.0
    ))
    
    # ===== 7. WATER HEATING SERVICE REQUEST - High =====
    rows.extend(make_row(
        {'Branch': f'CIMS.CAN.{province_code.upper()}.Residential.Water Heating.High Density',
         'Type': 'Service',
         'Service': 'Water Heating',
         'Parameter': 'service_request',
         'Target': f'CIMS.CAN.{province_code.upper()}.Residential.Water Heating.High Density',
         'Unit': 'GJ water heating/GJ water heating'},
        wh_high, 1.0
    ))
    
    # ===== 8. WATER HEATING TECHNOLOGIES - LowMed =====
    for wh_tech_name, wh_tech_data in wh_tech_lowmed.items():
        if wh_tech_data:
            rows.extend(make_row(
                {'Branch': f'CIMS.CAN.{province_code.upper()}.Residential.Water Heating.LowMed Density',
                 'Type': 'Service',
                 'Service': 'LowMed Density',
                 'Technology': wh_tech_name,
                 'Parameter': 'market_share_total',
                 'Unit': '% of GJ of water heating'},
                wh_tech_data, 1.0
            ))
    
    # ===== 9. WATER HEATING TECHNOLOGIES - High =====
    for wh_tech_name, wh_tech_data in wh_tech_high.items():
        if wh_tech_data:
            rows.extend(make_row(
                {'Branch': f'CIMS.CAN.{province_code.upper()}.Residential.Water Heating.High Density',
                 'Type': 'Service',
                 'Service': 'High Density',
                 'Technology': wh_tech_name,
                 'Parameter': 'market_share_total',
                 'Unit': '% of GJ of water heating'},
                wh_tech_data, 1.0
            ))
    
    # ===== 10. VINTAGE BINS - High Density =====
    for vint_tech, vint_data in vintage_bins_high.items():
        rows.extend(make_row(
            {'Branch': f'CIMS.CAN.{province_code.upper()}.Residential.Dwellings.Building Type.High Density.Vintage',
             'Type': 'Service',
             'Service': 'Vintage',
             'Technology': vint_tech,
             'Parameter': 'market_share_total',
             'Unit': '% of m2'},
            vint_data, 1.0
        ))
    
    # ===== 11. VINTAGE BINS - LowMed Density =====
    for vint_tech, vint_data in vintage_bins_lowmed.items():
        rows.extend(make_row(
            {'Branch': f'CIMS.CAN.{province_code.upper()}.Residential.Dwellings.Building Type.LowMed Density.Vintage',
             'Type': 'Service',
             'Service': 'Vintage',
             'Technology': vint_tech,
             'Parameter': 'market_share_total',
             'Unit': '% of m2'},
            vint_data, 1.0
        ))
    
    # ===== 12-15. HEATING TECHNOLOGIES =====
    
    # Function to export heating for a specific climate
    def export_heating_for_climate(heating_lowmed_data, heating_high_data, climate_label, heating_tech_map):
        """Export heating technologies for one climate type"""
        # LowMed Density Heating
        for vint_tech in vintage_bins_lowmed.keys():
            for h_tech_output, h_tech_source in heating_tech_map.items():
                tech_data = heating_lowmed_data.get(h_tech_source, {}) if h_tech_source else {}
                rows.extend(make_row(
                    {'Branch': f'CIMS.CAN.{province_code.upper()}.Residential.Dwellings.Building Type.LowMed Density.Vintage.{vint_tech} Bldg Code.Heating ({climate_label})',
                     'Type': 'Service',
                     'Service': 'Heating',
                     'Technology': h_tech_output,
                     'Parameter': 'market_share_total',
                     'Unit': '% of GJ of heat'},
                    tech_data, 1.0
                ))
        
        # High Density Heating
        for vint_tech in vintage_bins_high.keys():
            for h_tech_output, h_tech_source in heating_tech_map.items():
                tech_data = heating_high_data.get(h_tech_source, {}) if h_tech_source else {}
                rows.extend(make_row(
                    {'Branch': f'CIMS.CAN.{province_code.upper()}.Residential.Dwellings.Building Type.High Density.Vintage.{vint_tech} Bldg Code.Heating ({climate_label})',
                     'Type': 'Service',
                     'Service': 'Heating',
                     'Technology': h_tech_output,
                     'Parameter': 'market_share_total',
                     'Unit': '% of GJ of heat'},
                    tech_data, 1.0
                ))
    
    # Define heating technology mappings
    if is_bc:
        # BC: Export BOTH Marine and Cold climates
        
        # Marine climate tech map (9 technologies)
        # IMPORTANT: Values must match keys in heating_lowmed_marine/heating_high_marine dictionaries
        heating_tech_map_marine = {
            "NG - Low Efficiency": "NG - Low Efficiency",
            "NG - Medium Efficiency": "NG - Medium Efficiency",
            "NG - High Efficiency": "NG - High Efficiency",
            "Electric - Resistance": "Electric - Resistance",
            "Heating Oil - Low Efficiency": "Heating Oil - Low Efficiency",
            "Heating Oil - Medium Efficiency": "Heating Oil - Medium Efficiency",
            "Wood": "Wood",
            "NG - ASHP": "NG - ASHP",
            "Electric - ASHP": "Electric - ASHP"
        }
        
        # Cold climate tech map (10 technologies)
        # IMPORTANT: Values must match keys in heating_lowmed_cold/heating_high_cold dictionaries
        heating_tech_map_cold = {
            "NG - Low Efficiency": "NG - Low Efficiency",
            "NG - Medium Efficiency": "NG - Medium Efficiency",
            "NG - High Efficiency": "NG - High Efficiency",
            "Electric - Resistance": "Electric - Resistance",
            "Heating Oil - Low Efficiency": "Heating Oil - Low Efficiency",
            "Heating Oil - Medium Efficiency": "Heating Oil - Medium Efficiency",
            "Wood": "Wood",
            "NG - ASHP / NG - backup": "NG - ASHP / NG - backup",
            "Electric - ASHP / NG - backup": "Electric - ASHP / NG - backup",
            "Electric - ASHP / Electric - backup": "Electric - ASHP / Electric - backup"
        }
        
        # Export Marine climate
        export_heating_for_climate(heating_lowmed_marine, heating_high_marine, "Marine", heating_tech_map_marine)
        
        # Export Cold climate
        export_heating_for_climate(heating_lowmed_cold, heating_high_cold, "Cold", heating_tech_map_cold)
        
    else:
        # Other provinces: Only Cold climate
        # IMPORTANT: Values must match keys in heating_lowmed_cold/heating_high_cold dictionaries
        heating_tech_map_cold = {
            "NG - Low Efficiency": "NG - Low Efficiency",
            "NG - Medium Efficiency": "NG - Medium Efficiency",
            "NG - High Efficiency": "NG - High Efficiency",
            "Electric - Resistance": "Electric - Resistance",
            "Heating Oil - Low Efficiency": "Heating Oil - Low Efficiency",
            "Heating Oil - Medium Efficiency": "Heating Oil - Medium Efficiency",
            "Wood": "Wood",
            "NG - ASHP / NG - backup": "NG - ASHP / NG - backup",
            "Electric - ASHP / NG - backup": "Electric - ASHP / NG - backup",
            "Electric - ASHP / Electric - backup": "Electric - ASHP / Electric - backup"
        }
        
        # Export Cold climate only
        export_heating_for_climate(heating_lowmed_cold, heating_high_cold, "Cold", heating_tech_map_cold)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Ensure proper column order
    column_order = ['Branch', 'Type', 'Region', 'Sector', 'Service', 'Technology', 
                    'Parameter', 'Context', 'Sub_Context', 'Target', 'Source', 'Unit', 'Year', 'Value']
    df = df[column_order]
    
    # Sort for better readability
    if not df.empty:
        df = df.sort_values(['Branch', 'Technology', 'Year'])
    
    # Save to CSV
    output_path = Path(output_file)
    df.to_csv(output_path, index=False)
    
    print(f"  ✅ Saved {len(df):,} rows to {output_path}")
    
    return str(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract residential data from CEUD and export to CIMS-formatted CSV files"
    )
    parser.add_argument(
        "--output-dir",
        default=r"C:\cims\data\calibration\residential",
        help=r"Output directory for CSV files (default: C:\cims\data\calibration\residential)"
    )
    
    args = parser.parse_args()
    
    # Always extract all provinces with projections
    province_codes = list(PROVINCES.keys())
    output_dir = args.output_dir
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    results = {}
    failed = []
    
    print("="*80)
    print("RESIDENTIAL DATA EXTRACTION - ALL PROVINCES")
    print("="*80)
    print(f"Provinces: {', '.join(province_codes)}")
    print(f"Projections: ENABLED")
    print(f"Output format: CIMS")
    print(f"Output directory: {output_dir}")
    print("="*80)
    
    for prov in province_codes:
        try:
            print(f"\n{prov} - {PROVINCES[prov.upper()]}:")
            
            # Extract data (always with projections)
            data = extract_all_data(prov, apply_projections=True)
            results[prov] = data
            
            print(f"  ✅ Extraction complete")
            
            # Save to CIMS-formatted CSV
            output_file = Path(output_dir) / f"residential_{prov.upper()}.csv"
            format_to_cims(data, output_file, prov)
                
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            import traceback
            traceback.print_exc()
            failed.append((prov, str(e)))
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✅ Successful: {len(results)}/{len(province_codes)} provinces")
    
    if failed:
        print(f"❌ Failed: {len(failed)} provinces")
        for prov, error in failed:
            print(f"  • {prov}: {error}")
    
    print("="*80)
    print(f"\n✅ Complete! Extracted data for {len(results)} provinces.")
    print(f"CSV files saved to: {output_dir}")
