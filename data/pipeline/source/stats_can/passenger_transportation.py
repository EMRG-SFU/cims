
from __future__ import annotations
import argparse, csv, re
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import pandas as pd

DEFAULT_STATCAN_FILE = Path(r"C:\cims\data\raw_data\stats_can\market_shares\2010002501-eng.csv")
DEFAULT_EPA_FILE = Path(r"C:\cims\data\raw_data\epa\table_export.csv")
DEFAULT_OUTPUT_FILE = Path("passenger_transportation_market_shares.csv")
HISTORICAL_START_YEAR = 2000
FIRST_OBSERVED_YEAR = 2017
LAST_OBSERVED_YEAR = 2025
TOTAL_FUEL = "All fuel types"
PROXY_REGIONS: Dict[str, str] = {"Alberta":"Saskatchewan","Newfoundland and Labrador":"Nova Scotia","Nunavut":"Northwest Territories"}
EXCLUDE_REGIONS = {"Canada"}
TECHNOLOGIES = ["Gasoline Existing","Gasoline Standard","Gasoline Efficient","Diesel Existing","Diesel Standard","Diesel Efficient","Hybrid","Plug-in Hybrid","BEV500","BEV800","Other Fuel Types"]
EFFICIENT_PACKAGES = {"GDPI, Fixed Valve Timing, Multi-Valve","GDPI, Variable Valve Timing, Multi-Valve","GDI, Fixed Valve Timing, Multi-Valve","GDI, Variable Valve Timing, Multi-Valve","GDI, Variable Valve Timing, Two-Valve"}
STANDARD_PACKAGES = {"Port, Variable Valve Timing, Multi-Valve","Carb, Fixed Valve Timing, Two-Valve","Carb, Fixed Valve Timing, Multi-Valve","TBI, Fixed Valve Timing, Two-Valve","TBI, Fixed Valve Timing, Multi-Valve","Port, Fixed Valve Timing, Two-Valve","Port, Fixed Valve Timing, Multi-Valve","Port, Variable Valve Timing, Two-Valve"}
FUEL_NAME_MAP = {"Other fuel types 4":"Other fuel types","Newfoundland and Labrador 2":"Newfoundland and Labrador","Alberta 2":"Alberta"}

def resolve_path(path: str|Path) -> Path:
    p = Path(path)
    if p.exists(): return p
    basename = str(path).replace("\\", "/").split("/")[-1]
    for loc in [Path.cwd()/basename, Path('/mnt/data')/basename]:
        if loc.exists(): return loc
    return p

def clean_name(v) -> str:
    s = "" if v is None else str(v).strip().lstrip("\ufeff")
    s = FUEL_NAME_MAP.get(s, s)
    s = re.sub(r"\s+\d+$", "", s).strip()
    return FUEL_NAME_MAP.get(s, s)

def to_number(v) -> float:
    if v is None: return np.nan
    if isinstance(v,(int,float)): return float(v)
    s = str(v).strip().replace(",", "")
    if s in {"","..","-","nan","NaN"}: return np.nan
    try: return float(s)
    except ValueError: return np.nan

def parse_year(v) -> Optional[int]:
    m = re.search(r"(19|20)\d{2}", str(v)) if v is not None else None
    return int(m.group(0)) if m else None

def read_statscan_vehicle_sales(csv_path, first_observed_year=FIRST_OBSERVED_YEAR, last_observed_year=LAST_OBSERVED_YEAR) -> pd.DataFrame:
    with open(resolve_path(csv_path), 'r', encoding='utf-8-sig', newline='') as f:
        rows = list(csv.reader(f))
    geography_row_idx = quarter_row_idx = units_row_idx = None
    for i,row in enumerate(rows):
        if row and clean_name(row[0]) == 'Geography': geography_row_idx = i
        elif row and clean_name(row[0]) == 'Fuel type': quarter_row_idx = i
        elif row and any(clean_name(c) == 'Units' for c in row[:3]): units_row_idx = i; break
    if geography_row_idx is None or quarter_row_idx is None or units_row_idx is None:
        raise ValueError('Could not find Geography, Fuel type, and Units rows in StatCan CSV.')
    geography_row, quarter_row = rows[geography_row_idx], rows[quarter_row_idx]
    geographies, current_geo = [], None
    for col in range(max(len(geography_row), len(quarter_row))):
        if col == 0: geographies.append(None); continue
        geo = clean_name(geography_row[col]) if col < len(geography_row) else ''
        if geo: current_geo = geo
        geographies.append(current_geo)
    rec=[]
    for row in rows[units_row_idx+1:]:
        if not row: continue
        fuel = clean_name(row[0])
        if not fuel or fuel in {'Symbol legend:','Footnotes:','How to cite:'}: break
        for col in range(1, min(len(row), len(quarter_row), len(geographies))):
            region, year = geographies[col], parse_year(quarter_row[col])
            if region is None or year is None or year < first_observed_year or year > last_observed_year: continue
            rec.append({'region_original':clean_name(region),'year':year,'fuel_type':fuel,'registrations':to_number(row[col])})
    df=pd.DataFrame(rec)
    if df.empty: raise ValueError('No vehicle registration data were read from StatCan CSV.')
    return df.groupby(['region_original','year','fuel_type'], as_index=False)['registrations'].sum(min_count=1).rename(columns={'registrations':'annual_sales'})

def apply_region_proxies(annual, proxy_regions):
    annual=annual.copy(); annual['region']=annual['region_original']; annual['source_region']=annual['region_original']; annual['is_proxy']=False
    non_proxy=annual[~annual['region_original'].isin(proxy_regions.keys())].copy(); frames=[]
    for target, source in proxy_regions.items():
        proxy=annual[annual['region_original']==source].copy()
        if proxy.empty: raise ValueError(f'Proxy source region {source!r} has no data for {target!r}.')
        proxy['region']=target; proxy['source_region']=source; proxy['is_proxy']=True; frames.append(proxy)
    return pd.concat([non_proxy]+frames, ignore_index=True).query('region not in @EXCLUDE_REGIONS').copy()

def cagr(start,end,periods):
    if pd.isna(start) or pd.isna(end) or periods <= 0: return np.nan
    if start == 0 and end == 0: return 0.0
    if start == 0 and end != 0: return 0.0
    if start > 0 and end == 0: return -1.0
    if start < 0 or end < 0: return np.nan
    return (end/start)**(1/periods)-1

def backcast_group(g):
    key=g[['region','source_region','is_proxy','fuel_type']].iloc[0].to_dict(); series=g.set_index('year')['annual_sales'].to_dict()
    start,end=series.get(FIRST_OBSERVED_YEAR,np.nan),series.get(LAST_OBSERVED_YEAR,np.nan); gr=cagr(start,end,LAST_OBSERVED_YEAR-FIRST_OBSERVED_YEAR)
    out=[]
    for y in range(HISTORICAL_START_YEAR,LAST_OBSERVED_YEAR+1):
        if y in series and not pd.isna(series[y]): val=series[y]
        elif y < FIRST_OBSERVED_YEAR: val=np.nan if pd.isna(start) or pd.isna(gr) else (0.0 if start==0 or gr<=-1 else start/((1+gr)**(FIRST_OBSERVED_YEAR-y)))
        else: val=np.nan
        out.append({**key,'year':y,'annual_sales':val})
    return pd.DataFrame(out)

def add_backcast_history(df):
    return pd.concat([backcast_group(g) for _,g in df.groupby(['region','source_region','is_proxy','fuel_type'], dropna=False)], ignore_index=True)

def classify_engine_package(pkg):
    pkg=clean_name(pkg)
    if pkg in EFFICIENT_PACKAGES: return 'Efficient'
    if pkg in STANDARD_PACKAGES: return 'Standard'
    return None

def load_gasoline_technology_shares(epa_path):
    df=pd.read_csv(resolve_path(epa_path), dtype=str)
    for col in ['Engine Package','Model Year','Production (000)']:
        if col not in df.columns: raise ValueError(f'EPA file missing required column: {col}')
    df['year']=df['Model Year'].map(parse_year); df['class']=df['Engine Package'].map(classify_engine_package); df['production']=df['Production (000)'].map(to_number)
    df=df[df['year'].notna() & df['class'].notna() & df['production'].notna() & (df['production']>0)].copy(); df['year']=df['year'].astype(int)
    wide=df.groupby(['year','class'], as_index=False)['production'].sum().pivot(index='year', columns='class', values='production').fillna(0).reset_index()
    for col in ['Standard','Efficient']:
        if col not in wide: wide[col]=0.0
    wide['total']=wide['Standard']+wide['Efficient']; wide=wide[wide['total']>0].copy()
    wide['standard_share']=wide['Standard']/wide['total']; wide['efficient_share']=wide['Efficient']/wide['total']
    return wide[['year','standard_share','efficient_share']].sort_values('year').reset_index(drop=True)

def lookup_share(shares, year):
    if year in shares: return shares[year]
    yrs=sorted(shares); prior=[y for y in yrs if y <= year]
    return shares[prior[-1] if prior else yrs[0]]

def expand_vehicle_technologies(fuel_df, shares_df):
    shares=shares_df.set_index('year')[['standard_share','efficient_share']].to_dict('index'); rows=[]
    for _,r in fuel_df.iterrows():
        fuel,y,sales=clean_name(r['fuel_type']),int(r['year']),r['annual_sales']
        base={'region':r['region'],'source_region':r.get('source_region',r['region']),'is_proxy':r.get('is_proxy',False),'year':y}
        if fuel in {TOTAL_FUEL,'All zero-emission vehicles'}: continue
        if fuel == 'Gasoline': splits=[('Gasoline Existing',1.0),('Gasoline Standard',0.0),('Gasoline Efficient',0.0)] if y==2000 else [('Gasoline Existing',0.0),('Gasoline Standard',lookup_share(shares,y)['standard_share']),('Gasoline Efficient',lookup_share(shares,y)['efficient_share'])]
        elif fuel == 'Diesel': splits=[('Diesel Existing',1.0),('Diesel Standard',0.0),('Diesel Efficient',0.0)] if y==2000 else [('Diesel Existing',0.0),('Diesel Standard',lookup_share(shares,y)['standard_share']),('Diesel Efficient',lookup_share(shares,y)['efficient_share'])]
        elif fuel == 'Battery electric': splits=[('BEV500',1.0),('BEV800',0.0)]
        elif fuel == 'Plug-in hybrid electric': splits=[('Plug-in Hybrid',1.0)]
        elif fuel == 'Hybrid electric': splits=[('Hybrid',1.0)]
        elif fuel == 'Other fuel types': splits=[('Other Fuel Types',1.0)]
        else: continue
        for tech,sh in splits: rows.append({**base,'fuel_type':tech,'annual_sales':sales*sh})
    tech_df=pd.DataFrame(rows)
    region_years=tech_df[['region','source_region','is_proxy','year']].drop_duplicates(); existing=tech_df[tech_df['fuel_type']=='BEV800'][['region','year']].drop_duplicates(); add=[]
    for _,r in region_years.iterrows():
        if not ((existing['region']==r['region']) & (existing['year']==r['year'])).any(): add.append({**r.to_dict(),'fuel_type':'BEV800','annual_sales':0.0})
    return pd.concat([tech_df,pd.DataFrame(add)], ignore_index=True) if add else tech_df

def calculate_market_shares(tech_df):
    totals=tech_df.groupby(['region','year'], as_index=False)['annual_sales'].sum(min_count=1).rename(columns={'annual_sales':'total'})
    out=tech_df.merge(totals,on=['region','year'],how='left'); out['market_share']=np.where(out['total'].notna() & (out['total']!=0), out['annual_sales']/out['total'], np.nan)
    return out.groupby(['region','year','fuel_type'], as_index=False)['market_share'].sum(min_count=1)

def build_market_shares(statcan_file=DEFAULT_STATCAN_FILE, epa_file=DEFAULT_EPA_FILE, output_file=DEFAULT_OUTPUT_FILE):
    annual=read_statscan_vehicle_sales(statcan_file); annual=apply_region_proxies(annual, PROXY_REGIONS); annual=add_backcast_history(annual)
    shares=load_gasoline_technology_shares(epa_file); tech_sales=expand_vehicle_technologies(annual, shares); result=calculate_market_shares(tech_sales)
    order={t:i for i,t in enumerate(TECHNOLOGIES)}; result=result[result['fuel_type'].isin(TECHNOLOGIES)].copy(); result['_order']=result['fuel_type'].map(order)
    result=result.sort_values(['region','year','_order']).drop(columns='_order')
    Path(output_file).parent.mkdir(parents=True, exist_ok=True) if Path(output_file).parent != Path('.') else None
    result.to_csv(output_file,index=False); return result

def parse_args():
    ap=argparse.ArgumentParser(description='Build passenger transportation market shares for CIMS.')
    ap.add_argument('--statcan-file', default=str(DEFAULT_STATCAN_FILE)); ap.add_argument('--epa-file', default=str(DEFAULT_EPA_FILE)); ap.add_argument('--output-file', default=str(DEFAULT_OUTPUT_FILE)); return ap.parse_args()
if __name__ == '__main__':
    args=parse_args(); df=build_market_shares(args.statcan_file,args.epa_file,args.output_file); print(f'Wrote {len(df):,} rows to {args.output_file}')
