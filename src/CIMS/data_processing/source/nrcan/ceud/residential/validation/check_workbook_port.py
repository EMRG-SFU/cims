"""
Check that residential_heating_intensity.py reproduces the REM699 workbook.

Feeds the workbook's own CEUD inputs (AB_CEUD_Res, 2000-2020, extracted to
workbook_ab_fixture.csv) through the module with WORKBOOK_OPTIONS and compares
against AB_CIMS_Input_Res rows 365-368 (High Density) and 370-373 (LowMed).

The fixture's sh_by_fuel values are the workbook's plugged space-heating fuel
totals (rows 511-535), including its coal/propane range error, so that the
comparison isolates the calculation chain from there onward.

Then switches each fix on one at a time and prints the 2015 values.

Run:  python -m CIMS.data_processing.source.nrcan.ceud.residential.validation.check_workbook_port
"""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

import CIMS.data_processing.source.nrcan.ceud.residential.residential_heating_intensity as hi

FIXTURE = Path(__file__).with_name('workbook_ab_fixture.csv')
TOL = 1e-9


def load_fixture(path: Path = FIXTURE) -> tuple[hi.RegionInputs, pd.DataFrame]:
    raw = pd.read_csv(path, encoding='utf-8')
    parts = raw['key'].str.split('|', expand=True)
    raw['kind'], raw['a'], raw['b'] = parts[0], parts[1], parts[2]

    def wide(kind, col='a', sub=None):
        d = raw[raw['kind'] == kind]
        if sub is not None:
            d = d[d['a'] == sub]
            col = 'b'
        return d.pivot(index='year', columns=col, values='value')

    vintages = hi.BASE_VINTAGES
    inp = hi.RegionInputs(
        region='AB',
        vintages=vintages,
        stock={bt: wide('stock', sub=bt).reindex(columns=hi.SYSTEMS) for bt in hi.BUILDING_TYPES},
        eff_single=wide('eff')[list(hi.SINGLE_SYSTEMS)] / 100.0,
        floor={bt: wide('floor', sub=bt)[list(vintages)] for bt in hi.BUILDING_TYPES},
        sh_by_fuel=wide('sh_by_fuel')[list(hi.FUELS)],
        energy_by_type=wide('energy_by_type')[list(hi.BUILDING_TYPES)],
        sh_by_vintage=wide('sh_by_vintage')[list(vintages)],
        floor_total=wide('floor_total')[list(hi.BUILDING_TYPES)],
    )
    exp = raw[raw['kind'] == 'expected'].rename(columns={'a': 'variable', 'b': 'bin'})
    return inp, exp[['variable', 'bin', 'year', 'value']]


def main() -> bool:
    inp, expected = load_fixture()
    got = hi.compute_region(inp, hi.WORKBOOK_OPTIONS)['intensity']
    cmp = expected.merge(got, on=['variable', 'bin', 'year'], how='left', suffixes=('_wb', '_py'))
    cmp['diff'] = (cmp['value_py'] - cmp['value_wb']).abs()
    worst = cmp['diff'].max()
    ok = bool(cmp['value_py'].notna().all() and worst < TOL)
    print(f'Workbook reproduction: {len(cmp)} cells, max |diff| = {worst:.3e} -> {"PASS" if ok else "FAIL"}')
    if not ok:
        print(cmp.sort_values('diff', ascending=False).head(10).to_string())

    # effect of the bin-weighting fix alone (the other fixes need tables the
    # workbook did not carry: Tables 6, 8 and 33)
    opts = replace(hi.WORKBOOK_OPTIONS, bin_weighting='floor')
    fixed = hi.compute_region(inp, opts)['intensity']
    view = (cmp[cmp['year'] == 2015][['variable', 'bin', 'value_wb']]
            .merge(fixed[fixed['year'] == 2015], on=['variable', 'bin']))
    print('\n2015, workbook vs floor-weighted bins:')
    print(view.rename(columns={'value': 'floor_weighted'}).to_string(index=False))
    return ok


if __name__ == '__main__':
    raise SystemExit(0 if main() else 1)
