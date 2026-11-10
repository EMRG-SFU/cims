"""
Linear interpolation of parameter values between the years they are given (#619).

Interpolation only fills years with no value. A constant (blank Year) value applies to every year
not given explicitly, as for any other parameter, so a parameter with a constant is never
interpolated.
"""
from numbers import Real

from numpy import interp

from . import list as PARAM
from .construction import create_value_dict


def interpolate_between(anchors, years):
    """
    Linearly interpolate values for `years` that fall strictly between two anchor years.

    Parameters
    ----------
    anchors : dict {int: float}
        Years with a given value, and that value.
    years : iterable of int
        Years to interpolate for.

    Returns
    -------
    dict {int: float}
        The interpolated value for each year in `years` that isn't an anchor and lies between the
        first and last anchor. Years outside that range are left out.
    """
    if not anchors:
        return {}
    anchor_years = sorted(anchors)
    targets = [year for year in years
               if anchor_years[0] < year < anchor_years[-1] and year not in anchors]
    values = interp(targets, anchor_years, [anchors[year] for year in anchor_years])
    return dict(zip(targets, values.tolist()))


def _is_anchor(value_dict):
    """A numeric value from the model description."""
    if not isinstance(value_dict, dict):
        return False
    val = value_dict.get(PARAM.year_value)
    return (value_dict.get(PARAM.param_source) == 'model'
            and isinstance(val, Real) and not isinstance(val, bool))


def interpolate_parameter(year_dicts, param):
    """
    Interpolate `param` between the years it is given, in place, filling only years with no value.

    Earlier interpolated values are removed first, so re-running after a scenario overlay
    recalculates them from the current values. Only plain value dicts are interpolated; values
    keyed by context or target are left as they are.

    Parameters
    ----------
    year_dicts : dict {str: dict}
        The year dicts of one node or technology, keyed by year. Years where the node or
        technology has no year dict can't be filled, and are left out.
    param : str
        The parameter to interpolate.
    """
    for year_dict in year_dicts.values():
        value_dict = year_dict.get(param)
        if isinstance(value_dict, dict) and value_dict.get(PARAM.param_source) == 'interpolation':
            del year_dict[param]

    anchors = {int(year): year_dict[param]
               for year, year_dict in year_dicts.items() if _is_anchor(year_dict.get(param))}
    empty_years = [int(year) for year, year_dict in year_dicts.items() if param not in year_dict]
    values = interpolate_between({year: d[PARAM.year_value] for year, d in anchors.items()},
                                 empty_years)
    for year, val in values.items():
        # Take the unit from the nearest given year before this one
        unit = anchors[max(y for y in anchors if y < year)][PARAM.unit]
        year_dicts[str(year)][param] = create_value_dict(val, unit=unit,
                                                         param_source='interpolation')
