from ..utils.graph.traversals import find_loops
from ..utils.parameter import list as PARAM


def market_share_total_after_base_year(validator, model):
    """
    Identify technologies with a market_share_total after the base year, in a year where their
    lifetime is greater than 1.

    Stock retirement assumes stock lasts `lifetime` years, so an exogenous total market share after
    the base year only gives consistent stocks when lifetime is 1 (#515). Values are read from the
    constructed graph, so they are the ones the model uses: base and scenario files combined,
    constant (blank Year) values applied, and lifetime interpolated where listed (#619) or
    defaulted.
    """
    later_years = [str(y) for y in model._year_list if int(y) > int(model.base_year)]

    concerns = []
    for node, data in model.graph.nodes(data=True):
        bad_years = {}
        for year in later_years:
            for tech, tech_data in (data.get(year, {}).get(PARAM.technologies) or {}).items():
                market_share = tech_data.get(PARAM.market_share_total)
                given = (isinstance(market_share, dict)
                         and market_share.get(PARAM.param_source) == 'model'
                         and market_share.get(PARAM.year_value) is not None)
                if given and model.get_param(PARAM.lifetime, node, year, tech=tech) > 1:
                    bad_years.setdefault(tech, []).append(year)
        concerns.extend((node, tech, ", ".join(years)) for tech, years in bad_years.items())

    concern_desc = "technologies have a market_share_total after the base year and a lifetime > 1"
    return concerns, concern_desc


def loops_aggregation_cannot_break(validator, model):
    """
    Identify loops that the aggregation step of a run cannot break: loops with no supply node,
    and loops made only of supply nodes.

    After each year's equilibrium, quantities and emissions are aggregated bottom-up. To break a
    loop, loop_resolution.aggregation_resolution removes the loop's supply nodes and picks one of
    the remaining nodes. A loop with no supply node is still a loop after that, and a loop of only
    supply nodes leaves nothing to pick, so either one fails the run with "max() iterable argument
    is empty" (#507).
    """
    supply_nodes = set(model.supply_nodes)

    concerns = []
    for loop in find_loops(model.graph):
        n_supply = sum(node in supply_nodes for node in loop)
        if n_supply == 0:
            concerns.append((loop, "no supply node"))
        elif n_supply == len(loop):
            concerns.append((loop, "only supply nodes"))

    concern_desc = "loops have no supply node or only supply nodes, so a run cannot break them"
    return concerns, concern_desc
