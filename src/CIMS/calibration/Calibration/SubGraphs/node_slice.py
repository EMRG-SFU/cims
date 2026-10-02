"""
Minimal model slices for calibrating one tech-compete node.

A market-share fit at node N recalculates only N: its LCC and its stock
allocation. Measured on a residential heating node, everything the fit reads
or writes lives in N itself and in the nodes N requests services from
(fuel/supply nodes and other request targets, via `request_provide` edges).
Ancestors are not read at runtime: inheritable parameters were copied down to
N when the model was initialised. So a slice of N plus its request targets
reproduces the full-model fit at a fraction of the memory — ~2.8 MB of node
data instead of 37 MB at a 116-node AB residential model, and far less than
that against a full-sector model.

The one thing that reaches further is declining capital cost (DCC): the cost
of a technology in a DCC class depends on cumulative stock at EVERY member of
that class, across the whole model. During N's fit those other members are
fixed — only N's own stock changes — so their contribution is a constant per
class and year. `build_node_slice` computes it once from the full model and
stores it on the slice as `_dcc_external`; `declining_costs._calc_all_stock`
adds it to the sum over the members that are inside the slice. The result
equals the full-model sum up to floating-point summation order.

Do NOT rebuild `dcc_classes` from a slice's graph (`model._dcc_classes()`),
as the older subgraph helpers in this package do: that silently drops every
class member outside the slice, and DCC then sees a fraction of the real
cumulative stock.

Anything else a fit might read from a node that is not in the slice raises
`KeyError` (graph lookups on a missing node fail), so a slice that is too
small fails loudly rather than producing different numbers.
"""
import copy

import networkx as nx

from CIMS.utils.parameter import list as PARAM


def _request_targets(model, node):
    """Nodes `node` requests services from, by edge type and by service_request keys."""
    graph = model.graph
    targets = {v for v in graph.successors(node)
               if 'request_provide' in (graph.edges[node, v].get('edge') or [])}
    # The fit resolves targets from each technology's `service_request` keys,
    # not from edges — include those too, in case the two ever disagree.
    data = graph.nodes[node]
    for year in model.years:
        year_data = data.get(str(year)) or {}
        for tech_data in (year_data.get(PARAM.technologies) or {}).values():
            requests = tech_data.get('service_request')
            if isinstance(requests, dict):
                targets.update(requests.keys())
        requests = year_data.get('service_request')
        if isinstance(requests, dict):
            targets.update(requests.keys())
    targets.discard(node)
    return targets


def slice_node_names(model, node):
    """The node set a fit at `node` needs: the node and its request targets."""
    return {node} | _request_targets(model, node)


def _dcc_external_stock(model, keep, dcc_classes):
    """
    {dcc_class: {year: stock}} — the cumulative-stock contribution of every
    class member OUTSIDE `keep`, for each model year, computed exactly as
    `declining_costs._calc_all_stock` would (base-year stock_base plus
    stock_new at every calendar year before `year`, mapped onto model years,
    each divided by the member's base-year multiplier_load_factor).
    """
    base_year = int(model.base_year)
    base_str = str(base_year)
    step = model.step
    years = [int(y) for y in model.years]

    external = {}
    for dcc_class, members in dcc_classes.items():
        outside = [(n, t) for n, t in members if n not in keep]
        if not outside:
            continue
        base_sum = 0.0
        new_by_year = {}                     # model year -> sum of stock_new
        for node_k, tech_k in outside:
            unit = model.get_param(PARAM.multiplier_load_factor, node_k, base_str, tech=tech_k)
            bs = model.get_param(PARAM.stock_base, node_k, base_str, tech=tech_k)
            if bs is not None:
                base_sum += bs / unit
            for y in years:
                if y >= years[-1]:
                    continue                 # never needed: only years strictly before are summed
                ns = model.get_param(PARAM.stock_new, node_k, str(y), tech=tech_k)
                new_by_year[y] = new_by_year.get(y, 0.0) + ns / unit

        per_year = {}
        for y in years:
            total = base_sum
            for j in range(base_year, y):
                ref = (j - base_year) // step * step + base_year
                total += new_by_year.get(ref, 0.0)
            per_year[str(y)] = total
        external[dcc_class] = per_year
    return external


def build_node_slice(model, node, drop_attributes=('change_history',)):
    """
    A copy of `model` holding only what a market-share fit at `node` reads.

    The graph is cut to `slice_node_names(model, node)` and its node data is
    deep-copied, so fitting the slice never touches `model`. Model-level
    attributes (defaults, inheritable params, years, supply_nodes, ...) are
    shared by reference except those in `drop_attributes`, which are emptied
    (`change_history` grows with every set_param and is not read by a fit).
    DCC classes are reduced to their in-slice members, and the frozen
    contribution of the rest is stored as `_dcc_external`.

    Intended to be pickled and sent to a worker process. Fitting a slice in
    the process that holds `model` also works, but model-level attributes are
    shared, so do not mutate those.
    """
    keep = slice_node_names(model, node)
    missing = keep - set(model.graph.nodes)
    if missing:
        raise KeyError(f"request targets of {node} not in the model graph: {sorted(missing)}")

    sliced = copy.copy(model)

    graph = nx.DiGraph()
    graph.graph.update(copy.deepcopy(model.graph.graph))
    for n in keep:
        graph.add_node(n, **copy.deepcopy(dict(model.graph.nodes[n])))
    for u, v, data in model.graph.subgraph(keep).edges(data=True):
        graph.add_edge(u, v, **copy.deepcopy(data))
    sliced.graph = graph

    dcc_classes = getattr(model, 'dcc_classes', None) or {}
    sliced.dcc_classes = {c: [(n, t) for n, t in members if n in keep]
                          for c, members in dcc_classes.items()
                          if any(n in keep for n, _ in members)}
    sliced._dcc_external = _dcc_external_stock(
        model, keep, {c: dcc_classes[c] for c in sliced.dcc_classes})
    if hasattr(sliced, '_dcc_cache'):
        sliced._dcc_cache = None

    for attr in drop_attributes:
        if hasattr(sliced, attr):
            value = getattr(model, attr)
            try:
                setattr(sliced, attr, type(value)())
            except TypeError:
                setattr(sliced, attr, None)

    sliced._slice_of = node
    return sliced
