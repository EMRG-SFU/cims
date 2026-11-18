"""
Flow ledger: the physical flows and direct emissions of every node and technology in a year,
recorded once the year's equilibrium and aggregation are done.

The ledger does not change any model result. It is the raw record that reporting conventions
(RESD energy balances, NIR emissions by category, ...) are applied to, so that each convention
is a config over the same flows rather than a different aggregation in the model. See
tech_data/design/reporting-accounting.md, section 4.

Flows (one row per node/technology and commodity):
    direction  role        meaning
    in         use         a positive service request: the tech draws `quantity` of `commodity`
    out        byproduct   a negative service request: the tech supplies `quantity` of `commodity`
                           as a by-product (e.g. electricity from cogeneration, coke oven gas)
    out        product     what the node provides; for a node with technologies, the node's
                           provided quantity x each tech's total market share

Carbon on a flow is `quantity` x the carbon in one unit of the commodity, as t CO2:
`t_co2` for fossil carbon and `t_co2_biomass` for biogenic carbon. `carbon_basis` says where
the per-unit value came from:
    carbon_content   the node's `carbon_content` / `carbon_content_biomass` (either or both set)
    emission_factor  neither is set, so the node's CO2 Combustion emission factors are used:
                     `emissions` (fossil) and `emissions_biomass` (biogenic). For a fuel that is
                     burned these equal its carbon, since inventory factors assume complete
                     oxidation. Process factors are not used: they are net of the carbon stored
                     in products (e.g. feedstocks), so those nodes need `carbon_content`.
    (blank)          no carbon value is known for the commodity

Emissions (one row per node/technology, emission group, source, GHG and emission type) are the
technologies' direct emission totals (emissions_total_direct_{net, avoided, negative, bio}),
keyed as CIMS keys them: source branch -> GHG -> emission type.

Quantities for inputs come from the provided quantities the stock allocation records
(market share x service request x assessed demand); outputs and emissions use the node's
provided quantity, as CIMS's direct-emission totals do. The two differ only where a node's
assessed demand includes exports or a macro-economic multiplier.
"""
from __future__ import annotations

from typing import Dict, List

from .quantities import ProvidedQuantity
from .utils.parameter import list as PARAM

FLOW_COLUMNS = ["year", "node", "technology", "direction", "role", "commodity", "quantity",
                "t_co2", "t_co2_biomass", "carbon_basis"]
EMISSION_COLUMNS = ["year", "node", "technology", "group", "source", "ghg", "emission_type", "value"]

EMISSION_GROUPS = {
    "net": PARAM.emissions_total_direct_net,
    "avoided": PARAM.emissions_total_direct_avoided,
    "negative": PARAM.emissions_total_direct_negative,
    "bio": PARAM.emissions_total_direct_bio,
}


def _year_value(node_year: dict, param: str):
    entry = node_year.get(param)
    if isinstance(entry, dict):
        return entry.get(PARAM.year_value)
    return None


def _split_key(key: str) -> tuple[str, str | None]:
    """ProvidedQuantity keys are 'node[tech]'; tech is 'None' for node-level requests."""
    node, tech = key.split('[', 1)
    tech = tech[:-1]
    return node, (None if tech == 'None' else tech)


def _co2_combustion(node_year: dict, param: str):
    """The CO2 Combustion value of an emission-factor parameter ({ghg: {type: value_dict}})."""
    factors = node_year.get(param)
    if not isinstance(factors, dict):
        return None
    entry = factors.get("CO2", {}).get("Combustion") if isinstance(factors.get("CO2"), dict) else None
    return entry.get(PARAM.year_value) if isinstance(entry, dict) else None


def carbon_per_unit(model, node: str, year: str) -> tuple:
    """(fossil t CO2/unit, biogenic t CO2/unit, basis) for the commodity `node` provides."""
    node_year = model.graph.nodes[node].get(year, {})
    fossil = _year_value(node_year, PARAM.carbon_content)
    bio = _year_value(node_year, PARAM.carbon_content_biomass)
    if fossil is not None or bio is not None:
        return fossil or 0.0, bio or 0.0, "carbon_content"
    fossil = _co2_combustion(node_year, PARAM.emissions)
    bio = _co2_combustion(node_year, PARAM.emissions_biomass)
    if fossil is not None or bio is not None:
        return fossil or 0.0, bio or 0.0, "emission_factor"
    return None, None, None


def build_year(model, year: str) -> Dict[str, List[dict]]:
    """Build the ledger for one year: {'flows': [...], 'emissions': [...]}."""
    graph = model.graph
    flows, emissions = [], []

    per_unit = {}

    def carbon(commodity, qty):
        """Carbon columns for `qty` of `commodity`."""
        if commodity not in per_unit:
            per_unit[commodity] = carbon_per_unit(model, commodity, year)
        fossil, bio, basis = per_unit[commodity]
        if basis is None:
            return {"t_co2": None, "t_co2_biomass": None, "carbon_basis": None}
        return {"t_co2": qty * fossil, "t_co2_biomass": qty * bio, "carbon_basis": basis}

    for node in graph.nodes:
        node_year = graph.nodes[node].get(year)
        if not node_year:
            continue
        pq = _year_value(node_year, PARAM.provided_quantities)
        if not isinstance(pq, ProvidedQuantity):
            continue

        # Requests of this node, seen from the requester: uses and by-products.
        for key, amount in pq.provided_quantities.items():
            if not amount:
                continue
            requester, tech = _split_key(key)
            qty = abs(amount)
            flows.append({"year": year, "node": requester, "technology": tech,
                          "direction": "in" if amount > 0 else "out",
                          "role": "use" if amount > 0 else "byproduct",
                          "commodity": node, "quantity": qty, **carbon(node, qty)})

        # What the node (or each of its technologies) provides.
        total = pq.sum_provided_by_total()
        techs = node_year.get(PARAM.technologies) or {}
        if total > 0:
            if techs:
                for tech in techs:
                    ms = model.get_param(PARAM.market_share_total, node, year, tech=tech) or 0
                    if ms:
                        qty = total * ms
                        flows.append({"year": year, "node": node, "technology": tech,
                                      "direction": "out", "role": "product", "commodity": node,
                                      "quantity": qty, **carbon(node, qty)})
            else:
                flows.append({"year": year, "node": node, "technology": None,
                              "direction": "out", "role": "product", "commodity": node,
                              "quantity": total, **carbon(node, total)})

        # Direct emissions of each technology.
        for tech, tech_data in techs.items():
            for group, param in EMISSION_GROUPS.items():
                em = _year_value(tech_data, param)
                for source, ghgs in (getattr(em, "emissions", None) or {}).items():
                    for ghg, types in ghgs.items():
                        for emission_type, value_dict in types.items():
                            value = value_dict.get(PARAM.year_value) if isinstance(value_dict, dict) else value_dict
                            if value:
                                emissions.append({"year": year, "node": node, "technology": tech,
                                                  "group": group, "source": source, "ghg": ghg,
                                                  "emission_type": emission_type, "value": value})
    return {"flows": flows, "emissions": emissions}


def log_ledger(model, flows_file, emissions_file=None, *, ensure_dir: bool = True):
    """
    Write the model's flow ledger (all years) to CSV.

    Parameters
    ----------
    model : CIMS.Model
        A model that has been run.
    flows_file : str | Path
        Output CSV for the flows.
    emissions_file : str | Path, optional
        Output CSV for the emissions. Defaults to `flows_file` with '_emissions' before the
        extension.

    Returns
    -------
    tuple of pandas.DataFrame : (flows, emissions)
    """
    from pathlib import Path
    import pandas as pd

    ledger = getattr(model, "ledger", None)
    if not ledger:
        raise ValueError("The model has no flow ledger. Run the model first.")
    flows_file = Path(flows_file)
    if emissions_file is None:
        emissions_file = flows_file.with_name(f"{flows_file.stem}_emissions{flows_file.suffix}")
    emissions_file = Path(emissions_file)
    if ensure_dir:
        flows_file.parent.mkdir(parents=True, exist_ok=True)
        emissions_file.parent.mkdir(parents=True, exist_ok=True)

    flows = pd.DataFrame([r for y in ledger.values() for r in y["flows"]], columns=FLOW_COLUMNS)
    emissions = pd.DataFrame([r for y in ledger.values() for r in y["emissions"]],
                             columns=EMISSION_COLUMNS)
    flows.to_csv(flows_file, index=False)
    emissions.to_csv(emissions_file, index=False)
    return flows, emissions
