from ..quantities import DistributedSupply
from .aggregation_utils import find_structural_children, find_req_prov_children
from ..utils.parameter import list as PARAM
from ..utils.parameter.construction import create_value_dict

def aggregate_distributed_supplies(model, node, year):
    """
    We want to aggregate up the structural relationships in the tree. This means there are two
    different locations within the tree we need to think about:

    (1) @ a Node without techs — Find any distributed supply that has been generated at the
        node. Add any distributed supplies from structural children.
    (2) @ a Node with tech — For each tech, find the distributed supply generated at that node. Sum
        up the distributed supplies across all techs.

    When doing sums, there is no need to worry about multiply by weights or service request
    ratios, since each node only has a single structural parent, everything will flow through
    that path.
    """

    node_distributed_supply = DistributedSupply()

    if PARAM.technologies in model.graph.nodes[node][year]:
        # @ a Node with techs
        # Find distributed supply generated at the tech
        for tech in model.graph.nodes[node][year][PARAM.technologies]:
            tech_distributed_supply = DistributedSupply()
            distributed_supplies = _get_direct_distributed_supply(model, node, year, tech)
            for service, amount in distributed_supplies:
                tech_distributed_supply.record_distributed_supply(service, node, amount)
                node_distributed_supply.record_distributed_supply(service, node, amount)
            model.graph.nodes[node][year][PARAM.technologies][tech][PARAM.distributed_supply] = \
                tech_distributed_supply
    else:
        # @ a Node without techs
        node_distributed_supply = DistributedSupply()

        # Find distributed supply generated at the node
        distributed_supplies = _get_direct_distributed_supply(model, node, year)
        for service, amount in distributed_supplies:
            node_distributed_supply.record_distributed_supply(service, node, amount)

    # Find distributed supply from structural children
    structural_children = find_structural_children(model.graph, node)
    for child in structural_children:
        node_distributed_supply += model.get_param(PARAM.distributed_supply, child, year)

    model.graph.nodes[node][year][PARAM.distributed_supply] = \
        create_value_dict(node_distributed_supply, param_source='calculation')


def _get_direct_distributed_supply(model, node, year, tech=None):
    """
    Find the distributed supply originating at the node/tech.

    Parameters
    ----------
    model : CIMS.Model
        The model containing the information of interest.
    node : str
        The node whose distributed supply we are interested in finding.
    year : str
        The year in which we want to find
    tech : str, optional
        Optional. If supplied, the tech whose distributed supply we are finding. Otherwise,
        we will look for distributed supply at the node itself.

    Returns
    -------
    List[(str, float)]
        A list of tuples is returned. Each tuple has two entries. The first corresponds to the
        service being provided through distributed supply. The second is the amount of that
        service being provided.

    """
    children = find_req_prov_children(model.graph, node, year, tech)

    distributed_supply = []

    for child in children:
        child_provided_quantities = model.get_param(PARAM.provided_quantities, child, year=year)

        # Find the quantities provided by child to the node/tech
        if tech is None:
            quantity_provided_to_node_tech = \
                child_provided_quantities.get_quantity_provided_to_node(node)
        else:
            quantity_provided_to_node_tech = \
                child_provided_quantities.get_quantity_provided_to_tech(node, tech)

        if quantity_provided_to_node_tech < 0:
            # Record quantities provided directly to the node/tech from child
            distributed_supply.append((child, -1 * quantity_provided_to_node_tech))

    return distributed_supply

