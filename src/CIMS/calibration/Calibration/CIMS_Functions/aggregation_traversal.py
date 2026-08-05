
from CIMS.utils.graph import traversals
from CIMS.utils.graph import query as graph_query
from CIMS.utils.graph import loop_resolution

def aggregation_traversal( model, *args, **kwargs ):


    # ::TODO:: Do we need to separately calculate the set of supply nodes here,
    #          so that it's just operating on the subgraph where we're doing these
    #          traversals?

    local_supply_nodes = graph_query.get_supply_nodes(model.graph)
    
    #  Aha! Incorrect! Supply_nodes and supply_side_nodes are two different things!
    #local_supply_nodes = graph_query.get_supply_side_nodes(calModel.model.graph)
    
    for year in model.years:
        # Requested Quantities
        traversals.bottom_up_traversal(model.graph,
                                       model._aggregate_requested_quantities,
                                       year,
                                       loop_resolution_func = loop_resolution.aggregation_resolution,
                                       supply_nodes = local_supply_nodes)
        
        # Direct Emissions
        traversals.bottom_up_traversal(model.graph,
                                       model._aggregate_direct_emissions,
                                       year,
                                       loop_resolution_func = loop_resolution.aggregation_resolution,
                                       supply_nodes = local_supply_nodes)

        # Cumulative Emissions
        traversals.bottom_up_traversal(model.graph,
                                       model._aggregate_cumulative_emissions,
                                       year,
                                       loop_resolution_func = loop_resolution.aggregation_resolution,
                                       supply_nodes = local_supply_nodes)

        # Distributed Supply
        traversals.bottom_up_traversal(model.graph,
                                       model._aggregate_distributed_supplies,
                                       year)
