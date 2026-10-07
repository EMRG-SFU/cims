
from collections.abc import Mapping, Sequence, Iterable
import re

from CIMS.calibration.list_utils import intersect_sublists, union_of_sublists

# The helpers below live in the shared graph_utils module, where VizServer uses
# them too. They are imported here under the names this module has always used.
from CIMS.calibration.graph_utils import (
    ##################################
    #  Parameter Information

    ## Non-Yearly Node Params
    list_all_node_params as list_nonYearly_nodeParams,

    ## All Years for Node
    get_all_node_years as list_years,

    ## All Techs for Node
    get_all_tech_names_strict as list_techs,

    ## Yearly Node Params
    list_year_node_params_at as list_yearly_nodeParams_atYear,             # At a specific year
    list_year_node_params_union as list_yearly_nodeParams_union,           # The union across all years
    list_year_node_params_intersect as list_yearly_nodeParams_intersect,   # The intersection across all years (most useful)
    list_year_node_params_strict as list_yearly_nodeParams,                # Errors unless the set is the same at each year

    ## Yearly Tech Params
    list_yearly_tech_params_union as list_yearly_techParams_union,
    list_yearly_tech_params_intersect as list_yearly_techParams_intersect,
    list_tech_params_strict as list_yearly_techParams,                     # Errors unless the set is the same at each year

    ##################################
    #  Parameter search
    search_for_param as searchForParam,
    search_for_param_any_years as searchForParam_anyYears,

    ##################################
    #  Parameter Access
    get_tech_param_over_time as getTechParamOverTime,
    get_node_param_over_time as getParamOverTime,
)


def find_nodes_with_parameter(model, paramRE):
    return(
        sorted(
            list(
                set(
                    [a['node'] for a in searchForParam_anyYears(model.graph, paramRE)]
                    )
                )
            )
        )
