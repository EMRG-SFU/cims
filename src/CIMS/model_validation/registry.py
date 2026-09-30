"""
CIMS ModelValidator – Central Check Registry

This module defines:
- Data structures for describing validation checks (`CheckSpec`)
- The global registry (`REGISTRY`) of all available checks
- Helper functions for resolving runtime arguments (`resolve_kwargs`)

Purpose
-------
Provides a central, declarative place to register validation checks.
Each check is defined elsewhere (e.g., in `file_checks.py`) and added here
using a consistent pattern that specifies its phase, severity, and inputs.

Design notes
------------
- `resolve_kwargs()` injects `validator` automatically into every check call.
- `argmap` is only for additional inputs (e.g., providers, requested) that are
  resolved from keys in the per-run `context` dict passed from ModelValidator.

Example
-------
To register a new FILE-phase error check from `file_checks.py`:

    REGISTRY.register(
        "my_new_check",
        CheckSpec(
            fn=file_warnings.my_new_check,
            phase=Phase.FILE,
            severity=Severity.ERROR,
            argmap={},  # validator is provided automatically
        ),
    )
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional
from . import file_errors
from . import file_warnings

# ---------------------------------------------------------------------------
# Core registry tags
# ---------------------------------------------------------------------------
class Phase:
    """Validation phase tags."""
    FILE = "file"
    GRAPH = "graph"

class Severity:
    """Severity levels for validation checks."""
    ERROR = "error"
    WARNING = "warning"

# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------
@dataclass
class CheckSpec:
    fn: Callable
    phase: str          # "file" or "graph"
    severity: str       # "error" or "warning"
    argmap: Dict[str, str]
    short_desc: str = ""  # one-line summary shown in explain() summary table
    help_text: str = ""   # output format + interpretation, shown by validator.explain(key)


class CheckRegistry:
    def __init__(self) -> None:
        # name -> spec
        self._checks: Dict[str, "CheckSpec"] = {}

    def register(self, name: str, spec: "CheckSpec") -> None:
        if name in self._checks:
            raise ValueError(f"Duplicate check name: {name}")
        self._checks[name] = spec

    def get(self, name: str) -> "CheckSpec":
        return self._checks[name]

    def iter(self, *, phase: Optional[str] = None, severity: Optional[str] = None):
        """Iterate (name, spec) with optional filters; keyword-only for clarity."""
        for name, spec in self._checks.items():
            if (phase is None or spec.phase == phase) and (severity is None or spec.severity == severity):
                yield name, spec

# ---------------------------------------------------------------------------
# Argument resolution helper
# ---------------------------------------------------------------------------
def resolve_kwargs(validator, argmap: Dict[str, str], context: Dict[str, object]) -> Dict[str, object]:
    """
    Build kwargs for a check function.

    Every check always gets 'validator'.
    Extra parameters listed in argmap are resolved from either:
      - validator.<name>  if it exists as an attribute, otherwise
      - context[<name>]   if present there.
    """
    out: Dict[str, object] = {"validator": validator}
    for param, src in argmap.items():
        if hasattr(validator, src):
            out[param] = getattr(validator, src)
        else:
            out[param] = context.get(src)
    return out

# ---------------------------------------------------------------------------
# Global registry instance
# ---------------------------------------------------------------------------
REGISTRY = CheckRegistry()


# ---------------------------------------------------------------------------
# FILE-PHASE CHECKS
# ---------------------------------------------------------------------------
# All of these are run by ModelValidator.file_warningss() via:
#   REGISTRY.iter(phase=Phase.FILE, severity=...)
#
# Errors first, then warnings.
# ---------------------------------------------------------------------------

# ---- Errors ----
REGISTRY.register("invalid_competition_type", CheckSpec(
    fn=file_errors.invalid_competition_type,
    phase=Phase.FILE,
    severity=Severity.ERROR,
    argmap={},
    short_desc="nodes have an invalid or missing competition type",
    help_text=(
        "Output: [(row_index, node)]\n"
        "  Each entry is a node whose competition parameter is blank or not in the\n"
        "  valid competition types list; row_index is that node's competition row\n"
        "  in model_df, node is the branch name.\n\n"
        "Reading: check the Value column at the flagged row to see what value was\n"
        "set. This check is skipped entirely when no list CSV is provided."
    ),
))
REGISTRY.register("nodes_requesting_self", CheckSpec(
    fn=file_errors.nodes_requesting_self,
    phase=Phase.FILE,
    severity=Severity.ERROR,
    argmap={},
    short_desc="nodes have a service_request targeting themselves",
    help_text=(
        "Output: [(row_index, node)]\n"
        "  Each entry is a node with a service_request row pointing to itself as\n"
        "  the target; row_index is that row in model_df.\n\n"
        "Reading: a node cannot request a service from itself. Correct the Target\n"
        "column on the flagged row to point to the intended provider node."
    ),
))
REGISTRY.register("supply_without_lcc_or_price", CheckSpec(
    fn=file_errors.supply_without_lcc_or_price,
    phase=Phase.FILE,
    severity=Severity.ERROR,
    argmap={},
    short_desc="supply nodes are missing a price in the base year",
    help_text=(
        "Output: [(row_index, node)]\n"
        "  Each entry is a supply node (is_supply = True) with no lcc_financial,\n"
        "  price, or cost_curve_price value in the base year; row_index is a\n"
        "  representative row for that node in model_df.\n\n"
        "Reading: supply nodes require at least one price parameter in the base\n"
        "year. Add the missing price row or check whether the base year value\n"
        "is blank rather than absent."
    ),
))
REGISTRY.register("lcc_at_tech_node", CheckSpec(
    fn=file_errors.lcc_at_tech_node,
    phase=Phase.FILE,
    severity=Severity.ERROR,
    argmap={},
    short_desc="tech compete nodes have LCC defined at node level (should be at tech level)",
    help_text=(
        "Output: [(row_index, node)]\n"
        "  Each entry is a tech-compete node with an LCC parameter on a node-level\n"
        "  row (no Technology value); row_index is that row in model_df.\n\n"
        "Reading: LCC is computed endogenously and must not be set exogenously\n"
        "at any level. Remove the flagged LCC row."
    ),
))
REGISTRY.register("lcc_at_tech", CheckSpec(
    fn=file_errors.lcc_at_tech,
    phase=Phase.FILE,
    severity=Severity.ERROR,
    argmap={},
    short_desc="technologies have LCC defined exogenously (LCC is computed, not set)",
    help_text=(
        "Output: [(row_index, node)]\n"
        "  Each entry is a node with an LCC parameter on a technology-level row;\n"
        "  row_index is a representative row per (node, technology) pair.\n\n"
        "Reading: LCC is computed endogenously and must not be set exogenously\n"
        "on any technology. Remove the flagged LCC rows."
    ),
))
REGISTRY.register("nodes_with_zero_output", CheckSpec(
    fn=file_errors.nodes_with_zero_output,
    phase=Phase.FILE,
    severity=Severity.ERROR,
    argmap={},
    short_desc="nodes or technologies have an Output value of 0",
    help_text=(
        "Output: [(row_index, node, years)]\n"
        "  Each entry is a node or technology with at least one Output value of 0;\n"
        "  years is the list of years where Output is 0, row_index is a\n"
        "  representative row for that node in model_df.\n\n"
        "Reading: an Output of 0 can cause divide-by-zero errors later in the\n"
        "model run. Verify whether this is intentional (e.g. a phased-out\n"
        "technology) or a data-entry error."
    ),
))
REGISTRY.register("undefined_nodes", CheckSpec(
    fn=file_errors.undefined_nodes,
    phase=Phase.FILE,
    severity=Severity.ERROR,
    argmap={"providers": "providers", "requested": "requested"},
    short_desc="nodes referenced in service_requests but not defined in the model",
    help_text=(
        "Output: {node: [row_indexes]}\n"
        "  Each key is an undefined node name — referenced by a service_request\n"
        "  but not defined anywhere in the model.\n"
        "  Each value is the list of row indexes in model_df where it is requested.\n\n"
        "Reading: the row indexes point to service_request rows whose Target column\n"
        "names this undefined node. Check whether the name is a misspelling of an\n"
        "existing node, or whether a service_provide row for it is genuinely missing\n"
        "from the model."
    ),
))
REGISTRY.register("inconsistent_tech_refs", CheckSpec(
    fn=file_errors.inconsistent_tech_refs,
    phase=Phase.FILE,
    severity=Severity.ERROR,
    argmap={},
    short_desc="tech names in data rows don't match the node's technology declarations",
    help_text=(
        "Output: {node: {tech_name: [row_indexes]}}\n"
        "  Each outer key is a node that has the problem.\n"
        "  Each inner key is a technology name that appears in the Technology column\n"
        "  of data rows at that node but has no corresponding Parameter='technology'\n"
        "  declaration row.\n"
        "  Each value is the list of row indexes in model_df where that technology\n"
        "  name appears.\n\n"
        "Reading: the most common cause is a row being copied in with the Technology\n"
        "column not updated — the value still refers to a technology from the source\n"
        "node. To find the declared technologies at that node, filter model_df for\n"
        "Parameter='technology' rows at that node and compare against the inner key."
    ),
))
REGISTRY.register("tech_compete_nodes_no_techs", CheckSpec(
    fn=file_errors.tech_compete_nodes_no_techs,
    phase=Phase.FILE,
    severity=Severity.ERROR,
    argmap={},
    short_desc="tech compete nodes have no declared technologies",
    help_text=(
        "Output: [(row_index, node)]\n"
        "  Each entry is a tech-compete node (competition = compete) with no\n"
        "  Parameter='technology' declaration rows; row_index is the competition\n"
        "  row for that node in model_df.\n\n"
        "Reading: a tech-compete node must declare at least one technology. Either\n"
        "add technology declaration rows or correct the competition type if this\n"
        "node was not intended to compete by technology."
    ),
))
REGISTRY.register("techs_no_base_market_share", CheckSpec(
    fn=file_errors.techs_no_base_market_share,
    phase=Phase.FILE,
    severity=Severity.ERROR,
    argmap={},
    short_desc="technologies are missing a base year market share",
    help_text=(
        "Output: [(row_index, node, tech)]\n"
        "  Each entry is a technology with market_share_new values defined for\n"
        "  some years but not the base year; row_index is a representative row\n"
        "  for that (node, technology) in model_df. The base year is shown in\n"
        "  the printed count line.\n\n"
        "Reading: a base year market share is required for model initialisation.\n"
        "Add a market_share_new row for the base year, or verify that the base\n"
        "year is correct."
    ),
))
REGISTRY.register("service_req_at_tech_node", CheckSpec(
    fn=file_errors.service_req_at_tech_node,
    phase=Phase.FILE,
    severity=Severity.ERROR,
    argmap={},
    short_desc="tech compete nodes have a node-level service_request (should be at tech level)",
    help_text=(
        "Output: [(row_index, node)]\n"
        "  Each entry is a tech-compete node with a service_request row where\n"
        "  the Technology column is blank; row_index is that row in model_df.\n\n"
        "Reading: at a tech-compete node, every service_request row must have a\n"
        "Technology value — the model uses this to allocate demand across\n"
        "competing technologies. A blank Technology value means the request\n"
        "applies to the node as a whole, which is invalid here. Fill in the\n"
        "Technology column on the flagged row, or move the service_request to\n"
        "the appropriate technology rows."
    ),
))
REGISTRY.register("revenue_recycling_at_techs", CheckSpec(
    fn=file_errors.revenue_recycling_at_techs,
    phase=Phase.FILE,
    severity=Severity.ERROR,
    argmap={},
    short_desc="revenue recycling defined at tech level (should only appear at node level)",
    help_text=(
        "Output: [(row_index, node, tech)]\n"
        "  Each entry is a technology with a revenue_recycled parameter row;\n"
        "  row_index is that row in model_df.\n\n"
        "Reading: revenue recycling is applied at the node level only. Remove\n"
        "the revenue_recycled row from the technology and add it to the node."
    ),
))
REGISTRY.register("both_cop_p2000_defined", CheckSpec(
    fn=file_errors.both_cop_p2000_defined,
    phase=Phase.FILE,
    severity=Severity.ERROR,
    argmap={},
    short_desc="nodes have both COP and P2000 exogenously defined (only one allowed)",
    help_text=(
        "Output: [(row_index, node)]\n"
        "  Each entry is a node with both COP and P2000 values defined;\n"
        "  row_index is a representative COP or P2000 row for that node.\n\n"
        "Reading: COP and P2000 are mutually exclusive — only one may be\n"
        "exogenously specified. Remove whichever is not applicable for this node."
    ),
))
REGISTRY.register("min_max_conflicts", CheckSpec(
    fn=file_errors.min_max_conflicts,
    phase=Phase.FILE,
    severity=Severity.ERROR,
    argmap={},
    short_desc="market share min/max limits conflict (min > max)",
    help_text=(
        "Output: [(node, tech, years)]\n"
        "  Each entry is a (node, technology) pair where market_share_new_min\n"
        "  exceeds market_share_new_max; years is the list of conflicting years.\n\n"
        "Reading: correct the min or max value for the listed years so that\n"
        "min ≤ max at every year."
    ),
))
REGISTRY.register("new_techs_in_scenario", CheckSpec(
    fn=file_errors.new_techs_in_scenario,
    phase=Phase.FILE,
    severity=Severity.ERROR,
    argmap={},
    short_desc="new technologies in scenario files are missing a technology declaration",
    help_text=(
        "Output: [(node, tech)]\n"
        "  Each entry is a (node, technology) pair that appears in scenario files\n"
        "  with data rows but no Parameter='technology' declaration, and was not\n"
        "  present in the base files.\n\n"
        "Reading: possible fixes — (1) add a Parameter='technology' declaration\n"
        "row to the scenario file, (2) add the technology to the base file instead,\n"
        "or (3) check for a misspelling if the technology was intended to exist."
    ),
))
REGISTRY.register("base_year_market_share_not_one", CheckSpec(
    fn=file_errors.base_year_market_share_not_one,
    phase=Phase.FILE,
    severity=Severity.ERROR,
    argmap={},
    short_desc="base year market shares don't sum to 1",
    help_text=(
        "Output: [(row_index, node, total_share)]\n"
        "  Each entry is a node whose technology market shares don't sum to 1.0\n"
        "  in the base year; total_share is the actual sum.\n\n"
        "Reading: total_share < 1 means shares are missing across technologies;\n"
        "total_share > 1 suggests double-counting or overlapping shares. The\n"
        "base year is inferred as the minimum Year value in the data."
    ),
))
REGISTRY.register("nodes_missing_service_provide", CheckSpec(
    fn=file_errors.nodes_missing_service_provide,
    phase=Phase.FILE,
    severity=Severity.ERROR,
    argmap={},
    short_desc="nodes are missing a Service Provide parameter",
    help_text=(
        "Output: [(row_index, node)]\n"
        "  Each entry is a node with no Parameter='service_provide' row;\n"
        "  row_index is a representative row for that node in model_df.\n\n"
        "Reading: every node must declare what service it provides. Either add\n"
        "a service_provide row, or check whether the node name in the Branch\n"
        "column is misspelled (which would prevent the service_provide row from\n"
        "being matched to this node)."
    ),
))
REGISTRY.register("nodes_missing_competition", CheckSpec(
    fn=file_errors.nodes_missing_competition,
    phase=Phase.FILE,
    severity=Severity.ERROR,
    argmap={},
    short_desc="nodes are missing a Competition parameter",
    help_text=(
        "Output: [(row_index, node)]\n"
        "  Each entry is a node with no Parameter='competition' row;\n"
        "  row_index is a representative row for that node in model_df.\n\n"
        "Reading: every node must declare a competition type. Add a competition\n"
        "row with the appropriate type."
    ),
))

REGISTRY.register("currency_table_coverage", CheckSpec(
    fn=file_errors.currency_table_coverage,
    phase=Phase.FILE,
    severity=Severity.ERROR,
    argmap={},
    short_desc="monetary units in the data cannot be converted — year or currency missing from deflator/exchange tables",
    help_text=(
        "Output: [\"SRC_CURRENCY SRC_YEAR → TARGET_CURRENCY TARGET_YEAR: <reason>\"]\n"
        "  Each entry is a unique conversion that failed; reason is the missing key\n"
        "  (e.g. 'Year 1800 not found in deflator table for CAD (available: 1995-2024)').\n\n"
        "Reading: open the deflator CSV and confirm the target dollar-year row exists\n"
        "for every currency present in the data. Open the exchange CSV and confirm the\n"
        "required cross-currency rate exists for the target year. This check is skipped\n"
        "when no target_units are configured."
    ),
))

REGISTRY.register("no_structural_parent_node_exists", CheckSpec(
    fn=file_errors.no_structural_parent_node_exists,
    phase=Phase.FILE,
    severity=Severity.ERROR,
    argmap={},
    short_desc="nodes whose structural parent is not defined in the model",
    help_text=(
        "Output: {missing_parent: [child_nodes]}\n"
        "  Each key is a parent node that is not defined in the model.\n"
        "  Each value is the list of child nodes that require it.\n\n"
        "Reading: e.g. CIMS.CAN.AB requires CIMS.CAN to also be defined. A\n"
        "parent with many children likely needs to be added to the model. A\n"
        "parent with only one child may indicate a misspelling in that child's\n"
        "Branch name."
    ),
))


# ---- Warnings ----
REGISTRY.register("missing_parameter_default", CheckSpec(
    fn=file_warnings.missing_parameter_default,
    phase=Phase.FILE,
    severity=Severity.WARNING,
    argmap={},
    short_desc="parameters used in the model have no entry in the defaults file",
    help_text=(
        "Output: [(parameter, count)]\n"
        "  Each entry is a parameter name used in the model with no entry in the\n"
        "  defaults file; count is the number of rows in model_df that use it.\n\n"
        "Reading: parameters without a default may behave unexpectedly if not\n"
        "explicitly set everywhere they are needed. Add the parameter to the\n"
        "defaults file or confirm it is always set explicitly in the model data.\n"
        "Higher counts indicate more widely used parameters — prioritise those."
    ),
))
REGISTRY.register("unrequested_nodes", CheckSpec(
    fn=file_warnings.unrequested_nodes,
    phase=Phase.FILE,
    severity=Severity.WARNING,
    argmap={"providers": "providers", "requested": "requested"},
    short_desc="nodes are defined in the model but never requested by any other node",
    help_text=(
        "Output: [(row_index, node)]\n"
        "  Each entry is a non-root node that is defined in the model but never\n"
        "  appears as the target of any service_request row; row_index is a\n"
        "  representative row for that node in model_df.\n\n"
        "Reading: this node provides a service that nothing consumes. It may be\n"
        "a recently disconnected node or a node from an incomplete model file."
    ),
))
REGISTRY.register("nodes_no_requested_service", CheckSpec(
    fn=file_warnings.nodes_no_requested_service,
    phase=Phase.FILE,
    severity=Severity.WARNING,
    argmap={},
    short_desc="nodes or technologies make no service requests",
    help_text=(
        "Output: [(row_index, node, tech)]\n"
        "  Each entry is a node or technology with no service_request rows;\n"
        "  tech is None for node-level entries and the technology name otherwise.\n"
        "  row_index is a representative row for that node or (node, tech) pair.\n\n"
        "Reading: every node and technology should consume at least one service.\n"
        "A missing service_request may mean a row was accidentally deleted or\n"
        "the node/technology is incomplete."
    ),
))
REGISTRY.register("duplicate_service_requests", CheckSpec(
    fn=file_warnings.duplicate_service_requests,
    phase=Phase.FILE,
    severity=Severity.WARNING,
    argmap={},
    short_desc="same (node, tech, target) service_request rows appear more than once",
    help_text=(
        "Output: {(node, tech, target): {year: count}}\n"
        "  Each key is a (node, technology, target) combination with duplicate\n"
        "  service_request rows. Each value maps year → total row count for that\n"
        "  year (2 = one duplicate, 3 = two duplicates, etc.).\n\n"
        "Reading: uniform count across all years means the same block is duplicated\n"
        "— likely within one file, but possibly across multiple. Elevated count for\n"
        "only some years points to a data-entry error in those years."
    ),
))
REGISTRY.register("bad_service_req", CheckSpec(
    fn=file_warnings.bad_service_req,
    phase=Phase.FILE,
    severity=Severity.WARNING,
    argmap={},
    short_desc="service_request rows are all 0 or blank (node requests nothing)",
    help_text=(
        "Output: [(row_index, node, tech, target)]\n"
        "  Each entry is a (node, technology, target) combination whose\n"
        "  service_request values are all 0 or blank across every year;\n"
        "  row_index is a representative row for that combination in model_df.\n\n"
        "Reading: this particular service request never has a non-zero value.\n"
        "Check whether the values are missing or whether the request row\n"
        "should be removed."
    ),
))
REGISTRY.register("zero_requested_nodes", CheckSpec(
    fn=file_warnings.zero_requested_nodes,
    phase=Phase.FILE,
    severity=Severity.WARNING,
    argmap={"providers": "providers"},
    short_desc="nodes are only requested via zero-valued service_requests (receives no demand)",
    help_text=(
        "Output: [(row_index, node)]\n"
        "  Each entry is a node that appears as a target in service_request rows\n"
        "  but all those requests have a value of 0; row_index is a representative\n"
        "  row for that node in model_df.\n\n"
        "Reading: something is pointing at this node but the demand is always 0.\n"
        "Check whether the requesting nodes' service_request values are\n"
        "intentionally 0 or whether they should be non-zero."
    ),
))
REGISTRY.register("cost_params_missing_currency_unit", CheckSpec(
    fn=file_warnings.cost_params_missing_currency_unit,
    phase=Phase.FILE,
    severity=Severity.WARNING,
    argmap={},
    short_desc="rows with a '$' unit are missing a YYYY_CCC monetary year prefix",
    help_text=(
        "Output: [(row_index, node, parameter, unit)]\n"
        "  Each entry is a row whose Unit contains '$' but lacks a YYYY_CCC prefix;\n"
        "  row_index is that row in model_df.\n\n"
        "Reading: currency conversion uses the year prefix to identify which rows to\n"
        "convert. Without it those values are silently skipped. Update the Unit column\n"
        "to include the dollar-year and currency code\n"
        "(e.g. '$/GJ' → '2010_CAD/GJ', '$' → '2010_CAD').\n\n"
        "This check is skipped when no target_units are configured."
    ),
))

# ---------------------------------------------------------------------------
# GRAPH-PHASE CHECKS (to be added)
# ---------------------------------------------------------------------------
# All of these are run by ModelValidator.file_warnings() via:
#   REGISTRY.iter(phase=Phase.GRAPH, severity=...)
#
# Errors first, then warnings.
# ---------------------------------------------------------------------------
# ---- Errors ----

# ---- Warnings ----
