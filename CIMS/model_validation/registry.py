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
))
REGISTRY.register("nodes_no_provided_service", CheckSpec(
    fn=file_errors.nodes_no_provided_service,
    phase=Phase.FILE,
    severity=Severity.ERROR,
    argmap={}
))
REGISTRY.register("nodes_requesting_self", CheckSpec(
    fn=file_errors.nodes_requesting_self,
    phase=Phase.FILE,
    severity=Severity.ERROR,
    argmap={}
))
REGISTRY.register("supply_without_lcc_or_price", CheckSpec(
    fn=file_errors.supply_without_lcc_or_price,
    phase=Phase.FILE,
    severity=Severity.ERROR,
    argmap={}
))
REGISTRY.register("lcc_at_tech_node", CheckSpec(
    fn=file_errors.lcc_at_tech_node,
    phase=Phase.FILE,
    severity=Severity.ERROR,
    argmap={}
))
REGISTRY.register("lcc_at_tech", CheckSpec(
    fn=file_errors.lcc_at_tech,
    phase=Phase.FILE,
    severity=Severity.ERROR,
    argmap={}
))
REGISTRY.register("nodes_with_zero_output", CheckSpec(
    fn=file_errors.nodes_with_zero_output,
    phase=Phase.FILE,
    severity=Severity.ERROR,
    argmap={}
))
REGISTRY.register("undefined_nodes", CheckSpec(
    fn=file_errors.undefined_nodes,
    phase=Phase.FILE,
    severity=Severity.ERROR,
    argmap={"providers": "providers", "requested": "requested"}
))
REGISTRY.register("inconsistent_tech_refs", CheckSpec(
    fn=file_errors.inconsistent_tech_refs,
    phase=Phase.FILE,
    severity=Severity.ERROR,
    argmap={}
))
REGISTRY.register("tech_compete_nodes_no_techs", CheckSpec(
    fn=file_errors.tech_compete_nodes_no_techs,
    phase=Phase.FILE,
    severity=Severity.ERROR,
    argmap={}
))
REGISTRY.register("techs_no_base_market_share", CheckSpec(
    fn=file_errors.techs_no_base_market_share,
    phase=Phase.FILE,
    severity=Severity.ERROR,
    argmap={}
))
REGISTRY.register("service_req_at_tech_node", CheckSpec(
    fn=file_errors.service_req_at_tech_node,
    phase=Phase.FILE,
    severity=Severity.ERROR,
    argmap={}
))
REGISTRY.register("revenue_recycling_at_techs", CheckSpec(
    fn=file_errors.revenue_recycling_at_techs,
    phase=Phase.FILE,
    severity=Severity.ERROR,
    argmap={}
))
REGISTRY.register("both_cop_p2000_defined", CheckSpec(
    fn=file_errors.both_cop_p2000_defined,
    phase=Phase.FILE,
    severity=Severity.ERROR,
    argmap={}
))
REGISTRY.register("min_max_conflicts", CheckSpec(
    fn=file_errors.min_max_conflicts,
    phase=Phase.FILE,
    severity=Severity.ERROR,
    argmap={}
))
REGISTRY.register("new_nodes_in_scenario", CheckSpec(
    fn=file_errors.new_nodes_in_scenario,
    phase=Phase.FILE,
    severity=Severity.ERROR,
    argmap={}
))
REGISTRY.register("new_techs_in_scenario", CheckSpec(
    fn=file_errors.new_techs_in_scenario,
    phase=Phase.FILE,
    severity=Severity.ERROR,
    argmap={}
))
REGISTRY.register("base_year_market_share_not_one", CheckSpec(
    fn=file_errors.base_year_market_share_not_one,
    phase=Phase.FILE,
    severity=Severity.ERROR,
    argmap={}
))
REGISTRY.register("nodes_missing_service_provide", CheckSpec(
    fn=file_errors.nodes_missing_service_provide,
    phase=Phase.FILE,
    severity=Severity.ERROR,
    argmap={}
))
REGISTRY.register("nodes_missing_competition", CheckSpec(
    fn=file_errors.nodes_missing_competition,
    phase=Phase.FILE,
    severity=Severity.ERROR,
    argmap={}
))

            
# ---- Warnings ----
REGISTRY.register("missing_parameter_default", CheckSpec(
    fn=file_warnings.missing_parameter_default, 
    phase=Phase.FILE,
    severity=Severity.WARNING,
    argmap={}
))
REGISTRY.register("unrequested_nodes", CheckSpec(
    fn=file_warnings.unrequested_nodes, 
    phase=Phase.FILE, 
    severity=Severity.WARNING, 
    argmap={"providers":"providers", "requested": "requested"}    
))
REGISTRY.register("nodes_no_requested_service", CheckSpec(
    fn = file_warnings.nodes_no_requested_service, 
    phase=Phase.FILE,
    severity=Severity.WARNING,
    argmap={}
))
REGISTRY.register("duplicate_service_requests", CheckSpec(
    fn = file_warnings.duplicate_service_requests, 
    phase=Phase.FILE,
    severity=Severity.WARNING,
    argmap={}
))
REGISTRY.register("bad_service_req", CheckSpec(
    fn = file_warnings.bad_service_req, 
    phase=Phase.FILE,
    severity=Severity.WARNING,
    argmap={}
))
REGISTRY.register("zero_requested_nodes", CheckSpec(
    fn = file_warnings.zero_requested_nodes, 
    phase=Phase.FILE,
    severity=Severity.WARNING,
    argmap={"providers":"providers"}
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
