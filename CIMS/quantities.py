from typing import Dict, Iterable, Mapping, Tuple, Union
import copy
from .utils.parameter import list as PARAM

Number = Union[int, float]

# TODO: make quantity classes into a superclass with methods on dictionaries

class ProvidedQuantity:
    def __init__(self) -> None:
        self.provided_quantities: Dict[str, Number] = {}

    def provide_quantity(self, 
                         amount: Number, 
                         requesting_node: str, 
                         requesting_technology: str | None =None
                         ) -> None:
        """Record a provided amount for a requesting node/technology pair."""
        node_tech = f"{requesting_node}[{requesting_technology}]"
        self.provided_quantities[node_tech] = amount

    def sum_provided_by_total(self) -> Number:
        """
        Return the total provided amount (floored at zero).

        Note: If distributed supply exceeds the sum of positive provided quantities,
        this total may differ from the raw sum of provided values.
        """
        total: Number = sum(self.provided_quantities.values())
        return total if total > 0 else 0

    def sum_provided_to_node(self, node: str) -> Number:
        """Return the total provided amount to a node across all technologies."""
        total = 0
        for key, amount in self.provided_quantities.items():
            pq_node, _ = key.split('[', 1)
            if pq_node == node:
                total += amount
        return total

    def sum_provided_to_tech(self, node: str, tech: str) -> Number:
        """Return the provided amount to a specific node/technology (0 if none)."""
        node_tech = f"{node}[{tech}]"
        return self.provided_quantities.get(node_tech, 0)

    def calculate_proportion(self, node: str, tech: str | None = None) -> Number:
        """Return the share of non-negative provided units allocated to a node/technology."""
        total_for_target = self.sum_provided_to_node(node) if tech is None else self.sum_provided_to_tech(node, tech)
        non_negative_total = sum(a for a in self.provided_quantities.values() if a > 0)
        return (total_for_target / non_negative_total) if non_negative_total and total_for_target >= 0 else 0

class RequestedQuantity:
    def __init__(self) -> None:
        self.requested_quantities: Dict[str, Dict[str, Number]] = {}

    def record_requested_quantity(self, energy: str, service: str, amount: Number) -> None:
        """Record a requested amount for an energy/service pair."""
        self.requested_quantities.setdefault(energy, {}).setdefault(service, 0)
        self.requested_quantities[energy][service] += amount

    def sum_requested_by_energy_service(self) -> Dict[str, Dict[str, Number]]:
        """Return totals requested, grouped by energy then service (energy -> service -> amount)."""
        quantities: Dict[str, Dict[str, Number]] = {}
        
        for energy, services in self.requested_quantities.items():
            for service, amt in services.items():
                quantities.setdefault(energy, {}).setdefault(service, 0)
                quantities[energy][service] += amt 
                
        return quantities

    def sum_requested_by_service(self) -> Dict[str, Number]:
        """Return totals requested, grouped by service across all energies (service -> amount)."""
        quantities: Dict[str, Number]= {}
        
        for services in self.requested_quantities.values():
            for service, amt in services.items():
                quantities.setdefault(service, 0)
                quantities[service] += amt

        return quantities

    def sum_requested_by_energy(self) -> Dict[str, Number]:
        """Return totals requested, grouped by energy across all services (energy -> amount)."""
        quantities: Dict[str, Number] = {}
        
        for energy, services in self.requested_quantities.items():
            quantities.setdefault(energy, 0)
            for amt in services.values():
                quantities[energy] += amt

        return quantities

    def sum_requested_by_total(self) -> Number:
        """Return the total requested amount across all energies and services."""
        total: Number = 0
        for services in self.requested_quantities.values():
            for amt in services.values():
                total += amt
        return total

class DistributedSupply:
    """
    Class to record distributed supplies in the model.
    Note: Negative service requests are stored as positive distributed supply values.
    """
    def __init__(self) -> None:
        self.distributed_supply: Dict[str, Dict[str, Number]] = {}

    def __add__(self, other: "DistributedSupply") -> "DistributedSupply":
        """Return a new DistributedSupply equal to the elementwise sum of two supplies."""
        result = copy.deepcopy(self)
        
        for supply_node, nodes in other.distributed_supply.items():
            result.distributed_supply.setdefault(supply_node, {})
            for node, amt in nodes.items():
                result.distributed_supply[supply_node].setdefault(node, 0)
                result.distributed_supply[supply_node][node] += amt
 
        return result

    def record_distributed_supply(self, supply_node: str, distributed_supply_node: str, amount: Number) -> None:
        """Record an amount of supply provided by one node to another."""
        self.distributed_supply.setdefault(supply_node, {}).setdefault(distributed_supply_node, 0)
        self.distributed_supply[supply_node][distributed_supply_node] += amount

    def sum_distributed_by_energy(self) -> Dict[str, Number]:
        """Return totals distributed, grouped by energy (energy -> amount)."""
        supply: Dict[str, Number] = {}
        
        for energy, nodes in self.distributed_supply.items():
            supply.setdefault(energy, 0)
            for amt in nodes.values():
                supply[energy] += amt                
        
        return supply

    def sum_distributed_by_total(self) -> Number:
        total: Number = 0
        
        for nodes in self.distributed_supply.values():
            for amt in nodes.values():
                total += amt
                
        return total
