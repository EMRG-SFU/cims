import copy
from .utils.parameter import list as PARAM

# TODO: make quantity classes into a superclass with methods on dictionaries

class ProvidedQuantity:
    def __init__(self):
        self.provided_quantities = {}

    def provide_quantity(self, amount, requesting_node, requesting_technology=None):
        node_tech = '{}[{}]'.format(requesting_node, requesting_technology)
        self.provided_quantities[node_tech] = amount

    def sum_provided_by_total(self):
        # Note, the result of sum_provided_by_total() will not equal the sum across
        # self.provided_quantities values when distributed supply is greater than the sum of
        # positive provided quantities.
        total = 0
        for amount in self.provided_quantities.values():
            total += amount

        if total < 0:
            total = 0

        return total

    def sum_provided_to_node(self, node):
        """
        Find the quantity being provided to a specific node, across all it's technologies
        """
        total_provided_to_node = 0
        for pq in self.provided_quantities:
            pq_node, pq_tech = pq.split('[', 1)
            if pq_node == node:
                total_provided_to_node += self.provided_quantities[pq]
        return total_provided_to_node

    def sum_provided_to_tech(self, node, tech):
        node_tech = '{}[{}]'.format(node, tech)

        if node_tech in self.provided_quantities:
            return self.provided_quantities[node_tech]
        else:
            return 0

    def calculate_proportion(self, node, tech=None):
        """
        Find the proportion of non-negative units provided to a particular node/tech combination.
        """
        proportion = 0

        if tech is None:
            total_provided_node_tech = self.sum_provided_to_node(node)
        else:
            total_provided_node_tech = self.sum_provided_to_tech(node, tech)

        non_negative_total = 0
        for amount in self.provided_quantities.values():
            if amount > 0:
                non_negative_total += amount

        if total_provided_node_tech >= 0:
            proportion = total_provided_node_tech / non_negative_total

        return proportion


class RequestedQuantity:
    def __init__(self):
        self.requested_quantities = {}

    def record_requested_quantity(self, energy, service, amount):
        if energy in self.requested_quantities:
            if service in self.requested_quantities[energy]:
                self.requested_quantities[energy][service] += amount
            else:
                self.requested_quantities[energy][service] = amount

        else:
            self.requested_quantities[energy] = {service: amount}

    def sum_requested_by_energy_service(self):
        quantities = {}
        for energy in self.requested_quantities:
            for service in self.requested_quantities[energy]:
                if energy not in quantities:
                    quantities[energy] = {}
                if service not in quantities[energy]:
                    quantities[energy][service] = 0
                quantities[energy][service] += self.requested_quantities[energy][service]
        return quantities

    def sum_requested_by_service(self):
        quantities = {}
        for energy in self.requested_quantities:
            for service in self.requested_quantities[energy]:
                if service not in quantities:
                    quantities[service] = 0
                quantities[service] += self.requested_quantities[energy][service]
        return quantities

    def sum_requested_by_energy(self):
        quantities = {}
        for energy in self.requested_quantities:
            if energy not in quantities:
                quantities[energy] = 0
            for service in self.requested_quantities[energy]:
                quantities[energy] += self.requested_quantities[energy][service]
        return quantities

    def sum_requested_by_total(self):
        quantities = 0
        for energy in self.requested_quantities:
            for service in self.requested_quantities[energy]:
                quantities += self.requested_quantities[energy][service]
        return quantities


class DistributedSupply:
    """
    Class to help record distributed supplies in the model.
    Note, negative service request values are recorded as positive Distributed Supply values.
    """
    def __init__(self):
        self.distributed_supply = {}

    def __add__(self, other):
        result = copy.deepcopy(self)
        for supply_node in other.distributed_supply:
            if supply_node not in result.distributed_supply:
                result.distributed_supply[supply_node] = {}
            for node in other.distributed_supply[supply_node]:
                if node not in result.distributed_supply[supply_node]:
                    result.distributed_supply[supply_node][node] = 0
                result.distributed_supply[supply_node][node] += other.distributed_supply[supply_node][node]
        return result

    def record_distributed_supply(self, supply_node, distributed_supply_node, amount):
        """Records amount of supply provided by the distributed_supply_node"""
        if supply_node in self.distributed_supply:
            if distributed_supply_node in self.distributed_supply[supply_node]:
                self.distributed_supply[supply_node][distributed_supply_node] += amount
            else:
                self.distributed_supply[supply_node][distributed_supply_node] = amount

        else:
            self.distributed_supply[supply_node] = {distributed_supply_node: amount}

    def sum_distributed_by_energy(self):
        supply = {}
        for energy in self.distributed_supply:
            if energy not in supply:
                supply[energy] = 0
            for distributed_supply_node in self.distributed_supply[energy]:
                supply[energy] += self.distributed_supply[energy][distributed_supply_node]
        return supply

    def sum_distributed_by_total(self):
        supply = 0
        for energy in self.distributed_supply:
            for distributed_supply_node in self.distributed_supply[energy]:
                supply += self.distributed_supply[energy][distributed_supply_node]
        return supply