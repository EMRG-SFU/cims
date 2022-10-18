import copy

# TODO: make quantity classes into a superclass with methods on dictionaries

class ProvidedQuantity:
    def __init__(self):
        self.provided_quantities = {}

    def provide_quantity(self, amount, requesting_node, requesting_technology=None):
        node_tech = '{}[{}]'.format(requesting_node, requesting_technology)
        self.provided_quantities[node_tech] = amount

    def get_total_quantity(self):
        # Note, the result of get_total_quantity() will not equal the sum across
        # self.provided_quantities values when distributed supply is greater than the sum of
        # positive provided quantities.
        total = 0
        for amount in self.provided_quantities.values():
            total += amount

        if total < 0:
            total = 0

        return total

    def calculate_proportion(self, node, tech=None):
        """
        Find the proportion of non-negative units provided to a particular node/tech combination.
        """
        # Note, the result of get_total_quantity() will not equal the sum across
        # self.provided_quantities values when distributed supply is greater than the sum of
        # positive provided quantities.
        proportion = 0

        if tech is None:
            total_provided_node_tech = self.get_quantity_provided_to_node(node)
        else:
            total_provided_node_tech = self.get_quantity_provided_to_tech(node, tech)

        non_negative_total = 0
        for amount in self.provided_quantities.values():
            if amount > 0:
                non_negative_total += amount

        if total_provided_node_tech >= 0:
            proportion = total_provided_node_tech / non_negative_total

        return proportion

    def get_quantity_provided_to_node(self, node):
        """
        Find the quantity being provided to a specific node, across all it's technologies
        """
        # Note, the result of get_total_quantity() will not equal the sum across
        # self.provided_quantities values when distributed supply is greater than the sum of
        # positive provided quantities.

        total_provided_to_node = 0
        for pq in self.provided_quantities:
            pq_node, pq_tech = pq.split('[', 1)
            if pq_node == node:
                total_provided_to_node += self.provided_quantities[pq]
        return total_provided_to_node

    def get_quantity_provided_to_tech(self, node, tech):
        # Note, the result of get_total_quantity() will not equal the sum across
        # self.provided_quantities values when distributed supply is greater than the sum of
        # positive provided quantities.
        node_tech = '{}[{}]'.format(node, tech)

        if node_tech in self.provided_quantities:
            return self.provided_quantities[node_tech]
        else:
            return 0


class RequestedQuantity:
    def __init__(self):
        self.requested_quantities = {}

    def record_requested_quantity(self, providing_node, child, amount):
        if providing_node in self.requested_quantities:
            if child in self.requested_quantities[providing_node]:
                self.requested_quantities[providing_node][child] += amount
            else:
                self.requested_quantities[providing_node][child] = amount

        else:
            self.requested_quantities[providing_node] = {child: amount}

    def get_total_quantities_requested(self):
        total_quants = {}
        for service in self.requested_quantities:
            total_service = 0
            for child, quantity in self.requested_quantities[service].items():
                total_service += quantity
            total_quants[service] = total_service
        return total_quants

    def sum_requested_quantities(self):
        total_quantity = 0
        for fuel in self.requested_quantities:
            fuel_rq = self.requested_quantities[fuel]
            for source in fuel_rq:
                total_quantity += fuel_rq[source]
        return total_quantity


class DistributedSupply:
    """
    Class to help record distributed supplies in the model.
    Note, negative service request values are recorded as positive Distributed Supply values.
    """
    def __init__(self):
        self.distributed_supply = {}

    def __add__(self, other):
        result = copy.deepcopy(self)
        for fuel in other.distributed_supply:
            if fuel not in result.distributed_supply:
                result.distributed_supply[fuel] = {}
            for node in other.distributed_supply[fuel]:
                if node not in result.distributed_supply[fuel]:
                    result.distributed_supply[fuel][node] = 0
                result.distributed_supply[fuel][node] += other.distributed_supply[fuel][node]
        return result

    def record_distributed_supply(self, fuel, distributed_supply_node, amount):
        """Records amount of fuel supplyed by the distributed_supply_node"""
        if fuel in self.distributed_supply:
            if distributed_supply_node in self.distributed_supply[fuel]:
                self.distributed_supply[fuel][distributed_supply_node] += amount
            else:
                self.distributed_supply[fuel][distributed_supply_node] = amount

        else:
            self.distributed_supply[fuel] = {distributed_supply_node: amount}

    def summarize_distributed_supply(self):
        """
        Summarize the distributed supply across all supplying_nodes, aggregating to the fuel/service
        being provided.
        """
        distributed_supply = {}
        for fuel in self.distributed_supply:
            fuel_distributed_supply = 0
            for child, quantity in self.distributed_supply[fuel].items():
                fuel_distributed_supply += quantity
            distributed_supply[fuel] = fuel_distributed_supply
        return distributed_supply
