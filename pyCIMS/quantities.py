class ProvidedQuantity:
    def __init__(self):
        self.provided_quantities = {}

    def provide_quantity(self, amount, requesting_node, requesting_technology=None):
        node_tech = '{}[{}]'.format(requesting_node, requesting_technology)
        self.provided_quantities[node_tech] = amount

    def get_total_quantity(self):
        total = 0
        for amount in self.provided_quantities.values():
            total += amount
        return total

    def get_quantity_provided_to_node(self, node):
        """
        Find the quantity being provided to a specific node, across all it's technologies
        """
        total_provided_to_node = 0
        for pq in self.provided_quantities:
            pq_node, pq_tech = pq.split('[', 1)
            if pq_node == node:
                total_provided_to_node += self.provided_quantities[pq]
        return total_provided_to_node


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
