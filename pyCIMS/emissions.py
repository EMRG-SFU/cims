import copy


class EmissionRates:
    def __init__(self):
        self.net_emission_rates = {}
        self.removed_emission_rates = {}

    def multiply_rates(self, amount):
        net_emission_totals = copy.deepcopy(self.net_emission_rates)
        for source_branch in net_emission_totals:
            for ghg in net_emission_totals[source_branch]:
                for emission_type in net_emission_totals[source_branch][ghg]:
                    net_emission_totals[source_branch][ghg][emission_type] *= amount

        removed_emission_totals = copy.deepcopy(self.removed_emission_rates)
        for source_branch in removed_emission_totals:
            for ghg in removed_emission_totals[source_branch]:
                for emission_type in removed_emission_totals[source_branch][ghg]:
                    removed_emission_totals[source_branch][ghg][emission_type] *= amount

        return net_emission_totals, removed_emission_totals


class Emissions:
    def __init__(self):
        # self.gross_emissions_per_unit = {}
        # self.net_emissions_per_unit = {}
        # self.removed_emissions_per_unit = {}
        #
        # self.number_of_units = 0
        # self.gross_emissions = {}
        self.net_emissions = {}
        self.removed_emissions = {}

    def __add__(self, other):
        result = self

        for source_branch in other.net_emissions:
            for ghg in other.net_emissions[source_branch]:
                for emission_type in other.net_emissions[source_branch][ghg]:
                    amount = other.net_emissions[source_branch][ghg][emission_type]
                    if source_branch not in result.net_emissions:
                        result.net_emissions[source_branch] = {}
                    if ghg not in result.net_emissions[source_branch]:
                        result.net_emissions[source_branch][ghg] = {}
                    if emission_type not in result.net_emissions[source_branch][ghg]:
                        result.net_emissions[source_branch][ghg][emission_type] = amount
                    else:
                        result.net_emissions[source_branch][ghg][emission_type] += amount

        for source_branch in other.removed_emissions:
            for ghg in other.removed_emissions[source_branch]:
                for emission_type in other.removed_emissions[source_branch][ghg]:
                    amount = other.removed_emissions[source_branch][ghg][emission_type]
                    if source_branch not in result.removed_emissions:
                        result.removed_emissions[source_branch] = {}
                    if ghg not in result.removed_emissions[source_branch]:
                        result.removed_emissions[source_branch][ghg] = {}
                    if emission_type not in result.removed_emissions[source_branch][ghg]:
                        result.removed_emissions[source_branch][ghg][emission_type] = amount
                    else:
                        result.removed_emissions[source_branch][ghg][emission_type] += amount

        return result

    # def check_emission_exists(self, ghg, emission_type):
    #     if ghg not in self.gross_emissions_per_unit:
    #         self.gross_emissions_per_unit[ghg] = {}
    #         self.net_emissions_per_unit[ghg] = {}
    #         self.removed_emissions_per_unit[ghg] = {}
    #     if emission_type not in self.gross_emissions_per_unit:
    #         self.gross_emissions_per_unit[ghg][emission_type] = 0
    #         self.net_emissions_per_unit[ghg][emission_type] = 0
    #         self.removed_emissions_per_unit[ghg][emission_type] = 0

    # def add_emissions(self, ghg, emission_type, amount):
    #     self.check_emission_exists(ghg, emission_type)
    #     self.gross_emissions_per_unit[ghg][emission_type] += amount
    #     self.net_emissions_per_unit[ghg][emission_type] += amount
    #
    # def remove_emissions(self, ghg, emission_type, amount):
    #     self.check_emission_exists(ghg, emission_type)
    #     self.net_emissions_per_unit[ghg][emission_type] -= amount
    #     self.removed_emissions_per_unit[ghg][emission_type] += amount

    def calculate_total_emissions(self):
        self.gross_emissions = copy.deepcopy(self.gross_emissions_per_unit)
        self.net_emissions = copy.deepcopy(self.net_emissions_per_unit)
        self.removed_emissions = copy.deepcopy(self.removed_emissions_per_unit)

        for emission_dict in [self.gross_emissions, self.net_emissions, self.removed_emissions]:
            for ghg in emission_dict:
                for emission_type in emission_dict:
                    emission_dict[ghg][emission_type] *= self.number_of_units



def _prep_emissions_to_save(calculated_emissions, model_emissions):
    """
    Transform the nested emissions dictionary used during calc_emission_cost() into a dictionary
    ready to be saved in the model.

    Parameters
    ----------
    calculated_emissions : dict
        The emissions dictionary created in the calc_emission_cost() function.
        e.g. {'pyCIMS.Canada.Alberta.Natural Gas': {'CO2': {'Combustion': 0.2,
                                                            'Process': 0.087}}}

    model_emissions : list or dict
        The dictionary (or list of dictionaries) containing the emissions that already exist in the
        model.

    Returns
    -------
        list or dict :
        A version of the calculated_emissions dictionary that matches the dictionary format used
        throughout the model. e.g. [{'branch': None,
                                     'param_source': 'calculation',
                                     'source': None,
                                     'sub_param': 'Combustion',
                                     'unit': None,
                                     'value': 'CO2',
                                     'year_value': 0.2}]
    """
    if isinstance(model_emissions, dict):
        model_emissions = [model_emissions]

    for branch in calculated_emissions:
        for ghg in calculated_emissions[branch]:
            for emission_type in calculated_emissions[branch][ghg]:
                val = calculated_emissions[branch][ghg][emission_type]
                ghg_type_emissions = [e for e in model_emissions if
                                      (e['value'] == ghg) & (e['sub_param'] == emission_type)]
                model_emissions = [e for e in model_emissions if
                                   not ((e['value'] == ghg) & (e['sub_param'] == emission_type))]
                if len(ghg_type_emissions) < 1:
                    model_emissions.append({'value': ghg,
                                            'source': None,
                                            'branch': None,
                                            'sub_param': emission_type,
                                            'unit': None,
                                            'year_value': val,
                                            'param_source': 'calculation'})
                elif len(ghg_type_emissions) == 1:
                    entry_to_update = ghg_type_emissions[0]
                    entry_to_update['year_value'] += val
                    entry_to_update['param_source'] = 'calculation'
                    model_emissions.append(entry_to_update)
                elif len(ghg_type_emissions) > 1:
                    raise ValueError

    # Return the result
    if len(model_emissions) == 1:
        return model_emissions[0]
    else:
        return model_emissions
