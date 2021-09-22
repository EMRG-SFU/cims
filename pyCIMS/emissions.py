"""
Module containing the custom Emission and EmissionRates classes used during emission aggregation.
"""
import copy
from . import utils


class EmissionRates:
    """Class for storing the emission rates of technologies. An emission rate is the amount of
       emissions per unit of a technology."""
    def __init__(self, emission_rates=None):
        self.emission_rates = emission_rates if emission_rates is not None else {}

    def multiply_rates(self, amount):
        """
        Multiply, each emission rate in self.emission_rates by amount. Amount will usually be the
        total number of units of a given tech. Then rate * units will give the total emissions for
        that tech.

        Parameters
        ----------
        amount : float or int
            The value to multiply each emission rate by.

        Returns
        -------
        dict :
            Returns a dictionary where each emission rate has been multiplied by amount.
        """
        emission_totals = copy.deepcopy(self.emission_rates)
        for source_branch in emission_totals:
            for ghg in emission_totals[source_branch]:
                for emission_type in emission_totals[source_branch][ghg]:
                    emission_totals[source_branch][ghg][emission_type]['year_value'] *= amount

        return emission_totals

    def summarize_rates(self):
        """
        Sum emission rates across GHG/Emission Type pairs. This removes the differentiation between
        which node an emission rate originated from.

        Returns
        -------
        dict :
            Returns a dictionary containing the summarized version of self.emission_rates. This
            summary dictionary will follow the form {GHG: {Emission Type: rate}}.
        """
        summary_rates = {}
        for source in self.emission_rates:
            for ghg in self.emission_rates[source]:
                if ghg not in summary_rates:
                    summary_rates[ghg] = {}
                for emission_type in self.emission_rates[source][ghg]:
                    if emission_type not in summary_rates[ghg]:
                        summary_rates[ghg][emission_type] = 0
                    summary_rates[ghg][emission_type] += \
                        self.emission_rates[source][ghg][emission_type]['year_value']
        return summary_rates


class Emissions:
    """Class for """
    def __init__(self, emissions=None):
        self.emissions = emissions if emissions is not None else {}

    def __add__(self, other):
        result = copy.deepcopy(self)

        for source_branch in other.emissions:
            if source_branch not in result.emissions:
                result.emissions[source_branch] = {}
            for ghg in other.emissions[source_branch]:
                if ghg not in result.emissions[source_branch]:
                    result.emissions[source_branch][ghg] = {}
                for emission_type in other.emissions[source_branch][ghg]:
                    if emission_type not in result.emissions[source_branch][ghg]:
                        result.emissions[source_branch][ghg][emission_type] = \
                            utils.create_value_dict(0)
                    amount = other.emissions[source_branch][ghg][emission_type]
                    result.emissions[source_branch][ghg][emission_type]['year_value'] += amount['year_value']

        return result

    def __mul__(self, other):
        result = copy.deepcopy(self)
        for source_branch in self.emissions:
            for ghg in self.emissions[source_branch]:
                for emission_type in self.emissions[source_branch][ghg]:
                    result.emissions[source_branch][ghg][emission_type]['year_value'] *= other

        return result

    def summarize_emissions(self):
        """
        Sum emissions across GHG/Emission Type pairs. This removes the  differentiation between
        which node an emission originated from.

        Returns
        -------
        dict :
            Returns a dictionary containing the summarized version of self.emissions. This summary
            dictionary will follow the form {GHG: {Emission Type: X1}}.
        """
        summary_emissions = {}
        for source in self.emissions:
            for ghg in self.emissions[source]:
                if ghg not in summary_emissions:
                    summary_emissions[ghg] = {}
                for emission_type in self.emissions[source][ghg]:
                    if emission_type not in summary_emissions[ghg]:
                        summary_emissions[ghg][emission_type] = 0
                    summary_emissions[ghg][emission_type] += \
                        self.emissions[source][ghg][emission_type]['year_value']

        return summary_emissions
