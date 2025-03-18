import pytest
from pyCIMS.model import Model, load_model
import pyCIMS.utils 
import pyCIMS
import os
import re

class TestUtils:
    @pytest.fixture(scope="session")
    def create_model(self):
        # Model Reader
        test_path = os.path.dirname(os.path.abspath(__file__))
        model_description_file = test_path + '/resources/pyCIMS_model_description_AB_value.xlsb'
        model_reader = pyCIMS.ModelReader(infile=model_description_file,
                                        sheet_map={'model': 'Model', 
                                                   'default_tech': 'Technology_Node templates'},
                                        node_col='Node')

        # Model
        model = pyCIMS.Model(model_reader)
        return model
    
    @pytest.mark.parametrize("new_val, param, node, year, sub_param, save", 
    [(0.7, 'Service requested', 'pyCIMS.Canada.Alberta.Transportation Personal.Passenger Vehicles.Existing', '2000', 'Recent Car', False),
    (3.0, 'Price Multiplier', 'pyCIMS.Canada.Alberta.Transportation Personal', '2040', 'Heavy Fuel Oil', False),
    (50, 'Tax', 'pyCIMS.Canada.Alberta.Transportation Freight', '2040', 'CO2', False),
    (1.3, 'Heterogeneity', 'pyCIMS.Canada.Alberta.Residential.Buildings.Floorspace.Solar Electricity', '2000', None, False)])
    def test_set_node_param(self, create_model, new_val, param, node, year, sub_param, save):
        model = create_model
        pyCIMS.utils.set_node_param(new_val, param, model, node, year, sub_param, save)
        assert model.get_param(param, node, year, sub_param=sub_param) == new_val

    @pytest.mark.parametrize("new_val, param, node, year, tech, sub_param, save", 
    [(0.8, 'Market share', 'pyCIMS.Canada.Alberta.Residential.Buildings.Floorspace.Lighting', '2000', 'Incandescent', None, False),
    (0.2, 'Market share', 'pyCIMS.Canada.Alberta.Residential.Buildings.Floorspace.Lighting', '2000', 'CFL', None, False),])
    def test_set_tech_param(self, create_model, new_val, param, node, year, tech, sub_param, save):
        model = create_model
        pyCIMS.utils.set_tech_param(new_val, param, model, node, year, tech, sub_param, save)
        assert model.get_param(param, node, year, tech, sub_param) == new_val
    