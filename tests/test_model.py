import pytest
from pyCIMS.model import Model, load_model
import pyCIMS
import os
import re

class TestModel:
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

    @pytest.fixture(scope="session")
    def set_params_file(self):
        return os.path.dirname(os.path.abspath(__file__)) + '/resources/SetParams_script.csv'

    @pytest.fixture(scope="session")
    def model_file(self):
        return os.path.dirname(os.path.abspath(__file__)) + '/resources/model_file.pkl'

    @pytest.fixture(scope="session")
    def change_log_file(self):
        return os.path.dirname(os.path.abspath(__file__)) + '/resources/change_log.csv'
    
    @pytest.mark.parametrize("val, param, node, year, tech, sub_param, save", 
    [(0.8, 'Market share', 'pyCIMS.Canada.Alberta.Residential.Buildings.Floorspace.Lighting', '2000', 'Incandescent', None, False), 
    (0.7, 'Service requested', 'pyCIMS.Canada.Alberta.Transportation Personal.Passenger Vehicles.Existing', '2000', None, 'Recent Car', False),
    (3.0, 'Price Multiplier', 'pyCIMS.Canada.Alberta.Transportation Personal', '2040', None, 'Heavy Fuel Oil', False)])
    def test_set_param(self, create_model, val, param, node, year, tech, sub_param, save):
        model = create_model
        model.set_param(val, param, node, year, tech, sub_param, save)
        assert model.get_param(param, node, year, tech, sub_param) == val

    @pytest.mark.parametrize("val, param, node_regex, year, tech, sub_param, save",
    [(0.6, 'Heterogeneity', r'.*Pumping\.Precision\.Small$', '2020', None, None, False),
    (1.3, 'Heterogeneity', r'^pyCIMS\.Canada\.Alberta\.Residential\.Buildings\.Floorspace\.Lighting\..*', '2000', None, None, False)])
    def test_set_param_wildcard(self, create_model, val, param, node_regex, year, tech, sub_param, save):
        model = create_model
        model.set_param_wildcard(val, param, node_regex, year)
        for node in model.graph.nodes:
            if re.search(node_regex, node) != None:
                assert model.get_param(param, node, year) == val

    @pytest.mark.parametrize("val, param, node, year, tech, sub_param, val_operator, create_missing, row_index",
    [(50, 'Tax', 'pyCIMS.Canada.Alberta.Residential', '2025', None, 'CO2', '>=', True, None),
    (0.2, 'Market share', 'pyCIMS.Canada.Alberta.Residential.Buildings.Floorspace.Lighting', '2000', 'CFL', None, '==', False, None)])
    def test_set_param_search(self, create_model, val, param, node, year, tech, sub_param, val_operator, create_missing, row_index):
        model = create_model
        model.set_param_search(val, param, node, year, tech, sub_param, val_operator, create_missing, row_index)
        assert eval("model.get_param(param, node, year, tech, sub_param)" + val_operator + "val")
    
    @pytest.mark.parametrize("val, param, node, year, tech, sub_param, row_index", 
    [(50, 'Tax', 'pyCIMS.Canada.Alberta.Ethanol', '2025', None, 'CO2', None),
    (30, 'Tax', 'pyCIMS.Canada.Alberta.Biodiesel', '2050', None, 'CO2', 1),
    (20, 'Tax', 'pyCIMS.Canada.Alberta.Residential', '2030', None, 'CO2', None)])
    def test_create_param(self, create_model, val, param, node, year, tech, sub_param, row_index):
        model = create_model
        model.create_param(val, param, node, year, tech, sub_param, row_index)
        model.get_param(param, node, year, tech, sub_param) # test that this runs without error
        assert True

    def test_set_param_file(self, create_model, set_params_file):
        model = create_model
        model.set_param_file(set_params_file)
        assert len(model.change_history) > 10

    def test_set_param_log(self, create_model, change_log_file):
        if os.path.isfile(change_log_file):
            os.remove(change_log_file)
        model = create_model
        model.set_param_log(change_log_file)
        assert os.path.isfile(change_log_file)
    
    def test_save_model(self, create_model, model_file):
        if os.path.isfile(model_file):
            os.remove(model_file)
        model = create_model
        model.save_model(model_file, False)
        assert os.path.isfile(model_file)

    def test_load_model(self, model_file):
        assert load_model(model_file)    