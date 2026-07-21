from .helpers import filter_model_data
from .model_reader import ModelReader

class ScenarioReader(ModelReader):

    def _get_model_df(self):
        return filter_model_data(self.csv_files, self.sector_list, self.year_list, self.col_list)