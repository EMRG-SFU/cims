import pandas as pd
from .model_reader import ModelReader
from .reader_utils import _bool_as_string, get_node_cols
import copy
import numpy as np

class ScenarioReader(ModelReader):
    def __init__(self, infile, sheet_map, col_list, year_list, sector_list,
                 default_values=None, node_col="Branch", root_node="CIMS"):
        
        self.na_values = copy.deepcopy(pd._libs.parsers.STR_NA_VALUES)
        self.na_values.remove('None')

        super().__init__(
            infile, 
            sheet_map, 
            col_list, 
            year_list, 
            sector_list,
            default_values=default_values, 
            node_col=node_col, 
            root_node=root_node)
        
    def _get_model_df(self):
        appended_data = []
        for sheet in self.sheet_map:
            try:
                mixed_type_columns = ['Context']
                sheet_df = pd.read_excel(
                    self.infile,
                    sheet_name=sheet,
                    header=1,
                    na_values=self.na_values,
                    keep_default_na=False,
                    converters={c: _bool_as_string for c in mixed_type_columns},
                    engine=self.excel_engine).replace(
                        {np.nan: None, 'False': False, 'True': True})
                appended_data.append(sheet_df)

            except ValueError:
                print(f"Warning: {sheet} not included in {self.infile}. Sheet was not imported into model.")

        model_df = pd.concat(appended_data, ignore_index=True)  # Add province sheets together and re-index
        model_df.index += 3  # Adjust index to correspond to Excel line numbers
        # (+1: 0 vs 1 origin, +1: header skip, +1: column headers)
        model_df.columns = [str(c) for c in
                            model_df.columns]  # Convert all column names to strings (years were ints)
        n_cols, y_cols = get_node_cols(model_df, self.node_col)  # Find columns, separated year cols from non-year cols
        n_cols = [n_col for n_col in n_cols if n_col in self.col_list]
        y_cols = [y_col for y_col in y_cols if y_col in self.year_list]
        all_cols = n_cols + y_cols

        mdf = model_df.loc[1:, all_cols]  # Create df, drop irrelevant columns & skip first, empty row

        return mdf