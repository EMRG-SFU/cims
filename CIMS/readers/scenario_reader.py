import pandas as pd
import polars as pl
from .model_reader import ModelReader
from .reader_utils import _bool_as_string, get_node_cols
import copy
import numpy as np

class ScenarioReader(ModelReader):
    def __init__(self, csv_file_paths, col_list, year_list, sector_list,
                 default_values_csv_path=None, node_col="Branch", root_node="CIMS"):
        
        self.na_values = copy.deepcopy(pd._libs.parsers.STR_NA_VALUES)
        self.na_values.remove('None')

        super().__init__(
            csv_file_paths, 
            col_list, 
            year_list, 
            sector_list,
            default_values_csv_path=default_values_csv_path, 
            node_col=node_col, 
            root_node=root_node)
        
    def _get_model_df(self):
        appended_data = []
        for csv_file in self.csv_files:
            try:
                mixed_type_columns = ['Context']
                sheet_df = pl.read_csv(
                    csv_file,
                    skip_rows=1,
                    use_pyarrow=False,
                    infer_schema_length=0
                    ).with_columns(pl.all().replace(
                        {np.nan: None}
                    )).to_pandas()
                appended_data.append(sheet_df)

            except ValueError:
                print(f"Warning: Unable to parse scenario csv_path at {csv_file}. Skipping.")

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