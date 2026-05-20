import numpy as np
import pandas as pd
import polars as pl

from .model_reader import ModelReader
from ..utils.model_description import column_list as COL

class ScenarioReader(ModelReader):
    def __init__(self, csv_file_paths, col_list, year_list, sector_list,
                 default_values_csv_path=None, node_col=COL.branch, root_node="CIMS"):
        # TODO: __init__ is now a passthrough — remove once confirmed no callers rely on it.
        # Previously built a custom na_values list to preserve the string "None" when reading
        # scenario CSVs with pandas, which treated "None" as NaN by default.
        # This is no longer needed since we read with polars (infer_schema_length=0).
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
                sheet_df = pl.read_csv(
                    csv_file,
                    use_pyarrow=False,
                    infer_schema_length=0,
                ).with_columns(pl.all().replace({np.nan: None})).to_pandas()
                appended_data.append(sheet_df)

            except ValueError:
                print(f"Warning: Unable to parse scenario csv_path at {csv_file}. Skipping.")

        if not appended_data:
            return pd.DataFrame(columns=list(self.col_list) + ["Year", "Value"])

        model_df = pd.concat(appended_data, ignore_index=True)

        meta_cols = [c for c in model_df.columns if c not in ("Year", "Value") and c in self.col_list]
        year_mask = model_df["Year"].isin(self.year_list) | model_df["Year"].isna()
        return model_df[year_mask][meta_cols + ["Year", "Value"]]