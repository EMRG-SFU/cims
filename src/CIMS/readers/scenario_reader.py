import numpy as np
import pandas as pd
import polars as pl

from .model_reader import ModelReader
from ..utils.model_description import column_list as COL

class ScenarioReader(ModelReader):

    def _get_model_df(self):
        appended_data = []
        for csv_file in self.csv_files:
            try:
                sheet_df = pl.read_csv(
                    csv_file,
                    use_pyarrow=False,
                    infer_schema_length=0,
                ).to_pandas().replace({np.nan: None, "": None})
                appended_data.append(sheet_df)

            except ValueError:
                print(f"Warning: Unable to parse scenario csv_path at {csv_file}. Skipping.")

        if not appended_data:
            return pd.DataFrame(columns=list(self.col_list) + ["Year", "Value"])

        model_df = pd.concat(appended_data, ignore_index=True)

        meta_cols = [c for c in model_df.columns if c not in ("Year", "Value") and c in self.col_list]
        year_mask = model_df["Year"].isin(self.year_list) | model_df["Year"].isna()
        return model_df[year_mask][meta_cols + ["Year", "Value"]]