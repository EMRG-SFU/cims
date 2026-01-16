import pandas as pd
from typing import Iterable, List


def build_col_list(list_csv_path: str, year_list: Iterable) -> List[str]:
    """
    Build the column list from defaults_Lists.csv plus the provided years.
    """
    base_cols = (
        pd.read_csv(list_csv_path)["Columns"]
        .dropna()
        .astype(str)
        .tolist()
    )
    year_cols = [str(y) for y in year_list]
    return base_cols + year_cols
