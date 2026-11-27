import pandas as pd
from typing import List


def build_col_list(list_csv_path: str) -> List[str]:
    """Build the metadata column allowlist from config/model-default/default-list.csv's Columns field."""
    return (
        pd.read_csv(list_csv_path)["Columns"]
        .dropna()
        .astype(str)
        .tolist()
    )
