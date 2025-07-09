import pandas as pd
from source.handle_missing_values import DropMissingValues, FillMissingValues, MissingValuesHandler

from zenml import step

@step
def handle_missing_values_step(df:pd.DataFrame, strategy:str = "mean") -> pd.DataFrame:
    if strategy == "drop":
        handler = MissingValuesHandler(DropMissingValues(axis=0))

    elif strategy in ["mean", "mode" ,"median", "constant"]:
        handler = MissingValuesHandler( FillMissingValues(method=strategy))

    else:
        raise ValueError(f"Unsupported missing value handling strategy ")

    cleaned_df = handler.execute_strategy(df)
    return cleaned_df        
