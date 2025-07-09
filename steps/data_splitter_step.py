from typing import Tuple
import pandas as pd
from source.data_splitter import DataSplitter, SimpleTrainTestSpliter
from zenml import step

@step
def data_splitter_step(df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    splitter = DataSplitter(strategy = SimpleTrainTestSpliter())
    x_train, x_test, y_train, y_test = splitter.split(df, target_column)
    return x_train, x_test, y_train, y_test