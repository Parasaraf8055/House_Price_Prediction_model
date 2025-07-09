from zenml import step
from source.feature_engineering import FeatureEngineer, LogTransformation, MinMaxScaler, OneHotEncoder, StandardScaler
import pandas as pd

@step
def feature_engineering_step(df:pd.DataFrame, strategy: str = "log", features: list = None) -> pd.DataFrame:
    if features is None:
        features = []

    if strategy == "log":
        engineer = FeatureEngineer(LogTransformation(features=features))    

    elif strategy == "minmaxscaling":
        engineer = FeatureEngineer(MinMaxScaler(features=features))

    elif strategy == "standardscaling":
        engineer = FeatureEngineer(StandardScaler(features=features))

    elif strategy == "onehotencoding":
        engineer = FeatureEngineer(OneHotEncoder(features=features))

    else:
        raise ValueError(f"unsupported feature engineering strategy {strategy}")

    transformed_df = engineer.apply_feature_engineering(df)
    return transformed_df            