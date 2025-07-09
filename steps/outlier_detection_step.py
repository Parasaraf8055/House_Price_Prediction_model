import logging
import pandas as pd
from source.outlier_detection import OutlierDetector, ZScoreOutlierDetection, IQROutlierDetection
from zenml import step

@step
def outlier_detection_step(df:pd.DataFrame, column_name:str) -> pd.DataFrame:

    logging.info(f"start outlier detection step with dataset {df}")
    if df is None:
        logging.info("No dataset provided")
        raise ValueError("No dataset provided")
    
    if not isinstance(df, pd.DataFrame):
        logging.info(f"expected pandas dataframe but got {type(df)}")
        raise ValueError(f"expected pandas dataframe but got {type(df)}")
    
    if column_name not in df.columns:
        logging.info(f"column {column_name} not found in dataset")
        raise ValueError(f"column {column_name} not found in dataset")

    df_numeric = df.select_dtypes(include=[int, float])

    outlier_detector = OutlierDetector(ZScoreOutlierDetection(threshold=3))
    outliers = outlier_detector.detect(df_numeric)
    df_cleaned = outlier_detector.handle_outliers(df_numeric, method="remove", axis=1)

    return df_cleaned    