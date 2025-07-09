import Analysis.basic_data_inspection
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class OutlierDetection(ABC):
    @abstractmethod
    def detect(self, df:pd.DataFrame) -> pd.DataFrame:
        pass


class ZScoreOutlierDetection(OutlierDetection):
    def __init__(self, threshold=3):
        self.threshold = threshold

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("detecting outliers with Z Score method")
        z_score = np.abs((df - df.mean()) / df.std())
        outliers = z_score > self.threshold
        logging.info(f"outlier detected with Z score {z_score}")
        return outliers


class IQROutlierDetection(OutlierDetection):
    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("detecting outliers with IQR method")
        q1 = df.quantile(0.25)
        q3 = df.quantile(0.75)

        iqr = q3 - q1
        outliers = (df < (q1 - 1.5*iqr) ) | (df > (q3 + 1.5*iqr))

        logging.info(f"outlier detected with IQR {iqr}")
        return outliers
    

class OutlierDetector:
    def __init__(self, strategy: OutlierDetection):
        self.strategy = strategy

    def set_strategy(self, strategy: OutlierDetection):
        self.strategy = strategy
    
    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.strategy.detect(df)
    

    def handle_outliers(self, df:pd.DataFrame, method="remove", axis=1) -> pd.DataFrame:
        outliers = self.detect(df)
        if method == "remove":
            logging.info("removing outliers from dataset")
            df_cleaned = df[(~outliers).all(axis=1)]

        elif method =="cap":
            logging.info("capping outliers")
            df_cleaned = df.clip(lower=df.quantile(0.01), upper=df.quantile(0.99), axis=axis)

        else:
            logging.warning(f"unknown methos '{method}'. no outlier detected")
            return df

        logging.info("outlier handling completed... ")
        return df_cleaned


    def visualize_outliers(self, df:pd.DataFrame, features:list):
        logging.info(f"Visualizing the outliers removed for features {features}")
        for feature in features:
            plt.figure(figsize=(12,8))
            sns.boxplot(x=df[feature])
            plt.title(f"box plot of feature {feature}")
            plt.show()
        

        