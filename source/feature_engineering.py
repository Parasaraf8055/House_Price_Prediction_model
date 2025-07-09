import logging
import pandas as pd
from abc import ABC, abstractmethod
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class FeatureEngineering(ABC):
    @abstractmethod
    def apply_transformation(self,df:pd.DataFrame) -> pd.DataFrame:
        pass

class LogTransformation(FeatureEngineering):
    def __init__(self,features):
        self.features = features
# for handling skewness

    def apply_transformation(self, df:pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Applying log Transformation on {self.features}")
        df_transformed = df.copy()
        for feature in self.features:
            df_transformed[feature] = np.log1p(df_transformed[feature])
        logging.info("log transformation applied")    
        return df_transformed



class StandardScaling(FeatureEngineering):
    def __init__(self,features):
        self.features = features
        self.scaler = StandardScaler()

    def apply_transformation(self, df:pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Applying Standard Scaling on {self.features}")
        df_transformed = df.copy()
        df_transformed[self.features] = self.scaler.fit_transform(df_transformed[self.features])
        logging.info("Standard Scaling applied")
        return df_transformed

class MinMaxScaling(FeatureEngineering):
    def __init__(self,features, feature_range=(0,1)):
        self.features = features
        self.scaler = MinMaxScaler(feature_range=feature_range)


    def apply_transformation(self, df:pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Applying Min Max Scaling on {self.features}")
        df_transformed = df.copy()
        df_transformed[self.features] = self.scaler.fit_transform(df_transformed[self.features])
        logging.info("MinMax Scaling applied")
        return df_transformed


class OneHotEncoding(FeatureEngineering):
    def __init__(self, features):
        self.features = features 
        self.encoder = OneHotEncoder(sparse = False, drop = "first")

    def apply_transformation(self, df:pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Applying One Hot Encoding on {self.features}")
        df_transformed = df.copy()
        encoded_df = pd.DataFrame(
            self.encoder.fit_transform(df[self.features]),
            columns = self.encoder.get_feature_names_out(self.features)
        )
        df_transformed = df_transformed.drop(columns=self.features)

        df_transformed = pd.concat([df_transformed, encoded_df],inplace=True)
        logging.info("onehot encoding is completed")
        return df_transformed


class FeatureEngineer:
    def __init__(self, strategy: FeatureEngineering):
        self.strategy = strategy   

    def set_strategy(self, strategy: FeatureEngineering):
        logging.info(f"switching startegy to {self.strategy}")
        self.strategy = strategy 


    def apply_feature_engineering(self, df:pd.DataFrame) -> pd.DataFrame:
        logging.info("applying feature engineering")
        return self.strategy.apply_transformation(df)


                   