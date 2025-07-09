import logging
import pandas as pd
from abc import ABC, abstractmethod

from typing import Any
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class ModelBuilding(ABC):
    @abstractmethod
    def build_and_train_model(self, x_train: pd.DataFrame, y_train: pd.Series) -> RegressorMixin:
        # regressormixin - a trained scikit learn model instance
        pass


class LinearRegressionStrategy(ModelBuilding):
    def build_and_train_model(self, x_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        if not isinstance(x_train, pd.DataFrame):
            raise TypeError("x_train must be a pandas DataFrame")

        if not isinstance(y_train, pd.Series):
            raise TypeError("y_train must be a pandas Series")
        
        logging.info("Initializing the Linear regression model")

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LinearRegression()),
        ])

        logging.info("training linear regression model")

        pipeline.fit(x_train, y_train)

        logging.info("model training completed!")

        return pipeline
    

class ModelBuilder:
    def __init__(self,strategy: ModelBuilding):
        self.strategy = strategy

    def set_strategy(self,strategy: ModelBuilding):
        logging.info("switching strategy model building strategy")
        self.strategy = strategy

    def build_and_train_model(self, x_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        return self.strategy.build_and_train_model(x_train, y_train)


