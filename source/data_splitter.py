from abc import ABC, abstractmethod
import pandas as pd
from sklearn.model_selection import train_test_split
import source.data_splitter
import logging


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class DataSplitting(ABC):
    @abstractmethod
    def split_data(self, df:pd.DataFrame, target_column:str):
        pass


class SimpleTrainTestSpliter(DataSplitting):
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state

    def split_data(self, df: pd.DataFrame, target_column:str):
        logging.info("performing simple train test split")
        x = df.drop(columns=[target_column])
        y = df[target_column]

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=self.test_size, random_state=self.random_state)

        logging.info("simple train test split completed")
        return x_train, x_test, y_train, y_test


class DataSplitter:
    def __init__(self, strategy:DataSplitting):
        self.strategy = strategy

    def set_strategy(self, strategy:DataSplitting):
        self.strategy = strategy

    def split(self, df:pd.DataFrame, target_column:str):
        logging.info("splitting data using selected strategy")
        return self.strategy.split_data(df, target_column)                    