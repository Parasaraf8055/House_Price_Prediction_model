import logging
from abc import ABC, abstractmethod
import pandas as pd
from Analysis.basic_data_inspection import DataInspectionStrategy

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class MissingValuesHandling(ABC):  # reason of ABC is to any subclass which going to inherite this class must implement this method
    @abstractmethod
    def handle(self, df:pd.DataFrame) -> pd.DataFrame:
        pass


class DropMissingValues(MissingValuesHandling):
    def __init__(self, axis=0, thresh=None):
        """ 
        Initialize the dropmissingvalue class with specific parameters

        parameters - 
        axis(init)- 0 to drop rows with missing values, 1 to drop columns with missing values
        thresh(init)- threshold for non-NA values. 

        """
        self.axis = axis
        self.thresh = thresh

    def handle(self, df:pd.DataFrame) -> pd.DataFrame:
        logging.info(f"dropping missing values with axis={self.axis} and thresh={self.thresh}")
        df_cleaned = df.dropna(axis=self.axis, thresh=self.thresh)
        logging.info("missing values dropped successfully")
        return df_cleaned


class FillMissingValues(MissingValuesHandling):
    def __init__(self, method="mean",fill_value=None):
        self.method=method
        self.fill_value=fill_value


    def handle(self,df:pd.DataFrame) -> pd.DataFrame:
        logging.info(f"filling missing values with method={self.method} and fill_value={self.fill_value}")
        df_cleaned = df.copy()
        if self.method=="mean":
            numerical_columns = df_cleaned.select_dtypes(include="number").columns
            df_cleaned[numerical_columns] = df_cleaned[numerical_columns].fillna(
                df[numerical_columns].mean()
            )    

        elif self.method == "median":
            numerical_columns = df_cleaned.select_dtypes(include="number").columns
            df_cleaned[numerical_columns] = df_cleaned[numerical_columns].fillna(
                df[numerical_columns].median()
            )     

        elif self.method == "mode":
            for column in df_cleaned.columns:
                df_cleaned[column].fillna(df[column].mode().iloc[0], inplace=True)

        elif self.method == "constant":
            df_cleaned.fillna(self.fill_value, inplace=True)

        logging.info("missing values filled successfully")
        return df_cleaned


class MissingValuesHandler:
    def __init__(self, strategy: MissingValuesHandling):
        self.strategy = strategy


    def set_strategy(self, strategy: MissingValuesHandling):
        logging.info(f"setting strategy to {strategy}")
        self.strategy = strategy

    def execute_strategy(self, df:pd.DataFrame):
        logging.info("executing the strategy.")
        return self.strategy.handle(df)


    

