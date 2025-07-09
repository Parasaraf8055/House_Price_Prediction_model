#basis inspection about data

from abc import ABC, abstractmethod
import pandas as pd

# defining abstract class for implementing  multiple  strategies with using  strategy pattern

class DataInspectionStrategy(ABC):
    @abstractmethod
    def inspect(self, df: pd.DataFrame):
        """perform specific type of data inspection """
        pass


#type of strategies

class DatatypesInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        print("\n data types and non-null counts")
        print(df.info())


class SummaryStatisticsInspection(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        print("\n summary statistics - numerical columns")
        print(df.describe())
        
        print("\n summary statistics - categorical columns")
        print(df.describe(include=['object']))


class DataInspector:
    def __init__(self, strategy: DataInspectionStrategy):
        self.strategy = strategy

    def set_strategy(self, strategy: DataInspectionStrategy):
        self.strategy = strategy

    def execute_strategy(self, df:pd.DataFrame):
        self.strategy.inspect(df)                
