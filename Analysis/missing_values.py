from abc import ABC, abstractmethod
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class MissingValuesInspection(ABC):
    def analyze(self, df:pd.DataFrame):
        self.identify_missing_values(df)
        self.visualize_missing_values(df)


    @abstractmethod
    def identify_missing_values(self, df: pd.DataFrame):
        pass

    @abstractmethod
    def visualize_missing_values(self, df: pd.DataFrame):
        pass


class SimpleMissingValues(MissingValuesInspection):
    def identify_missing_values(self, df):
        print("\n missing values ")

        missing_values = df.isnull().sum()

        print(missing_values[missing_values>0])


    def visualize_missing_values(self, df):
        print("\n visualizing missing values")

        plt.figure(figsize=(10,5))
        sns.heatmap(df.isnull(), cmap='viridis',cbar=False,annot=True)
        plt.title("missing values heatmap")
        plt.show()
        
                        