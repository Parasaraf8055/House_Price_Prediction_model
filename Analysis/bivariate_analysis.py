from abc import ABC, abstractmethod
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class BivariateAnalysis(ABC):
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        pass



class NumericalVsNumericalBivariateAnalysis(BivariateAnalysis):
    def analyze(self, df: pd.DataFrame, feature1:str, feature2:str):

        plt.figure(figsize=(12,8))
        plt.title(f"Bivariate analysis between {feature1} vs {feature2}")
        sns.scatterplot(x=feature1, y=feature2, data=df)
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.show()
        



class CategoricalVsNumericalBivariateAnalysis(BivariateAnalysis):
    def analyze(self, df: pd.DataFrame, feature1:str, feature2:str):

        plt.figure(figsize=(12,8))
        plt.title(f"Bivariate analysis between {feature1} vs {feature2}")
        sns.boxplot(x=feature1, y=feature2, data=df)
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.xticks(rotation=45)
        plt.show()



class BivariateAnalyzer:
    def __init__(self,strategy):
        self.strategy = strategy

    def set_strategy(self,strategy):
        self.strategy = strategy

    def execute_strategy(self, df: pd.DataFrame , feature1:str, feature2:str):
        self.strategy.analyze(df,feature1,feature2)            