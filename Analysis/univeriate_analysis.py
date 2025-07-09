from abc import ABC, abstractmethod
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class UniveriateAnalysis(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature: str):
        pass


class NumericalUniveriateAnalysis(UniveriateAnalysis):
    def analyze(self, df: pd.DataFrame, feature: str):
        """plotting distribution KDE plot for numerical features"""


        plt.figure(figsize=(12,8))
        plt.title(f"Distribution of {feature}:")
        sns.histplot(df[feature], kde=True, bins=30)
        plt.xlabel(feature)
        plt.ylabel("frequency")
        plt.show()


class CategoricalUniveriateAnalysis(UniveriateAnalysis):
    def analyze(self, df: pd.DataFrame, feature:str):
        plt.figure(figsize=(12,8))
        plt.title(f"Distribution of {feature}:")
        sns.countplot(x=feature, data=df , palette='muted')
        plt.xlabel(feature,rotation=60)
        plt.xticks(rotation=45)
        plt.ylabel("count")
        plt.show()


class UniveriateAnalyzer:
    def __init__(self, strategy: UniveriateAnalysis):
        self.strategy = strategy

    def set_strategy(self, strategy: UniveriateAnalysis):
        self.strategy = strategy

    def execute_strategy(self, df: pd.DataFrame , feature: str):
        self.strategy.analyze(df, feature)                
