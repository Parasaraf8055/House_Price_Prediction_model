from abc import ABC , abstractmethod

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class MultivariateAnalysis(ABC):

    def analyze(self,df:pd.DataFrame):
        

        self.generate_correlation_heatmap(df)
        self.generate_pairplot(df)

    @abstractmethod
    def generate_correlation_heatmap(self,df:pd.DataFrame):
        pass

    @abstractmethod
    def generate_pairplot(self,df:pd.DataFrame):
        pass


class SimpleMultiveriateAnalysis(MultivariateAnalysis):
    def generate_correlation_heatmap(self, df):

        plt.figure(figsize=(12,8))
        sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="viridis")
        plt.title("correlation heatmap")
        plt.show()


    def generate_pairplot(self,df):

        plt.figure(figsize=(12,8))
        sns.pairplot(df)
        plt.subtitle("pair plot of selected figures",y=1.02)
        plt.show()     


