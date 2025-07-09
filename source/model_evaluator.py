import logging
from abc import ABC, abstractmethod

from sklearn.base import RegressorMixin
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score



logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class ModelEvaluation(ABC):
    @abstractmethod
    def evaluate_model(self, x_test: pd.DataFrame, y_test:pd.Series, model=RegressorMixin) -> dict:
        pass
        

class RegressionModelEvaluation(ModelEvaluation):
    def evaluate_model(self, x_test:pd.DataFrame, y_test: pd.Series, model:RegressorMixin) -> dict:
        logging.info("adding constant to test data for intercept")
        

        logging.info("predicting using trained model")
        y_pred = model.predict(x_test)

        logging.info("calculating evaluation matrics")
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        metrics = {"Mean Square Error":mse, "R2 score":r2}

        logging.info(f"model evaluation metrics:{metrics}")
        return metrics


class ModelEvaluator:
    def __init__(self, strategy: ModelEvaluation):
        self.strategy = strategy      

    def set_strategy(self, strategy: ModelEvaluation):
        self.strategy = strategy


    def evaluate(self, x_test: pd.DataFrame, y_test:pd.Series, model:RegressorMixin) -> dict:
        return self.strategy.evaluate_model(x_test,y_test,model)             
