import pandas as pd
import logging
from typing import Tuple
from source.model_evaluator import ModelEvaluator, RegressionModelEvaluation
from zenml import step
from sklearn.pipeline import Pipeline
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@step(enable_cache=False)
def model_evaluator_step( trained_model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> Tuple[dict, Optional[float | None]]:
    #Ensure the inputs are of the correct type
    if not isinstance(x_test, pd.DataFrame):
        raise TypeError("X_test must be a pandas DataFrame.")
    if not isinstance(y_test, pd.Series):
        raise TypeError("y_test must be a pandas Series.")

    logging.info("Evaluating model using full pipeline.")

    # Use the full pipeline for evaluation
    evaluator = ModelEvaluator(strategy=RegressionModelEvaluation())

    evaluation_metrics = evaluator.evaluate(
        model=trained_model,
        x_test=x_test,
        y_test=y_test
    )

    #Ensure that the evaluation metrics are returned as a dictionary
    if not isinstance(evaluation_metrics, dict):
        raise ValueError("Evaluation metrics must be returned as a dictionary.")
    mse = evaluation_metrics.get("Mean Squared Error", None)
    return evaluation_metrics, mse



