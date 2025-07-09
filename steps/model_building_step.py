import logging
import pandas as pd 
from typing import Annotated

import mlflow
import mlflow.sklearn
from sklearn.base import RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


from zenml import step, ArtifactConfig
from zenml.client import Client

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# get active experiment tracker from zenml 
experiment_tracker = Client().active_stack.experiment_tracker
from zenml import Model

if experiment_tracker is None:
    raise ValueError("Experiment Tracker is not set in the active ZenML stack.")

model = Model(
    name="price_predictor",
    version=None,
    license="apache 2.0",
    description="price prediction model for houses"
)

@step(enable_cache=False, experiment_tracker=experiment_tracker.name, model=model)
def model_building_step(
    x_train: pd.DataFrame,
    y_train: pd.Series
) -> Annotated[Pipeline, ArtifactConfig(name="sklearn-pipeline", is_model_artifact=True)]:
    # this returns a trained sklearn pipeline including preprocessing and model training

    if not isinstance(x_train, pd.DataFrame):
        raise TypeError("x_train must be a pandas DataFrame")

    if not isinstance(y_train, pd.Series):
        raise TypeError("y_train must be a pandas Series")
    
    categorical_col = x_train.select_dtypes(include=["object", "category"]).columns
    numerical_col = x_train.select_dtypes(exclude=["object","category"]).columns

    logging.info(f"categorical columns : {categorical_col.tolist()}")
    logging.info(f"numerical columns : {numerical_col.tolist()}")

    # defining preprocessing for features values
    numerical_tranform = SimpleImputer(strategy="mean")
    categorical_transform = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # bundle preprocessing for both columns
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_tranform, numerical_col),
            ("cat", categorical_transform, categorical_col)
        ]
    )

    # define model training pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor",preprocessor),
        ("model", LinearRegression()),
    ])
    
    if not mlflow.active_run():
        mlflow.start_run() 

    try:

        mlflow.sklearn.autolog()

        logging.info("Building and training the linear regression model")

        pipeline.fit(x_train, y_train)

        mlflow.sklearn.log_model(pipeline, "model")

        # Register the model (optional but recommended for serving)
        run = mlflow.active_run()
        run_id = run.info.run_id
        model_uri = f"runs:/{run_id}/model"
        mlflow.register_model(model_uri, "house_price_predictor")

        logging.info(f"Model registered from {model_uri}")

        logging.info("model training completed")

        onehot_encoder = (
            pipeline.named_steps["preprocessor"].transformers_[1][1].named_steps["onehot"]
        )

        onehot_encoder.fit(x_train[categorical_col])
        expected_columns = numerical_col.tolist() + list(onehot_encoder.get_feature_names_out(categorical_col))

        logging.info(f"model expects the following columns:{expected_columns}")

        
    except Exception as e:
        logging.error(f"error during model training : {e}")
        raise e

    finally:
        # End the MLflow run
        mlflow.end_run()
 

    return pipeline            


   