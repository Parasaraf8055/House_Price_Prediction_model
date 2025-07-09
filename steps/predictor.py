import json

import numpy as np
import pandas as pd
from zenml import step
from zenml.integrations.mlflow.services import MLFlowDeploymentService


@step(enable_cache=False)
def predictor(
    service: MLFlowDeploymentService,
    input_data: str,
) -> np.ndarray:
    """Run an inference request against a prediction service.

    Args:
        service (MLFlowDeploymentService): The deployed MLFlow service for prediction.
        input_data (str): The input data as a JSON string.

    Returns:
        np.ndarray: The model's prediction.
    """

    # Start the service (should be a NOP if already started)
    if not service.is_running:
        raise RuntimeError("MLflow prediction service is not running. Please start it manually before running this step.")

    # Load the input data from JSON string
    data = json.loads(input_data)

    # Extract the actual data and expected columns
    data.pop("columns", None)  # Remove 'columns' if it's present
    data.pop("index", None)  # Remove 'index' if it's present

    # Define the columns the model expects
    expected_columns = [
        "Order",
        "PID",
        "MS SubClass",
        "Lot Frontage",
        "Lot Area",
        "Overall Qual",
        "Overall Cond",
        "Year Built",
        "Year Remod/Add",
        "Mas Vnr Area",
        "BsmtFin SF 1",
        "BsmtFin SF 2",
        "Bsmt Unf SF",
        "Total Bsmt SF",
        "1st Flr SF",
        "2nd Flr SF",
        "Low Qual Fin SF",
        "Gr Liv Area",
        "Bsmt Full Bath",
        "Bsmt Half Bath",
        "Full Bath",
        "Half Bath",
        "Bedroom AbvGr",
        "Kitchen AbvGr",
        "TotRms AbvGrd",
        "Fireplaces",
        "Garage Yr Blt",
        "Garage Cars",
        "Garage Area",
        "Wood Deck SF",
        "Open Porch SF",
        "Enclosed Porch",
        "3Ssn Porch",
        "Screen Porch",
        "Pool Area",
        "Misc Val",
        "Mo Sold",
        "Yr Sold",
    ]

    # Convert the data into a DataFrame with the correct columns
    df = pd.DataFrame(data["data"], columns=expected_columns)

    # Convert DataFrame to JSON list for prediction
    data_array = df.to_numpy()


    # Run the prediction
    prediction = service.predict(data_array)

    return prediction