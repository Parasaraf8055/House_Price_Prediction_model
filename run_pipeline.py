import click
import zenml
from Pipelines.training_pipeline import ml_pipeline
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

@click.command()
def main():
    # run ml pipeline and mlflow ui
    run = ml_pipeline()

    print(
        "now run \n" 
        f"mlflow ui --backend store--'{get_tracking_uri()}'"
        "to inspect experiment run within mlflow ui\n"
    )

if __name__=="__main__":
    main()
