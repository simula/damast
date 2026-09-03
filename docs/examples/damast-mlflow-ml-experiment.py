import mlflow.keras
import yaml

from damast.integrations.mlflow_tracker import track_pipeline
from damast.ml.experiments import Experiment

experiment = Experiment.from_file("experiment.yaml")

mlflow.keras.autolog()  # per-epoch metrics + the trained model, straight from model.fit()

with track_pipeline(experiment.learning_task.pipeline, run_name=experiment.label) as tracker:
    report_path = experiment.run()  # computes features, trains, evaluates, writes the yaml report

    with open(report_path) as f:
        report = yaml.safe_load(f)

    # training_parameters/evaluation aren't part of any AnnotatedDataFrame's metadata, so log
    # them directly rather than through tracker.log_result()
    tracker.log_params({
        f"training.{k}": v for k, v in report["learning_task"]["training_parameters"].items()
    })
    for model_name, results in report["evaluation"].items():
        tracker.log_metrics({f"eval.{model_name}.{k}": v for k, v in results.items()})
    tracker.log_artifact(report_path)
