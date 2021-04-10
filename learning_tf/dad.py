import mlflow
import paths

uri = f"file:///{(paths.storage_path / 'mlflow' / 'mlruns').as_posix()}"
print(uri)

mlflow.set_tracking_uri(uri)
tracking_uri = mlflow.get_tracking_uri()
print("Current tracking uri: {}".format(tracking_uri))

# mlflow.set_tracking_uri(uri=uri)
mlflow.set_experiment("Test4")

mlflow.start_run()

mlflow.log_param("Test", 1)

mlflow.end_run()
