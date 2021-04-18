import pickle as pkl

import mlflow

import paths


class DatasetLoader:

    def __init__(self, run_ids: list):
        self.run_ids = run_ids

        self.data = []

    def load(self):
        for run_id in self.run_ids:
            from_run = mlflow.get_run(run_id)
            from_artifact_path = paths.artifact_path(from_run.info.artifact_uri)

            with open(from_artifact_path / "timeseries.pkl", "rb") as f:
                self.data.append(pkl.load(f))
        return self.data

    def merge(self):
        flattened = [item for sublist in self.data for item in sublist]
        self.data = flattened
        return self.data


if __name__ == "__main__":
    mlflow.set_tracking_uri(paths.mlflow_path)
    mlflow.set_experiment("Datasets")

    run_ids = ["58d6faa3746b46b7839f62fcb03239ea", "82e8310bd4134c45a071ce6d5175b297"]
    dl = DatasetLoader(run_ids)
    a = dl.load()
    merged = dl.merge()
    print()

