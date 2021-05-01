import pickle as pkl

import mlflow

import paths


class DatasetLoader:
    kind_dict = {"nn": "nn_input.pkl", "cnn": "cnn_input.pkl"}

    def __init__(self, run_ids: list, kind: str):
        if kind not in ["nn", "cnn"]:
            raise ValueError

        mlflow.set_tracking_uri(paths.mlflow_path)
        mlflow.set_experiment("Datasets")

        self.run_ids = run_ids

        self.data = []
        self.kind = kind

    def load(self):
        for run_id in self.run_ids:
            from_run = mlflow.get_run(run_id)
            from_artifact_path = paths.artifact_path(from_run.info.artifact_uri)

            with open(from_artifact_path / self.kind_dict[self.kind], "rb") as f:
                self.data.append(pkl.load(f))
        return self.data

    def merge(self):
        self.load()
        flattened = [item for sublist in self.data for item in sublist]
        self.data = flattened
        return self.data


if __name__ == "__main__":
    mlflow.set_tracking_uri(paths.mlflow_path)
    mlflow.set_experiment("Datasets")

    run_ids = ["3bf57f5ad0d94c0f8ab0848438b78808"]
    dl = DatasetLoader(run_ids, "cnn").merge()

    print()
