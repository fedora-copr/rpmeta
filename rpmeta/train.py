# TODO: this is just copy paste from my private jupyter notebook, fine tuning and real work is
# happenning there, this is just a proof of concept

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor


class Trainer:
    """
    Trainer class for training the model
    """

    def __init__(
        self,
        dataset_path: str,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> None:
        self._test_size = test_size
        self._random_state = random_state

        if not Path(dataset_path).exists():
            raise FileNotFoundError(f"Dataset file {dataset_path} not found")

        self._df = pd.read_json(dataset_path)

        self._categorical_features = [
            "package_name",
            "epoch",
            "version",
            "release",
            "mock_chroot_name",
            "cpu_model",
            "cpu_arch",
            "cpu_model_name",
        ]
        self._numerical_features = ["cpu_cores", "ram", "swap", "bogomips"]

    def _preprocess_data(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        self._df["build_duration"] = self._df["build_duration"].astype(
            int,
            errors="ignore",
        )

        # drop nulls
        self._df.dropna(subset=["build_duration"], inplace=True)

        # unpick hw info - TODO: what to do with nones? probably drop? we have ton of data so
        self._df["cpu_model"] = self._df["hw_info"].apply(
            lambda x: x["cpu_model"] if x else None,
        )
        self._df["cpu_arch"] = self._df["hw_info"].apply(
            lambda x: x["cpu_arch"] if x else None,
        )
        self._df["cpu_model_name"] = self._df["hw_info"].apply(
            lambda x: x["cpu_model_name"] if x else None,
        )
        self._df["cpu_cores"] = self._df["hw_info"].apply(
            lambda x: x["cpu_cores"] if x else None,
        )
        self._df["ram"] = self._df["hw_info"].apply(lambda x: x["ram"] if x else None)
        self._df["swap"] = self._df["hw_info"].apply(lambda x: x["swap"] if x else None)
        self._df["bogomips"] = self._df["hw_info"].apply(
            lambda x: x["bogomips"] if x else None,
        )

        # get rid of empty string
        self._df["bogomips"] = self._df["hw_info"].apply(
            lambda x: None if not x or not x.get("bogomips") else x["bogomips"],
        )

        # drop these, no longer needed
        self._df.drop(
            columns=["hw_info", "type", "path_to_hw_info"],
            inplace=True,
            errors="ignore",
        )

        # yep so here drop null for time being
        self._df.dropna(
            subset=[
                "cpu_model",
                "cpu_arch",
                "cpu_model_name",
                "cpu_cores",
                "ram",
                "swap",
                "bogomips",
            ],
            inplace=True,
        )

        self._df["cpu_cores"] = self._df["cpu_cores"].astype(int)
        self._df["ram"] = self._df["ram"].astype(int)
        self._df["swap"] = self._df["swap"].astype(int)
        self._df["bogomips"] = self._df["bogomips"].astype(float)

        self._df = self._df.drop_duplicates()

        x = self._df.drop(columns=["build_duration"])
        y = self._df["build_duration"]

        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=self._test_size,
            random_state=self._random_state,
        )
        return x_train, x_test, y_train, y_test

    def train(self) -> None:
        """
        Train the model on the dataset
        """
        x_train, _, y_train, _ = self._preprocess_data()
        categorical_transformer = OneHotEncoder(handle_unknown="ignore")
        numerical_transformer = StandardScaler()

        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", categorical_transformer, self._categorical_features),
                ("num", numerical_transformer, self._numerical_features),
            ],
        )

        model_xgb = XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            random_state=self._random_state,
            max_depth=6,
        )

        self.model_pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("regressor", model_xgb),
            ],
        )

        print("Training XGBoost Regressor...")
        self.model_pipeline.fit(x_train, y_train)
        print("Training finished.")

    def save(self, model_path: str) -> None:
        """
        Save the trained model to the given path
        """
        if Path(model_path).exists():
            raise ValueError(f"File {model_path} already exists, won't overwrite it")

        joblib.dump(self.model_pipeline, model_path)
