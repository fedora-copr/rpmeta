import logging
from pathlib import Path

import numpy as np
import pandas as pd
from optuna import Study
from sklearn.model_selection import train_test_split

from rpmeta.constants import ALL_FEATURES, CATEGORICAL_FEATURES, TARGET
from rpmeta.train.base import BaseModel, BestModelResult, TrialResult

logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, data: pd.DataFrame, model_allowlist: set[str] = None) -> None:
        self.df = data.copy()
        self._preprocess_dataset()
        for col in CATEGORICAL_FEATURES:
            self.df[col] = self.df[col].astype("category")

        self.X = self.df[ALL_FEATURES]
        self.y = self.df[TARGET]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=0.2,
            random_state=42,
        )

        if model_allowlist is None:
            model_allowlist = {model.name for model in BaseModel.get_registry()}

        self.model_allowlist = {model.lower() for model in model_allowlist}

    def _get_filtered_models(self) -> list[BaseModel]:
        models = []
        for model in BaseModel.get_registry():
            if model.name.lower() in self.model_allowlist:
                models.append(model)

        return models

    @staticmethod
    def _remove_outliers_iqr(group: pd.DataFrame) -> pd.DataFrame:
        q1 = group[TARGET].quantile(0.25)
        q3 = group[TARGET].quantile(0.75)

        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        result = group[(group[TARGET] >= lower) & (group[TARGET] <= upper)]
        if len(result) == 0:
            # to prevent complete data loss
            group[TARGET] = group[TARGET].clip(lower=lower, upper=upper)
            return group

        return result

    def _preprocess_dataset(self) -> None:
        # most of the preprocessing is done in the dataset.py
        logger.info("Preprocessing dataset")
        self.df["version"] = self.df["version"].str.replace(r"[\^~].*", "", regex=True)
        self.df[TARGET] = np.round(self.df[TARGET]).astype(int)
        self.df["ram"] = np.round(self.df["ram"] / 100000).astype(int)
        self.df["swap"] = np.round(self.df["swap"] / 100000).astype(int)
        self.df = self.df[(self.df[TARGET] >= 15) & (self.df[TARGET] <= 115000)]

        logger.info("Removing duplicates and outliers")
        dupes = self.df[self.df.duplicated(subset=ALL_FEATURES, keep=False)].copy()
        not_dupes = self.df[~self.df.index.isin(dupes.index)]

        logger.info(f"Found {len(dupes)} duplicates")
        logger.info(f"Found {len(not_dupes)} non-duplicates")

        logger.info("Removing outliers using IQR method")
        filtered_dupes = dupes.groupby(ALL_FEATURES, group_keys=False).apply(
            self._remove_outliers_iqr,
        )
        logger.info(f"Filtered {len(dupes) - len(filtered_dupes)} outliers from duplicates")
        self.df = pd.concat([filtered_dupes, not_dupes], ignore_index=True)

        # aggregate the target by mean
        logger.info("Aggregating target by mean")
        self.df = self.df.groupby(ALL_FEATURES, as_index=False).agg({TARGET: "mean"})
        logger.info(f"Aggregated target by mean, resulting in {len(self.df)} records.")
        self.df[TARGET] = np.round(self.df[TARGET]).astype(int)

        logger.info("Removing duplicates and NaN values")
        self.df = self.df.drop_duplicates(keep=False)
        self.df = self.df.dropna()
        logger.info(f"Removed duplicates and NaN values, resulting in {len(self.df)} records.")

        self.df[TARGET] = self.df[TARGET].apply(np.ceil).astype(int)

    def run_all_studies(
        self,
        n_trials: int = 200,
    ) -> tuple[dict[str, list[TrialResult]], dict[str, BestModelResult], dict[str, Study]]:
        all_results = {}
        best_models = {}
        studies = {}

        for model in self._get_filtered_models():
            logger.info(f"Training model: {model.name}")
            study, trials, best = model.run_study(
                self.X_train,
                self.X_test,
                self.y_train,
                self.y_test,
                n_trials=n_trials,
            )
            studies[model.name] = study
            all_results[model.name] = trials
            best_models[model.name] = best

        return all_results, best_models, studies

    def run(self, result_dir: Path) -> Path:
        """
        Run the model training and save the results.
        """
        models = []
        for model in self._get_filtered_models():
            logger.info(f"Starting training for model: {model.name}")
            models.append(model.run(self.X, self.y, result_dir))

        logger.info("Training completed for all models.")
        return models
