import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from optuna import Study
from sklearn.model_selection import train_test_split

from rpmeta.config import Config
from rpmeta.constants import ALL_FEATURES, CATEGORICAL_FEATURES, DIVIDER, TARGET, ModelEnum
from rpmeta.trainer.base import BestModelResult, ModelTrainer, TrialResult
from rpmeta.trainer.models import get_all_model_trainers

logger = logging.getLogger(__name__)


class ModelTrainingManager:
    def __init__(
        self,
        data: pd.DataFrame,
        config: Config,
        model_allowlist: Optional[set[ModelEnum]] = None,
    ) -> None:
        self.df = data
        self.config = config

        self._preprocess_dataset()

        category_dtypes = self._categorize_get_categories_mapping()
        category_dtypes_path = (
            Path(self.config.result_dir) / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
        )

        with category_dtypes_path.open("w") as f:
            json.dump(category_dtypes, f, indent=4)

        logger.info("Saved category dtypes to %s", category_dtypes_path)

        self.X = self.df[ALL_FEATURES]
        self.y = self.df[TARGET]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=self.config.model.test_size,
            random_state=self.config.model.random_state,
        )

        if model_allowlist is None:
            model_allowlist_str = set(ModelEnum.get_all_model_names())
        else:
            model_allowlist_str = {model.value for model in model_allowlist}

        self.model_allowlist = model_allowlist_str
        logger.info("Model allowlist: %s", self.model_allowlist)

    def _get_filtered_models(self) -> list[ModelTrainer]:
        models = []
        for model in get_all_model_trainers(self.config):
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

    @staticmethod
    def _aggregate_group(group: pd.DataFrame) -> pd.Series:
        return pd.Series({TARGET: int(group[TARGET].mean())})

    def _preprocess_dataset(self) -> None:
        # rest of the preprocessing is done in the dataset.py
        logger.info("Preprocessing dataset")
        self.df["version"] = self.df["version"].str.replace(r"[\^~].*", "", regex=True)
        self.df[TARGET] = self.df[TARGET].astype(int)  # this has to be in minutes

        self.df["hw_info.ram"] = np.round(
            self.df["hw_info.ram"] / DIVIDER,
        ).astype(int)
        self.df["hw_info.swap"] = np.round(
            self.df["hw_info.swap"] / DIVIDER,
        ).astype(int)
        self.df = self.df[(self.df[TARGET] >= 0) & (self.df[TARGET] <= 1917)]

        logger.info("Removing duplicates and NaN values")
        self.df.drop_duplicates(inplace=True, keep="first")
        self.df.dropna(inplace=True)
        logger.info("Removed duplicates and NaN values, resulting in %d records.", len(self.df))

        logger.info("Removing duplicates and outliers")
        dupes = self.df[self.df.duplicated(subset=ALL_FEATURES, keep=False)].copy()
        not_dupes = self.df[~self.df.index.isin(dupes.index)]

        logger.info("Found %d duplicates", len(dupes))
        logger.info("Found %d non-duplicates", len(not_dupes))

        logger.info("Removing outliers using IQR method")
        filtered_dupes = dupes.groupby(ALL_FEATURES, group_keys=False).apply(
            self._remove_outliers_iqr,
        )
        logger.info("Filtered %d outliers from duplicates", len(dupes) - len(filtered_dupes))
        self.df = pd.concat([filtered_dupes, not_dupes], ignore_index=True)

        # aggregate the target by mean
        logger.info("Aggregating target by mean")
        group_cols = [col for col in self.df.columns if col != TARGET]
        self.df = (
            self.df.groupby(group_cols, group_keys=False)
            .apply(self._aggregate_group, include_groups=False)
            .reset_index()
        )

        logger.info("Aggregated target by mean, resulting in %d records.", len(self.df))
        self.df[TARGET] = np.round(self.df[TARGET]).astype(int)

        self.df[TARGET] = self.df[TARGET].apply(np.ceil).astype(int)
        logger.info("Preprocessing completed")
        logger.info("Preprocessed dataset, resulting in %d records.", len(self.df))

    def _categorize_get_categories_mapping(self) -> dict[str, list[str]]:
        result = {}
        for col in CATEGORICAL_FEATURES:
            self.df[col] = self.df[col].astype("category")
            result[col] = self.df[col].cat.categories.tolist()

        logger.info("Categorized columns: %s", list(result.keys()))
        return result

    def run_all_studies(
        self,
        n_trials: int = 100,
    ) -> tuple[dict[str, list[TrialResult]], dict[str, BestModelResult], dict[str, Study]]:
        all_results = {}
        best_models = {}
        studies = {}

        train_packages = set(self.X_train["package_name"].unique())
        unknown_mask = ~self.X_test["package_name"].isin(train_packages)

        num_unknown = unknown_mask.sum()
        print(f"Count of unknown packages in train data: {num_unknown}")

        # filter unseen data by training - response on unseen data is -1 nevertheless
        known_mask = ~unknown_mask
        self.X_test = self.X_test[known_mask]
        self.y_test = self.y_test[known_mask]

        for model in self._get_filtered_models():
            logger.info("Training model: %s", model.name)
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

            logger.info("Best model for %s: %s", model.name, best.model_name)

        logger.info("All models trained.")
        return all_results, best_models, studies

    def run(self) -> list[Path]:
        """
        Run the model training and save the results.

        Returns:
            list[Path]: List of paths to the saved models
        """
        models = []
        for model in self._get_filtered_models():
            logger.info("Starting training for model: %s", model.name)
            models.append(model.run(self.X, self.y))

        logger.info("Training completed for all models.")
        return models
