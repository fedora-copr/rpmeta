import json
import logging
import os
import tempfile
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna.visualization as vis
import pandas as pd
import seaborn as sn
from optuna import Study

from rpmeta.config import Config
from rpmeta.train.base import BestModelResult, TrialResult

logger = logging.getLogger(__name__)


@dataclass
class VisualizeConfig:
    results_json: str = "best_models_results.json"
    performance_json: str = "model_performance.json"
    dist_csv: str = "distribution_data.csv"


class ResultsHandler:
    def __init__(
        self,
        all_trials: dict[str, list[TrialResult]],
        best_models: dict[str, BestModelResult],
        studies: dict[str, Study],
        X_test: pd.DataFrame,  # noqa: N803
        y_test: pd.Series,
        config: Config,
    ) -> None:
        self.all_trials = all_trials
        self.bests = best_models
        self.studies = studies
        self.X_test = X_test
        self.y_test = y_test

        self.config = config
        self._tune_dir = self.config.result_dir / "hyperparameter_tuning"
        self._optuna_dir = self._tune_dir / "optuna_plots"
        self._plot_dir = self._tune_dir / "regular_plots"
        self._data_dir = self._tune_dir / "data"

        self._optuna_dir.mkdir(parents=True, exist_ok=True)
        self._plot_dir.mkdir(parents=True, exist_ok=True)
        self._data_dir.mkdir(parents=True, exist_ok=True)

        sn.set_theme(style="darkgrid")
        plt.grid(axis="y", linestyle="dotted")
        plt.grid(axis="x", linestyle="dotted")

    def _save_figure(self, fig: plt.Figure, name: str) -> None:
        path = os.path.join(self._plot_dir, f"{name}.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        fig.close()
        logger.info(f"Saved figure {name}.png to {self._plot_dir}")

    def plot_trials(self) -> None:
        for model_name, results in self.all_trials.items():
            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values("test_score").reset_index(drop=True)
            results_df["trial_number"] = results_df.index

            plt.figure(figsize=(12, 7))
            sn.scatterplot(
                data=results_df,
                x="trial_number",
                y="test_score",
                hue="fit_time",
                palette="mako",
            )

            sn.set_theme(style="darkgrid")
            plt.grid(axis="y", linestyle="dotted")
            plt.grid(axis="x", linestyle="dotted")

            best_row = results_df.iloc[-1]
            best_score = best_row["test_score"]
            best_index = best_row["trial_number"]

            best_params = {k: v for k, v in best_row.items() if k not in {"model_name"}}
            best_params |= best_params["params"]
            del best_params["params"]

            best_params = {
                k: (
                    int(v)
                    if isinstance(v, float) and v.is_integer()
                    else round(v, 4)
                    if isinstance(v, float)
                    else v
                )
                for k, v in best_params.items()
                if k not in {"trial", "trial_number", "model"}
            }

            plt.title(
                f"Model performance: {model_name}\n"
                f"Best neg. RMSE: {best_score:.3f} | "
                f"Configurations tried: {len(results_df)}",
            )

            plt.xlabel("Configuration index (sorted by negative RMSE)")
            plt.ylabel("Negative RMSE (higher is better)")
            plt.legend(title="Fit time")
            plt.grid(True, linestyle="--")

            text_params = "\n".join(
                [f"{key}: {value}" for key, value in best_params.items()],
            )
            if text_params:
                plt.annotate(
                    text_params,
                    xy=(best_index, best_score),
                    xytext=(1.05, 0.95),
                    textcoords="axes fraction",
                    ha="left",
                    va="top",
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.6", fc="lightyellow", ec="gray"),
                    arrowprops=dict(
                        arrowstyle="->",
                        connectionstyle="arc3,rad=0.2",
                        color="gray",
                    ),
                )

            self._save_figure(plt, f"{model_name}_performance")

    def print_trials_table(self):
        for _, results in self.all_trials.items():
            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values(
                "test_score",
                ascending=False,
            ).reset_index(drop=True)
            print(results_df)

    def plot_predictions(self):
        for model_name, best in self.bests.items():
            y_pred = best.model.predict(self.X_test)

            y_test_np = np.array(self.y_test)
            y_pred_np = np.array(y_pred)

            valid_mask_positive_nums = (
                ~np.isnan(y_test_np) & ~np.isnan(y_pred_np) & (y_test_np >= 0) & (y_pred_np >= 0)
            )

            mask_5k = (self.y_test <= 5000) & (y_pred <= 5000)

            for suffix, y_test_f, y_pred_f, title in [
                ("", self.y_test, y_pred, f"{model_name} Prediction vs. Reality"),
                (
                    "_5k",
                    self.y_test[mask_5k],
                    y_pred[mask_5k],
                    f"{model_name} Prediction vs. Reality (5k)",
                ),
                (
                    "_log_scale",
                    np.log1p(y_test_np[valid_mask_positive_nums]),
                    np.log1p(y_pred_np[valid_mask_positive_nums]),
                    f"{model_name} Prediction vs. Reality (log-scale)",
                ),
            ]:
                plt.figure(figsize=(8, 6))
                plt.scatter(
                    y_test_f,
                    y_pred_f,
                    alpha=0.3,
                    s=15,
                    color="royalblue",
                    edgecolor="black",
                    linewidth=0.3,
                )

                sn.set_theme(style="darkgrid")
                plt.grid(axis="y", linestyle="dotted")
                plt.grid(axis="x", linestyle="dotted")

                plt.plot(
                    [y_test_f.min(), y_test_f.max()],
                    [y_test_f.min(), y_test_f.max()],
                    "k--",
                )
                plt.title(title)
                plt.xlabel("Real build_duration")
                plt.ylabel("Predicted build_duration")
                plt.grid(True, linestyle="--")
                plt.tight_layout()

                self._save_figure(plt, f"{model_name}_pred_vs_real{suffix}")

    def plot_test_value_compare(self):
        best_df = pd.DataFrame.from_dict(self.bests, orient="index")

        plt.figure(figsize=(10, 6))
        sn.barplot(data=best_df, x="model_name", y="neg_rmse", color="royalblue")

        sn.set_theme(style="darkgrid")
        plt.grid(axis="y", linestyle="dotted")
        plt.grid(axis="x", linestyle="dotted")

        plt.title("Comparison of the best models by negative RMSE")
        plt.ylabel("Negative RMSE")
        plt.xlabel("Model")
        plt.tight_layout()

        self._save_figure(plt, "model_test_metric_comparison")

    def save_best_json(self):
        out = {
            name: {
                "r2": bm.r2,
                "neg_rmse": bm.neg_rmse,
                "neg_mae": bm.neg_mae,
                "params": bm.params,
            }
            for name, bm in self.bests.items()
        }
        with open(self._data_dir / "best_models_test_scores.json", "w") as f:
            json.dump(out, f, indent=4)

    def plot_ditribution(self):
        plt.figure(figsize=(12, 6))
        y_test_log = np.log1p(self.y_test)
        for model_name, best_model in self.bests.items():
            y_pred = best_model.model.predict(self.X_test)

            y_pred = np.array(y_pred)
            valid_mask = (y_pred >= 0) & ~np.isnan(y_pred)
            y_pred_log = np.log1p(y_pred[valid_mask])

            sn.kdeplot(y_pred_log, label=model_name)

        sn.kdeplot(y_test_log, label="Reality", linestyle="--", color="black")

        sn.set_theme(style="darkgrid")
        plt.grid(axis="y", linestyle="dotted")
        plt.grid(axis="x", linestyle="dotted")

        plt.title("Distribution of predicted values vs. reality (log-scale)")
        plt.xlabel("build_duration")
        plt.grid(True, linestyle="--")
        plt.legend()
        plt.tight_layout()
        self._save_figure(plt, "model_distribution_real_vs_predicted")

        dist_data = {"y_test": self.y_test}
        for model_name, best_model in self.bests.items():
            dist_data[model_name] = best_model.model.predict(self.X_test)

        df_dist = pd.DataFrame(dist_data)
        path_to_save = self._data_dir / "distribution_data.csv"
        df_dist.to_csv(path_to_save, index=False)

    def _plot_metric(self, title, data, ylabel, name):
        plt.figure(figsize=(9, 4))
        plot_df = pd.DataFrame(
            {"Model": list(data.keys()), "Value": list(data.values())},
        )

        sn.barplot(
            data=plot_df,
            x="Model",
            y="Value",
            hue="Model",
            palette="mako",
            legend=False,
        )

        sn.set_theme(style="darkgrid")
        plt.grid(axis="y", linestyle="dotted")
        plt.grid(axis="x", linestyle="dotted")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        self._save_figure(plt, name)

    def plot_model_performance(self, tempdir: Path):
        prediction_times = {}
        memory_usages = {}
        model_file_sizes = {}
        reload_memory_usages = {}

        # TODO: again this wastes a lot of memory... save the model to disk and load back
        for model_name, best_model in self.bests.items():
            model_path = tempdir / f"{model_name}.joblib"
            joblib.dump(best_model.model, model_path)

            model_file_sizes[model_name] = os.path.getsize(model_path) / 1024**2

            tracemalloc.start()
            start_time = time.perf_counter()

            _ = best_model.model.predict(self.X_test)

            end_time = time.perf_counter()
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            prediction_times[model_name] = end_time - start_time
            memory_usages[model_name] = peak / 1024**2

            tracemalloc.start()
            loaded_model = joblib.load(model_path)

            _ = loaded_model.predict(self.X_test)

            _, peak_reload = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            reload_memory_usages[model_name] = peak_reload / 1024**2

            print(f"{model_name}:")
            print(f"  Prediction time: {prediction_times[model_name]:.4f} s")
            print(f"  Peak RAM (original): {memory_usages[model_name]:.2f} MB")
            print(f"  Model file size: {model_file_sizes[model_name]:.2f} MB")
            print(
                f"  Peak RAM (after reload): {reload_memory_usages[model_name]:.2f} MB",
            )
            print("-" * 50)

        self._plot_metric("Prediction Time", prediction_times, "Time (s)", "prediction_time")
        self._plot_metric(
            "RAM Usage During Predict",
            memory_usages,
            "Memory (MB)",
            "ram_usage_run",
        )
        self._plot_metric(
            "Model File Size on Disk",
            model_file_sizes,
            "Size (MB)",
            "model_size",
        )
        self._plot_metric(
            "RAM Usage After Reload",
            reload_memory_usages,
            "Memory (MB)",
            "model_ram_usage",
        )

        performance_data = {
            "prediction_times": prediction_times,
            "memory_usages": memory_usages,
            "reload_memory_usages": reload_memory_usages,
            "model_file_sizes": model_file_sizes,
        }

        with open(self._data_dir / "model_performance.json", "w") as f:
            json.dump(performance_data, f, indent=4)

    def plot_optuna_plots(self):
        for model_name, study in self.studies.items():
            try:
                print(f"\nModel: {model_name}")

                # Optimization history
                fig = vis.plot_optimization_history(study, target_name="RMSE")
                fig.write_image(f"{self._optuna_dir}/{model_name}_opt_history.png", scale=2)
                fig.write_html(f"{self._optuna_dir}/{model_name}_opt_history.html")

                # Param importances
                fig = vis.plot_param_importances(study)
                fig.write_image(
                    f"{self._optuna_dir}/{model_name}_param_importance.png",
                    scale=2,
                )
                fig.write_html(f"{self._optuna_dir}/{model_name}_param_importance.html")

                # Parallel coordinates
                fig = vis.plot_parallel_coordinate(study)
                fig.write_image(f"{self._optuna_dir}/{model_name}_parallel.png", scale=2)
                fig.write_html(f"{self._optuna_dir}/{model_name}_parallel.html")

                # Slice plot
                fig = vis.plot_slice(study, target_name="RMSE")
                fig.write_image(f"{self._optuna_dir}/{model_name}_slice.png", scale=2)
                fig.write_html(f"{self._optuna_dir}/{model_name}_slice.html")

                # Contour plot
                importances = study.best_trial.params.keys()
                top_params = list(importances)[:2]
                if len(top_params) >= 2:
                    fig = vis.plot_contour(study, params=top_params)
                    fig.write_image(f"{self._optuna_dir}/{model_name}_contour.png", scale=2)
                    fig.write_html(f"{self._optuna_dir}/{model_name}_contour.html")

                # EDF plot
                fig = vis.plot_edf(study)
                fig.write_image(f"{self._optuna_dir}/{model_name}_edf.png", scale=2)
                fig.write_html(f"{self._optuna_dir}/{model_name}_edf.html")

            except Exception as e:
                print(f"Error for model {model_name}: {e}")

    def run_all(self):
        self.save_best_json()
        self.plot_trials()
        self.print_trials_table()
        self.plot_predictions()
        self.plot_test_value_compare()
        self.plot_ditribution()
        with tempfile.TemporaryDirectory() as tempdir:
            self.plot_model_performance(Path(tempdir))

        # kaleido is required and it is not packaged in fedora
        try:
            import kaleido  # noqa: F401
        except ImportError:
            logger.error("Kaleido package is not installed. Optuna plots will not be generated.")
            return

        self.plot_optuna_plots()
