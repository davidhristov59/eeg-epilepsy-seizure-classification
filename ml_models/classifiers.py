# classifiers.py (modified, SVM removed)
import os
import time
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.base import clone
import logging


@dataclass
class ModelConfig:
    name: str
    model: Any
    params: Dict[str, Any]


@dataclass
class EvaluationMetrics:
    accuracy: float
    tpr: float  # True Positive Rate (Recall/Sensitivity)
    fpr: float  # False Positive Rate
    tnr: float  # True Negative Rate (Specificity)
    precision: float
    f1_score: float
    confusion_matrix: np.ndarray
    training_time: float


class SeizureClassifier:
    def __init__(self, input_file: str, output_dir: str):
        self.input_file = input_file
        self.output_dir = output_dir

        os.makedirs(output_dir, exist_ok=True)

        self.models = self._get_model_configs()
        self.scaler = StandardScaler()
        self.variance_selector = VarianceThreshold(threshold=0.01)
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.output_dir, 'classification.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _get_model_configs(self) -> List[ModelConfig]:
        """Define model configurations (SVM removed for speed)"""
        return [
            ModelConfig(
                "MLP",
                MLPClassifier,
                {
                    "hidden_layer_sizes": (100, 50),
                    "max_iter": 1000,
                    "activation": "relu",
                    "solver": "adam",
                    "random_state": 42,
                    "early_stopping": True,
                    "validation_fraction": 0.1
                },
            ),
            ModelConfig(
                "RandomForest",
                RandomForestClassifier,
                {
                    "n_estimators": 100,
                    "min_samples_split": 5,
                    "min_samples_leaf": 2,
                    "class_weight": "balanced",
                    "random_state": 42,
                    "n_jobs": -1
                },
            ),
            ModelConfig(
                "AdaBoost",
                AdaBoostClassifier,
                {
                    "n_estimators": 100,
                    "random_state": 42,
                    "algorithm": "SAMME"
                }
            ),
            ModelConfig(
                "KNN",
                KNeighborsClassifier,
                {
                    "n_neighbors": 5,
                    "weights": "distance",
                    "metric": "minkowski"
                }
            ),
        ]

    def load_and_preprocess_data(self) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Load dataset and prepare features, labels, and groups"""
        self.logger.info("Loading data from %s", self.input_file)

        try:
            df = pd.read_csv(self.input_file)
        except FileNotFoundError:
            self.logger.error("Input file not found: %s", self.input_file)
            raise
        except Exception as e:
            self.logger.error("Error loading data: %s", str(e))
            raise

        if 'seizure' not in df.columns:
            raise ValueError("'seizure' column not found in the dataset")
        if 'recording_id' not in df.columns:
            raise ValueError("'recording_id' column not found in the dataset")

        feature_cols = [col for col in df.columns if col not in ['seizure', 'start_time', 'subject', 'recording_id']]
        X = df[feature_cols].values
        y = df['seizure'].values
        groups = df['recording_id'].values

        self.logger.info("Dataset shape: %s", X.shape)
        return df, X, y, groups, feature_cols

    def _calculate_metrics(self, y_true, y_pred, training_time=0.0) -> EvaluationMetrics:
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0

        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total > 0 else 0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1_score = 2 * (precision * tpr) / (precision + tpr) if (precision + tpr) > 0 else 0

        return EvaluationMetrics(accuracy, tpr, fpr, tnr, precision, f1_score, cm, training_time)

    def evaluate_model(self, model, X_train, X_test, y_train, y_test, groups_train):
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        cv_metrics = []

        self.logger.info("Starting 5-fold StratifiedGroupKFold cross-validation")

        for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_train, y_train, groups_train), 1):
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

            try:
                fold_model = clone(model)
                fold_model.fit(X_fold_train, y_fold_train)
                y_pred = fold_model.predict(X_fold_val)
                metrics = self._calculate_metrics(y_fold_val, y_pred)
                cv_metrics.append(metrics)
                self.logger.info("Fold %d - Acc: %.3f, F1: %.3f", fold, metrics.accuracy, metrics.f1_score)
            except Exception as e:
                self.logger.error("Error in fold %d: %s", fold, str(e))

        self.logger.info("Training on full training set")
        start = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        train_time = time.time() - start
        test_metrics = self._calculate_metrics(y_test, y_pred, train_time)
        self.logger.info("Test Accuracy: %.3f, F1: %.3f", test_metrics.accuracy, test_metrics.f1_score)

        return cv_metrics, test_metrics

    def save_results(self, model_name, cv_metrics, test_metrics):
        results = {
            "model": model_name,
            "cv_accuracy": np.mean([m.accuracy for m in cv_metrics]),
            "cv_tpr": np.mean([m.tpr for m in cv_metrics]),
            "cv_fpr": np.mean([m.fpr for m in cv_metrics]),
            "test_accuracy": test_metrics.accuracy,
            "test_tpr": test_metrics.tpr,
            "test_fpr": test_metrics.fpr,
            "training_time": test_metrics.training_time,
        }

        df = pd.DataFrame([results])
        output_file = os.path.join(self.output_dir, f"{model_name}_results.csv")
        df.to_csv(output_file, index=False)
        pd.DataFrame(test_metrics.confusion_matrix).to_csv(
            os.path.join(self.output_dir, f"{model_name}_confusion_matrix.csv"), index=False
        )
        self.logger.info("Saved results for %s", model_name)

    def run_classification(self):
        df, X_all, y_all, groups_all, feature_cols = self.load_and_preprocess_data()

        recording_df = df[['recording_id', 'seizure']].groupby('recording_id').max().reset_index()
        rec_ids = recording_df['recording_id'].values
        rec_labels = recording_df['seizure'].values

        train_rec, test_rec = train_test_split(rec_ids, test_size=0.2, stratify=rec_labels, random_state=42)
        train_mask = df['recording_id'].isin(train_rec)
        test_mask = df['recording_id'].isin(test_rec)

        X_train = df.loc[train_mask, feature_cols].values
        y_train = df.loc[train_mask, 'seizure'].values
        groups_train = df.loc[train_mask, 'recording_id'].values
        X_test = df.loc[test_mask, feature_cols].values
        y_test = df.loc[test_mask, 'seizure'].values

        X_train = self.variance_selector.fit_transform(X_train)
        X_test = self.variance_selector.transform(X_test)
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        all_results = []
        for config in self.models:
            self.logger.info("=" * 40)
            self.logger.info("Training %s", config.name)
            self.logger.info("=" * 40)

            try:
                model = config.model(**config.params)
                cv_metrics, test_metrics = self.evaluate_model(model, X_train, X_test, y_train, y_test, groups_train)
                self.save_results(config.name, cv_metrics, test_metrics)

                all_results.append({
                    "model": config.name,
                    "cv_accuracy": np.mean([m.accuracy for m in cv_metrics]),
                    "cv_tpr": np.mean([m.tpr for m in cv_metrics]),
                    "cv_fpr": np.mean([m.fpr for m in cv_metrics]),
                    "cv_precision": np.mean([m.precision for m in cv_metrics]),
                    "cv_f1_score": np.mean([m.f1_score for m in cv_metrics]),
                    "test_accuracy": test_metrics.accuracy,
                    "test_tpr": test_metrics.tpr,
                    "test_fpr": test_metrics.fpr,
                    "test_precision": test_metrics.precision,
                    "test_f1_score": test_metrics.f1_score,
                    "training_time": test_metrics.training_time,
                })
            except Exception as e:
                self.logger.error("Error training %s: %s", config.name, str(e))

        # ✅ Combine all results into one summary CSV
        self._save_all_results(all_results)

        self.logger.info("Training complete for all models.")
        for r in all_results:
            self.logger.info(
                "%s - Acc: %.3f, F1: %.3f, Prec: %.3f",
                r["model"], r["test_accuracy"], r["test_f1_score"], r["test_precision"]
            )

    def _save_all_results(self, all_results):
        """Combine all model results into one summary CSV."""
        all_df = pd.DataFrame(all_results)
        output_path = os.path.join(self.output_dir, "all_results.csv")
        all_df.to_csv(output_path, index=False)
        self.logger.info("✅ Combined results saved to %s", output_path)


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(project_root, 'processed_data', 'subjects.csv')
    output_dir = os.path.join(project_root, 'output', 'classification_results')

    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        return

    classifier = SeizureClassifier(input_file, output_dir)
    classifier.run_classification()


if __name__ == "__main__":
    main()
