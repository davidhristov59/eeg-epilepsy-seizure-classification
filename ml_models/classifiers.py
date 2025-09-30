import os
import time
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import VarianceThreshold
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
        self.models = self._get_model_configs()
        self.scaler = StandardScaler()
        self.variance_selector = VarianceThreshold(threshold=0.01)  # Remove near-zero variance features
        self._setup_logging()
        os.makedirs(output_dir, exist_ok=True)

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
        """Define model configurations with balanced parameters"""
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
                "SVM",
                SVC,
                {
                    "kernel": "rbf",
                    "class_weight": "balanced",
                    "random_state": 42,
                    "probability": True
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

    def load_and_preprocess_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess the dataset"""
        self.logger.info("Loading data from %s", self.input_file)

        try:
            data = pd.read_csv(self.input_file)
        except FileNotFoundError:
            self.logger.error("Input file not found: %s", self.input_file)
            raise
        except Exception as e:
            self.logger.error("Error loading data: %s", str(e))
            raise

        # Check required columns
        if 'seizure' not in data.columns:
            raise ValueError("'seizure' column not found in the dataset")

        # Separate features and target
        feature_cols = [col for col in data.columns if col not in ['seizure', 'start_time', 'subject']]
        X = data[feature_cols].values
        y = data['seizure'].values

        self.logger.info("Original dataset shape: %s", X.shape)
        self.logger.info("Total seizure samples: %d (%.2f%%)",
                         np.sum(y), (np.sum(y) / len(y)) * 100)

        # Remove low-variance features
        X = self.variance_selector.fit_transform(X)
        self.logger.info("After variance threshold: %s", X.shape)

        # Standardize features
        X = self.scaler.fit_transform(X)

        return X, y

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                           training_time: float = 0.0) -> EvaluationMetrics:
        """Calculate comprehensive evaluation metrics"""
        cm = confusion_matrix(y_true, y_pred)

        # Handle edge cases where confusion matrix might not be 2x2
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            # If only one class present in predictions
            if len(np.unique(y_pred)) == 1:
                if y_pred[0] == 0:  # All predicted as non-seizure
                    tn = np.sum(y_true == 0)
                    fp = 0
                    fn = np.sum(y_true == 1)
                    tp = 0
                else:  # All predicted as seizure
                    tn = 0
                    fp = np.sum(y_true == 0)
                    fn = 0
                    tp = np.sum(y_true == 1)
            else:
                tn = fp = fn = tp = 0

        # Calculate metrics with zero-division handling
        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total > 0 else 0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensitivity/Recall
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # Specificity
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1_score = 2 * (precision * tpr) / (precision + tpr) if (precision + tpr) > 0 else 0

        return EvaluationMetrics(
            accuracy=accuracy,
            tpr=tpr,
            fpr=fpr,
            tnr=tnr,
            precision=precision,
            f1_score=f1_score,
            confusion_matrix=cm,
            training_time=training_time
        )

    def evaluate_model(
            self,
            model: Any,
            X_train: np.ndarray,
            X_test: np.ndarray,
            y_train: np.ndarray,
            y_test: np.ndarray,
    ) -> Tuple[List[EvaluationMetrics], EvaluationMetrics]:
        """Evaluate model using cross-validation and test set"""

        # Use StratifiedKFold to maintain class distribution
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_metrics = []

        self.logger.info("Starting 5-fold cross-validation")

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

            self.logger.info("Fold %d - Train: %s, Val: %s",
                             fold, X_fold_train.shape, X_fold_val.shape)

            try:
                # Create a fresh model instance for each fold
                fold_model = model.__class__(**model.get_params())
                fold_model.fit(X_fold_train, y_fold_train)
                y_pred = fold_model.predict(X_fold_val)

                fold_metrics = self._calculate_metrics(y_fold_val, y_pred)
                cv_metrics.append(fold_metrics)

                self.logger.info("Fold %d - Acc: %.3f, TPR: %.3f, F1: %.3f",
                                 fold, fold_metrics.accuracy, fold_metrics.tpr, fold_metrics.f1_score)

            except Exception as e:
                self.logger.error("Error in fold %d: %s", fold, str(e))
                continue

        # Train on full training set and evaluate on test set
        self.logger.info("Training on full training set")
        start_time = time.time()

        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            training_time = time.time() - start_time

            test_metrics = self._calculate_metrics(y_test, y_pred, training_time)

            self.logger.info("Test Set Results:")
            self.logger.info("Accuracy: %.3f", test_metrics.accuracy)
            self.logger.info("TPR (Sensitivity): %.3f", test_metrics.tpr)
            self.logger.info("TNR (Specificity): %.3f", test_metrics.tnr)
            self.logger.info("Precision: %.3f", test_metrics.precision)
            self.logger.info("F1 Score: %.3f", test_metrics.f1_score)
            self.logger.info("Training Time: %.2f seconds", test_metrics.training_time)

        except Exception as e:
            self.logger.error("Error in final model training: %s", str(e))
            raise

        return cv_metrics, test_metrics

    def save_results(
            self,
            model_name: str,
            cv_metrics: List[EvaluationMetrics],
            test_metrics: EvaluationMetrics,
    ):
        """Save evaluation results to CSV"""
        if not cv_metrics:
            self.logger.warning("No cross-validation metrics to save for %s", model_name)
            return

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

        # Also save confusion matrix
        cm_file = os.path.join(self.output_dir, f"{model_name}_confusion_matrix.csv")
        pd.DataFrame(test_metrics.confusion_matrix).to_csv(cm_file, index=False)

        self.logger.info("Results saved to %s", output_file)

    def run_classification(self):
        """Run the complete classification pipeline"""
        try:
            # Load and preprocess data
            X, y = self.load_and_preprocess_data()

            # Split data with stratification to maintain class balance
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            self.logger.info("Train set: %s, Test set: %s", X_train.shape, X_test.shape)
            self.logger.info("Train seizures: %d, Test seizures: %d",
                             np.sum(y_train), np.sum(y_test))

            all_results = []

            # Train and evaluate each model
            for config in self.models:
                self.logger.info("\n" + "=" * 50)
                self.logger.info("Training %s", config.name)
                self.logger.info("=" * 50)

                try:
                    model = config.model(**config.params)
                    cv_metrics, test_metrics = self.evaluate_model(
                        model, X_train, X_test, y_train, y_test
                    )
                    self.save_results(config.name, cv_metrics, test_metrics)
                    all_results.append({
                        "model": config.name,
                        "test_metrics": test_metrics,
                        "cv_metrics": cv_metrics
                    })

                except Exception as e:
                    self.logger.error("Error training %s: %s", config.name, str(e))
                    continue

            # Combine and save all results
            if all_results:
                self._save_combined_results(all_results)
                self._print_summary(all_results)
            else:
                self.logger.error("No models were successfully trained!")

        except Exception as e:
            self.logger.error("Fatal error in classification pipeline: %s", str(e))
            raise

    def _save_combined_results(self, all_results: List[Dict]):
        """Save combined results from all models"""
        try:
            combined_dfs = []
            for result in all_results:
                model_file = os.path.join(self.output_dir, f"{result['model']}_results.csv")
                if os.path.exists(model_file):
                    combined_dfs.append(pd.read_csv(model_file))

            if combined_dfs:
                combined_results = pd.concat(combined_dfs, ignore_index=True)
                combined_file = os.path.join(self.output_dir, "all_results.csv")
                combined_results.to_csv(combined_file, index=False)
                self.logger.info("Combined results saved to %s", combined_file)

        except Exception as e:
            self.logger.error("Error saving combined results: %s", str(e))

    def _print_summary(self, all_results: List[Dict]):
        """Print a summary of all model performances"""
        self.logger.info("\n" + "=" * 70)
        self.logger.info("FINAL RESULTS SUMMARY")
        self.logger.info("=" * 70)

        for result in all_results:
            metrics = result['test_metrics']
            self.logger.info("%s - Acc: %.3f, F1: %.3f, TPR: %.3f, TNR: %.3f",
                             result['model'], metrics.accuracy, metrics.f1_score,
                             metrics.tpr, metrics.tnr)


def main():
    """Main execution function"""
    # Setup directories
    input_file = "processed_data/subjects.csv"
    output_dir = "output/classification_results"

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found!")
        print("Please ensure the data file exists before running the classifier.")
        return

    # Create and run classifier
    classifier = SeizureClassifier(input_file=input_file, output_dir=output_dir)
    classifier.run_classification()


if __name__ == "__main__":
    main()