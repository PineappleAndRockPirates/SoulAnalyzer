from soulanalyzer.backend.MLData import MLData

import joblib
import numpy as np
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score


class TrainingResult:

    def __init__(self, model, score, path_to_model):
        self.model = model
        self.score = score
        self.path_to_model = path_to_model


class SoulAnalyzer:

    def __init__(self, dataset: MLData, random_seed: int = 42):
        self.dataset = dataset
        self.random_seed = random_seed

    def classification_for_different_classifiers(self):
        max_tree_depth = self._calculate_max_tree_depth(self.dataset.train.x.shape[1])
        k_min = self._calculate_k_min(self.dataset.train.x.shape[0])

        classifiers_hyper = [
            (DecisionTreeClassifier(random_state=self.random_seed), "max_depth", range(1, max_tree_depth)),
            (RandomForestClassifier(random_state=self.random_seed), "max_depth", range(1, max_tree_depth)),
            (KNeighborsClassifier(), "n_neighbors", range(k_min, k_min + 10)),
        ]

        return self._train_models(classifiers_hyper)

    def _calculate_max_tree_depth(self, n_features):
        return int(np.log2(n_features)) + 1

    def _calculate_k_min(self, n_samples):
        return int(np.sqrt(n_samples))

    def _k_fold_cross_validation(self, model, hyperparam_name, value):
        if hyperparam_name:
            model.set_params(**{hyperparam_name: value})
        accuracy = cross_val_score(model, self.dataset.train.x, self.dataset.train.y, cv=5)
        return accuracy.mean()

    def _train_w_best_hyperparam(self, model_hyper):
        model, hyperparam_name, hyperparam_values = model_hyper

        best_hyperparam_value = None
        if hyperparam_name:
            accuracies = [self._k_fold_cross_validation(model, hyperparam_name, value) for value in hyperparam_values]
            best_hyperparam_value = hyperparam_values[np.argmax(accuracies)]
            model.set_params(**{hyperparam_name: best_hyperparam_value})

        model.fit(self.dataset.train.x, self.dataset.train.y)
        return model, hyperparam_name, best_hyperparam_value

    def _train_model(self, model_hyper):
        model, hyperparam, best_hyperparam_value = self._train_w_best_hyperparam(model_hyper)

        # Evaluate the model for never seen data.
        y_pred = model.predict(self.dataset.valid.x)
        acc_on_valid = accuracy_score(self.dataset.valid.y, y_pred)
        print(
            f'Best Model for {type(model).__name__} with {hyperparam}={best_hyperparam_value}, Accuracy for '
            f'Validation Set: {acc_on_valid}')

        # Save the trained model
        output_dir = Path(f"results/trained_models/")
        output_dir.mkdir(parents=True, exist_ok=True)
        model_file = output_dir / f"{type(model).__name__}_model_{self.random_seed}.joblib"
        joblib.dump(model, model_file, compress=False)
        return model, acc_on_valid, model_file

    def _train_models(self, models_hyper):
        training_results = []
        for model_hyper in models_hyper:
            training_results.append(TrainingResult(*self._train_model(model_hyper)))

        scores = [result.score for result in training_results]
        best_performance = accuracy_score(scores)
        print(
            f'Best Model: {type(models_hyper[best_performance].model).__name__} with Accuracy on Validation Set: '
            f'{training_results[best_performance].score}')
        return training_results[best_performance].path_to_model, training_results[best_performance].score
