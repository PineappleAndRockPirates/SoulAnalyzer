from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class FeatureTargetMatrix:
    x: np.ndarray
    y: np.ndarray


class MLData:
    def __init__(self, csv: Path = Path("data/heart_failure_clinical_records.csv"), test_size: float = 0.2,
                 random_seed: int = 42
                 ) -> None:
        self.df = pd.read_csv(csv)
        self.test_size = test_size
        self.random_seed = random_seed
        self.complete_data_set = self._get_complete_data_set()
        self.train_set, self.test_set = self._get_prepared_matrices()

    def _get_complete_data_set(self) -> FeatureTargetMatrix:
        x = self.df.drop(columns=["DEATH_EVENT"])
        y = self.df["DEATH_EVENT"]

        return FeatureTargetMatrix(x.values, y.values)

    def _get_prepared_matrices(self) -> tuple[FeatureTargetMatrix, FeatureTargetMatrix]:
        x_train, x_test, y_train, y_test = train_test_split(
            self.complete_data_set.x,
            self.complete_data_set.y,
            test_size=self.test_size,
            random_state=self.random_seed,
        )
        return FeatureTargetMatrix(x_train, y_train), FeatureTargetMatrix(x_test, y_test)
