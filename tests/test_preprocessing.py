import unittest

import numpy as np
from sklearn.model_selection import train_test_split

from preprocessing import prepare_train_test_data, resample_training_data


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.features = np.column_stack(
            (np.arange(100, dtype=float), np.arange(100, dtype=float) * 10)
        )
        self.target = np.array([0] * 80 + [1] * 20)

    def test_scaler_is_fitted_on_training_partition_only(self):
        expected_train, _, _, _ = train_test_split(
            self.features,
            self.target,
            test_size=0.2,
            random_state=42,
            stratify=self.target,
        )

        X_train, X_test, y_train, y_test, scaler = prepare_train_test_data(
            self.features, self.target
        )

        np.testing.assert_allclose(scaler.mean_, expected_train.mean(axis=0))
        np.testing.assert_allclose(X_train.mean(axis=0), [0.0, 0.0], atol=1e-12)
        self.assertEqual(len(X_test), 20)
        self.assertEqual(len(y_test), 20)
        self.assertEqual(len(y_train), 80)

    def test_holdout_keeps_original_imbalanced_class_distribution(self):
        _, _, _, y_test, _ = prepare_train_test_data(self.features, self.target)

        unique, counts = np.unique(y_test, return_counts=True)
        self.assertEqual(dict(zip(unique, counts)), {0: 16, 1: 4})

    def test_smote_balances_only_the_supplied_training_partition(self):
        X_train, X_test, y_train, y_test, _ = prepare_train_test_data(
            self.features, self.target
        )
        X_train_before = X_train.copy()
        y_train_before = y_train.copy()
        X_test_before = X_test.copy()
        y_test_before = y_test.copy()

        _, y_resampled = resample_training_data(X_train, y_train)

        counts = np.unique(y_resampled, return_counts=True)[1]
        self.assertEqual(counts[0], counts[1])
        self.assertGreater(len(y_resampled), len(y_train))
        np.testing.assert_array_equal(X_train, X_train_before)
        np.testing.assert_array_equal(y_train, y_train_before)
        np.testing.assert_array_equal(X_test, X_test_before)
        np.testing.assert_array_equal(y_test, y_test_before)

    def test_rejects_mismatched_feature_and_target_lengths(self):
        with self.assertRaisesRegex(ValueError, "same number of rows"):
            prepare_train_test_data(self.features, self.target[:-1])


if __name__ == "__main__":
    unittest.main()
