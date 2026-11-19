"""Leakage-safe preprocessing helpers for the credit-card fraud notebook."""

import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def prepare_train_test_data(
    features,
    target,
    *,
    test_size=0.2,
    random_state=42,
):
    """Split with stratification, then fit the scaler on training rows only."""
    feature_array = np.asarray(features)
    target_array = np.asarray(target)

    if feature_array.ndim != 2:
        raise ValueError("features must be a two-dimensional array")
    if target_array.ndim != 1:
        raise ValueError("target must be a one-dimensional array")
    if feature_array.shape[0] != target_array.shape[0]:
        raise ValueError("features and target must contain the same number of rows")

    X_train, X_test, y_train, y_test = train_test_split(
        feature_array,
        target_array,
        test_size=test_size,
        random_state=random_state,
        stratify=target_array,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def resample_training_data(features, target, *, random_state=42):
    """Apply SMOTE to a training partition without modifying the input arrays."""
    feature_array = np.asarray(features)
    target_array = np.asarray(target)

    if feature_array.ndim != 2:
        raise ValueError("features must be a two-dimensional array")
    if target_array.ndim != 1:
        raise ValueError("target must be a one-dimensional array")
    if feature_array.shape[0] != target_array.shape[0]:
        raise ValueError("features and target must contain the same number of rows")

    return SMOTE(random_state=random_state).fit_resample(feature_array, target_array)
