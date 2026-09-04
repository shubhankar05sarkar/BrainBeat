import numpy as np
import joblib
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier


def train_eeg_knn(X, y, n_splits=5, k=5):

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    all_y_true = []
    all_y_pred = []

    model = KNeighborsClassifier(n_neighbors=k)

    fold = 1
    for train_idx, test_idx in skf.split(X, y):
        print(f"\n--- Fold {fold} ---")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

        fold += 1

    print("\n==============================")
    print("EEG Model: KNN (5-Fold CV)")
    print("==============================")
    print(confusion_matrix(all_y_true, all_y_pred))
    print(classification_report(all_y_true, all_y_pred, digits=4))

    # Save model
    os.makedirs("saved_models", exist_ok=True)
    joblib.dump(model, "saved_models/eeg_knn_model.pkl")

    print("\nEEG KNN model saved at saved_models/eeg_knn_model.pkl")

    return model
