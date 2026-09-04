from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import joblib
import os


def train_ecg_knn(X, y, k=5):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    model = KNeighborsClassifier(n_neighbors=k)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n==============================")
    print("ECG Model: KNN")
    print("==============================")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, digits=4))

    # Save model
    os.makedirs("saved_models", exist_ok=True)
    joblib.dump(model, "saved_models/ecg_knn_model.pkl")

    print("\nECG KNN model saved at saved_models/ecg_knn_model.pkl")

    return model