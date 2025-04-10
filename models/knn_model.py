# models/knn_model.py

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class KNNModel:
    def __init__(self, n_neighbors=3):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)

    def train(self, X_train, y_train):
        print("📚 Training KNN model...")
        self.model.fit(X_train, y_train)
        print("✅ Training complete.")

    def evaluate(self, X_test, y_test):
        print("📈 Evaluating KNN model...")
        y_pred = self.model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        print(f"✅ Accuracy: {acc:.4f}")
        print("📊 Classification Report:\n", report)
        return acc, cm, report
