# models/svm_model.py

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class SVMModel:
    def __init__(self, kernel='rbf', C=1.0):
        self.model = SVC(kernel=kernel, C=C)

    def train(self, X_train, y_train):
        print("📚 Training SVM model...")
        self.model.fit(X_train, y_train)
        print("✅ Training complete.")

    def evaluate(self, X_test, y_test):
        print("📈 Evaluating SVM model...")
        y_pred = self.model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        print(f"✅ Accuracy: {acc:.4f}")
        print("📊 Classification Report:\n", report)
        return acc, cm, report
