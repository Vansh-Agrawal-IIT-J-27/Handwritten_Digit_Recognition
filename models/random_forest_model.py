# models/random_forest_model.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class RandomForestModel:
    def __init__(self, n_estimators=100):
        self.model = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1)

    def train(self, X_train, y_train):
        print("📚 Training Random Forest model...")
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        print("📈 Evaluating Random Forest model...")
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        print(f"✅ Accuracy: {acc:.4f}")
        print("📊 Classification Report:\n", report)
        return acc, cm, report
