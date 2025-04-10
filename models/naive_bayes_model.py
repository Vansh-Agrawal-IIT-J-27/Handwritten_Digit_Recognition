# models/naive_bayes_model.py

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class NaiveBayesModel:
    def __init__(self):
        self.model = GaussianNB()

    def train(self, X_train, y_train):
        print("📚 Training Naive Bayes model...")
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        print("📈 Evaluating Naive Bayes model...")
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        print(f"✅ Accuracy: {acc:.4f}")
        print("📊 Classification Report:\n", report)
        return acc, cm, report
