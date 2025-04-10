# models/lda_model.py

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class LDAModel:
    def __init__(self):
        self.model = LinearDiscriminantAnalysis()

    def train(self, X_train, y_train):
        print("📚 Training LDA model...")
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        print("📈 Evaluating LDA model...")
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        print(f"✅ Accuracy: {acc:.4f}")
        print("📊 Classification Report:\n", report)
        return acc, cm, report
