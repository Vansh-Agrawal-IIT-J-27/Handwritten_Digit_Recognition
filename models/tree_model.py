# models/tree_model.py

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class DecisionTreeModel:
    def __init__(self, max_depth=None):
        self.model = DecisionTreeClassifier(max_depth=max_depth)

    def train(self, X_train, y_train):
        print("📚 Training Decision Tree model...")
        self.model.fit(X_train, y_train)
        print("✅ Training complete.")

    def evaluate(self, X_test, y_test):
        print("📈 Evaluating Decision Tree model...")
        y_pred = self.model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        print(f"✅ Accuracy: {acc:.4f}")
        print("📊 Classification Report:\n", report)
        return acc, cm, report
