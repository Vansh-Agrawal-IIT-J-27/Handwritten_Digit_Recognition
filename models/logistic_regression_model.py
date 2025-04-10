from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class LogisticRegressionModel:
    def __init__(self):
        self.model = LogisticRegression(
            max_iter=300,
            solver='saga',
            n_jobs=-1
        )

    def train(self, X_train, y_train):
        print("📚 Training Logistic Regression model on 10k samples...")
        # Limit data to 10k samples to avoid long training time
        X_small = X_train[:10000]
        y_small = y_train[:10000]
        self.model.fit(X_small, y_small)

    def evaluate(self, X_test, y_test):
        print("📈 Evaluating Logistic Regression model...")
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        print(f"✅ Accuracy: {acc:.4f}")
        print("📊 Classification Report:\n", report)
        return acc, cm, report
