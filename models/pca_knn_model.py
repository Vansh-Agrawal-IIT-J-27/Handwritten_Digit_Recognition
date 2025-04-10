# models/pca_knn_model.py

from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class PCA_KNN_Model:
    def __init__(self, n_components=50, n_neighbors=3):
        self.pca = PCA(n_components=n_components)
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    def train(self, X_train, y_train):
        print("📉 Reducing data with PCA...")
        self.X_train_pca = self.pca.fit_transform(X_train)
        print("📚 Training KNN on PCA-reduced data...")
        self.knn.fit(self.X_train_pca, y_train)

    def evaluate(self, X_test, y_test):
        X_test_pca = self.pca.transform(X_test)
        y_pred = self.knn.predict(X_test_pca)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        print(f"✅ PCA + KNN Accuracy: {acc:.4f}")
        print("📊 Classification Report:\n", report)
        return acc, cm, report

    def predict(self, X):  # for confusion matrix plot
        return self.knn.predict(self.pca.transform(X))
