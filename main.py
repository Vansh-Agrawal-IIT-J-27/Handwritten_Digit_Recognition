# main.py

import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from models.knn_model import KNNModel
from models.svm_model import SVMModel
from models.tree_model import DecisionTreeModel
from models.pca_knn_model import PCA_KNN_Model
from models.naive_bayes_model import NaiveBayesModel
from models.logistic_regression_model import LogisticRegressionModel
from models.random_forest_model import RandomForestModel
from models.lda_model import LDAModel
from utils.evaluation import plot_confusion_matrix
from utils.plot_summary import plot_accuracy_comparison

def load_data():
    print("📦 Loading MNIST dataset from data/mnist.npz...")
    with np.load("data/mnist.npz") as data:
        X_train = data["x_train"]
        y_train = data["y_train"]
        X_test = data["x_test"]
        y_test = data["y_test"]

    # Flatten images from (28, 28) → (784,)
    X_train = X_train.reshape(-1, 28 * 28).astype("float32")
    X_test = X_test.reshape(-1, 28 * 28).astype("float32")

    print(f"✅ Loaded: {X_train.shape[0] + X_test.shape[0]} samples, each with {X_train.shape[1]} features")
    return X_train, X_test, y_train, y_test

def visualize_samples(X, y):
    print("🖼️ Showing 10 sample digits...")
    plt.figure(figsize=(10, 2))
    for i in range(10):
        plt.subplot(1, 10, i + 1)
        plt.imshow(X[i].reshape(28, 28), cmap="gray")
        plt.title(str(int(y[i])))
        plt.axis("off")
    plt.tight_layout()
    plt.show()

def preprocess(X_train, X_test):
    print("🧪 Normalizing data using StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# ================= MAIN =================
if __name__ == "__main__":
    # Step 1: Load & preprocess data
    X_train, X_test, y_train, y_test = load_data()
    visualize_samples(X_train, y_train)
    X_train, X_test = preprocess(X_train, X_test)
    print("✅ Preprocessing complete. Data is ready for ML models.\n")

    # ========== KNN ==========
    knn = KNNModel(n_neighbors=3)
    knn.train(X_train, y_train)
    knn_acc, knn_cm, _ = knn.evaluate(X_test, y_test)
    plot_confusion_matrix(y_test, knn.model.predict(X_test), title="KNN Confusion Matrix", save_path="confusion_matrices/knn.png")
    print(f"✅ KNN Accuracy: {knn_acc:.4f}\n")

    # ========== SVM ==========
    svm = SVMModel(kernel='rbf', C=1.0)
    svm.train(X_train, y_train)
    svm_acc, svm_cm, _ = svm.evaluate(X_test, y_test)
    plot_confusion_matrix(y_test, svm.model.predict(X_test), title="SVM Confusion Matrix", save_path="confusion_matrices/svm.png")
    print(f"✅ SVM Accuracy: {svm_acc:.4f}\n")

    # ========== Decision Tree ==========
    tree = DecisionTreeModel(max_depth=20)
    tree.train(X_train, y_train)
    tree_acc, tree_cm, _ = tree.evaluate(X_test, y_test)
    plot_confusion_matrix(y_test, tree.model.predict(X_test), title="Decision Tree Confusion Matrix", save_path="confusion_matrices/tree.png")
    print(f"✅ Decision Tree Accuracy: {tree_acc:.4f}\n")

    # ========== PCA + KNN ==========
    pca_knn = PCA_KNN_Model(n_components=50, n_neighbors=3)
    pca_knn.train(X_train, y_train)
    pca_knn_acc, pca_knn_cm, _ = pca_knn.evaluate(X_test, y_test)
    plot_confusion_matrix(y_test, pca_knn.predict(X_test), title="PCA + KNN Confusion Matrix", save_path="confusion_matrices/pca_knn.png")
    print(f"✅ PCA + KNN Accuracy: {pca_knn_acc:.4f}\n")
    
    # ========== Naive Bayes ==========
    nb = NaiveBayesModel()
    nb.train(X_train, y_train)
    nb_acc, nb_cm, _ = nb.evaluate(X_test, y_test)
    plot_confusion_matrix(y_test, nb.model.predict(X_test), title="Naive Bayes Confusion Matrix", save_path="confusion_matrices/nb.png")
    print(f"✅ Naive Bayes Accuracy: {nb_acc:.4f}\n")

    # ========== Logistic Regression ==========
    lr = LogisticRegressionModel()
    lr.train(X_train, y_train)
    lr_acc, lr_cm, _ = lr.evaluate(X_test, y_test)
    plot_confusion_matrix(y_test, lr.model.predict(X_test), title="Logistic Regression Confusion Matrix", save_path="confusion_matrices/lr.png")
    print(f"✅ Logistic Regression Accuracy: {lr_acc:.4f}\n")

    # ========== Random Forest ==========
    rf = RandomForestModel(n_estimators=100)
    rf.train(X_train, y_train)
    rf_acc, rf_cm, _ = rf.evaluate(X_test, y_test)
    plot_confusion_matrix(y_test, rf.model.predict(X_test), title="Random Forest Confusion Matrix", save_path="confusion_matrices/rf.png")
    print(f"✅ Random Forest Accuracy: {rf_acc:.4f}\n")

    # ========== LDA ==========
    lda = LDAModel()
    lda.train(X_train, y_train)
    lda_acc, lda_cm, _ = lda.evaluate(X_test, y_test)
    plot_confusion_matrix(y_test, lda.model.predict(X_test), title="LDA Confusion Matrix", save_path="confusion_matrices/lda.png")
    print(f"✅ LDA Accuracy: {lda_acc:.4f}\n")

    # ========== Summary ==========

    print("📊 Model Evaluation Summary:")
    print(f"KNN Accuracy: {knn_acc:.4f}")
    print(f"SVM Accuracy: {svm_acc:.4f}")
    print(f"Decision Tree Accuracy: {tree_acc:.4f}")
    print(f"Naive Bayes Accuracy: {nb_acc:.4f}")
    print(f"Logistic Regression Accuracy: {lr_acc:.4f}")
    print(f"Random Forest Accuracy: {rf_acc:.4f}")
    print(f"LDA Accuracy: {lda_acc:.4f}")
    print("✅ All models evaluated successfully!")

    with open("results.txt", "w") as f:
        f.write(f"KNN Accuracy: {knn_acc:.4f}\n")
        f.write(f"SVM Accuracy: {svm_acc:.4f}\n")
        f.write(f"Decision Tree Accuracy: {tree_acc:.4f}\n")
        f.write(f"PCA + KNN Accuracy: {pca_knn_acc:.4f}\n")
        f.write(f"Naive Bayes Accuracy: {nb_acc:.4f}\n")
        f.write(f"Logistic Regression Accuracy: {lr_acc:.4f}\n")
        f.write(f"Random Forest Accuracy: {rf_acc:.4f}\n")
        f.write(f"LDA Accuracy: {lda_acc:.4f}\n")
    print("📝 All accuracies saved to results.txt")

    model_names = ["KNN", "SVM", "Decision Tree", "PCA+KNN", "Naive Bayes", "LogReg", "Rand Forest", "LDA"]
    accuracies = [knn_acc, svm_acc, tree_acc, pca_knn_acc, nb_acc, lr_acc, rf_acc, lda_acc]

    plot_accuracy_comparison(model_names, accuracies)
    print("📈 Accuracy comparison plot saved as accuracy_comparison.png")

    joblib.dump(svm_model, 'model/svm_model.pkl')

    print("✅ All tasks completed successfully!")
    print("🔚 End of script.")
    print("======================================")
    
