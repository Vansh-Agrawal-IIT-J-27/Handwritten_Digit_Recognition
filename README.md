# Handwritten Digit Recognition

This project demonstrates handwritten digit recognition using multiple traditional machine learning models applied to the MNIST dataset. We evaluated and compared the performance of each model and visualized results using confusion matrices and accuracy charts.

---

## Setup Instructions

1. **Clone the Repository**
```bash
git clone https://github.com/Vansh-Agrawal-IIT-J-27/Handwritten_Digit_Recognition.git
cd Handwritten_Digit_Recognition
```

2. **Install Requirements**
```bash
pip install -r requirements.txt
```

3. **Run the Main Script**
```bash
python main.py
```

All confusion matrices and model accuracy summaries will be saved automatically.

---

## Models Implemented

-  K-Nearest Neighbors (KNN)
-  Support Vector Machine (SVM)
-  Decision Tree
-  PCA + KNN
-  Naive Bayes
-  Logistic Regression
-  Random Forest
-  Linear Discriminant Analysis (LDA)

Each model is evaluated using classification reports and confusion matrices.

---

##  Evaluation Outputs

| File                             | Description                              |
|----------------------------------|------------------------------------------|
| `results.txt`                    | Accuracy of all models                   |
| `confusion_matrices/*.png`       | Confusion matrices for each model        |
| `model_accuracy_comparison.png`  | Visual bar chart of model accuracies     |

---

##  Team Members & Contributions

| S. No. | Name                  | Contribution                                  |
|--------|-----------------------|-----------------------------------------------|
| 1.     | **Vansh Agrawal**     | Code architecture, model training, GitHub     |
| 2.     | Chandavath Akhil      | Data preprocessing, SVM & KNN models          |
| 3.     | Sapavath Gharulal     | Logistic Regression, accuracy evaluation      |
| 4.     | Jatavath Sudheer      | Decision Tree, Random Forest models           |
| 5.     | Banoth Mallesh        | LDA, Naive Bayes implementation               |
| 6.     | Golla Sathvik         | Confusion matrix visualization, summary chart |

---

##  Folder Structure

```
Handwritten_Digit_Recognition/
│
├── main.py
├── results.txt
├── requirements.txt
├── README.md
├── model_accuracy_comparison.png
│
├── data/
│   └── mnist.npz
│
├── models/
│   ├── knn_model.py
│   ├── svm_model.py
│   ├── tree_model.py
│   ├── pca_knn_model.py
│   ├── naive_bayes_model.py
│   ├── logistic_regression_model.py
│   ├── random_forest_model.py
│   └── lda_model.py
│
├── utils/
│   ├── evaluation.py
│   └── plot_summary.py
│
└── confusion_matrices/
    ├── knn.png
    ├── svm.png
    ├── tree.png
    ├── pca_knn.png
    ├── nb.png
    ├── lr.png
    ├── rf.png
    └── lda.png
```

---

##  License

This project is for academic and learning purposes only.
