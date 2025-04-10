# utils/plot_summary.py

import matplotlib.pyplot as plt

def plot_accuracy_comparison(model_names, accuracies):
    plt.figure(figsize=(10, 5))
    bars = plt.bar(model_names, accuracies, color='mediumseagreen')
    plt.xticks(rotation=45)
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.ylim(0.5, 1.0)

    for bar, acc in zip(bars, accuracies):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f"{acc:.4f}", ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig("model_accuracy_comparison.png")
    print("📊 Accuracy chart saved to model_accuracy_comparison.png")
    plt.close()
