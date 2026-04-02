import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("dataset/final.csv")

X = df['text']
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Load models
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))
nb = pickle.load(open("model/nb_model.pkl", "rb"))
lr = pickle.load(open("model/lr_model.pkl", "rb"))
svm = pickle.load(open("model/svm_model.pkl", "rb"))

# Transform test data
X_test_vec = vectorizer.transform(X_test)

# Predictions
nb_pred = nb.predict(X_test_vec)
lr_pred = lr.predict(X_test_vec)
svm_pred = svm.predict(X_test_vec)

# Ensemble (Voting)
ensemble_pred = []
for i in range(len(nb_pred)):
    votes = [nb_pred[i], lr_pred[i], svm_pred[i]]
    final = int(round(sum(votes)/len(votes)))
    ensemble_pred.append(final)

# Function to calculate metrics
def evaluate(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred)
    }

# Evaluate all models
results = {
    "Naive Bayes": evaluate(y_test, nb_pred),
    "Logistic Regression": evaluate(y_test, lr_pred),
    "SVM": evaluate(y_test, svm_pred),
    "Ensemble": evaluate(y_test, ensemble_pred)
}

# Print results
for model, metrics in results.items():
    print(f"\n{model}")
    for k, v in metrics.items():
        print(f"{k}: {round(v*100, 2)}%")

# Prepare data for graph
models = list(results.keys())
accuracy = [results[m]["accuracy"]*100 for m in models]

plt.figure()
plt.bar(models, accuracy)

plt.xlabel("Models")
plt.ylabel("Accuracy (%)")
plt.title("Model Accuracy Comparison")

for i, v in enumerate(accuracy):
    plt.text(i, v + 0.5, f"{v:.2f}%", ha='center')

plt.savefig("accuracy_graph.png")
plt.show()

precision = [results[m]["precision"]*100 for m in models]
recall = [results[m]["recall"]*100 for m in models]
f1 = [results[m]["f1"]*100 for m in models]

x = range(len(models))

plt.figure()
plt.bar(x, precision)
plt.bar(x, recall, bottom=precision)
plt.bar(x, f1, bottom=[i+j for i,j in zip(precision, recall)])

plt.xticks(x, models)
plt.title("Precision, Recall, F1 Comparison")

plt.savefig("metrics_graph.png")
plt.show()

import numpy as np

cm = confusion_matrix(y_test, ensemble_pred)

plt.figure()
plt.imshow(cm)

plt.title("Confusion Matrix - Ensemble")
plt.xlabel("Predicted")
plt.ylabel("Actual")

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i][j], ha='center')

plt.savefig("confusion_matrix.png")
plt.show()