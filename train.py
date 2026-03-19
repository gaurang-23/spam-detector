import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# Load dataset
df = pd.read_csv("dataset/final.csv")

X = df['text']
y = df['label']

# TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_vec = vectorizer.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

# Models
nb = MultinomialNB()
lr = LogisticRegression()
svm = LinearSVC()

# Train
nb.fit(X_train, y_train)
lr.fit(X_train, y_train)
svm.fit(X_train, y_train)

# Save
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))
pickle.dump(nb, open("model/nb_model.pkl", "wb"))
pickle.dump(lr, open("model/lr_model.pkl", "wb"))
pickle.dump(svm, open("model/svm_model.pkl", "wb"))

print("Training complete!")