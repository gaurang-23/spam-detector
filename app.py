from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load models
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))
nb = pickle.load(open("model/nb_model.pkl", "rb"))
lr = pickle.load(open("model/lr_model.pkl", "rb"))
svm = pickle.load(open("model/svm_model.pkl", "rb"))

def explain(text):
    words = text.split()
    word_scores = {}

    for word in words:
        vec = vectorizer.transform([word])
        prob = nb.predict_proba(vec)[0][1]
        word_scores[word] = prob

    top_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
    return top_words[:5]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['message']

    if not text.strip():
        return render_template('index.html', error="Empty input!")

    vec = vectorizer.transform([text])

    # Predictions
    nb_pred = nb.predict(vec)[0]
    lr_pred = lr.predict(vec)[0]
    svm_pred = svm.predict(vec)[0]

    votes = [int(nb_pred), int(lr_pred), int(svm_pred)]
    final = int(round(sum(votes)/len(votes)))

    # Confidence
    nb_prob = nb.predict_proba(vec)[0][1]
    lr_prob = lr.predict_proba(vec)[0][1]
    prob = (0.4 * nb_prob) + (0.6 * lr_prob)

    explanation = explain(text)

    return render_template(
        'index.html',
        prediction="Spam" if final else "Ham",
        confidence=round(prob*100, 2),
        votes=votes,
        explanation=explanation
    )

if __name__ == "__main__":
    app.run()