import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.externals import joblib
import pdb

app = Flask(__name__)
loaded_model = joblib.load('./model.pkl')
loaded_vec = joblib.load('./vectorizer.pkl')
label = {0: 'negative', 1: 'positive'}


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.form['review-text']
    vectorized_input = loaded_vec.transform([data])
    emotion = loaded_model.predict(vectorized_input)[0]
    proba = np.max(loaded_model.predict_proba(vectorized_input))

    return render_template('index.html', prediction_text='Emotion: {} with a probability of {}'.format(label[emotion], proba))


if __name__ == "__main__":
    app.run(debug=True)
