from flask import Flask, render_template, request
import pickle
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the vectorizer
with open('vectorization.pkl', 'rb') as file:
    vectorization = pickle.load(file)

# Load the saved model
with open('rfc_model.pkl', 'rb') as file:
    rfc = pickle.load(file)

def wordopt(text):
    # Convert into lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'https?://\S+|www.\S+', '', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Remove digits
    text = re.sub(r'\d', '', text)

    # Remove newline characters
    text = re.sub(r'\n', ' ', text)

    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    text = wordopt(text)
    x = vectorization.transform([text])
    pred = rfc.predict(x)
    prediction = "Uh-oh! Your article is likely fake." if pred[0] == 0 else "Great news! Your article is likely genuine."
    css_class = "fake" if pred[0] == 0 else "genuine"
    return render_template('predict.html', prediction=prediction, css_class=css_class)

if __name__ == '__main__':
    app.run(debug=True)