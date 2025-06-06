import pandas as pd
import joblib

model = joblib.load('xss_classifier.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

def predict(text):
    try:
        text_series = pd.Series([text])  # Convert the single string to a pandas Series
        text_vectorized = vectorizer.transform(text_series)
        prediction = model.predict(text_vectorized)[0] # Get the prediction, [0] to get scalar instead of array
        return int(prediction)  # Return as an integer (0 or 1)
    except Exception as e:
        print(f"Prediction error: {e}")
        return -1  # Or raise an appropriate HTTP error in the Flask app

