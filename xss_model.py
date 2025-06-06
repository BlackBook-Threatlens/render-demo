import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import joblib


df=pd.read_csv("XSS_dataset.csv")

df=df.drop(columns=['Unnamed: 0'])

vectorizer = TfidfVectorizer(max_features=5000)  # Limit to top 5000 features
X = vectorizer.fit_transform(df['Sentence'])
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt=DecisionTreeClassifier()
dt.fit(X_train,y_train)

y_pred = dt.predict(X_test)

# Save model and vectorizer
#joblib.dump(dt, 'xss_classifier.pkl')
#joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

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

