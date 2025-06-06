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

'''df.isnull().sum()
df.head()
df.tail()'''
df=df.drop(columns=['Unnamed: 0'])

#df.head()
#df.info()

vectorizer = TfidfVectorizer(max_features=5000)  # Limit to top 5000 features
X = vectorizer.fit_transform(df['Sentence'])
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt=DecisionTreeClassifier()
dt.fit(X_train,y_train)

y_pred = dt.predict(X_test)

'''print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n", classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", conf_matrix)



sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greens')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()'''



# Save model and vectorizer
joblib.dump(dt, 'xss_classifier.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

new_sentences = ["<div class=\"thumbcaption\">", "<svg onload=\"alert('XSS')\"></svg>"]

loaded_model = joblib.load('xss_classifier.pkl')
loaded_vectorizer = joblib.load('tfidf_vectorizer.pkl')

new_X = loaded_vectorizer.transform(new_sentences)
new_predictions = loaded_model.predict(new_X)

print("Predictions:", new_predictions)

import joblib
import pandas as pd

model = joblib.load('xss_classifier.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

def predict(text):
    text_series = pd.Series([text])  # Convert the single string to a pandas Series
    text_vectorized = vectorizer.transform(text_series)
    prediction = model.predict(text_vectorized)[0] # Get the prediction, [0] to get scalar instead of array
    return int(prediction)  # Return as an integer (0 or 1)

