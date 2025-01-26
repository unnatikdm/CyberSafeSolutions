"""
Spam Detection Model
This implements a spam detection system using the Multinomial Naive Bayes algorithm.
It preprocesses text data, balances the dataset, and uses TF-IDF for feature extraction.
The model is tuned using GridSearchCV and evaluated using accuracy, precision, recall, and F1-score.
A real-time prediction function is provided to classify messages as spam or legitimate.
"""

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import resample
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

df = pd.read_csv('spam.csv', encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

df['message'] = df['message'].apply(preprocess_text)

df['label'] = df['label'].map({'spam': 1, 'ham': 0})

df_spam = df[df['label'] == 1]
df_ham = df[df['label'] == 0]
df_spam_upsampled = resample(df_spam, replace=True, n_samples=len(df_ham), random_state=42)
df_balanced = pd.concat([df_ham, df_spam_upsampled])

X_train, X_test, y_train, y_test = train_test_split(df_balanced['message'], df_balanced['label'], test_size=0.2, random_state=42)

tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

param_grid = {'alpha': [0.1, 0.5, 1.0, 2.0]}
grid_search = GridSearchCV(MultinomialNB(), param_grid, cv=5, scoring='recall')
grid_search.fit(X_train_tfidf, y_train)

best_model = grid_search.best_estimator_
best_model.fit(X_train_tfidf, y_train)

y_pred_best = best_model.predict(X_test_tfidf)

print("\nPerformance Metrics (Best Model):")
print("- Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred_best) * 100))
print("- Precision (Spam): {:.2f}%".format(classification_report(y_test, y_pred_best, target_names=['ham', 'spam'], output_dict=True)['spam']['precision'] * 100))
print("- Recall (Spam): {:.2f}%".format(classification_report(y_test, y_pred_best, target_names=['ham', 'spam'], output_dict=True)['spam']['recall'] * 100))
print("- F1-Score (Spam): {:.2f}%".format(classification_report(y_test, y_pred_best, target_names=['ham', 'spam'], output_dict=True)['spam']['f1-score'] * 100))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_best))

def predict_spam(message, threshold=0.7):
    message_tfidf = tfidf.transform([message])
    prediction_proba = best_model.predict_proba(message_tfidf)[:, 1]
    prediction = 1 if prediction_proba >= threshold else 0
    return "Spam" if prediction == 1 else "Legitimate"
