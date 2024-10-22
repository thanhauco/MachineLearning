"""
using the IMDB Movie Reviews dataset. We'll create a simple sentiment classifier using a Naive Bayes model. This dataset is available through TensorFlow datasets, which we'll use to fetch the data online.
Here's a Python script for sentiment analysis on IMDB movie reviews:
"""

"""
This script does the following:
Loads the IMDB movie reviews dataset using TensorFlow Datasets.
Converts the dataset to numpy arrays for easier processing.
Splits the training data into train and validation sets.
Creates a bag of words representation of the text data using CountVectorizer.
Trains a Multinomial Naive Bayes classifier.
Evaluates the model on validation and test sets.
Prints a classification report with precision, recall, and F1-score.
Provides a function to predict the sentiment of new reviews.
The script will download the IMDB dataset automatically when you run it for the first time. This might take a few minutes depending on your internet connection.
This is a basic approach to sentiment analysis. To improve the model, you could:
Use more advanced text representation techniques like TF-IDF or word embeddings.
Try more sophisticated models like LSTM or BERT.
Perform more extensive text preprocessing (e.g., removing stopwords, stemming).
Use cross-validation for more robust evaluation.

"""

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

import ssl

ssl._create_default_https_context = ssl._create_unverified_context


# Load the 20 Newsgroups dataset
print("Loading 20 Newsgroups dataset...")
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)

print("Dataset loaded successfully!")
print(f"Number of training examples: {len(twenty_train.data)}")
print(f"Number of testing examples: {len(twenty_test.data)}")

# Create a bag of words representation
print("\nCreating bag of words representation...")
vectorizer = CountVectorizer(max_features=5000)
X_train_bow = vectorizer.fit_transform(twenty_train.data)
X_test_bow = vectorizer.transform(twenty_test.data)

# Train a Naive Bayes classifier
print("Training Naive Bayes classifier...")
clf = MultinomialNB()
clf.fit(X_train_bow, twenty_train.target)

# Make predictions on test set
test_predictions = clf.predict(X_test_bow)

# Calculate accuracy on test set
test_accuracy = accuracy_score(twenty_test.target, test_predictions)
print(f"\nTest Accuracy: {test_accuracy:.4f}")

# Print classification report
print("\nClassification Report:")
print(classification_report(twenty_test.target, test_predictions, target_names=twenty_train.target_names))

# Function to predict category of a new text
def predict_category(text):
    vectorized_text = vectorizer.transform([text])
    prediction = clf.predict(vectorized_text)
    category = twenty_train.target_names[prediction[0]]
    return category

# Example predictions
print("\nExample predictions:")
texts = [
    "The Bible teaches us about God's love.",
    "New graphics card released with improved performance.",
    "Recent medical study shows promising results for cancer treatment.",
    "Discussing the existence of a higher power."
]

for text in texts:
    category = predict_category(text)
    print(f"Text: '{text}'")
    print(f"Predicted category: {category}\n")