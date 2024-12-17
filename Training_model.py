import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
import joblib
import pickle
import os

# Read dataset
file_path = 'C:\\Users\\admin\\Programs\\.venv\VSCode\\Projects\\Practice\\7th-sem\\attempt-2\\share\\a1_RestaurantReviews_HistoricDump.tsv'
df = pd.read_csv(file_path, delimiter='\t')

# Display the dataset shape and an example review
print("Original dataset shape:", df.shape)
print(f"Example review: {df['Review'].values[0]}")

# Reduce the total rows in dataset for testing (optional)
# Uncomment the lines below to reduce the dataset size for quicker testing
# df = df.head(900)
# print("Reduced dataset shape:", df.shape)

# Example of a review before cleaning
example_review = df['Review'].values[100]
print(f"Example review before cleaning: {example_review}")

# Data Cleaning
nltk.download('stopwords')  # Download stopwords from nltk
ps = PorterStemmer()
stop_words = stopwords.words('english')
stop_words.remove('not')  # Retain 'not' as it is important for sentiment

corpus = []  # To store cleaned customer reviews
for i in range(0, 900):  # Iterate through the reviews
    review = re.sub('[^a-zA-Z]', ' ', df['Review'][i])  # Remove non-alphabet characters
    review = review.lower()  # Convert text to lowercase
    review = review.split()  # Split into words
    review = [ps.stem(word) for word in review if word not in stop_words]  # Stemming and removing stopwords
    review = ' '.join(review)  # Join words back into a string
    corpus.append(review)

# Inspect the first 5 cleaned reviews
print("First 5 cleaned reviews:")
print(corpus[:5])

# Dataset Transformation (Bag of Words Representation)
cv = CountVectorizer(max_features=1000)  # Convert text to Bag of Words, keep top 1000 features
X = cv.fit_transform(corpus).toarray()  # BoW features matrix
y = df.iloc[:, -1].values  # Target labels (assuming labels are in the last column)
print("Shape of Bag of Words matrix:", X.shape)

# Save BoW dictionary for later use
bow_model_path = 'C:\\Users\\admin\\Programs\\.venv\VSCode\\Projects\\Practice\\7th-sem\\attempt-2\\share\\BoW_Sentiment_Model.pkl'
pickle.dump(cv, open(bow_model_path, "wb"))

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Model Fitting (Naive Bayes Classifier)
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Ensure the model is saved properly
classifier_model_path = 'C:\\Users\\admin\\Programs\\.venv\VSCode\\Projects\\Practice\\7th-sem\\attempt-2\\share\\Classifier_Sentiment_Model'

# Check if the classifier model file exists
if not os.path.exists(classifier_model_path):
    print(f"Model file does not exist at {classifier_model_path}. Saving a new model...")
    # Save the trained Naive Bayes model
    joblib.dump(classifier, classifier_model_path)
else:
    print(f"Model file already exists at {classifier_model_path}. Re-saving to ensure no corruption...")
    # Re-save to avoid potential corruption or incorrect encoding
    joblib.dump(classifier, classifier_model_path)

# Model Evaluation (Testing)
y_pred = classifier.predict(X_test)

# Confusion Matrix and Accuracy Score
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {accuracy:.4f}")

# Optional: Plot the review count distribution by rating (uncomment to visualize)
# ax = df['rating'].value_counts().sort_index().plot(kind='bar', title='Ratings Count by Stars', figsize=(10, 5))
# ax.set_xlabel('Review Stars')
# plt.show()
