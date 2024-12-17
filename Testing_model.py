import numpy as np
import pandas as pd
import re
import nltk
import pickle
import joblib
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

# Download NLTK data
nltk.download('stopwords')

# Read dataset
df = pd.read_csv('C:\\Users\\admin\\Programs\\.venv\VSCode\\Projects\\Practice\\7th-sem\\attempt-2\\share\\a2_RestaurantReviews_FreshDump.tsv', delimiter='\t')

# Initialize stopwords and PorterStemmer
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')

# Create a cleaned and stemmed corpus
corpus = []
for i in range(0, len(df)):
    review = re.sub('[^a-zA-Z]', ' ', df['Review'][i])  # Keep only letters
    review = review.lower()  # Convert to lowercase
    review = review.split()  # Tokenize
    review = [ps.stem(word) for word in review if word not in set(all_stopwords)]
    review = ' '.join(review)  # Rejoin tokens into a string
    corpus.append(review)

# Load CountVectorizer
cvFile = 'C:\\Users\\admin\\Programs\\.venv\VSCode\\Projects\\Practice\\7th-sem\\attempt-2\\share\\BoW_Sentiment_Model.pkl'
try:
    with open(cvFile, 'rb') as f:
        cv = pickle.load(f)
    print("CountVectorizer loaded successfully from training.")
    # Print the vocabulary or feature names
    print("Vocabulary in CountVectorizer:")
    print(cv.get_feature_names_out())
except FileNotFoundError:
    raise FileNotFoundError("BoW_Sentiment_Model.pkl not found. Ensure the training script saved the correct file.")
    cv = CountVectorizer(max_features=1000)
    cv.fit(corpus)
    with open(cvFile, 'wb') as f:
        pickle.dump(cv, f)
    print("CountVectorizer created and saved successfully.")

# Transform the corpus into feature vectors
X_fresh = cv.transform(corpus).toarray()

# Load the pre-trained classifier
classifierFile = 'C:\\Users\\admin\\Programs\\.venv\VSCode\\Projects\\Practice\\7th-sem\\attempt-2\\share\\Classifier_Sentiment_Model'
try:
    classifier = joblib.load(classifierFile)
    print("Classifier loaded successfully.")
    # Inspect the classifier's parameters (e.g., coefficients for each feature)
    print("Classifier coefficients:")
    print(classifier.theta_)  # For GaussianNB, this contains the log-probabilities of each feature
except FileNotFoundError:
    print("Classifier file not found. Please ensure the correct file path.")

# Predict sentiments
y_pred = classifier.predict(X_fresh)

# Add predictions to the DataFrame
df['predicted_label'] = y_pred.tolist()

# Save the DataFrame with predictions
output_file = "C:\\Users\\admin\\Programs\\.venv\VSCode\\Projects\\Practice\\7th-sem\\attempt-2\\share\\Final_Predicted_Sentiments_Fresh_Dump.tsv"
print(f"File saved to: {output_file}")
df.to_csv(output_file, sep='\t', encoding='UTF-8', index=False)

