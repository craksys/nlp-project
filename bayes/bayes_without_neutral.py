import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
try:
    word_tokenize("test")
except LookupError:
    nltk.download('punkt')
try:
    WordNetLemmatizer().lemmatize("test")
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')


csv_file_name = "tripadvisor_hotel_reviews.csv"
df = pd.read_csv(csv_file_name)

print("Dataset loaded successfully.")
print("Dataset head:\n", df.head())
print("\nDataset info:\n")
df.info()

df.dropna(subset=['Review', 'Rating'], inplace=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and word.isalpha()]
    return ' '.join(tokens)

print("\nPreprocessing reviews...")
df['Processed_Review'] = df['Review'].apply(preprocess_text)
print("Preprocessing complete.")
print("Example of processed review:\n", df[['Review', 'Processed_Review']].head())

def categorize_rating(rating):
    if rating <= 3: # Changed condition for 'bad'
        return 'bad'
    else:
        return 'good'

df['Sentiment'] = df['Rating'].apply(categorize_rating)
print("\nSentiment distribution:\n", df['Sentiment'].value_counts())

X = df['Processed_Review']
y = df['Sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nTraining Naive Bayes model...")
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
print("Naive Bayes model trained.")

y_pred = nb_model.predict(X_test_tfidf)

print("\nModel Evaluation:")
# Ensure target_names match the actual unique sorted labels in y_test or y_pred
# For binary 'bad', 'good', sorted unique labels will be ['bad', 'good']
print(classification_report(y_test, y_pred, target_names=sorted(y.unique())))

new_reviews = [
       "This hotel was fantastic, the service was excellent!",
        "The room was dirty and the staff were rude.",
        "It was an okay experience, nothing special."
    ]

print("\nPreprocessing new reviews for prediction...")
processed_new_reviews = [preprocess_text(review) for review in new_reviews]
print("Preprocessing of new reviews complete.")

new_reviews_tfidf = vectorizer.transform(processed_new_reviews)
predictions = nb_model.predict(new_reviews_tfidf)

print("\nPredictions for new reviews:")
for original_review, processed_review, sentiment in zip(new_reviews, processed_new_reviews, predictions):
    print(f"Original Review: \"{original_review}\"")
    print(f"Processed Review: \"{processed_review}\" -> Predicted Sentiment: {sentiment}")
