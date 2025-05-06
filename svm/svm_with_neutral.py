import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import zipfile
import os


# Unzip the dataset
csv_file_name = "tripadvisor_hotel_reviews.csv"
df = pd.read_csv(csv_file_name)

# Load the dataset
print("Dataset loaded successfully.")
print("Dataset head:\n", df.head())
print("\nDataset info:\n")
df.info()

    # Preprocessing
    # Drop rows with missing reviews or ratings if any
df.dropna(subset=['Review', 'Rating'], inplace=True)

    # Define a function to categorize ratings
def categorize_rating(rating):
    if rating <= 2:
        return 'bad'
    elif rating == 3:
        return 'neutral'
    else:
        return 'good'

    # Apply the categorization
df['Sentiment'] = df['Rating'].apply(categorize_rating)
print("\nSentiment distribution:\n", df['Sentiment'].value_counts())

# Select features (X) and target (y)
X = df['Review']
y = df['Sentiment']

    # Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

    # Train SVM model
print("\nTraining SVM model...")
svm_model = SVC(kernel='linear', C=1.0, random_state=42) # Linear kernel is often good for text
svm_model.fit(X_train_tfidf, y_train)
print("SVM model trained.")

    # Make predictions
y_pred = svm_model.predict(X_test_tfidf)

    # Evaluate the model
print("\nModel Evaluation:")
print(classification_report(y_test, y_pred, target_names=['bad', 'good', 'neutral'])) # Ensure order matches labels
   # Example of classifying new reviews
new_reviews = [
       "This hotel was fantastic, the service was excellent!",
        "The room was dirty and the staff were rude.",
        "It was an okay experience, nothing special."
    ]
new_reviews_tfidf = vectorizer.transform(new_reviews)
predictions = svm_model.predict(new_reviews_tfidf)

print("\nPredictions for new reviews:")
for review, sentiment in zip(new_reviews, predictions):
    print(f"Review: \"{review}\" -> Predicted Sentiment: {sentiment}")

