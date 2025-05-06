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

    # Target variable is the 'Rating' itself
    # Ensure 'Rating' is treated as a categorical label, converting to string is a safe way
df['Sentiment_Category'] = df['Rating'].astype(str) 
print("\nRating distribution:\n", df['Rating'].value_counts().sort_index())

# Select features (X) and target (y)
X = df['Review']
y = df['Sentiment_Category'] # Use the string version of ratings as target

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
# Ensure target_names match the unique sorted string values in y_test
target_names = sorted(y_test.unique()) 
print(classification_report(y_test, y_pred, target_names=target_names)) 
   # Example of classifying new reviews
new_reviews = [
       "This hotel was fantastic, the service was excellent!",
        "The room was dirty and the staff were rude.",
        "It was an okay experience, nothing special."
    ]
new_reviews_tfidf = vectorizer.transform(new_reviews)
predictions = svm_model.predict(new_reviews_tfidf)

print("\nPredictions for new reviews:")
for review, rating_prediction in zip(new_reviews, predictions):
    print(f"Review: \"{review}\" -> Predicted Rating: {rating_prediction} stars")
