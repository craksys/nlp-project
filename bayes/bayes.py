import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

class NaiveBayesClassifier:
    def __init__(self, data_path="tripadvisor_hotel_reviews.csv"):
        self.data_path = data_path
        self.df = pd.read_csv(data_path)
        print("Dataset loaded successfully.")
        print("Dataset head:\n", self.df.head())
        print("\nDataset info:\n")
        self.df.info()
        
        self.df.dropna(subset=['Review', 'Rating'], inplace=True)
        self.setup_preprocessing()
        
    def setup_preprocessing(self):
        """Initialize NLTK components and download required data"""
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
            
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess_text(self, text):
        """Preprocess text by lowercasing, removing special characters, and lemmatizing"""
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens 
                 if word not in self.stop_words and word.isalpha()]
        return ' '.join(tokens)
    
    def categorize_rating(self, rating, mode='1_5'):
        """
        Categorize ratings based on the specified mode
        mode: '1_5' for raw ratings, 'with_neutral' for 3 classes, 'without_neutral' for binary
        """
        if mode == '1_5':
            return str(rating)
        elif mode == 'with_neutral':
            if rating <= 2:
                return 'negative'
            elif rating == 3:
                return 'neutral'
            else:
                return 'positive'
        else:  # without_neutral
            return 'positive' if rating >= 4 else 'negative'
    
    def prepare_data(self, mode='1_5', test_size=0.2, random_state=42):
        """
        Prepare data for training and testing
        mode: '1_5', 'with_neutral', or 'without_neutral'
        """
        print("\nPreprocessing reviews...")
        self.df['Processed_Review'] = self.df['Review'].apply(self.preprocess_text)
        print("Preprocessing complete.")
        print("Example of processed review:\n", self.df[['Review', 'Processed_Review']].head())
        
        self.df['Sentiment'] = self.df['Rating'].apply(
            lambda x: self.categorize_rating(x, mode=mode))
        print("\nRating distribution:\n", self.df['Sentiment'].value_counts().sort_index())
        
        X = self.df['Processed_Review']
        y = self.df['Sentiment']
        
        return train_test_split(X, y, test_size=test_size, 
                              random_state=random_state, stratify=y)
    
    def train_and_evaluate(self, mode='1_5', test_size=0.2, random_state=42, X_train=None, X_test=None, y_train=None, y_test=None):
        """
        Train the Naive Bayes model and evaluate its performance
        Returns: model, vectorizer, metrics dictionary, and predictions
        """
        if X_train is None or X_test is None or y_train is None or y_test is None:
            X_train, X_test, y_train, y_test = self.prepare_data(
                mode=mode, test_size=test_size, random_state=random_state)
        
        # Vectorize the text data
        vectorizer = TfidfVectorizer(max_features=5000)
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        
        # Train the model
        print("\nTraining Naive Bayes model...")
        model = MultinomialNB()
        model.fit(X_train_tfidf, y_train)
        print("Naive Bayes model trained.")
        
        # Make predictions
        y_pred = model.predict(X_test_tfidf)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted')
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'classification_report': classification_report(y_test, y_pred, target_names=sorted(y_test.unique()))
        }
        
        return model, vectorizer, metrics, y_pred, y_test
    
    def predict_new_reviews(self, model, vectorizer, new_reviews):
        """Predict sentiment for new reviews"""
        print("\nPreprocessing new reviews for prediction...")
        processed_reviews = [self.preprocess_text(review) for review in new_reviews]
        print("Preprocessing of new reviews complete.")
        
        new_reviews_tfidf = vectorizer.transform(processed_reviews)
        predictions = model.predict(new_reviews_tfidf)
        
        results = []
        for original, processed, pred in zip(new_reviews, processed_reviews, predictions):
            results.append({
                'original_review': original,
                'processed_review': processed,
                'predicted_rating': pred
            })
        
        return results 