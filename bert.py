import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, roc_auc_score
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

class BERTClassifier:
    def __init__(self, data_path="tripadvisor_hotel_reviews.csv", model_name="google-bert/bert-base-uncased"):
        self.data_path = data_path
        self.model_name = model_name
        self.df = pd.read_csv(data_path)
        self.df.dropna(subset=['Review', 'Rating'], inplace=True)
        self.setup_preprocessing()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.BATCH_SIZE = 32
        self.EPOCHS = 1
        self.MAX_LEN = 128
        self.LEARNING_RATE = 2e-5
        self.RANDOM_STATE = 42
        
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
        text = re.sub(r'[^a-z0-9\s]', '', text)  # Keep numbers for BERT
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
        
        X = self.df['Processed_Review'].values
        y = self.df['Sentiment'].values
        
        return train_test_split(X, y, test_size=test_size, 
                              random_state=random_state, stratify=y)
    
    def tokenize_data(self, texts):
        """Tokenize the input texts for BERT"""
        input_ids = []
        attention_masks = []
        
        for text in texts:
            encoded_dict = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.MAX_LEN,
                padding='max_length',
                return_attention_mask=True,
                return_tensors='pt',
                truncation=True
            )
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])
            
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        return input_ids, attention_masks
    
    def train_and_evaluate(self, mode='1_5', test_size=0.2, random_state=42, X_train=None, X_test=None, y_train=None, y_test=None):
        """
        Train the BERT model and evaluate its performance
        Returns: model, tokenizer, metrics dictionary, and predictions
        """
        if X_train is None or X_test is None or y_train is None or y_test is None:
            X_train, X_test, y_train, y_test = self.prepare_data(
                mode=mode, test_size=test_size, random_state=random_state)
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)
        num_labels = len(label_encoder.classes_)
        print(f"\nEncoded labels: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
        
        # Initialize tokenizer and model
        print(f"\nLoading BERT tokenizer: {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        print("Tokenizing training data...")
        train_input_ids, train_attention_masks = self.tokenize_data(X_train)
        print("Tokenizing test data...")
        test_input_ids, test_attention_masks = self.tokenize_data(X_test)
        
        # Create DataLoader
        train_labels = torch.tensor(y_train_encoded)
        test_labels = torch.tensor(y_test_encoded)
        
        train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=self.BATCH_SIZE)
        
        test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_labels)
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=self.BATCH_SIZE)
        
        # Initialize model
        print(f"\nLoading BERT model for sequence classification: {self.model_name}...")
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels,
            output_attentions=False,
            output_hidden_states=False
        )
        model.to(self.device)
        
        # Setup optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=self.LEARNING_RATE, eps=1e-8)
        total_steps = len(train_dataloader) * self.EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        # Training loop
        print("\nStarting training...")
        for epoch_i in range(0, self.EPOCHS):
            print(f"\n======== Epoch {epoch_i + 1} / {self.EPOCHS} ========")
            total_train_loss = 0
            model.train()
            
            for step, batch in enumerate(train_dataloader):
                if step % 50 == 0 and not step == 0:
                    print(f"  Batch {step} of {len(train_dataloader)}.")
                
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)
                
                model.zero_grad()
                outputs = model(
                    b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask,
                    labels=b_labels
                )
                loss = outputs.loss
                total_train_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
            
            avg_train_loss = total_train_loss / len(train_dataloader)
            print(f"  Average training loss: {avg_train_loss:.2f}")
        
        print("\nTraining complete.")
        
        # Evaluation
        print("\nEvaluating model...")
        model.eval()
        all_preds = []
        all_pred_probas = []
        all_true_labels = []
        
        for batch in test_dataloader:
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            b_labels = batch[2].to(self.device)
            
            with torch.no_grad():
                outputs = model(
                    b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask
                )
            
            logits = outputs.logits
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            
            all_preds.extend(np.argmax(logits, axis=1).flatten())
            all_pred_probas.extend(logits)
            all_true_labels.extend(label_ids.flatten())
        
        # Convert predictions back to original labels
        y_pred = label_encoder.inverse_transform(all_preds)
        y_test = label_encoder.inverse_transform(all_true_labels)
        y_pred_proba = np.array(all_pred_probas)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted')
        
        # Calculate ROC AUC
        if mode == 'without_neutral':  # Binary classification
            roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        else:  # Multi-class classification
            roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'classification_report': classification_report(y_test, y_pred, target_names=label_encoder.classes_)
        }
        
        return model, self.tokenizer, metrics, y_pred, y_test
    
    def predict_new_reviews(self, model, tokenizer, new_reviews):
        """Predict sentiment for new reviews"""
        print("\nPreprocessing new reviews for prediction...")
        processed_reviews = [self.preprocess_text(review) for review in new_reviews]
        print("Preprocessing of new reviews complete.")
        
        model.eval()
        
        encoded_batch = tokenizer.batch_encode_plus(
            processed_reviews,
            add_special_tokens=True,
            max_length=self.MAX_LEN,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoded_batch['input_ids'].to(self.device)
        attention_mask = encoded_batch['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = model(input_ids, token_type_ids=None, attention_mask=attention_mask)
        
        logits = outputs.logits
        predictions_indices = torch.argmax(logits, dim=1).cpu().numpy()
        
        results = []
        for original, processed, pred in zip(new_reviews, processed_reviews, predictions_indices):
            results.append({
                'original_review': original,
                'processed_review': processed,
                'predicted_rating': pred
            })
        
        return results 