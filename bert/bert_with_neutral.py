import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
import numpy as np
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


MODEL_NAME = "google-bert/bert-base-uncased"
NUM_LABELS = 3
BATCH_SIZE = 16
EPOCHS = 3
MAX_LEN = 128
LEARNING_RATE = 2e-5
RANDOM_STATE = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

csv_file_name = "tripadvisor_hotel_reviews.csv"
if not os.path.exists(csv_file_name):
    print(f"Error: {csv_file_name} not found. Please ensure the dataset is in the same directory.")
    exit()

df = pd.read_csv(csv_file_name, on_bad_lines='skip')
print("Dataset loaded successfully.")

df.dropna(subset=['Review', 'Rating'], inplace=True)

lemmatizer = WordNetLemmatizer()
stop_words_nltk = set(stopwords.words('english'))

def preprocess_text_bert(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words_nltk and word.isalpha()]
    return ' '.join(tokens)

print("\nPreprocessing reviews for BERT...")
df['Processed_Review'] = df['Review'].apply(preprocess_text_bert)
print("Preprocessing complete.")
print("Example of processed review for BERT:\n", df[['Review', 'Processed_Review']].head())

def categorize_rating(rating):
    if rating <= 2:
        return 'bad'
    elif rating == 3:
        return 'neutral'
    else:
        return 'good'

df['Sentiment'] = df['Rating'].apply(categorize_rating)
print("\nSentiment distribution:\n", df['Sentiment'].value_counts())

label_encoder = LabelEncoder()
df['SentimentEncoded'] = label_encoder.fit_transform(df['Sentiment'])
class_names = label_encoder.classes_
print(f"\nEncoded labels: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")


X = df['Processed_Review'].values
y = df['SentimentEncoded'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

print(f"\nLoading BERT tokenizer: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_data(texts, max_len):
    input_ids = []
    attention_masks = []
    for text in texts:
        encoded_dict = tokenizer.encode_plus(
                            text,
                            add_special_tokens=True,
                            max_length=max_len,
                            pad_to_max_length=True,
                            return_attention_mask=True,
                            return_tensors='pt',
                            truncation=True
                       )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    return input_ids, attention_masks

print("Tokenizing training data...")
train_input_ids, train_attention_masks = tokenize_data(X_train, MAX_LEN)
print("Tokenizing test data...")
test_input_ids, test_attention_masks = tokenize_data(X_test, MAX_LEN)

train_labels = torch.tensor(y_train)
test_labels = torch.tensor(y_test)

train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=BATCH_SIZE)

test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_labels)
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=BATCH_SIZE)

print(f"\nLoading BERT model for sequence classification: {MODEL_NAME}...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    output_attentions=False,
    output_hidden_states=False
)
model.to(device)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)
total_steps = len(train_dataloader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=total_steps)

print("\nStarting training...")
for epoch_i in range(0, EPOCHS):
    print(f"\n======== Epoch {epoch_i + 1} / {EPOCHS} ========")
    total_train_loss = 0
    model.train()
    for step, batch in enumerate(train_dataloader):
        if step % 50 == 0 and not step == 0:
            print(f"  Batch {step} of {len(train_dataloader)}.")
        
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        model.zero_grad()
        outputs = model(b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask,
                        labels=b_labels)
        loss = outputs.loss
        total_train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_train_loss / len(train_dataloader)
    print(f"  Average training loss: {avg_train_loss:.2f}")

print("\nTraining complete.")

print("\nEvaluating model...")
model.eval()
all_preds = []
all_true_labels = []

for batch in test_dataloader:
    b_input_ids = batch[0].to(device)
    b_input_mask = batch[1].to(device)
    b_labels = batch[2].to(device)

    with torch.no_grad():
        outputs = model(b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask)
    logits = outputs.logits
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()

    all_preds.extend(np.argmax(logits, axis=1).flatten())
    all_true_labels.extend(label_ids.flatten())

print("\nModel Evaluation:")
print(classification_report(all_true_labels, all_preds, target_names=class_names))


new_reviews_raw = [
    "This hotel was fantastic, the service was excellent!",
    "The room was dirty and the staff were rude.",
    "It was an okay experience, nothing special."
]
print("\nPreprocessing new reviews for BERT prediction...")
new_reviews_processed = [preprocess_text_bert(review) for review in new_reviews_raw]
print("Preprocessing of new reviews complete.")

model.eval()

encoded_batch = tokenizer.batch_encode_plus(
    new_reviews_processed,
    add_special_tokens=True,
    max_length=MAX_LEN,
    padding='max_length',
    truncation=True,
    return_attention_mask=True,
    return_tensors='pt'
)

input_ids = encoded_batch['input_ids'].to(device)
attention_mask = encoded_batch['attention_mask'].to(device)

with torch.no_grad():
    outputs = model(input_ids, token_type_ids=None, attention_mask=attention_mask)

logits = outputs.logits
predictions_indices = torch.argmax(logits, dim=1).cpu().numpy()
predicted_sentiments = label_encoder.inverse_transform(predictions_indices)

for original_review, processed_review, sentiment in zip(new_reviews_raw, new_reviews_processed, predicted_sentiments):
    print(f"Original Review: \"{original_review}\"")
    print(f"Processed Review: \"{processed_review}\" -> Predicted Sentiment: {sentiment}")