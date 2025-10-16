import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load the original dataset CSV file
df = pd.read_csv("dataset_2.csv")

# Additional positive examples with the word "impeccable" to improve prediction
additional_data = [
    {'Text': 'He is impeccable in every way.', 'Sentiment': 'positive', '#tags': '#excellent', 'Platform': 'LinkedIn'},
    {'Text': 'Her performance was impeccable.', 'Sentiment': 'positive', '#tags': '#perfect', 'Platform': 'Twitter'},
    {'Text': 'Impeccable service and quality, highly recommend!', 'Sentiment': 'positive', '#tags': '#awesome', 'Platform': 'Facebook'}
]
additional_df = pd.DataFrame(additional_data)

# Append additional rows to the original dataframe
df = pd.concat([df, additional_df], ignore_index=True)

# Map sentiment labels and filter known classes
map_3class = {'positive': 'positive', 'negative': 'negative', 'neutral': 'neutral'}
df['Sentiment'] = df['Sentiment'].astype(str).str.strip()
df['Sentiment_Option1'] = df['Sentiment'].map(map_3class).fillna('Other')
df_option1 = df[df['Sentiment_Option1'] != 'Other'].copy()

# Text cleaning function
def clean_text_basic(text):
    custom_stopwords = set(stopwords.words('english'))
    text = text.encode('ascii', 'ignore').decode()
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in custom_stopwords]
    return " ".join(tokens)

# Clean the text column
df_option1['clean_text'] = df_option1['Text'].astype(str).apply(clean_text_basic)

# Vectorize cleaned text using TF-IDF
vectorizer = TfidfVectorizer(max_features=2000)
X = vectorizer.fit_transform(df_option1['clean_text'])

# Encode labels
le = LabelEncoder()
y = le.fit_transform(df_option1['Sentiment_Option1'])

# Split data into train and test sets with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42, stratify=y
)

# Train logistic regression classifier
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Print classification report
print("=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Plot and save confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix - Sentiment Classification")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

# Save model and vectorizer for future use
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(le, 'label_encoder.pkl')

# Save classification report as JSON file
report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
with open('classification_report.json', 'w') as f:
    json.dump(report, f)

# Example predictions on new text samples
examples = [
    "I really enjoyed this product, highly recommend it!",
    "This is the worst experience I've ever had.",
    "He is impeccable",
    "It's okay, nothing special about it.",
    "Absolutely fantastic service and great quality!",
    "I am disappointed with the delay in delivery.",
]

cleaned_examples = [clean_text_basic(text) for text in examples]
example_vectors = vectorizer.transform(cleaned_examples)
example_preds = model.predict(example_vectors)
decoded_preds = le.inverse_transform(example_preds)

print("\n--- Example Predictions ---")
for text, sentiment in zip(examples, decoded_preds):
    print(f'"{text}" → {sentiment}')