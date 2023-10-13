import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from transformers import pipeline
from datasets import concatenate_datasets, load_dataset_builder, get_dataset_config_names, load_dataset


ds_builder = load_dataset_builder("amazon_reviews_multi")
print(ds_builder.info.description)
configs = get_dataset_config_names("amazon_reviews_multi")
print(configs)

# train
trainen = load_dataset("amazon_reviews_multi", "en", split="train")
trainde = load_dataset("amazon_reviews_multi", "de", split="train")
tr = concatenate_datasets([trainen, trainde])

#test
testen = load_dataset("amazon_reviews_multi", "en", split="test")
testde = load_dataset("amazon_reviews_multi", "de", split="test")
te = concatenate_datasets([testen, testde])


def count_words(text):
    # Split the text by spaces to count words
    words = text.split()
    return len(words)


def calculate_word_counts(dat, col):
    dat[f"{col}_word_count"] = count_words(dat[col])
    return dat


def char_counts(dat, col):
    dat[f"{col}_char_count"] = len(dat[col])
    return dat


def calculate_average_word_length(example, column_name):
    words = example[column_name].split()
    total_word_length = sum(len(word) for word in words)
    example[f"{column_name}_avg_word_length"] = total_word_length / len(words) if len(words) > 0 else 0
    return example


def calculate_unique_word_count(dat, col):
    words = dat[col].split()
    unique_words = set(words)
    dat[f"{col}_unique_word_count"] = len(unique_words)
    return dat


def calculate_uppercase_letter_count(dat, col):
    uppercase_count = sum(1 for letter in dat[col] if letter.isupper())
    dat[f"{col}_uppercase_letter_count"] = uppercase_count
    return dat


def calculate_numeric_character_count(dat, col):
    numeric_count = sum(1 for char in dat[col] if char.isnumeric())
    dat[f"{col}_numeric_char_count"] = numeric_count
    return dat


def apply_calculate_counts(dat):
    dat = calculate_word_counts(dat, "review_body")
    dat = calculate_word_counts(dat, "review_title")
    dat = char_counts(dat, "review_body")
    dat = char_counts(dat, "review_title")
    dat = calculate_average_word_length(dat, "review_body")
    dat = calculate_average_word_length(dat, "review_title")
    dat = calculate_unique_word_count(dat, "review_body")
    dat = calculate_unique_word_count(dat, "review_title")
    dat = calculate_uppercase_letter_count(dat, "review_body")
    dat = calculate_uppercase_letter_count(dat, "review_title")
    return dat


tr = tr.map(apply_calculate_counts)
te = te.map(apply_calculate_counts)

labels_tr = tr["stars"]
data_tr = tr.remove_columns("stars")
labels_te = te["stars"]
data_te = te.remove_columns("stars")

# Combine text from the first 7 columns into a single text field
data_tr = data_tr.to_pandas()
data_te = data_te.to_pandas()
def combine_text_columns(row):
    return ' '.join(row)

data_tr['combined_text'] = data_tr.iloc[:, :7].apply(combine_text_columns, axis=1)
data_te['combined_text'] = data_te.iloc[:, :7].apply(combine_text_columns, axis=1)

# Use the last 13 integer columns as additional features
train_features = data_tr.iloc[:, 7:17]
test_features = data_te.iloc[:, 7:17]

# Step 2: Preprocess the text data using TF-IDF
tfidf_vectorizer = TfidfVectorizer(min_df=100, token_pattern=r'[a-zA-Z]+')
X_train = tfidf_vectorizer.fit_transform(data_tr['combined_text'])
X_test = tfidf_vectorizer.transform(data_te['combined_text'])

print(X_train.shape)  # Check the shape of X_train
print(train_features.shape)  # Check the shape of train_features

# Combine text features and integer features
import scipy.sparse as sp
from scipy.sparse import hstack

X_combined_train = hstack([X_train, train_features.values], format='csr')
X_combined_test = hstack([X_test, test_features.values], format='csr')

# Step 4: Prepare labels (assuming your target column is in a DataFrame)
y_train = labels_tr
y_test = labels_te

# Step 5: Train the Decision Tree Classifier
clf = DecisionTreeClassifier()
clf.fit(X_combined_train, y_train)

# Step 6: Make Predictions on the test set
y_pred = clf.predict(X_combined_test)

# Step 7: Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("\nClassification Report:")
print(report)

# Convert the dataset to a pandas DataFrame
df = tr.to_pandas()

# Print the DataFrame
print(df)

# Step 8: Train the Naive Bayes Classifier
nb_clf = MultinomialNB()
nb_clf.fit(X_combined_train, y_train)

# Step 9: Make Predictions on the test set with Naive Bayes
y_pred_nb = nb_clf.predict(X_combined_test)

# Step 10: Evaluate the Naive Bayes Model
accuracy_nb = accuracy_score(y_test, y_pred_nb)
report_nb = classification_report(y_test, y_pred_nb)

print("\nNaive Bayes Results")
print(f"Accuracy: {accuracy_nb}")
print("\nClassification Report:")
print(report_nb)

# Convert the dataset to a pandas DataFrame
df = tr.to_pandas()

# Print the DataFrame
print(df)