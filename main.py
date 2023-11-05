import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from datasets import load_dataset
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

# Load the Social Bias Frames dataset
social_bias_frames = load_dataset("social_bias_frames")

# Check if links and emojis in text
def has_links(text):
    # Define a regular expression for URLs
    url_pattern = r'https?://\S+|www\.\S+'
    return int(bool(re.search(url_pattern, text)))

def has_emojis(text):
    # Define a regular expression for emojis
    emoji_pattern = r'[\U00010000-\U0010ffff]'
    return int(bool(re.search(emoji_pattern, text)))

# Apply the custom functions to the dataset
def apply_custom_fields(data_field):
    data_field['Has links'] = has_links(data_field['post'])
    data_field['HasEmojis Y/N'] = has_emojis(data_field['post'])
    return data_field

social_bias_frames = social_bias_frames.map(apply_custom_fields)

# Convert the dataset to a pandas DataFrame
df = social_bias_frames['train'].to_pandas()

# Print the DataFrame
print(df)

# Assuming 'offensiveYN' is the target column and 'post' is the text column you want to use
labels = df['offensiveYN']
data = df.drop(columns=['offensiveYN'])

# Combine text from the 'post' column into a single text field
data['combined_text'] = data['post']

# Preprocess the text data using TF-IDF
tfidf_vectorizer = TfidfVectorizer(min_df=100, token_pattern=r'[a-zA-Z]+')
X = tfidf_vectorizer.fit_transform(data['combined_text'])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Train the Decision Tree Classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Make Predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the Model
accuracy_dt = accuracy_score(y_test, y_pred)
report_dt = classification_report(y_test, y_pred, zero_division=1)

print(f"Decision Tree Results:")
print(f"Accuracy: {accuracy_dt}")
print("\nDecision Tree Classification Report:")
print(report_dt)

# Train the Naive Bayes Classifier
nb_clf = MultinomialNB()
nb_clf.fit(X_train, y_train)

# Make Predictions on the test set with Naive Bayes
y_pred_nb = nb_clf.predict(X_test)

# Evaluate the Naive Bayes Model
accuracy_nb = accuracy_score(y_test, y_pred_nb)
report_nb = classification_report(y_test, y_pred_nb, zero_division=1)

print("\nNaive Bayes Results:")
print(f"Accuracy: {accuracy_nb}")
print("\nNaive Bayes Classification Report:")
print(report_nb)

# Standardize the features before feeding them into MLP
scaler = StandardScaler(with_mean=False)  # with_mean=False to support sparse input
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Multi-layer Perceptron Classifier
mlp_clf = MLPClassifier(random_state=1, max_iter=300)
mlp_clf.fit(X_train_scaled, y_train)

# Make Predictions on the test set with MLP
y_pred_mlp = mlp_clf.predict(X_test_scaled)

# Evaluate the MLP Model
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
report_mlp = classification_report(y_test, y_pred_mlp, zero_division=1)

print("\nMulti-layer Perceptron Results")
print(f"Accuracy: {accuracy_mlp}")
print("\nMulti-layer Perceptron Classification Report:")
print(report_mlp)

# Train the Logistic Regression Classifier
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# Make Predictions on the test set with Logistic Regression
y_pred_lr = log_reg.predict(X_test)

# Evaluate the Logistic Regression Model
accuracy_lr = accuracy_score(y_test, y_pred_lr)
report_lr = classification_report(y_test, y_pred_lr, zero_division=1)

print("\nLogistic Regression Results")
print(f"Accuracy: {accuracy_lr}")
print("\nLogistic Regression Classification Report:")
print(report_lr)