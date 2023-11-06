import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
import re
from scipy.sparse import csr_matrix, hstack
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from datasets import concatenate_datasets, load_dataset_builder, get_dataset_config_names, load_dataset

# Load Social Bias Frames dataset
ds_builder = load_dataset_builder("social_bias_frames")
print(ds_builder.info.description)
configs = get_dataset_config_names("social_bias_frames")
print(configs)

# train
train_data = load_dataset("social_bias_frames", split="train")
# test
test_data = load_dataset("social_bias_frames", split="test")


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
def apply_custom_features(dataset):
    dataset = dataset.map(lambda x: {"HasLinksYN": 'Y' if has_links(x['post']) else 'N'})
    dataset = dataset.map(lambda x: {"HasEmojisYN": 'Y' if has_emojis(x['post']) else 'N'})
    return dataset


train_data = apply_custom_features(train_data)
test_data = apply_custom_features(test_data)

# Separate the features and labels for training and testing data
labels_tr = train_data["offensiveYN"]
data_tr = train_data.remove_columns("offensiveYN")
labels_te = test_data["offensiveYN"]
data_te = test_data.remove_columns("offensiveYN")

# Convert the dataset to a pandas DataFrame
data_tr = data_tr.to_pandas()
data_te = data_te.to_pandas()

# Step 2: Preprocess the text data using TF-IDF
tfidf_vectorizer = TfidfVectorizer(min_df=10, token_pattern=r'[a-zA-Z]+')
X_train = tfidf_vectorizer.fit_transform(data_tr['post'])
X_test = tfidf_vectorizer.transform(data_te['post'])

print(X_train.shape)  # Check the shape of X_train

# Combine text features and additional features
data_tr['HasLinksYN'] = data_tr['HasLinksYN'].map({'N': 0, 'Y': 1})
data_tr['HasEmojisYN'] = data_tr['HasEmojisYN'].map({'N': 0, 'Y': 1})

data_te['HasLinksYN'] = data_te['HasLinksYN'].map({'N': 0, 'Y': 1})
data_te['HasEmojisYN'] = data_te['HasEmojisYN'].map({'N': 0, 'Y': 1})

# Select only the numeric columns for converting to sparse matrix
numeric_columns_tr = ['HasLinksYN', 'HasEmojisYN'] 
numeric_columns_te = ['HasLinksYN', 'HasEmojisYN']

# Convert additional features to sparse matrix
data_tr_sparse = csr_matrix(data_tr[numeric_columns_tr].values)
data_te_sparse = csr_matrix(data_te[numeric_columns_te].values)

# Combine text features and additional features
X_combined_train = hstack([X_train, data_tr_sparse], format='csr')
X_combined_test = hstack([X_test, data_te_sparse], format='csr')

# Step 4: Prepare labels
y_train = labels_tr
y_test = labels_te

# # Define the parameter grid to search for the best 'max_depth'
# param_grid = {'max_depth': range(1, 21)}  # Searching from 1 to 20
# 
# # Initialize the decision tree classifier
# dtree = DecisionTreeClassifier(random_state=42)
#
# # Initialize GridSearchCV with cross-validation
# grid_search = GridSearchCV(dtree, param_grid, cv=5, scoring='accuracy')
#
# # Fit GridSearchCV on the training data
# grid_search.fit(X_combined_train, y_train)
#
# # Print the best parameter and the corresponding score from the training (cross-validation)
# print(f"Best Parameter from training: {grid_search.best_params_}")
# print(f"Best Score from training: {grid_search.best_score_}")


# Step 5: Train the Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=1)
clf.fit(X_combined_train, y_train)

# Step 6: Make Predictions on the test set
y_pred = clf.predict(X_combined_test)

# Step 7: Evaluate the Model
accuracy_dt = accuracy_score(y_test, y_pred)
report_dt = classification_report(y_test, y_pred, zero_division=0)

print(f"Decision Tree Results:")
print(f"Accuracy: {accuracy_dt}")
print("\nDecision Tree Classification Report:")
print(report_dt)

# Step 8: Train the Naive Bayes Classifier
nb_clf = MultinomialNB()
nb_clf.fit(X_combined_train, y_train)

# Step 9: Make Predictions on the test set with Naive Bayes
y_pred_nb = nb_clf.predict(X_combined_test)

# Step 10: Evaluate the Naive Bayes Model
accuracy_nb = accuracy_score(y_test, y_pred_nb)
report_nb = classification_report(y_test, y_pred_nb, zero_division=0)

print("\nNaive Bayes Results:")
print(f"Accuracy: {accuracy_nb}")
print("\nNaive Bayes Classification Report:")
print(report_nb)

# Standardize the features before feeding them into MLP
scaler = StandardScaler(with_mean=False)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#MLP Hyperparameters
MLPlearn = 0.01
svd = TruncatedSVD(n_components=20)
trunc = svd.fit_transform(X_train_scaled)
trunc_test = svd.transform(X_test_scaled)

# Train the Multi-layer Perceptron Classifier
mlp_clf = MLPClassifier(random_state=1, activation='relu', max_iter=300, learning_rate='adaptive', learning_rate_init=MLPlearn)
mlp_clf.fit(trunc, y_train)

# Make Predictions on the test set with MLP
y_pred_mlp = mlp_clf.predict(trunc_test)

# Evaluate the MLP Model
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
report_mlp = classification_report(y_test, y_pred_mlp, zero_division=0)

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
report_lr = classification_report(y_test, y_pred_lr, zero_division=0)

print("\nLogistic Regression Results")
print(f"Accuracy: {accuracy_lr}")
print("\nLogistic Regression Classification Report:")
print(report_lr)