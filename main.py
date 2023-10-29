import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from transformers import pipeline
from datasets import concatenate_datasets, load_dataset_builder, get_dataset_config_names, load_dataset
import warnings

# mute warnings
warnings.filterwarnings('ignore')

# prepare data
ds_builder = load_dataset_builder("social_bias_frames")
print(ds_builder.info.description)
configs = get_dataset_config_names("social_bias_frames")

# train
tr = load_dataset("social_bias_frames", split="train")

#test
te = load_dataset("social_bias_frames", split="test")

labels_tr = tr["offensiveYN"]
data_tr = tr.remove_columns("offensiveYN")
labels_te = te["offensiveYN"]
data_te = te.remove_columns("offensiveYN")

data_tr = data_tr.to_pandas().head(5000)
data_te = data_te.to_pandas().head(5000)
labels_tr = labels_tr[:5000]
labels_te = labels_te[:5000]


def HasLinks(dat):
    dat['hasLink'] = dat['post'].str.contains('http')


def HasEmojis(dat):
    dat['hasEmoji'] = dat['post'].str.contains('&#')


# Preprocessing
HasLinks(data_tr)
HasEmojis(data_tr)
HasLinks(data_te)
HasEmojis(data_te)

from sklearn.preprocessing import LabelEncoder

# Convert string categorical variables into integer-encoded variables
label_encoders = {}
text_columns = ['post', 'targetStereotype', 'sexPhrase', 'sexReason']

# First, handle the categorical columns with LabelEncoder
for col in data_tr.columns:
    if data_tr[col].dtype == 'object' and col not in text_columns:
        le = LabelEncoder()
        le.fit(pd.concat([data_tr[col], data_te[col]]))  # Fit on both train and test data
        data_tr[col] = le.transform(data_tr[col])
        # Transform test data with handling unseen categories
        data_te[col] = data_te[col].map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)

# Next, handle the text columns with TF-IDF vectorization
tfidf_vectorizers = {}

for col in text_columns:
    tfidf = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(data_tr[col])
    X_test_tfidf = tfidf.transform(data_te[col])
    
    # Drop the original column and add the TF-IDF features
    data_tr = data_tr.drop(columns=[col])
    data_te = data_te.drop(columns=[col])
    
    # Concatenate the TF-IDF features
    data_tr = pd.concat([data_tr, pd.DataFrame(X_train_tfidf.toarray(), index=data_tr.index)], axis=1)
    data_te = pd.concat([data_te, pd.DataFrame(X_test_tfidf.toarray(), index=data_te.index)], axis=1)
    
    tfidf_vectorizers[col] = tfidf

# Convert all column names to string
data_tr.columns = data_tr.columns.astype(str)
data_te.columns = data_te.columns.astype(str)

# Decision Tree Classifier
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(data_tr, labels_tr)
# Make Predictions on the test set
dt_predictions = dt_classifier.predict(data_te)
print("Decision Tree Classifier Accuracy:", accuracy_score(labels_te, dt_predictions))
print("Decision Tree Classifier Report:")
print(classification_report(labels_te, dt_predictions))

# Naive Bayes Classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(data_tr, labels_tr)
nb_predictions = nb_classifier.predict(data_te)
print("Naive Bayes Classifier Accuracy:", accuracy_score(labels_te, nb_predictions))
print("Naive Bayes Classifier Report:")
print(classification_report(labels_te, nb_predictions))
