import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, TruncatedSVD
import re
from scipy.sparse import csr_matrix, hstack
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.tree import plot_tree
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from datasets import concatenate_datasets, load_dataset_builder, get_dataset_config_names, load_dataset
import argparse
from halo import Halo

class MlClassifier:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(min_df=10, token_pattern=r'[a-zA-Z]+')
        self.scaler = StandardScaler(with_mean=False)
        self.svd = TruncatedSVD(n_components=20)

    def load_data(self, dataset_name="social_bias_frames"):
        train_data = load_dataset(dataset_name, split="train")
        test_data = load_dataset(dataset_name, split="test")
        return train_data, test_data

    def preprocess_data(self, data):
        data = data.map(lambda x: {"HasLinksYN": 'Y' if self.has_links(x['post']) else 'N'})
        data = data.map(lambda x: {"HasEmojisYN": 'Y' if self.has_emojis(x['post']) else 'N'})
        return data

    @staticmethod
    def has_links(text):
        url_pattern = r'https?://\S+|www\.\S+'
        return int(bool(re.search(url_pattern, text)))

    @staticmethod
    def has_emojis(text):
        emoji_pattern = r'[\U00010000-\U0010ffff]'
        return int(bool(re.search(emoji_pattern, text)))

    def extract_features(self, train_data, test_data):
        # Fit the vectorizer on the training data
        X_train = self.tfidf_vectorizer.fit_transform(train_data['post'])

        # Transform the test data using the same vectorizer
        X_test = self.tfidf_vectorizer.transform(test_data['post'])

        # Process additional features for training data
        train_data['HasLinksYN'] = train_data['HasLinksYN'].map({'N': 0, 'Y': 1})
        train_data['HasEmojisYN'] = train_data['HasEmojisYN'].map({'N': 0, 'Y': 1})
        additional_features_train = csr_matrix(train_data[['HasLinksYN', 'HasEmojisYN']].values)

        test_data['HasLinksYN'] = test_data['HasLinksYN'].map({'N': 0, 'Y': 1})
        test_data['HasEmojisYN'] = test_data['HasEmojisYN'].map({'N': 0, 'Y': 1})
        additional_features_test = csr_matrix(test_data[['HasLinksYN', 'HasEmojisYN']].values)

        # Combine text features and additional features
        X_combined_train = hstack([X_train, additional_features_train], format='csr')
        X_combined_test = hstack([X_test, additional_features_test], format='csr')

        return X_combined_train, X_combined_test

    def train_and_evaluate_decision_tree(self, X_train, y_train, X_test, y_test):
        dtree = DecisionTreeClassifier(max_depth=100)
        dtree.fit(X_train, y_train)
        return self.evaluate_model(dtree, X_test, y_test)

    def train_and_evaluate_naive_bayes(self, X_train, y_train, X_test, y_test):
        nb = MultinomialNB()
        nb.fit(X_train, y_train)
        return self.evaluate_model(nb, X_test, y_test)

    def train_and_evaluate_mlp(self, X_train, y_train, X_test, y_test):
        mlp = MLPClassifier(random_state=1, activation='relu', max_iter=300, learning_rate='adaptive', learning_rate_init=0.01)
        mlp.fit(X_train, y_train)
        return self.evaluate_model(mlp, X_test, y_test)

    def train_and_evaluate_logistic_regression(self, X_train, y_train, X_test, y_test):
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_train, y_train)
        return self.evaluate_model(lr, X_test, y_test)

    def train_and_evaluate_svm(self, X_train, y_train, X_test, y_test):
        svm = LinearSVC(dual=False, tol=1e-4, max_iter=3000)
        svm.fit(X_train, y_train)
        return self.evaluate_model(svm, X_test, y_test)

    def evaluate_model(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, zero_division=0)
        return accuracy, report

    def run(self, classifier):
        train_data, test_data = self.load_data()
        train_data = self.preprocess_data(train_data)
        test_data = self.preprocess_data(test_data)

        labels_tr = train_data["offensiveYN"]
        data_tr = train_data.remove_columns("offensiveYN")
        labels_te = test_data["offensiveYN"]
        data_te = test_data.remove_columns("offensiveYN")

        X_train, X_test = self.extract_features(data_tr.to_pandas(), data_te.to_pandas())


        # Train and evaluate Decision Tree
        with Halo(text=f'Training and evaluating {classifier}...', spinner='dots'):
            if classifier == 'decision_tree':
                accuracy, report = self.train_and_evaluate_decision_tree(X_train, labels_tr, X_test, labels_te)
            elif classifier == 'naive_bayes':
                accuracy, report = self.train_and_evaluate_naive_bayes(X_train, labels_tr, X_test, labels_te)
            elif classifier == 'mlp':
                accuracy, report = self.train_and_evaluate_mlp(X_train, labels_tr, X_test, labels_te)
            elif classifier == 'logistic_regression':
                accuracy, report = self.train_and_evaluate_logistic_regression(X_train, labels_tr, X_test, labels_te)
            elif classifier == 'svm':
                accuracy, report = self.train_and_evaluate_svm(X_train, labels_tr, X_test, labels_te)
            else:
                raise ValueError("Unknown classifier")

        print(f"{classifier.capitalize()} Results:")
        print(f"Accuracy: {accuracy}")
        print("Classification Report:")
        print(report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run different ML classifiers')
    parser.add_argument('-c', '--classifier', required=True, choices=['decision_tree', 'naive_bayes', 'mlp', 'logistic_regression', 'svm'],
                        help='The machine learning algorithm to run')

    args = parser.parse_args()

    classifier = MlClassifier()
    classifier.run(args.classifier)

