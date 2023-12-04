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
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load Social Bias Frames dataset
ds_builder = load_dataset_builder("social_bias_frames")
configs = get_dataset_config_names("social_bias_frames")

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

# Define the parameter grid to search for the best 'max_depth'
param_grid = {'max_depth': range(1, 21)}  # Searching from 1 to 20

# Initialize the decision tree classifier
dtree = DecisionTreeClassifier(random_state=42)

# Initialize GridSearchCV with cross-validation
grid_search = GridSearchCV(dtree, param_grid, cv=5, scoring='accuracy')

# Fit GridSearchCV on the training data
grid_search.fit(X_combined_train, y_train)

# Print the best parameter and the corresponding score from the training (cross-validation)
print(f"Best Parameter from training: {grid_search.best_params_}")
print(f"Best Score from training: {grid_search.best_score_}")

###########################################################
# Step 5: Train the Decision Tree Classifier
training_accuracy = []
testing_accuracy = []

training_precision = []
testing_precision = []

training_recall = []
testing_recall = []

training_f1 = []
testing_f1 = []
start_depth = 1
end_depth = 45 
for depth in range(start_depth, end_depth):
    clf = DecisionTreeClassifier(max_depth=depth)
    clf = clf.fit(X_train, y_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    training_accuracy.append(metrics.accuracy_score(y_train, y_pred_train))
    testing_accuracy.append(metrics.accuracy_score(y_test, y_pred_test))
    
    training_precision.append(metrics.precision_score(y_train, y_pred_train, average='macro', zero_division=np.nan))
    testing_precision.append(metrics.precision_score(y_test, y_pred_test, average='macro', zero_division=np.nan))

    training_recall.append(metrics.recall_score(y_train, y_pred_train, average='macro'))
    testing_recall.append(metrics.recall_score(y_test, y_pred_test, average='macro'))

    training_f1.append(metrics.f1_score(y_train, y_pred_train, average='macro'))
    testing_f1.append(metrics.f1_score(y_test, y_pred_test, average='macro'))

    print("Depth: ", depth)
    print("Training Accuracy:", '{:.2f}'.format(metrics.accuracy_score(y_train, y_pred_train)))
    print("Testing Accuracy:", '{:.2f}'.format(metrics.accuracy_score(y_test, y_pred_test)))
    print("")

def make_plot(axs):
    ax1 = axs[0,0]
    ax1.plot(range(start_depth, end_depth), training_accuracy, label="Training Accuracy")
    ax1.plot(range(start_depth, end_depth), testing_accuracy, label="Testing Accuracy")
    ax1.set_ylabel("Accuracy")
    ax1.set_xlabel("Tree Depth")
    ax1.set_title("Accuracy vs. Tree Depth")
    ax1.legend()

    ax2 = axs[0,1]
    ax2.plot(range(start_depth, end_depth), training_precision, label="Training Precision")
    ax2.plot(range(start_depth, end_depth), testing_precision, label="Testing Precision")
    ax2.set_ylabel("Precision")
    ax2.set_xlabel("Tree Depth")
    ax2.set_title("Precision vs. Tree Depth")
    ax2.legend()

    ax3 = axs[1,0]
    ax3.plot(range(start_depth, end_depth), training_recall, label="Training Recall")
    ax3.plot(range(start_depth, end_depth), testing_recall, label="Testing Recall")
    ax3.set_ylabel("Recall")
    ax3.set_xlabel("Tree Depth")
    ax3.set_title("Recall vs. Tree Depth")
    ax3.legend()

    ax4 = axs[1,1]
    ax4.plot(range(start_depth, end_depth), training_f1, label="Training F1")
    ax4.plot(range(start_depth, end_depth), testing_f1, label="Testing F1")
    ax4.set_ylabel("F1")
    ax4.set_xlabel("Tree Depth")
    ax4.set_title("F1 vs. Tree Depth")
    ax4.legend()

fig, axs = plt.subplots(2, 2, figsize=(15, 15))
make_plot(axs)
fig.suptitle("Accuracy Metrics against Tree Depth (with 85%/15% Test/Train Split)")
plt.show()

###########################################################
# Plot 1: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(set(y_test)))
plt.xticks(tick_marks, set(y_test), rotation=45)
plt.yticks(tick_marks, set(y_test))
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()

###########################################################
# Step 8: Train the Naive Bayes Classifier
# Define the parameter grid
# Define the alpha values to test
alpha_values = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]

# Lists to store metrics for each alpha
training_accuracy_nb = []
testing_accuracy_nb = []
training_precision_nb = []
testing_precision_nb = []
training_recall_nb = []
testing_recall_nb = []
training_f1_nb = []
testing_f1_nb = []

# Loop over each alpha value
for alpha in alpha_values:
    # Train the model
    nb_clf = MultinomialNB(alpha=alpha)
    nb_clf.fit(X_combined_train, y_train)

    # Predict on train and test set
    y_pred_train_nb = nb_clf.predict(X_combined_train)
    y_pred_test_nb = nb_clf.predict(X_combined_test)

    # Calculate metrics
    training_accuracy_nb.append(metrics.accuracy_score(y_train, y_pred_train_nb))
    testing_accuracy_nb.append(metrics.accuracy_score(y_test, y_pred_test_nb))
    training_precision_nb.append(metrics.precision_score(y_train, y_pred_train_nb, average='macro', zero_division=np.nan))
    testing_precision_nb.append(metrics.precision_score(y_test, y_pred_test_nb, average='macro', zero_division=np.nan))
    training_recall_nb.append(metrics.recall_score(y_train, y_pred_train_nb, average='macro'))
    testing_recall_nb.append(metrics.recall_score(y_test, y_pred_test_nb, average='macro'))
    training_f1_nb.append(metrics.f1_score(y_train, y_pred_train_nb, average='macro'))
    testing_f1_nb.append(metrics.f1_score(y_test, y_pred_test_nb, average='macro'))

# Function to make plot for Naive Bayes
def make_plot_nb(axs, alpha_values):
    ax1 = axs[0,0]
    ax1.plot(alpha_values, training_accuracy_nb, label="Training Accuracy")
    ax1.plot(alpha_values, testing_accuracy_nb, label="Testing Accuracy")
    ax1.set_ylabel("Accuracy")
    ax1.set_xlabel("Alpha")
    ax1.set_title("Naive Bayes: Accuracy vs. Alpha")
    ax1.legend()

    ax2 = axs[0,1]
    ax2.plot(alpha_values, training_precision_nb, label="Training Precision")
    ax2.plot(alpha_values, testing_precision_nb, label="Testing Precision")
    ax2.set_ylabel("Precision")
    ax2.set_xlabel("Alpha")
    ax2.set_title("Naive Bayes: Precision vs. Alpha")
    ax2.legend()

    ax3 = axs[1,0]
    ax3.plot(alpha_values, training_recall_nb, label="Training Recall")
    ax3.plot(alpha_values, testing_recall_nb, label="Testing Recall")
    ax3.set_ylabel("Recall")
    ax3.set_xlabel("Alpha")
    ax3.set_title("Naive Bayes: Recall vs. Alpha")
    ax3.legend()

    ax4 = axs[1,1]
    ax4.plot(alpha_values, training_f1_nb, label="Training F1")
    ax4.plot(alpha_values, testing_f1_nb, label="Testing F1")
    ax4.set_ylabel("F1 Score")
    ax4.set_xlabel("Alpha")
    ax4.set_title("Naive Bayes: F1 Score vs. Alpha")
    ax4.legend()

# Create the plot
fig, axs = plt.subplots(2, 2, figsize=(15, 15))
make_plot_nb(axs, alpha_values)
fig.suptitle("Naive Bayes Performance Metrics for Different Alpha Values")
plt.xscale('log')  # Setting x-axis to log scale for better visualization
plt.show()

class_names = ['N', 'Y']  # 'N' for non-offensive posts and 'Y' for offensive posts

mat = confusion_matrix(y_test, y_pred_nb)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('True label')
plt.ylabel('Predicted label')
plt.show()

# Scale data
scaler = StandardScaler(with_mean=False)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reduce dimensionality
svd = TruncatedSVD(n_components=20)
X_train_reduced = svd.fit_transform(X_train_scaled)
X_test_reduced = svd.transform(X_test_scaled)

# Define the parameter grid to search
learning_rates = [0.0001, 0.001, 0.01, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7]

# Lists to store metrics for each alpha
training_accuracy_nb = []
testing_accuracy_nb = []
training_precision_nb = []
testing_precision_nb = []
training_recall_nb = []
testing_recall_nb = []
training_f1_nb = []
testing_f1_nb = []

scaler = StandardScaler(with_mean=False)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#MLP Hyperparameters
svd = TruncatedSVD(n_components=20)
trunc = svd.fit_transform(X_train_scaled)
trunc_test = svd.transform(X_test_scaled)
# Loop over each alpha value
for MLPlearn in learning_rates:
    # Train the model
    mlp_clf = MLPClassifier(random_state=1, activation='relu', max_iter=300, learning_rate='adaptive', learning_rate_init=MLPlearn)
    mlp_clf.fit(X_combined_train, y_train)

    # Predict on train and test set
    y_pred_train_nb = mlp_clf.predict(X_combined_train)
    y_pred_test_nb = mlp_clf.predict(X_combined_test)

    # Calculate metrics
    training_accuracy_nb.append(metrics.accuracy_score(y_train, y_pred_train_nb))
    testing_accuracy_nb.append(metrics.accuracy_score(y_test, y_pred_test_nb))
    training_precision_nb.append(metrics.precision_score(y_train, y_pred_train_nb, average='macro', zero_division=np.nan))
    testing_precision_nb.append(metrics.precision_score(y_test, y_pred_test_nb, average='macro', zero_division=np.nan))
    training_recall_nb.append(metrics.recall_score(y_train, y_pred_train_nb, average='macro'))
    testing_recall_nb.append(metrics.recall_score(y_test, y_pred_test_nb, average='macro'))
    training_f1_nb.append(metrics.f1_score(y_train, y_pred_train_nb, average='macro'))
    testing_f1_nb.append(metrics.f1_score(y_test, y_pred_test_nb, average='macro'))

# Function to make plot for MLP
def make_plot_nb(axs, learning_rates):
    ax1 = axs[0,0]
    ax1.plot(learning_rates, training_accuracy_nb, label="Training Accuracy")
    ax1.plot(learning_rates, testing_accuracy_nb, label="Testing Accuracy")
    ax1.set_ylabel("Accuracy")
    ax1.set_xlabel("Learning Rate")
    ax1.set_title("MLPs: Accuracy vs. Learning Rate")
    ax1.legend()

    ax2 = axs[0,1]
    ax2.plot(learning_rates, training_precision_nb, label="Training Precision")
    ax2.plot(learning_rates, testing_precision_nb, label="Testing Precision")
    ax2.set_ylabel("Precision")
    ax2.set_xlabel("Learning Rate")
    ax2.set_title("MLP: Precision vs. Learning Rate")
    ax2.legend()

    ax3 = axs[1,0]
    ax3.plot(learning_rates, training_recall_nb, label="Training Recall")
    ax3.plot(learning_rates, testing_recall_nb, label="Testing Recall")
    ax3.set_ylabel("Recall")
    ax3.set_xlabel("Learning Rate")
    ax3.set_title("MLP: Recall vs. Learning Rate")
    ax3.legend()

    ax4 = axs[1,1]
    ax4.plot(learning_rates, training_f1_nb, label="Training F1")
    ax4.plot(learning_rates, testing_f1_nb, label="Testing F1")
    ax4.set_ylabel("F1 Score")
    ax4.set_xlabel("Learning Rate")
    ax4.set_title("MLP: F1 Score vs. Learning Rate")
    ax4.legend()

# Create the plot
fig, axs = plt.subplots(2, 2, figsize=(15, 15))
make_plot_nb(axs, learning_rates)
fig.suptitle("MLP Performance Metrics for Different Learning Rates Values")
plt.xscale('log')  # Setting x-axis to log scale for better visualization
plt.show()

scaler = StandardScaler(with_mean=False)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Logistic Regression Classifier
# Parameters for Logistic Regression (using C as inverse of regularization strength)
C_values = np.logspace(-4, 2, 7)

# Lists to store metrics for each C value
training_accuracy_lr = []
testing_accuracy_lr = []
training_precision_lr = []
testing_precision_lr = []
training_recall_lr = []
testing_recall_lr = []
training_f1_lr = []
testing_f1_lr = []

# Loop over each C value
for C in C_values:
    # Train the model
    lr_clf = LogisticRegression(C=C, max_iter=1000)
    lr_clf.fit(X_train, y_train)

    # Predict on train and test set
    y_pred_train_lr = lr_clf.predict(X_train)
    y_pred_test_lr = lr_clf.predict(X_test)

    # Calculate metrics
    training_accuracy_lr.append(accuracy_score(y_train, y_pred_train_lr))
    testing_accuracy_lr.append(accuracy_score(y_test, y_pred_test_lr))
    training_precision_lr.append(precision_score(y_train, y_pred_train_lr, average='macro', zero_division=0))
    testing_precision_lr.append(precision_score(y_test, y_pred_test_lr, average='macro', zero_division=0))
    training_recall_lr.append(recall_score(y_train, y_pred_train_lr, average='macro'))
    testing_recall_lr.append(recall_score(y_test, y_pred_test_lr, average='macro'))
    training_f1_lr.append(f1_score(y_train, y_pred_train_lr, average='macro'))
    testing_f1_lr.append(f1_score(y_test, y_pred_test_lr, average='macro'))

# Function to make plot for Logistic Regression
def make_plot_lr(axs, C_values):
    ax1 = axs[0,0]
    ax1.plot(C_values, training_accuracy_lr, label="Training Accuracy")
    ax1.plot(C_values, testing_accuracy_lr, label="Testing Accuracy")
    ax1.set_ylabel("Accuracy")
    ax1.set_xlabel("C (Inverse of Regularization Strength)")
    ax1.set_title("Logistic Regression: Accuracy vs. C")
    ax1.legend()

    ax2 = axs[0,1]
    ax2.plot(C_values, training_precision_lr, label="Training Precision")
    ax2.plot(C_values, testing_precision_lr, label="Testing Precision")
    ax2.set_ylabel("Precision")
    ax2.set_xlabel("C")
    ax2.set_title("Logistic Regression: Precision vs. C")
    ax2.legend()

    ax3 = axs[1,0]
    ax3.plot(C_values, training_recall_lr, label="Training Recall")
    ax3.plot(C_values, testing_recall_lr, label="Testing Recall")
    ax3.set_ylabel("Recall")
    ax3.set_xlabel("C")
    ax3.set_title("Logistic Regression: Recall vs. C")
    ax3.legend()

    ax4 = axs[1,1]
    ax4.plot(C_values, training_f1_lr, label="Training F1")
    ax4.plot(C_values, testing_f1_lr, label="Testing F1")
    ax4.set_ylabel("F1 Score")
    ax4.set_xlabel("C")
    ax4.set_title("Logistic Regression: F1 Score vs. C")
    ax4.legend()

# Create the plot
fig, axs = plt.subplots(2, 2, figsize=(15, 15))
make_plot_lr(axs, C_values)
fig.suptitle("Logistic Regression Performance Metrics for Different C Values")
plt.xscale('log')  # Setting x-axis to log scale for better visualization
plt.show()

# Step 11: Train the SVM Classifier
# Values for the regularization parameter C
C_values = np.logspace(-4, 2, 7)

# Lists to store metrics
training_accuracy_svm = []
testing_accuracy_svm = []
training_precision_svm = []
testing_precision_svm = []
training_recall_svm = []
testing_recall_svm = []
training_f1_svm = []
testing_f1_svm = []

# Scaling the data
scaler = StandardScaler(with_mean=False)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Loop over each value of C
for C in C_values:
    # Initialize the SVM classifier with the current value of C
    svm_clf = LinearSVC(C=C, dual=False, tol=1e-4, max_iter=3000)
    
    # Fit the SVM classifier on the scaled training data
    svm_clf.fit(X_train_scaled, y_train)
    
    # Predict on train and test set
    y_pred_train_svm = svm_clf.predict(X_train_scaled)
    y_pred_test_svm = svm_clf.predict(X_test_scaled)
    
    # Calculate metrics
    training_accuracy_svm.append(accuracy_score(y_train, y_pred_train_svm))
    testing_accuracy_svm.append(accuracy_score(y_test, y_pred_test_svm))
    training_precision_svm.append(precision_score(y_train, y_pred_train_svm, average='macro', zero_division=0))
    testing_precision_svm.append(precision_score(y_test, y_pred_test_svm, average='macro', zero_division=0))
    training_recall_svm.append(recall_score(y_train, y_pred_train_svm, average='macro'))
    testing_recall_svm.append(recall_score(y_test, y_pred_test_svm, average='macro'))
    training_f1_svm.append(f1_score(y_train, y_pred_train_svm, average='macro'))
    testing_f1_svm.append(f1_score(y_test, y_pred_test_svm, average='macro'))

# Function to make plot for Logistic Regression
def make_plot_svm(axs, C_values):
    ax1 = axs[0,0]
    ax1.plot(C_values, training_accuracy_svm, label="Training Accuracy")
    ax1.plot(C_values, testing_accuracy_svm, label="Testing Accuracy")
    ax1.set_ylabel("Accuracy")
    ax1.set_xlabel("C (Regularization Strength)")
    ax1.set_title("SVM: Accuracy vs. C")
    ax1.legend()

    ax2 = axs[0,1]
    ax2.plot(C_values, training_precision_svm, label="Training Precision")
    ax2.plot(C_values, testing_precision_svm, label="Testing Precision")
    ax2.set_ylabel("Precision")
    ax2.set_xlabel("C")
    ax2.set_title("SVM: Precision vs. C")
    ax2.legend()

    ax3 = axs[1,0]
    ax3.plot(C_values, training_recall_svm, label="Training Recall")
    ax3.plot(C_values, testing_recall_svm, label="Testing Recall")
    ax3.set_ylabel("Recall")
    ax3.set_xlabel("C")
    ax3.set_title("SVM: Recall vs. C")
    ax3.legend()

    ax4 = axs[1,1]
    ax4.plot(C_values, training_f1_svm, label="Training F1")
    ax4.plot(C_values, testing_f1_svm, label="Testing F1")
    ax4.set_ylabel("F1 Score")
    ax4.set_xlabel("C")
    ax4.set_title("SVM: F1 Score vs. C")
    ax4.legend()

# Create the plot
fig, axs = plt.subplots(2, 2, figsize=(15, 15))
make_plot_svm(axs, C_values)
fig.suptitle("SVM Performance Metrics for Different C Values")
plt.xscale('log')  # Setting x-axis to log scale for better visualization
plt.show()
