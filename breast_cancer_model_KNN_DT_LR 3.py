"""
    A.  We would like to perform a predictive modeling analysis on this breast cancer dataset using the
            a) Decision tree,
            b) K-NN technique and
            c) Logistic regression technique.
    B.  Build and visualize a learning curve for the logistic regression technique (visualize the
        performance for both training and test data in the same plot).
    C.  Build a fitting graph for different depths of the decision tree (visualize the performance
        for both training and test data in the same plot).
    D.  Create an ROC curve for k-NN, decision tree, and logistic regression.

    Showing AUC estimators in the ROC graph
"""

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



def chiSquaredPearsonCorrCof(X,y):
    # Normalize the features using Min-Max scaling (to ensure non-negative values)
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)

    # Method 1: Chi-squared feature selection
    # Select the top k features using chi-squared test
    k_chi2 = 10  # Number of top features to select
    chi2_selector = SelectKBest(score_func=chi2, k=k_chi2)
    X_chi2 = chi2_selector.fit_transform(X_normalized, y)

    # Method 2: Pearson Correlation Coefficient feature selection
    # Calculate Pearson correlation coefficients between features and the target
    correlation_scores = []
    for feature_idx in range(X_normalized.shape[1]):
        corr, _ = pearsonr(X_normalized[:, feature_idx], y)
        correlation_scores.append(abs(corr))

    # Select the top k features with the highest absolute correlation
    k_pearson = 10  # Number of top features to select
    top_pearson_indices = np.argsort(correlation_scores)[-k_pearson:]
    X_pearson = X_normalized[:, top_pearson_indices]

    # Print the selected feature indices for each method
    print("Selected Feature Indices (Chi-squared):", chi2_selector.get_support(indices=True))
    print("Selected Feature Indices (Pearson Correlation):", top_pearson_indices)


def histogram(breast_cancer_df):
    selected_features = ['radius_mean', 'texture_mean', 'smoothness_mean', 'compactness_mean',
                         'symmetry_se', 'radius_worst', 'texture_worst', 'smoothness_worst'
                         ]

    # Set up subplots for histograms
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(14, 8))

    # Create histograms for selected features
    for i, feature in enumerate(selected_features):
        row, col = divmod(i, 4)
        ax = axes[row, col]
        ax.hist(breast_cancer_df[feature], bins=20, color='skyblue', alpha=0.7)
        ax.set_title(f'Histogram of {feature}')
        ax.set_xlabel(feature)
        ax.set_ylabel('Frequency')

    # Adjust layout and display the histograms
    plt.tight_layout()
    plt.show()


def all_roc_curve(X, y):
    """
    Showing all the ROC graphs in one single plot
    Showing AUC estimators in the ROC graph
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    knn = KNeighborsClassifier(algorithm='auto', leaf_size=20, metric='euclidean', n_neighbors=5, p=1,
                               weights='uniform')
    tree = DecisionTreeClassifier(random_state=42, criterion='entropy', min_samples_leaf=1,
                                  min_samples_split=5, splitter='best')
    logreg = LogisticRegression(C=10, fit_intercept=True, max_iter=100, penalty='l1', random_state=42,
                                solver='liblinear')

    classifiers = [('K-NN', knn), ('Decision Tree', tree), ('Logistic Regression', logreg)]

    plt.figure(figsize=(10, 8))

    for name, clf in classifiers:
        clf.fit(X_train, y_train)
        y_prob = clf.predict_proba(X_test)[:, 1]

        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, lw=2, label='{} (AUC = {:.2f})'.format(name, roc_auc))
        # Annotate the AUC value on the graph
        plt.annotate('AUC = {:.2f}'.format(roc_auc), xy=(0.6, 0.4), xytext=(0.7, 0.6),
                     arrowprops=dict(facecolor='black', shrink=0.05))

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Multiple Classifiers')
    plt.legend(loc='lower right')
    plt.show()


def fittingGraphDT(X, y):
    """
    Build a fitting graph for different depths of the decision tree
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    max_depths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]  # Different depths to test
    train_scores = []
    test_scores = []

    for depth in max_depths:
        clf = DecisionTreeClassifier(max_depth=depth, random_state=42, criterion='entropy', min_samples_leaf=1,
                                     min_samples_split=2, splitter='best')
        clf.fit(X_train, y_train)
        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        train_scores.append(train_accuracy)
        test_scores.append(test_accuracy)
    plt.figure(figsize=(10, 6))
    plt.plot(max_depths, train_scores, marker='o', label='Training Accuracy - In sample')
    plt.plot(max_depths, test_scores, marker='o', label='Test Accuracy - Out of Sample')
    plt.title('Decision Tree Performance vs. Depth')
    plt.xlabel('Max Depth of Tree')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()


def lrLearningCurve(X, y):
    """
    Build and visualize a learning curve for the logistic regression technique
    (visualize the performance for both training and test data in the same plot).
    """

    train_performance = []
    test_performance = []
    sizes = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # 15 different sizes from 5% to 90%

    for size in sizes:
        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - size, random_state=42)

        # Initialize and train a logistic regression model
        model = LogisticRegression(C=10, fit_intercept=True, max_iter=100, penalty='l1', random_state=42,
                                   solver='liblinear')
        model.fit(X_train, y_train)

        # Make predictions on training and test data
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Calculate accuracy for both training and test data
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        # Append the accuracy scores to the lists
        train_performance.append(train_acc)
        test_performance.append(test_acc)

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the training performance
    ax.plot(sizes, train_performance, marker='o', label='Training Performance - In sample')

    # Plot the test performance
    ax.plot(sizes, test_performance, marker='o', label='Test Performance - Out of Sample')

    # Set labels and title
    ax.set_xlabel('Dataset Size')
    ax.set_ylabel('Accuracy')
    ax.set_title('Learning Curve for Breast Cancer Dataset')

    # Add a legend
    ax.legend()

    # Display the plot
    plt.grid(True)

    plt.show()


def lrModel(X, y):
    """
    param X: Dependent variable
    param y: Target variable
    return:
    """
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # Normalize the features using StandardScaler
    scaler = StandardScaler()
    X_train_normalized = scaler.fit_transform(X_train)
    X_test_normalized = scaler.transform(X_test)

    # Method 1: Recursive Feature Elimination (RFE) with Logistic Regression
    # Select the top 10 features using RFE
    logistic_regression = LogisticRegression(C=100, penalty='l1', random_state=42, solver='liblinear')
    rfe_selector = RFE(estimator=logistic_regression, n_features_to_select=10, step=1)
    X_train_rfe = rfe_selector.fit_transform(X_train_normalized, y_train)
    X_test_rfe = rfe_selector.transform(X_test_normalized)

    # Get the selected features
    selected_features = np.where(rfe_selector.support_)[0]

    # Use the selected features for modeling
    X_selected = X_train_normalized[:, selected_features]

    # Train a model with the selected features
    logistic_regression.fit(X_train_rfe, y_train)

    # Perform nested cross-validation using the selected features
    nested_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(logistic_regression, X_selected, y_train, cv=nested_cv, scoring='accuracy')

    # Calculate and print the mean accuracy of the nested cross-validation folds
    mean_accuracy = np.mean(cv_scores)
    print("Mean Accuracy (Nested Cross-Validation with Feature Selection): %.2f" % mean_accuracy)

    # Make predictions on the test set using the selected features
    y_pred_rfe = logistic_regression.predict(X_test_rfe)

    # Calculate and print performance metrics for the test set
    log_reg_accuracy = accuracy_score(y_test, y_pred_rfe)
    log_reg_precision = precision_score(y_test, y_pred_rfe)
    log_reg_recall = recall_score(y_test, y_pred_rfe)
    log_reg_f1 = f1_score(y_test, y_pred_rfe)
    log_reg_cm = confusion_matrix(y_test, y_pred_rfe)

    print("Logistic Regression Performance on Test Set:")
    print("Accuracy: %.2f" % log_reg_accuracy)
    print("Precision: %.2f" % log_reg_precision)
    print("Recall: %.2f" % log_reg_recall)
    print("F1 Score: %.2f" % log_reg_f1)
    print("Confusion Matrix:\n", log_reg_cm)


def knnModel(X, y):
    """
    param X: Dependant variable
    param y: Target variable
    return:
    """

    # Define the parameter grid for KNN
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],  # Number of neighbors to consider
        'weights': ['uniform', 'distance'],  # Weighting scheme
        'p': [1, 2],  # Power parameter for Minkowski distance (1 for Manhattan, 2 for Euclidean)
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],  # Algorithm used to compute neighbors
        'leaf_size': [20, 30, 40],  # Leaf size for efficient neighbor searches
        'metric': ['euclidean', 'manhattan', 'minkowski'],  # Distance metric
    }

    # Normalize the features using StandardScaler
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)

    # Create a KNN classifier
    knn_classifier = KNeighborsClassifier()

    # Create a GridSearchCV object for hyperparameter tuning with nested cross-validation
    nested_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        knn_classifier, param_grid, scoring='recall', cv=nested_cv, n_jobs=-1
    )
    grid_search.fit(X_normalized, y)

    # Get the best hyperparameters
    best_params = grid_search.best_params_
    print("BEST_PARAMETERS: ", best_params)
    # Create a KNN classifier with the best hyperparameters
    best_knn_classifier = KNeighborsClassifier(**best_params)

    # Perform nested cross-validation
    cv_scores = cross_val_score(
        best_knn_classifier, X_normalized, y, cv=nested_cv, scoring='recall'
    )

    # Calculate and print the mean Recall of the nested cross-validation folds
    mean_recall = np.mean(cv_scores)
    print("Mean Recall (Nested Cross-Validation): %.2f" % mean_recall)

    # Fit the best model on the entire dataset
    best_knn_classifier.fit(X_normalized, y)

    # Make predictions on the test set
    y_pred = best_knn_classifier.predict(X_normalized)

    # Calculate and print additional performance metrics
    knn_accuracy = accuracy_score(y, y_pred)
    knn_precision = precision_score(y, y_pred)
    knn_recall = recall_score(y, y_pred)
    knn_f1 = f1_score(y, y_pred)
    knn_cm = confusion_matrix(y, y_pred)

    print("K-Nearest Neighbors (KNN) Performance:")
    print("Accuracy: %.2f" % knn_accuracy)
    print("Precision: %.2f" % knn_precision)
    print("Recall: %.2f" % knn_recall)
    print("F1 Score: %.2f" % knn_f1)
    print("Confusion Matrix:\n", knn_cm)


def decisionTree(X, y):
    # HEATMATP
    # plt.figure(figsize=(18, 10))
    # sns.heatmap(X.corr(), annot=True)
    # plt.show()
    # Normalize the features using StandardScaler
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)

    # Define the parameter grid for the decision tree classifier
    param_grid = {
        'criterion': ['gini', 'entropy'],  # Splitting criterion
        'max_depth': [None, 5, 10, 15, 20],  # Maximum depth of the tree
        'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
        'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be a leaf node
    }

    # Create a DecisionTreeClassifier
    decision_tree_classifier = DecisionTreeClassifier(random_state=42)

    # Create a GridSearchCV object for hyperparameter tuning with nested cross-validation
    nested_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        decision_tree_classifier, param_grid, scoring='recall', cv=nested_cv, n_jobs=-1
    )
    grid_search.fit(X_normalized, y)

    # Get the best hyperparameters
    best_params = grid_search.best_params_
    print("BEST_PARAMETERS: ", best_params)

    # Create a DecisionTreeClassifier with the best hyperparameters
    best_decision_tree_classifier = DecisionTreeClassifier(random_state=42, **best_params)

    # Perform nested cross-validation
    cv_scores = cross_val_score(
        best_decision_tree_classifier, X_normalized, y, cv=nested_cv, scoring='recall'
    )

    # Calculate and print the mean Recall of the nested cross-validation folds
    mean_recall = np.mean(cv_scores)
    print("Mean Recall (Nested Cross-Validation): %.2f" % mean_recall)

    # Fit the best model on the entire dataset
    best_decision_tree_classifier.fit(X_normalized, y)

    # Make predictions on the test set (not needed for nested cross-validation)
    y_pred = best_decision_tree_classifier.predict(X_normalized)

    # Calculate and print additional performance metrics (not needed for nested cross-validation)
    dt_accuracy = accuracy_score(y, y_pred)
    dt_precision = precision_score(y, y_pred)
    dt_recall = recall_score(y, y_pred)
    dt_f1 = f1_score(y, y_pred)
    dt_cm = confusion_matrix(y, y_pred)

    print("Decision Tree Performance:")
    print("Accuracy: %.2f" % dt_accuracy)
    print("Precision: %.2f" % dt_precision)
    print("Recall: %.2f" % dt_recall)
    print("F1 Score: %.2f" % dt_f1)
    print("Confusion Matrix:\n", dt_cm)


# Data loading
import urllib.request
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data'
filename = 'wdbc.data.txt'
urllib.request.urlretrieve(url, filename)
#Reading the data set into a data frame called breast_cancer_df
breast_cancer_df = pd.read_csv(filename)

# Give feature names.
breast_cancer_df.columns = ['ID', 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',
                            'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
                            'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
                            'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
                            'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
                            'fractal_dimension_se', 'radius_worst', 'texture_worst',
                            'perimeter_worst', 'area_worst', 'smoothness_worst',
                            'compactness_worst', 'concavity_worst', 'concave points_worst',
                            'symmetry_worst', 'fractal_dimension_worst']

# To label the data in the second column of a CSV file (without column names) into numerical format
breast_cancer_df['diagnosis'] = breast_cancer_df['diagnosis'].map({'M': 0, 'B': 1})

# Create a dataframe with all training data except the target column
X = breast_cancer_df.drop(columns=['diagnosis', 'ID'])
y = breast_cancer_df['diagnosis']
y = y.astype('int')

"""
Best Parameters for DT Model: {'criterion': 'entropy', 'max_depth': None, 'max_features': None, 'min_samples_leaf': 1, 'min_samples_split': 5, 'splitter': 'best'}
"""
# CALL TO FUNCTION KNOW ABOUT DT MODEL
decisionTree(X, y)

"""
Best Parameters for KNN Model: {'algorithm': 'auto', 'leaf_size': 20, 'metric': 'manhattan', 'n_neighbors': 7, 'p': 1, 'weights': 'uniform'}
"""
# CALL TO FUNCTION KNOW ABOUT KNN MODEL
knnModel(X, y)


"""
Best Parameters for LR Model: {'C': 10, 'fit_intercept': True, 'max_iter': 100, 'penalty': 'l1', 'random_state': 42, 'solver': 'liblinear'}
"""
# CALL TO FUNCTION KNOW ABOUT LR MODEL
lrModel(X, y)

## LEARNING CURVE
lrLearningCurve(X, y)


## FITTING GRAPH
fittingGraphDT(X, y)

## ALL ROC CURVE WITH AUC
all_roc_curve(X, y)

## HISTOGRAM
histogram(breast_cancer_df)


## Chi squared and Pearson Corrletaion Cofficient
chiSquaredPearsonCorrCof(X,y)