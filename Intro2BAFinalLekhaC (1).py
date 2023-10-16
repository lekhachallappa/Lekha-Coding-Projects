##INTRO TO BA FINAL: Lekha Challappa
##Citations:
##DATA SET: https://colab.research.google.com/drive/1W6TprjcxOdXsNwswkpm_XX2U_xld9_zZ#sandboxMode=true&scrollTo=fmJ2snhqx1YD
##Scholarly:
#1. https://www.sciencedirect.com/science/article/pii/S0957417422010405
#2. https://link.springer.com/article/10.1007/s00521-023-08313-6
#3. https://jfin-swufe.springeropen.com/articles/10.1186/s40854-022-00431-9
#4. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9601484/
#5. https://www.softwareimpacts.com/article/S2665-9638(22)00069-0/fulltext
##CODE-RELATED:
#1. https://www.analyticsvidhya.com/blog/2022/05/handling-imbalanced-data-with-imbalance-learn-in-python/
#2. https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
#3. https://www.kaggle.com/code/residentmario/oversampling-with-smote-and-adasyn
#4. https://colab.research.google.com/drive/17ZRZmBtDX9meojL7LidDfJFc3FDXbfNO?usp=sharing
#5. https://colab.research.google.com/drive/1aR6boevRpfqzZYWwVYYsdyMz8rsxrqmm?usp=sharing
#6. https://colab.research.google.com/drive/19rnGbCCoFA8n0FI96-0zFihfR3QKXv0m?usp=sharing
#7. https://colab.research.google.com/drive/1C8JHLwMi0hDGFMknsr_UITbDHHK8AS7J?usp=sharing
#8. https://colab.research.google.com/drive/1Tk3iWD1MgSIUrhbvrEobmcA5PNG7Njkw?usp=sharing
## Importing
import pandas as pd
import sklearn
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE
from sklearn.ensemble import BaggingClassifier
import matplotlib as plt
import seaborn as sns
import random
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.over_sampling import ADASYN
##STEP 1: LOAD/CLEAN
# Load the dataset
file_path = "/Users/lekhac/Desktop/RIG.csv"
rig_df = pd.read_csv(file_path)
# Iterate over each column/ convert to numeric
for col in rig_df.columns:
    if col != 'date':
        rig_df[col] = pd.to_numeric(rig_df[col], errors='coerce')
# Drop duplicate
rig_df.drop_duplicates(inplace=True)
# Removing data prior to 2001, in order to maintain nature of HFT environment
rig_df = rig_df[~(rig_df['date'].str.startswith(tuple(['1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000'])))]
#convert date from object-datetime
rig_df['date'] = pd.to_datetime(rig_df['date'])
#STEP 2: REDUCING THE PERIODS
#Randomizing the period that we select (5,12)
# Constant columns to be retained
constant_columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'openint']
# Unique indicators
indicators = ['sma', 'ema', 'wma', 'bbands_high', 'bbands_low', 'per_b_high', 'per_b_low', 'trima',
              'rsi', 'willr', 'atr', 'trange', 'plus_di', 'minus_di', 'dx', 'adx', 'roc', 'macd',
              'macd_histogram', 'cci', 'aroon_osc', 'adl', 'chaikin_osc', 'chaikin_mf', 'obv',
              'stoch_per_k', 'stoch_per_d', 'ichi_dist_day', 'ichi_dist_week', 'ichi_dist_month']
periods = set([col.split('_')[-1] for col in rig_df.columns if '_' in col])
# Constant columns to be retained
constant_columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'openint']
# Unique indicators
indicators = ['sma', 'ema', 'wma', 'bbands_high', 'bbands_low', 'per_b_high', 'per_b_low', 'trima',
              'rsi', 'willr', 'atr', 'trange', 'plus_di', 'minus_di', 'dx', 'adx', 'roc', 'macd',
              'macd_histogram', 'cci', 'aroon_osc', 'adl', 'chaikin_osc', 'chaikin_mf', 'obv',
              'stoch_per_k', 'stoch_per_d', 'ichi_dist_day', 'ichi_dist_week', 'ichi_dist_month']
# selected periods (randomized search loop ran prior)
selected_periods = ['5', '12']
selected_columns = []
# Loop through the indicators to filter columns based on the selected periods
for indicator in indicators:
    for period in selected_periods:
        # Main indicator column for the period
        column_name = f"{indicator}_{period}"
        if column_name in rig_df.columns:
            selected_columns.append(column_name)

        # Associated 'label', 'weight', 'return' columns for the selected period
        label_col = f"{indicator}_{period}_label"
        weight_col = f"{indicator}_{period}_weight"
        return_col = f"{indicator}_{period}_return"
        if label_col in rig_df.columns:
            selected_columns.append(label_col)
        if weight_col in rig_df.columns:
            selected_columns.append(weight_col)
        if return_col in rig_df.columns:
            selected_columns.append(return_col)

# Combine constant columns and selected columns
columns_to_keep = constant_columns + selected_columns
rig_df = rig_df[columns_to_keep]
##STEP 3: DEFINING ARBITRAGE BASED ON EXTERNAL RESEARCH PAPERS
#arbitrage "formula"
def define_arbitrage_potential(row):
    # RSI: Stock might be oversold or overbought
    if row['rsi_5'] < 25 or row['rsi_12'] < 25:  # Strong oversold condition
        return 1  # Potential long arbitrage
    elif row['rsi_5'] > 75 or row['rsi_12'] > 75:  # Strong overbought condition
        return 1  # Potential short arbitrage

    # MACD: Signal line crossovers
    if (row['macd_5'] > row['macd_histogram_5'] and row['macd_5'] < 0) or \
       (row['macd_12'] > row['macd_histogram_12'] and row['macd_12'] < 0):
        return 1  # Bullish momentum: Potential long arbitrage
    elif (row['macd_5'] < row['macd_histogram_5'] and row['macd_5'] > 0) or \
         (row['macd_12'] < row['macd_histogram_12'] and row['macd_12'] > 0):
        return 1  # Bearish momentum: Potential short arbitrage

    # Bollinger Bands: Price moving outside the bands can signal potential arbitrage opportunities
    if row['close'] > row['bbands_high_5'] or row['close'] > row['bbands_high_12']:  # Price moves above upper Bollinger Band
        return 1  # Potential short arbitrage
    elif row['close'] < row['bbands_low_5'] or row['close'] < row['bbands_low_12']:  # Price moves below lower Bollinger Band
        return 1  # Potential long arbitrage

    # Chaikin Oscillator: Measures momentum of the Accumulation Distribution Line
    # Positive values-buying pressure; negative values-selling pressure
    if (row['chaikin_osc_5'] > 0 and row['adx_5'] > 25) or \
       (row['chaikin_osc_12'] > 0 and row['adx_12'] > 25):
        return 1  # Strong bullish trend with buying pressure: Potential long arbitrage
    elif (row['chaikin_osc_5'] < 0 and row['adx_5'] > 25) or \
         (row['chaikin_osc_12'] < 0 and row['adx_12'] > 25):
        return 1  # Strong bearish trend with selling pressure: Potential short arbitrage

    # Price-based arbitrage: Significant intraday movement might indicate news or events affecting the stock
    price_diff_percentage = abs((row['close'] - row['open']) / row['open'])
    if price_diff_percentage > 0.03:  # Example: 3% price difference between open and close
        return 1  # Potential arbitrage due to significant intraday movement

    return 0  # No arbitrage detected
#arbitrage labeling
rig_df['arbitrage_label'] = rig_df.apply(define_arbitrage_potential, axis=1)
##STEP 4: KMEANS IN ORDER TO SEE THE DISTRIBUTION OF THESE FEATURES
#determing right K before k-means (elbow and silh.)
# Scale the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(rig_df.drop(columns=['date']))
#Results of Elbow: 3-5 clusters, Silh: 3 clusters, tried 3 clusters=BAD, 2=right number
#K-means
# Perform K-Means clustering with 3 clusters FIRST, THEN 2
kmeans = KMeans(n_clusters=2, random_state=42)
rig_df['cluster'] = kmeans.fit_predict(data_scaled)
# number of stocks in each cluster
cluster_counts = rig_df['cluster'].value_counts()
print(cluster_counts)
for cluster in range(2):
    print(f"\nCluster {cluster} samples:")
    print(rig_df[rig_df['cluster'] == cluster].head())
# Extract cluster centroids
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
# Get the correct column names for the centroids
column_names = rig_df.drop(columns=['arbitrage_label', 'cluster']).columns
# Create the centroid DataFrame with the correct columns
centroid_df = pd.DataFrame(centroids, columns=column_names)
# Print the centroid DataFrame
print(centroid_df)
# Get the centroids
centroids = kmeans.cluster_centers_
# Create a new column in the dataframe to store the distance from centroid
rig_df['distance_from_centroid'] = 0
# distance for each point from its cluster centroid
for i, row in rig_df.iterrows():
    cluster_label = row['cluster']
    # Calculate distance from the cluster's centroid
    centroid = centroids[cluster_label]
    # Exclude non-numeric columns
    data_point = row.drop(labels=['date', 'arbitrage_label', 'cluster']).values
    dist = distance.euclidean(data_point, centroid)
    rig_df.at[i, 'distance_from_centroid'] = dist

#STEP 5: HANDLING OUTLIERS/ONLY REMOVING OUTLIERS NOT RELATED TO ARBIT. ALGOR
# threshold for outliers
threshold = rig_df['distance_from_centroid'].quantile(0.95)
rig_df['outlier'] = rig_df['distance_from_centroid'] > threshold
#comparing the outliers to the arbitrage formula
relevant_outliers_df = rig_df[(rig_df['outlier'] == True) & (rig_df['arbitrage_label'] == 1)]
# Mask for irrelevant outliers
irrelevant_outliers_mask = (rig_df['outlier'] == True) & (rig_df['arbitrage_label'] == 0)
# Remove irrelevant outliers
rig_df_cleaned = rig_df[~irrelevant_outliers_mask]

##STEP 6: FEATURE ELIMINATION
# Get the cluster centroids
centroids_df = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_),
                            columns=rig_df_cleaned.drop(
                                columns=['date', 'arbitrage_label', 'cluster', 'distance_from_centroid',
                                        ]).columns)

# Calculate variance for each feature across centroids
feature_variance = centroids_df.var(axis=0)
# Display features with variance below a certain threshold, say 1% of the maximum variance
threshold = 0.01 * feature_variance.max()
low_variance_features = feature_variance[feature_variance < threshold].index.tolist()
# List to store features that are highly correlated in any cluster
redundant_features = []
# Updated threshold for high correlation
correlation_threshold = 0.95

for cluster_label in rig_df_cleaned['cluster'].unique():
    cluster_data = rig_df_cleaned[rig_df_cleaned['cluster'] == cluster_label]
    correlation_matrix = cluster_data.drop(
        columns=['date', 'arbitrage_label', 'cluster', 'distance_from_centroid']).corr()

    # Find features with high correlation
    for column in correlation_matrix.columns:
        # Check if the feature is already identified as redundant
        if column not in redundant_features:
            correlated_features = correlation_matrix[column][(correlation_matrix[column] > correlation_threshold) &
                                                             (correlation_matrix[column] < 1)].index.tolist()
            redundant_features.extend(correlated_features)

# Remove duplicates
redundant_features = list(set(redundant_features))
# Essential features based on the define_arbitrage_potential function
essential_features = ['rsi_5', 'rsi_12', 'macd_5', 'macd_histogram_5', 'macd_12', 'macd_histogram_12',
                      'bbands_high_5', 'bbands_high_12', 'bbands_low_5', 'bbands_low_12',
                      'chaikin_osc_5', 'adx_5', 'chaikin_osc_12', 'adx_12', 'close', 'open']
features_to_drop = set(low_variance_features + redundant_features)
# Remove essential features from features_to_drop
features_to_drop = [feature for feature in features_to_drop if feature not in essential_features]
# Drop the identified non-essential features from rig_df_cleaned
rig_df_cleaned.drop(columns=features_to_drop, inplace=True)

##STEP 7: CLUSTER-WISE SMOTE TO HANDLE IMBALANCE
# Separate the dataset based on clusters
clusters = rig_df_cleaned['cluster'].unique()
# Initiate SMOTE
smote = SMOTE(sampling_strategy=0.7)  # Adjust this value as per your needs
new_samples = []
for cluster in clusters:
    cluster_data = rig_df_cleaned[rig_df_cleaned['cluster'] == cluster].drop(columns=['cluster'])

    # Separate the 'date' column
    dates = cluster_data['date']
    cluster_data = cluster_data.drop(columns='date')

    X = cluster_data.drop(columns='arbitrage_label')
    y = cluster_data['arbitrage_label']

    # If the cluster has a low number of the minority class, apply SMOTE
    if sum(y) < len(y) * 0.1:  # Example threshold, adjust as needed
        X_resampled, y_resampled = smote.fit_resample(X, y)
        # Add back the 'date' column
        X_resampled['date'] = dates.iloc[:len(X_resampled)].values
        resampled_data = pd.concat([X_resampled, y_resampled], axis=1)
        new_samples.append(resampled_data)
# Merge the original data and the new samples
oversampled_data = pd.concat([rig_df_cleaned, *new_samples], ignore_index=True)
# Re-run clustering
kmeans_new = KMeans(n_clusters=2, random_state=42)  # Or another suitable number of clusters
oversampled_data['cluster'] = kmeans_new.fit_predict(oversampled_data.drop(columns=['arbitrage_label', 'date']))
# Display the number of stocks in each new cluster
cluster_counts = oversampled_data['cluster'].value_counts()
## 0-3482, 1-737


##STEP 8: SUPERVISED LEARNING PRE
##A. TRAIN TEST SPLIT
# Drop the cluster column for supervised learning
data = oversampled_data.drop(columns='cluster')
# Define features/target variable
X = data.drop(columns=['date', 'arbitrage_label'])
y = data['arbitrage_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

##B. Apply ENN to further handle minority class imbalance
enn = EditedNearestNeighbours()
X_resampled, y_resampled = enn.fit_resample(X_train, y_train)

##C. Use ADASYN to oversample the minority class (AMAZING RESULTS)
adasyn = ADASYN(sampling_strategy='auto', random_state=42)  # Adjust 'sampling_strategy' if necessary
X_resampled_ad, y_resampled_ad = adasyn.fit_resample(X_resampled, y_resampled)
# Scale the features
scaler = StandardScaler()
X_resampled_scaled = scaler.fit_transform(X_resampled_ad)
X_test_scaled = scaler.transform(X_test)

##STEP 9: K-NN (UNCOMMENT AFTER LOG/DT)-average results, low recall
# Hyperparameter grid/Bagging
# Instantiate a K-NN classifier
#knn = KNeighborsClassifier()

# Use bagging with K-NN as the base estimator
#bagged_knn = BaggingClassifier(base_estimator=knn, random_state=42)

# Hyperparameter grid for both bagging and K-NN
#param_grid_bagged_knn = {
    #'base_estimator__n_neighbors': list(range(1, 31)),
    #'base_estimator__weights': ['uniform', 'distance'],
    #'base_estimator__metric': ['euclidean', 'manhattan', 'minkowski'],
    #'n_estimators': [10, 30, 50, 70, 90]  # Number of K-NN instances to train
#}
# GridSearchCV
#grid_search_bagged_knn = GridSearchCV(bagged_knn, param_grid_bagged_knn, cv=5, scoring='f1', n_jobs=-1)
#grid_search_bagged_knn.fit(X_train_scaled, y_train)
# best estimator
#best_bagged_knn = grid_search_bagged_knn.best_estimator_
# Predict on the test
#y_pred_best_bagged_knn = best_bagged_knn.predict(X_test_scaled)
# Evaluate the K-NN performance
#print("Optimized Bagged K-NN Performance:")
#print(confusion_matrix(y_test, y_pred_best_bagged_knn))
#print(classification_report(y_test, y_pred_best_bagged_knn))

##STEP 10: LOGISTIC REGRESSION- better results, still not great
# # Instantiate the logistic regression classifier
# log_reg = LogisticRegression(max_iter=10000, random_state=42)
# # hyperparameter grid
# # elasticnet
# param_grid_elasticnet = {
#     'penalty': ['elasticnet'],
#     'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
#     'solver': ['saga'],
#     'l1_ratio': [0, 0.5, 1]
# }
#
# # l1 and l2 penalties
# param_grid_l1_l2 = {
#     'penalty': ['l1', 'l2'],
#     'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
#     'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
# }
#
# # Combine the parameter grids
# param_grid_log_reg = [param_grid_elasticnet, param_grid_l1_l2]
#
# # GridSearchCV
# grid_search_log_reg = GridSearchCV(log_reg, param_grid_log_reg, cv=5, scoring='f1', n_jobs=-1)
# grid_search_log_reg.fit(X_resampled_scaled, y_resampled_ad)
#
# # best estimator
# best_log_reg = grid_search_log_reg.best_estimator_
#
# # Predict on the test
# y_pred_best_log_reg = best_log_reg.predict(X_test_scaled)
#
# # Evaluate the performance
# print("Optimized Logistic Regression Performance:")
# print(confusion_matrix(y_test, y_pred_best_log_reg))
# print(classification_report(y_test, y_pred_best_log_reg))

##STEP 11: DT CLASSIFIER-AMAZING RESULTS
# Parameter grid
param_grid = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2', None]
}
dtree = DecisionTreeClassifier()
grid_search = GridSearchCV(dtree, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_resampled_scaled, y_resampled_ad)
# best estimator
best_dtree = grid_search.best_estimator_
y_pred_dtree = best_dtree.predict(X_test_scaled)
print("Optimized Decision Tree Performance:")
print(confusion_matrix(y_test, y_pred_dtree))
print(classification_report(y_test, y_pred_dtree))
