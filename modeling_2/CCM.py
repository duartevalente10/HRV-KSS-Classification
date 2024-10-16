# append the path of the parent directory
import sys
sys.path.append("..")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from pre_process_1 import pca, svd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

# get data
df_2_min = pd.read_csv('../pre_process_2/datasets/supervised/supervised_HRV_KSS_5_min_filtered.csv')
df_5_min = pd.read_csv('../pre_process_2/datasets/supervised/supervised_HRV_KSS_5_min_filtered.csv')

# get features names
df_time_names = pd.read_csv('../pre_process_2/datasets/hrv/hrv_time_domain_2_min.csv').columns.to_list()
df_freq_names = pd.read_csv('../pre_process_2/datasets/hrv/hrv_freq_domain_2_min.csv').columns.to_list()

# add target name
df_time_names.append('kss_answer')
df_freq_names.append('kss_answer')

# select data from superdised dataset 2 min
df_time_2_min = df_2_min[df_time_names]
df_freq_2_min = df_2_min[df_freq_names]

# select data from superdised dataset 5 min
df_time_5_min = df_5_min[df_time_names]
df_freq_5_min = df_5_min[df_freq_names]

# remove variables that dont relate to the objective of this thesis
df_2_min = df_2_min[(df_2_min.columns.difference(['HRV_SDANN1','HRV_SDNNI1','HRV_SDANN2','HRV_SDNNI2','HRV_SDANN5','HRV_SDNNI5','HRV_ULF','HRV_VLF','Interval_Start', 'Interval_End', 'Filename'], sort=False))]
df_5_min = df_5_min[(df_5_min.columns.difference(['HRV_SDANN1','HRV_SDNNI1','HRV_SDANN2','HRV_SDNNI2','HRV_SDANN5','HRV_SDNNI5','HRV_ULF','HRV_VLF','Interval_Start', 'Interval_End', 'Filename'], sort=False))]
df_time_2_min = df_time_2_min[(df_time_2_min.columns.difference(['HRV_SDANN1','HRV_SDNNI1','HRV_SDANN2','HRV_SDNNI2','HRV_SDANN5','HRV_SDNNI5','Interval_Start', 'Interval_End', 'Filename'], sort=False))]
df_time_5_min = df_time_5_min[(df_time_5_min.columns.difference(['HRV_SDANN1','HRV_SDNNI1','HRV_SDANN2','HRV_SDNNI2','HRV_SDANN5','HRV_SDNNI5','Interval_Start', 'Interval_End', 'Filename'], sort=False))]
df_freq_2_min = df_freq_2_min[(df_freq_2_min.columns.difference(['HRV_ULF','HRV_VLF','Interval_Start', 'Interval_End', 'Filename'], sort=False))]
df_freq_5_min = df_freq_5_min[(df_freq_5_min.columns.difference(['HRV_ULF','HRV_VLF','Interval_Start', 'Interval_End', 'Filename'], sort=False))]

# drop null instances
df_2_min = df_2_min.dropna()
df_time_2_min = df_time_2_min.dropna()
df_freq_2_min = df_freq_2_min.dropna()
df_5_min = df_5_min.dropna()
df_time_5_min = df_time_5_min.dropna()
df_freq_5_min = df_freq_5_min.dropna()

# reformat 'kss_answer' column with new conditions
df_2_min['kss_answer'] = df_2_min['kss_answer'].apply(lambda x: 0 if x < 7 else 1)
df_time_2_min['kss_answer'] = df_time_2_min['kss_answer'].apply(lambda x: 0 if x < 7 else 1)
df_freq_2_min['kss_answer'] = df_freq_2_min['kss_answer'].apply(lambda x: 0 if x < 7 else 1)
df_5_min['kss_answer'] = df_5_min['kss_answer'].apply(lambda x: 0 if x < 7 else 1)
df_time_5_min['kss_answer'] = df_time_5_min['kss_answer'].apply(lambda x: 0 if x < 7 else 1)
df_freq_5_min['kss_answer'] = df_freq_5_min['kss_answer'].apply(lambda x: 0 if x < 7 else 1)

### Prepare data ###

# set target feature
target_2_min = df_2_min['kss_answer']
target_time_2_min = df_time_2_min['kss_answer']
target_freq_2_min = df_freq_2_min['kss_answer']
target_5_min = df_5_min['kss_answer']
target_time_5_min = df_time_5_min['kss_answer']
target_freq_5_min = df_freq_5_min['kss_answer']

# remove target from data
data_2_min = df_2_min.drop('kss_answer', axis=1)
data_time_2_min = df_time_2_min.drop('kss_answer', axis=1)
data_freq_2_min = df_freq_2_min.drop('kss_answer', axis=1)
data_5_min = df_5_min.drop('kss_answer', axis=1)
data_time_5_min = df_time_5_min.drop('kss_answer', axis=1)
data_freq_5_min = df_freq_5_min.drop('kss_answer', axis=1)

### Normalize Data ####

scaler = StandardScaler()

# fit and transform the features
scaled_features_2_min = scaler.fit_transform(data_2_min)
scaled_features_time_2_min = scaler.fit_transform(data_time_2_min)
scaled_features_freq_2_min = scaler.fit_transform(data_freq_2_min)
scaled_features_5_min = scaler.fit_transform(data_5_min)
scaled_features_time_5_min = scaler.fit_transform(data_time_5_min)
scaled_features_freq_5_min = scaler.fit_transform(data_freq_5_min)

# create a dataFrame with the scaled features
data_2_min = pd.DataFrame(scaled_features_2_min, columns=data_2_min.columns)
data_time_2_min = pd.DataFrame(scaled_features_time_2_min, columns=data_time_2_min.columns)
data_freq_2_min = pd.DataFrame(scaled_features_freq_2_min, columns=data_freq_2_min.columns)
data_5_min = pd.DataFrame(scaled_features_5_min, columns=data_5_min.columns)
data_time_5_min = pd.DataFrame(scaled_features_time_5_min, columns=data_time_5_min.columns)
data_freq_5_min = pd.DataFrame(scaled_features_freq_5_min, columns=data_freq_5_min.columns)

# # FR PCA
# data_2_min, _ = pca(data_2_min, None, 0.99, debug=True)
# data_time_2_min, _ = pca(data_time_2_min, None, 0.99, debug=True)
# data_freq_2_min, _ = pca(data_freq_2_min, None, 0.99, debug=True)
# data_5_min, _ = pca(data_5_min, None, 0.99, debug=True)
# data_time_5_min, _ = pca(data_time_5_min, None, 0.99, debug=True)
# data_freq_5_min, _ = pca(data_freq_5_min, None, 0.99, debug=True)

# FR SVD

n_components_2_min = data_2_min.shape[1] - 1
_, _, best_n_comp_2_min = svd(data_2_min, None, n_components_2_min, 0.99, debug=False)
data_2_min, _, _ = svd(data_2_min, None, best_n_comp_2_min, 0.99, debug=False)
n_components_time_2_min = data_time_2_min.shape[1] - 1
_, _, best_n_comp_time_2_min = svd(data_time_2_min, None, n_components_time_2_min, 0.99, debug=False)
data_time_2_min, _, _ = svd(data_time_2_min, None, best_n_comp_time_2_min, 0.99, debug=False)
n_components_freq_2_min = data_freq_2_min.shape[1] - 1
_, _, best_n_comp_freq_2_min = svd(data_freq_2_min, None, n_components_freq_2_min, 0.99, debug=False)
data_freq_2_min, _, _ = svd(data_freq_2_min, None, best_n_comp_freq_2_min, 0.99, debug=False)
n_components_5_min = data_5_min.shape[1] - 1
_, _, best_n_comp_5_min = svd(data_5_min, None, n_components_5_min, 0.99, debug=False)
data_5_min, _, _ = svd(data_5_min, None, best_n_comp_5_min, 0.99, debug=False)
n_components_time_5_min = data_time_5_min.shape[1] - 1
_, _, best_n_comp_time_5_min = svd(data_time_5_min, None, n_components_time_5_min, 0.99, debug=False)
data_time_5_min, _, _ = svd(data_time_5_min, None, best_n_comp_time_5_min, 0.99, debug=False)
n_components_freq_5_min = data_freq_5_min.shape[1] - 1
_, _, best_n_comp_freq_5_min = svd(data_freq_5_min, None, n_components_freq_5_min, 0.99, debug=False)
data_freq_5_min, _, _ = svd(data_freq_5_min, None, best_n_comp_freq_5_min, 0.99, debug=False)

# define the radom forest classifiers with vest params
rf_2_min = RandomForestClassifier(criterion='entropy', min_samples_split=0.02, n_estimators=200, random_state=42)
rf_time_2_min = RandomForestClassifier(criterion='gini', min_samples_split=0.02, n_estimators=100, random_state=42)
rf_freq_2_min = RandomForestClassifier(criterion='gini', min_samples_split=0.05, n_estimators=150, random_state=42)
rf_5_min = RandomForestClassifier(criterion='gini', min_samples_split=0.02, n_estimators=150, random_state=42)
rf_time_5_min = RandomForestClassifier(criterion='entropy', min_samples_split=0.02, n_estimators=150, random_state=42)
rf_freq_5_min = RandomForestClassifier(criterion='gini', min_samples_split=0.05, n_estimators=200, random_state=42)

# split train and test data
X_train_2_min, X_test_2_min, y_train_2_min, y_test_2_min = train_test_split(data_2_min, target_2_min, test_size=0.2, random_state=42)
X_train_time_2_min, X_test_time_2_min, y_train_time_2_min, y_test_time_2_min = train_test_split(data_time_2_min, target_time_2_min, test_size=0.2, random_state=42)
X_train_freq_2_min, X_test_freq_2_min, y_train_freq_2_min, y_test_freq_2_min = train_test_split(data_freq_2_min, target_freq_2_min, test_size=0.2, random_state=42)
X_train_5_min, X_test_5_min, y_train_5_min, y_test_5_min = train_test_split(data_5_min, target_5_min, test_size=0.2, random_state=42)
X_train_time_5_min, X_test_time_5_min, y_train_time_5_min, y_test_time_5_min = train_test_split(data_time_5_min, target_time_5_min, test_size=0.2, random_state=42)
X_train_freq_5_min, X_test_freq_5_min, y_train_freq_5_min, y_test_freq_5_min = train_test_split(data_freq_5_min, target_freq_5_min, test_size=0.2, random_state=42)

#y_test_2_min y_test_time_2_min y_test_freq_2_min y_test_5_min y_train_time_5_min y_train_freq_5_min

print(len(y_test_2_min))
print(len(y_test_time_2_min))
print(len(y_test_freq_2_min))
print(len(y_test_5_min))
print(len(y_test_time_5_min))
print(len(y_test_freq_5_min))

# train classifiers
rf_2_min.fit(X_train_2_min, y_train_2_min)
rf_time_2_min.fit(X_train_time_2_min, y_train_time_2_min)
rf_freq_2_min.fit(X_train_freq_2_min, y_train_freq_2_min)
rf_5_min.fit(X_train_5_min, y_train_5_min)
rf_time_5_min.fit(X_train_time_5_min, y_train_time_5_min)
rf_freq_5_min.fit(X_train_freq_5_min, y_train_freq_5_min)

# rpedict
pred_rf_2_min = rf_2_min.predict(X_test_2_min)
pred_rf_time_2_min = rf_time_2_min.predict(X_test_time_2_min)
pred_rf_freq_2_min = rf_freq_2_min.predict(X_test_freq_2_min)
pred_rf_5_min = rf_5_min.predict(X_test_5_min)
pred_rf_time_5_min = rf_time_5_min.predict(X_test_time_5_min)
pred_rf_freq_5_min = rf_freq_5_min.predict(X_test_freq_5_min)

# merge predictions
meta_features = np.column_stack((pred_rf_2_min, pred_rf_time_2_min, pred_rf_freq_2_min, pred_rf_5_min, pred_rf_time_5_min, pred_rf_freq_5_min))

# NN to score the output
meta_clf = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1000, random_state=42)

# train the NN with the scores of RF classifeiers
meta_clf.fit(meta_features, y_test_2_min) 

# output predictions
meta_predictions = meta_clf.predict(meta_features)

# evaluate
accuracy = accuracy_score(y_test_2_min, meta_predictions)
precision = precision_score(y_test_2_min, meta_predictions)
recall = recall_score(y_test_2_min, meta_predictions)
f1 = f1_score(y_test_2_min, meta_predictions)
print("Final Combined Classifier Test Results: ")
print(f'Accuracy: {accuracy:.3f}')
print(f'Precision: {precision:.3f}')
print(f'Recall: {recall:.3f}')
print(f'F1 Score: {f1:.3f}')

# ROC curve
meta_probabilities = meta_clf.predict_proba(meta_features)[:, 1]

# FPR TPR 
fpr, tpr, _ = roc_curve(y_test_2_min, meta_probabilities)

# AUC
roc_auc = auc(fpr, tpr)

# plot RC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='tab:blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve of the Combined Classifier')
plt.legend(loc="lower right")
plt.show()

# CM
cm = confusion_matrix(y_test_2_min, meta_predictions)

# plot CM
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix of the Combined Classifier')
plt.show()



