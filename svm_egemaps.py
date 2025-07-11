import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sc
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
import metrics
from sklearn.metrics import classification_report

# load the data
folder_train = "egemaps_LA_train_labeled.csv"
df = pd.read_csv(folder_train)
y_train = df["label"]
x_train = df.drop(columns=["label", "audio"])

folder_eval = "egemaps_LA_eval_labeled.csv"
#folder_eval = "/home/juanjo/Documentos/eGeMAPS_embedding/egemaps_HABLA_labeled.csv"
df_eval = pd.read_csv(folder_eval)
y_eval = df_eval["label"]
x_eval = df_eval.drop(columns=["label", "audio"])

# start with a linear SVC
"""
svm = Pipeline([
    ("scaler", StandardScaler()),
    ("linear_svc", LinearSVC(C=1, loss="hinge", class_weight='balanced')),
])
"""
# svc with polynomial kernel
svm = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel="linear", C=0.1))
])


# fit the model with the training data
svm.fit(x_train.values, y_train.values)

# make predictions in the eval dataset
y_pred = svm.predict(x_eval.values)

y_scores = svm.decision_function(x_eval.values)

# compute accuracy and confusion matrix
acc, m = metrics.evaluate_model(y_true=y_eval, y_pred=y_pred)
eer = metrics.compute_eer(y_true=y_eval, y_scores=y_scores)
roc_auc, mas_cosas = metrics.plot_roc_curve(y_true=y_eval.values, y_scores=y_scores)

print("accuracy: ", acc, "EER: ", eer, "AUC: ", roc_auc)
print("Reporte ")
print(classification_report(y_eval, y_pred))