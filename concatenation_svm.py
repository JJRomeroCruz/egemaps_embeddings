import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import sklearn as sc

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import metrics
from sklearn.metrics import classification_report
#from wav2vec2_svm import concatenate_label_data
from sklearn.model_selection import cross_val_score

def concatenate_label_data(folder, label_file):
    list_df = []
    df = pd.DataFrame()
    labels = pd.read_csv(label_file)
    for file in os.listdir(folder):
        list_df.append(pd.read_csv(os.path.join(folder, file)))
    df = pd.concat(list_df, ignore_index=True)
    df["embedding"] = df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=' '))
    df["audio"] = df["audio"].apply(lambda x: x.replace(".flac", ""))
    df["audio"] = df["audio"].apply(lambda x: x.replace(".wav", ""))
    df = pd.merge(df, labels.filter(items=["audio", "label_x"]), on="audio")
    print(len(df))
    return df

# load the data
"""
df1_train = concatenate_label_data(
    "/home/juanjo/Documentos/eGeMAPS_embedding/embeddings_LA_train", "/home/juanjo/Documentos/eGeMAPS_embedding/egemaps_LA_train_labeled.csv"
    )
df1_eval = concatenate_label_data(
    "/home/juanjo/Documentos/eGeMAPS_embedding/embeddings_LA_eval", "/home/juanjo/Documentos/eGeMAPS_embedding/egemaps_LA_eval_labeled.csv"
)

df2_train = pd.read_csv(
    "/home/juanjo/Documentos/eGeMAPS_embedding/egemaps_LA_train_labeled.csv"
    )
df2_eval = pd.read_csv(
    "/home/juanjo/Documentos/eGeMAPS_embedding/egemaps_LA_eval_labeled.csv"
    )
"""

df1_train = concatenate_label_data(
    "/home/juanjo/Documentos/eGeMAPS_embedding/embeddings_LA_train", "/home/juanjo/Documentos/eGeMAPS_embedding/egemaps_LA_train_labeled.csv"
    )
df1_eval = concatenate_label_data(
    "/home/juanjo/Documentos/eGeMAPS_embedding/embeddings_HABLA", "/home/juanjo/Documentos/eGeMAPS_embedding/egemaps_HABLA_labeled.csv"
)


df2_train = pd.read_csv(
    "/home/juanjo/Documentos/eGeMAPS_embedding/egemaps_LA_train_labeled.csv"
    )
df2_train["audio"] = df2_train["audio"].apply(lambda x: x.replace(".wav", ""))
df2_eval = pd.read_csv(
    "/home/juanjo/Documentos/eGeMAPS_embedding/egemaps_HABLA_labeled.csv"
    )
print("Vamos a mergear")
df_train = pd.merge(df1_train, df2_train, on="audio")
df_eval = pd.merge(df1_eval, df2_eval, on="audio")
#print(df_train.columns, len(df_train))
#print("Train: ", df_train.columns(), len(df_train))
#print("Eval: ", df_eval.columns(), len(df_columns))
print("Vamos a concatenar")
features_df = pd.DataFrame(df_train["embedding"].tolist(),
                           index=df_train.index).add_prefix("feat_")
df_train = df_train.drop(columns=["embedding"])
df_train = pd.concat([df_train, features_df], axis=1)
#x_train = df_train.drop(columns=["label_x", "audio", "label_y", "Unnamed: 0.1", "Unnamed: 0"])
x_train = df_train.drop(columns=["audio", "label", "Unnamed: 0.1", "Unnamed: 0"])
y_train = df_train["label"]

features_df = pd.DataFrame(df_eval["embedding"].tolist(),
                           index=df_eval.index).add_prefix("feat_")
df_eval = df_eval.drop(columns=["embedding"])
df_eval = pd.concat([df_eval, features_df], axis=1)
x_eval = df_eval.drop(columns=["audio", "label", "Unnamed: 0.1", "Unnamed: 0", "h1", "h3", "sw"])
y_eval=df_eval["label"]

print("fit the model")
# fit the svc
svm = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel="linear", C=0.1))
])

svm.fit(x_train.values, y_train.values)

# evaluate the model
y_pred = svm.predict(x_eval.values)
y_scores = svm.decision_function(x_eval.values)

# compute accuracy and confusion matrix
acc, m = metrics.evaluate_model(y_true=y_eval, y_pred=y_pred)
eer = metrics.compute_eer(y_true=y_eval, y_scores=y_scores)
roc_auc, mas_cosas = metrics.plot_roc_curve(y_true=y_eval.values, y_scores=y_scores)

print("accuracy: ", acc, "EER: ", eer, "AUC: ", roc_auc)
print("Reporte ")
print(classification_report(y_eval, y_pred))

"""
# svc with polynomial kernel
for kernel in ['linear', 'poly', 'rbf']:
    for degree in ([2, 3] if kernel == 'poly' else [None]):
        clf = SVC(kernel=kernel, degree=degree) if degree else SVC(kernel=kernel)
        scores = cross_val_score(clf, x.values, y.values, cv=5)
        print(f"Kernel={kernel}, Degree={degree}: Mean CV accuracy={scores.mean():.4f}") 
"""

