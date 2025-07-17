import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
import metrics
from sklearn.metrics import classification_report

# load the trainning and the evaluation data
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
    df = pd.merge(df, labels.filter(items=["audio", "label"]), on="audio")
    print(len(df))
    return df

df_train = concatenate_label_data(
    "/home/juanjo/Documentos/eGeMAPS_embedding/embeddings_LA_train", "/home/juanjo/Documentos/eGeMAPS_embedding/egemaps_LA_train_labeled.csv"
    )

"""
df_eval = concatenate_label_data(
    "/home/juanjo/Documentos/eGeMAPS_embedding/embeddings_LA_eval", "/home/juanjo/Documentos/eGeMAPS_embedding/egemaps_LA_eval_labeled.csv"
)
"""
df_eval = concatenate_label_data(
    "/home/juanjo/Documentos/eGeMAPS_embedding/embeddings_HABLA", "/home/juanjo/Documentos/eGeMAPS_embedding/egemaps_HABLA_labeled.csv"
)

# train the linear svm
x_train = df_train.drop(columns=["label", "audio"])
y_train = df_train["label"]
x_eval = df_eval.drop(columns=["label", "audio"])
y_eval = df_eval["label"]

# expandimos el array de los embeddings en columnas
x_train = np.vstack(x_train["embedding"].values)
x_eval = np.vstack(x_eval["embedding"].values)

#svm = Pipeline([
#    ("scaler", StandardScaler()),
#    ("linear_svc", LinearSVC(C=1, loss="hinge", class_weight='balanced')),
#])
# probamos con kernel polinomico
svm = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel="linear", C=0.1))
])

svm.fit(x_train, y_train.values)

# evaluate the model
y_pred = svm.predict(x_eval)
y_scores = svm.decision_function(x_eval)

# compute accuracy and confusion matrix
acc, m = metrics.evaluate_model(y_true=y_eval, y_pred=y_pred)
eer = metrics.compute_eer(y_true=y_eval, y_scores=y_scores)
roc_auc, mas_cosas = metrics.plot_roc_curve(y_true=y_eval.values, y_scores=y_scores)

print("accuracy: ", acc, "EER: ", eer, "AUC: ", roc_auc)
print("Reporte ")
print(classification_report(y_eval, y_pred))