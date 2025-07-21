import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sklearn
from sklearn.preprocessing import LabelEncoder

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import metrics
from sklearn.metrics import classification_report

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
    labels["audio"] = labels["audio"].apply(lambda x: x.replace(".wav", ""))
    #df = pd.merge(df, labels.filter(items=["audio", "label_x"]), on="audio")
    #print(labels.columns.to_list)
    df = pd.merge(df, labels.filter(items=["audio", "label"]), on="audio")
    print(len(df))
    return df

def pred_late_fusion(models, list_eval):
    # los dataframes de los datasets tienen que estar ordenados primeramente
    n = len(models) # number of models we trained
    p = [models[i].decision_function(list_eval[i]) for i in range(n)]

    y_scores = [0 for i in range(len(list_eval[0]))]
    for pred in p:
        for i in range(len(pred)):
            y_scores[i] += pred[i]
    return [x/float(n) for x in y_scores]

# modelo con embeddings
df1_train = concatenate_label_data(
    "/home/juanjo/Documentos/eGeMAPS_embedding/embeddings_LA_train", "/home/juanjo/Documentos/eGeMAPS_embedding/egemaps_LA_train_labeled.csv"
    )
"""
df1_eval = concatenate_label_data(
    "/home/juanjo/Documentos/eGeMAPS_embedding/embeddings_LA_eval", "/home/juanjo/Documentos/eGeMAPS_embedding/egemaps_LA_eval_labeled.csv"
)

df1_eval = concatenate_label_data(
    "/home/juanjo/Documentos/eGeMAPS_embedding/embeddings_HABLA", "/home/juanjo/Documentos/eGeMAPS_embedding/egemaps_HABLA_labeled.csv"
)
"""
df1_eval = concatenate_label_data(
    "/home/juanjo/Documentos/eGeMAPS_embedding/embeddings_WILD", "/home/juanjo/Documentos/eGeMAPS_embedding/egemaps_wild_labeled.csv"
)

df1_train = df1_train.sort_values(by="audio").reset_index(drop=True)
df1_eval = df1_eval.sort_values(by="audio").reset_index(drop=True)

x1_train = df1_train.drop(columns=["label", "audio"])
y1_train = df1_train["label"]

x1_eval = df1_eval.drop(columns=["label", "audio"])
y1_eval = df1_eval["label"]

x1_train = np.vstack(x1_train["embedding"].values)
x1_eval = np.vstack(x1_eval["embedding"].values)

svm1 = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel="linear", C=0.1))
])
svm1.fit(x1_train, y1_train.values)

# modelo con egemaps
df2_train = pd.read_csv("/home/juanjo/Documentos/eGeMAPS_embedding/egemaps_LA_train_labeled.csv")
df2_train["audio"] = df2_train["audio"].apply(lambda x: x.replace(".wav", ""))

#df2_eval = pd.read_csv("/home/juanjo/Documentos/eGeMAPS_embedding/egemaps_LA_eval_labeled.csv")
#df2_eval = pd.read_csv("/home/juanjo/Documentos/eGeMAPS_embedding/egemaps_HABLA_labeled.csv")
df2_eval = pd.read_csv("/home/juanjo/Documentos/eGeMAPS_embedding/egemaps_wild_labeled.csv")
df2_eval["audio"] = df2_eval["audio"].apply(lambda x: x.replace(".wav", ""))

df2_eval = df2_eval.sort_values(by="audio").reset_index(drop=True)
df2_train = df2_train.sort_values(by="audio").reset_index(drop=True)

cols_gmaps = ['F0semitoneFrom27.5Hz_sma3nz_amean', 'F0semitoneFrom27.5Hz_sma3nz_stddevNorm', 'F0semitoneFrom27.5Hz_sma3nz_percentile20.0', 'F0semitoneFrom27.5Hz_sma3nz_percentile50.0', 'F0semitoneFrom27.5Hz_sma3nz_percentile80.0', 'F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2', 'F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope', 'F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope', 'F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope', 'F0semitoneFrom27.5Hz_sma3nz_stddevFallingSlope', 'loudness_sma3_amean', 'loudness_sma3_stddevNorm', 'loudness_sma3_percentile20.0', 'loudness_sma3_percentile50.0', 'loudness_sma3_percentile80.0', 'loudness_sma3_pctlrange0-2', 'loudness_sma3_meanRisingSlope', 'loudness_sma3_stddevRisingSlope', 'loudness_sma3_meanFallingSlope', 'loudness_sma3_stddevFallingSlope', 'spectralFlux_sma3_amean', 'spectralFlux_sma3_stddevNorm', 'mfcc1_sma3_amean', 'mfcc1_sma3_stddevNorm', 'mfcc2_sma3_amean', 'mfcc2_sma3_stddevNorm', 'mfcc3_sma3_amean', 'mfcc3_sma3_stddevNorm', 'mfcc4_sma3_amean', 'mfcc4_sma3_stddevNorm', 'jitterLocal_sma3nz_amean', 'jitterLocal_sma3nz_stddevNorm', 'shimmerLocaldB_sma3nz_amean', 'shimmerLocaldB_sma3nz_stddevNorm', 'HNRdBACF_sma3nz_amean', 'HNRdBACF_sma3nz_stddevNorm', 'logRelF0-H1-H2_sma3nz_amean', 'logRelF0-H1-H2_sma3nz_stddevNorm', 'logRelF0-H1-A3_sma3nz_amean', 'logRelF0-H1-A3_sma3nz_stddevNorm', 'F1frequency_sma3nz_amean', 'F1frequency_sma3nz_stddevNorm', 'F1bandwidth_sma3nz_amean', 'F1bandwidth_sma3nz_stddevNorm', 'F1amplitudeLogRelF0_sma3nz_amean', 'F1amplitudeLogRelF0_sma3nz_stddevNorm', 'F2frequency_sma3nz_amean', 'F2frequency_sma3nz_stddevNorm', 'F2bandwidth_sma3nz_amean', 'F2bandwidth_sma3nz_stddevNorm', 'F2amplitudeLogRelF0_sma3nz_amean', 'F2amplitudeLogRelF0_sma3nz_stddevNorm', 'F3frequency_sma3nz_amean', 'F3frequency_sma3nz_stddevNorm', 'F3bandwidth_sma3nz_amean', 'F3bandwidth_sma3nz_stddevNorm', 'F3amplitudeLogRelF0_sma3nz_amean', 'F3amplitudeLogRelF0_sma3nz_stddevNorm', 'alphaRatioV_sma3nz_amean', 'alphaRatioV_sma3nz_stddevNorm', 'hammarbergIndexV_sma3nz_amean', 'hammarbergIndexV_sma3nz_stddevNorm', 'slopeV0-500_sma3nz_amean', 'slopeV0-500_sma3nz_stddevNorm', 'slopeV500-1500_sma3nz_amean', 'slopeV500-1500_sma3nz_stddevNorm', 'spectralFluxV_sma3nz_amean', 'spectralFluxV_sma3nz_stddevNorm', 'mfcc1V_sma3nz_amean', 'mfcc1V_sma3nz_stddevNorm', 'mfcc2V_sma3nz_amean', 'mfcc2V_sma3nz_stddevNorm', 'mfcc3V_sma3nz_amean', 'mfcc3V_sma3nz_stddevNorm', 'mfcc4V_sma3nz_amean', 'mfcc4V_sma3nz_stddevNorm', 'alphaRatioUV_sma3nz_amean', 'hammarbergIndexUV_sma3nz_amean', 'slopeUV0-500_sma3nz_amean', 'slopeUV500-1500_sma3nz_amean', 'spectralFluxUV_sma3nz_amean', 'loudnessPeaksPerSec', 'VoicedSegmentsPerSec', 'MeanVoicedSegmentLengthSec', 'StddevVoicedSegmentLengthSec', 'MeanUnvoicedSegmentLength', 'StddevUnvoicedSegmentLength', 'equivalentSoundLevel_dBp']

x2_train = df2_train[cols_gmaps]
y2_train = df2_train["label"]

x2_eval = df2_eval[cols_gmaps]
y2_eval = df2_eval["label"]

svm2 = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel="linear", C=0.1))
])
svm2.fit(x2_train.values, y2_train.values)

# predictions
models = [svm1, svm2]
list_eval = [x1_eval, x2_eval.values]

y_scores = pred_late_fusion(models=models, list_eval=list_eval)
#y_pred = [lambda x: int(0) if x < 0.5 else int(1) for x in y_scores]
y_pred = [0 if x < 0.5 else 1 for x in y_scores]

#le = LabelEncoder()
#y_pred = le.transform(y_pred)

# evaluate the model
# compute accuracy and confusion matrix
acc, m = metrics.evaluate_model(y_true=y2_eval, y_pred=y_pred)
eer = metrics.compute_eer(y_true=y2_eval, y_scores=y_scores)
roc_auc, mas_cosas = metrics.plot_roc_curve(y_true=y2_eval.values, y_scores=y_scores)

print("accuracy: ", acc, "EER: ", eer, "AUC: ", roc_auc)
print("Reporte ")
print(classification_report(y2_eval, y_pred))