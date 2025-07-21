import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mlxtend.classifier import StackingClassifier
from mlxtend.feature_selection import ColumnSelector
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import copy
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

# load data
    # embeddings
df1_train = concatenate_label_data(
    "/home/juanjo/Documentos/eGeMAPS_embedding/embeddings_LA_train", "/home/juanjo/Documentos/eGeMAPS_embedding/egemaps_LA_train_labeled.csv"
    )
"""
df1_eval = concatenate_label_data(
    "/home/juanjo/Documentos/eGeMAPS_embedding/embeddings_WILD", "/home/juanjo/Documentos/eGeMAPS_embedding/egemaps_wild_labeled.csv"
)

df1_eval = concatenate_label_data(
    "/home/juanjo/Documentos/eGeMAPS_embedding/embeddings_LA_eval", "/home/juanjo/Documentos/eGeMAPS_embedding/egemaps_LA_eval_labeled.csv"
)
"""

df1_eval = concatenate_label_data(
    "/home/juanjo/Documentos/eGeMAPS_embedding/embeddings_HABLA", "/home/juanjo/Documentos/eGeMAPS_embedding/egemaps_HABLA_labeled.csv"
)
    # egemaps
df2_train = pd.read_csv("/home/juanjo/Documentos/eGeMAPS_embedding/egemaps_LA_train_labeled.csv")
#df2_eval = pd.read_csv("/home/juanjo/Documentos/eGeMAPS_embedding/egemaps_wild_labeled.csv")
#df2_eval = pd.read_csv("/home/juanjo/Documentos/eGeMAPS_embedding/egemaps_LA_eval_labeled.csv")
df2_eval = pd.read_csv("/home/juanjo/Documentos/eGeMAPS_embedding/egemaps_HABLA_labeled.csv")

cols_gmaps = ['F0semitoneFrom27.5Hz_sma3nz_amean', 'F0semitoneFrom27.5Hz_sma3nz_stddevNorm', 'F0semitoneFrom27.5Hz_sma3nz_percentile20.0', 'F0semitoneFrom27.5Hz_sma3nz_percentile50.0', 'F0semitoneFrom27.5Hz_sma3nz_percentile80.0', 'F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2', 'F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope', 'F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope', 'F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope', 'F0semitoneFrom27.5Hz_sma3nz_stddevFallingSlope', 'loudness_sma3_amean', 'loudness_sma3_stddevNorm', 'loudness_sma3_percentile20.0', 'loudness_sma3_percentile50.0', 'loudness_sma3_percentile80.0', 'loudness_sma3_pctlrange0-2', 'loudness_sma3_meanRisingSlope', 'loudness_sma3_stddevRisingSlope', 'loudness_sma3_meanFallingSlope', 'loudness_sma3_stddevFallingSlope', 'spectralFlux_sma3_amean', 'spectralFlux_sma3_stddevNorm', 'mfcc1_sma3_amean', 'mfcc1_sma3_stddevNorm', 'mfcc2_sma3_amean', 'mfcc2_sma3_stddevNorm', 'mfcc3_sma3_amean', 'mfcc3_sma3_stddevNorm', 'mfcc4_sma3_amean', 'mfcc4_sma3_stddevNorm', 'jitterLocal_sma3nz_amean', 'jitterLocal_sma3nz_stddevNorm', 'shimmerLocaldB_sma3nz_amean', 'shimmerLocaldB_sma3nz_stddevNorm', 'HNRdBACF_sma3nz_amean', 'HNRdBACF_sma3nz_stddevNorm', 'logRelF0-H1-H2_sma3nz_amean', 'logRelF0-H1-H2_sma3nz_stddevNorm', 'logRelF0-H1-A3_sma3nz_amean', 'logRelF0-H1-A3_sma3nz_stddevNorm', 'F1frequency_sma3nz_amean', 'F1frequency_sma3nz_stddevNorm', 'F1bandwidth_sma3nz_amean', 'F1bandwidth_sma3nz_stddevNorm', 'F1amplitudeLogRelF0_sma3nz_amean', 'F1amplitudeLogRelF0_sma3nz_stddevNorm', 'F2frequency_sma3nz_amean', 'F2frequency_sma3nz_stddevNorm', 'F2bandwidth_sma3nz_amean', 'F2bandwidth_sma3nz_stddevNorm', 'F2amplitudeLogRelF0_sma3nz_amean', 'F2amplitudeLogRelF0_sma3nz_stddevNorm', 'F3frequency_sma3nz_amean', 'F3frequency_sma3nz_stddevNorm', 'F3bandwidth_sma3nz_amean', 'F3bandwidth_sma3nz_stddevNorm', 'F3amplitudeLogRelF0_sma3nz_amean', 'F3amplitudeLogRelF0_sma3nz_stddevNorm', 'alphaRatioV_sma3nz_amean', 'alphaRatioV_sma3nz_stddevNorm', 'hammarbergIndexV_sma3nz_amean', 'hammarbergIndexV_sma3nz_stddevNorm', 'slopeV0-500_sma3nz_amean', 'slopeV0-500_sma3nz_stddevNorm', 'slopeV500-1500_sma3nz_amean', 'slopeV500-1500_sma3nz_stddevNorm', 'spectralFluxV_sma3nz_amean', 'spectralFluxV_sma3nz_stddevNorm', 'mfcc1V_sma3nz_amean', 'mfcc1V_sma3nz_stddevNorm', 'mfcc2V_sma3nz_amean', 'mfcc2V_sma3nz_stddevNorm', 'mfcc3V_sma3nz_amean', 'mfcc3V_sma3nz_stddevNorm', 'mfcc4V_sma3nz_amean', 'mfcc4V_sma3nz_stddevNorm', 'alphaRatioUV_sma3nz_amean', 'hammarbergIndexUV_sma3nz_amean', 'slopeUV0-500_sma3nz_amean', 'slopeUV500-1500_sma3nz_amean', 'spectralFluxUV_sma3nz_amean', 'loudnessPeaksPerSec', 'VoicedSegmentsPerSec', 'MeanVoicedSegmentLengthSec', 'StddevVoicedSegmentLengthSec', 'MeanUnvoicedSegmentLength', 'StddevUnvoicedSegmentLength', 'equivalentSoundLevel_dBp']

# merge all the features and
df_train = pd.merge(df1_train, df2_train, on="audio")
features_df = pd.DataFrame(
    df_train["embedding"].tolist(),
    index=df_train.index).add_prefix("feat_")
feat_names = features_df.columns.to_list()
df_train = df_train.drop(columns=["embedding"])
df_train = pd.concat([df_train, features_df], axis=1)
y_train = df_train["label_x"]
x_train = df_train.filter(items=cols_gmaps + feat_names)

df1_eval["audio"] = df1_eval["audio"].apply(lambda x: x.replace(".wav", ""))
df2_eval["audio"] = df2_eval["audio"].apply(lambda x: x.replace(".wav", ""))
df_eval = pd.merge(df1_eval, df2_eval, on="audio")
features_df = pd.DataFrame(df_eval["embedding"].tolist(),
                           index=df_eval.index).add_prefix("feat_")
df_eval = df_eval.drop(columns=["embedding"])
df_eval = pd.concat([df_eval, features_df], axis=1)
y_eval = df_eval["label_x"]
x_eval = df_eval.filter(items=cols_gmaps + feat_names)

# definimos los dos clasificadores
svm1 = Pipeline([
    ("columnselector", ColumnSelector(cols=cols_gmaps)),
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel="linear", C=0.1))
])

svm2 = Pipeline([
    ("columnselector", ColumnSelector(cols=feat_names)),
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel="linear", C=0.1))
])

svm1.fit(x_train, y_train)
svm2.fit(x_train, y_train)

# definimoos el meta clasificador y lo entrenamos
sclf = StackingClassifier(classifiers=[svm1, svm2],
                          meta_classifier=LogisticRegression(), 
                          fit_base_estimators=True)
sclf.fit(x_train, y_train)
print("Entrenado")

# evaluamos
#y_score = sclf.predict_proba(x_eval)
y_score = sclf.decision_function(x_eval)
print(y_score)
y_pred = [0 if x < 0.5 else 1 for x in y_score]

acc, m = metrics.evaluate_model(y_true=y_eval, y_pred=y_pred)
eer = metrics.compute_eer(y_true=y_eval, y_scores=y_score)
roc_auc, mas_cosas = metrics.plot_roc_curve(y_true=y_eval.values, y_scores=y_score)

print("accuracy: ", acc, "EER: ", eer, "AUC: ", roc_auc)
print("Reporte ")
print(classification_report(y_eval, y_pred))