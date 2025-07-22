import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import  sklearn as sc

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

from mlxtend.classifier import StackingClassifier
from mlxtend.feature_selection import ColumnSelector
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import copy

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

def egemaps_svm(folder_eval, folder_train = "/home/juanjo/Documentos/eGeMAPS_embedding/egemaps_LA_train_labeled.csv"):
    # load data
    cols_gmaps = ['F0semitoneFrom27.5Hz_sma3nz_amean', 'F0semitoneFrom27.5Hz_sma3nz_stddevNorm', 'F0semitoneFrom27.5Hz_sma3nz_percentile20.0', 'F0semitoneFrom27.5Hz_sma3nz_percentile50.0', 'F0semitoneFrom27.5Hz_sma3nz_percentile80.0', 'F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2', 'F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope', 'F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope', 'F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope', 'F0semitoneFrom27.5Hz_sma3nz_stddevFallingSlope', 'loudness_sma3_amean', 'loudness_sma3_stddevNorm', 'loudness_sma3_percentile20.0', 'loudness_sma3_percentile50.0', 'loudness_sma3_percentile80.0', 'loudness_sma3_pctlrange0-2', 'loudness_sma3_meanRisingSlope', 'loudness_sma3_stddevRisingSlope', 'loudness_sma3_meanFallingSlope', 'loudness_sma3_stddevFallingSlope', 'spectralFlux_sma3_amean', 'spectralFlux_sma3_stddevNorm', 'mfcc1_sma3_amean', 'mfcc1_sma3_stddevNorm', 'mfcc2_sma3_amean', 'mfcc2_sma3_stddevNorm', 'mfcc3_sma3_amean', 'mfcc3_sma3_stddevNorm', 'mfcc4_sma3_amean', 'mfcc4_sma3_stddevNorm', 'jitterLocal_sma3nz_amean', 'jitterLocal_sma3nz_stddevNorm', 'shimmerLocaldB_sma3nz_amean', 'shimmerLocaldB_sma3nz_stddevNorm', 'HNRdBACF_sma3nz_amean', 'HNRdBACF_sma3nz_stddevNorm', 'logRelF0-H1-H2_sma3nz_amean', 'logRelF0-H1-H2_sma3nz_stddevNorm', 'logRelF0-H1-A3_sma3nz_amean', 'logRelF0-H1-A3_sma3nz_stddevNorm', 'F1frequency_sma3nz_amean', 'F1frequency_sma3nz_stddevNorm', 'F1bandwidth_sma3nz_amean', 'F1bandwidth_sma3nz_stddevNorm', 'F1amplitudeLogRelF0_sma3nz_amean', 'F1amplitudeLogRelF0_sma3nz_stddevNorm', 'F2frequency_sma3nz_amean', 'F2frequency_sma3nz_stddevNorm', 'F2bandwidth_sma3nz_amean', 'F2bandwidth_sma3nz_stddevNorm', 'F2amplitudeLogRelF0_sma3nz_amean', 'F2amplitudeLogRelF0_sma3nz_stddevNorm', 'F3frequency_sma3nz_amean', 'F3frequency_sma3nz_stddevNorm', 'F3bandwidth_sma3nz_amean', 'F3bandwidth_sma3nz_stddevNorm', 'F3amplitudeLogRelF0_sma3nz_amean', 'F3amplitudeLogRelF0_sma3nz_stddevNorm', 'alphaRatioV_sma3nz_amean', 'alphaRatioV_sma3nz_stddevNorm', 'hammarbergIndexV_sma3nz_amean', 'hammarbergIndexV_sma3nz_stddevNorm', 'slopeV0-500_sma3nz_amean', 'slopeV0-500_sma3nz_stddevNorm', 'slopeV500-1500_sma3nz_amean', 'slopeV500-1500_sma3nz_stddevNorm', 'spectralFluxV_sma3nz_amean', 'spectralFluxV_sma3nz_stddevNorm', 'mfcc1V_sma3nz_amean', 'mfcc1V_sma3nz_stddevNorm', 'mfcc2V_sma3nz_amean', 'mfcc2V_sma3nz_stddevNorm', 'mfcc3V_sma3nz_amean', 'mfcc3V_sma3nz_stddevNorm', 'mfcc4V_sma3nz_amean', 'mfcc4V_sma3nz_stddevNorm', 'alphaRatioUV_sma3nz_amean', 'hammarbergIndexUV_sma3nz_amean', 'slopeUV0-500_sma3nz_amean', 'slopeUV500-1500_sma3nz_amean', 'spectralFluxUV_sma3nz_amean', 'loudnessPeaksPerSec', 'VoicedSegmentsPerSec', 'MeanVoicedSegmentLengthSec', 'StddevVoicedSegmentLengthSec', 'MeanUnvoicedSegmentLength', 'StddevUnvoicedSegmentLength', 'equivalentSoundLevel_dBp']
    df_train = pd.read_csv(folder_train)
    y_train = df_train["label"]
    x_train = df_train[cols_gmaps]

    df_eval = pd.read_csv(folder_eval)
    y_eval = df_eval["label"]
    x_eval = df_eval[cols_gmaps]

    # linear svc
    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="linear", C=0.1))
    ])
    svm.fit(x_train.values, y_train.values)

    # evaluation
    y_pred = svm.predict(x_eval.values)
    y_scores = svm.decision_function(x_eval.values)

    return y_pred, y_scores, y_eval 

def wav2vec2_svm(eval, train = ["/home/juanjo/Documentos/eGeMAPS_embedding/egemaps_LA_train_labeled.csv", "/home/juanjo/Documentos/eGeMAPS_embedding/embeddings_LA_train"]):
    folder_train = train[1]
    label_train = train[0]
    folder_eval = eval[1]
    label_eval = eval[0]
    
    # load data
    df_train = concatenate_label_data(
    folder_train, label_train
    )
    df_train = df_train.sort_values(by="audio").reset_index(drop=True)
    features_df = pd.DataFrame(
    df_train["embedding"].tolist(),
    index=df_train.index).add_prefix("feat_")

    feat_names = features_df.columns.to_list()
    df_train = df_train.drop(columns=["embedding"])
    df_train = pd.concat([df_train, features_df], axis=1)
    y_train = df_train["label"]
    x_train = df_train.filter(items=feat_names)

    df_eval = concatenate_label_data(
        folder_eval, label_eval
    )
    df_eval = df_eval.sort_values(by="audio").reset_index(drop=True)
    df_eval["audio"] = df_eval["audio"].apply(lambda x: x.replace(".wav", ""))
    features_df = pd.DataFrame(df_eval["embedding"].tolist(),
                           index=df_eval.index).add_prefix("feat_")
    df_eval = df_eval.drop(columns=["embedding"])
    df_eval = pd.concat([df_eval, features_df], axis=1)
    y_eval = df_eval["label"]
    x_eval = df_eval.filter(items=feat_names)

    # svc
    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="linear", C=0.1))
    ])

    svm.fit(x_train, y_train)

    # evaluate the model
    y_pred = svm.predict(x_eval)
    y_scores = svm.decision_function(x_eval)

    return y_pred, y_scores, y_eval

def concatenation_svm(eval, train=["/home/juanjo/Documentos/eGeMAPS_embedding/egemaps_LA_train_labeled.csv", "/home/juanjo/Documentos/eGeMAPS_embedding/embeddings_LA_train"]):

    folder_train_egemaps = train[0]
    folder_train_em = train[1]

    folder_eval_egemaps = eval[0]
    folder_eval_em = eval[1]

    cols_gmaps = ['F0semitoneFrom27.5Hz_sma3nz_amean', 'F0semitoneFrom27.5Hz_sma3nz_stddevNorm', 'F0semitoneFrom27.5Hz_sma3nz_percentile20.0', 'F0semitoneFrom27.5Hz_sma3nz_percentile50.0', 'F0semitoneFrom27.5Hz_sma3nz_percentile80.0', 'F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2', 'F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope', 'F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope', 'F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope', 'F0semitoneFrom27.5Hz_sma3nz_stddevFallingSlope', 'loudness_sma3_amean', 'loudness_sma3_stddevNorm', 'loudness_sma3_percentile20.0', 'loudness_sma3_percentile50.0', 'loudness_sma3_percentile80.0', 'loudness_sma3_pctlrange0-2', 'loudness_sma3_meanRisingSlope', 'loudness_sma3_stddevRisingSlope', 'loudness_sma3_meanFallingSlope', 'loudness_sma3_stddevFallingSlope', 'spectralFlux_sma3_amean', 'spectralFlux_sma3_stddevNorm', 'mfcc1_sma3_amean', 'mfcc1_sma3_stddevNorm', 'mfcc2_sma3_amean', 'mfcc2_sma3_stddevNorm', 'mfcc3_sma3_amean', 'mfcc3_sma3_stddevNorm', 'mfcc4_sma3_amean', 'mfcc4_sma3_stddevNorm', 'jitterLocal_sma3nz_amean', 'jitterLocal_sma3nz_stddevNorm', 'shimmerLocaldB_sma3nz_amean', 'shimmerLocaldB_sma3nz_stddevNorm', 'HNRdBACF_sma3nz_amean', 'HNRdBACF_sma3nz_stddevNorm', 'logRelF0-H1-H2_sma3nz_amean', 'logRelF0-H1-H2_sma3nz_stddevNorm', 'logRelF0-H1-A3_sma3nz_amean', 'logRelF0-H1-A3_sma3nz_stddevNorm', 'F1frequency_sma3nz_amean', 'F1frequency_sma3nz_stddevNorm', 'F1bandwidth_sma3nz_amean', 'F1bandwidth_sma3nz_stddevNorm', 'F1amplitudeLogRelF0_sma3nz_amean', 'F1amplitudeLogRelF0_sma3nz_stddevNorm', 'F2frequency_sma3nz_amean', 'F2frequency_sma3nz_stddevNorm', 'F2bandwidth_sma3nz_amean', 'F2bandwidth_sma3nz_stddevNorm', 'F2amplitudeLogRelF0_sma3nz_amean', 'F2amplitudeLogRelF0_sma3nz_stddevNorm', 'F3frequency_sma3nz_amean', 'F3frequency_sma3nz_stddevNorm', 'F3bandwidth_sma3nz_amean', 'F3bandwidth_sma3nz_stddevNorm', 'F3amplitudeLogRelF0_sma3nz_amean', 'F3amplitudeLogRelF0_sma3nz_stddevNorm', 'alphaRatioV_sma3nz_amean', 'alphaRatioV_sma3nz_stddevNorm', 'hammarbergIndexV_sma3nz_amean', 'hammarbergIndexV_sma3nz_stddevNorm', 'slopeV0-500_sma3nz_amean', 'slopeV0-500_sma3nz_stddevNorm', 'slopeV500-1500_sma3nz_amean', 'slopeV500-1500_sma3nz_stddevNorm', 'spectralFluxV_sma3nz_amean', 'spectralFluxV_sma3nz_stddevNorm', 'mfcc1V_sma3nz_amean', 'mfcc1V_sma3nz_stddevNorm', 'mfcc2V_sma3nz_amean', 'mfcc2V_sma3nz_stddevNorm', 'mfcc3V_sma3nz_amean', 'mfcc3V_sma3nz_stddevNorm', 'mfcc4V_sma3nz_amean', 'mfcc4V_sma3nz_stddevNorm', 'alphaRatioUV_sma3nz_amean', 'hammarbergIndexUV_sma3nz_amean', 'slopeUV0-500_sma3nz_amean', 'slopeUV500-1500_sma3nz_amean', 'spectralFluxUV_sma3nz_amean', 'loudnessPeaksPerSec', 'VoicedSegmentsPerSec', 'MeanVoicedSegmentLengthSec', 'StddevVoicedSegmentLengthSec', 'MeanUnvoicedSegmentLength', 'StddevUnvoicedSegmentLength', 'equivalentSoundLevel_dBp']

    # load the data
    df1_train = concatenate_label_data(
    folder_train_em, folder_train_egemaps
    )
    df1_train = df1_train.sort_values(by="audio").reset_index(drop=True)
    df1_train["audio"] = df1_train["audio"].apply(lambda x: x.replace(".wav", ""))

    features_df = pd.DataFrame(
    df1_train["embedding"].tolist(),
    index=df1_train.index).add_prefix("feat_")

    feat_names = features_df.columns.to_list()
    df1_train = df1_train.drop(columns=["embedding"])
    df1_train = pd.concat([df1_train, features_df], axis=1)

    df2_train = pd.read_csv(folder_train_egemaps)
    df2_train["audio"] = df2_train["audio"].apply(lambda x: x.replace(".wav", ""))
    df_train = pd.merge(df1_train, df2_train, on="audio")
    
    x_train = df_train.filter(items=cols_gmaps + feat_names)
    y_train = df_train["label_x"]


    df1_eval = concatenate_label_data(
    folder_eval_em, folder_eval_egemaps
    )
    df1_eval["audio"] = df1_eval["audio"].apply(lambda x: x.replace(".wav", ""))
    features_df = pd.DataFrame(
    df1_eval["embedding"].tolist(),
    index=df1_eval.index).add_prefix("feat_")

    feat_names = features_df.columns.to_list()
    df1_eval = df1_eval.drop(columns=["embedding"])
    df1_eval = pd.concat([df1_eval, features_df], axis=1)

    df2_eval = pd.read_csv(folder_eval_egemaps)
    df2_eval["audio"] = df2_eval["audio"].apply(lambda x: x.replace(".wav", ""))

    df_eval = pd.merge(df1_eval, df2_eval, on="audio")

    x_eval = df_eval.filter(items=cols_gmaps + feat_names)
    y_eval = df_eval["label_x"]

    # svc
    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="linear", C=0.1))
    ])

    svm.fit(x_train.values, y_train.values)

    # evaluate
    y_pred = svm.predict(x_eval.values)
    y_scores = svm.decision_function(x_eval.values)

    return y_pred, y_scores, y_eval

def late_fusion(eval, train = ["/home/juanjo/Documentos/eGeMAPS_embedding/egemaps_LA_train_labeled.csv", "/home/juanjo/Documentos/eGeMAPS_embedding/embeddings_LA_train"]):

    res1 = egemaps_svm(folder_train=train[0], folder_eval=eval[0])
    res2 = wav2vec2_svm(train=train, eval=eval)

    y_scores, y_eval = 0.5*(res1[1] + res2[1]), res1[2]
    y_pred = [0 if x < 0.5 else 1 for x in y_scores]
    return y_pred, y_scores, y_eval

def stacking(eval, train=["/home/juanjo/Documentos/eGeMAPS_embedding/egemaps_LA_train_labeled.csv", "/home/juanjo/Documentos/eGeMAPS_embedding/embeddings_LA_train"]):

    folder_train_egemaps = train[0]
    folder_train_em = train[1]

    folder_eval_egemaps = eval[0]
    folder_eval_em = eval[1]

    cols_gmaps = ['F0semitoneFrom27.5Hz_sma3nz_amean', 'F0semitoneFrom27.5Hz_sma3nz_stddevNorm', 'F0semitoneFrom27.5Hz_sma3nz_percentile20.0', 'F0semitoneFrom27.5Hz_sma3nz_percentile50.0', 'F0semitoneFrom27.5Hz_sma3nz_percentile80.0', 'F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2', 'F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope', 'F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope', 'F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope', 'F0semitoneFrom27.5Hz_sma3nz_stddevFallingSlope', 'loudness_sma3_amean', 'loudness_sma3_stddevNorm', 'loudness_sma3_percentile20.0', 'loudness_sma3_percentile50.0', 'loudness_sma3_percentile80.0', 'loudness_sma3_pctlrange0-2', 'loudness_sma3_meanRisingSlope', 'loudness_sma3_stddevRisingSlope', 'loudness_sma3_meanFallingSlope', 'loudness_sma3_stddevFallingSlope', 'spectralFlux_sma3_amean', 'spectralFlux_sma3_stddevNorm', 'mfcc1_sma3_amean', 'mfcc1_sma3_stddevNorm', 'mfcc2_sma3_amean', 'mfcc2_sma3_stddevNorm', 'mfcc3_sma3_amean', 'mfcc3_sma3_stddevNorm', 'mfcc4_sma3_amean', 'mfcc4_sma3_stddevNorm', 'jitterLocal_sma3nz_amean', 'jitterLocal_sma3nz_stddevNorm', 'shimmerLocaldB_sma3nz_amean', 'shimmerLocaldB_sma3nz_stddevNorm', 'HNRdBACF_sma3nz_amean', 'HNRdBACF_sma3nz_stddevNorm', 'logRelF0-H1-H2_sma3nz_amean', 'logRelF0-H1-H2_sma3nz_stddevNorm', 'logRelF0-H1-A3_sma3nz_amean', 'logRelF0-H1-A3_sma3nz_stddevNorm', 'F1frequency_sma3nz_amean', 'F1frequency_sma3nz_stddevNorm', 'F1bandwidth_sma3nz_amean', 'F1bandwidth_sma3nz_stddevNorm', 'F1amplitudeLogRelF0_sma3nz_amean', 'F1amplitudeLogRelF0_sma3nz_stddevNorm', 'F2frequency_sma3nz_amean', 'F2frequency_sma3nz_stddevNorm', 'F2bandwidth_sma3nz_amean', 'F2bandwidth_sma3nz_stddevNorm', 'F2amplitudeLogRelF0_sma3nz_amean', 'F2amplitudeLogRelF0_sma3nz_stddevNorm', 'F3frequency_sma3nz_amean', 'F3frequency_sma3nz_stddevNorm', 'F3bandwidth_sma3nz_amean', 'F3bandwidth_sma3nz_stddevNorm', 'F3amplitudeLogRelF0_sma3nz_amean', 'F3amplitudeLogRelF0_sma3nz_stddevNorm', 'alphaRatioV_sma3nz_amean', 'alphaRatioV_sma3nz_stddevNorm', 'hammarbergIndexV_sma3nz_amean', 'hammarbergIndexV_sma3nz_stddevNorm', 'slopeV0-500_sma3nz_amean', 'slopeV0-500_sma3nz_stddevNorm', 'slopeV500-1500_sma3nz_amean', 'slopeV500-1500_sma3nz_stddevNorm', 'spectralFluxV_sma3nz_amean', 'spectralFluxV_sma3nz_stddevNorm', 'mfcc1V_sma3nz_amean', 'mfcc1V_sma3nz_stddevNorm', 'mfcc2V_sma3nz_amean', 'mfcc2V_sma3nz_stddevNorm', 'mfcc3V_sma3nz_amean', 'mfcc3V_sma3nz_stddevNorm', 'mfcc4V_sma3nz_amean', 'mfcc4V_sma3nz_stddevNorm', 'alphaRatioUV_sma3nz_amean', 'hammarbergIndexUV_sma3nz_amean', 'slopeUV0-500_sma3nz_amean', 'slopeUV500-1500_sma3nz_amean', 'spectralFluxUV_sma3nz_amean', 'loudnessPeaksPerSec', 'VoicedSegmentsPerSec', 'MeanVoicedSegmentLengthSec', 'StddevVoicedSegmentLengthSec', 'MeanUnvoicedSegmentLength', 'StddevUnvoicedSegmentLength', 'equivalentSoundLevel_dBp']

    # load the data
    df1_train = concatenate_label_data(
    folder_train_em, folder_train_egemaps
    )
    df1_train = df1_train.sort_values(by="audio").reset_index(drop=True)
    features_df = pd.DataFrame(
    df1_train["embedding"].tolist(),
    index=df1_train.index).add_prefix("feat_")

    feat_names = features_df.columns.to_list()
    df1_train = df1_train.drop(columns=["embedding"])
    df1_train = pd.concat([df1_train, features_df], axis=1)

    df2_train = pd.read_csv(folder_train_egemaps)
    df2_train["audio"] = df2_train["audio"].apply(lambda x: x.replace(".wav", ""))
    df_train = pd.merge(df1_train, df2_train, on="audio")
    
    x_train = df_train.filter(items=cols_gmaps + feat_names)
    y_train = df_train["label_x"]


    df1_eval = concatenate_label_data(
    folder_eval_em, folder_eval_egemaps
    )
    df1_eval["audio"] = df1_eval["audio"].apply(lambda x: x.replace(".wav", ""))
    features_df = pd.DataFrame(
    df1_eval["embedding"].tolist(),
    index=df1_eval.index).add_prefix("feat_")

    feat_names = features_df.columns.to_list()
    df1_eval = df1_eval.drop(columns=["embedding"])
    df1_eval = pd.concat([df1_eval, features_df], axis=1)

    df2_eval = pd.read_csv(folder_eval_egemaps)
    df2_eval["audio"] = df2_eval["audio"].apply(lambda x: x.replace(".wav", ""))

    df_eval = pd.merge(df1_eval, df2_eval, on="audio")

    x_eval = df_eval.filter(items=cols_gmaps + feat_names)
    y_eval = df_eval["label_x"]

    # definimos los calsificadores
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

    # evaluate the model
    y_scores = sclf.decision_function(x_eval)
    print(y_scores)
    y_pred = [0 if x < 0.5 else 1 for x in y_scores]

    return y_pred, y_scores, y_eval