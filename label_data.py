import os
import numpy as np
import pandas as pd

# aqui vamos a etiquetar los datos
df_train = pd.read_csv("egemaps_LA.csv")
df_eval = pd.read_csv("egemaps_LA_eval.csv")

# le quitamos el .flac a los nombres de los audios
df_train["audio"] = df_train["audio"].apply(lambda x: x.replace(".flac", ""))
df_eval["audio"] = df_eval["audio"].apply(lambda x: x.replace(".flac", ""))

# etiquetamos ambos
labels_train = pd.read_csv("/home/juanjo/Documentos/eGeMAPS_embedding/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt", 
                     sep=' ', header=None, names=["h1", "audio", "h2", "h3", "label"])
labels_train = labels_train.filter(items=["audio", "label"])
labels_eval = pd.read_csv("/home/juanjo/Documentos/eGeMAPS_embedding/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt", 
                     sep=' ', header=None, names=["h1", "audio", "h2", "h3", "label"])
labels_eval = labels_eval.filter(items=["audio", "label"])

# hacemos merge
df_train = pd.merge(df_train, labels_train, on="audio")
df_eval = pd.merge(df_eval, labels_eval, on="audio")

# cambiamos las etiquetas
df_train["label"] = df_train["label"].apply(lambda x: 1 if x == "bonafide" else 0 if x == "spoof" else x)
df_eval["label"] = df_eval["label"].apply(lambda x: 1 if x == "bonafide" else 0 if x == "spoof" else x)

# guardamos los csv
df_train.to_csv("egemaps_LA_train_labeled.csv")
df_eval.to_csv("egemaps_LA_eval_labeled.csv")