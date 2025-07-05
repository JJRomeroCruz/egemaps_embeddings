import numpy as np
import pandas as pd
from use_wav2vec2 import get_embedding
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# vamos a empezar usando el modelo solamente preentrenado, sin entrenarlo para la tarea

# recorremos la carpeta de los audios para sacarsus embedding
folder = "/home/juanjo/Documentos/eGeMAPS_embedding/ASVspoof2019_LA_eval/flac"
datos = []
cols = ["audio", "embedding"]
contador = 0
for file in os.listdir(folder):
    print(contador)
    row = [file]
    emb = get_embedding(os.path.join(folder, file))
    row.append(emb)
    datos.append(row)
    contador += 1

df = pd.DataFrame(datos, columns=cols)

# label the data
df_label = pd.read_csv("/home/juanjo/Documentos/eGeMAPS_embedding/egemaps_LA_eval_labeled.csv")
df = pd.merge(df, df_label.filter(items=["audios", "label"], axis=1), on="audios")

df.to_csv("embeddings_LA_eval.csv")


#emb = get_embedding("/home/juanjo/Documentos/eGeMAPS_embedding/ASVspoof2019_LA_train/flac/LA_T_1000137.flac")

# vamos 
#print(emb)