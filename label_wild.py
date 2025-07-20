import os
import numpy as np
import pandas as pd

# etiquetamos los datos
df = pd.read_csv("/home/juanjo/Documentos/eGeMAPS_embedding/egemaps_wild_egemaps.csv")

# etiquetas
labels = pd.read_csv("/home/juanjo/Documentos/eGeMAPS_embedding/in_the_wild/meta.csv")
labels = labels.rename(columns={"file":"audio"})
labels = labels.filter(items=["audio", "label"])

# mergeamos
df = pd.merge(df, labels, on="audio")

# cambiammos las etiquetas
df["label"] = df["label"].apply(lambda x: 1 if x == "bona-fide" else 0 if x == "spoof" else x)
df = df.drop(columns = ["Unnamed: 0"])
print(df.columns.tolist())
# guardamos los csv
df.to_csv("egemaps_wild_labeled.csv")