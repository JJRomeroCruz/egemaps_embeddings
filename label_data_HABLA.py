import numpy as np
import pandas as pd

# generate the dataframe of labels
cols = ["h1", "audio", "h3", "sw", "label"]
df_labels = pd.read_csv("/home/juanjo/Documentos/eGeMAPS_embedding/Latin_America_Spanish_anti_spoofing_dataset/protocol.txt", 
                        sep=' ', header=None, names= cols)

# merge the two both datasets
df = pd.read_csv("egemaps_HABLA.csv")
df["audio"] = df["audio"].apply(lambda x: x.replace(".wav", ""))
df = pd.merge(df, df_labels, on="audio")
df["label"] = df["label"].apply(lambda x: 1 if x == "bonafide" else 0 if x == "spoof" else x)
print(sum(df["label"]))
# save the data
df.to_csv("egemaps_HABLA_labeled.csv")