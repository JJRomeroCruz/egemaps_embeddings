import egemaps
#import use_wav2vec2
import pandas as pd
import numpy as np

# sacamos los egemaps
#folder_train = "/home/juanjo/Documentos/eGeMAPS_embedding/ASVspoof2019_LA_train/flac"
folder_eval = "/home/juanjo/Documentos/eGeMAPS_embedding/ASVspoof2019_LA_eval/flac"

df = egemaps.egemaps_from_folder(folder_eval)

print(df.columns.to_list())

# save the data
df.to_csv("egemaps_LA_eval.csv")