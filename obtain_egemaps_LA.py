import egemaps
#import use_wav2vec2
import pandas as pd
import numpy as np

# sacamos los egemaps
#folder_train = "/home/juanjo/Documentos/eGeMAPS_embedding/ASVspoof2019_LA_train/flac"
#folder_eval = "/home/juanjo/Documentos/eGeMAPS_embedding/ASVspoof2019_LA_eval/flac"
folder = "/home/juanjo/Documentos/eGeMAPS_embedding/in_the_wild/release_in_the_wild"

df = egemaps.egemaps_from_folder(folder)

print(df.columns.to_list())

# save the data
#df.to_csv("egemaps_LA_eval.csv")
df.to_csv("egemaps_wild_egemaps.csv")