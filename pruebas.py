import egemaps
#import use_wav2vec2
import pandas as pd
import numpy as np

# sacamos los egemaps
folder = "/home/juanjo/Documentos/eGeMAPS_embedding/ASVspoof2019_LA_train/flac"

df = egemaps.egemaps_from_folder(folder)

print(df.columns.to_list())