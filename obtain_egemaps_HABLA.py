import egemaps
import pandas as pd
import numpy as np

folder = "/home/juanjo/Documentos/eGeMAPS_embedding/Latin_America_Spanish_anti_spoofing_dataset/FinalDataset_16khz/"

df = egemaps.egemaps_from_tree_folder(folder=folder)

print(df.columns.to_list())

# save the data
df.to_csv("egemaps_HABLA.csv")

