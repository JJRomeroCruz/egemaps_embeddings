import egemaps
import pandas as pd
import numpy as np

folder = "/home/juanjo/Documentos/eGeMAPS_embedding/MLAAD/fake/"

df = egemaps.egemaps_from_tree_folder(folder=folder)

print(df.columns.to_list())

# save the data
df.to_csv("egemaps_MLAAD.csv")

# aqui todos son fake