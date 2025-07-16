import os
import pandas as pd
import numpy as np

# load the MLAAD egemaps data
df = pd.read_csv("/home/juanjo/Documentos/eGeMAPS_embedding/egemaps_MLAAD.csv")

# generate the label
df["label"] = df.apply(lambda x: 0, axis= 1)

# save the data
df.to_csv("egemaps_MLAAD_labeled.csv")