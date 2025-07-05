import numpy as np
import pandas as pd
import os

# load the data
df = pd.read_csv("egemaps_MLAAD.csv")

# generate the label
df["label"] = df["label"].apply(lambda x: "spoof")

# save the data
df.to_csv("egemaps_MLAAD.csv")