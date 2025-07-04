import numpy as np
import pandas as pd
from use_wav2vec2 import get_embedding

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# recorremos la carpeta de los audios para sacarsus embedding
emb = get_embedding("/home/juanjo/Documentos/eGeMAPS_embedding/ASVspoof2019_LA_train/flac/LA_T_1000137.flac")

print(emb)