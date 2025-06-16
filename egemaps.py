import numpy as np
import pandas as pd
import os
import opensmile
import soundfile as sf
import parselmouth as p
import matplotlib.pyplot as plt

# requiere que en la carpeta solo hayan audios
def egemaps_from_folder(folder):
    datos = []
    for file in os.listdir(folder):
        # cargamos el audio
        row = [file]
        extractor = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv01a,
            feature_level = opensmile.FeatureLevel.Functionals
        )
        signal, sr = sf.read(os.path.join(folder, file))
        features = extractor.process(signal, sr)
        for x in features[2].tolist()[0]:
            row.append(x)
        datos.append(row)
    cols = ["audio"]
    for x in extractor.feature_names:
        cols.append(x)
    df = pd.DataFrame(datos, columns=cols)

    return df

