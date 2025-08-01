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
    contador = 0
    for file in os.listdir(folder):
        # cargamos el audio
        print(contador)
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
        contador += 1
        print(contador)
    cols = ["audio"]
    for x in extractor.feature_names:
        cols.append(x)
    df = pd.DataFrame(datos, columns=cols)

    return df

def egemaps_from_tree_folder(folder):
    datos = []
    contador = 0
    for root, dirs, files in os.walk(folder, topdown = False):
        for file in files:
            if file.endswith('.wav'):
                # cargamos el audio
                print(contador)
                row = [file]
                
                extractor = opensmile.Smile(
                    opensmile.FeatureSet.eGeMAPSv01a,
                    feature_level = opensmile.FeatureLevel.Functionals
                )
                signal, sr = sf.read(os.path.join(root, file))
                features = extractor.process(signal, sr)
                for x in features[2].tolist()[0]:
                    row.append(x)
                datos.append(row)
                contador += 1
    cols = ["audio"]
    for x in extractor.feature_names:
        cols.append(x)
    df = pd.DataFrame(datos, columns=cols)

    return df