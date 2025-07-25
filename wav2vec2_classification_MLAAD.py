import numpy as np
import pandas as pd
from use_wav2vec2 import get_embedding
import os

# --- CONFIGURACIÓN ---
#folder = "/home/juanjo/Documentos/eGeMAPS_embedding/ASVspoof2019_LA_train/flac"
folder = "/home/juanjo/Documentos/eGeMAPS_embedding/MLAAD/fake"
output_prefix = "embeddings_MLAAD"
batch_size = 1000

# --- VARIABLES DE TRABAJO ---
datos_batch = []
cols = ["audio", "embedding"]
contador = 0
lote = 7

# --- PROCESAMIENTO ---
for root, dirs, files in os.walk(folder, topdown=False):
    for file in files:
        if file.endswith('.wav'):
            if contador > 7000:
                full_path = os.path.join(root, file)
                print(f"[{contador}] Procesando: {file}")
                
                try:

                    emb = get_embedding(full_path)
                #if emb is None:
                #    continue # salta si ha habido error ocn el audio

                    datos_batch.append([file, emb])
                    contador += 1
                except RuntimeError as e:
                    print(f"Error procesando {file}:{e}")
                    continue

                # guardamos cada N audios
                if contador % batch_size == 0:
                    df = pd.DataFrame(datos_batch, columns=cols)
                    df.to_csv(f"{output_prefix}_batch{lote}.csv", index=False)
                    print(f" Guardado: {output_prefix}_batch{lote}.csv con {len(df)} filas")
                    datos_batch = []
                    lote += 1
            else:
                contador += 1
                print(contador)

# --- GUARDAR RESTO ---
if datos_batch:
    df = pd.DataFrame(datos_batch, columns=cols)
    df.to_csv(f"{output_prefix}_batch{lote}.csv", index=False)
    print(f"Guardado final: {output_prefix}_batch{lote}.csv con {len(df)} filas")