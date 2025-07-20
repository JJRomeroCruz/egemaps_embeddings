import numpy as np
import pandas as pd
from use_wav2vec2 import get_embedding
import os

# --- CONFIGURACIÓN ---
#folder = "/home/juanjo/Documentos/eGeMAPS_embedding/ASVspoof2019_LA_train/flac"
#folder = "/home/juanjo/Documentos/eGeMAPS_embedding/ASVspoof2019_LA_eval/flac"
folder = "/home/juanjo/Documentos/eGeMAPS_embedding/in_the_wild/release_in_the_wild"
#output_prefix = "embeddings_LA_train"
output_prefix = "embeddings_wild"
batch_size = 1000

# --- VARIABLES DE TRABAJO ---
datos_batch = []
cols = ["audio", "embedding"]
lote = 22
contador = 0
# --- PROCESAMIENTO ---
for file in os.listdir(folder):
    if file.endswith(".wav"):
        if contador >= int(lote*1000):
            full_path = os.path.join(folder, file)
            print(f"[{contador}] Procesando: {file}")

            emb = get_embedding(full_path)
            if emb is None:
                continue  # saltar si hubo error con el audio

            datos_batch.append([file, emb])
            contador += 1

            # guardar cada N audios
            if contador % batch_size == 0:
                df = pd.DataFrame(datos_batch, columns=cols)
                df.to_csv(f"{output_prefix}_batch{lote}.csv", index=False)
                print(f" Guardado: {output_prefix}_batch{lote}.csv con {len(df)} filas")
                datos_batch = []
                lote += 1
        else:
            print(contador)
            contador += 1
            

# --- GUARDAR RESTO ---
if datos_batch:
    df = pd.DataFrame(datos_batch, columns=cols)
    df.to_csv(f"{output_prefix}_batch{lote}.csv", index=False)
    print(f"Guardado final: {output_prefix}_batch{lote}.csv con {len(df)} filas")
