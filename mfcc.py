import numpy as np
import pandas as pd
import os

import torchaudio
import torch
from glob import glob
from tqdm import tqdm

class PreEmphasis(torch.nn.Module):
    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input.unsqueeze(1)
        input = torch.nn.functional.pad(input, (1, 0), 'reflect')
        return torch.nn.functional.conv1d(input, self.flipped_filter).squeeze(1)

# Configuración del transformador MelSpectrogram
sample_rate = 16000
n_mels = 80

torchfbank = torch.nn.Sequential(
    PreEmphasis(),
    torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=512,
        win_length=400,
        hop_length=160,
        f_min=20,
        f_max=7600,
        window_fn=torch.hamming_window,
        n_mels=n_mels
    )
)

# Rutas
input_dir = "/home/juanjo/Documentos/eGeMAPS_embedding/MLAAD/fake"
output_dir = "/home/juanjo/Documentos/eGeMAPS_embedding/mel_MLAAD"
os.makedirs(output_dir, exist_ok=True)

"""
# Procesamiento por lote
audio_paths = glob(os.path.join(input_dir, "*.wav"))

contador = 0
for path in tqdm(audio_paths):
    name = os.path.basename(path).replace('.wav', '.npy')
    waveform, sr = torchaudio.load(path)
    print("[", contador, "]", " ", path)

    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)

    with torch.no_grad():
        x = waveform[0].unsqueeze(0)  # [1, T]
        mel = torchfbank(x) + 1e-6
        mel = mel.log()
        mel = mel - mel.mean(dim=-1, keepdim=True)
        mel = mel.squeeze(0).cpu().numpy()  # [n_mels, time]

    np.save(os.path.join(output_dir, name), mel)
"""

batch_size = 1000
datos_batch = []
cols = ["audio", "mfcc"]
contador = 0
lote = 174

# --- PROCESAMOS ---
for root, dirs, files in os.walk(input_dir, topdown=False):
    for file in files:
        if file.endswith('.wav'):
            if contador > int(lote*1000):
                full_path = os.path.join(root, file)
                print(f"[{contador}] Procesando: {file}")

                try:
                    # cargamos audio
                    waveform, sr = torchaudio.load(full_path)

                    # le cambiamos la sample rate
                    if sr != sample_rate:
                        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)

                    # le sacamos el mel
                    with torch.no_grad():
                        x = waveform[0].unsqueeze(0)  # [1, T]
                        mel = torchfbank(x) + 1e-6
                        mel = mel.log()
                        mel = mel - mel.mean(dim=-1, keepdim=True)
                        mel = mel.squeeze(0).cpu().numpy()  # [n_mels, time]

                    # guardamos en el batch
                    datos_batch.append([file, mel])
                    contador += 1
                except RuntimeError as e:
                    print(f"Error procesando {file}:{e}")
                    continue

                if contador % batch_size == 0:
                    df = pd.DataFrame(datos_batch, columns=cols)
                    df.to_csv(f"{output_dir}_batch{lote}.csv", index=False)
                    print(f" Guardado: {output_dir}_batch{lote}.csv con {len(df)} filas")
                    datos_batch = []
                    lote += 1
            else:
                contador +=1
                print(contador)

# --- GUARDAR EL RESTO ---
if datos_batch:
    df = pd.DataFrame(datos_batch, columns=cols)
    df.to_csv(f"{output_dir}_batch{lote}.csv", index=False)
    print(f"Guardado final: {output_dir}_batch{lote}.csv con {len(df)} filas")