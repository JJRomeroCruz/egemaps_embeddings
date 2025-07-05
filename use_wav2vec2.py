import numpy as np
import os
import torch
import torchaudio
import librosa
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# load the model and the processor, direccion del hash
#processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
#model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
path = "/home/juanjo/Documentos/eGeMAPS_embedding/wav2vec2/models--facebook--wav2vec2-base-960h/snapshots/22aad52d435eb6dbaf354bdad9b0da84ce7d6156"

processor = Wav2Vec2Processor.from_pretrained(path)
model = Wav2Vec2Model.from_pretrained(path)
model.eval()

# definimos una funcion para sacar los embeddings de cada audio
def get_embedding(audio_path):
    
    # caragamos el audio
    #signal, sr = torchaudio.load(audio_path)
    signal, sr = librosa.load(audio_path, sr=16000)
    signal = torch.tensor(signal)
    
    # queremos que la sample rate sea de 16 KHz
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        signal = resampler(signal)
    
    # definimos los inputs que le vamos a pasar 
    inputs = processor(signal.squeeze(), sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        hidden_states = outputs.last_hidden_state # (batch, time, feature_dim)
        mean_embedding = hidden_states.mean(dim=1).squeeze().numpy() # vector 
    return mean_embedding

