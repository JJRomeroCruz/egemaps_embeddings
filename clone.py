import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
import os

model = ChatterboxTTS.from_pretrained(device = "cpu")
text = "We were away a year ago"

folder = "/home/juanjo/Documentos/eGeMAPS_embedding/Audio Files"
output_folder = "/home/juanjo/Documentos/eGeMAPS_embedding/Audio Files Cloned"
os.makedirs(output_folder, exist_ok=True)

for file in os.listdir(folder):
    wav = model.generate(text, audio_prompt_path = os.path.join(folder, file))
    ta.save(os.path.join(folder, file), wav, model.sr)