from transformers import Wav2Vec2Processor, Wav2Vec2Model

# descargamos y guardamos el modelo wav2vec2
folder = "/home/juanjo/Documentos/eGeMAPS_embedding/wav2vec2"

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h", cache_dir=folder)
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h", cache_dir=folder)

