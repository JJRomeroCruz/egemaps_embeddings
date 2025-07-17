
from datasets import load_dataset, DownloadConfig

dataset = load_dataset("mueller91/ReplayDF", num_proc=1)


print(len(dataset))