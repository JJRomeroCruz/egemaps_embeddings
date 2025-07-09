from datasets import load_dataset, DownloadConfig

dataset = load_dataset("jungjee/asvspoof5", num_proc=1)


print(len(dataset))