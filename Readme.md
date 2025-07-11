This experiment is inspired in [this paper][https://arxiv.org/abs/2408.15775]
# Install the repository

1. Install it by writing in the bash: 

```
git clone https://github.com/JJRomeroCruz/egemaps_embeddings.git 
```

2. Then create a new enviroment in python 3.11 version

```
conda create -n egemaps python=3.11
conda activate egemaps
```

3. Install the requirements

```
pip install -r requirements.txt
```

# Use

1. Download the self-supervised model wav2vec2-base

```
python download_wav2vec2.py
```

2. Download the dataset, which is the [ASVspoof 5 dataset][https://huggingface.co/datasets/jungjee/asvspoof5], [HABLA][https://zenodo.org/records/7370805] and [MLAAD][https://huggingface.co/datasets/mueller91/MLAAD].

```
python download_asvspoof5.py
```

3. Obtain the eGeMAPS and the Wav2vec2 representations. 

```
python wav2vec2_calssification_HABLA.py
python wav2vec2_classification_LA_train.py
python wav2vec2_classification_LA_eval.py
```

4. Run the classifiers scripts

```

```

# License

This project is licensed under the **GNU General Public License v3.0** – see the [LICENSE](./LICENSE) file for details.

# Contact

Created by Juan José Romero - juanjorcr98@gmail.com

[Linkedin][https://www.linkedin.com/in/juan-jos%C3%A9-romero-cruz-ba1289191/]

