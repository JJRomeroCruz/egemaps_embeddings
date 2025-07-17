import tensorflow as tf
import numpy as np
from LCNN import build_lcnn
import pandas as pd
from sklearn.utils import shuffle as sk_shuffle
import os


def parse_hdf5(dataset_path):
    # Implementar lógica de map para datasets TF desde h5py
    pass

def data_generator(hdf5_path, batch_size=32, shuffle=True):
    # Crea tf.data.Dataset que carga mel‑spectrograms y labels
    pass

def load_data(path, label_csv, batch_size=32, shuffle=True, val_split=0.2):
    # load the training data (asvspoof2021)
    folder = path
    list_df = []
    for file in os.listdir(folder):
        df = pd.read_csv(os.path.join(folder, file))
        list_df.append(df)
    df_train = pd.concat(list_df, ignore_index=True)

    # label the data
    labels = pd.read_csv(label_csv)
    df_train = pd.merge(df_train, labels.filter(items=["audio", "label"]), on="audio")

    # Asegurar que los MELS están como arrays y no como strings
    if isinstance(df['mel'].iloc[0], str):
        df['mel'] = df['mel'].apply(lambda x: np.array(eval(x), dtype=np.float32))

    # x, y = df_train['mel'], df_train['label']
    x = np.stack(df_train['mel'].values)
    y = df_train['label'].values

    # separar entre entrenamiento y validacion
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=val_split, stratify=y, random_state=42)

    if shuffle:
        x_train, y_train = sk_shuffle(x_train, y_train, random_state=42)
        x_val, y_val = sk_shuffle(x_val, y_val, random_state=42)
    
    # Crear datasets de tendroflow haciendo batching
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds

def train():
    model = build_lcnn(num_classes=2)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    train_ds, val_ds = load_data(
        path="/home/juanjo/Documentos/eGeMAPS_embedding/mel_LA_train",
        label_csv= "/home/juanjo/Documentos/eGeMAPS_embedding/egemaps_LA_train_labeled.csv",
        batch_size=64,
        shuffle=True, 
        val_split=0.2
    )

    model.fit(train_ds, validation_data=val_ds, epochs=20)
    model.save('lcnn_model.h5')

if __name__ == '__main__':
    train()