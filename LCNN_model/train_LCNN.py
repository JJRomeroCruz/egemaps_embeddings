import tensorflow as tf
import numpy as np
from lcnn_tf import build_lcnn

def parse_hdf5(dataset_path):
    # Implementar lógica de map para datasets TF desde h5py
    pass

def data_generator(hdf5_path, batch_size=32, shuffle=True):
    # Crea tf.data.Dataset que carga mel‑spectrograms y labels
    pass

def train():
    model = build_lcnn(num_classes=2)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    train_ds = data_generator("train_mels.h5", 64)
    val_ds = data_generator("val_mels.h5", 64, shuffle=False)

    model.fit(train_ds, validation_data=val_ds, epochs=20)
    model.save('lcnn_tf_model.h5')

if __name__ == '__main__':
    train()