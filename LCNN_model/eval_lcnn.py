# eval_lcnn_tf.py
import tensorflow as tf
import numpy as np
from lcnn_tf import build_lcnn

def load_data(hdf5_path):
    # Devuelve tuplas (inputs, labels) numpy
    pass

def evaluate():
    model = tf.keras.models.load_model('lcnn_tf_model.h5', compile=False)
    X, y = load_data("test_mels.h5")
    preds, _ = model.predict(X, batch_size=64)
    y_hat = np.argmax(preds, axis=1)

    acc = np.mean(y_hat == y)
    print(f"Test accuracy: {acc:.4f}")

if __name__ == '__main__':
    evaluate()
