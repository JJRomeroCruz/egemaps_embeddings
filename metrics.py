import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

def plot_roc_curve(y_true, y_scores):
    """
    funcion que me da la curva ROC, así como la AUC
    reutrns AUC, array con fpr, tpr y las thresholds
    y_true: las etiquetas verdaderas
    y_score: los scores de las prediciones que da el modelo
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize = (8, 6))
    plt.plot(fpr, tpr, color='blue', lw = 2, label = f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', lw=1, linestyle='--')
    plt.xlabel("Tasa de falsos positivos (FPR)")
    plt.ylabel("Tasa de verdaderos positivos (TPR)")
    plt.title("Curva ROC")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
    plt.savefig("ROC_curve", format="png")
    return roc_auc, [fpr, tpr, thresholds]

def compute_eer(y_true, y_scores):
    """
    funcion que me calcula la eer
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr

    # tenemos que encontrar un punto donde la fpr y la fnr sean iguales
    eer_threshold_index = np.nanargmin(np.abs(fnr-fpr))
    eer = (fpr[eer_threshold_index] + fnr[eer_threshold_index])/2.0
    eer_threshold = thresholds[eer_threshold_index]

    return eer, eer_threshold

def evaluate_model(y_true, y_pred):
    """
    a aprtir de las etiquetas reales y las predicciones del modelo, nos da la matriz de confusion y 
    la accuracy
    """
    # primero de todo la accuracy
    acc = accuracy_score(y_true, y_pred)

    # matriz de confusion
    m = confusion_matrix(y_true, y_pred)

    # de paso, ploteamos la matriz de confusion
    plt.figure(figsize=(6, 5))
    sns.heatmap(m, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Matriz de confusion")
    plt.xlabel("Prediccion")
    plt.ylabel("Ground truth")
    plt.show()

    return acc, m