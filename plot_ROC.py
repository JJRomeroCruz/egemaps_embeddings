import matplotlib.pyplot as plt
import models
import seaborn as sns
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix
import metrics

def plot_roc_models(eval, train=["egemaps_LA_train_labeled.csv", "embeddings_LA_train"]):
    
    # sacamos los y_pred, y_scores, y_eval de cada modelo
    res = [models.egemaps_svm(folder_eval=eval[0]), 
           models.wav2vec2_svm(eval=eval), 
           models.concatenation_svm(eval=eval), models.late_fusion(eval=eval),
           models.stacking(eval=eval)]
    
    titulos = ["eGeMAPS", "wav2vec2", "Early Fusion", "Late Fusion", "Stacking"]

    plt.figure(figsize=(16, 12))
    for i in range(len(res)):
        y_true, y_scores = res[i][2], res[i][1]
        fpr, tpr, thres = roc_curve(y_true, y_scores)
        plt.plot(fpr, tpr, label=titulos[i])
    
    plt.plot([0, 1], [0, 1], color='red', lw=1, linestyle="--")
    plt.xlabel("Tasa de falsos positivos (FPR)")
    plt.ylabel("Tasa de verdaderos positivos (TPR)")
    plt.title("Curva ROC HABLA")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
    
    return

#plot_roc_models(eval=["egemaps_LA_eval_labeled.csv", "embeddings_LA_eval"])
#plot_roc_models(eval=["/home/juanjo/Documentos/eGeMAPS_embedding/egemaps_wild_labeled.csv", "/home/juanjo/Documentos/eGeMAPS_embedding/embeddings_WILD"])
plot_roc_models(eval=["/home/juanjo/Documentos/eGeMAPS_embedding/egemaps_HABLA_labeled.csv", "/home/juanjo/Documentos/eGeMAPS_embedding/embeddings_HABLA"])