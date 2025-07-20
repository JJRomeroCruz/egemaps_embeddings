import time
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import shap


# load the data
X_train, X_test, Y_train, Y_test = train_test_split(*shap.datasets.iris(), test_size=0.2, random_state=0)

# rather than use the whole training set to estimate expected values, we could summarize with
# a set of weighted kmeans, each weighted by the number of points they represent. But this dataset
# is so small we don't worry about it
# X_train_summary = shap.kmeans(X_train, 50)


def print_accuracy(f):
    print(f"Accuracy = {100 * np.sum(f(X_test) == Y_test) / len(Y_test)}%")
    time.sleep(0.5)  # to let the print get out before any progress bars

shap.initjs()

# define SVM with a linear kernel
svc_linear = sklearn.svm.SVC(kernel="linear", probability=True)
svc_linear.fit(X_train, Y_train)
print_accuracy(svc_linear.predict)

# explain all the predictions in the test set
explainer = shap.KernelExplainer(svc_linear.predict_proba, X_train)
shap_values = explainer.shap_values(X_test)

# this is multiclass so we only visualize the contributions to first class (hence index 0)
force_plot = shap.force_plot(explainer.expected_value[0], shap_values[..., 0], X_test)

# guardar en un archivo html
shap.save_html("force_plot.html", force_plot)