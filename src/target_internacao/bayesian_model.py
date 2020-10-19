# Code for Predicting special care during the COVID-19 pandemic: A machine learning approach #
# by Vitor Bezzan (vitor@bezzan.com) and Cleber Rocco (cdrocco@unicamp.br)                   #
#                                                                                            #
# São Paulo, 2020                                                                            #

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shap

from scikitplot.metrics import plot_roc, plot_precision_recall_curve

from src.bayesian_classifier import BayesianxgClassifier

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTEENN

from sklearn.calibration import calibration_curve

from scipy import stats

data = pd.read_parquet("./data/primary/hsl.parquet")
train, test = train_test_split(data, test_size=0.15, random_state=123)

X_train, y_train = train.drop(["target_internacao", "TI"], axis=1), train["target_internacao"]
X_test, y_test = test.drop(["target_internacao", "TI"], axis=1), test["target_internacao"]

steps = [
    ("input_values", SimpleImputer(strategy="median")),
    ("balance_targets", SMOTEENN(random_state=123)),
    ("model_xgb", BayesianxgClassifier(CV=4))
]

pipeline = Pipeline(steps)
model = pipeline.fit(X_train, y_train)

roc = roc_auc_score(y_test, model.predict(X_test)[:,1])

sns.set_style("ticks")
sns.set(font="Arial")
plot_roc(y_test, model.predict(X_test))
plt.savefig("./results/xgb_results/target_internacao/roc_plot_bayes_TI.png")
plt.close()

sns.set_style("ticks")
sns.set(font="Arial")
plot_precision_recall_curve(y_test, model.predict(X_test))
plt.savefig("./results/xgb_results/target_internacao/precision_recall_bayes_TI.png")
plt.close()

explainer = shap.KernelExplainer(pipeline.predict, shap.sample(X_train, 2000), link="logit")
shap_values = explainer.shap_values(shap.sample(X_test, 100), nsamples=100)

sns.set_style("ticks")
sns.set(font="Arial")
fpos, mean = calibration_curve(y_train, model.predict(X_train)[:, 1], n_bins=5)
plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
plt.plot(mean, fpos, label="Model Calibration")
plt.legend()
plt.savefig("./results/xgb_results/target_internacao/calibration.png")
plt.close()