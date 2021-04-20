# Code for Predicting special care during the COVID-19 pandemic: A machine learning approach #
# by Vitor Bezzan (vitor@bezzan.com) and Cleber Rocco (cdrocco@unicamp.br)                   #
#                                                                                            #
# São Paulo, 2020                                                                            #

import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import shap
import seaborn as sns

from scikitplot.metrics import plot_roc, plot_precision_recall_curve

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline as skPipeline

from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTEENN

data = pd.read_parquet("./data/primary/hsl.parquet")
train, test = train_test_split(data, test_size=0.15, random_state=123)

X_train, y_train = (
    train.drop(["target_internacao", "TI"], axis=1),
    train["target_internacao"],
)
X_test, y_test = (
    test.drop(["target_internacao", "TI"], axis=1),
    test["target_internacao"],
)

steps = [
    ("input_values", SimpleImputer(strategy="median")),
    ("balance_targets", SMOTEENN(random_state=123)),
    (
        "model_xgb",
        xgb.XGBClassifier(
            objective="binary:logistic",
        ),
    ),
]

grid = {
    "model_xgb__n_estimators": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250],
    "model_xgb__max_depth": [3, 4, 5, 6],
}

pipeline = Pipeline(steps)
model = GridSearchCV(pipeline, grid, cv=5, n_jobs=-1, verbose=2).fit(X_train, y_train)

new_steps = [
    ("input_values", model.best_estimator_[0]),
    ("model_xgb", model.best_estimator_[-1]),
]

final_model = skPipeline(new_steps)
roc = roc_auc_score(y_test, final_model.predict_proba(X_test)[:, 1])

sns.set_style("ticks")
sns.set(font="Arial")
plot_roc(y_test, model.predict_proba(X_test))
plt.savefig("./results/xgb_results/roc_plot")
plt.close()

sns.set_style("ticks")
sns.set(font="Arial")
plot_precision_recall_curve(y_test, model.predict_proba(X_test))
plt.savefig("./results/xgb_results/precision_recall")
plt.close()

explainer = shap.TreeExplainer(model.best_estimator_[-1], X_train, link="logit")
shap_values = explainer.shap_values(X_test, check_additivity=False)

explanations = pd.DataFrame(shap_values, columns=X_test.columns, index=X_test.index)

results = pd.DataFrame()
results["# Estimators"] = model.cv_results_["param_model_xgb__n_estimators"].data
results["Max Depth"] = model.cv_results_["param_model_xgb__max_depth"].data
results["Model Score"] = model.cv_results_["mean_test_score"].data
results["Model Score (std)"] = model.cv_results_["std_test_score"].data

sns.set_style("ticks")
sns.set(font="Arial")
ax = sns.lineplot(
    x="# Estimators", y="Model Score", hue="Max Depth", markers=True, data=results
)
plt.xlabel("Number of Estimators")
plt.ylabel("Validation Score (5 fold-validation)")
plt.title("Validation Scores (Grid-search Hyperparameters)")
plt.savefig("./results/xgb_results/training_plot")
plt.close()

sns.set_style("ticks")
sns.set(font="Arial")
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.savefig("./results/xgb_results/importance_plot")
plt.close()
