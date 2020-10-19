# Code for Predicting special care during the COVID-19 pandemic: A machine learning approach #
# by Vitor Bezzan (vitor@bezzan.com) and Cleber Rocco (cdrocco@unicamp.br)                   #
#                                                                                            #
# São Paulo, 2020                                                                            #

import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import shap
import math

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.dummy import DummyRegressor

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTEENN

from scipy.stats import spearmanr, kendalltau


from fancyimpute import SoftImpute

class SoftImputeDf:
    def fit(self, X, y):
        return self
    def transform(self, X):
        return SoftImpute().fit_transform(X)

data = pd.read_parquet("./data/primary/hsl.parquet")
train, test = train_test_split(data, test_size=0.15, random_state=123)

train = train[pd.notna(train["TI"])]
test = test[pd.notna(test["TI"])]

treat_anomalous_train = train[(train["TI"] > 0) & (train["target_internacao"] == 0)].index
treat_anomalous_test = test[(test["TI"] > 0) & (test["target_internacao"] == 0)].index

train.loc[treat_anomalous_train, "TI"] = 0
test.loc[treat_anomalous_test, "TI"] = 0

X_train, y_train = train.drop(["target_internacao", "TI"], axis=1), train["TI"]
X_test, y_test = test.drop(["target_internacao", "TI"], axis=1), test["TI"]

steps = [
    ("input_values", SoftImputeDf()),
    ("model_xgb", xgb.XGBRegressor())
]

grid = {"model_xgb__n_estimators": [30, 40, 50, 100], "model_xgb__max_depth": [3, 4, 5, 6]}

pipeline = Pipeline(steps)
model = GridSearchCV(pipeline, grid, cv=5, n_jobs=-1, verbose=2).fit(X_train, y_train)
dummy = DummyRegressor().fit(X_train, y_train)

mse = mean_squared_error(y_test, model.predict(X_test))
mae = mean_absolute_error(y_test, model.predict(X_test))

mse_baseline = mean_squared_error(y_test, dummy.predict(X_test))
mae_baseline = mean_absolute_error(y_test, dummy.predict(X_test))

spearman = spearmanr(y_test, model.predict(X_test))
kendall = kendalltau(y_test, model.predict(X_test))

residuals = y_test - model.predict(X_test)

table = pd.DataFrame(columns=["Model", "Baseline"], index=["RMSE", "MAE"])
table.loc["RMSE", "Model"] = round(math.sqrt(mse), 2)
table.loc["MAE", "Model"] = round(mae, 2)

table.loc["RMSE", "Baseline"] = round(math.sqrt(mse_baseline), 2)
table.loc["MAE", "Baseline"] = round(mae_baseline, 2)

table.loc["RMSE", "Improvement (%)"] = round((1 - (mse/mse_baseline)) * 100, 2)
table.loc["MAE", "Improvement (%)"] = round((1 - (mae/mae_baseline)) * 100, 2)

table.loc["Spearman", "Model"] = round(spearman.correlation, 2)
table.loc["Kendall", "Model"] = round(kendall.correlation, 2)

string = table.to_latex()

with open("./results/tables/results_days.tbl", "w") as file:
    file.write(string)
    file.close()