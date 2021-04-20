# Code for Predicting special care during the COVID-19 pandemic: A machine learning approach #
# by Vitor Bezzan (vitor@bezzan.com) and Cleber Rocco (cdrocco@unicamp.br)                   #
#                                                                                            #
# São Paulo, 2020                                                                            #

import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import math

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.dummy import DummyRegressor

from imblearn.pipeline import Pipeline

from src.bayesian_regressor import BayesianxgRegressor

data = pd.read_parquet("./data/primary/hsl.parquet")
train, test = train_test_split(data, test_size=0.15, random_state=123)

train = train[pd.notna(train["TI"])]
test = test[pd.notna(test["TI"])]

treat_anomalous_train = train[
    (train["TI"] > 0) & (train["target_internacao"] == 0)
].index
treat_anomalous_test = test[(test["TI"] > 0) & (test["target_internacao"] == 0)].index

train.loc[treat_anomalous_train, "TI"] = 0
test.loc[treat_anomalous_test, "TI"] = 0

X_train, y_train = train.drop(["target_internacao", "TI"], axis=1), train["TI"]
X_test, y_test = test.drop(["target_internacao", "TI"], axis=1), test["TI"]

steps = [("model_xgb", BayesianxgRegressor(trial_number=100))]

pipeline = Pipeline(steps)
model = pipeline.fit(X_train, y_train)
dummy = DummyRegressor().fit(X_train, y_train)

mse = mean_squared_error(y_test, model.predict(X_test))
mae = mean_absolute_error(y_test, model.predict(X_test))
r2 = r2_score(y_test, model.predict(X_test))

mse_baseline = mean_squared_error(y_test, dummy.predict(X_test))
mae_baseline = mean_absolute_error(y_test, dummy.predict(X_test))
r2_baseline = r2_score(y_test, dummy.predict(X_test))

residuals = y_test - model.predict(X_test)

table = pd.DataFrame(columns=["Model", "Baseline"], index=["RMSE", "MAE", "R2"])
table.loc["RMSE", "Model"] = round(math.sqrt(mse), 2)
table.loc["MAE", "Model"] = round(mae, 2)
table.loc["R2", "Model"] = round(r2, 2)

table.loc["RMSE", "Baseline"] = round(math.sqrt(mse_baseline), 2)
table.loc["MAE", "Baseline"] = round(mae_baseline, 2)
table.loc["R2", "Baseline"] = round(r2_baseline, 2)

table.loc["RMSE", "Improvement (%)"] = round((1 - (mse / mse_baseline)) * 100, 2)
table.loc["MAE", "Improvement (%)"] = round((1 - (mae / mae_baseline)) * 100, 2)
table.loc["R2", "Improvement (%)"] = round((1 - (r2 / r2_baseline)) * 100, 2)

string = table.to_latex()

with open("./results/tables/results_days.tbl", "w") as file:
    file.write(string)
    file.close()

plot = pd.concat([pd.Series(y_test.values), pd.Series(model.predict(X_test))], axis=1)
plot.columns = ["Ground Truth", "Predicted"]

sns.set_style("ticks")
sns.set(font="Arial")
plt.scatter(plot["Ground Truth"], plot["Predicted"])
plt.xlabel("Ground Truth")
plt.ylabel("Predictions")
plt.savefig("./results/xgb_results/target_tempo_internacao/dist.png")
plt.close()
