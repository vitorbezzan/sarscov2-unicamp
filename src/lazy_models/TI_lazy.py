# Code for Predicting special care during the COVID-19 pandemic: A machine learning approach #
# by Vitor Bezzan (vitor@bezzan.com) and Cleber Rocco (cdrocco@unicamp.br)                   #
#                                                                                            #
# São Paulo, 2020                                                                            #

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.base import RegressorMixin
from imblearn.pipeline import Pipeline
from lazypredict.Supervised import LazyRegressor


class IdentityTransformer(RegressorMixin):
    """
    Returns same data as input
    """
    def __init__(self, strategy: str = "none"):
        self.strategy = strategy

    def fit(self, X: pd.DataFrame, y):
        return self

    def transform(self, X: pd.DataFrame, y=None):
        if self.strategy == "none":
            return X
        else:
            return X.mean(axis=1)

    def predict(self, X: pd.DataFrame, y=None):
        if self.strategy == "none":
            return X
        else:
            return X.mean(axis=1)


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
    ("input_values", SimpleImputer(strategy="median")),
    ("data", IdentityTransformer())
]

pipeline = Pipeline(steps).fit(X_train, y_train)

X_train = pipeline.predict(X_train)
X_test = pipeline.predict(X_test)

clf = LazyRegressor(verbose=2, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)

with open("./results/tables/results_lazy_TI.tbl", "w") as file:
    file.write(models.to_latex())
    file.close()
