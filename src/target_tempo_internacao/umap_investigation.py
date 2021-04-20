# Code for Predicting special care during the COVID-19 pandemic: A machine learning approach #
# by Vitor Bezzan (vitor@bezzan.com) and Cleber Rocco (cdrocco@unicamp.br)                   #
#                                                                                            #
# São Paulo, 2020                                                                            #

import pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn as sns
import umap

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


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

y = pd.concat([y_train, y_test], axis=0).sort_values()
X = pd.concat([X_train, X_test], axis=0).loc[y.index]

X = pd.DataFrame(
    SimpleImputer(strategy="median").fit_transform(X), columns=X.columns, index=X.index
)

neighbors = [50, 100, 200]
spread = [1, 5, 10]

plt.rcParams["figure.figsize"] = [12, 12]
figure, axis = plt.subplots(3, 3)
for i in range(0, len(neighbors)):
    for j in range(0, len(spread)):
        result = umap.UMAP(spread=spread[j], n_neighbors=neighbors[i]).fit_transform(X)
        axis[i, j].scatter(result[:, 0], result[:, 1], c=y)
        axis[i, j].set_title(" Neighbors = %s, Spread = %s" % (neighbors[i], spread[j]))
        print(i, j)
