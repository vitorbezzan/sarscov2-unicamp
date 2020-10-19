# Code for Predicting special care during the COVID-19 pandemic: A machine learning approach #
# by Vitor Bezzan (vitor@bezzan.com) and Cleber Rocco (cdrocco@unicamp.br)                   #
#                                                                                            #
# São Paulo, 2020                                                                            #

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from scipy.stats import ks_2samp


data = pd.read_parquet("./data/primary/hsl.parquet")
train, test = train_test_split(data, test_size=0.15, random_state=123)

X_train = train.drop(["TI"], axis=1)
X_test = test.drop(["TI"], axis=1)

stats = X_train.describe()
stats.loc["Empty"] = np.nan
stats.loc["Coverage"] = np.nan
stats.loc["KS Statistic"] = np.nan


for column in X_train.columns:
    stats.loc["Empty", column] = pd.isna(X_train[column]).sum()
    stats.loc["Coverage", column] = stats.loc["count", column] / (stats.loc["Empty", column] + stats.loc["count", column])

    try:
        stats.loc["KS Statistic", column] = ks_2samp(data[column][data["target_internacao"] == 1].dropna(),
                                                     data[column][data["target_internacao"] == 0].dropna())[1]
    except:
        stats.loc["KS Statistic", column] = np.nan

    sns.set_style("ticks")
    sns.set(font="Arial")
    sns.distplot(data[column][data["target_internacao"] == 1], color="skyblue", label="Hospitalar Care")
    sns.distplot(data[column][data["target_internacao"] == 0], color="red", label="Non-Hospitalar Care")
    plt.legend()
    plt.xlabel(column)
    plt.ylabel("Histogram and Adjusted Kernel")
    plt.savefig("./results/histograms/" + column + ".png")
    plt.close()

table = stats.loc[:,stats.loc["Coverage"].sort_values(ascending=False)[:30].index].T.round(decimals=2)
table["IQR"] = table["75%"] - table["25%"]
table = table.drop(["count", "Empty", "75%", "25%", "50%"], axis=1)
table = table[["mean", "std", "min", "IQR", "max", "Coverage", "KS Statistic"]]
table["Coverage"] = table["Coverage"] * 100
table = table.round(decimals=2)
with open("./results/tables/description.tbl", "w") as file:
    file.write(table.to_latex())
    file.close()

selected = data[["Leucocytes", "Lymphocytes", "Monocites", "Neutrophils", "target_internacao"]]
selected.columns = ["Leukocytes", "Lymphocytes", "Monocytes", "Neutrofils", "Special Care"]
sns.set_style("ticks")
sns.set(font="Arial")
g = sns.pairplot(selected, hue="Special Care", corner=False)
plt.savefig("./results/histograms/pairplot.png")
plt.close()