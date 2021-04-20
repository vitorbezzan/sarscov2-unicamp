# Code for Predicting special care during the COVID-19 pandemic: A machine learning approach #
# by Vitor Bezzan (vitor@bezzan.com) and Cleber Rocco (cdrocco@unicamp.br)                   #
#                                                                                            #
# São Paulo, 2020                                                                            #

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


data = pd.read_parquet("./data/primary/einstein.parquet")

selected = data[
    [
        "Hematocrit",
        "Hemoglobin",
        "Red blood Cells",
        "Mean corpuscular hemoglobin concentration (MCHC)",
        "Mean corpuscular hemoglobin (MCH)",
        "Mean corpuscular volume (MCV)",
        "Red blood cell distribution width (RDW)",
        "target_covid",
    ]
]
selected.columns = [
    "Hematocrit",
    "Hemoglobin",
    "Red blood Cells",
    "MCHC",
    "MCH",
    "MCV",
    "RDW",
    "Covid-19 Status",
]

sns.set_style("ticks")
sns.set(font="Arial")
g = sns.pairplot(selected, hue="Covid-19 Status", corner=False)
plt.savefig("./results/einstein/pairplot_covid_redseries.png")
plt.close()


selected = data[
    [
        "Platelets",
        "Mean platelet volume ",
        "Lymphocytes",
        "Leukocytes",
        "Basophils",
        "Eosinophils",
        "Monocytes",
        "target_covid",
    ]
]
selected.columns = [
    "Platelets",
    "Mean platelet volume ",
    "Lymphocytes",
    "Leukocytes",
    "Basophils",
    "Eosinophils",
    "Monocytes",
    "Covid-19 Status",
]

sns.set_style("ticks")
sns.set(font="Arial")
g = sns.pairplot(selected, hue="Covid-19 Status", corner=False)
plt.savefig("./results/einstein/pairplot_covid_whiteseries.png")
plt.close()


selected = data[data["target_covid"] == 1][
    [
        "Platelets",
        "Mean platelet volume ",
        "Lymphocytes",
        "Leukocytes",
        "Basophils",
        "Eosinophils",
        "Monocytes",
        "target_attention",
    ]
]
selected.columns = [
    "Platelets",
    "Mean platelet volume ",
    "Lymphocytes",
    "Leukocytes",
    "Basophils",
    "Eosinophils",
    "Monocytes",
    "Care",
]

sns.set_style("ticks")
sns.set(font="Arial")
g = sns.pairplot(selected, hue="Care", corner=False)
plt.savefig("./results/einstein/pairplot_covid_specialcare_infected.png")
plt.close()
