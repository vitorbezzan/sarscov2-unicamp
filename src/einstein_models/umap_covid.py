# Code for Predicting special care during the COVID-19 pandemic: A machine learning approach #
# by Vitor Bezzan (vitor@bezzan.com) and Cleber Rocco (cdrocco@unicamp.br)                   #
#                                                                                            #
# São Paulo, 2020                                                                            #

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import umap

from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from scipy.stats import ttest_ind, kstest


data = pd.read_parquet("./data/primary/einstein.parquet")
data = data[
    [
        "Patient age quantile",
        "Hematocrit",
        "Hemoglobin",
        "Platelets",
        "Mean platelet volume ",
        "Red blood Cells",
        "Lymphocytes",
        "Mean corpuscular hemoglobin concentration (MCHC)",
        "Leukocytes",
        "Basophils",
        "Mean corpuscular hemoglobin (MCH)",
        "Eosinophils",
        "Mean corpuscular volume (MCV)",
        "Monocytes",
        "Red blood cell distribution width (RDW)",
        "target_covid",
        "target_regular",
        "target_special",
        "target_attention",
    ]
].dropna()

red_series = [
    "Hematocrit",
    "Hemoglobin",
    "Red blood Cells",
    "Mean corpuscular hemoglobin concentration (MCHC)",
    "Mean corpuscular hemoglobin (MCH)",
    "Mean corpuscular volume (MCV)",
    "Red blood cell distribution width (RDW)",
]

data["target_covid_attention"] = data["target_covid"] * data["target_attention"]

# Set variables for umap analysis
X = data.drop(columns=[column for column in data.columns if "target" in column])

# Model and cluster
neighbors = [10, 25, 50, 100, 150]
spreads = [0.01, 0.05, 0.1, 0.25]

results_table = pd.DataFrame(columns=["Neighbors", "Spread", "EPS", "Labels", "Score"])
N = 0
for neighbor in neighbors:
    for spread in spreads:
        for epss in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
            model = umap.UMAP(
                n_neighbors=neighbor, spread=spread, min_dist=0.01, random_state=42
            ).fit(X)
            transformed_data = model.transform(X)
            cluster = DBSCAN(eps=epss).fit(transformed_data)

            if np.sum(cluster.labels_) > 0:
                results_table.loc[N, "Neighbors"] = neighbor
                results_table.loc[N, "Spread"] = spread
                results_table.loc[N, "EPS"] = epss
                results_table.loc[N, "Labels"] = np.max(cluster.labels_) + 1
                results_table.loc[N, "Score"] = silhouette_score(
                    X, cluster.labels_, metric="euclidean"
                )
            N += 1

results_table.sort_values(by="Score", ascending=False, inplace=True)
selected = results_table.iloc[0]
model = umap.UMAP(
    n_neighbors=selected["Neighbors"],
    spread=selected["Spread"],
    min_dist=0.1,
    random_state=42,
).fit(X)
transformed_data = pd.DataFrame(model.transform(X), columns=["X", "Y"], index=X.index)
cluster = DBSCAN(eps=selected["EPS"]).fit(transformed_data.values)
transformed_data["Cluster"] = cluster.labels_
transformed_data = transformed_data[transformed_data["Cluster"] != -1]

colors = data["target_covid"] * transformed_data["Cluster"]
fig, (ax1, ax2) = plt.subplots(1, 2)
sns.set(font="Arial")
scatter = ax1.scatter(
    x=transformed_data["X"].values,
    y=transformed_data["Y"].values,
    c=transformed_data["Cluster"],
)
fig.colorbar(scatter, ticks=range(transformed_data["Cluster"].max() + 1))
ax1.set_title("Cluster Positions")

ax2.scatter(
    x=transformed_data[data["target_covid"] == 1]["X"].values,
    y=transformed_data[data["target_covid"] == 1]["Y"].values,
    c=transformed_data[data["target_covid"] == 1]["Cluster"],
)
ax2.set_title("COVID-19 patients")
plt.savefig("./results/einstein/cluster_covid.png")
plt.close()

transformed_data = transformed_data.merge(
    X, how="left", left_index=True, right_index=True
).drop(columns=["X", "Y"])
groupby = transformed_data.groupby(by="Cluster").mean()
groupby["Covid-19 (%)"] = (
    data["target_covid"].groupby(by=transformed_data["Cluster"]).mean()
)
groupby["# Patients"] = (
    data["target_covid"].groupby(by=transformed_data["Cluster"]).count()
)
groupby.sort_values(by="Covid-19 (%)", ascending=False, inplace=True)

with open("./results/tables/results_umap_covid_red.tbl", "w") as file:
    file.write(groupby[[*red_series, "Covid-19 (%)"]].to_latex())
    file.close()

with open("./results/tables/results_umap_covid_white.tbl", "w") as file:
    file.write(groupby.drop([*red_series, "Patient age quantile"], axis=1).to_latex())
    file.close()
