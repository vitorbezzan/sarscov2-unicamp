# Code for Predicting special care during the COVID-19 pandemic: A machine learning approach #
# by Vitor Bezzan (vitor@bezzan.com) and Cleber Rocco (cdrocco@unicamp.br)                   #
#                                                                                            #
# São Paulo, 2020                                                                            #

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data = pd.read_csv("./data/raw/einstein_dataset.csv")
data = data.set_index("Patient ID")

# Replace exam indicators for 0/1
data = data.replace({"negative": 0, "positive": 1})

# Replace other exam indicators for 0/1
data = data.replace({"not_detected": 0, "detected": 1})
data = data.replace(
    {"not_done": np.nan, "Não Realizado": np.nan, "Ausentes": 0, "absent": 0}
)
data = data.drop(
    columns=[
        "Urine - Esterase",
        "Urine - Aspect",
        "Urine - pH",
        "Urine - Hemoglobin",
        "Urine - Bile pigments",
        "Urine - Ketone Bodies",
        "Urine - Nitrite",
        "Urine - Density",
        "Urine - Urobilinogen",
        "Urine - Protein",
        "Urine - Sugar",
        "Urine - Leukocytes",
        "Urine - Crystals",
        "Urine - Red blood cells",
        "Urine - Hyaline cylinders",
        "Urine - Granular cylinders",
        "Urine - Yeasts",
        "Urine - Color",
    ]
)

# Rename targets
data["target_covid"] = data["SARS-Cov-2 exam result"]
data["target_regular"] = data["Patient addmited to regular ward (1=yes, 0=no)"]
data["target_special"] = (
    (
        data["Patient addmited to semi-intensive unit (1=yes, 0=no)"]
        + data["Patient addmited to intensive care unit (1=yes, 0=no)"]
    )
    >= 1
).astype(int)
data["target_attention"] = (
    (data["target_regular"] + data["target_special"]) >= 1
).astype(int)

data = data.drop(
    columns=[
        "SARS-Cov-2 exam result",
        "Patient addmited to regular ward (1=yes, 0=no)",
        "Patient addmited to semi-intensive unit (1=yes, 0=no)",
        "Patient addmited to intensive care unit (1=yes, 0=no)",
    ]
)

# All data columns should be numerical
for column in data.columns:
    data[column] = pd.to_numeric(data[column])

data.to_parquet("./data/primary/einstein.parquet")
