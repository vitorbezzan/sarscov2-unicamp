# Code for Predicting special care during the COVID-19 pandemic: A machine learning approach #
# by Vitor Bezzan (vitor@bezzan.com) and Cleber Rocco (cdrocco@unicamp.br)                   #
#                                                                                            #
# São Paulo, 2020                                                                            #

import datetime as dt
import pandas as pd


# ====================================================================
# Treatments database
# ====================================================================
encounter = pd.read_csv("./data/raw/hsl_desfecho_1.csv", sep="|")
encounter["dt_desfecho"] = pd.to_datetime(
    encounter["dt_desfecho"], errors="coerce", format="%Y-%m-%d"
).dt.date

encounter["dt_atendimento"] = pd.to_datetime(
    encounter["dt_atendimento"], errors="coerce", format="%Y-%m-%d"
).dt.date

encounter["TI"] = (encounter["dt_desfecho"] - encounter["dt_atendimento"]).dt.days

encounter["de_tipo_atendimento"] = encounter["de_tipo_atendimento"].str.lower()
encounter["de_desfecho"] = encounter["de_desfecho"].str.lower()

encounter["target_internacao"] = (
    encounter["de_tipo_atendimento"] == "internado"
).astype(int)

encounter = encounter[["id_paciente", "id_atendimento", "target_internacao", "TI"]]


# ====================================================================
# Treating patient information
# ====================================================================
patient = pd.read_csv("./data/raw/hsl_patient_1.csv", sep="|")
patient["age"] = dt.datetime.now().year - pd.to_numeric(
    patient["aa_nascimento"], errors="coerce"
)
patient["sex"] = patient["ic_sexo"].apply(lambda x: 1 if x == "M" else 0)
patient = patient[["id_paciente", "age", "sex"]]


# ====================================================================
# Treating results table
# ====================================================================
results = pd.read_csv("./data/raw/hsl_exames_tratado.csv")

results["DT_COLETA"] = pd.to_datetime(
    results["DT_COLETA"], errors="coerce", format="%m/%d/%Y"
).dt.date

results = results[
    ["ID_PACIENTE", "ID_ATENDIMENTO", "DT_COLETA", "PARA_ANALITO", "DE_RESULTADO"]
]
results["DE_RESULTADO"] = pd.to_numeric(
    results["DE_RESULTADO"].str.replace(",", "."), errors="coerce"
)

results = results.dropna()

results = results.pivot_table(
    index=["ID_PACIENTE", "ID_ATENDIMENTO", "DT_COLETA"],
    columns="PARA_ANALITO",
    values="DE_RESULTADO",
    aggfunc="mean",
).reset_index()

final = encounter.merge(patient, on="id_paciente", how="left")
final = final.merge(
    results,
    left_on=["id_paciente", "id_atendimento"],
    right_on=["ID_PACIENTE", "ID_ATENDIMENTO"],
    how="left",
)
final.drop(["ID_PACIENTE", "ID_ATENDIMENTO"], axis=1, inplace=True)
final.sort_values(["id_atendimento", "DT_COLETA"], inplace=True)

final = (
    final.drop("id_paciente", axis=1)
    .groupby("id_atendimento")
    .first()
    .drop("DT_COLETA", axis=1)
)

final.to_parquet("./data/primary/hsl.parquet")
