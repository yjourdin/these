import matplotlib.pyplot as plt
import seaborn as sns
from pandas import read_csv

# MIP benchmark

MIP_df = read_csv(
    "results_MIP.csv",
    dtype={
        "N_tr": "category",
        "N_te": "category",
        "M": "category",
        "K_o": "category",
        "K_e": "category",
        # "N_bc": "category",
        "Method": "category",
        "Model": "category",
    },
)

sns.relplot(
    data=MIP_df,
    x="N_bc",
    y="Time",
    style="K_e",
    hue="K_e",
    col="M",
    kind="line",
    errorbar="sd",
    err_style="bars",
    err_kws={"capsize": 5},
)

plt.savefig("MIP_time.png")

sns.relplot(
    data=MIP_df,
    x="N_bc",
    y="Train accuracy",
    style="K_e",
    hue="K_e",
    col="M",
    kind="line",
    errorbar="sd",
    err_style="bars",
    err_kws={"capsize": 5},
)

plt.savefig("MIP_acc.png")


# Evolutionnary benchmark

evo_df = read_csv(
    "results_evo.csv",
    dtype={
        "N_tr": "category",
        "N_te": "category",
        "M": "category",
        "K_o": "category",
        "K_e": "category",
        # "N_bc": "category",
        "Method": "category",
        "Model": "category",
    },
)

sns.relplot(
    data=evo_df,
    x="N_bc",
    y="Time",
    row="K_e",
    col="M",
    kind="line",
    errorbar="sd",
    err_style="bars",
    err_kws={"capsize": 5},
)

plt.savefig("evo_time.png")

sns.relplot(
    data=evo_df,
    x="N_bc",
    y="Train accuracy",
    row="K_e",
    col="M",
    kind="line",
    errorbar="sd",
    err_style="bars",
    err_kws={"capsize": 5},
)

plt.savefig("evo_acc.png")
