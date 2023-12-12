import matplotlib.pyplot as plt
import seaborn as sns
from pandas import read_csv

gen_df = read_csv(
    "csv/results_gen.csv",
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
    data=gen_df,
    x="N_bc",
    y="Train accuracy",
    row="K_e",
    col="M",
    kind="line",
    errorbar=("ci", 95),
    err_style="bars",
    err_kws={"capsize": 5},
)

plt.savefig("plots/gen_acc.png")
