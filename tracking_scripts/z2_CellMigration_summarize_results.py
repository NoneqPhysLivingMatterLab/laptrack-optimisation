#%%
from glob import glob
import re
import pandas as pd
from os import path
from matplotlib import pyplot as plt
import yaml
import numpy as np
from IPython.display import display

from utils.common import score_name_map
#%%

csvs = glob("../results/CellMigration_use_0.05/*.csv")
# %%
filenames = [path.basename(c) for c in csvs]
pattern = "(.+)_Sparse(\d)_(\d)+.csv"
splitted = []
for f in filenames:
    print(f)
    splitted.append(re.search(pattern, f).groups())
# %%
files_df = (
    pd.DataFrame(
        {
            "filename": csvs,
            "skip": [s[1] for s in splitted],
            "method": [s[0] for s in splitted],
        }
    )
    .sort_values(
        [
            "skip",
            "method",
        ]
    )
    .reset_index(drop=True)
)
files_df["best_result_dir"] = files_df.apply(
    lambda row: path.join(
        "../results/CellMigration_use_0.05/best_results",
        path.basename(row["filename"]).replace(".csv",""),
    ),
    axis=1,
)
display(files_df)
# %%
score_keys = None
for i, row in files_df.iterrows():
    with open(path.join(row["best_result_dir"], "0","best_model_score.yaml")) as f:
        score = yaml.safe_load(f)
    for k, v in score.items():
        files_df.loc[i, k] = v
    score_keys = list(score.keys())

for i, row in files_df.iterrows():
    top_row = pd.read_csv(row["filename"]).iloc[0]
    for k in score_keys:
        files_df.loc[i, "tune_" + k] = top_row[k]
display(files_df)
# %%
score_keys2 = ["target_effectiveness","track_purity", "union_ratio", "division_recovery"]
colors = [plt.cm.tab10(i / 10) for i in range(10)]
symbols = "osv^><*Dd8"
fig, axes = plt.subplots(2, 2, figsize=(9, 7), gridspec_kw=dict(hspace=0.4))
for j, (k, ax) in enumerate(zip(score_keys2, np.ravel(axes))):
    for l, (c, s, (method, grp)) in enumerate(
        zip(colors, symbols, files_df.groupby("method"))
    ):
        if l > 4:
            continue
        ax.plot(
            grp["skip"],
            grp[k],
            s,
            c="none",
            mec=c,
            label=method if j == 3 else None,
        )
    ax.set_title(k)
    ax.set_xlabel("skip")
    ax.set_ylim(0,1.05)

axes[-1, -1].legend(loc="lower left", bbox_to_anchor=(0, 0))
# %%
#axes[-1, -1].axis("off")
plt.rcParams['font.family'] = "Arial"
fig, axes = plt.subplots(2, 2, figsize=(7, 5), 
    gridspec_kw=dict(hspace=0.5))
fig2, ax2 = plt.subplots(1,1, figsize=(2.25, 3))
skips = list(files_df.skip.unique())
methods = list(files_df.method.unique())
method_map = {
    "01_Simple_LAP" : "Without drift",
    "02_Simple_LAP_with_drift" : "With drift"
}
for j, (k, ax) in enumerate(zip(score_keys2, np.ravel(axes))):
    for l, (c, s, (method, grp)) in enumerate(
        zip(colors, symbols, files_df.groupby("method"))
    ):
        if k == "target_effectiveness":
            _axes = [ax, ax2]
        else:
            _axes = [ax]
        for m,_ax in  enumerate(_axes):
            if l > 4:
                continue
            if method in method_map:
                _ax.plot(
                    [skips.index(s) for s in grp["skip"]],
                    grp[k],
                    "o-",
                    color=c,
                    label=method_map[method] if j == len(score_keys2)-1 or m==1 else None,
                )
            _ax.set_title(score_name_map[k])
for ax in list(np.ravel(axes))+ [ax2]:
    ax.set_xlabel("Frames skipped")
    ax.set_ylim(0,1.05)
    ax.set_xticks(range(len(skips)),skips)

axes[-1, -1].legend(loc="lower left", bbox_to_anchor=(0, 0))
ax2.legend(loc="lower left", bbox_to_anchor=(0, 0))
fig.savefig("../plots/fig_SX_drift_scores.pdf",bbox_inches="tight")
fig2.subplots_adjust(left=0.2, right=0.95, bottom=0.15, top=0.85)
fig2.savefig("../plots/fig_2b_drift_score.pdf")
