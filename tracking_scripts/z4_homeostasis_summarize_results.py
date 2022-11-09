#%%
from glob import glob
import re
import pandas as pd
import os
from os import path
from matplotlib import pyplot as plt
import numpy as np
from IPython.display import display
import yaml
from laptrack.scores import calc_scores
from utils.common import np_array_to_edge_set, score_name_map
score_name_map["mitotic_branching_correctness"]="Mitotic branching\ncorrectness"

do_calc_score = False

accept_names={
    "homeostasis":[
        "01_*",
        "04.+overlap$",
    ]
}

params = [
    {"name":"homeostasis","base_dir": "../results/homeostasis", "prefix": "area1"},
    {"name":"homeostasis","base_dir": "../results/homeostasis", "prefix": "area2"},
]

method_map={
    "01_Simple_LAP" : "Centroid",
    "04_simple_LAP_with_overlap_dist_sum_quantile0.999_factor1.50_overlap" : "Overlap",
}

plt.rcParams['font.family'] = "Arial"
fig, axes = plt.subplots(3, 2, figsize=(7, 9), gridspec_kw=dict(hspace=0.4))
fig2, ax2 = plt.subplots(1, 1, figsize=(1.75, 3))
total_col = 0

xs = []
tick_labels = []

for p_i, param in enumerate(params):
    name = param["name"]
    base_dir = param["base_dir"]
    prefix = param["prefix"]

    if do_calc_score:
        for d in glob(path.join(base_dir, "best_results", "*", "0")):
            true_edges = np_array_to_edge_set(np.load(path.join(d, "true_edges.npy")))
            predicted_edges = np_array_to_edge_set(
                np.load(path.join(d, "predicted_edges.npy"))
            )
            fit_edges_path = path.join(
                base_dir,
                "fitting_data",
                path.basename(path.dirname(d)),
                "fit_edgess.npy",
            )
            fit_edgess = np.load(fit_edges_path, allow_pickle=True)
            fit_edgess = [np_array_to_edge_set(fit_edges) for fit_edges in fit_edgess]
            assert len(fit_edgess) == 1
            scores = calc_scores(
                true_edges, predicted_edges, exclude_true_edges=fit_edgess[0]
            )
            print(scores)
            with open(path.join(d, "best_model_score.yaml"), "w") as f:
                yaml.dump(scores, f)

    csvs = glob(path.join(base_dir, "*.csv"))
    filenames = [path.basename(c) for c in csvs]
    pattern = f"(.+)_{prefix}_(\d)+.csv"
    splitted = []
    found_csvs = []
    for f in filenames:
        print(f)
        res = re.search(pattern, f)
        if res:
            splitted.append(res.groups())
            found_csvs.append(path.join(base_dir, f))
    files_df = (
        pd.DataFrame(
            {
                "filename": found_csvs,
                "id": [s[0] + "_lineage1" for s in splitted],
                "best_result_dir": [
                    path.join(base_dir, "best_results", path.basename(f)[:-4])
                    for f in found_csvs
                ],
                "method": [s[0] for s in splitted],
            }
        )
        .sort_values(["method"])
        .reset_index(drop=True)
    )

    display(files_df)
    score_keys = None
    trials = [
        p
        for p in os.listdir(files_df.iloc[0]["best_result_dir"])
        if path.isdir(path.join(files_df.iloc[0]["best_result_dir"], p))
    ]
    if len(trials) == 0:
        trials = [""]
    
    if name in accept_names.keys():
        files_df = files_df[files_df["method"].apply(lambda x:
            any([re.search(p,x) for p in accept_names[name]]))]

    for m,trial in enumerate(trials):
        for i, row in files_df.iterrows():
            best_result_dir = path.join(row["best_result_dir"], trial)
            try:
                with open(path.join(best_result_dir, "best_model_score.yaml")) as f:
                    score = yaml.safe_load(f)
            except FileNotFoundError:
                continue
            for k, v in score.items():
                files_df.loc[i, k] = v
            score_keys = list(score.keys())

        for i, row in files_df.iterrows():
            top_row = pd.read_csv(row["filename"]).iloc[0]
            for k in score_keys:
                if k in top_row.keys():
                    files_df.loc[i, "tune_" + k] = top_row[k]
        display(files_df)
        colors = [plt.cm.tab20(i / 10) for i in range(20)]
        colors = colors + colors

        methods = list(files_df["method"].unique())

        score_keys2 = ["Jaccard_index", "true_positive_rate", "precision", "mitotic_branching_correctness", "target_effectiveness","track_purity"]
        assert set(score_keys2) == set(score_keys)
        for j, (k, ax) in enumerate(zip(score_keys2, np.ravel(axes))):
            if k=="mitotic_branching_correctness":
                _axes = [ax,ax2]
            else:
                _axes = [ax]
            for n,_ax in enumerate(_axes):
                for l, (c, (method, grp)) in enumerate(
                    zip(colors, files_df.groupby("method"))
                ):
                    assert len(grp) == 5
                    col = methods.index(method)
                    _ax.bar(
                        col+total_col,
                        grp[k].median(),
                        width=1,
                        color=c,
                        label=method_map[method] if 
                            p_i==0 and (((j == 0) and (m==0)) or (n==1)) else None,
                        yerr=grp[k].std(),
                        capsize=2,
                    )
    #                if l == 0:
    #                    ax.hlines(
    #                        grp[k].median(), -0.5, len(methods) - 0.5, color="k", ls="--"
    #                    )
#                _ax.xaxis.set_visible(False)
                _ax.set_ylim(0.0, 1)
                _ax.set_title(score_name_map[k])
        xs.append(total_col + len(methods)/2-0.5)
        tick_labels.append(prefix.capitalize())
        total_col += len(methods) + 0.5 

axes[0, 0].legend(loc="lower left", bbox_to_anchor=(0., 0))
ax2.legend(loc="lower left", bbox_to_anchor=(0., 0))
for ax in np.ravel(axes):
    ax.set_xticks(xs, tick_labels)
ax2.set_xticks(xs, tick_labels)

fig.savefig(
    path.join("../plots/", f"figS_summary_{prefix}_all.pdf"), 
    bbox_inches="tight"
)
fig2.subplots_adjust(left=0.2, right=0.95, bottom=0.15, top=0.85)
fig2.savefig(
    path.join("../plots/", f"fig2_summary_{prefix}.pdf")
)


