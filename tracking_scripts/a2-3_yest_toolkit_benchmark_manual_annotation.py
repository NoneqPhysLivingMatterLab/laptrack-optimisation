# %%
from utils.data_loader import read_data
from utils.common import read_yaml, to_tree
import numpy as np
from laptrack import LapTrack
from laptrack.scores import calc_scores
from tqdm import tqdm
import pandas as pd
from matplotlib import pyplot as plt
from copy import deepcopy

# params = read_yaml("../setting_yaml/HL60_live.yaml")
# coords, track_labels, true_edges, GT_TRA_images = read_data(
#     "../data/HL60_live/organized_data", params["regionprop_keys"]
# )

params = read_yaml("../setting_yaml/yeast_image_toolkit_benchmark.yaml")
coords, track_labels, true_edges, GT_TRA_images = read_data(
    f"../data/yeast_image_toolkit_benchmark/organized_data/TestSet{i}", params["regionprop_keys"]
)


# %%
dt = [e[1][0] - e[0][0] for e in true_edges]
assert np.unique(dt) ==[1]
# %%

base_lt = LapTrack(
    track_cost_cutoff=14**2,
    gap_closing_cost_cutoff=25**2,
    splitting_cost_cutoff=25**2,
)

tree = base_lt.predict(coords)
# %%
scoress = []
scores = calc_scores(true_edges,tree.edges())
scores.update({
    "GT_edge_ratio":0,
    "use_GT_edge":False
})
scoress.append(scores)

# %%

unpredicted_edges = set(true_edges)-set(tree.edges())
unpredicted_nodes = list(set([e[0] for e in unpredicted_edges]))
print(len(unpredicted_nodes)) # == 247

# %%
gt_tree = to_tree(coords,true_edges)

def get_edges(GT_edge_count, unpredicted_nodes):
    N=len(unpredicted_nodes)
    inds = np.random.choice(N, GT_edge_count, replace=False)
    nodes = [unpredicted_nodes[i] for i in inds]
    edges = [e for e in gt_tree.edges() if e[0] in nodes]
    return edges

# %%
np.random.seed(24)

manual_fix_count = 5

# fix some of the edges and added to the GT edge list
# track again w/ freezing edges, measure remaining number of unpredicted_nodes

repeat=5
remaining_unpredicted_nodes=[]
threshold_remaining_node_count=10

for r in range(repeat):
    unpredicted_nodes2 = deepcopy(unpredicted_nodes)
    remaining_unpredicted_nodes.append({
            "repeat": r,
            "manual_fix_i":0,
            "unpredicted_node_count":len(unpredicted_nodes2),
            "unpredicted_node_count2":len(unpredicted_nodes2)
    })
    all_fixed_edges = []
    for manual_fix_i in range(45):
        print(manual_fix_i)
        try:
            fix_edges = get_edges(manual_fix_count,unpredicted_nodes2)
            all_fixed_edges.extend(fix_edges)
            tree1 = base_lt.predict(coords, 
                connected_edges=all_fixed_edges,
                split_merge_validation=False)
            unpredicted_edges = set(true_edges)-set(tree1.edges())
            unpredicted_nodes2 = list(set([e[0] for e in unpredicted_edges]))
            count1=len(unpredicted_nodes2)
        except ValueError:
            count1=np.nan

        try:
            fix_edges2 = get_edges(manual_fix_count*(manual_fix_i+1),unpredicted_nodes)
            tree2 = base_lt.predict(coords, 
                connected_edges=fix_edges2,
                split_merge_validation=False)
            unpredicted_edges = set(true_edges)-set(tree2.edges())
            unpredicted_nodes3 = list(set([e[0] for e in unpredicted_edges]))
            count2=len(unpredicted_nodes3)
        except ValueError:
            count2=np.nan

        remaining_unpredicted_nodes.append({
                    "repeat": r,
                    "manual_fix_i":manual_fix_i+1,
                    "unpredicted_node_count":count1,
                    "unpredicted_node_count2":count2
            })
# %%

remaining_unpredicted_nodes_df = pd.DataFrame.from_records(remaining_unpredicted_nodes)
remaining_unpredicted_nodes_df

mean_df=remaining_unpredicted_nodes_df.groupby("manual_fix_i").agg(["mean","std"]).reset_index()

plt.errorbar(
    mean_df["manual_fix_i"],
    mean_df["unpredicted_node_count"]["mean"],
    mean_df["unpredicted_node_count"]["std"],
    label="re-tracking"
)
plt.errorbar(
    mean_df["manual_fix_i"],
    mean_df["unpredicted_node_count2"]["mean"],
    mean_df["unpredicted_node_count2"]["std"],
    label="no re-tracking"
)
plt.xlabel("Manual fix rounds")
plt.ylabel("Remaining misconnected nodes")
plt.legend()


# %%


#    scores1 = calc_scores(true_edges,tree1.edges(), exclude_true_edges=edges)
#    unpredicted_edges = set(true_edges)-set(tree.edges())
#    unpredicted_nodes = list(set([e[0] for e in unpredicted_edges]))
#
#    tree2 = base_lt.predict(coords)
#    scores2 = calc_scores(true_edges,tree2.edges(), exclude_true_edges=edges)
#
#    scores1.update({
#        "manual_fix_i":manual_fix_i,
#        "use_GT_edge":True
#    })
#    scoress.append(scores1)
#    scores2.update({
#        "manual_fix_i":manual_fix_i,
#        "use_GT_edge":False
#    })
#    scoress.append(scores2)


# %%
import os
os.makedirs("../results/manual_annotation_effects", exist_ok=True)
scores_df = pd.DataFrame.from_records(scoress)
scores_df.to_csv("../results/manual_annotation_effects/HL60_live.csv")

# %%
scores_df = pd.read_csv("../results/manual_annotation_effects/HL60_live.csv",index_col=0)
scores_df

# %%
summarized=[]
keys = set(scores_df.columns) - set(["GT_edge_ratio","use_GT_edge"])
for GT_edge_ratio, grp in scores_df.groupby("GT_edge_ratio"):
    if GT_edge_ratio == 0:
        r1 = grp.iloc[0]
        r2 = grp.iloc[0]
    else:
        r1 = grp[grp["use_GT_edge"]].iloc[0]
        r2 = grp[~grp["use_GT_edge"]].iloc[0]

    s={
        "GT_edge_ratio" : GT_edge_ratio,
    }
    for k in keys:
        s.update({
            k+"_control" : r2[k],
            k+"_use_GT" : r1[k],
            k+"_diff" : r1[k] - r2[k],
        })
    summarized.append(s)
summarized_df=pd.DataFrame.from_records(summarized)

# %%
plt.figure(figsize=(4,3))
colors=[plt.cm.tab10(0), plt.cm.tab10(0.1)]
plt.plot(summarized_df["GT_edge_ratio"],summarized_df["track_purity_control"],  "o",
         c=colors[0], ls="--",
         label = "track purity control")
plt.plot(summarized_df["GT_edge_ratio"],summarized_df["target_effectiveness_control"], "o",
         c=colors[1], ls="--",
         label = "target effectiveness control",)
plt.plot(summarized_df["GT_edge_ratio"],summarized_df["track_purity_use_GT"],  "o",
         c=colors[0], ls = "-",
         label = "track purity GT used")
plt.plot(summarized_df["GT_edge_ratio"],summarized_df["target_effectiveness_use_GT"], "o",
         c=colors[1], ls = "-",
         label = "target effectiveness GT used",)


plt.legend(loc="lower left", bbox_to_anchor=(0, 1))
plt.xlabel("used edge ratio")
plt.ylabel("score")
plt.savefig("../plots/fig3b_scores_homeostasis.pdf",bbox_inches="tight")

# %%
!cp ../plots/fig3b_scores*.pdf ~/myworks/papers/2208_LapTrack/fig3
# %%
