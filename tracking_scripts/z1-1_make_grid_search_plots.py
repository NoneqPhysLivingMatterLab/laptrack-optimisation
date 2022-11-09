# %%
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from os import path

csvs = [
    "../results/C2C12_grid_search/C2C12_grid_search.csv",
    "../results/homeostasis_grid_search/homeostasis_grid_search_area1.csv",
    "../results/homeostasis_grid_search/homeostasis_grid_search_area2.csv",
]

_ranges = [
    (0.91, 1.),
    (0.91, 0.95),
    (0.91, 0.95),
]

score_key = "target_effectiveness"

k1 = "config.max_distance"
k2 = "config.splitting_max_distance"

def plot_score_key(score_key, ranges=None, xskips=None):
    if not ranges:
        ranges = [(0.7,1.0)]*len(csvs)
    if not xskips:
        xskips = [1]*len(csvs)
    for csv,r,xskip in zip(csvs,ranges,xskips):
        vmin, vmax = r
        df = pd.read_csv(csv)
        plt.figure(figsize=(10, 5))
        for j, (gap_closing, grp) in enumerate(df.groupby("config.gap_closing")):
            plt.subplot(1, 2, j + 1)
            df2 = grp.sort_values([k1, k2])
            k1_vals = sorted(df2[k1].unique())
            k2_vals = sorted(df2[k2].unique())
            vals = df2[score_key].values.reshape(len(k1_vals), len(k2_vals)).T
            plt.pcolormesh(k1_vals, k2_vals, vals, vmin=vmin, vmax=vmax)
            plt.xlabel("max_distance")
            plt.ylabel("splitting_max_distance")
            plt.colorbar()
            plt.xlabel("max_distance")
            plt.ylabel("splitting_max_distance")
            plt.gca().set_xticks(k1_vals[::xskip])
            plt.gca().set_yticks(k2_vals)
            plt.gca().set_aspect("equal")
            plt.title(f"gap_closing={gap_closing}")
        plt.tight_layout()
        plt.savefig(csv.replace(".csv", f"_{score_key}.pdf"))
        plt.show()

ranges = [
    [0.95,1.0],
    [0.9,0.95],
    [0.8,1.0],
    [0.8,0.9]
]
plot_score_key(score_key,ranges=ranges,xskips=[1,1,1,4])
ranges = [(100,300)]*len(csvs)
#plot_score_key("time_this_iter_s",ranges)


# %%
plt.rcParams['font.family'] = "Arial"
names = []
for j, (csv, r) in enumerate(zip(np.array(csvs),_ranges)):
    plt.rcParams['font.family'] = "Arial"
    fig, ax = plt.subplots(1,1,figsize=(4*(3/3.5), 3*(3/3.5)), 
                            gridspec_kw=dict(wspace=-0.3))
    df = pd.read_csv(csv)
    grp = df[df["config.gap_closing"]==1]
    df2 = grp.sort_values([k1, k2])
    k1_vals = np.array(sorted(df2[k1].unique()))
    k2_vals = np.array(sorted(df2[k2].unique()))
    vals = df2[score_key].values.reshape(len(k1_vals), len(k2_vals)).T
    im=ax.pcolormesh(k1_vals, k2_vals, vals, vmin=r[0], vmax=r[1])
    ax.set_xlabel("max_distance")
    ax.set_ylabel("splitting_max_distance")
    name = path.basename(csv)[:-4]
#    axes[j].set_xticks(k1_vals[::4])
#    axes[j].set_yticks(k2_vals)
#    axes[j].set_aspect("equal")
    if j>0:
        max_pos = np.array(np.nonzero(vals==np.max(vals)))
        ax.scatter(k1_vals[max_pos[1]], k2_vals[max_pos[0]], c="red")
    if j == 0:
        ax.set_title(f"{np.max(vals):.3f}")
        ax.set_aspect("equal")
    fig.colorbar(im,label="Target effectiveness")
    fig.savefig(f"../plots/fig2a2_{name}.pdf", bbox_inches='tight')
# %%
# !cp ../plots/fig2a2_C2C12_grid_search.pdf /Users/fukai/myworks/papers/2208_LapTrack2/figS_C2C12_grid_search.pdf
# %%
df1 = pd.read_csv(csvs[1])
df2 = pd.read_csv(csvs[2])
# %%
df_all = df1.merge(df2,on=["config.max_distance","config.splitting_max_distance","config.gap_closing"])
df_all
# %%
plt.rcParams['font.family'] = "Arial"
plt.figure(figsize=(5,2.5))
for j,k in enumerate(["target_effectiveness","track_purity"]):
    k2=k.replace('_',' ').capitalize()
    plt.subplot(1,2,j+1)
    xs, ys =df_all[f"{k}_x"], df_all[f"{k}_y"]
    plt.plot(xs,ys,".")
    plt.xlim(0.75,1)
    plt.ylim(0.75,1)
    plt.xlabel(f"{k2} at area 1")
    if j == 0:
        plt.ylabel(f"{k2} at area 2", y=0.4)
    else:
        plt.ylabel(f"{k2} at area 2")
    ind = (xs>0.75) & (ys>0.75)
    r = pearsonr(xs[ind],ys[ind])
    plt.text(0.835,0.758,rf"$r={r.statistic:.2f}$"+"\n"+fr"(score$>0.75$)")
    plt.plot([0.75,1],[0.75,1],"--k")
    plt.gca().set_aspect('equal')
plt.tight_layout()
plt.savefig("../plots/figS_score_corr.pdf",bbox_inches='tight')


# %%
# !cp ../plots/figS_score_corr.pdf /Users/fukai/myworks/papers/2208_LapTrack2/
# %%
