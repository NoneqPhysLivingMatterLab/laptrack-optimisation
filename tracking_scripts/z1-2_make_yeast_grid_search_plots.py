# %%
from re import I
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from os import path
from glob import glob
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
# %%
basedir="../results/yeast_image_toolkit_benchmark"
csvs=glob(path.join(basedir,"yeast_*.csv"))
dfs = []
for csv in csvs:
    df=pd.read_csv(csv)
    df["TestSet"] = int(csv[:-4].split("TestSet")[-1])
    dfs.append(df)
score_df = pd.concat(dfs)

score_df2=pd.read_csv(path.join(basedir,"evaluation_platform_res.csv"))

k1 = "max_distance"
k2 = "gap_closing_max_distance"
score_df = pd.merge(score_df,score_df2,
    left_on=["TestSet",f"config/{k1}",f"config/{k2}"],
    right_on=["TestSet",k1,k2],
                    )

assert len(score_df) == len(score_df2)
score_df
plt.rcParams['font.family'] = "Arial"

# %%
previous_score_df = pd.read_csv("yeast_benchmark_previous_results.tsv",delimiter="\t")
previous_score_df
# %%
methods = previous_score_df["method"].drop_duplicates().to_list()
print(methods)
symbols = "o*s^v><"
assert len(symbols) >= len(methods)

# %%
score_keys = [
    "Jaccard_index_original",
    "Tracking F",
    "Long-time tracking F",
]

def plot_score_key(grp,score_key, fig, ax, r=None, xskip=None):
    if not r:
        r = (0.7,1.0)
    if not xskip:
        xskip = 1
    vmin, vmax = r
    df2 = grp.sort_values([k1, k2])
    k1_vals = sorted(df2[k1].unique())
    k2_vals = sorted(df2[k2].unique())
    vals = df2[score_key].values.reshape(len(k1_vals), len(k2_vals)).T
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    im=ax.pcolormesh(k1_vals, k2_vals, vals, norm=norm)
#    fig.colorbar(label=score_key.replace("_original","").replace("_"," "))
    #ax.set_ylabel(k2)
    #ax.set_xticks(k1_vals[::xskip])
    #ax.set_yticks(k2_vals[::xskip])
    ax.set_aspect("equal")
    return im, norm

for score_key in score_keys:
    fig,axes = plt.subplots(2,5,figsize=(8*(3/3.5), 4*(3/3.5)))
    for ax, (TestSet, grp) in zip(np.ravel(axes),score_df.groupby("TestSet")):
        divider = make_axes_locatable(ax)
        ax_top = divider.append_axes("top", size="15%", pad=0.05)
        im, norm= plot_score_key(grp,score_key,fig,ax,xskip=5)

        previous_df = previous_score_df[previous_score_df["TestSet"]=="TS"+str(TestSet)]
        if score_key in previous_df.columns:
            previous_res = previous_df[["method",score_key]].set_index("method")
            N_methods=len(methods)
            for i, method in enumerate(methods):
                try:
                    res = previous_res.loc[method]
                    val = res[score_key]
                    if np.isfinite(val):
                        ax_top.scatter([i],[0],marker=symbols[i], 
                                       c=val,norm=norm,cmap=plt.cm.viridis)
                except KeyError:
                    pass
        ax_top.set_xlim([-0.5,len(methods)-0.5])
        ax_top.axis("off")
        ax_top.set_title(f"TestSet {TestSet}")
        if not TestSet in [1,6]:
            ax.set_yticks([])
    fig.supxlabel(k1)
    fig.supylabel(k2)
    fig.show()
    fig.colorbar(im,ax=axes.ravel().tolist(),
                 label=score_key.replace("_original","").replace("_"," "))
    markers = [Line2D([0],[0], color=plt.cm.viridis(0.5), marker=m, lw=0) 
               for m in symbols]
    fig.legend(markers,methods,handletextpad=0.01,columnspacing=0.01,
               ncol=len(markers),
               loc="lower left", bbox_to_anchor=(0,0.9))
    fig.savefig(f"../plots/figS_yeast_benchmark_{score_key}.pdf")


# %%
