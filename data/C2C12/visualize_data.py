# %%
import napari
from os import path
import os
import zarr
import pandas as pd
from dask import array as da
import numpy as np
from glob import glob
from dask_image import imread
from IPython.display import display
from matplotlib import pyplot as plt

# %%

viewer = napari.Viewer()

# %%
basedir = "/Volumes/common2/TEMPORARY/TimelapseExamples/TrackingProject/data/C2C12/090303-C2C12P15-FGF2,BMP2/hzd5p/osfstorage/"
images = imread.imread(path.join(basedir,"*.tif"))#[::5]
image_paths = sorted(glob(path.join(basedir,"*.tif")))
print(image_paths[0])
image_id = np.array([int(p.split("-")[-1].split(".")[0]) for p in image_paths])
ng_ids = image_id[:-1][image_id[1:] - image_id[:-1]>1]
print("----------------")
for i in range(1,800):
    p = f'/Volumes/common2/TEMPORARY/TimelapseExamples/TrackingProject/data/C2C12/090303-C2C12P15-FGF2,BMP2/hzd5p/osfstorage/exp1_F0009-{i:05d}.tif'
    if p not in image_paths:
        print(p)
    

# %%
viewer.add_image(images)
#viewer.add_labels(labels[target_Ts,0])

track_df=pd.read_csv("./organized_data/BMP2/090303-C2C12P15-FGF2,BMP2_9_all/regionprops.csv",index_col=0, comment="#")
division_df=pd.read_csv("./organized_data/BMP2/090303-C2C12P15-FGF2,BMP2_9_all/02_GT/TRA/man_track.txt",comment="#", delimiter=" ",
                        names=["track_id","start","end","parent_id"])

display(track_df)
display(division_df)
# %%
track_df["frame"]=track_df["frame"]+1
viewer.add_tracks(
    track_df[["track","frame","y","x"]].values
)
# %%

ok_frames = np.zeros(track_df["frame"].max(),dtype=int)
ng_frames = np.zeros(track_df["frame"].max(),dtype=int)
for i, grp in track_df.groupby("track"):
    frame = grp["frame"].values
    pos = grp[["x","y"]].values
    diff=np.linalg.norm(pos[1:]-pos[:-1],axis=1)
    ok_frames[frame[1:]] += diff!=0
    ng_frames[frame[1:]] += diff==0
    plt.plot(frame[1:],diff)
plt.show()
# %%
plt.plot(ng_frames / ok_frames)
# %%
root=path.dirname(path.dirname(path.dirname(basedir)))
for d in os.listdir(root):
    d2=path.join(root,d,"osfstorage")
    print(d2)
    print(len(glob(path.join(d2,"*.tif"))))

# %%
assert all(division_df[division_df["track_id"]==116]["parent_id"] == 0)
assert all(division_df[division_df["track_id"]==117]["parent_id"] == 0)

wrong_ids = []
candidates = [0,1,2]

while len(candidates) > 0:
    c = candidates.pop()
    wrong_ids.append(c)
    df = division_df[division_df["parent_id"]==c]
    if len(df) > 0:
        if c==0:
            children_id = [116,117]
        else:
            children_id = df["track_id"].to_list()
        candidates.extend(children_id)
wrong_ids = sorted(wrong_ids)
print(wrong_ids)

# %%
track_df2 = track_df[~track_df["track"].isin(wrong_ids)]
division_df2 = division_df[~division_df["track_id"].isin(wrong_ids)]

# %%
track_df2

# %%
viewer.add_tracks(
    track_df2[["track","frame","y","x"]].values
)
# %%
track_df2.to_csv("./organized_data/BMP2_updated/regionprops.csv")
os.makedirs("./organized_data/BMP2_updated/02_GT/TRA/",exist_ok=True)
division_df2.to_csv("./organized_data/BMP2_updated/02_GT/TRA/man_track.txt", 
                    index=False, sep=" ", header=False)

# %%
