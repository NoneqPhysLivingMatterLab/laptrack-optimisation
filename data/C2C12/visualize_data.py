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

# %%

viewer = napari.Viewer()

# %%
basedir = "/Volumes/common2/TEMPORARY/TimelapseExamples/TrackingProject/data/C2C12/090303-C2C12P15-FGF2,BMP2/hzd5p/osfstorage/"
images = imread.imread(path.join(basedir,"*.tif"))#[::5]
image_paths = glob(path.join(basedir,"*.tif"))
print(image_paths[0])

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
