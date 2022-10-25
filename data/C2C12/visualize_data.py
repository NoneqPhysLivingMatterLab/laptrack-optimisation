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

# %%

viewer = napari.Viewer()

# %%
basedir = "/Volumes/common2/TEMPORARY/TimelapseExamples/TrackingProject/data/C2C12/090303-C2C12P15-FGF2,BMP2/hzd5p/osfstorage/"
images = imread.imread(path.join(basedir,"*.tif"))
image_paths = glob(path.join(basedir,"*.tif"))
print(image_paths[0])

# %%
viewer.add_image(images)
#viewer.add_labels(labels[target_Ts,0])

track_df=pd.read_csv("./organized_data/BMP2/090303-C2C12P15-FGF2,BMP2_9_all/regionprops.csv",index_col=0)
# %%
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
