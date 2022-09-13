# %%
# data  ... from https://doi.org/10.5281/zenodo.6087728
import numpy as np
import napari
viewer = napari.Viewer()

# %%
from skimage.io import imread
images = imread("organized_data/Sparse1/image.tif")
# 1.5779 px = 1um
viewer.add_image(images, scale=[1,1/1.5779,1/1.5779])
# %%
import pandas as pd


# %%
track_df = pd.read_csv("organized_data/Sparse1/regionprops.csv").dropna()
track_df["track"]=track_df["track"].astype(np.int)
track_df["centroid-0"] = track_df["centroid-0"] / 1.5779
track_df["centroid-1"] = track_df["centroid-1"] / 1.5779
# %%
viewer.add_tracks(
    track_df[["track","frame","centroid-0","centroid-1"]].values
)
# %%
viewer.scale_bar.unit="um"
viewer.scale_bar.visible=True
viewer.scale_bar.position="top right"

# %%
