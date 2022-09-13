# %%
import numpy as np
import napari
from tqdm import tqdm
from skimage.io import imread
from skimage.transform import rescale
import pandas as pd
viewer = napari.Viewer()

# %%
factor = 10
track_df = pd.read_csv("organized_data/write_every_175/0/regionprops.csv").dropna()
track_df["track"]=track_df["track"].astype(np.int32)
track_df["y"] = track_df["y"] * factor
track_df["x"] = track_df["x"] * factor
# %%
viewer.add_tracks(
    track_df[["track","frame","x","y"]].values,
    properties=dict(null=np.ones(len(track_df)))
)
# %%
track_df

# %%
image = np.zeros((3,100,20*factor,20*factor))
size=1 
yy, xx = np.meshgrid(np.arange(20*factor),np.arange(20*factor))
for frame in tqdm(range(100)):
    df = track_df[track_df["frame"]==frame]
    y0s = df["y"].values
    x0s = df["x"].values
    dists = (yy[:,:,np.newaxis] - y0s) ** 2 + (xx[:,:,np.newaxis] - x0s)**2
    for channel in range(3):
        image[channel,frame] += np.sum(np.exp(-dists/2*size**2) * df[f"channel_{channel}"].values,axis=-1)


# %%
viewer.add_image(image, channel_axis=0)
# %%
viewer.add_image(image, scale=(factor, factor),channel_axis=0)

# %%
factor2=5
image2=rescale(image[:,:,:100,:100],(1,1,factor2,factor2),mode="edge")

# %%
viewer.add_image(image2, channel_axis=0, scale=[1/factor/factor2]*2)
# %%
viewer.scale_bar.visible=True

# %%
track_df = pd.read_csv("organized_data/write_every_10/0/regionprops.csv").dropna()
track_df["track"]=track_df["track"].astype(np.int32)
track_df["y"] = track_df["y"] 
track_df["x"] = track_df["x"] 
viewer.add_tracks(
    track_df[["track","frame","x","y"]].values,
    properties=dict(null=np.ones(len(track_df)))
)
# %%

# %%
