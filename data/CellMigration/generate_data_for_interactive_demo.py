# %%
from skimage.io import imread,imsave
import numpy as np
from laptrack import LapTrack
from laptrack.data_conversion import convert_dataframe_to_coords, convert_tree_to_dataframe
import pandas as pd
images = imread("organized_data/Sparse5/image.tif")
mask = imread("organized_data/Sparse5/predicted_label.tif")
track_df = pd.read_csv("organized_data/Sparse5/regionprops.csv")
coords2 = convert_dataframe_to_coords(track_df,["centroid-0","centroid-1","label"])
coords = [c[:,:-1] for c in coords2]
labels = [c[:,-1] for c in coords2]
# %%
lt=LapTrack(
    track_cost_cutoff=100**2,
    gap_closing_max_frame_count=1
)
tree=lt.predict(coords)

# %%
df, _, _ = convert_tree_to_dataframe(tree)
df=df.reset_index()

# %%
df["frame"] = df["frame"].astype(np.int32)
df["index"] = df["index"].astype(np.int32)
df["label"] = df.apply(lambda row: labels[row["frame"]][row["index"]],axis=1).astype(np.int32)

# %%
df

# %%
new_mask = np.zeros_like(mask)
for frame in range(len(labels)):
    for _, row in df[df["frame"]==frame].iterrows():
        new_mask[frame][mask[frame]==row["label"]]=row["track_id"]

# %%
import napari

viewer = napari.Viewer()
viewer.add_image(images)


# %%
viewer.add_labels(new_mask)
# %%
window=(slice(12,None),slice(800,1000),slice(400,600))
images2=images[window]
new_mask2=new_mask[window]

# %%

viewer.add_image(images2)
l=viewer.add_labels(new_mask2)
# %%
imsave("example_images.tif",images2)
imsave("example_label.tif",new_mask2)

