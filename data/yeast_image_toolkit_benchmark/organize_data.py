# %% 
import numpy as np
import pandas as pd
from pathlib import Path
from skimage.io import imread
import napari
from napari_animation import Animation
import os

# %%
base_dir=Path("/Volumes/common2/TEMPORARY/TimelapseExamples/TrackingProject/data/YIT-Benchmark2/")

def organize_data(base_dir,name):
    viewer = napari.Viewer()
    images = np.array([imread(im) for im in list((base_dir / "RawData").glob("*.tif"))])
    viewer.add_image(images)
    poss_df = pd.read_csv(base_dir / "GroundTruth" / "GroundTruth_Segmentation.csv", skipinitialspace=True)
    poss_df[["Frame_number"]]=poss_df[["Frame_number"]]-1
    viewer.add_points(poss_df[["Frame_number","Position_Y","Position_X"]])
    track_df = pd.read_csv(base_dir / "GroundTruth" / "GroundTruth_Tracking.csv", skipinitialspace=True)
    track_df[["Frame_number"]]=track_df[["Frame_number"]]-1
    assert len(track_df) == len(poss_df)
    poss_df2=pd.merge(track_df,poss_df,on=["Frame_number","Cell_number"])
    track_df_final=poss_df2.rename(columns={
        "Frame_number":"frame",
        "Unique_cell_number":"track",
    })
    viewer.add_tracks(
        track_df_final[["track","frame","Position_Y","Position_X"]]
    )
    animation = Animation(viewer)
    viewer.dims.current_step=(0,0,0,)
    animation.capture_keyframe()
    viewer.dims.current_step=(len(images)-1,0,0)
    animation.capture_keyframe()
    animation.animate(f'movies/{name}.mov', canvas_only=True, fps=5)
    
    track_path=f"organized_data/{name}/02_GT/TRA"
    os.makedirs(track_path)
    track_df_final.to_csv(f"organized_data/{name}/regionprops.csv")
    tracking_txt_df=[]
    for cell_id, grp in track_df_final.groupby("track",dropna=False):
        parent_id=0
        tracking_txt_df.append(
            [
                cell_id,
                grp["frame"].min(),
                grp["frame"].max(),
                parent_id]
        )
    track_txt_path=Path(track_path)/"man_track.txt"
    np.savetxt(track_txt_path,np.array(tracking_txt_df,dtype=np.uint32),fmt="%d")

for i in range(1,11):
    organize_data(base_dir/f"TestSet{i}",f"TestSet{i}")

# %%
# %%
