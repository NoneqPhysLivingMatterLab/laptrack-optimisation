# %%
import pandas as pd
from os import path
import numpy as np
import os

regionprops_df = pd.read_csv("organized_data/BMP2/090303-C2C12P15-FGF2,BMP2_9_all/regionprops.csv",index_col=0)
track_txt_df_orig = pd.read_csv("organized_data/BMP2/090303-C2C12P15-FGF2,BMP2_9_all/02_GT/TRA/man_track.txt", 
                           delimiter=" ",names=["cell_id","min_T","max_T","parent_id"])
# %%
skip=5
df=regionprops_df.copy()
print(len(df))
df=df[df["frame"].isin(np.arange(0,df["frame"].max()+1,skip))]
print(len(df))
frame_map={f:j for j,f in enumerate(sorted(df["frame"].unique()))}
df["frame"]=df["frame"].map(frame_map)
output_path=f"organized_data/Sparse{skip}"
os.makedirs(output_path, exist_ok=True)
df.to_csv(path.join(output_path,"regionprops.csv"))

# %%

def get_parent_id(cell_id):
    row = track_txt_df_orig[track_txt_df_orig["cell_id"]==cell_id]
    assert len(row) == 1
    return row["parent_id"].iloc[0]

tracking_txt_df=[]
for cell_id, grp in df.groupby("track",dropna=False):
    parent_id = get_parent_id(cell_id)
    if parent_id != 0:
        if parent_id not in df["track"].to_list():
            parent_id = get_parent_id(parent_id)
            if parent_id not in df["track"].to_list():
                parent_id = 0
            
    tracking_txt_df.append(
        [
            cell_id,
            grp["frame"].min(),
            grp["frame"].max(),
            parent_id]
    )
track_txt_path=path.join(output_path,"02_GT/TRA/man_track.txt")
os.makedirs(path.dirname(track_txt_path),exist_ok=True)
np.savetxt(track_txt_path,np.array(tracking_txt_df,dtype=np.uint32),fmt="%d")
# %%
