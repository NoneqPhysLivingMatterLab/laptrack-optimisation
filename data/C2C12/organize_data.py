# %%
from os import path
import pandas as pd
import os
from IPython.display import display
import numpy as np
from tqdm import tqdm
import zarr
import shutil
from skimage.measure import regionprops_table
from skimage.filters import gaussian
from skimage.io import imread, imsave
from glob import glob
import xmltodict

basedir = "/mnt/showers2/TEMPORARY/TimelapseExamples/TrackingProject/data/C2C12"
organized_data_dir = "/work/fukai/C2C12_organized_data"
organized_data_dir2 = "./organized_data"

organization_file = path.join(basedir, "41597_2018_BFsdata2018237_MOESM98_ESM","a_Campbell004A_assay.txt")
organization_df = pd.read_csv(organization_file, sep="\t")
organization_df[["Sample Name","Raw Data File","Assay Name"]]
#organization_df

# %%
files = glob(path.join(basedir, "*","*","osfstorage","*"))
# %%
files_df=pd.DataFrame(files, columns=["path"])
files_df["filename"] = files_df["path"].apply(path.basename)
files_df["experiment_set"] = files_df["path"].apply(lambda x: x.split(path.sep)[-4])
# %%
tiff_files_df = files_df[files_df["filename"].str.contains("\.tif")]
tiff_files_df["run"] = tiff_files_df["filename"].apply(lambda x: int(x[6:10]))
tiff_files_df
# %%
len(tiff_files_df.groupby(["experiment_set","filename"])) == len(tiff_files_df)

conditions=["control"]*4+["FGF2"]*4+["BMP2"]*4+["FGF2 BMP2"]*4
os.makedirs(organized_data_dir, exist_ok=True)
for c in set(conditions):
    os.makedirs(path.join(organized_data_dir,c), exist_ok=True)

for experiment_set, grp in tiff_files_df.groupby(["experiment_set"]):
    for condition, (run, grp2) in zip(conditions, grp.groupby(["run"])):
        print(condition,run)
        tiff_files = sorted(grp2["path"].tolist())
        output_path = path.join(organized_data_dir,condition,f"{experiment_set}_{run}")
        os.makedirs(output_path, exist_ok=True)
        if path.isfile(path.join(output_path,"images.npy")):
            continue
#        for f in tiff_files:
#            print(f)
#            imread(f)
#        np.save(path.join(output_path,"images.npy"), [imread(im) for im in tqdm(tiff_files)])
        output_path2 = path.join(organized_data_dir2,condition,f"{experiment_set}_{run}")
        os.makedirs(output_path2, exist_ok=True)

# %%
other_files_df = files_df[~files_df["filename"].str.contains("\.tif")]
other_files_df["ext"] = other_files_df["filename"].apply(lambda x: x.split(".")[-1])
other_files_df["ext"].unique()
# %%
csv_files_df=other_files_df[other_files_df["ext"]=="csv"]
test_csv_file=csv_files_df.iloc[0]["path"]
display(csv_files_df)
df = pd.read_csv(test_csv_file)
df
# csv file ... label mean stdev mode min max of the images (not useful)

# %%
def wrap_list(x):
    if isinstance(x, list):
        return x
    else:
        return [x]

def get_props(cell_node):
    # cell attributes
    # i={[0,inf]} Specifies the frame index.
    # x={[0.0,inf]} Specifies the cell’s x-position in the image.
    # y={[0.0,inf]} Specifies the cell’s y-position in the image.
    # f={“ ”} Specifies whether the result was generated from interpolation or “I” (for human-generated ground truths).
    # s={[0,18]}	 Specifies cell state (Table 3).
    column_map={"@i":"frame", "@x":"x", "@y":"y", "@f":"interpolation", "@s":"state"}
    coord_data=cell_node["ss"]["s"]
    if "as" in cell_node:
        children_nodes = wrap_list(cell_node["as"]["a"])
    else:
        children_nodes = []
#    print(coord_data)
    coord_data = wrap_list(coord_data)
    coord_df = pd.DataFrame.from_records(coord_data).rename(columns=column_map,errors="ignore")
    return coord_df, children_nodes

xml_files_df=other_files_df[other_files_df["ext"]=="xml"]

def parse_xml(xml_file):
    with open(xml_file, "r") as f:
        xml_content = "".join(f.readlines())
    xml_dict = xmltodict.parse(xml_content)
    vals=xml_dict["AnnotationDocument"]["fs"]["f"]

    vals = wrap_list(vals)
    cell_nodes = list(enumerate(sum([wrap_list(val["as"]["a"]) for val in vals], [])))
    new_track_id = len(cell_nodes)

    coord_dfs = []
    parent_track_ids = []
    while len(cell_nodes) > 0:
        track_id, cell_node = cell_nodes.pop(0)
        try:
            coord_df, children_nodes = get_props(cell_node)
        except TypeError as e:
            print(cell_node)
            raise e
        coord_df["track"] = track_id
        coord_dfs.append(coord_df)

        for child_node in children_nodes:
            _coord_df, _ = get_props(child_node)
            diff= (_coord_df["frame"].astype(int).min()-coord_df["frame"].astype(int).max()) 
            if int(_coord_df["state"].iloc[0]) == 2 and diff == 1:
                parent_track_ids.append([track_id, new_track_id])
            cell_nodes.append((new_track_id, child_node))
            new_track_id += 1

        coord_df=pd.concat(coord_dfs)
    coord_df["frame"]=coord_df["frame"].astype(int)

    parent_track_ids = np.array(parent_track_ids)
    parent_track_ids

    track_data=[]
    for track_id, grp in coord_df.groupby("track"):
        parent_track_id=parent_track_ids[parent_track_ids[:,1]==track_id,0]
        assert len(parent_track_id)<=1
        if len(parent_track_id)==0:
            parent_track_id=0
        else:
            parent_track_id=parent_track_id[0]

        track_data.append([
            track_id, grp["frame"].min(), grp["frame"].max(),parent_track_id,
        ])
    return coord_df, track_data

test_xml_file=xml_files_df.loc[32159]["path"]
print(test_xml_file)
coord_df, track_data = parse_xml(test_xml_file)

# %%
coord_df
track_data
# %%
xml_files_df[xml_files_df["filename"].str.contains("Full ")]
# %%
xml_files_df2 = xml_files_df[
    xml_files_df["filename"].str.contains("Human ") & 
    ~xml_files_df["filename"].str.contains("Full ") 
    ].copy()
#xml_files_df2["filename"]=xml_files_df2["filename"].apply(lambda x: x.replace(" Full Annotation",""))
xml_files_df2["run"] = xml_files_df2["filename"].apply(lambda x: int(x[12:16]))
# %%

for experiment_set, grp in xml_files_df2.groupby("experiment_set"):
    df = grp.sort_values("run", ascending=True)
    for condition, (i, row) in zip(conditions, df.iterrows()):
        xml_files_df2.loc[i,"condition"]=condition
#        print(i)
#        run = row["run"]
#
#        output_path = path.join(organized_data_dir2,condition,f"{experiment_set}_{run}")
#        print(output_path)
#        assert path.isdir(output_path)
#        coord_df, track_data = parse_xml(row["path"])
#
#        coord_df.to_csv(path.join(output_path,"regionprops.csv"))
#        track_txt_path=path.join(output_path,"02_GT/TRA/man_track.txt")
#        os.makedirs(path.dirname(track_txt_path),exist_ok=True)
#        np.savetxt(track_txt_path,np.array(track_data,dtype=np.uint32),fmt="%d")

 
# %%
xml_files_df2=xml_files_df2.sort_values(["experiment_set","run"])
xml_files_df2
# %%
xml_files_df3 = xml_files_df[xml_files_df["filename"].str.contains("Full ")].copy()
#xml_files_df2["filename"]=xml_files_df2["filename"].apply(lambda x: x.replace(" Full Annotation",""))
xml_files_df3["run"] = xml_files_df3["filename"].apply(lambda x: int(x[28:32]))
print(xml_files_df3)
# %%

max_T = 779 # "Graph Neural Network for Cell Tracking in Microscopy Videos" says the dataset is 780 frames
for experiment_set, grp in xml_files_df3.groupby("experiment_set"):
    df = grp.sort_values("run", ascending=True)
    for (i, row) in df.iterrows():
        condition = xml_files_df2[
            (xml_files_df2["experiment_set"]==experiment_set) &
            (xml_files_df2["run"]==row["run"])
        ].iloc[0]["condition"]
        print(i,condition)
        run = row["run"]

        output_path = path.join(organized_data_dir2,condition,f"{experiment_set}_{run}_all")
        os.makedirs(output_path,exist_ok=True)
        coord_df, track_data = parse_xml(row["path"])

        coord_df = coord_df[coord_df["frame"]<=max_T]
        track_data = np.array(track_data)
        track_data = track_data[track_data[:,1]<=max_T]
        track_data[:,2] = np.clip(track_data[:,2],0,max_T)

        coord_df.to_csv(path.join(output_path,"regionprops.csv"))
        track_txt_path=path.join(output_path,"02_GT/TRA/man_track.txt")
        os.makedirs(path.dirname(track_txt_path),exist_ok=True)
        np.savetxt(track_txt_path,np.array(track_data,dtype=np.uint32),fmt="%d")

 
# %%
# %%
