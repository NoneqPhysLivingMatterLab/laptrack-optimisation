# %%
"""
the following commands need to be executed before analysis.
the notebook was executed as a Jupyter notebook in evaluate_platform environment.

cd ~/.local/src/
git clone https://github.com/Fafa87/EP/
cd EP
git checkout 76d7c12b637169a088a6eb40f040d609e2b5d9dd
# change ep/evalplatform/plot_comparison.py line 28 to "output_evaluation_details = 1"
conda create -y -n evaluate_platform python=3.7
conda activate evaluate_platform
pip install -r requirements.txt
pip install pandas jupyter
"""

import os
from os import path
from glob import glob
from subprocess import run
from multiprocessing import Pool
from functools import partial
import pandas as pd

EP_path = path.join(os.getenv("HOME"),".local/src/EP")
assert path.isdir(EP_path)
print(EP_path)
results_dir = path.abspath("../results/yeast_image_toolkit_benchmark")

def build_command(detailed_results_dir,seg_df_path):
    tra_df_path=seg_df_path.replace("seg","tra") 
    assert path.isfile(tra_df_path)
    args=[
         detailed_results_dir,
         "GroundTruth",
         "GroundTruth_Segmentation.csv",
         "GroundTruth_Tracking.csv",
         "predicted",
         "predicted", 
         path.basename(seg_df_path),
         path.basename(tra_df_path),
    ]
    command = f"cd {EP_path} ; conda activate evaluate_platform ; python -m ep.evaluate {' '.join(args)}"
    return command

# %%
   
for i in range(1,11):
    print("analyzing", i)
    detailed_results_dir=path.join(results_dir,"detailed_tracking_results",f"TestSet{i}")
    seg_df_paths=glob(path.join(detailed_results_dir,"predicted","res_seg_*.txt"))
    assert len(seg_df_paths) > 0
    commands = []
    for seg_df_path in seg_df_paths:
        track_results_file=path.join(detailed_results_dir,"predicted","Output",
                                     f"{path.basename(seg_df_path)}.merged2.tmp.predicted.eval.summary.txt")
        if not path.exists(track_results_file):
            command=build_command(detailed_results_dir,seg_df_path)
            commands.append(command)
    pool = Pool(processes=20)
    pool.map(partial(run,shell=True,capture_output=True), commands) 
    pool.close()
    pool.join()

# %%

records=[]
for i in range(1,11):
    print("analyzing", i)
    detailed_results_dir=path.join(results_dir,"detailed_tracking_results",f"TestSet{i}")
    seg_df_paths=glob(path.join(detailed_results_dir,"predicted","res_seg_*.txt"))
    assert len(seg_df_paths) > 0
    print("finished")
    for seg_df_path in seg_df_paths:
        track_results_file=path.join(detailed_results_dir,"predicted","Output",
                                     f"{path.basename(seg_df_path)}.merged2.tmp.predicted.eval.summary.txt")
        if not path.exists(track_results_file):
            command=build_command(detailed_results_dir,seg_df_path)
            run(command,shell=True)
        with open(track_results_file,"r") as f:
            res=f.readlines()
            res_dict={ "Tracking "+r.split(":")[0]:float(r.split(":")[1].replace("\n","")) for r in res[-7:-4]}
            res_dict2={ "Long-time tracking "+r.split(":")[0]:float(r.split(":")[1].replace("\n","")) for r in res[-3:]}
            res_dict.update(res_dict2)
        print(res_dict)
        record={
            "TestSet":i,
            "seg_df_path":path.basename(seg_df_path),
            **res_dict
        }
        records.append(record)

# %%
results_df=pd.DataFrame.from_records(records)
results_df["max_distance"] =results_df["seg_df_path"].apply(lambda x:int(x.split("_")[2]))
results_df["gap_closing_max_distance"] =results_df["seg_df_path"].apply(lambda x:int(x[:-4].split("_")[3]))
results_df.to_csv(path.join(results_dir,"evaluation_platform_res.csv"),index=False)

# %%
