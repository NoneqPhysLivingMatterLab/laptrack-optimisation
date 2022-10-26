# %%
"""
the following commands need to be executed before analysis

cd ~/.local/src/
git clone https://github.com/Fafa87/EP/
cd EP
git checkout 76d7c12b637169a088a6eb40f040d609e2b5d9dd
# change ep/evalplatform/plot_comparison.py line 28 to "output_evaluation_details = 1"
conda create -y -n evaluate_platform python=3.7
conda activate evaluate_platform
pip install -r requirements.txt
"""
import os
from os import path
from glob import glob
from subprocess import call

EP_path = path.join(os.getenv("HOME"),".local/src/EP")
assert path.isdir(EP_path)
print(EP_path)
# %%
results_dir = path.abspath("../results/yeast_image_toolkit_benchmark")
records=[]
for i in range(4,5):
    print("analyzing", i)
    detailed_results_dir=path.join(results_dir,"detailed_tracking_results",f"TestSet{i}")
    seg_df_paths=glob(path.join(detailed_results_dir,"predicted","res_seg_*.txt"))
    assert len(seg_df_paths) > 0
    for seg_df_path in seg_df_paths:
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
        call(command,shell=True)
        track_results_file=path.join(detailed_results_dir,"predicted","Output",f"{path.basename(seg_df_path)}.merged2.tmp.predicted.eval.summary.txt")
        assert path.exists(track_results_file)
        with open(track_results_file,"r") as f:
            res=f.readlines()
            res_dict={ r.split(":")[0]:float(r.split(":")[1].replace("\n","")) for r in res[-3:]}
        print(res_dict)
        record={
            "TestSet":i,
            "seg_df_path":path.basename(seg_df_path),
            **res_dict
        }
        records.append(record)
# %%
record
# %%
