# %%
# Importing packages 

from laptrack import LapTrack
from functools import partial
from itertools import product
from ray import tune
from ray.tune.search import BasicVariantGenerator
import numpy as np
import os
from os import path
import networkx as nx
import pandas as pd

from utils.common import power_dist, read_yaml
from utils.data_loader import read_data
from utils.yeast_evaluation import save_evaluation_platform_input
from laptrack.scores import calc_scores

LAP_NAME = "01-2_Simple_LAP_baseline_grid"

max_dists = np.linspace(2, 47, 16).tolist()
gap_closing_max_dists = np.linspace(2, 47, 16).tolist()

config = {
    "max_distance": tune.grid_search(max_dists),
    "gap_closing_max_distance": tune.grid_search(gap_closing_max_dists),
}
initial_configs = [
    {
        "max_distance": max_dist,
        "gap_closing_max_distance": gap_closing_max_dist,
    }
    for max_dist, gap_closing_max_dist in product(
        max_dists, gap_closing_max_dists
    )
]


def get_tracker(config, regionprop_keys=None):
    ws = [1, 1] + [0] * (len(regionprop_keys) - 1)
    dist_power = 2
    return LapTrack(
        track_cost_cutoff=config["max_distance"] ** dist_power,
        gap_closing_cost_cutoff=config["gap_closing_max_distance"] ** dist_power,
        gap_closing_max_frame_count=1, 
        track_dist_metric=partial(power_dist, ws=ws, power=dist_power),
    )


def main():
    yaml_path = "../setting_yaml/yeast_image_toolkit_benchmark.yaml"
    results_dir = path.abspath("../results/yeast_image_toolkit_benchmark")
    for i in range(1,11):
        print("analyzing", i)
        base_dir = f"../data/yeast_image_toolkit_benchmark/organized_data/TestSet{i}"
        os.makedirs(results_dir, exist_ok=True)

        single_shot_count = 30

        yaml_params = read_yaml(yaml_path)
        regionprop_keys = yaml_params["regionprop_keys"]
        coords, track_labels, true_edges, GT_TRA_images = read_data(
            base_dir, regionprop_keys
        )

        def calc_fitting_score(config, report=True):
            lt = get_tracker(
                config,
                regionprop_keys=regionprop_keys,
            )
            track_tree = lt.predict(coords)
            predicted_edges = list(track_tree.edges())

            score_dict = calc_scores(true_edges, predicted_edges)

            ### output result for evaluation by evaluation platform (yeast image toolkit) ###
            # 
            trial_str = f'{int(config["max_distance"]):02d}_{int(config["gap_closing_max_distance"]):02d}'
            
            detailed_results_dir=path.join(results_dir,"detailed_tracking_results",f"TestSet{i}")
            os.makedirs(detailed_results_dir,exist_ok=True)
            output_dir=path.join(detailed_results_dir,"predicted")
            os.makedirs(output_dir,exist_ok=True)
            save_evaluation_platform_input(coords,predicted_edges,output_dir,trial_str)
            #
            ##################################################################################

            if report:
                tune.report(**score_dict)

        config2 = config.copy()
        search_alg = BasicVariantGenerator(
            points_to_evaluate=initial_configs,
            max_concurrent=single_shot_count,
        )
        analysis = tune.run(
            calc_fitting_score,
            config=config2,
            metric="Jaccard_index",
            mode="max",
            search_alg=search_alg,
        )
        analysis_df = analysis.results_df.sort_values(by="Jaccard_index", ascending=False)
        analysis_df.to_csv(path.join(results_dir, f"yeast_image_toolkit_grid_search_TestSet{i}.csv"))

if __name__ == "__main__":
    main()

# %%
