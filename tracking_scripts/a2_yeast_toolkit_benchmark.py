# %%
from laptrack import LapTrack
from functools import partial
from itertools import product
from ray import tune
from ray.tune.search import BasicVariantGenerator
import numpy as np
import os
from os import path
import networkx as nx

from utils.common import power_dist, read_yaml
from utils.data_loader import read_data
from laptrack.scores import calc_scores
from laptrack.utils import order_edges

LAP_NAME = "01-2_Simple_LAP_baseline_grid"

max_dists = np.linspace(2, 47, 16).tolist()
split_max_dists = np.linspace(2, 47, 16).tolist()
gap_closings = [0, 1]

config = {
    "max_distance": tune.grid_search(max_dists),
    "splitting_max_distance": tune.grid_search(split_max_dists),
    #    "gap_closing_max_distance": tune.grid_search(np.linspace(4, 20, 4)) ,
    "gap_closing": tune.grid_search(gap_closings),
}
initial_configs = [
    {
        "max_distance": max_dist,
        "splitting_max_distance": split_max_dist,
        #    "gap_closing_max_distance": 20,
        "gap_closing": gap_closing,
    }
    for max_dist, split_max_dist, gap_closing in product(
        max_dists, split_max_dists, gap_closings
    )
]


def get_tracker(config, regionprop_keys=None):
    ws = [1, 1] + [0] * (len(regionprop_keys) - 1)
    dist_power = 2
    return LapTrack(
        track_cost_cutoff=config["max_distance"] ** dist_power,
        splitting_cost_cutoff=config["splitting_max_distance"] ** dist_power,
        gap_closing_cost_cutoff=config["splitting_max_distance"] ** dist_power,
        gap_closing_max_frame_count=config["gap_closing"],
        track_dist_metric=partial(power_dist, ws=ws, power=dist_power),
        splitting_dist_metric=partial(power_dist, ws=ws, power=dist_power),
    )


def main():
    for i in range(1,10):
        print("analyzing", i)
        base_dir = f"../data/yeast_image_toolkit_benchmark/organized_data/TestSet{i}"
        yaml_path = "../setting_yaml/yeast_image_toolkit_benchmark.yaml"
        results_dir = "../results/yeast_image_toolkit_benchmark"
        os.makedirs(results_dir, exist_ok=True)

        single_shot_count = 30

        yaml_params = read_yaml(yaml_path)
        regionprop_keys = yaml_params["regionprop_keys"]
        normalize_exclude_keys = yaml_params["normalize_exclude_keys"]
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

            score_dict_original = calc_scores(true_edges, predicted_edges)
            score_dict_original = {k+"_original":v for k,v in score_dict_original.items()}

            ## ignore the splitting as the original score does not include division
            gt_tree = nx.from_edgelist(order_edges(true_edges), create_using=nx.DiGraph)
            pred_tree = nx.from_edgelist(
                    order_edges(predicted_edges), create_using=nx.DiGraph
            )
            true_edges2 = [e for e in true_edges if len(list(gt_tree.successors(e[0]))) < 2]
            predicted_edges2 = [e for e in predicted_edges if len(list(pred_tree.successors(e[0]))) < 2]
            score_dict = calc_scores(true_edges2, predicted_edges2)

            if report:
                tune.report(**score_dict_original, **score_dict)

        #    # test run
        #    test_config = initial_configs[0].copy()
        #    calc_fitting_score(test_config, report=False)

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
            #                resources_per_trial={"cpu": single_shot_count*4}
        )
        analysis_df = analysis.results_df.sort_values(by="Jaccard_index", ascending=False)
        analysis_df.to_csv(path.join(results_dir, f"yeast_image_toolkit_grid_search_TestSet{i}.csv"))

if __name__ == "__main__":
    main()

# %%
