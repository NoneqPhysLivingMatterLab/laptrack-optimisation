################################################
# the script to calculate tracking scores 
# with the overlap-based metric
################################################

from functools import partial
from fire import Fire
from ray import tune
import numpy as np


from utils.common import main, read_yaml, power_dist
from laptrack import LapTrack

LAP_NAME = "04_simple_LAP_with_overlap_dist_sum"

def get_initial_configs_csv_pattern(yaml_params, prefix):
    if yaml_params["drift"]:
        return f"02_Simple_LAP_with_drift_{prefix}*.csv"
    else:
        return f"01_Simple_LAP_{prefix}*.csv"

# we define the metric. the overlap is pre-computed and stored in overlap_df.
# see https://github.com/yfukai/laptrack/blob/202b9ec0d24fab97406a073ac8c78d00a97a2e3f/docs/examples/overlap_tracking.ipynb
# for the simplified example.
def metric_overlap(
           c1s,c2s, 
           overlap_df,
           max_dist,
           dist_weight,
           weighted_dist_weight,
           overlap_type="ratio_2",
           use_overlap=True,
           use_euclidean_dist=False,
           use_weighted_dist=False,
           nll_offset=0.001):
    _c1s, frame1, label1 = c1s[:-2], c1s[-2], c1s[-1]
    _c2s, frame2, label2 = c2s[:-2], c2s[-2], c2s[-1]
    euclidean_dist = np.linalg.norm(_c1s[:2]-_c2s[:2])
    if euclidean_dist > max_dist:
        return np.infty
    w_euclidean_dist = np.linalg.norm(_c1s[2:4]-_c2s[2:4])

    if frame1 > frame2:
        tmp = _c1s, frame1, label1
        _c1s, frame1, label1 = _c2s, frame2, label2 
        _c2s, frame2, label2 = tmp

    score = 0
    if use_overlap:
        ind=(int(frame1),int(label1),int(label2))
        overlap, iou, ratio_1, ratio_2 = overlap_df.loc[ind, ["overlap", "iou", "ratio_1", "ratio_2"]]
        if overlap_type == "ratio_2":
            val = ratio_2
        elif overlap_type == "iou":
            val = iou
        score = score - np.log((val+nll_offset)/(1+nll_offset))
    if use_euclidean_dist:
        score = score + dist_weight * euclidean_dist ** 2
    if use_weighted_dist:
        score = score + weighted_dist_weight * w_euclidean_dist ** 2 
    return score

# the function to return the LapTrack objects 
# with given parameters in the "config".
def get_tracker(
    config, 
    division,
    overlap_df,
    regionprop_keys=None, 
    second_only=False, 
    use_overlap=True,
    use_euclidean_dist=False,
    use_weighted_dist=False,
    use_iou=False,
):
    ws = [1, 1,] + [
        0
    ] * (len(regionprop_keys) )

    # load the parameter values
    if "dist_weight" in config.keys():
        dist_weight = config["dist_weight"]
    else:
        dist_weight = 0
    if "weighted_dist_weight" in config.keys():
        weighted_dist_weight = config["weighted_dist_weight"]
    else:
        weighted_dist_weight = 0
    if "nll_offset" in config.keys():
        nll_offset = config["nll_offset"]
    else:
        nll_offset = 0
 
    # define the metric function
    metric = partial(
        metric_overlap,
        overlap_df=overlap_df,       
        dist_weight=dist_weight,
        weighted_dist_weight=weighted_dist_weight,
        use_overlap=use_overlap,
        use_euclidean_dist=use_euclidean_dist,
        use_weighted_dist=use_weighted_dist,
        nll_offset=nll_offset,
    )
 
    # depending on the condition, use different metrics
    get_metric = lambda max_dist, overlap_type: partial(
        metric, max_dist=max_dist, overlap_type=overlap_type, 
    )
    if not second_only:
        # both of the tracking and splitting detections uses the overlap
        track_dist_metric = get_metric(config["max_distance"], "iou" if use_iou else "ratio_2")
    else:
        # only splitting detections uses the overlap
        track_dist_metric = partial(power_dist, power=2, ws=ws)

    # calculate the cost cutoffs
    if use_overlap:
        common_cost_cutoff_max =  - np.log(0.01) 
    else:
        common_cost_cutoff_max = 0
    dist_cost_coef = dist_weight+weighted_dist_weight 
    if second_only:
        track_cost_cutoff = config["max_distance"] ** 2 
    else:
        track_cost_cutoff = dist_cost_coef * config["max_distance"] ** 2 + common_cost_cutoff_max
    gap_closing_cost_cutoff = dist_cost_coef * config["gap_closing_max_distance"] ** 2 + common_cost_cutoff_max
    if division:
        splitting_cost_cutoff = dist_cost_coef * config["splitting_max_distance"] ** 2 + common_cost_cutoff_max
    else:
        splitting_cost_cutoff = False

    # define the LapTrack object and return
    return LapTrack(
        track_dist_metric=track_dist_metric,
        track_cost_cutoff=track_cost_cutoff,

        gap_closing_max_frame_count=config["gap_closing"],
        gap_closing_dist_metric=get_metric(
            config["gap_closing_max_distance"], "iou" if use_iou else "ratio_2"
        ),
        gap_closing_cost_cutoff=gap_closing_cost_cutoff,
        splitting_dist_metric=get_metric(
            config["splitting_max_distance"], "ratio_2"
        ) if division else "none",
        splitting_cost_cutoff=splitting_cost_cutoff,

        alternative_cost_percentile=config["alternative_cost_percentile"]
        if "alternative_cost_percentile" in config.keys()
        else 90,
    )

# the main function to launch the program
def main2(
    base_dirs,  # separaed by ":"
    results_dir,
    prefix,
    yaml_path,
    *,
    fitting_use_ratio=None,
    division_fitting_use_ratio=None,
    max_dist_quantile=0.999,
    max_dist_quantile_factor=1.5,
    score_target="true_positive_rate",
    test_base_dirs=None,  # separated by ":"
    second_only=False, 
    use_overlap=True,
    use_euclidean_dist=False,
    use_weighted_dist=False,
    use_iou=False,
    change_percentile=False,
    fix_configs=False,
):

    config = {}
    initial_configs = [{}]

    if change_percentile:
        config.update(
            {
                "alternative_cost_percentile": tune.uniform(90, 100),
            }
        )
        initial_configs[0].update(
            {
                "alternative_cost_percentile": 90,
            }
        )

    if use_overlap:
        config.update({"nll_offset": tune.uniform(0.01, 0.5),})
        initial_configs[0].update({"nll_offset": 0.01,})
    if use_euclidean_dist:
        config.update({"dist_weight": tune.uniform(0, 1),})
        initial_configs[0].update({"dist_weight": 0,})
    if use_weighted_dist:
        config.update({"weighted_dist_weight": tune.uniform(0, 1),})
        initial_configs[0].update({"weighted_dist_weight": 0,})

    division = read_yaml(yaml_path)["division"]
    distance_keys = [
        "max_distance",
        "gap_closing_max_distance",
    ]
    if division:
        distance_keys.append(
            "splitting_max_distance"
        )

    common_params = dict(
        model_include_drift=True,
        divide_training=False,
        guess_dist_cutoff_keys=distance_keys,
        read_overlap_df=True
    )

    if fix_configs:
        common_params.update(dict(
            fix_configs=distance_keys,
        ))
    
    lap_name = f"{LAP_NAME}_quantile{max_dist_quantile:.3f}_factor{max_dist_quantile_factor:.2f}"
    if second_only:
        lap_name = lap_name + "_second_only"
    if use_overlap:
        lap_name = lap_name + "_overlap"
    if use_euclidean_dist:
        lap_name = lap_name + "_euclidean_dist"
    if use_weighted_dist:
        lap_name = lap_name + "_weighted_dist"
    if use_iou:
        lap_name = lap_name + "_use_iou"
    if fix_configs:
        lap_name = lap_name + "_fix_configs"
    if change_percentile:
        lap_name = lap_name + "_change_percentile"

    return main(
        base_dirs=base_dirs, 
        results_dir=results_dir,
        prefix=prefix,
        yaml_path=yaml_path,
        fitting_use_ratio=fitting_use_ratio,
        division_fitting_use_ratio=division_fitting_use_ratio,
        score_target=score_target,
        test_base_dirs=test_base_dirs,  
        max_dist_quantile=max_dist_quantile,
        max_dist_quantile_factor=max_dist_quantile_factor,
        lap_name=lap_name,
        get_tracker=partial(get_tracker,
            second_only=second_only, 
            use_overlap=use_overlap,
            use_euclidean_dist=use_euclidean_dist,
            use_weighted_dist=use_weighted_dist,
            use_iou=use_iou,
        ),
        config=config,
        initial_configs=initial_configs,
        initial_configs_csv_pattern=get_initial_configs_csv_pattern,
        **common_params,
    )


if __name__ == "__main__":
    Fire(main2)


