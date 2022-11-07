################################################
# the script to calculate tracking scores 
# with the squared-distance-based metric
# using coordinates and features 
################################################

from laptrack import LapTrack
from functools import partial
from ray import tune
from fire import Fire

# power_dist is the weighted distance-power weight with the drift term
from utils.common import power_dist_with_drift, main

LAP_NAME = "03_simple_LAP_with_similarity"

# the search range of the "feature_weight" is set to (0,10)
config = {
    "feature_weight": tune.uniform(0.,10.)
}
# the initial value is set to zero
initial_configs = [{
    "feature_weight":0
}]

def get_tracker(config, division, regionprop_keys=None):
    # used the all dimensions, but those later than the first two dimensions are weighted
    ws = [1, 1] + [config["feature_weight"]] * (len(regionprop_keys) - 2) + [0]
    # use the first two dimension for splitting detection
    ws2 = [1, 1] + [0] * (len(regionprop_keys) - 1)
    # the power is set to be 2 (square)
    dist_power = 2
    track_metric=partial(
            power_dist_with_drift,
            ws=ws,
            power=dist_power,
            drift_x=config["drift_x"],
            drift_y=config["drift_y"],
    )
    return LapTrack(
        track_cost_cutoff=config["max_distance"] ** dist_power,
        splitting_cost_cutoff=config["splitting_max_distance"] ** dist_power if division else False,
        gap_closing_max_frame_count=config["gap_closing"],
        track_dist_metric=track_metric,
        gap_closing_dist_metric=track_metric,
        splitting_dist_metric=partial(
            power_dist_with_drift,
            ws=ws2,
            power=dist_power,
            drift_x=config["drift_x"],
            drift_y=config["drift_y"],
        ) if division else "none",
    )

# read the data from the simple_LAP results
def get_initial_configs_csv_pattern(yaml_params, prefix):
    if yaml_params["drift"]:
        return f"02_Simple_LAP_with_drift_{prefix}*.csv"
    else:
        return f"01_Simple_LAP_{prefix}*.csv"

main = partial(
    main,
    lap_name=LAP_NAME,
    get_tracker=get_tracker,
    config=config,
    initial_configs=initial_configs,
    model_include_drift=True,
    initial_configs_csv_pattern=get_initial_configs_csv_pattern,
    guess_dist_cutoff_keys=[
        "max_distance",
        "splitting_max_distance",
        "segment_connect_max_distance",
    ],
    only_division_configs=["splitting_max_distance"],
)

if __name__ == "__main__":
    Fire(main)
