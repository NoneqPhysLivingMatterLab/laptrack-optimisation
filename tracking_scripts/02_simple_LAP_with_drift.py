from laptrack import LapTrack
from functools import partial
from pathlib import Path
import os
from ray import tune
from fire import Fire

from utils.common import power_dist_with_drift, main

LAP_NAME = "02_Simple_LAP_with_drift"

config = {}
initial_configs = [{}]


def get_tracker(config, division, regionprop_keys=None):
    ws = [1, 1] + [0] * (len(regionprop_keys) - 1)
    dist_power = 2
    return LapTrack(
        track_cost_cutoff=config["max_distance"] ** dist_power,
        splitting_cost_cutoff=config["splitting_max_distance"] ** dist_power,
        gap_closing_cost_cutoff=config["gap_closing_max_distance"] ** dist_power,
        gap_closing_max_frame_count=config["gap_closing"],
        track_dist_metric=partial(
            power_dist_with_drift,
            ws=ws,
            power=dist_power,
            drift_x=config["drift_x"],
            drift_y=config["drift_y"],
        ),
        splitting_dist_metric=partial(
            power_dist_with_drift,
            ws=ws,
            power=dist_power,
            drift_x=config["drift_x"],
            drift_y=config["drift_y"],
        ),
    )


main = partial(
    main,
    lap_name=LAP_NAME,
    get_tracker=get_tracker,
    config=config,
    initial_configs=initial_configs,
    model_include_drift=True,
    guess_dist_cutoff_keys=[
        "max_distance",
        "splitting_max_distance",
        "gap_closing_max_distance",
    ],
)


if __name__ == "__main__":
    Fire(main)
