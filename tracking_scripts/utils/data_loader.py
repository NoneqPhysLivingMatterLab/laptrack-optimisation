import numpy as np
from glob import glob
from natsort import natsorted
from os import path
from skimage.io import imread
import pandas as pd
from matplotlib import pyplot as plt
from ray import tune


def read_data(base_dir, regionprop_keys):
    GT_TRA_files = natsorted(glob(path.join(base_dir, "02_GT/TRA/man_track*.tif")))
    GT_TRA_images = np.array(list(map(imread, GT_TRA_files)))
    GT_track_df = pd.read_csv(
        path.join(base_dir, "02_GT/TRA/man_track.txt"),
        names=["track", "first_frame", "last_frame", "parent_track"],
        sep=" ",
    )
    regionprops_df = pd.read_csv(path.join(base_dir, "regionprops.csv"), index_col=0)
    regionprops_df = regionprops_df[["frame", "track"] + list(regionprop_keys)]
    regionprops_df = regionprops_df.dropna()
    max_frame = regionprops_df["frame"].max()
    coords = [None for _ in range(max_frame + 1)]
    track_labels = [None for _ in range(max_frame + 1)]

    for frame, grp in regionprops_df.groupby("frame"):
        df = grp.sort_values("track")
        coords[frame] = np.concatenate(
            [df[regionprop_keys].values, [[frame]] * len(df)], axis=1
        )
        assert len(set(df["track"].values)) == len(grp)
        track_labels[frame] = list(
            int(v) if np.isfinite(v) else None for v in df["track"].values
        )

    assert all([c is not None for c in coords])

    def to_index(frame, track_label):
        return track_labels[frame].index(track_label)

    true_edges = []
    for i, row in GT_track_df.iterrows():
        frame = row["first_frame"]
        while frame < row["last_frame"]:  # correct
            try:
                next_index = to_index(frame + 1, row["track"])
                frame_add = 1
            except ValueError:
                print("Missing frame:", frame + 1, row["track"])
                next_index = to_index(frame + 2, row["track"])
                frame_add = 2
            true_edges.append(
                (
                    (frame, to_index(frame, row["track"])),
                    (frame + frame_add, next_index),
                )
            )
            frame += frame_add
        if row["parent_track"] > 0:
            true_edges.append(
                (
                    (
                        row["first_frame"] - 1,
                        to_index(row["first_frame"] - 1, row["parent_track"]),
                    ),
                    (row["first_frame"], to_index(row["first_frame"], row["track"])),
                )
            )
    return coords, track_labels, true_edges, GT_TRA_images


def visualize_tracks(edges, coords, frame):
    plt.plot(*coords[frame].T[:2], "or")
    plt.plot(*coords[frame + 1].T[:2], "ob")
    for edge in edges:
        (frame1, ind1), (frame2, ind2) = edge
        if (frame1 == frame and frame1 < frame2) or (
            frame2 == frame and frame2 < frame1
        ):
            edge_coords = np.array(
                [coords[frame1][ind1][:2], coords[frame2][ind2][:2]]
            ).T
            plt.plot(*edge_coords, "-k")
    plt.show()


def guess_drift(coords, true_edges, use_drift):
    drifts = []
    for e in true_edges:
        (f1, i1), (f2, i2) = e
        if f1 > f2:
            (f2, i2), (f1, i1) = e
        drifts.append(coords[f2][i2] - coords[f1][i1])
    m = np.mean(drifts, axis=0)
    s = np.std(drifts, axis=0)
    if use_drift:
        init_config = {
            "drift_x": m[0],
            "drift_y": m[1],
        }
        config = {
            "drift_x": tune.uniform(m[0] - s[0], m[0] + s[0]),
            "drift_y": tune.uniform(m[1] - s[1], m[1] + s[1]),
        }
    else:
        init_config = {}
        config = {
            "drift_x": 0,
            "drift_y": 0,
        }
    return init_config, config


def guess_drift2(coordss, true_edgess, use_drift):
    drifts = []
    for coords, true_edges in zip(coordss, true_edgess):
        for e in true_edges:
            (f1, i1), (f2, i2) = e
            if f1 > f2:
                (f2, i2), (f1, i1) = e
            drifts.append(coords[f2][i2] - coords[f1][i1])
    m = np.mean(drifts, axis=0)
    s = np.std(drifts, axis=0)
    if use_drift:
        init_config = {
            "drift_x": m[0],
            "drift_y": m[1],
        }
        config = {
            "drift_x": tune.uniform(m[0] - s[0], m[0] + s[0]),
            "drift_y": tune.uniform(m[1] - s[1], m[1] + s[1]),
        }
    else:
        init_config = {}
        config = {
            "drift_x": 0,
            "drift_y": 0,
        }
    return init_config, config
