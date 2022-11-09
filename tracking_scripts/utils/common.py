import pickle
import numpy as np
from numpy.linalg import norm
from ray import tune
import yaml
from .data_loader import guess_drift2, read_data
import os
from .ray_tune_search import ray_tune_search
from pathlib import Path
from os import path
import pandas as pd
from glob import glob
import networkx as nx
from laptrack.scores import calc_scores
from datetime import datetime
import random


############# distance functions #############


def power_dist(c1s, c2s, ws, power):
    return norm((c1s - c2s) * np.array(ws)) ** power


def power_dist_with_drift(c1s, c2s, ws, power, drift_x, drift_y):
    frame1, frame2 = c1s[-1], c2s[-1]
    if frame1 > frame2:  # ensure frame1 < frame2
        _tmp = c2s
        c2s = c1s
        c1s = _tmp
    drift = np.array([drift_x, drift_y] + [0] * (len(ws) - 2))
    return norm((c1s - c2s + drift) * np.array(ws)) ** power


############## networkx tree conversion utilities ###############


def to_tree(coords, edges):
    tree = nx.DiGraph()
    tree.add_nodes_from(
        sum([[(f, i) for i in range(len(coords[f]))] for f in range(len(coords))], [])
    )
    edges = [(n1, n2) if n1[0] < n2[0] else (n2, n1) for n1, n2 in edges]
    tree.add_edges_from(edges)
    return tree


def to_tree2(nodes, edges):
    tree = nx.DiGraph()
    tree.add_nodes_from(nodes)
    edges = [(n1, n2) if n1[0] < n2[0] else (n2, n1) for n1, n2 in edges]
    tree.add_edges_from(edges)
    return tree


############## filtering subset of edges ###############


def get_fit_edges(coords, true_edges, fitting_use_ratio, division_fitting_use_ratio):
    tree = to_tree(coords, true_edges)
    if fitting_use_ratio:
        nodes = list(tree.nodes)
        if not division_fitting_use_ratio:
            inds = np.random.choice(
                len(nodes), int(len(nodes) * fitting_use_ratio), replace=False
            )
            fit_nodes = set([nodes[i] for i in inds])
            fit_edges = [e for e in true_edges if e[0] in fit_nodes]
        else:
            fit_nodes = []
            fit_edges = []
            children_counts = np.array([len(list(tree.successors(n))) for n in nodes])
            for target_children_count, target_ratio in zip(
                [1, 2], [fitting_use_ratio, division_fitting_use_ratio]
            ):
                _nodes = [
                    n
                    for n, cond in zip(nodes, children_counts == target_children_count)
                    if cond
                ]
                inds = np.random.choice(
                    len(_nodes), int(len(_nodes) * target_ratio), replace=False
                )
                _fit_nodes = set([_nodes[i] for i in inds])
                fit_nodes.extend(_fit_nodes)
                fit_edges.extend([e for e in true_edges if e[0] in _fit_nodes])
        return fit_nodes, fit_edges
    else:
        return tree.nodes(), true_edges.copy()


################ misc ########################


def read_yaml(yaml_path):
    with open(yaml_path) as f:
        yaml_params = yaml.safe_load(f)
    return yaml_params


################# main function to test tracking methods ########################


def main(
    base_dirs,  # separaed by ":"
    results_dir,
    prefix,
    yaml_path,
    lap_name,
    get_tracker,
    config,
    initial_configs,
    *,
    initial_configs_csv_pattern=None,
    discrete_configs={},
    guess_dist_cutoff_keys=None,
    fix_configs=[],
    only_division_configs=[],
    model_include_drift=False,
    fitting_use_ratio=None,
    division_fitting_use_ratio=None,
    divide_training=False,
    config_update=None,
    params_update=None,
    coords_update=None,
    score_target="true_positive_rate",
    test_base_dirs=None,  # separated by ":"
    max_dist_quantile=0.999,
    max_dist_quantile_factor=1.5,
    read_overlap_df=False
):

    base_dirs = base_dirs.split(":")
    if test_base_dirs is not None:
        test_base_dirs = test_base_dirs.split(":")

    run_params = dict(locals())
    prefix2 = lap_name + "_" + prefix
    datetime_str = datetime.strftime(datetime.now(), "%Y%m%d%H%M%S") + f"_{random.randint(0,100):03d}"
    name_id = f"{prefix2}_{datetime_str}"

    ################### read yaml params ###################
    yaml_params = read_yaml(yaml_path)
    regionprop_keys = yaml_params["regionprop_keys"]
    single_shot_count = yaml_params["single_shot_count"]
    iterations = yaml_params["iterations"]
    use_drift = yaml_params["drift"]
    division = yaml_params["division"]

    ################### read best parameter from previous run ###################
    initial_configs2 = initial_configs.copy()
    if initial_configs_csv_pattern:
        if callable(initial_configs_csv_pattern):
            initial_configs_csv_pattern = initial_configs_csv_pattern(
                yaml_params, prefix
            )
        print(path.join(results_dir, initial_configs_csv_pattern))
        df = pd.concat(
            [
                pd.read_csv(f)
                for f in glob(path.join(results_dir, initial_configs_csv_pattern))
            ]
        )
        used_keys = set(
            list(config.keys()) + (guess_dist_cutoff_keys or []) + list(fix_configs)
        )
        if model_include_drift:
            used_keys = used_keys.union(["drift_x", "drift_y"])
        previous_configs = {
            k[7:]: v
            for k, v in df.sort_values(score_target)
            .iloc[-1][[k for k in df.columns if "config" in k]]
            .to_dict()
            .items()
            if k[7:] in used_keys
        }
        assert all([k in previous_configs.keys() for k in fix_configs])
        fixed_previous_configs = {
            k: v for k, v in previous_configs.items() if k in fix_configs
        }
        initial_previous_configs = {
            k: v for k, v in previous_configs.items() if k not in fix_configs
        }
        initial_configs2[0].update(initial_previous_configs)
    else:
        fixed_previous_configs = {}

    initial_configs = initial_configs2.copy()

    ################## make directory for results ##################
    plots_dir = Path(results_dir) / "plots"
    fitting_data_dir = Path(results_dir) / "fitting_data"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(fitting_data_dir, exist_ok=True)

    ################## read data ##################
    print("read data ...")
    data = [read_data(base_dir, regionprop_keys) for base_dir in base_dirs]
    coordss,  track_labelss, true_edgess, GT_TRA_imagess = zip(*data)
    fit_nodess, fit_edgess = zip(
        *[
            get_fit_edges(
                coords, true_edges, fitting_use_ratio, division_fitting_use_ratio
            )
            for coords, true_edges in zip(coordss, true_edgess)
        ]
    )
    print(len(fit_nodess[0]))
    if read_overlap_df:
        overlap_dfs=[]
        coordss2=[]
        for coords, track_labels, base_dir in zip(coordss,track_labelss,base_dirs):
            # include track label at the last dimension
            coords = [np.concatenate([c,np.array(tl)[:,np.newaxis]],axis=1) 
                            for c,tl in zip(coords,track_labels)]
            coordss2.append(coords)
            df=pd.read_csv(path.join(base_dir,"02_GT/TRA/overlaps.csv"),index_col=False)
            df=df.set_index(["frame","label1","label2"])
            overlap_dfs.append(df)
        coordss=coordss2

    ################## update the drift parameters ##################
    print("update drift params ...")
    if model_include_drift:
        drift_init_config, drift_config = guess_drift2(coordss, fit_edgess, use_drift)
        initial_configs[0].update(drift_init_config)
        config.update(drift_config)
        if use_drift:
            drift_init = np.array(
                [drift_init_config["drift_x"], drift_init_config["drift_y"]]
            )
        else:
            drift_init = np.zeros(2)
    else:
        drift_init = np.zeros(2)

    ################## update the dist cutoff ##################
    print("update cutoff params ...")
    if guess_dist_cutoff_keys:
        distss = []
        for coords, fit_edges in zip(coordss, fit_edgess):
            edge_coords = np.array(
                [
                    (coords[e[0][0]][e[0][1]], coords[e[1][0]][e[1][1]])
                    for e in fit_edges
                ]
            )
            dists = np.linalg.norm(
                edge_coords[:, 0, :2] - edge_coords[:, 1, :2] + drift_init, axis=1
            )
            distss.append(dists)
        qs = np.quantile(np.concatenate(distss), [0.5, 0.9, max_dist_quantile])
        max_distance = qs[2] * max_dist_quantile_factor
        for k in guess_dist_cutoff_keys:
            config[k] = tune.uniform(qs[0], max_distance)
            initial_configs[0][k] = qs[1]
    else:
        max_distance = None

    ################## update the coords and config ##################
    print("update coords ...")
    if callable(coords_update):
        coordss = [
            coords_update(coords, max_distance, yaml_params) for coords in coordss
        ]
    if callable(config_update):
        config, initial_configs = config_update(
            config, initial_configs, regionprop_keys
        )

    ################### organize configs ###################
    for k in fix_configs:
        print(fixed_previous_configs)
        config[k] = fixed_previous_configs[k]

    ################## update the parameters (can train models, etc...) ##################
    print("update parameters ...")
    if callable(params_update):
        if read_overlap_df:
            base_tracker_params = params_update(
                coordss,
                fit_nodess,
                fit_edgess,
                overlap_dfs,
                plots_dir,
                name_id,
                use_drift,
                config=config,
                max_distance=max_distance,
            )
        else:
            base_tracker_params = params_update(
                coordss,
                fit_nodess,
                fit_edgess,
                plots_dir,
                name_id,
                use_drift,
                config=config,
                max_distance=max_distance,
            )
    else:
        base_tracker_params = dict()

    if not divide_training:
        ################## define the fitting score function ##################
        params_update_framesss = [[[] for _ in range(len(coordss))]]
        tracker_paramss = [base_tracker_params]
    else:
        ################## update the parameters (can train models, etc...) for separated datasets ##################
        # use_frames dimension  (get_tracker_paramss shares this dimension)
        # base_dir dimension
        # times
        if divide_training is True:
            params_update_framesss = (
                [np.arange(0, len(coords), 2) for coords in coordss],
                [np.arange(1, len(coords), 2) for coords in coordss],
            )
        else:
            raise ValueError("general divide_training not implemented")

        tracker_paramss = []
        if callable(params_update):
            for j, params_update_framess in enumerate(params_update_framesss):
                fit_nodess_train = [
                    [n for n in fit_nodes if n[0] in params_update_frames]
                    for fit_nodes, params_update_frames in zip(
                        fit_nodess, params_update_framess
                    )
                ]
                assert 0 < len(fit_nodess_train[0])
                assert len(fit_nodess_train[0]) < len(fit_nodess[0])
                fit_edgess_train = [
                    [e for e in fit_edges if e[0][0] in params_update_frames]
                    for fit_edges, params_update_frames in zip(
                        fit_edgess, params_update_framess
                    )
                ]
                assert 0 < len(fit_edgess_train[0])
                assert len(fit_edgess_train[0]) < len(fit_edgess[0])
                if read_overlap_df:
                    _tracker_params = params_update(
                        coordss,
                        fit_nodess_train,
                        fit_edgess_train,
                        overlap_dfs,
                        plots_dir,
                        name_id + f"_params_update_frames{j}",
                        use_drift,
                        config=config,
                        max_distance=max_distance,
                    )
                else:
                    _tracker_params = params_update(
                        coordss,
                        fit_nodess_train,
                        fit_edgess_train,
                        plots_dir,
                        name_id + f"_params_update_frames{j}",
                        use_drift,
                        config=config,
                        max_distance=max_distance,
                    )
                tracker_paramss.append(_tracker_params)
        else:
            tracker_paramss = [dict()] * len(params_update_framesss)

    ################## define the fitting score function ##################
    def calc_fitting_score(config, report=True):
        score_dicts = []
        for _tracker_params, params_update_framess in zip(
            tracker_paramss, params_update_framesss
        ):
            for j, (coords, fit_edges, params_update_frames) in enumerate(zip(
                coordss, fit_edgess, params_update_framess
            )):
                if read_overlap_df:
                    lt = get_tracker(
                        config,
                        regionprop_keys=regionprop_keys,
                        division=division,
                        overlap_df=overlap_dfs[j],
                        **_tracker_params,
                    )
                else:
                    lt = get_tracker(
                        config,
                        regionprop_keys=regionprop_keys,
                        division=division,
                        **_tracker_params,
                    )
                track_tree = lt.predict(coords)
                predicted_edges = list(track_tree.edges())
                include_frames = [
                    i for i in range(len(coords)) if i not in params_update_frames
                ]
                # requires all the edges to calculate the track overlap scores (target effectiveness and track purity)
                score_dict = calc_scores(
                    fit_edges, predicted_edges, 
                    include_frames=include_frames, 
                    track_scores=False
                )
                score_dicts.append(score_dict)
        keys = score_dicts[0].keys()
        score_dict = {k: np.mean([d[k] for d in score_dicts]) for k in keys}
        if report:
            tune.report(**score_dict)

    ################### organize configs ###################
    for k in fix_configs:
        print(fixed_previous_configs)
        config[k] = fixed_previous_configs[k]
        del initial_configs[0][k]
    if not division:
        for k in only_division_configs:
            del config[k]
            del initial_configs[0][k]

    ################### test run fitting ###################
    print("test run ...")
    test_config = initial_configs[0].copy()
    for k, v in discrete_configs.items():
        test_config.update({k: v[0]})
    test_config.update({"gap_closing": 0})
    test_config.update(fixed_previous_configs)
    if model_include_drift and not use_drift:
        test_config.update(
            {
                "drift_x": 0,
                "drift_y": 0,
            }
        )
    calc_fitting_score(test_config, report=False)

    ################### run fitting ###################
    print("production run ...")
    print(config)
    print(initial_configs)
    # production run
    all_analysis_df, name_id = ray_tune_search(
        name_id,
        calc_fitting_score,
        config,
        initial_configs.copy(),
        prefix2,
        single_shot_count,
        iterations,
        results_dir,
        plots_dir,
        score_target=score_target,
        discrete_configs=discrete_configs,
    )
    os.makedirs(fitting_data_dir / name_id, exist_ok=True)
    np.save(fitting_data_dir / name_id / "coordss.npy", np.array(coordss, dtype=object))
    np.save(
        fitting_data_dir / name_id / "true_edgess.npy",
        np.array(true_edgess, dtype=object),
    )
    np.save(
        fitting_data_dir / name_id / "fit_edgess.npy",
        np.array(fit_edgess, dtype=object),
    )

    ################### save the best config and get tracker ###################
    best_model = all_analysis_df.sort_values(score_target, ascending=False).iloc[0]
    best_model_config = {
        k[7:]: v for k, v in best_model.to_dict().items() if "config" in k
    }
    print(best_model_config)
    print(config)
    print(initial_configs)
    initial_config = initial_configs[0].copy()
    for k, v in discrete_configs.items():
        initial_config.update({k: v[0]})
    initial_config.update({"gap_closing": 1})
    initial_config.update(fixed_previous_configs)
    if model_include_drift and not use_drift:
        initial_config.update(
            {
                "drift_x": 0,
                "drift_y": 0,
            }
        )
    #################### validation fitting ####################
    if test_base_dirs:
        run_paramss = []
        val_coordss = []
        val_true_edgess = []
        val_overlap_dfs = []
        for test_base_dir in test_base_dirs:
            coords, track_labels, true_edges, _ = read_data(test_base_dir, regionprop_keys)
            if read_overlap_df:
            # include track label at the last dimension
                coords = [np.concatenate([c,np.array(tl)[:,np.newaxis]],axis=1) 
                                for c,tl in zip(coords,track_labels)]
                df=pd.read_csv(path.join(test_base_dir,"02_GT/TRA/overlaps.csv"),index_col=False)
                df=df.set_index(["frame","label1","label2"])
                val_overlap_dfs.append(df)
            if coords_update:
                coords = coords_update(coords, max_distance, yaml_params)
            val_coordss.append(coords)
            val_true_edgess.append(true_edges)
            run_params2 = run_params.copy()
            run_params2.update(
                dict(
                    test_base_dir=test_base_dir,
                )
            )
            run_paramss.append(run_params2)
        connected_edgess = [[]] * len(test_base_dirs)

    else:
        run_paramss = [run_params] * len(coordss)
        val_coordss = coordss
        val_true_edgess = true_edgess
        connected_edgess = fit_edgess
        if read_overlap_df:
            val_overlap_dfs = overlap_dfs

    #################### save the validation result ####################

    for i, (val_coords, val_true_edges, connected_edges, run_params2) in enumerate(
        zip(val_coordss, val_true_edgess, connected_edgess, run_paramss)
    ):
        if read_overlap_df:
            lt = get_tracker(
                best_model_config,
                division=division,
                regionprop_keys=regionprop_keys,
                overlap_df=val_overlap_dfs[i],
                **base_tracker_params,
            )
    
            lt_init = get_tracker(
                initial_config,
                division=division,
                regionprop_keys=regionprop_keys,
                overlap_df=val_overlap_dfs[i],
                **base_tracker_params,
            )
        else:
            lt = get_tracker(
                best_model_config,
                division=division,
                regionprop_keys=regionprop_keys,
                **base_tracker_params,
            )
    
            lt_init = get_tracker(
                initial_config,
                division=division,
                regionprop_keys=regionprop_keys,
                **base_tracker_params,
            )

        track_tree = lt.predict(
            val_coords, connected_edges=connected_edges, split_merge_validation=False
        )
        score_dict = calc_scores(
            val_true_edges, list(track_tree.edges()), exclude_true_edges=connected_edges
        )

        track_tree_init = lt_init.predict(
            val_coords, connected_edges=connected_edges, split_merge_validation=False
        )
        score_dict_init = calc_scores(
            val_true_edges,
            list(track_tree_init.edges()),
            exclude_true_edges=connected_edges,
        )

        best_result_dir = Path(results_dir) / "best_results" / name_id / str(i)
        os.makedirs(best_result_dir, exist_ok=True)
        try:
            with open(best_result_dir / "best_tracker.pickle", "wb") as f:
                pickle.dump(lt, f)
        except:
            pass

        with open(best_result_dir / "best_model_config.yaml", "w") as f:
            yaml.dump(best_model_config, f)
        with open(best_result_dir / "best_model_score.yaml", "w") as f:
            yaml.dump(score_dict, f)
        with open(best_result_dir / "best_model_run_params.yaml", "w") as f:
            yaml.dump(run_params2, f)

        with open(best_result_dir / "initial_model_config.yaml", "w") as f:
            yaml.dump(initial_config, f)
        with open(best_result_dir / "initial_model_score.yaml", "w") as f:
            yaml.dump(score_dict_init, f)

        np.save(best_result_dir / "coords.npy", np.array(val_coords, dtype=object))
        np.save(best_result_dir / "true_edges.npy", val_true_edges)
        np.save(best_result_dir / "predicted_edges.npy", list(track_tree.edges()))
        np.save(
            best_result_dir / "predicted_edges_init.npy", list(track_tree_init.edges())
        )
        np.save(best_result_dir / "connected_edges.npy", connected_edges)


def np_array_to_edge_set(edges):
    return set([tuple(map(tuple, e)) for e in edges])

score_name_map = {
    "Jaccard_index": "Connection Jaccard index",
    "true_positive_rate" : "Connection true positive rate",
    "precision" : "Connection precision",
    "mitotic_branching_correctness" : "Mitotic branching correctness",
    "target_effectiveness" : "Target effectiveness",
    "track_purity" : "Track purity",
}
