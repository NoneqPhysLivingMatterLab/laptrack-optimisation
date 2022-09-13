# Scripts for "LapTrack: Linear assignment particle tracking with tunable metrics"
Parameter optimization for [LapTrack](https://github.com/yfukai/laptrack) with [Ray-Tune](https://www.ray.io/ray-tune).

## Executing the analysis

```bash
conda env create -f conda_env_minimum.yaml
conda activate optlaptrack

python execute.py --n-jobs=1 conditions_drift.yaml # for the cell migration dataset
python execute.py --n-jobs=1 conditions_synthetic.yaml # for the coloured particles dataset
python execute.py --n-jobs=1 conditions_homeostasis.yaml # for the mouse epidermis dataset

cd tracking_scripts
python a1_homeostasis_simple_LAP_baseline_grid.py # to perform grid search for the mouse epidermis dataset
```

## Directories
- **data** the organized datasets
- **tracking_scripts** the tracking and plotting scripts
- **setting_yaml** YAML setting files for tracking scripts
- **results** the tracking results
- **plots** the result plots

## Tracking scripts
Located in `tracking_scripts`, executed via `execute.py`.
- **01_simple_LAP.py** LAP with Euclidean distance only
- **02_simple_LAP_with_drift.py** LAP with Euclidean distance and the drift term
- **03_simple_LAP_with_similarity-simple.py** LAP with the feature Euclidean distance term
- **04_simple_LAP_with_overlap_dist_sum.py** LAP with overlap cost function 

## Plotting scripts
Located in `tracking_scripts`, executed as Jupyter notebooks in, e.g., Visual Studio Code.
- **z1_make_grid_search_plots.py** Generate Fig. 2(b) and S1.
- **z2_CellMigration_summarize_results.py** Generate Fig. 2(d) and S2.
- **z3_synthetic_summarize_results.py** Generate Fig. 2(f) and S3.
- **z4_homeostasis_summarize_results.py** Generate Fig. 2(g).
- **z5_summarize_properties.py** Summarize particle counts.

## Other scripts
Located in `tracking_scripts`.
- **a1_homeostasis_simple_LAP_baseline_grid.py** Performs grid search for the mouse epidermis dataset.

## Datasets
Located in `data`. 
- **CellMigration** Data for the cell migration dataset.
- **homeostasis** Data for the mouse epidermis dataset.
- **synthetic** Data for the coloured particles dataset.

## Results
Located in `results`.
- **CellMigration_use_0.05** Results for the cell migration dataset.
- **homeostasis** Results for the mouse epidermis dataset.
- **homeostasis_grid_search** Results for parameter grid search with the mouse epidermis dataset.
- **synthetic** Results for the coloured particles dataset.

## Interactive annotation example
- The example for interactive annotation with napari is located in [The LapTrack repository](https://github.com/yfukai/laptrack/blob/main/examples/napari_interactive_example.ipynb).

## Credits

- The data in `data/CellMigration` are generated from data in [10.5281/zenodo.6087728](https://doi.org/10.5281/zenodo.6087728), which are distributed with [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/legalcode).
- The data in `data/homeostasis` are generated from data in the [Cell interaction paper repository](https://github.com/NoneqPhysLivingMatterLab/cell_interaction_gnn).
