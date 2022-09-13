from ray import tune
from ray.tune.suggest import BasicVariantGenerator
from ray.tune.suggest.optuna import OptunaSearch
from ray.tune.suggest import ConcurrencyLimiter

# from ray.tune.suggest.nevergrad import NevergradSearch
# from ray.tune.suggest.flaml import CFO
# import nevergrad as ng
from matplotlib import pyplot as plt
from os import path
import numpy as np
from IPython.display import display
from datetime import datetime
import pandas as pd
from itertools import product

# names = ["RandomSearch", "CFO", "OptunaSearch", "NevergradSearch"]
# searchs = [BasicVariantGenerator, CFO, OptunaSearch, NevergradSearch]
names = ["RandomSearch", "OptunaSearch"]
searchs = [BasicVariantGenerator, OptunaSearch]
# names = ["RandomSearch"]
# searchs = [BasicVariantGenerator]
get_additional_configs = lambda single_shot_count: [
    dict(max_concurrent=single_shot_count),
    #        dict(),
    dict(),
    #        dict(
    #            optimizer=ng.optimizers.OnePlusOne,
    #        ),
]


def ray_tune_search(
    name_id,
    calc_fitting_score,
    config,
    initial_configs,
    suffix,
    single_shot_count,
    iterations,
    results_dir,
    plots_dir,
    score_target,
    discrete_configs={},
):
    additional_configs = get_additional_configs(single_shot_count)
    all_dfs = []

    fig_score_dist, ax_score_dist = plt.subplots(1, 1)
    fig_learning, ax_learning = plt.subplots(1, 1)

    discrete_configs.update(dict(gap_closing=[0, 1]))
    discrete_config_product = product(*discrete_configs.values())

    for (name, search, additional_config), discrete_config in product(
        list(zip(names, searchs, additional_configs))[:], discrete_config_product
    ):
        config2 = config.copy()
        for k, v in zip(discrete_configs.keys(), discrete_config):
            config2[k] = v
        search_alg = search(
            points_to_evaluate=initial_configs,
            **additional_config,
        )
        if name not in "RandomSearch":
            search_alg = ConcurrencyLimiter(search_alg, single_shot_count)
        num_samples = single_shot_count * iterations

        analysis = tune.run(
            calc_fitting_score,
            config=config2,
            metric=score_target,
            mode="max",
            search_alg=search_alg,
            num_samples=num_samples,
            #                resources_per_trial={"cpu": single_shot_count*4}
        )
        analysis_df = analysis.results_df.sort_values(by=score_target, ascending=False)
        display(analysis_df.head())
        analysis_df["search_name"] = name
        all_dfs.append(analysis_df)

        label = f"{name}"
        ax_score_dist.hist(
            analysis_df[score_target],
            bins=20,
            range=(0, 1),
            label=label,
            histtype="step",
        )

        df = analysis_df.sort_values("date")[score_target].cummax().values
        try:
            df = [np.max(d) for d in np.split(df, iterations)]
            ax_learning.plot(range(len(df)), df, label=label)
        except ValueError:
            pass

    all_analysis_df = pd.concat(all_dfs).sort_values(score_target, ascending=False)
    all_analysis_df.to_csv(path.join(results_dir, name_id + ".csv"))

    ax_score_dist.set_xlabel(score_target)
    ax_score_dist.set_ylabel("count")
    ax_score_dist.legend()
    fig_score_dist.savefig(
        path.join(plots_dir, name_id + "_score.png"), bbox_inches="tight"
    )

    ax_learning.set_xlabel("iter")
    ax_learning.set_ylabel(f"max {score_target}")
    ax_learning.legend()
    fig_learning.savefig(
        path.join(plots_dir, name_id + "_learning.png"), bbox_inches="tight"
    )

    return all_analysis_df, name_id
