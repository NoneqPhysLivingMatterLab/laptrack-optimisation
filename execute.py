#! /usr/bin/env python3

from itertools import product
from multiprocessing import Pool
from subprocess import call
from fire import Fire
import yaml
import os
from os import path

qsub_command="""#!/bin/sh
#PJM -j
#PJM -L rscunit=bwmpc
#PJM -L rscgrp=batch
#PJM -L elapse=24:00:00
#PJM -L vnode=1
#PJM -L vnode-core=40
#PJM -L proc-crproc=4096

. ${{HOME}}/.miniconda3/etc/profile.d/conda.sh
conda activate optlaptrack
ray start --head --port=6379 --num-cpus=10 
sleep 10

{command}
"""

def main(
        conditions_yaml_path, 
        n_jobs = 4, 
        include_programs = None, 
        include_conditions=None, 
        repeat=1,
        qsub=False,
    ):

    script_dir = path.dirname(path.abspath(__file__))
    os.chdir(script_dir)


    with open(conditions_yaml_path) as f:
        params = yaml.safe_load(f)
    programss = params["programs"]
    conditions = params["conditions"]

    if include_programs:
        if not isinstance(include_programs, tuple):
            include_programs = include_programs.split(",")
        all_programs = sum(programss,[])
        assert all([p in all_programs for p in include_programs]), str(include_programs)+str(all_programs)
        programss = [
            [k for k in programs if k in include_programs]
            for programs in programss]

    if include_conditions:
        if not isinstance(include_conditions, tuple):
            include_conditions = include_conditions.split(",")
        assert all([c in conditions for c in include_conditions])
        conditions = {k:v for k,v in conditions.items() if k in include_conditions}

    for programs in programss:
        commands = []
        for name, condition in conditions.items():
            for program in programs:
                prog_params=params["program_params"].get(program,{})
                prog_keys = prog_params.keys()
                prog_valuess = prog_params.values()

                cond_keys = condition.keys()
                cond_values = condition.values()
                cond_values_count = [len(v) if isinstance(v, list) else 1 for v in cond_values]
                max_values_count = max(cond_values_count)
                assert set(cond_values_count) == set([1, max_values_count]), f"{cond_values_count}"
                cond_valuess = list(zip(*[[v]*max_values_count if not isinstance(v, list) else v for v in cond_values]))

                for _prog_values in list(product(*prog_valuess))[::-1]:
                    prog_args = sum([[f"--{k}",str(v)] for k,v in zip(prog_keys,_prog_values)],[])
                    for _cond_values in cond_valuess:
                        cond_args = sum([[f"--{k}",str(v)] for k,v in zip(cond_keys,_cond_values)],[])
                        commands.append(["nice","-n","19","python3", "tracking_scripts/"+program, *prog_args, *cond_args])
        commands = commands * repeat
        print(commands)
        if not qsub:
            pool = Pool(processes=n_jobs)
            pool.map(call, commands) 
            pool.close()
            pool.join()
        else:
            for j,c in enumerate(commands):
                qsub_command_connected = qsub_command.format(
                   command=" ".join(c[3:]) 
                )
                with open(f"qsub/qsub_job{j:03d}.sh","w") as f:
                    f.write(qsub_command_connected)

if __name__ == "__main__":
    Fire(main)