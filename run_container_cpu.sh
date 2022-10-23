#!/bin/bash
# runs the container in offline mode - remove --env-file otherwise
docker run -it --env-file offline-env.env \
  -v `pwd`/mimic_summ_data/:/home/mimic_summ_data/ \
  -v `pwd`/cg_summ_data/:/home/cg_summ_data/ \
  -v `pwd`/experiment_cfg/:/home/experiment_cfg/ \
  -v `pwd`/outputs/:/home/outputs/ \
  -v `pwd`/guidance_experiment_cfg/:/home/guidance_experiment_cfg/ \
  -v `pwd`/guidance-outputs/:/home/guidance-outputs/ \
  tsearle/summ_exp:latest bash
