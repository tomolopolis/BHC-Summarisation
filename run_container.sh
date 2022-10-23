#!/bin/bash
# runs the container in offline mode - remove --env-file otherwise
docker run -it --env-file offline-env.env --gpus all \
  -v `pwd`/../mimic_summ_data/:/home/mimic_summ_data/ \
  -v `pwd`/../cg_summ_data/:/home/cg_summ_data/ \
  -v `pwd`/experiment_cfg/:/home/experiment_cfg/ \
  -v `pwd`/model-outputs/:/home/model-outputs/ \
  tsearle/summ_exp-bart:latest bash
