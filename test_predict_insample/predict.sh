#!/bin/bash
#bash script to run inferrence using model from train_benchmark.py
#PREREQUISITE: copy inputs/predict.json into working directory and modify accordingly
#PREREQUISITE: copy model from training into working directory
LSTMDIR=/home/craig/programs/rtlstm
NWCALCNAME=h2_plus_rttddft
NDENS=100
python ${LSTMDIR}/src/extract_densities.py --n-densities $NDENS perm/${NWCALCNAME}.rt_restart
python ${LSTMDIR}/src/predict_config.py predict.json
