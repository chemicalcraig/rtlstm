#!/bin/bash
#bash script to run inferrence using model from train_benchmark.py
#Be sure to copy inputs/predict.json into working directory and modify accordingly
LSTMDIR=/home/craig/programs/rtlstm
NWCALCNAME=h2_plus_rttddft
NDENS=20
python ${LSTMDIR}/src/extract_densities.py --n-densities $NDENS ${NWCALCNAME}.rt_restart
python ${LSTMDIR}/src/predict_config.py predict.json
