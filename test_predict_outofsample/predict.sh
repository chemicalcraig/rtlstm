#!/bin/bash
#bash script to run inferrence using model from train_benchmark.py
#PREREQUISITE: copy inputs/predict.json into working directory and modify accordingly
#PREREQUISITE: copy model from training into working directory
mkdir densities
mkdir rt_data 
mkdir data
LSTMDIR=/home/craig/programs/rtlstm
NWCALCNAME=h2_plus_rttddft
NDENS=100
echo ">>>>>		"
echo ">>>>>		Parsing Real-Time data"
echo ">>>>>		"
python ${LSTMDIR}/src/rtparse.py ${NWCALCNAME}.out && mv *.dat rt_data/
echo ">>>>>		"
echo ">>>>>		Parsing Overlap Matrix, S"
echo ">>>>>		"
python ${LSTMDIR}/src/get_overlap.py ${NWCALCNAME}.out && mv *overlap.npy data/ && mv *overlap.txt data/ && mv *overlap.py data/
echo ">>>>>		"
echo ">>>>>		Sync E-field data with restart density matrices"
echo ">>>>>		"
python ${LSTMDIR}/src/sync_field.py --densities perm/ --field rt_data/field.dat --out data/field.npy
python ${LSTMDIR}/src/extract_densities.py --n-densities $NDENS perm/${NWCALCNAME}.rt_restart
python ${LSTMDIR}/src/predict_config.py predict.json
