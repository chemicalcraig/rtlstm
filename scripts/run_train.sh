#!/bin/bash
#bash script to set up training with simple bootstrap
#Extract necessary information:
# rt_data/field.dat
# data/_overlap.npy
# data/_overlap.txt
# data/_overlap.npy
LSTMDIR=/home/craig/programs/rtlstm
NWCALCNAME=h2_plus_rttddft.out
mkdir densities && mkdir rt_data && mkdir data && mkdir perm
python ${LSTMDIR}/src/rtparse.py $NWCALCNAME && mv *.dat rt_data/
python ${LSTMDIR}/src/get_overlap.py $NWCALCNAME && mv *overlap.npy data/ && mv *overlap.txt data/ && mv *overlap.py data/

#Set up and train. Be sure to copy inputs/train.json into working directory and edit accordingly
python ${LSTMDIR}/src/sync_field.py --densities perm/ --field rt_data/field.dat --out data/field.npy
python ${LSTMDIR}/src/aggregate_data.py --dir perm/ --out densities/density_series.npy
python ${LSTMDIR}/src/train_benchmark.py --config train.json
