#!/bin/bash
#bash script to set up training with simple bootstrap
#Extract necessary information:
# rt_data/field.dat
# data/_overlap.npy
# data/_overlap.txt
# data/_overlap.npy
LSTMDIR=/home/craig/programs/rtlstm
NWCALCNAME=h2_plus_rttddft.out
CONFIGFILE=train.json
mkdir densities && mkdir rt_data && mkdir data && mkdir perm

echo ">>>>>		"
echo ">>>>>		Parsing Real-Time data"
echo ">>>>>		"
python ${LSTMDIR}/src/rtparse.py $NWCALCNAME && mv *.dat rt_data/
echo ">>>>>		"
echo ">>>>>		Parsing Overlap Matrix, S"
echo ">>>>>		"
python ${LSTMDIR}/src/get_overlap.py $NWCALCNAME && mv *overlap.npy data/ && mv *overlap.txt data/ && mv *overlap.py data/
#Set up and train. Be sure to copy inputs/train.json into working directory and edit accordingly
echo ">>>>>		"
echo ">>>>>		Sync E-field data with restart density matrices"
echo ">>>>>		"
python ${LSTMDIR}/src/sync_field.py --densities perm/ --field rt_data/field.dat --out data/field.npy
echo ">>>>>		"
echo ">>>>>		Aggregating restart density matrices into training data series"
echo ">>>>>		"
python ${LSTMDIR}/src/aggregate_data.py --dir perm/ --out densities/density_series.npy
echo ">>>>>		"
echo ">>>>>		Training with the following input config:"
echo ">>>>>		"
cat $CONFIGFILE
echo ">>>>>		"
echo ">>>>>		Training"
echo ">>>>>		"
#python ${LSTMDIR}/src/train_benchmark.py --config $CONFIGFILE
