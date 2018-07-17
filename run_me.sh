#!/bin/bash

# First load modules and activate virtual environment. 
source $HOME/.bashrc
source $HOME/env/bin/activate

# Now run an experiment on each GPU.
PARENT_PATH=$HOME/UvA/emergent_language/submodules/lang-emerge
DATA_PATH=$PARENT_PATH/data/toy64_split_0.8.json
EXP='VSize'
SAVE_PREFIX=$PARENT_PATH/models/$EXP
OUT_PATH=$PARENT_PATH/output/$EXP

export CUDA_VISIBLE_DEVICES=0

python3 $PARENT_PATH/train.py -learningRate 0.01 -hiddenSize 100 -batchSize 1000 \
                -imgFeatSize 20 -embedSize 20\
                -dataset $DATA_PATH\
                -aOutVocab 64 -qOutVocab 64\
		-remember -save_prefix $SAVE_PREFIX > ${OUT_PATH}_64.out 2>&1 & 

export CUDA_VISIBLE_DEVICES=1

python3 $PARENT_PATH/train.py -learningRate 0.01 -hiddenSize 100 -batchSize 1000 \
                -imgFeatSize 20 -embedSize 20\
                -dataset $DATA_PATH\
                -aOutVocab 32 -qOutVocab 32\
		-remember -save_prefix $SAVE_PREFIX > ${OUT_PATH}_32.out 2>&1 &

export CUDA_VISIBLE_DEVICES=2

python3 $PARENT_PATH/train.py -learningRate 0.01 -hiddenSize 100 -batchSize 1000 \
                -imgFeatSize 20 -embedSize 20\
                -dataset $DATA_PATH\
                -aOutVocab 16 -qOutVocab 16\
		-remember -save_prefix $SAVE_PREFIX > ${OUT_PATH}_16.out 2>&1 &

export CUDA_VISIBLE_DEVICES=3

python3 $PARENT_PATH/train.py -learningRate 0.01 -hiddenSize 100 -batchSize 1000 \
                -imgFeatSize 20 -embedSize 20\
                -dataset $DATA_PATH\
                -aOutVocab 8 -qOutVocab 8\
		-remember -save_prefix $SAVE_PREFIX > ${OUT_PATH}_8.out 2>&1 &


# Wait for each experiment to end. 
wait

# Finally, deactivate environment. 
deactivate
