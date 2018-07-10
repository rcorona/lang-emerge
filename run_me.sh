DATA_PATH='data/toy64_split_0.8.json'
SAVE_PREFIX='models/testing'
VISIBLE_CUDA_DEVICES=0
# script to run the program
python3 train.py -learningRate 0.01 -hiddenSize 50 -batchSize 1000 \
                -imgFeatSize 20 -embedSize 20\
                -dataset $DATA_PATH\
                -aOutVocab 64 -qOutVocab 64\
		-remember -save_prefix $SAVE_PREFIX 
