#!/usr/bin/env bash

DESCRIPTION="
DAGE loss applied to the model logits with varying batch size. 
The theory for graph embedding assumes that all data is available when setting up the weight matrix. 
Here, we test how the method reacts to batches of data."

EXPERIMENT_ID=dage_batch_size
METHOD=dage_full
ARCHITECTURE=two_stream_pair_logits
MODEL_BASE=none
FEATURES=vgg16
EPOCHS=20
AUGMENT=0
LOSS_WEIGHTS_EVEN=0
LOSS_ALPHA=0.75
GPU_ID=1

DIR_NAME=./runs/$METHOD/$EXPERIMENT_ID

mkdir $DIR_NAME -p
echo $DESCRIPTION > $DIR_NAME/description.txt

for BATCH_SIZE in 64 256 1024 #4096 16
do
    for SEED in 0 1 2 3 4
    do
        for SOURCE in A D W
        do
            for TARGET in A D W
            do
                if [ $SOURCE != $TARGET ]
                then
                    python3 run.py --method $METHOD --architecture $ARCHITECTURE --source $SOURCE --target $TARGET --model_base $MODEL_BASE --epochs $EPOCHS --seed $SEED --augment $AUGMENT --loss_alpha $LOSS_ALPHA --loss_weights_even $LOSS_WEIGHTS_EVEN --batch_size $BATCH_SIZE --gpu_id $GPU_ID --features $FEATURES --experiment_id $EXPERIMENT_ID
                fi
            done
        done
    done
done