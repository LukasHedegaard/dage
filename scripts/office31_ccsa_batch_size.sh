#!/usr/bin/env bash

DESCRIPTION="
CCSA loss applied to the model embeddings with varying batch size. 
We experience that large batch size deteriorates performance on DAGE. 
This experiment is used to check if it is a general phenomenon for the NN architecture or if it relates to DAGE specifically."

EXPERIMENT_ID=ccsa_batch_size
METHOD=ccsa
ARCHITECTURE=two_stream_pair_embeds
MODEL_BASE=none
FEATURES=vgg16
EPOCHS=20
AUGMENT=0
LOSS_WEIGHTS_EVEN=0
LOSS_ALPHA=0.25
GPU_ID=0

DIR_NAME=./runs/$METHOD/$EXPERIMENT_ID

mkdir $DIR_NAME -p
echo $DESCRIPTION > $DIR_NAME/description.txt

for BATCH_SIZE in 16 256 4096 
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