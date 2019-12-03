#!/usr/bin/env bash

DESCRIPTION="CCSA method trained on pre-extracted vgg16 features."

EXPERIMENT_ID=ccsa_from_feat
METHOD=ccsa
ARCHITECTURE=two_stream_pair_embeds
MODEL_BASE=none
FEATURES=vgg16
EPOCHS=20
AUGMENT=0
LOSS_WEIGHTS_EVEN=0
LOSS_ALPHA=0.25
BATCH_SIZE=16
GPU_ID=0

DIR_NAME=./runs/$METHOD/$EXPERIMENT_ID

mkdir $DIR_NAME -p
echo $DESCRIPTION > $DIR_NAME/description.txt

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
