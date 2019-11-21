#!/usr/bin/env bash

DESCRIPTION="DAGE method on two-stream architecture with an aux dense layer. Here, we experiment with varying the size of this aux dense layer, thus varying the target dimensionality of the graph embedding."

EXPERIMENT_ID=dage_aux_dense_low_bs
METHOD=dage_full
ARCHITECTURE=two_stream_pair_aux_dense
MODEL_BASE=none
FEATURES=vgg16
EPOCHS=20
AUGMENT=0
LOSS_WEIGHTS_EVEN=0
LOSS_ALPHA=0.75
BATCH_SIZE=16
GPU_ID=0

DIR_NAME=./runs/$METHOD/$EXPERIMENT_ID

mkdir $DIR_NAME -p
echo $DESCRIPTION > $DIR_NAME/description.txt

for AUX_DENSE_SIZE in 16 31 64
do
    for SEED in 0 1 2 3 4
    do
        for SOURCE in A D W
        do
            for TARGET in A D W
            do
                if [ $SOURCE != $TARGET ]
                then
                    python3 run.py --method $METHOD --architecture $ARCHITECTURE --aux_dense_size $AUX_DENSE_SIZE --source $SOURCE --target $TARGET --model_base $MODEL_BASE --epochs $EPOCHS --seed $SEED --augment $AUGMENT --loss_alpha $LOSS_ALPHA --loss_weights_even $LOSS_WEIGHTS_EVEN --batch_size $BATCH_SIZE --gpu_id $GPU_ID --features $FEATURES --experiment_id $EXPERIMENT_ID
                fi
            done
        done
    done
done