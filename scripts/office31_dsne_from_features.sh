#!/usr/bin/env bash

DESCRIPTION="dSNE from features."

EXPERIMENT_ID=dsne_from_features
METHOD=dsne

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
                python3 run.py \
                    --gpu_id            0 \
                    --experiment_id     $EXPERIMENT_ID \
                    --source            $SOURCE \
                    --target            $TARGET \
                    --seed              $SEED \
                    --method            $METHOD \
                    --architecture      two_stream_pair_embeds \
                    --model_base        none \
                    --features          vgg16 \
                    --epochs            20 \
                    --batch_size        16 \
                    --augment           0 \
                    --loss_alpha        0.25 \
                    --loss_weights_even 0 \

            fi
        done
    done
done

