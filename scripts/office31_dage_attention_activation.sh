#!/usr/bin/env bash

DESCRIPTION="DAGE variant resembling d-SNE."

EXPERIMENT_ID=dage_attention_activation
METHOD=dage_a

DIR_NAME=./runs/$METHOD/$EXPERIMENT_ID

mkdir $DIR_NAME -p
echo $DESCRIPTION > $DIR_NAME/description.txt

for ACTIVATION in softmax #tanh
do
    for SEED in 0 #1 2 3 4
    do
        for SOURCE in A #D W
        do
            for TARGET in A D #W
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
                        --architecture      two_stream_pair_embeds_attention_mid_classwise \
                        --model_base        none \
                        --features          vgg16 \
                        --epochs            20 \
                        --batch_size        16 \
                        --augment           0 \
                        --loss_alpha        0.75 \
                        --loss_weights_even 0 \
                        --connection_type                   source_target \
                        --connection_filter_type            all \
                        --penalty_connection_filter_type    all \
                        --weight_type                       indicator \
                        --attention_activation              $ACTIVATION \

                fi
            done
        done
    done
done
