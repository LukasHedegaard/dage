#!/usr/bin/env bash
DESCRIPTION="Test which optimizer performs better for domain adaptation."

for OPTIMIZER in sgd adam
do
    EXPERIMENT_ID=optimizer_test_$OPTIMIZER
    METHOD=dage

    DIR_NAME=./runs/$METHOD/$EXPERIMENT_ID

    mkdir $DIR_NAME -p
    echo $DESCRIPTION > $DIR_NAME/description.txt

    for LR in 0.1 0.01 0.001
    do
        for SEED in 0 1 2 4 5
        do
            for SOURCE in A 
            do
                for TARGET in D 
                do
                    if [ $SOURCE != $TARGET ]
                    then
                        python3 run.py \
                            --gpu_id            0 \
                            --learning_rate     $LR \
                            --optimizer         $OPTIMIZER \
                            --experiment_id     $EXPERIMENT_ID \
                            --source            $SOURCE \
                            --target            $TARGET \
                            --seed              $SEED \
                            --method            $METHOD \
                            --architecture      two_stream_pair_embeds \
                            --model_base        none \
                            --features          vgg16 \
                            --epochs            30 \
                            --batch_size        16 \
                            --augment           0 \
                            --loss_alpha        0.25 \
                            --loss_weights_even 1 \
                            --weight_type       indicator \
                            --connection_type                   source_target \
                            --connection_filter_type            knn \
                            --connection_filter_param           1 \
                            --penalty_connection_filter_type    knn \
                            --connection_filter_param           1 \
                            
                    fi
                done
            done
        done
    done
done