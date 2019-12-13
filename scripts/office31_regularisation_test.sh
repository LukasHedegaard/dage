#!/usr/bin/env bash
DESCRIPTION="Test which regularisation strategy works best. We test batch norm, dropout, and L2 here."

for BN in 0 1
do
    for L2 in 0.0001 0.00001 0
    do

        for DROPOUT in 0 0.25 0.5
        do
            EXPERIMENT_ID="regularization_test_bn${BN}_drop${DROPOUT}"
            METHOD=dage

            DIR_NAME=./runs/$METHOD/$EXPERIMENT_ID

            mkdir $DIR_NAME -p
            echo $DESCRIPTION > $DIR_NAME/description.txt

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
                                --learning_rate     0.001 \
                                --optimizer         adam \
                                --dropout           $DROPOUT \
                                --batch_norm        $BN \
                                --l2                $L2 \
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
                                --loss_alpha        0.75 \
                                --loss_weights_even 0 \
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
done