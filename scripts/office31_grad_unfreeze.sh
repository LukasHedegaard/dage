#!/usr/bin/env bash
DESCRIPTION="Train using gradual unfreeze. 
We perform the gradual unfreeze mechanism within this script, first training only new layers until convergence.
We then reduce the learing rate, and perform training again, with some base-layers unfrozen, this time using the weights from the previous iteration as starting point.
This is repeated, each time unfreezing more layers."

METHOD=tune_source
GPU_ID=1
OPTIMIZER=adam
ARCHITECTURE=single_stream
MODEL_BASE=vgg16
FEATURES=images
EPOCHS=20
BATCH_SIZE=12
AUGMENT=1
LOSS_ALPHA=0.25
LOSS_WEIGHTS_EVEN=1
WEIGHT_TYPE=indicator
CONNECTION_TYPE=source_target
CONNECTION_FILTER_TYPE=knn
CONNECTION_FILTER_PARAM=1
PENALTY_CONNECTION_FILTER_TYPE=knn
PENALTY_CONNECTION_FILTER_PARAM=1

EXPERIMENT_ID_BASE=gradual_unfreeze_vgg16

for SEED in 1 2 4 5
do
    for SOURCE in A #W D
    do
        for TARGET in D #A W
        do
            if [ $SOURCE != $TARGET ]
            then
                EXPERIMENT_ID="${EXPERIMENT_ID_BASE}_0"
                DIR_NAME=./runs/$METHOD/$EXPERIMENT_ID
                mkdir $DIR_NAME -p
                echo $DESCRIPTION > $DIR_NAME/description.txt

                TIMESTAMP_OLD=$(date '+%Y%m%d%H%M%S')

                python3 run.py \
                    --num_unfrozen_base_layers 0 \
                    --timestamp         $TIMESTAMP_OLD \
                    --learning_rate     0.001 \
                    --gpu_id            $GPU_ID \
                    --optimizer         $OPTIMIZER \
                    --experiment_id     $EXPERIMENT_ID \
                    --source            $SOURCE \
                    --target            $TARGET \
                    --seed              $SEED \
                    --method            $METHOD \
                    --architecture      $ARCHITECTURE \
                    --model_base        $MODEL_BASE \
                    --features          $FEATURES \
                    --epochs            $EPOCHS \
                    --batch_size        $BATCH_SIZE \
                    --augment           $AUGMENT \
                    --loss_alpha        $LOSS_ALPHA \
                    --loss_weights_even $LOSS_WEIGHTS_EVEN \
                    --weight_type       $WEIGHT_TYPE \
                    --connection_type                   $CONNECTION_TYPE \
                    --connection_filter_type            $CONNECTION_FILTER_TYPE \
                    --connection_filter_param           $CONNECTION_FILTER_PARAM \
                    --penalty_connection_filter_type    $PENALTY_CONNECTION_FILTER_TYPE \
                    --penalty_connection_filter_param   $PENALTY_CONNECTION_FILTER_PARAM \

                FROM_WEIGHTS="./runs/$METHOD/$EXPERIMENT_ID/${SOURCE}${TARGET}_${SEED}_${TIMESTAMP_OLD}/checkpoints/cp-best.ckpt"

                for UNFROZEN in 2 3 4 6 7 8 10 11 12
                do
                    EXPERIMENT_ID="${EXPERIMENT_ID_BASE}_${UNFROZEN}"
                    DIR_NAME=./runs/$METHOD/$EXPERIMENT_ID
                    mkdir $DIR_NAME -p
                    echo $DESCRIPTION > $DIR_NAME/description.txt

                    TIMESTAMP_NEW=$(date '+%Y%m%d%H%M%S')

                    python3 run.py \
                        --num_unfrozen_base_layers $UNFROZEN \
                        --timestamp         $TIMESTAMP_NEW \
                        --learning_rate     0.0001 \
                        --optimizer         $OPTIMIZER \
                        --gpu_id            $GPU_ID \
                        --from_weights      $FROM_WEIGHTS \
                        --experiment_id     $EXPERIMENT_ID \
                        --source            $SOURCE \
                        --target            $TARGET \
                        --seed              $SEED \
                        --method            $METHOD \
                        --architecture      $ARCHITECTURE \
                        --model_base        $MODEL_BASE \
                        --features          $FEATURES \
                        --epochs            $EPOCHS \
                        --batch_size        $BATCH_SIZE \
                        --augment           $AUGMENT \
                        --loss_alpha        $LOSS_ALPHA \
                        --loss_weights_even $LOSS_WEIGHTS_EVEN \
                        --weight_type       $WEIGHT_TYPE \
                        --connection_type                   $CONNECTION_TYPE \
                        --connection_filter_type            $CONNECTION_FILTER_TYPE \
                        --connection_filter_param           $CONNECTION_FILTER_PARAM \
                        --penalty_connection_filter_type    $PENALTY_CONNECTION_FILTER_TYPE \
                        --penalty_connection_filter_param   $PENALTY_CONNECTION_FILTER_PARAM \

                    TIMESTAMP_OLD=$TIMESTAMP_NEW
                    FROM_WEIGHTS="./runs/$METHOD/$EXPERIMENT_ID/${SOURCE}${TARGET}_${SEED}_${TIMESTAMP_OLD}/checkpoints/cp-best.ckpt"
                done
            fi
        done
    done
done