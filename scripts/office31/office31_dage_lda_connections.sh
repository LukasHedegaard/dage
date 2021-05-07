#!/usr/bin/env bash
DESCRIPTION="Test of the best connections for DAGE-LDA connection."

METHOD=dage
GPU_ID=2
OPTIMIZER=adam
LEARNING_RATE=1e-4
ARCHITECTURE=two_stream_pair_embeds
MODEL_BASE=none
EPOCHS=20
FEATURES=vgg16
BATCH_SIZE=16
AUGMENT=0
EXPERIMENT_ID_BASE="lda_connections_st_int_all_pen"
MODE="train_test_validate"
TRAINING_REGIMEN=regular
ALPHA=0.75

for CONNECTION_TYPE in 'ST_INT_ALL_PEN' 'SOURCE_TARGET' 'ALL' 
do
    for SEED in 1 2 3 4 5
    do
        for SOURCE in A D W
        do
            for TARGET in A D W
            do
                if [ $SOURCE != $TARGET ]
                then
                    EXPERIMENT_ID="${EXPERIMENT_ID_BASE}"
                    DIR_NAME=./runs/$METHOD/$EXPERIMENT_ID
                    mkdir $DIR_NAME -p
                    echo $DESCRIPTION > $DIR_NAME/description.txt

                    TIMESTAMP=$(date '+%Y%m%d%H%M%S')

                    python3 run.py \
                        --training_regimen  $TRAINING_REGIMEN \
                        --timestamp         $TIMESTAMP \
                        --learning_rate     $LEARNING_RATE \
                        --epochs            $EPOCHS \
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
                        --batch_size        $BATCH_SIZE \
                        --augment           $AUGMENT \
                        --loss_alpha        $ALPHA \
                        --mode              $MODE \
                        --connection_type                   $CONNECTION_TYPE \
                        --connection_filter_type            all \
                        --penalty_connection_filter_type    all \
                        --weight_type                       indicator \

                    # delete checkpoint
                    RUN_DIR=./runs/$METHOD/$EXPERIMENT_ID/${SOURCE}${TARGET}_${SEED}_${TIMESTAMP}

                    if [ ! -f "$RUN_DIR/report.json" ]; then
                        rm -rf $RUN_DIR
                    else
                        rm -rf $RUN_DIR/checkpoints
                    fi
                fi
            done
        done
    done
done

./scripts/notify.sh "Finished job: ${METHOD}/${EXPERIMENT_ID_BASE} on GPU ${GPU_ID}."