#!/usr/bin/env bash
DESCRIPTION="Test of the impact of batch size for CCSA."

METHOD=ccsa
GPU_ID=0
OPTIMIZER=adam
ARCHITECTURE=two_stream_pair_embeds
MODEL_BASE=none
EPOCHS=20
FEATURES=vgg16
AUGMENT=0
EXPERIMENT_ID_BASE="batch_size"
MODE="train_test"
TRAINING_REGIMEN=regular
LEARNING_RATE=0.0016
ALPHA=0.1

for BATCH_SIZE in 16 64 256 1024 4096
do
    # learning rate compensation
    # if [ $BATCH_SIZE == 16 ]; then LEARNING_RATE=1e-4; fi
    # if [ $BATCH_SIZE == 64 ]; then LEARNING_RATE=2e-4; fi
    # if [ $BATCH_SIZE == 256 ]; then LEARNING_RATE=4e-4; fi
    # if [ $BATCH_SIZE == 1024 ]; then LEARNING_RATE=8e-4; fi
    # if [ $BATCH_SIZE == 4094 ]; then LEARNING_RATE=16e-4; fi

    for SEED in 0 1 2 3 4
    do
        for SOURCE in A #D W
        do
            for TARGET in A D #W
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