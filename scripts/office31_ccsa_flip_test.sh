#!/usr/bin/env bash
DESCRIPTION="Test of the flipping vs non-flipping training regimen for CCSA."

METHOD=ccsa
GPU_ID=1
OPTIMIZER=adam
LEARNING_RATE=1e-4
ARCHITECTURE=two_stream_pair_embeds
MODEL_BASE=none
EPOCHS=20
FEATURES=vgg16
BATCH_SIZE=16
AUGMENT=0
ALPHA=0.25
MODE="train_test_validate"

for LOSS_WEIGHTING in 0 0.5 
do
    EXPERIMENT_ID_BASE="flip_test_${TRAINING_REGIMEN}_${LOSS_WEIGHTING}_v2"
    for SEED in 0 1 2 3 4
    do
        for SOURCE in W #A D W
        do
            for TARGET in A #D W
            do
                if [ $SOURCE != $TARGET ]
                then
                    EXPERIMENT_ID="${EXPERIMENT_ID_BASE}"
                    DIR_NAME=./runs/$METHOD/$EXPERIMENT_ID
                    mkdir $DIR_NAME -p
                    echo $DESCRIPTION > $DIR_NAME/description.txt

                    TIMESTAMP=$(date '+%Y%m%d%H%M%S')

                    python3 run.py \
                        --training_regimen  flipping \
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
                        --loss_weights_even $LOSS_WEIGHTING \

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