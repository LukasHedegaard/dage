#!/usr/bin/env bash
DESCRIPTION="Test which optimizer performs better for domain adaptation."

EXPERIMENT_ID_BASE=optimizer_test
METHOD=multitask

for OPTIMIZER in sgd sgd_mom adam 
do
    EXPERIMENT_ID=${EXPERIMENT_ID_BASE}_${OPTIMIZER}

    DIR_NAME=./runs/$METHOD/$EXPERIMENT_ID
    mkdir $DIR_NAME -p
    echo $DESCRIPTION > $DIR_NAME/description.txt

    for LR in 1 0.1 0.01 0.001 0.0001 
    do
        for SEED in 0 1 2 4 5
        do
            for SOURCE in W #A D W
            do
                for TARGET in A #D W
                do
                    if [ $SOURCE != $TARGET ]
                    then
                        TIMESTAMP=$(date '+%Y%m%d%H%M%S')

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
                            --timestamp         $TIMESTAMP \

                        FT_RUN_DIR=./runs/$METHOD/$EXPERIMENT_ID/${SOURCE}${TARGET}_${SEED}_${TIMESTAMP}

                        if [ ! -f "$FT_RUN_DIR/report.json" ]; then
                            rm -rf $FT_RUN_DIR
                        else
                            rm -rf $FT_RUN_DIR/checkpoints
                        fi
                    fi
                done
            done
        done
    done
done

./scripts/notify.sh "Finished job: ${METHOD}/${EXPERIMENT_ID_BASE} on GPU ${GPU_ID}."