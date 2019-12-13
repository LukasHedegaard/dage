#!/usr/bin/env bash
DESCRIPTION="Tune on source data using gradual unfreeze. 
We perform the gradual unfreeze mechanism within this script, first training only new layers until convergence.
We then reduce the learing rate, and perform training again, with some base-layers unfrozen, this time using the weights from the previous iteration as starting point.
This is repeated, each time unfreezing more layers."

METHOD=tune_source
GPU_ID=0
OPTIMIZER=adam
ARCHITECTURE=single_stream
MODEL_BASE=vgg16
FEATURES=images
BATCH_SIZE=12
AUGMENT=1
EXPERIMENT_ID_BASE="${MODEL_BASE}_aug"

for SEED in 0 1 2 3 4
do
    for SOURCE in A W D
    do
        for TARGET in D A W
        do
            if [ $SOURCE != $TARGET ]
            then
                EXPERIMENT_ID="${EXPERIMENT_ID_BASE}"
                DIR_NAME=./runs/$METHOD/$EXPERIMENT_ID
                mkdir $DIR_NAME -p
                echo $DESCRIPTION > $DIR_NAME/description.txt

                TIMESTAMP_OLD=$(date '+%Y%m%d%H%M%S')

                python3 run.py \
                    --num_unfrozen_base_layers 0 \
                    --training_regimen  regular \
                    --timestamp         $TIMESTAMP_OLD \
                    --learning_rate     1e-4 \
                    --epochs            15 \
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

                FROM_WEIGHTS="./runs/$METHOD/$EXPERIMENT_ID/${SOURCE}${TARGET}_${SEED}_${TIMESTAMP_OLD}/checkpoints/cp-best.ckpt"

                EXPERIMENT_ID="${EXPERIMENT_ID_BASE}_coarse_grad_ft"
                DIR_NAME=./runs/$METHOD/$EXPERIMENT_ID
                mkdir $DIR_NAME -p
                echo $DESCRIPTION > $DIR_NAME/description.txt

                python3 run.py \
                    --training_regimen  gradual_unfreeze \
                    --learning_rate     1e-5 \
                    --epochs            10 \
                    --optimizer         $OPTIMIZER \
                    --gpu_id            $GPU_ID \
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
                    --from_weights      $FROM_WEIGHTS \

            fi
        done
    done
done

./scripts/notify.sh "Finished job: ${METHOD}/${EXPERIMENT_ID_BASE} on GPU ${GPU_ID}."