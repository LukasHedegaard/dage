#!/usr/bin/env bash
DESCRIPTION="Tune on source data using gradual unfreeze. 
We perform the gradual unfreeze mechanism within this script, first training only new layers until convergence.
We then reduce the learing rate, and perform training again, with some base-layers unfrozen, this time using the weights from the previous iteration as starting point.
This is repeated, each time unfreezing more layers."

METHOD=tune_source
OPTIMIZER=adam
ARCHITECTURE=single_stream
MODEL_BASE=resnet152v2
FEATURES=images
BATCH_SIZE=32
AUGMENT=1
EXPERIMENT_ID_BASE="visda_${MODEL_BASE}_aug"
SOURCE="visda"
TARGET="visda"

SEED=1

EXPERIMENT_ID="${EXPERIMENT_ID_BASE}"
DIR_NAME=./runs/$METHOD/$EXPERIMENT_ID
mkdir $DIR_NAME -p
echo $DESCRIPTION > $DIR_NAME/description.txt

TIMESTAMP_OLD=$(date '+%Y%m%d%H%M%S')

python3 run.py \
    --training_regimen  regular \
    --timestamp         $TIMESTAMP_OLD \
    --learning_rate     1e-5 \
    --epochs            10 \
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
    --dense_size        2048 \
    --embed_size        512 \
    --shuffle_buffer_size   1000 \
    --num_unfrozen_base_layers 0 \
    --verbose 1 \
    --monitor accuracy \
    --dropout 0.5 \

./scripts/notify.sh "Finished job: ${METHOD}/${EXPERIMENT_ID_BASE} on GPU ${GPU_ID}."
