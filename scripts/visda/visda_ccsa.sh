#!/usr/bin/env bash
DESCRIPTION="CCSA on VisDA-C using hyperparameters from Office31 A->D"

METHOD=ccsa
OPTIMIZER=adam
MODEL_BASE=resnet152v2
FEATURES=images
BATCH_SIZE=16
AUGMENT=1
EXPERIMENT_ID_BASE="visda_${MODEL_BASE}_aug"
SOURCE="visda"
TARGET="visda"

FROM_WEIGHTS="./runs/tune_source/visda_resnet152v2_aug/visdavisda_1_20210908104045/checkpoints/cp-best.ckpt"

for SEED in 1 2 3 4 5
do
    EXPERIMENT_ID="${EXPERIMENT_ID_BASE}"
    DIR_NAME=./runs/$METHOD/$EXPERIMENT_ID
    mkdir $DIR_NAME -p
    echo $DESCRIPTION > $DIR_NAME/description.txt

    python3 run.py \
        --from_weights      $FROM_WEIGHTS \
        --epochs            20 \
        --experiment_id     $EXPERIMENT_ID \
        --source            $SOURCE \
        --target            $TARGET \
        --num_source_samples_per_class 100 \
        --num_target_samples_per_class 10 \
        --seed              $SEED \
        --method            $METHOD \
        --architecture      two_stream_pair_embeds \
        --model_base        $MODEL_BASE \
        --features          $FEATURES \
        --batch_size        $BATCH_SIZE \
        --augment           $AUGMENT \
        --monitor           accuracy \
        --verbose           1 \
        --dense_size        1024 \
        --embed_size        128 \
        --training_regimen  batch_repeat \
        --batch_repeats     2 \
        --shuffle_buffer_size   1000 \
        --batch_norm                1 \
        --optimizer                 adam \
        --learning_rate             1.316e-06 \
        --learning_rate_decay       6.322e-07 \
        --momentum                  0.5 \
        --num_unfrozen_base_layers  9 \
        --l2                        2.885e-07 \
        --dropout                   0.674 \
        --loss_alpha                0.1 \
        --loss_weights_even         0.136 \
        --ratio                     3 \

done

./scripts/notify.sh "Finished job: ${METHOD}/${EXPERIMENT_ID_BASE} on GPU ${GPU_ID}."
