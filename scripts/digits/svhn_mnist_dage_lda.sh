#!/usr/bin/env bash

DESCRIPTION="DAGE for USPS to MNIST domain adaptation. Using hyperparameters found in ~/notebooks/hypersearch-results.ipybn"

METHOD=dage
EXPERIMENT_ID=dage_digits_usps_mnist
DIR_NAME=./runs/$METHOD/$EXPERIMENT_ID

AUGMENT=1

mkdir $DIR_NAME -p
echo $DESCRIPTION > $DIR_NAME/description.txt

NUM_TGT_PER_CLASS=10

for SEED in 1 2 3 4 5
do
    python run.py \
        --source            usps \
        --target            mnist \
        --gpu_id            0 \
        --experiment_id     $EXPERIMENT_ID \
        --seed              $SEED \
        --augment           $AUGMENT \
        --monitor           acc \
        --architecture      two_stream_pair_embeds \
        --model_base        conv2 \
        --features          images \
        --epochs            20 \
        --batch_size        128 \
        --mode              train_and_test \
        --training_regimen  regular \
        --num_source_samples_per_class  700 \
        --num_target_samples_per_class  $NUM_TGT_PER_CLASS \
        --method                        $METHOD \
        --connection_type                   SOURCE_TARGET \
        --weight_type                       INDICATOR \
        --connection_filter_type            ALL \
        --penalty_connection_filter_type    ALL \
        --batch_norm                1 \
        --optimizer                 adam \
        --learning_rate             0.0035753161317240803 \
        --learning_rate_decay       1.572832625907872e-05 \
        --dropout                   0.2412034416347774 \
        --l2                        0.0003828726839315707 \
        --momentum                  0.984569 \
        --loss_alpha                0.47586281871846964 \
        --loss_weights_even         0.5632755719763838 \
        --ratio                     3 \
        --resize_mode               2 \

done
        
./scripts/notify.sh "Finished job: ${METHOD}/${EXPERIMENT_ID} on GPU ${GPU_ID}."