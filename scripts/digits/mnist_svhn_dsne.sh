#!/usr/bin/env bash

DESCRIPTION="d-SNE for MNIST to SVHN domain adaptation. Using hyperparameters found in ~/notebooks/hypersearch-results.ipybn"

METHOD=dsne
EXPERIMENT_ID=dsne_digits_mnist_svhn
DIR_NAME=./runs/$METHOD/$EXPERIMENT_ID

AUGMENT=1

mkdir $DIR_NAME -p
echo $DESCRIPTION > $DIR_NAME/description.txt

NUM_TGT_PER_CLASS=10

for SEED in 1 2 3 4 5
do
    python run.py \
        --source            mnist \
        --target            svhn \
        --gpu_id            0 \
        --experiment_id     $EXPERIMENT_ID \
        --seed              $SEED \
        --augment           $AUGMENT \
        --monitor           acc \
        --architecture      two_stream_pair_embeds \
        --model_base        conv2 \
        --features          images \
        --epochs            50 \
        --batch_size        128 \
        --mode              train_and_test \
        --training_regimen  regular \
        --num_source_samples_per_class  5000 \
        --num_target_samples_per_class  $NUM_TGT_PER_CLASS \
        --method                        $METHOD \
        --connection_filter_param   0.12437131980930234 \
        --batch_norm                0 \
        --optimizer                 adam \
        --learning_rate             0.0008369557444945195 \
        --learning_rate_decay       1e-07 \
        --dropout                   0.1 \
        --l2                        5.5623556981575945e-05 \
        --momentum                  0.99 \
        --loss_alpha                0.01 \
        --loss_weights_even         0.7417739602216188 \
        --ratio                     3 \
        --resize_mode               2 \

done
        
./scripts/notify.sh "Finished job: ${METHOD}/${EXPERIMENT_ID} on GPU ${GPU_ID}."