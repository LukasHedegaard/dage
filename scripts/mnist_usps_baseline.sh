#!/usr/bin/env bash

DESCRIPTION="Baseline for MNIST to USPS domain adaptation. Train on MNIST, test on USPS. Using hyperparameters found in ~/notebooks/hypersearch-results.ipybn"

METHOD=tune_source
EXPERIMENT_ID=baseline_mnist_usps
DIR_NAME=./runs/$METHOD/$EXPERIMENT_ID

GPU_ID=1
TEST_AS_VAL=1
AUGMENT=1

mkdir $DIR_NAME -p
echo $DESCRIPTION > $DIR_NAME/description.txt

for SEED in 1 2 3 4 5 6 7 8 9 10
do
    python run.py \
        --source            mnist \
        --target            usps \
        --gpu_id            $GPU_ID \
        --experiment_id     $EXPERIMENT_ID \
        --seed              $SEED \
        --augment           $AUGMENT \
        --test_as_val       $TEST_AS_VAL \
        --monitor           acc \
        --architecture      single_stream \
        --model_base        conv2 \
        --features          images \
        --epochs            50 \
        --batch_size        256 \
        --mode              train_and_test \
        --resize_mode       2 \
        --training_regimen  regular \
        --method                    tune_source \
        --batch_norm                1 \
        --optimizer                 adam \
        --learning_rate             0.00124 \
        --learning_rate_decay       1.38e-07 \
        --momentum                  0.97 \
        --l2                        7.1e-04 \
        --dropout                   0.65 \
        # --num_unfrozen_base_layers  -1 \
        # --loss_alpha                0.103 \
        # --loss_weights_even         0.984 \
        # --ratio                     3 \
done
        