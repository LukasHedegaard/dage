#!/usr/bin/env bash

DESCRIPTION="d-SNE for MNIST to USPS domain adaptation. Train on MNIST, test on USPS. Using hyperparameters found in ~/notebooks/hypersearch-results.ipybn"

METHOD=dsne
EXPERIMENT_ID=dsne_mnist_usps_v2
DIR_NAME=./runs/$METHOD/$EXPERIMENT_ID

GPU_ID=2
TEST_AS_VAL=1
AUGMENT=1

mkdir $DIR_NAME -p
echo $DESCRIPTION > $DIR_NAME/description.txt

for NUM_TGT_PER_CLASS in 1 3 5 7 
do
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
            --architecture      two_stream_pair_embeds \
            --model_base        conv2 \
            --features          images \
            --epochs            50 \
            --batch_size        256 \
            --mode              train_and_test \
            --resize_mode       1 \
            --training_regimen  regular \
            --num_source_samples_per_class  200 \
            --num_target_samples_per_class  $NUM_TGT_PER_CLASS \
            --method                        dsne \
            --batch_norm                0 \
            --optimizer                 adam \
            --learning_rate             0.00192 \
            --learning_rate_decay       1.716e-06 \
            --dropout                   0.46 \
            --l2                        2.678e-07 \
            --momentum                  0.944 \
            --loss_alpha                0.10 \
            --loss_weights_even         0.00 \
            --ratio                     3 \
            # --num_unfrozen_base_layers  -1 \

    done
done

./scripts/notify.sh "Finished job: ${METHOD}/${EXPERIMENT_ID} on GPU ${GPU_ID}."
