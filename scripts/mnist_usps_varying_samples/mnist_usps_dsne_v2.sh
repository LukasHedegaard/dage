#!/usr/bin/env bash

DESCRIPTION="d-SNE for MNIST to USPS domain adaptation. Train on MNIST, test on USPS. Using hyperparameters found in ~/notebooks/hypersearch-results.ipybn"

METHOD=dsne
EXPERIMENT_ID=dsne_mnist_usps_new
DIR_NAME=./runs/$METHOD/$EXPERIMENT_ID

GPU_ID=3
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
            --monitor           acc \
            --architecture      two_stream_pair_embeds \
            --model_base        conv2 \
            --features          images \
            --epochs            50 \
            --batch_size        128 \
            --mode              train_and_test \
            --training_regimen  regular \
            --num_source_samples_per_class  200 \
            --num_target_samples_per_class  $NUM_TGT_PER_CLASS \
            --method                        dsne \
            --connection_filter_param   0.001 \
            --batch_norm                1 \
            --optimizer                 adam \
            --learning_rate             0.005030119691186109 \
            --learning_rate_decay       4.128902544505756e-05 \
            --dropout                   0.1872781583734256 \
            --l2                        0.0005074072177526865 \
            --momentum                  0.896316 \
            --loss_alpha                0.06640039255598627 \
            --loss_weights_even         0.06847478236950102 \
            --ratio                     3 \
            --resize_mode               2 \

    done
done

./scripts/notify.sh "Finished job: ${METHOD}/${EXPERIMENT_ID} on GPU ${GPU_ID}."
