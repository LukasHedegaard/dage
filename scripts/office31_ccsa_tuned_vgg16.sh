#!/usr/bin/env bash

DESCRIPTION="CCSA evaluation using tuned hyper-parameters."

EXPERIMENT_ID=ccsa_vgg16_tuned
DIR_NAME=./runs/$METHOD/$EXPERIMENT_ID

GPU_ID=0
TEST_AS_VAL=1
AUGMENT=1

mkdir $DIR_NAME -p
echo $DESCRIPTION > $DIR_NAME/description.txt

for SEED in 1 2 3 4 5
do
    python run.py \
        --source            A \
        --target            D \
        --from_weights      "./runs/tune_source/vgg16_aug_ft_best/AD/checkpoints/cp-best.ckpt" \
        --gpu_id            $GPU_ID \
        --experiment_id     $EXPERIMENT_ID \
        --seed              $SEED \
        --augment           $AUGMENT \
        --test_as_val       $TEST_AS_VAL \
        --monitor           acc \
        --architecture      two_stream_pair_embeds \
        --model_base        vgg16 \
        --features          images \
        --epochs            30 \
        --batch_size        16 \
        --mode              train_and_test \
        --training_regimen  batch_repeat \
        --batch_repeats     2 \
        --method                    ccsa \
        --connection_filter_param   10 \
        --batch_norm                1 \
        --optimizer                 adam \
        --learning_rate             1.316e-06 \
        --learning_rate_decay       6.322e-07 \
        --momentum                  0.5 \
        --num_unfrozen_base_layers  9 \
        --l2                        2.885e-07 \
        --dropout                   0.674 \
        --loss_alpha                0.100 \
        --loss_weights_even         0.136 \
        --ratio                     3 \


    python run.py \
        --source            A \
        --target            W \
        --from_weights      "./runs/tune_source/vgg16_aug_ft_best/AW/checkpoints/cp-best.ckpt" \
        --gpu_id            $GPU_ID \
        --experiment_id     $EXPERIMENT_ID \
        --seed              $SEED \
        --augment           $AUGMENT \
        --test_as_val       $TEST_AS_VAL \
        --monitor           acc \
        --architecture      two_stream_pair_embeds \
        --model_base        vgg16 \
        --features          images \
        --epochs            30 \
        --batch_size        16 \
        --mode              train_and_test \
        --training_regimen  batch_repeat \
        --batch_repeats     2 \
        --method                    ccsa \
        --connection_filter_param   10 \
        --batch_norm                1 \
        --optimizer                 adam \
        --learning_rate             1.0e-03 \
        --learning_rate_decay       1.0e-02 \
        --momentum                  0.99 \
        --num_unfrozen_base_layers  2 \
        --l2                        1.0e-03 \
        --dropout                   0.1 \
        --loss_alpha                0.45 \
        --loss_weights_even         0.79 \
        --ratio                     3 \

    python run.py \
        --source            D \
        --target            A \
        --from_weights      "./runs/tune_source/vgg16_aug_ft_best/DA/checkpoints/cp-best.ckpt" \
        --gpu_id            $GPU_ID \
        --experiment_id     $EXPERIMENT_ID \
        --seed              $SEED \
        --augment           $AUGMENT \
        --test_as_val       $TEST_AS_VAL \
        --monitor           acc \
        --architecture      two_stream_pair_embeds \
        --model_base        vgg16 \
        --features          images \
        --epochs            30 \
        --batch_size        16 \
        --mode              train_and_test \
        --training_regimen  batch_repeat \
        --batch_repeats     2 \
        --method                    ccsa \
        --connection_filter_param   0.0296 \
        --batch_norm                1 \
        --optimizer                 adam \
        --learning_rate             1.0e-08 \
        --learning_rate_decay       1.0e-02 \
        --momentum                  0.94 \
        --num_unfrozen_base_layers  4 \
        --l2                        1.0e-03 \
        --dropout                   0.8 \
        --loss_alpha                0.1 \
        --loss_weights_even         1.0 \
        --ratio                     3 \

    python run.py \
        --source            D \
        --target            W \
        --from_weights      "./runs/tune_source/vgg16_aug_ft_best/DW/checkpoints/cp-best.ckpt" \
        --gpu_id            $GPU_ID \
        --experiment_id     $EXPERIMENT_ID \
        --seed              $SEED \
        --augment           $AUGMENT \
        --test_as_val       $TEST_AS_VAL \
        --monitor           acc \
        --architecture      two_stream_pair_embeds \
        --model_base        vgg16 \
        --features          images \
        --epochs            30 \
        --batch_size        16 \
        --mode              train_and_test \
        --training_regimen  batch_repeat \
        --batch_repeats     2 \
        --method                    ccsa \
        --connection_filter_param   0.499 \
        --batch_norm                1 \
        --optimizer                 adam \
        --learning_rate             1.954e-05 \
        --learning_rate_decay       9.480e-03 \
        --momentum                  0.881 \
        --num_unfrozen_base_layers  16 \
        --l2                        2.560e-04 \
        --dropout                   0.767 \
        --loss_alpha                0.101 \
        --loss_weights_even         0.405 \
        --ratio                     3 \


    python run.py \
        --source            W \
        --target            A \
        --from_weights      "./runs/tune_source/vgg16_aug_ft_best/WA/checkpoints/cp-best.ckpt" \
        --gpu_id            $GPU_ID \
        --experiment_id     $EXPERIMENT_ID \
        --seed              $SEED \
        --augment           $AUGMENT \
        --test_as_val       $TEST_AS_VAL \
        --monitor           acc \
        --architecture      two_stream_pair_embeds \
        --model_base        vgg16 \
        --features          images \
        --epochs            30 \
        --batch_size        16 \
        --mode              train_and_test \
        --training_regimen  batch_repeat \
        --batch_repeats     2 \
        --method                    ccsa \
        --connection_filter_param   1e-05 \
        --batch_norm                1 \
        --optimizer                 adam \
        --learning_rate             1.0e-03 \
        --learning_rate_decay       1.0e-02 \
        --momentum                  0.604 \
        --num_unfrozen_base_layers  2 \
        --l2                        4.215e-07 \
        --dropout                   0.8. \
        --loss_alpha                0.10 \
        --loss_weights_even         1.0 \
        --ratio                     3 \

    python run.py \
        --source            W \
        --target            D \
        --from_weights      "./runs/tune_source/vgg16_aug_ft_best/WD/checkpoints/cp-best.ckpt" \
        --gpu_id            $GPU_ID \
        --experiment_id     $EXPERIMENT_ID \
        --seed              $SEED \
        --augment           $AUGMENT \
        --test_as_val       $TEST_AS_VAL \
        --monitor           acc \
        --architecture      two_stream_pair_embeds \
        --model_base        vgg16 \
        --features          images \
        --epochs            30 \
        --batch_size        16 \
        --mode              train_and_test \
        --training_regimen  batch_repeat \
        --batch_repeats     2 \
        --method                    ccsa \
        --connection_filter_param   0.000195 \
        --batch_norm                1 \
        --optimizer                 adam \
        --learning_rate             5.969e-07 \
        --learning_rate_decay       0.00108 \
        --momentum                  0.625 \
        --num_unfrozen_base_layers  0 \
        --l2                        1.884e-07\
        --dropout                   0.638 \
        --loss_alpha                0.494 \
        --loss_weights_even         0.235 \
        --ratio                     2 \
        
done
        