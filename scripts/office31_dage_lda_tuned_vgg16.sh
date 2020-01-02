#!/usr/bin/env bash

DESCRIPTION="DAGE variant resembling Linear Discriminant Analysis with hyperparameters tuned for VGG16."

EXPERIMENT_ID=dage_lda_vgg16_tuned
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
        --method                            dage \
        --connection_type                   source_target \
        --weight_type                       indicator \
        --connection_filter_type            all \
        --penalty_connection_filter_type    all \
        --batch_norm                1 \
        --optimizer                 adam \
        --learning_rate             9.21e-05 \
        --learning_rate_decay       1.27e-06 \
        --momentum                  0.6 \
        --num_unfrozen_base_layers  11 \
        --l2                        1.93e-04 \
        --dropout                   0.674 \
        --loss_alpha                0.103 \
        --loss_weights_even         0.984 \
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
        --method                            dage \
        --connection_type                   source_target \
        --weight_type                       indicator \
        --connection_filter_type            all \
        --penalty_connection_filter_type    all \
        --batch_norm                1 \
        --optimizer                 adam \
        --learning_rate             1.01e-05 \
        --learning_rate_decay       2.29e-04 \
        --momentum                  0.982 \
        --num_unfrozen_base_layers  6 \
        --l2                        1.15e-07 \
        --dropout                   0.34 \
        --loss_alpha                0.11 \
        --loss_weights_even         0.73 \
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
        --method                            dage \
        --connection_type                   source_target \
        --weight_type                       indicator \
        --connection_filter_type            all \
        --penalty_connection_filter_type    all \
        --batch_norm                1 \
        --optimizer                 adam \
        --learning_rate             1.54e-08 \
        --momentum                  0.82 \
        --num_unfrozen_base_layers  3 \
        --l2                        0.000887 \
        --dropout                   0.105 \
        --loss_alpha                0.50 \
        --loss_weights_even         0.62 \
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
        --method                            dage \
        --connection_type                   source_target \
        --weight_type                       indicator \
        --connection_filter_type            all \
        --penalty_connection_filter_type    all \
        --batch_norm                1 \
        --optimizer                 adam \
        --learning_rate             3.848e-07 \
        --learning_rate_decay       1.014e-04 \
        --momentum                  0.981 \
        --num_unfrozen_base_layers  3 \
        --l2                        2.355e-06 \
        --dropout                   0.342 \
        --loss_alpha                0.283 \
        --loss_weights_even         0.567 \
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
        --method                            dage \
        --connection_type                   source_target \
        --weight_type                       indicator \
        --connection_filter_type            all \
        --penalty_connection_filter_type    all \
        --batch_norm                1 \
        --optimizer                 adam \
        --learning_rate             3.236e-06 \
        --learning_rate_decay       1e-2 \
        --momentum                  0.99 \
        --num_unfrozen_base_layers  3 \
        --l2                        1e-7 \
        --dropout                   0.1 \
        --loss_alpha                0.476 \
        --loss_weights_even         0.510 \
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
        --method                            dage \
        --connection_type                   source_target \
        --weight_type                       indicator \
        --connection_filter_type            all \
        --penalty_connection_filter_type    all \
        --batch_norm                1 \
        --optimizer                 adam \
        --learning_rate             8.118e-08 \
        --momentum                  0.981 \
        --num_unfrozen_base_layers  2 \
        --l2                        7.519e-04 \
        --dropout                   0.769 \
        --loss_alpha                0.919 \
        --loss_weights_even         0.154 \
        --ratio                     2 \
        
done
        