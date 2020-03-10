#!/usr/bin/env bash

DESCRIPTION="d-SNE evaluation using tuned hyper-parameters."

EXPERIMENT_ID=dsne_vgg16_tuned
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
        --method                    dsne \
        --connection_filter_param   0.001693 \
        --batch_norm                1 \
        --optimizer                 adam \
        --learning_rate             1.994467e-06 \
        --learning_rate_decay       8.941508e-05 \
        --momentum                  0.987766 \
        --num_unfrozen_base_layers  9 \
        --l2                        4.290299e-05 \
        --dropout                   0.767407 \
        --loss_alpha                0.102982 \
        --loss_weights_even         0.866917 \
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
        --method                    dsne \
        --connection_filter_param   1e-05 \
        --batch_norm                1 \
        --optimizer                 adam \
        --learning_rate             0.001 \
        --learning_rate_decay       0.01 \
        --momentum                  0.5 \
        --num_unfrozen_base_layers  6 \
        --l2                        1.0e-07 \
        --dropout                   0.8 \
        --loss_alpha                0.1 \
        --loss_weights_even         0.87 \
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
        --method                    dsne \
        --connection_filter_param   100.0 \
        --batch_norm                1 \
        --optimizer                 adam \
        --learning_rate             0.001 \
        --learning_rate_decay       0.01 \
        --momentum                  0.99 \
        --num_unfrozen_base_layers  16 \
        --l2                        0.001 \
        --dropout                   0.1 \
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
        --method                    dsne \
        --connection_filter_param   10.0 \
        --batch_norm                1 \
        --optimizer                 adam \
        --learning_rate             7.918177e-08 \
        --learning_rate_decay       1.665447e-07 \
        --momentum                  0.702352 \
        --num_unfrozen_base_layers  3 \
        --l2                        6.844385e-07 \
        --dropout                   0.665386 \
        --loss_alpha                0.151635 \
        --loss_weights_even         0.63797 \
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
        --method                    dsne \
        --connection_filter_param   27.0164 \
        --batch_norm                1 \
        --optimizer                 adam \
        --learning_rate             0.001 \
        --learning_rate_decay       1.777872e-06 \
        --momentum                  0.5 \
        --num_unfrozen_base_layers  3 \
        --l2                        0.0006777264 \
        --dropout                   0.610774 \
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
        --method                    dsne \
        --connection_filter_param   100 \
        --batch_norm                1 \
        --optimizer                 adam \
        --learning_rate             1e-08 \
        --learning_rate_decay       0.01 \
        --momentum                  0.5 \
        --num_unfrozen_base_layers  0 \
        --l2                        1e-07\
        --dropout                   0.8 \
        --loss_alpha                0.447352 \
        --loss_weights_even         0.0 \
        --ratio                     1 \
        
done
        