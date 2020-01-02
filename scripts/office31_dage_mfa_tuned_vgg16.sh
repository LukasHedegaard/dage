#!/usr/bin/env bash

DESCRIPTION="DAGE variant resembling Marginal Fisher Analysis with hyperparameters tuned for VGG16."

EXPERIMENT_ID=dage_mfa_vgg16_tuned
DIR_NAME=./runs/$METHOD/$EXPERIMENT_ID

GPU_ID=0
TEST_AS_VAL=1
AUGMENT=1

mkdir $DIR_NAME -p
echo $DESCRIPTION > $DIR_NAME/description.txt

for SEED in 0 1 2 3 4
do
    # python run.py \
    #     --source            D \
    #     --target            A \
    #     --from_weights      "./runs/tune_source/vgg16_aug_ft_best/DA/checkpoints/cp-best.ckpt" \
    #     --gpu_id            $GPU_ID \
    #     --experiment_id     $EXPERIMENT_ID \
    #     --seed              $SEED \
    #     --augment           $AUGMENT \
    #     --test_as_val       $TEST_AS_VAL \
    #     --monitor           acc \
    #     --architecture      two_stream_pair_embeds \
    #     --model_base        vgg16 \
    #     --features          images \
    #     --epochs            30 \
    #     --batch_size        16 \
    #     --mode              train_and_test \
    #     --training_regimen  batch_repeat \
    #     --batch_repeats     2 \
    #     --method                            dage \
    #     --connection_type                   source_target \
    #     --weight_type                       indicator \
    #     --connection_filter_type            knn \
    #     --penalty_connection_filter_type    ksd \
    #     --connection_filter_param           1 \
    #     --penalty_connection_filter_param   61 \
    #     --batch_norm                1 \
    #     --optimizer                 adam \
    #     --learning_rate             1e-7 \
    #     --momentum                  0.9 \
    #     --num_unfrozen_base_layers  10 \
    #     --l2                        0.001 \
    #     --dropout                   0.436 \
    #     --loss_alpha                0.526 \
    #     --loss_weights_even         1.0 \

    # python run.py \
    #     --source            W \
    #     --target            D \
    #     --from_weights      "./runs/tune_source/vgg16_aug_ft_best/WD/checkpoints/cp-best.ckpt" \
    #     --gpu_id            $GPU_ID \
    #     --experiment_id     $EXPERIMENT_ID \
    #     --seed              $SEED \
    #     --augment           $AUGMENT \
    #     --test_as_val       $TEST_AS_VAL \
    #     --monitor           acc \
    #     --architecture      two_stream_pair_embeds \
    #     --model_base        vgg16 \
    #     --features          images \
    #     --epochs            30 \
    #     --batch_size        16 \
    #     --mode              train_and_test \
    #     --training_regimen  batch_repeat \
    #     --batch_repeats     2 \
    #     --method                            dage \
    #     --connection_type                   source_target \
    #     --weight_type                       indicator \
    #     --connection_filter_type            knn \
    #     --penalty_connection_filter_type    ksd \
    #     --connection_filter_param           3 \
    #     --penalty_connection_filter_param   14 \
    #     --batch_norm                1 \
    #     --optimizer                 adam \
    #     --learning_rate             1.67e-07 \
    #     --momentum                  0.935 \
    #     --num_unfrozen_base_layers  0 \
    #     --l2                        0.0003 \
    #     --dropout                   0.122 \
    #     --loss_alpha                0.982 \
    #     --loss_weights_even         0.036 \

    # python run.py \
    #     --source            D \
    #     --target            W \
    #     --from_weights      "./runs/tune_source/vgg16_aug_ft_best/DW/checkpoints/cp-best.ckpt" \
    #     --gpu_id            $GPU_ID \
    #     --experiment_id     $EXPERIMENT_ID \
    #     --seed              $SEED \
    #     --augment           $AUGMENT \
    #     --test_as_val       $TEST_AS_VAL \
    #     --monitor           acc \
    #     --architecture      two_stream_pair_embeds \
    #     --model_base        vgg16 \
    #     --features          images \
    #     --epochs            30 \
    #     --batch_size        16 \
    #     --mode              train_and_test \
    #     --training_regimen  batch_repeat \
    #     --batch_repeats     2 \
    #     --method                            dage \
    #     --connection_type                   source_target \
    #     --weight_type                       indicator \
    #     --connection_filter_type            knn \
    #     --penalty_connection_filter_type    ksd \
    #     --connection_filter_param           3 \
    #     --penalty_connection_filter_param   122 \
    #     --batch_norm                1 \
    #     --optimizer                 adam \
    #     --learning_rate             7.17e-04 \
    #     --learning_rate_decay       0.000137 \
    #     --momentum                  0.97 \
    #     --num_unfrozen_base_layers  1 \
    #     --l2                        0.00014 \
    #     --dropout                   0.585 \
    #     --loss_alpha                0.95 \
    #     --loss_weights_even         0.995 \


    # python run.py \
    #     --source            W \
    #     --target            A \
    #     --from_weights      "./runs/tune_source/vgg16_aug_ft_best/WA/checkpoints/cp-best.ckpt" \
    #     --gpu_id            $GPU_ID \
    #     --experiment_id     $EXPERIMENT_ID \
    #     --seed              $SEED \
    #     --augment           $AUGMENT \
    #     --test_as_val       $TEST_AS_VAL \
    #     --monitor           acc \
    #     --architecture      two_stream_pair_embeds \
    #     --model_base        vgg16 \
    #     --features          images \
    #     --epochs            30 \
    #     --batch_size        16 \
    #     --mode              train_and_test \
    #     --training_regimen  batch_repeat \
    #     --batch_repeats     2 \
    #     --method                            dage \
    #     --connection_type                   source_target \
    #     --weight_type                       indicator \
    #     --connection_filter_type            knn \
    #     --penalty_connection_filter_type    ksd \
    #     --connection_filter_param           1 \
    #     --penalty_connection_filter_param   91 \
    #     --batch_norm                1 \
    #     --optimizer                 adam \
    #     --learning_rate             1e-3 \
    #     --learning_rate_decay       1e-6 \
    #     --momentum                  0.5 \
    #     --num_unfrozen_base_layers  2 \
    #     --l2                        1e-6 \
    #     --dropout                   0.6 \
    #     --loss_alpha                0.06 \
    #     --loss_weights_even         0.84 \

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
        --connection_filter_type            knn \
        --penalty_connection_filter_type    ksd \
        --connection_filter_param           1 \
        --penalty_connection_filter_param   86 \
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
        --connection_filter_type            knn \
        --penalty_connection_filter_type    ksd \
        --connection_filter_param           1 \
        --penalty_connection_filter_param   21 \
        --batch_norm                1 \
        --optimizer                 adam \
        --learning_rate             1.5e-4 \
        --learning_rate_decay       2.1e-3 \
        --momentum                  0.726 \
        --num_unfrozen_base_layers  4 \
        --l2                        1.8e-4 \
        --dropout                   0.68 \
        --loss_alpha                0.10 \
        --loss_weights_even         0.98 \
        
done
        