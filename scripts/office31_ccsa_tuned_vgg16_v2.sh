#!/usr/bin/env bash

DESCRIPTION="CCSA evaluation using tuned hyper-parameters (evaluated solely on validation split - test remains independent)"

EXPERIMENT_ID=ccsa_vgg16_tuned_v2
DIR_NAME=./runs/$METHOD/$EXPERIMENT_ID

GPU_ID=3
AUGMENT=1
TEST_AS_VAL=1

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
        --monitor           acc \
        --architecture      two_stream_pair_embeds \
        --model_base        vgg16 \
        --features          images \
        --epochs            40 \
        --batch_size        16 \
        --mode              train_and_test \
        --training_regimen  batch_repeat \
        --batch_repeats     2 \
        --method                    ccsa \
        --connection_filter_param   9.281724963398634 \
        --batch_norm                1 \
        --optimizer                 adam \
        --learning_rate             0.0019258768353446639 \
        --learning_rate_decay       1e-07 \
        --momentum                  0.653670 \
        --num_unfrozen_base_layers  8 \
        --l2                        2.885e-07 \
        --dropout                   0.8 \
        --loss_alpha                0.01 \
        --loss_weights_even         0.9591641239112747 \
        --ratio                     3 \
        --test_as_val               $TEST_AS_VAL \


    python run.py \
        --source            A \
        --target            W \
        --from_weights      "./runs/tune_source/vgg16_aug_ft_best/AW/checkpoints/cp-best.ckpt" \
        --gpu_id            $GPU_ID \
        --experiment_id     $EXPERIMENT_ID \
        --seed              $SEED \
        --augment           $AUGMENT \
        --monitor           acc \
        --architecture      two_stream_pair_embeds \
        --model_base        vgg16 \
        --features          images \
        --epochs            40 \
        --batch_size        16 \
        --mode              train_and_test \
        --training_regimen  batch_repeat \
        --batch_repeats     2 \
        --method                    ccsa \
        --connection_filter_param   10 \
        --batch_norm                1 \
        --optimizer                 adam \
        --learning_rate             0.0999999999999998 \
        --learning_rate_decay       1e-07 \
        --momentum                  0.99 \
        --num_unfrozen_base_layers  0 \
        --l2                        1e-07 \
        --dropout                   0.8 \
        --loss_alpha                0.01 \
        --loss_weights_even         1.0 \
        --ratio                     3 \
        --test_as_val               $TEST_AS_VAL \

    python run.py \
        --source            D \
        --target            A \
        --from_weights      "./runs/tune_source/vgg16_aug_ft_best/DA/checkpoints/cp-best.ckpt" \
        --gpu_id            $GPU_ID \
        --experiment_id     $EXPERIMENT_ID \
        --seed              $SEED \
        --augment           $AUGMENT \
        --monitor           acc \
        --architecture      two_stream_pair_embeds \
        --model_base        vgg16 \
        --features          images \
        --epochs            40 \
        --batch_size        16 \
        --mode              train_and_test \
        --training_regimen  batch_repeat \
        --batch_repeats     2 \
        --method                    ccsa \
        --connection_filter_param   0.001 \
        --batch_norm                1 \
        --optimizer                 adam \
        --learning_rate             0.005504406197621818 \
        --learning_rate_decay       1e-07 \
        --momentum                  0.500 \
        --num_unfrozen_base_layers  6 \
        --l2                        0.001 \
        --dropout                   0.29088097991920614 \
        --loss_alpha                0.08925549717143938 \
        --loss_weights_even         1.0 \
        --ratio                     3 \
        --test_as_val               $TEST_AS_VAL \

    python run.py \
        --source            D \
        --target            W \
        --from_weights      "./runs/tune_source/vgg16_aug_ft_best/DW/checkpoints/cp-best.ckpt" \
        --gpu_id            $GPU_ID \
        --experiment_id     $EXPERIMENT_ID \
        --seed              $SEED \
        --augment           $AUGMENT \
        --monitor           acc \
        --architecture      two_stream_pair_embeds \
        --model_base        vgg16 \
        --features          images \
        --epochs            40 \
        --batch_size        16 \
        --mode              train_and_test \
        --training_regimen  batch_repeat \
        --batch_repeats     2 \
        --method                    ccsa \
        --connection_filter_param   0.001 \
        --batch_norm                1 \
        --optimizer                 adam \
        --learning_rate             0.0008982166853572013 \
        --learning_rate_decay       0.0006535225267466245 \
        --momentum                  0.99 \
        --num_unfrozen_base_layers  16 \
        --l2                        2.219638245619972e-07 \
        --dropout                   0.2332844026654423 \
        --loss_alpha                0.01 \
        --loss_weights_even         0.7391198955493959 \
        --ratio                     3 \
    #     --test_as_val               $TEST_AS_VAL \

    echo $TEST_AS_VAL
    
    python run.py \
        --source            W \
        --target            A \
        --test_as_val       $TEST_AS_VAL \
        --from_weights      "./runs/tune_source/vgg16_aug_ft_best/WA/checkpoints/cp-best.ckpt" \
        --gpu_id            $GPU_ID \
        --experiment_id     $EXPERIMENT_ID \
        --seed              $SEED \
        --augment           $AUGMENT \
        --monitor           acc \
        --architecture      two_stream_pair_embeds \
        --model_base        vgg16 \
        --features          images \
        --epochs            40 \
        --batch_size        16 \
        --mode              train_and_test \
        --training_regimen  batch_repeat \
        --batch_repeats     2 \
        --method                    ccsa \
        --connection_filter_param   10.0 \
        --batch_norm                1 \
        --optimizer                 adam \
        --learning_rate             1e-06 \
        --learning_rate_decay       2.1402978472680186e-06 \
        --momentum                  0.99 \
        --num_unfrozen_base_layers  5
        --l2                        1e-07 \
        --dropout                   0.5741322462547626 \
        --loss_alpha                0.01 \
        --loss_weights_even         0.9649054568494194 \
        --ratio                     3 \
        

    python run.py \
        --source            W \
        --target            D \
        --from_weights      "./runs/tune_source/vgg16_aug_ft_best/WD/checkpoints/cp-best.ckpt" \
        --gpu_id            $GPU_ID \
        --experiment_id     $EXPERIMENT_ID \
        --seed              $SEED \
        --augment           $AUGMENT \
        --monitor           acc \
        --architecture      two_stream_pair_embeds \
        --model_base        vgg16 \
        --features          images \
        --epochs            40 \
        --batch_size        16 \
        --mode              train_and_test \
        --training_regimen  batch_repeat \
        --batch_repeats     2 \
        --method                    ccsa \
        --connection_filter_param   0.001 \
        --batch_norm                1 \
        --optimizer                 adam \
        --learning_rate             0.0001401939841589172 \
        --learning_rate_decay       0.0012009827970416939 \
        --momentum                  0.951664 \
        --num_unfrozen_base_layers  2 \
        --l2                        0.00011314411919848672\
        --dropout                   0.31905527271400036 \
        --loss_alpha                0.14573686239359052 \
        --loss_weights_even         0.11926772021493856 \
        --ratio                     3 \
        --test_as_val               $TEST_AS_VAL \
        
done

./scripts/notify.sh "Finished all jobs: '${EXPERIMENT_ID}-${METHOD}' on GPU ${GPU_ID}."
