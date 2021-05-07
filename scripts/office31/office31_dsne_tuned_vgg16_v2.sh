#!/usr/bin/env bash

DESCRIPTION="d-SNE evaluation using tuned hyper-parameters (evaluated solely on validation split - test remains independent)"

EXPERIMENT_ID=dsne_vgg16_tuned_v2
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
        --method                    dsne \
        --connection_filter_param   1.1173938542646114 \
        --batch_norm                1 \
        --optimizer                 adam \
        --learning_rate             0.1 \
        --learning_rate_decay       1e-07 \
        --momentum                  0.955818 \
        --num_unfrozen_base_layers  16 \
        --l2                        0.0002497151670863988 \
        --dropout                   0.630074631495249 \
        --loss_alpha                0.01 \
        --loss_weights_even         0.5046928524799381 \
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
        --method                    dsne \
        --connection_filter_param   10.0 \
        --batch_norm                1 \
        --optimizer                 adam \
        --learning_rate             1e-06 \
        --learning_rate_decay       1e-07 \
        --momentum                  0.99 \
        --num_unfrozen_base_layers  9 \
        --l2                        1e-07 \
        --dropout                   0.3472006273976478 \
        --loss_alpha                0.21847886773019742 \
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
        --method                    dsne \
        --connection_filter_param   3.9816444831848066 \
        --batch_norm                1 \
        --optimizer                 adam \
        --learning_rate             1e-06 \
        --learning_rate_decay       1e-07 \
        --momentum                  0.967139 \
        --num_unfrozen_base_layers  16 \
        --l2                        0.00016704646590781953 \
        --dropout                   0.5164691104065059 \
        --loss_alpha                0.01 \
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
        --method                    dsne \
        --connection_filter_param   3.8656695461522266 \
        --batch_norm                1 \
        --optimizer                 adam \
        --learning_rate             0.0007571967519844236 \
        --learning_rate_decay       0.00086222512977251 \
        --momentum                  0.926876 \
        --num_unfrozen_base_layers  15 \
        --l2                        1.1191957079089307e-06 \
        --dropout                   0.6937478671877458 \
        --loss_alpha                0.01 \
        --loss_weights_even         0.25312576364917916 \
        --ratio                     3 \
        --test_as_val               $TEST_AS_VAL \


    python run.py \
        --source            W \
        --target            A \
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
        --method                    dsne \
        --connection_filter_param   10.0 \
        --batch_norm                1 \
        --optimizer                 adam \
        --learning_rate             0.0006437615199162743 \
        --learning_rate_decay       0.0008806429613626282 \
        --momentum                  0.5 \
        --num_unfrozen_base_layers  6 \
        --l2                        3.132589223847626e-05 \
        --dropout                   0.8 \
        --loss_alpha                0.01 \
        --loss_weights_even         1.0 \
        --ratio                     3 \
        --test_as_val               $TEST_AS_VAL \

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
        --method                    dsne \
        --connection_filter_param   0.6063816037570542 \
        --batch_norm                1 \
        --optimizer                 adam \
        --learning_rate             5.4425537355368425e-0 \
        --learning_rate_decay       1e-07 \
        --momentum                  0.5 \
        --num_unfrozen_base_layers  1 \
        --l2                        2.8727330959846286e-07\
        --dropout                   0.3065450511281586 \
        --loss_alpha                0.01 \
        --loss_weights_even         0.7788139702663768 \
        --ratio                     3 \
        --test_as_val               $TEST_AS_VAL \
        
done

./scripts/notify.sh "Finished all jobs: '${EXPERIMENT_ID}-${METHOD}' on GPU ${GPU_ID}."
