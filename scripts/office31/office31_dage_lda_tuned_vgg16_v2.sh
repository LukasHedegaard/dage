#!/usr/bin/env bash

DESCRIPTION="DAGE variant resembling Linear Discriminant Analysis with hyperparameters tuned for VGG16 (evaluated solely on validation split - test remains independent)"

EXPERIMENT_ID=dage_lda_vgg16_tuned_v2
DIR_NAME=./runs/$METHOD/$EXPERIMENT_ID

GPU_ID=1
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
        --method                            dage \
        --connection_type                   source_target \
        --weight_type                       indicator \
        --connection_filter_type            all \
        --penalty_connection_filter_type    all \
        --batch_norm                0 \
        --optimizer                 adam \
        --learning_rate             0.1 \
        --learning_rate_decay       7.441033318021644e-05 \
        --momentum                  0.734622 \
        --num_unfrozen_base_layers  2 \
        --l2                        1.1770915686993488e-06 \
        --dropout                   0.45081593275003584 \
        --loss_alpha                0.01 \
        --loss_weights_even         0.6591223159783306 \
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
        --method                            dage \
        --connection_type                   source_target \
        --weight_type                       indicator \
        --connection_filter_type            all \
        --penalty_connection_filter_type    all \
        --batch_norm                1 \
        --optimizer                 adam \
        --learning_rate             1e-06 \
        --learning_rate_decay       2.29e-04 \
        --momentum                  0.500000 \
        --num_unfrozen_base_layers  0 \
        --l2                        0.001 \
        --dropout                   0.8 \
        --loss_alpha                0.01 \
        --loss_weights_even         1.00 \
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
        --method                            dage \
        --connection_type                   source_target \
        --weight_type                       indicator \
        --connection_filter_type            all \
        --penalty_connection_filter_type    all \
        --batch_norm                1 \
        --optimizer                 adam \
        --learning_rate             0.09343901699717491 \
        --learning_rate_decay       0.0010819470522114175 \
        --momentum                  0.910902 \
        --num_unfrozen_base_layers  7 \
        --l2                        4.2227731787064076e-07 \
        --dropout                   0.7220232911292304 \
        --loss_alpha                0.19206722852749758 \
        --loss_weights_even         0.5135148570915419 \
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
        --method                            dage \
        --connection_type                   source_target \
        --weight_type                       indicator \
        --connection_filter_type            all \
        --penalty_connection_filter_type    all \
        --batch_norm                1 \
        --optimizer                 adam \
        --learning_rate             1e-06 \
        --learning_rate_decay       0.01 \
        --momentum                  0.500 \
        --num_unfrozen_base_layers  16 \
        --l2                        3.665648387614544e-06 \
        --dropout                   0.3635848431426377 \
        --loss_alpha                0.8858444702410938 \
        --loss_weights_even         0.29271682692321105 \
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
        --method                            dage \
        --connection_type                   source_target \
        --weight_type                       indicator \
        --connection_filter_type            all \
        --penalty_connection_filter_type    all \
        --batch_norm                1 \
        --optimizer                 adam \
        --learning_rate             0.1 \
        --learning_rate_decay       1e-07 \
        --momentum                  0.962777 \
        --num_unfrozen_base_layers  3 \
        --l2                        0.001 \
        --dropout                   0.8 \
        --loss_alpha                0.01 \
        --loss_weights_even         0.7287425562747809 \
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
        --method                            dage \
        --connection_type                   source_target \
        --weight_type                       indicator \
        --connection_filter_type            all \
        --penalty_connection_filter_type    all \
        --batch_norm                1 \
        --optimizer                 adam \
        --learning_rate             0.08186552414176086 \
        --learning_rate_decay       1e-07 \
        --momentum                  0.500 \
        --num_unfrozen_base_layers  3 \
        --l2                        1.3900738319640757e-07 \
        --dropout                   0.1 \
        --loss_alpha                0.01 \
        --loss_weights_even         0.0 \
        --ratio                     3 \
        --test_as_val               $TEST_AS_VAL \
        
done

./scripts/notify.sh "Finished all jobs: 'resnet50-${EXPERIMENT_ID}-${METHOD}' on GPU ${GPU_ID}."
