#!/usr/bin/env bash

DESCRIPTION="DAGE variant resembling Linear Discriminant Analysis with hyperparameters tuned for ResNet50"

METHOD=dage
EXPERIMENT_ID=dage_lda_resnet50
DIR_NAME=./runs/$METHOD/$EXPERIMENT_ID

GPU_ID=0
AUGMENT=1
TEST_AS_VAL=0  # Whether to validate on test split, as is usually done

mkdir $DIR_NAME -p
echo $DESCRIPTION > $DIR_NAME/description.txt


for SEED in 1 2 3 4 5
do

    python run.py \
        --source            A \
        --target            D \
        --from_weights      "./runs/tune_source/resnet50_best/AD/checkpoints/cp-best.ckpt" \
        --gpu_id            $GPU_ID \
        --experiment_id     $EXPERIMENT_ID \
        --seed              $SEED \
        --augment           $AUGMENT \
        --monitor           acc \
        --architecture      two_stream_pair_embeds \
        --model_base        resnet50 \
        --features          images \
        --epochs            40 \
        --batch_size        14 \
        --mode              train_and_test \
        --training_regimen  batch_repeat \
        --batch_repeats     2 \
        --method                            $METHOD \
        --connection_type                   source_target \
        --weight_type                       indicator \
        --connection_filter_type            all \
        --penalty_connection_filter_type    all \
        --batch_norm                1 \
        --optimizer                 adam \
        --learning_rate             1e-07 \
        --learning_rate_decay       0.0004 \
        --momentum                  0.5 \
        --num_unfrozen_base_layers  16 \
        --l2                        0.001 \
        --dropout                   0.7 \
        --loss_alpha                0.9 \
        --loss_weights_even         0.0 \
        --ratio                     3 \
        --test_as_val               $TEST_AS_VAL \


    python run.py \
        --source            A \
        --target            W \
        --from_weights      "./runs/tune_source/resnet50_best/AW/checkpoints/cp-best.ckpt" \
        --gpu_id            $GPU_ID \
        --experiment_id     $EXPERIMENT_ID \
        --seed              $SEED \
        --augment           $AUGMENT \
        --monitor           acc \
        --architecture      two_stream_pair_embeds \
        --model_base        resnet50 \
        --features          images \
        --epochs            40 \
        --batch_size        14 \
        --mode              train_and_test \
        --training_regimen  batch_repeat \
        --batch_repeats     2 \
        --method                            $METHOD \
        --connection_type                   source_target \
        --weight_type                       indicator \
        --connection_filter_type            all \
        --penalty_connection_filter_type    all \
        --batch_norm                1 \
        --optimizer                 adam \
        --learning_rate             2.8581910442814035e-05 \
        --learning_rate_decay       1.0223003914901796e-07 \
        --momentum                  0.9633394555661385 \
        --num_unfrozen_base_layers  10 \
        --l2                        7.882227661731842e-05 \
        --dropout                   0.6547491309807579 \
        --loss_alpha                0.18753516715367935 \
        --loss_weights_even         0.9728201941282467 \
        --ratio                     3 \
        --test_as_val               $TEST_AS_VAL \


    python run.py \
        --source            D \
        --target            A \
        --from_weights      "./runs/tune_source/resnet50_best/DA/checkpoints/cp-best.ckpt" \
        --gpu_id            $GPU_ID \
        --experiment_id     $EXPERIMENT_ID \
        --seed              $SEED \
        --augment           $AUGMENT \
        --monitor           acc \
        --architecture      two_stream_pair_embeds \
        --model_base        resnet50 \
        --features          images \
        --epochs            40 \
        --batch_size        14 \
        --mode              train_and_test \
        --training_regimen  batch_repeat \
        --batch_repeats     2 \
        --method                            $METHOD \
        --connection_type                   source_target \
        --weight_type                       indicator \
        --connection_filter_type            all \
        --penalty_connection_filter_type    all \
        --batch_norm                1 \
        --optimizer                 adam \
        --learning_rate             6.860133325237891e-05 \
        --learning_rate_decay       1.2830450332840693e-06 \
        --momentum                  0.9775559683538072 \
        --num_unfrozen_base_layers  15 \
        --l2                        9.301551770733569e-05 \
        --dropout                   0.6923666267152143 \
        --loss_alpha                0.5066486737346229 \
        --loss_weights_even         0.8186001108645153 \
        --ratio                     3 \
        --test_as_val               $TEST_AS_VAL \

    python run.py \
        --source            D \
        --target            W \
        --from_weights      "./runs/tune_source/resnet50_best/DW/checkpoints/cp-best.ckpt" \
        --gpu_id            $GPU_ID \
        --experiment_id     $EXPERIMENT_ID \
        --seed              $SEED \
        --augment           $AUGMENT \
        --monitor           acc \
        --architecture      two_stream_pair_embeds \
        --model_base        resnet50 \
        --features          images \
        --epochs            40 \
        --batch_size        14 \
        --mode              train_and_test \
        --training_regimen  batch_repeat \
        --batch_repeats     2 \
        --method                            $METHOD \
        --connection_type                   source_target \
        --weight_type                       indicator \
        --connection_filter_type            all \
        --penalty_connection_filter_type    all \
        --batch_norm                1 \
        --optimizer                 adam \
        --learning_rate             0.0001 \
        --learning_rate_decay       1e-07 \
        --momentum                  0.99 \
        --num_unfrozen_base_layers  16 \
        --l2                        1e-07 \
        --dropout                   0.1082474988943114 \
        --loss_alpha                0.05 \
        --loss_weights_even         0.40446887430632233 \
        --ratio                     3 \
        --test_as_val               $TEST_AS_VAL \

    python run.py \
        --source            W \
        --target            A \
        --from_weights      "./runs/tune_source/resnet50_best/WA/checkpoints/cp-best.ckpt" \
        --gpu_id            $GPU_ID \
        --experiment_id     $EXPERIMENT_ID \
        --seed              $SEED \
        --augment           $AUGMENT \
        --monitor           acc \
        --architecture      two_stream_pair_embeds \
        --model_base        resnet50 \
        --features          images \
        --epochs            40 \
        --batch_size        14 \
        --mode              train_and_test \
        --training_regimen  batch_repeat \
        --batch_repeats     2 \
        --method                            $METHOD \
        --connection_type                   source_target \
        --weight_type                       indicator \
        --connection_filter_type            all \
        --penalty_connection_filter_type    all \
        --batch_norm                1 \
        --optimizer                 adam \
        --learning_rate             0.1 \
        --learning_rate_decay       1.8415741876643152e-07 \
        --momentum                  0.8656278183961982 \
        --num_unfrozen_base_layers  3 \
        --l2                        0.001 \
        --dropout                   0.7 \
        --loss_alpha                0.6066964950198714 \
        --loss_weights_even         1.0 \
        --ratio                     3 \
        --test_as_val               $TEST_AS_VAL \

    python run.py \
        --source            W \
        --target            D \
        --from_weights      "./runs/tune_source/resnet50_best/WD/checkpoints/cp-best.ckpt" \
        --gpu_id            $GPU_ID \
        --experiment_id     $EXPERIMENT_ID \
        --seed              $SEED \
        --augment           $AUGMENT \
        --monitor           acc \
        --architecture      two_stream_pair_embeds \
        --model_base        resnet50 \
        --features          images \
        --epochs            40 \
        --batch_size        14 \
        --mode              train_and_test \
        --training_regimen  batch_repeat \
        --batch_repeats     2 \
        --method                            $METHOD \
        --connection_type                   source_target \
        --weight_type                       indicator \
        --connection_filter_type            all \
        --penalty_connection_filter_type    all \
        --batch_norm                1 \
        --optimizer                 adam \
        --learning_rate             9.94871683276103e-06 \
        --learning_rate_decay       9.762574909039597e-07 \
        --momentum                  0.8989358320276817 \
        --num_unfrozen_base_layers  1 \
        --l2                        3.4017899911285423e-06 \
        --dropout                   0.4367460555086208 \
        --loss_alpha                0.8759552810807384 \
        --loss_weights_even         0.8489138242660841 \
        --ratio                     3 \
        --test_as_val               $TEST_AS_VAL \
        
done

./scripts/notify.sh "Finished all jobs: 'resnet50-${EXPERIMENT_ID}-${METHOD}' on GPU ${GPU_ID}."