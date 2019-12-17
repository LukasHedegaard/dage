#!/usr/bin/env bash
DESCRIPTION="Test which regularisation strategy works best. We test batch norm, dropout, and L2 here."

EXPERIMENT_ID_BASE=regularization_test
METHOD=multitask

for BN in 0 1
do
    for L2 in 0.0001 0.00001 0
    do
        for DROPOUT in 0 0.25 0.5
        do
            EXPERIMENT_ID="${EXPERIMENT_ID_BASE}_bn_${BN}_drop_${DROPOUT}_l2_${L2}"

            DIR_NAME=./runs/$METHOD/$EXPERIMENT_ID
            mkdir $DIR_NAME -p
            echo $DESCRIPTION > $DIR_NAME/description.txt

            for SEED in 0 1 2 4 5
            do
                for SOURCE in A 
                do
                    for TARGET in D 
                    do
                        if [ $SOURCE != $TARGET ]
                        then
                            TIMESTAMP=$(date '+%Y%m%d%H%M%S')

                            python3 run.py \
                                --gpu_id            1 \
                                --learning_rate     0.001 \
                                --optimizer         adam \
                                --dropout           $DROPOUT \
                                --batch_norm        $BN \
                                --l2                $L2 \
                                --experiment_id     $EXPERIMENT_ID \
                                --source            $SOURCE \
                                --target            $TARGET \
                                --seed              $SEED \
                                --method            $METHOD \
                                --architecture      two_stream_pair_embeds \
                                --model_base        none \
                                --features          vgg16 \
                                --epochs            30 \
                                --batch_size        16 \
                                --augment           0 \
                                --timestamp         $TIMESTAMP \

                            FT_RUN_DIR=./runs/$METHOD/$EXPERIMENT_ID/${SOURCE}${TARGET}_${SEED}_${TIMESTAMP}

                            if [ ! -f "$FT_RUN_DIR/report.json" ]; then
                                rm -rf $FT_RUN_DIR
                            else
                                rm -rf $FT_RUN_DIR/checkpoints
                            fi
                        fi
                    done
                done
            done
        done
    done
done

./scripts/notify.sh "Finished job: ${METHOD}/${EXPERIMENT_ID_BASE} on GPU ${GPU_ID}."