#!/usr/bin/env bash
DESCRIPTION="Hyperparameter search for Office31 experiments"

GPU_ID=2
METHOD="dage"
EXPERIMENT="office"

for SOURCE in A #W D
do
    for TARGET in D #A W
    do
        if [ $SOURCE != $TARGET ]
        then
            if python3 hypersearch.py   \
                --id                 ${EXPERIMENT}-${METHOD}-${SOURCE}-${TARGET}   \
                --verbose            1  \
                --n_random_starts   10  \
                --acq_func          EI  \
                --seed              42  \
                --method            $METHOD \
                --source            $SOURCE \
                --target            $TARGET \
                --experiment        $EXPERIMENT \
                --gpu_id            $GPU_ID \
            ; then
                ./scripts/notify.sh "Finished job: 'hypersearch-${EXPERIMENT}-${METHOD}-${SOURCE}-${TARGET}' on GPU ${GPU_ID}."
            else
                ./scripts/notify.sh "Error in job: 'hypersearch-${EXPERIMENT}-${METHOD}-${SOURCE}-${TARGET}' on GPU ${GPU_ID}."
                exit 1
            fi
        fi
    done
done

./scripts/notify.sh "Finished all jobs: 'hypersearch-${EXPERIMENT}-${METHOD}' on GPU ${GPU_ID}."
