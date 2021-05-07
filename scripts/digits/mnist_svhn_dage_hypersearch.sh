#!/usr/bin/env bash
DESCRIPTION="Hyperparameter search for digits experiments"

GPU_ID=0
METHOD="dage"
EXPERIMENT="digits"
SOURCE="mnist"
TARGET="svhn"

python3 hypersearch.py  \
    --id                 ${EXPERIMENT}-${METHOD}-${SOURCE}-${TARGET}   \
    --n_calls           50  \
    --verbose            1  \
    --n_random_starts   10  \
    --acq_func          EI  \
    --seed              42  \
    --method            $METHOD \
    --source            $SOURCE \
    --target            $TARGET \
    --experiment        $EXPERIMENT \
    --gpu_id            $GPU_ID \

./scripts/notify.sh "Finished all jobs: 'hypersearch-${EXPERIMENT}-${METHOD}' on GPU ${GPU_ID}."
