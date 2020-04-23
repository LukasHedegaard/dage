#!/usr/bin/env bash
DESCRIPTION="Hyperparameter search for MNIST -> USPS experiments"

GPU_ID=3
METHOD="dsne"
EXPERIMENT="digits"

if python3 hypersearch.py   \
    --id                 ${EXPERIMENT}-${METHOD}-M-U  \
    --n_calls          100  \
    --verbose            1  \
    --n_random_starts   10  \
    --acq_func          EI  \
    --seed              42  \
    --method            $METHOD \
    --source            "mnist" \
    --target            "usps" \
    --experiment        $EXPERIMENT \
    --gpu_id            $GPU_ID \
; then
    ./scripts/notify.sh "Finished job: 'hypersearch-${EXPERIMENT}-${METHOD}-M-U' on GPU ${GPU_ID}."
else
    ./scripts/notify.sh "Error in job: 'hypersearch-${EXPERIMENT}-${METHOD}-M-U' on GPU ${GPU_ID}."
    exit 1
fi

./scripts/notify.sh "Finished all jobs: 'hypersearch-${EXPERIMENT}-${METHOD}' on GPU ${GPU_ID}."
