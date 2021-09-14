#!/bin/bash

# If this script fails, please use the official instructions at 
# https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification

if [ ! -d "datasets" ]; then
    mkdir "datasets"
fi

if [ ! -d "datasets/visda-c" ]; then
    mkdir "datasets/visda-c"
fi

wget http://csr.bu.edu/ftp/visda17/clf/train.tar
tar xvf train.tar -C datasets/visda-c

wget http://csr.bu.edu/ftp/visda17/clf/validation.tar
tar xvf validation.tar -C datasets/visda-c
