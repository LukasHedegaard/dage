#!/usr/bin/env bash

TFDS_PATH=~/tensorflow_datasets/downloads/manual/
if [ ! -d $TFDS_PATH ]; then
    mkdir $TFDS_PATH
fi

# MNIST 
echo "MNIST is included in Tensorflow Datasets"

# SVHN is included in Tensorflow
echo "SVHN is included in Tensorflow Datasets"

# MNIST-M
echo "MNIST-M downloading"

./scripts/gdrivedl.sh \
    "https://drive.google.com/open?id=0B_tExHiYS-0veklUZHFYT19KYjg" \
    "${TFDS_PATH}tmp.tar.gz"

tar -xzf "${TFDS_PATH}tmp.tar.gz" -C "${TFDS_PATH}"
rm "${TFDS_PATH}tmp.tar.gz"

# ./scripts/gdrivedl.sh "https://drive.google.com/open?id=0B_tExHiYS-0veklUZHFYT19KYjg" tmp.tar.gz  
# tar -xzf tmp.tar.gz -C datasets/digits
# rm tmp.tar.gz

# USPS
echo "USPS downloading"
if [ ! -d "${TFDS_PATH}usps" ]; then
    mkdir "${TFDS_PATH}usps"
fi

wget https://cs.nyu.edu/~roweis/data/usps_all.mat -O "${TFDS_PATH}usps/usps_all.mat"