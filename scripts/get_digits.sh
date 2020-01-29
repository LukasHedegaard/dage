#!/usr/bin/env bash

if [ ! -d "datasets/digits" ]; then
    mkdir "datasets/digits"
fi

# MNIST 
echo "MNIST is included in Tensorflow"

# SVHN is included in Tensorflow
echo "SVHN is included in Tensorflow"

# MNIST-M
echo "MNIST-M downloading"
./scripts/gdrivedl.sh "https://drive.google.com/open?id=0B_tExHiYS-0veklUZHFYT19KYjg" tmp.tar.gz  
tar -xzf tmp.tar.gz -C datasets/digits
rm tmp.tar.gz

# USPS
echo "USPS downloading"
if [ ! -d "datasets/digits/usps" ]; then
    mkdir "datasets/digits/usps"
fi

wget https://cs.nyu.edu/~roweis/data/usps_all.mat -O datasets/Digits/usps/usps_all.mat