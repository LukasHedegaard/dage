#!/usr/bin/env bash
cd datasets

if [ ! -d "datasets" ]; then
    mkdir "datasets"
fi

if [ ! -d "Office31" ]; then
    mkdir "datasets/Office31"
fi

wget --no-check-certificate -O tmp.tar.gz "https://drive.google.com/uc?export=download&id=0B4IapRTv9pJ1WGZVd1VDMmhwdlE"
tar -xvzf tmp.tar.gz -C datasets/Office31
rm tmp.tar.gz