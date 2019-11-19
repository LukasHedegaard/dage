#!/bin/bash

if [ ! -d "datasets" ]; then
    mkdir "datasets"
fi

if [ ! -d "datasets/Office31" ]; then
    mkdir "datasets/Office31"
fi


# jpeg images
# wget --no-check-certificate -O tmp.tar.gz "https://drive.google.com/uc?export=download&id=0B4IapRTv9pJ1WGZVd1VDMmhwdlE"
./gdrivedl.sh "https://drive.google.com/uc?export=download&id=0B4IapRTv9pJ1WGZVd1VDMmhwdlE" tmp.tar.gz  
tar -xvzf tmp.tar.gz -C datasets/Office31
rm tmp.tar.gz

# decaf features
./gdrivedl.sh "https://drive.google.com/uc?export=download&confirm=3X1c&amp&id=0B4IapRTv9pJ1eloxMmVNQ2IzS00" tmp.tar.gz  
tar -xvzf tmp.tar.gz -C datasets/Office31
rm tmp.tar.gz
mv ./datasets/Office31/amazon/decaf-fts/ ./datasets/Office31/amazon/decaf/
mv ./datasets/Office31/webcam/decaf-fts/ ./datasets/Office31/webcam/decaf/
mv ./datasets/Office31/dslr/decaf-fts/ ./datasets/Office31/dslr/decaf/

# surf features
./gdrivedl.sh "https://drive.google.com/uc?id=0B4IapRTv9pJ1aWxLY0kxN0JJeXM&export=download" tmp.tar.gz  
tar -xvzf tmp.tar.gz -C datasets/Office31
rm tmp.tar.gz
mv ./datasets/Office31/amazon/interest_points/ ./datasets/Office31/amazon/surf/
mv ./datasets/Office31/webcam/interest_points/ ./datasets/Office31/webcam/surf/
mv ./datasets/Office31/dslr/interest_points/ ./datasets/Office31/dslr/surf/

mv ./datasets ../datasets

python3 ../utils/feature_gen.py --feature_extractor vgg16
python3 ../utils/feature_gen.py --feature_extractor resnet101v2