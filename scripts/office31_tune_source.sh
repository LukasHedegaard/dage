#!/usr/bin/env bash
# This scripts is used to tune the pretrained model for office 31 experiments

# TODO: modify to fit this code
python run.py --method tune_source --source A --target D # --bb vgg --nlayers 16 --pretrained --model-path pretrained/vgg16.params --hybridize --train-source --end-epoch 50 --postfix p0 --flip --random-crop --random-color --cfg cfg/office31-t.json
# python run.py --method tune_source --source A --target W --bb vgg --nlayers 16 --pretrained --model-path pretrained/vgg16.params --hybridize --train-source --end-epoch 50 --postfix p0 --flip --random-crop --random-color --cfg cfg/office31-t.json
# python run.py --method tune_source --source D --target A --bb vgg --nlayers 16 --pretrained --model-path pretrained/vgg16.params --hybridize --train-source --end-epoch 50 --postfix p0 --flip --random-crop --random-color --cfg cfg/office31-t.json
# python run.py --method tune_source --source D --target W --bb vgg --nlayers 16 --pretrained --model-path pretrained/vgg16.params --hybridize --train-source --end-epoch 50 --postfix p0 --flip --random-crop --random-color --cfg cfg/office31-t.json
# python run.py --method tune_source --source W --target A --bb vgg --nlayers 16 --pretrained --model-path pretrained/vgg16.params --hybridize --train-source --end-epoch 50 --postfix p0 --flip --random-crop --random-color --cfg cfg/office31-t.json
# python run.py --method tune_source --source W --target D --bb vgg --nlayers 16 --pretrained --model-path pretrained/vgg16.params --hybridize --train-source --end-epoch 50 --postfix p0 --flip --random-crop --random-color --cfg cfg/office31-t.json


# python run.py --method tune_source --source A --target D --bb vgg --nlayers 16 --pretrained --model-path pretrained/vgg16.params --hybridize --train-source --end-epoch 50 --postfix p0-l2n --flip --cfg cfg/office31-t.json --log-itv 10000 --l2n --lr 0.01
# python run.py --method tune_source --source A --target W --bb vgg --nlayers 16 --pretrained --model-path pretrained/vgg16.params --hybridize --train-source --end-epoch 50 --postfix p0-l2n --flip --cfg cfg/office31-t.json --log-itv 10000 --l2n --lr 0.01
# python run.py --method tune_source --source D --target A --bb vgg --nlayers 16 --pretrained --model-path pretrained/vgg16.params --hybridize --train-source --end-epoch 50 --postfix p0-l2n --flip --cfg cfg/office31-t.json --log-itv 10000 --l2n --lr 0.01
# python run.py --method tune_source --source D --target W --bb vgg --nlayers 16 --pretrained --model-path pretrained/vgg16.params --hybridize --train-source --end-epoch 50 --postfix p0-l2n --flip --cfg cfg/office31-t.json --log-itv 10000 --l2n --lr 0.01
# python run.py --method tune_source --source W --target A --bb vgg --nlayers 16 --pretrained --model-path pretrained/vgg16.params --hybridize --train-source --end-epoch 50 --postfix p0-l2n --flip --cfg cfg/office31-t.json --log-itv 10000 --l2n --lr 0.01
# python run.py --method tune_source --source W --target D --bb vgg --nlayers 16 --pretrained --model-path pretrained/vgg16.params --hybridize --train-source --end-epoch 50 --postfix p0-l2n --flip --cfg cfg/office31-t.json --log-itv 10000 --l2n --lr 0.01


# # Repeat 2
# python run.py --method tune_source --source A --target D --bb resnet --nlayers 101 --pretrained --model-path pretrained/resnet101_v2.params --hybridize --train-source --end-epoch 50 --postfix p1 --flip --cfg cfg/office31-t.json --log-itv 10000
# python run.py --method tune_source --source A --target W --bb resnet --nlayers 101 --pretrained --model-path pretrained/resnet101_v2.params --hybridize --train-source --end-epoch 50 --postfix p1 --flip --cfg cfg/office31-t.json --log-itv 10000
# python run.py --method tune_source --source D --target A --bb resnet --nlayers 101 --pretrained --model-path pretrained/resnet101_v2.params --hybridize --train-source --end-epoch 50 --postfix p1 --flip --cfg cfg/office31-t.json --log-itv 10000
# python run.py --method tune_source --source D --target W --bb resnet --nlayers 101 --pretrained --model-path pretrained/resnet101_v2.params --hybridize --train-source --end-epoch 50 --postfix p1 --flip --cfg cfg/office31-t.json --log-itv 10000
# python run.py --method tune_source --source W --target A --bb resnet --nlayers 101 --pretrained --model-path pretrained/resnet101_v2.params --hybridize --train-source --end-epoch 50 --postfix p1 --flip --cfg cfg/office31-t.json --log-itv 10000
# python run.py --method tune_source --source W --target D --bb resnet --nlayers 101 --pretrained --model-path pretrained/resnet101_v2.params --hybridize --train-source --end-epoch 50 --postfix p1 --flip --cfg cfg/office31-t.json --log-itv 10000

# python run.py --method tune_source --source A --target D --bb vgg --nlayers 16 --pretrained --model-path pretrained/vgg16.params --hybridize --train-source --end-epoch 50 --postfix p1-l2n --flip --cfg cfg/office31-t.json --log-itv 10000 --l2n --lr 0.01
# python run.py --method tune_source --source A --target W --bb vgg --nlayers 16 --pretrained --model-path pretrained/vgg16.params --hybridize --train-source --end-epoch 50 --postfix p1-l2n --flip --cfg cfg/office31-t.json --log-itv 10000 --l2n --lr 0.01
# python run.py --method tune_source --source D --target A --bb vgg --nlayers 16 --pretrained --model-path pretrained/vgg16.params --hybridize --train-source --end-epoch 50 --postfix p1-l2n --flip --cfg cfg/office31-t.json --log-itv 10000 --l2n --lr 0.01
# python run.py --method tune_source --source D --target W --bb vgg --nlayers 16 --pretrained --model-path pretrained/vgg16.params --hybridize --train-source --end-epoch 50 --postfix p1-l2n --flip --cfg cfg/office31-t.json --log-itv 10000 --l2n --lr 0.01
# python run.py --method tune_source --source W --target A --bb vgg --nlayers 16 --pretrained --model-path pretrained/vgg16.params --hybridize --train-source --end-epoch 50 --postfix p1-l2n --flip --cfg cfg/office31-t.json --log-itv 10000 --l2n --lr 0.01
# python run.py --method tune_source --source W --target D --bb vgg --nlayers 16 --pretrained --model-path pretrained/vgg16.params --hybridize --train-source --end-epoch 50 --postfix p1-l2n --flip --cfg cfg/office31-t.json --log-itv 10000 --l2n --lr 0.01

# # Repeat 3
# python run.py --method tune_source --source A --target D --bb vgg --nlayers 16 --pretrained --model-path pretrained/vgg16.params --hybridize --train-source --end-epoch 20 --postfix p2 --flip --cfg cfg/office31-t.json --log-itv 10000
# python run.py --method tune_source --source A --target W --bb vgg --nlayers 16 --pretrained --model-path pretrained/vgg16.params --hybridize --train-source --end-epoch 20 --postfix p2 --flip --cfg cfg/office31-t.json --log-itv 10000
# python run.py --method tune_source --source D --target A --bb vgg --nlayers 16 --pretrained --model-path pretrained/vgg16.params --hybridize --train-source --end-epoch 20 --postfix p2 --flip --cfg cfg/office31-t.json --log-itv 10000
# python run.py --method tune_source --source D --target W --bb vgg --nlayers 16 --pretrained --model-path pretrained/vgg16.params --hybridize --train-source --end-epoch 10 --postfix p2 --flip --cfg cfg/office31-t.json --log-itv 10000
# python run.py --method tune_source --source W --target A --bb vgg --nlayers 16 --pretrained --model-path pretrained/vgg16.params --hybridize --train-source --end-epoch 20 --postfix p2 --flip --cfg cfg/office31-t.json --log-itv 10000
# python run.py --method tune_source --source W --target D --bb vgg --nlayers 16 --pretrained --model-path pretrained/vgg16.params --hybridize --train-source --end-epoch 10 --postfix p2 --flip --cfg cfg/office31-t.json --log-itv 10000

# python run.py --method tune_source --source A --target D --bb resnet --nlayers 101 --pretrained --model-path pretrained/resnet101_v2.params --hybridize --train-source --end-epoch 50 --postfix p2 --flip --cfg cfg/office31-t.json --log-itv 10000
# python run.py --method tune_source --source A --target W --bb resnet --nlayers 101 --pretrained --model-path pretrained/resnet101_v2.params --hybridize --train-source --end-epoch 50 --postfix p2 --flip --cfg cfg/office31-t.json --log-itv 10000
# python run.py --method tune_source --source D --target A --bb resnet --nlayers 101 --pretrained --model-path pretrained/resnet101_v2.params --hybridize --train-source --end-epoch 50 --postfix p2 --flip --cfg cfg/office31-t.json --log-itv 10000
# python run.py --method tune_source --source D --target W --bb resnet --nlayers 101 --pretrained --model-path pretrained/resnet101_v2.params --hybridize --train-source --end-epoch 50 --postfix p2 --flip --cfg cfg/office31-t.json --log-itv 10000
# python run.py --method tune_source --source W --target A --bb resnet --nlayers 101 --pretrained --model-path pretrained/resnet101_v2.params --hybridize --train-source --end-epoch 50 --postfix p2 --flip --cfg cfg/office31-t.json --log-itv 10000
# python run.py --method tune_source --source W --target D --bb resnet --nlayers 101 --pretrained --model-path pretrained/resnet101_v2.params --hybridize --train-source --end-epoch 50 --postfix p2 --flip --cfg cfg/office31-t.json --log-itv 10000

# python run.py --method tune_source --source A --target D --bb vgg --nlayers 16 --pretrained --model-path pretrained/vgg16.params --hybridize --train-source --end-epoch 50 --postfix p2-l2n --flip --cfg cfg/office31-t.json --log-itv 10000 --l2n --lr 0.01
# python run.py --method tune_source --source A --target W --bb vgg --nlayers 16 --pretrained --model-path pretrained/vgg16.params --hybridize --train-source --end-epoch 50 --postfix p2-l2n --flip --cfg cfg/office31-t.json --log-itv 10000 --l2n --lr 0.01
# python run.py --method tune_source --source D --target A --bb vgg --nlayers 16 --pretrained --model-path pretrained/vgg16.params --hybridize --train-source --end-epoch 50 --postfix p2-l2n --flip --cfg cfg/office31-t.json --log-itv 10000 --l2n --lr 0.01
# python run.py --method tune_source --source D --target W --bb vgg --nlayers 16 --pretrained --model-path pretrained/vgg16.params --hybridize --train-source --end-epoch 50 --postfix p2-l2n --flip --cfg cfg/office31-t.json --log-itv 10000 --l2n --lr 0.01
# python run.py --method tune_source --source W --target A --bb vgg --nlayers 16 --pretrained --model-path pretrained/vgg16.params --hybridize --train-source --end-epoch 50 --postfix p2-l2n --flip --cfg cfg/office31-t.json --log-itv 10000 --l2n --lr 0.01
# python run.py --method tune_source --source W --target D --bb vgg --nlayers 16 --pretrained --model-path pretrained/vgg16.params --hybridize --train-source --end-epoch 50 --postfix p2-l2n --flip --cfg cfg/office31-t.json --log-itv 10000 --l2n --lr 0.01
