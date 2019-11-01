#!/usr/bin/env bash

python run.py --method ccsa --source A --target D --model_base vgg16 --epochs 20 --seed 1 --augment 1 --alpha 0
python run.py --method ccsa --source A --target D --model_base vgg16 --epochs 20 --seed 2 --augment 1 --alpha 0
python run.py --method ccsa --source A --target D --model_base vgg16 --epochs 20 --seed 3 --augment 1 --alpha 0

python run.py --method ccsa --source A --target D --model_base vgg16 --epochs 20 --seed 1 --augment 1 --alpha 0.25
python run.py --method ccsa --source A --target D --model_base vgg16 --epochs 20 --seed 2 --augment 1 --alpha 0.25
python run.py --method ccsa --source A --target D --model_base vgg16 --epochs 20 --seed 3 --augment 1 --alpha 0.25

python run.py --method ccsa --source A --target D --model_base vgg16 --epochs 20 --seed 1 --augment 1 --alpha 0.5
python run.py --method ccsa --source A --target D --model_base vgg16 --epochs 20 --seed 2 --augment 1 --alpha 0.5
python run.py --method ccsa --source A --target D --model_base vgg16 --epochs 20 --seed 3 --augment 1 --alpha 0.5
