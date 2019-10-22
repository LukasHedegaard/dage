#!/usr/bin/env bash
# This scripts is used to tune the pretrained model for office 31 experiments

# Repeat 1
python run.py --method tune_source --source A --target D --model_base vgg16 --epochs 75 --seed 1 --augment  True
python run.py --method tune_source --source A --target W --model_base vgg16 --epochs 75 --seed 1 --augment  True
python run.py --method tune_source --source D --target A --model_base vgg16 --epochs 75 --seed 1 --augment  True
python run.py --method tune_source --source D --target W --model_base vgg16 --epochs 75 --seed 1 --augment  True
python run.py --method tune_source --source W --target A --model_base vgg16 --epochs 75 --seed 1 --augment  True
python run.py --method tune_source --source W --target D --model_base vgg16 --epochs 75 --seed 1 --augment  True

# Repeat 2
python run.py --method tune_source --source A --target D --model_base vgg16 --epochs 75 --seed 2 --augment  True
python run.py --method tune_source --source A --target W --model_base vgg16 --epochs 75 --seed 2 --augment  True
python run.py --method tune_source --source D --target A --model_base vgg16 --epochs 75 --seed 2 --augment  True
python run.py --method tune_source --source D --target W --model_base vgg16 --epochs 75 --seed 2 --augment  True
python run.py --method tune_source --source W --target A --model_base vgg16 --epochs 75 --seed 2 --augment  True
python run.py --method tune_source --source W --target D --model_base vgg16 --epochs 75 --seed 2 --augment  True

# Repeat 3
python run.py --method tune_source --source A --target D --model_base vgg16 --epochs 75 --seed 3 --augment  True
python run.py --method tune_source --source A --target W --model_base vgg16 --epochs 75 --seed 3 --augment  True
python run.py --method tune_source --source D --target A --model_base vgg16 --epochs 75 --seed 3 --augment  True
python run.py --method tune_source --source D --target W --model_base vgg16 --epochs 75 --seed 3 --augment  True
python run.py --method tune_source --source W --target A --model_base vgg16 --epochs 75 --seed 3 --augment  True
python run.py --method tune_source --source W --target D --model_base vgg16 --epochs 75 --seed 3 --augment  True

# Repeat 4
python run.py --method tune_source --source A --target D --model_base vgg16 --epochs 75 --seed 4 --augment  True
python run.py --method tune_source --source A --target W --model_base vgg16 --epochs 75 --seed 4 --augment  True
python run.py --method tune_source --source D --target A --model_base vgg16 --epochs 75 --seed 4 --augment  True
python run.py --method tune_source --source D --target W --model_base vgg16 --epochs 75 --seed 4 --augment  True
python run.py --method tune_source --source W --target A --model_base vgg16 --epochs 75 --seed 4 --augment  True
python run.py --method tune_source --source W --target D --model_base vgg16 --epochs 75 --seed 4 --augment  True

# Repeat 5
python run.py --method tune_source --source A --target D --model_base vgg16 --epochs 75 --seed 5 --augment  True
python run.py --method tune_source --source A --target W --model_base vgg16 --epochs 75 --seed 5 --augment  True
python run.py --method tune_source --source D --target A --model_base vgg16 --epochs 75 --seed 5 --augment  True
python run.py --method tune_source --source D --target W --model_base vgg16 --epochs 75 --seed 5 --augment  True
python run.py --method tune_source --source W --target A --model_base vgg16 --epochs 75 --seed 5 --augment  True
python run.py --method tune_source --source W --target D --model_base vgg16 --epochs 75 --seed 5 --augment  True
