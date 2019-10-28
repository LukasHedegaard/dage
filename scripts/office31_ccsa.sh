#!/usr/bin/env bash

# Repeat 1
python run.py --method ccsa --source A --target D --model_base vgg16 --epochs 20 --seed 1 --augment True --alpha 0.01
python run.py --method ccsa --source A --target W --model_base vgg16 --epochs 20 --seed 1 --augment True --alpha 0.01
python run.py --method ccsa --source D --target A --model_base vgg16 --epochs 20 --seed 1 --augment True --alpha 0.01
python run.py --method ccsa --source D --target W --model_base vgg16 --epochs 20 --seed 1 --augment True --alpha 0.01
python run.py --method ccsa --source W --target A --model_base vgg16 --epochs 20 --seed 1 --augment True --alpha 0.01
python run.py --method ccsa --source W --target D --model_base vgg16 --epochs 20 --seed 1 --augment True --alpha 0.01

# Repeat 2
python run.py --method ccsa --source A --target D --model_base vgg16 --epochs 20 --seed 2 --augment True --alpha 0.01
python run.py --method ccsa --source A --target W --model_base vgg16 --epochs 20 --seed 2 --augment True --alpha 0.01
python run.py --method ccsa --source D --target A --model_base vgg16 --epochs 20 --seed 2 --augment True --alpha 0.01
python run.py --method ccsa --source D --target W --model_base vgg16 --epochs 20 --seed 2 --augment True --alpha 0.01
python run.py --method ccsa --source W --target A --model_base vgg16 --epochs 20 --seed 2 --augment True --alpha 0.01
python run.py --method ccsa --source W --target D --model_base vgg16 --epochs 20 --seed 2 --augment True --alpha 0.01

# Repeat 3
python run.py --method ccsa --source A --target D --model_base vgg16 --epochs 20 --seed 3 --augment True --alpha 0.01
python run.py --method ccsa --source A --target W --model_base vgg16 --epochs 20 --seed 3 --augment True --alpha 0.01
python run.py --method ccsa --source D --target A --model_base vgg16 --epochs 20 --seed 3 --augment True --alpha 0.01
python run.py --method ccsa --source D --target W --model_base vgg16 --epochs 20 --seed 3 --augment True --alpha 0.01
python run.py --method ccsa --source W --target A --model_base vgg16 --epochs 20 --seed 3 --augment True --alpha 0.01
python run.py --method ccsa --source W --target D --model_base vgg16 --epochs 20 --seed 3 --augment True --alpha 0.01

# Repeat 4
python run.py --method ccsa --source A --target D --model_base vgg16 --epochs 20 --seed 4 --augment True --alpha 0.01
python run.py --method ccsa --source A --target W --model_base vgg16 --epochs 20 --seed 4 --augment True --alpha 0.01
python run.py --method ccsa --source D --target A --model_base vgg16 --epochs 20 --seed 4 --augment True --alpha 0.01
python run.py --method ccsa --source D --target W --model_base vgg16 --epochs 20 --seed 4 --augment True --alpha 0.01
python run.py --method ccsa --source W --target A --model_base vgg16 --epochs 20 --seed 4 --augment True --alpha 0.01
python run.py --method ccsa --source W --target D --model_base vgg16 --epochs 20 --seed 4 --augment True --alpha 0.01

# Repeat 5
python run.py --method ccsa --source A --target D --model_base vgg16 --epochs 20 --seed 5 --augment True --alpha 0.01
python run.py --method ccsa --source A --target W --model_base vgg16 --epochs 20 --seed 5 --augment True --alpha 0.01
python run.py --method ccsa --source D --target A --model_base vgg16 --epochs 20 --seed 5 --augment True --alpha 0.01
python run.py --method ccsa --source D --target W --model_base vgg16 --epochs 20 --seed 5 --augment True --alpha 0.01
python run.py --method ccsa --source W --target A --model_base vgg16 --epochs 20 --seed 5 --augment True --alpha 0.01
python run.py --method ccsa --source W --target D --model_base vgg16 --epochs 20 --seed 5 --augment True --alpha 0.01