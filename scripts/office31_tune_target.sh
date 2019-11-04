#!/usr/bin/env bash

# Repeat 1
python run.py --method tune_target --source A --target D --model_base vgg16 --epochs 3000 --seed 1 --augment  1 --from_weights ~/domain-adaptation/runs/20191022151331_A_D_tune_both/checkpoints/cp-025.ckpt
python run.py --method tune_target --source A --target W --model_base vgg16 --epochs 3000 --seed 1 --augment  1 --from_weights ~/domain-adaptation/runs/20191022152940_A_W_tune_both/checkpoints/cp-025.ckpt
python run.py --method tune_target --source D --target A --model_base vgg16 --epochs 3000 --seed 1 --augment  1 --from_weights ~/domain-adaptation/runs/20191022154540_D_A_tune_both/checkpoints/cp-025.ckpt
python run.py --method tune_target --source D --target W --model_base vgg16 --epochs 3000 --seed 1 --augment  1 --from_weights ~/domain-adaptation/runs/20191022155009_D_W_tune_both/checkpoints/cp-025.ckpt
python run.py --method tune_target --source W --target A --model_base vgg16 --epochs 3000 --seed 1 --augment  1 --from_weights ~/domain-adaptation/runs/20191022155434_W_A_tune_both/checkpoints/cp-025.ckpt
python run.py --method tune_target --source W --target D --model_base vgg16 --epochs 3000 --seed 1 --augment  1 --from_weights ~/domain-adaptation/runs/20191022160028_W_D_tune_both/checkpoints/cp-025.ckpt

# Repeat 2
python run.py --method tune_target --source A --target D --model_base vgg16 --epochs 3000 --seed 2 --augment  1 --from_weights ~/domain-adaptation/runs/20191022160624_A_D_tune_both/checkpoints/cp-025.ckpt
python run.py --method tune_target --source A --target W --model_base vgg16 --epochs 3000 --seed 2 --augment  1 --from_weights ~/domain-adaptation/runs/20191022162237_A_W_tune_both/checkpoints/cp-025.ckpt
python run.py --method tune_target --source D --target A --model_base vgg16 --epochs 3000 --seed 2 --augment  1 --from_weights ~/domain-adaptation/runs/20191022163840_D_A_tune_both/checkpoints/cp-025.ckpt
python run.py --method tune_target --source D --target W --model_base vgg16 --epochs 3000 --seed 2 --augment  1 --from_weights ~/domain-adaptation/runs/20191022164308_D_W_tune_both/checkpoints/cp-025.ckpt
python run.py --method tune_target --source W --target A --model_base vgg16 --epochs 3000 --seed 2 --augment  1 --from_weights ~/domain-adaptation/runs/20191022164731_W_A_tune_both/checkpoints/cp-025.ckpt
python run.py --method tune_target --source W --target D --model_base vgg16 --epochs 3000 --seed 2 --augment  1 --from_weights ~/domain-adaptation/runs/20191022165325_W_D_tune_both/checkpoints/cp-025.ckpt

# Repeat 3
python run.py --method tune_target --source A --target D --model_base vgg16 --epochs 3000 --seed 3 --augment  1 --from_weights ~/domain-adaptation/runs/20191022165923_A_D_tune_both/checkpoints/cp-025.ckpt
python run.py --method tune_target --source A --target W --model_base vgg16 --epochs 3000 --seed 3 --augment  1 --from_weights ~/domain-adaptation/runs/20191022171553_A_W_tune_both/checkpoints/cp-025.ckpt
python run.py --method tune_target --source D --target A --model_base vgg16 --epochs 3000 --seed 3 --augment  1 --from_weights ~/domain-adaptation/runs/20191022173208_D_A_tune_both/checkpoints/cp-025.ckpt
python run.py --method tune_target --source D --target W --model_base vgg16 --epochs 3000 --seed 3 --augment  1 --from_weights ~/domain-adaptation/runs/20191022173640_D_W_tune_both/checkpoints/cp-025.ckpt
python run.py --method tune_target --source W --target A --model_base vgg16 --epochs 3000 --seed 3 --augment  1 --from_weights ~/domain-adaptation/runs/20191022174104_W_A_tune_both/checkpoints/cp-025.ckpt
python run.py --method tune_target --source W --target D --model_base vgg16 --epochs 3000 --seed 3 --augment  1 --from_weights ~/domain-adaptation/runs/20191022174706_W_D_tune_both/checkpoints/cp-025.ckpt

# Repeat 4
python run.py --method tune_target --source A --target D --model_base vgg16 --epochs 3000 --seed 4 --augment  1 --from_weights ~/domain-adaptation/runs/20191022175314_A_D_tune_both/checkpoints/cp-025.ckpt
python run.py --method tune_target --source A --target W --model_base vgg16 --epochs 3000 --seed 4 --augment  1 --from_weights ~/domain-adaptation/runs/20191022180951_A_W_tune_both/checkpoints/cp-025.ckpt
python run.py --method tune_target --source D --target A --model_base vgg16 --epochs 3000 --seed 4 --augment  1 --from_weights ~/domain-adaptation/runs/20191022182611_D_A_tune_both/checkpoints/cp-025.ckpt
python run.py --method tune_target --source D --target W --model_base vgg16 --epochs 3000 --seed 4 --augment  1 --from_weights ~/domain-adaptation/runs/20191022183040_D_W_tune_both/checkpoints/cp-025.ckpt
python run.py --method tune_target --source W --target A --model_base vgg16 --epochs 3000 --seed 4 --augment  1 --from_weights ~/domain-adaptation/runs/20191022183456_W_A_tune_both/checkpoints/cp-025.ckpt
python run.py --method tune_target --source W --target D --model_base vgg16 --epochs 3000 --seed 4 --augment  1 --from_weights ~/domain-adaptation/runs/20191022184038_W_D_tune_both/checkpoints/cp-025.ckpt

# Repeat 5
python run.py --method tune_target --source A --target D --model_base vgg16 --epochs 3000 --seed 5 --augment  1 --from_weights ~/domain-adaptation/runs/20191022184642_A_D_tune_both/checkpoints/cp-015.ckpt
python run.py --method tune_target --source A --target W --model_base vgg16 --epochs 3000 --seed 5 --augment  1 --from_weights ~/domain-adaptation/runs/20191022190310_A_W_tune_both/checkpoints/cp-025.ckpt
python run.py --method tune_target --source D --target A --model_base vgg16 --epochs 3000 --seed 5 --augment  1 --from_weights ~/domain-adaptation/runs/20191022191922_D_A_tune_both/checkpoints/cp-025.ckpt
python run.py --method tune_target --source D --target W --model_base vgg16 --epochs 3000 --seed 5 --augment  1 --from_weights ~/domain-adaptation/runs/20191022192357_D_W_tune_both/checkpoints/cp-025.ckpt
python run.py --method tune_target --source W --target A --model_base vgg16 --epochs 3000 --seed 5 --augment  1 --from_weights ~/domain-adaptation/runs/20191022192807_W_A_tune_both/checkpoints/cp-025.ckpt
python run.py --method tune_target --source W --target D --model_base vgg16 --epochs 3000 --seed 5 --augment  1 --from_weights ~/domain-adaptation/runs/20191022193359_W_D_tune_both/checkpoints/cp-025.ckpt

