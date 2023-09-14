#!/bin/bash

model=ae_transformer
dataset=10000
for seed in 0 17 1243 3674 7341; do
    echo $seed
    echo $model
    echo $dataset
    python train.py $model  $dataset  $seed > saved_logs/$model-$dataset-$seed.txt 
done
