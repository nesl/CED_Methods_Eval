#!/bin/bash

model=ae_transformer

for dataset in 10000 8000 6000 4000; do
    for seed in 0 17 1243 3674 7341; do
        echo $model
        echo $dataset
        echo $seed
        python train.py $model  $dataset  $seed > saved_logs/$model-$dataset-$seed.txt 
    done
done
