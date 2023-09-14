#!/bin/bash

model=tcn

for dataset in 10000 8000 6000 4000; do
    for seed in 53 97 103 191 99719; do
        echo $model
        echo $dataset
        echo $seed
        python train.py $model  $dataset  $seed > saved_logs/$model-$dataset-$seed.txt 
    done
done
