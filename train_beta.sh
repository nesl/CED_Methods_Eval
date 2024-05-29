#!/bin/bash

model=ae_transformer

for dataset in 10000; do
    for seed in 233; do
        echo $model
        echo $dataset
        echo $seed
        python train.py $model  $dataset  $seed #> saved_logs/$model-$dataset-$seed-test.txt 
    done
done
