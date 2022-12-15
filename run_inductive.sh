#!/bin/bash

datasets=("amazon-computers" "amazon-photos" "citeseer" "cora" "coauthor-cs" "coauthor-physics")

if [ $# -eq 0 ]; then
    echo "Error: a dataset is required"
    echo "Usage: run_inductive.sh [dataset_name], where [dataset_name] is one of [${datasets[*]}]"
    exit 1
fi

if [[ ! " ${datasets[*]} " == *"$1"* ]]; then
    echo "Unknown dataset: '$1'"
    exit 2
fi

cd src || exit 3
python train_nc.py --flagfile="config/inductive_$1.cfg" --split_method=inductive
