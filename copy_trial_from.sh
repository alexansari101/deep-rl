#!/bin/bash

#Copies files over to a remote and starts the training

# $1: computer
# $2: trial name


remote_machine="$1"
trial_dir="~/projects/deep-rl/trials"
remote=$1
remote_t="$remote:$trial_dir"
echo "Copying to ./tmp/$2"

scp -r $remote_t/$2 tmp/$2
