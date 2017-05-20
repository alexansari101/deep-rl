#!/bin/bash

#Copies files over to a remote and starts the training

# $1: computer
# $2: command


remote_machine="$1"
proj_dir="~/projects/deep-rl"
remote=$1
remote_proj="$remote:$proj_dir"
echo "Running in $remote_proj"

ssh $remote "mkdir -p $proj_dir"
#Copy files over
declare -a cp_files=("run_hA3C.py" "HA3C_2lvl.py")
for f in "${cp_files[@]}"
do
    scp $f $remote_proj >> /dev/null
done
ssh $remote "chmod +x $proj_dir/run_hA3C.py"



declare -a cp_folders=("util" "intrinsics" "agents" "environments")
for f in "${cp_folders[@]}"
do
    ssh $remote "mkdir -p $proj_dir/$f"
    scp -r $f/*.py $remote_proj/$f >> /dev/null
done

echo "Start script over ssh"
ssh $remote -t "bash --rcfile ~/.more.sh"
# ssh $remote "source ~/tensorflow/bin/activate; $proj_dir/run_hA3C.py --train --env Search"
# ssh $remote "source ~/tensorflow/bin/activate; python ./print.py"
# ssh $1 -t "bash -i"
# loadtf
# python $proj_dir/run_hA3C.py $3"
# exit
