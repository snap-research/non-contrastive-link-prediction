#!/bin/bash

# Update packages & install wget
echo "apt update"
apt-get update
echo "install wget"
apt-get install wget -y
echo "upgrade packages"
apt-get upgrade -y
echo "remove unused packages"
apt-get autoremove -y

echo "Installing miniconda"
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -p "$HOME/miniconda" -b

"$HOME/miniconda/bin/conda" shell.bash hook >> "$HOME/.bashrc"
eval "$("$HOME/miniconda/bin/conda" shell.bash hook)"

echo "Install torch, PyG and DGL"
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
conda install -y pyg -c pyg

echo "Install numpy, ogb, sklearn, TB"
pip install -U pip
pip install ogb numpy scikit-learn absl-py networkx tqdm
pip install tensorboard tensorboard-pytorch
pip install wandb
