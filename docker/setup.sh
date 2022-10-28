#!/bin/bash

# Update packages & install wget
echo "apt update"
apt update
echo "install wget"
apt install wget -y
echo "upgrade packages"
apt upgrade -y
echo "remove unused packages"
apt autoremove -y

echo "Installing miniconda"
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -p "$HOME/miniconda" -b

"$HOME/miniconda/bin/conda" shell.bash hook >> "$HOME/.bashrc"
eval "$("$HOME/miniconda/bin/conda" shell.bash hook)"

echo "Install torch, PyG and DGL"
conda install -y pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install -y pyg -c pyg
fconda install -y -c dglteam dgl-cuda11.3

echo "Install numpy, ogb, sklearn, TB"
pip install -U pip
pip install ogb numpy scikit-learn absl-py networkx tqdm
pip install tensorboard tensorboard-pytorch
pip install wandb
