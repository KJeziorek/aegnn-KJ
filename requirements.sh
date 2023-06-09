#! /usr/bin/bash

conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cu116.html

pip install pytorch-lightning==1.4.9
pip install torchmetrics==0.4.0
pip install mapcalc==0.2.2
pip install wandb
pip install matplotlib

pip install .
