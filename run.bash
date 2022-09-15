# !/bin/bash

# python3 LightGCN.py --data ciao --save_name ciao_lightgcn --epoch 200 --gnn_layer 2 --reg 1e-5 --gpu 0

# python3 SGL.py --data ciao --save_name ciao_sgl --epoch 200 --ssl_reg 1e-6 --gnn_layer 2 --keepRate 0.5 --temp 0.4 --reg 1e-5 --gpu 0

# python3 SHT.py --data ciao --save_name ciao_sht --epoch 300 --ssl_reg 1e-6 --edgeSampRate 0.1 --gnn_layer 3 --hgnn_layer 2 --hyperNum 128 --att_head 4 --leaky 0.5 --reg 1e-2 --gpu 0

python3 our.py --data ciao --save_name ciao_our --epoch 300 --mult 1e2 --uuPre_reg 1e-3 --sal_reg 1e-5 --ssl_reg 1e-6 --edgeSampRate 0.1 --gnn_layer 3 --hgnn_layer 2 --hyperNum 128 --att_head 4 --keepRate 0.5 --temp 1.0 --leaky 0.5 --reg 1e-2 --gpu 0


