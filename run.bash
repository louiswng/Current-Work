# !/bin/bash

# python3 our.py --data ciao --save_name ciao_our --epoch 150 --lr 1e-3 --batch 2048 --latdim 64 --gnn_layer 4 --sBatch 1024 --uuPre_reg 1e-1 --uugnn_layer 1 --sal_reg 1e-5 --ssl_reg 1e-8 --temp 1.0 --reg 1e-5 --gpu 0
# python3 our.py --data epinions --save_name epinions_our --epoch 200 --lr 1e-3 --batch 2048 --latdim 64 --gnn_layer 4 --sBatch 2048 --uuPre_reg 1e-3 --uugnn_layer 4 --sal_reg 1e-5 --ssl_reg 1e-7 --temp 1.0 --reg 1e-5 --gpu 0
python3 our.py --data yelp --save_name yelp_our --epoch 200 --lr 1e-3 --batch 2048 --latdim 64 --gnn_layer 4 --sBatch 2048 --uuPre_reg 1e-2 --uugnn_layer 4 --sal_reg 1e-6 --ssl_reg 1e-7 --temp 1.0 --reg 1e-5 --gpu 0