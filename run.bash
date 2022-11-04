# !/bin/bash

# python3 our.py --data CiaoDVD --save_name ciao_our --epoch 150 --lr 1e-3 --batch 2048 --latdim 512 --gnn_layer 2 --sBatch 2048 --uuPre_reg 1e-1 --uugnn_layer 1 --sal_reg 1e-5 --reg 1e-5 --gpu 0
# python3 our.py --data Epinions --save_name epinions_our --epoch 200 --lr 5e-3 --batch 2048 --latdim 128 --gnn_layer 2 --sBatch 1024 --uuPre_reg 1e-2 --uugnn_layer 4 --sal_reg 1e-5 --reg 1e-5 --gpu 0
# python3 our.py --data Yelp --save_name yelp_our --epoch 200 --lr 1e-3 --batch 4096 --latdim 256 --gnn_layer 2 --sBatch 4096 --uuPre_reg 1e0 --uugnn_layer 2 --sal_reg 1e-5 --reg 1e-6 --gpu 0