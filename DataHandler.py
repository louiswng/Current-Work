from Params import args
import os
from random import shuffle
import pandas as pd
import numpy as np
import pickle
import scipy.sparse as sp
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
import torch as t
from torchvision import datasets
from torchvision.io import read_image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor

device = "cuda" if t.cuda.is_available() else "cpu"

class DataHandler():
	def __init__(self):
		if args.data == 'yelp':
			predir = 'Data/yelp/'
		elif args.data == 'ciaodvd':
			predir = 'Data/ciaodvd/'
		elif args.data == 'epinions':
			predir = 'Data/epinions/'
		self.predir = predir
		self.trnfile = predir + 'trnMat.pkl'
		self.tstfile = predir + 'tstMat.pkl'
        # self.uufile = predir + 'trust.csv'
		
	def LoadOneFile(self, filename):
		with open(filename, 'rb') as fs:
			ret = (pickle.load(fs) != 0).astype(np.float32)
		if type(ret) != coo_matrix:
			ret = sp.coo_matrix(ret)
		return ret
	
	def normalizeAdj(self, mat): # 对称归一化的拉普拉斯矩阵
		degree = np.array(mat.sum(axis=-1)) # degree (sum) of each row
		dInvSqrt = np.reshape(np.power(degree, -0.5), [-1]) # 1/sqrt(degree)
		dInvSqrt[np.isinf(dInvSqrt)] = 0.0 # inf 的行置 0=度为 0 的列置 0
		dInvSqrtMat = sp.diags(dInvSqrt)
		return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()
		
	def makeTorchAdj(self, mat):
		# make ui adj
		a = sp.csr_matrix((args.user, args.user))
		b = sp.csr_matrix((args.item, args.item))
		mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
		mat = (mat != 0) * 1.0
		mat = (mat+ sp.eye(mat.shape[0])) * 1.0
		mat = self.normalizeAdj(mat)
		
		# make cuda tensor
		idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
		vals = t.from_numpy(mat.data.astype(np.float32))
		shape = t.Size(mat.shape)

		# rowD = np.squeeze(np.array(1 / (np.sqrt(np.sum(mat, axis=1) + 1e-8) +1e-8)))
		# colD = np.squeeze(np.array(1 / (np.sqrt(np.sum(mat, axis=0) + 1e-8) +1e-8)))
		# for i in range(len(vals)):
		# 	row = idxs[0, i]
		# 	col = idxs[1, i]
		# 	vals[i] *= rowD[row] * colD[col]

		return t.sparse.FloatTensor(idxs, vals, shape).to(device)
		
	def LoadData(self):
		trnMat = self.LoadOneFile(self.trnfile)
		tstMat = self.LoadOneFile(self.tstfile)
		args.user, args.item = trnMat.shape
		
		self.torchAdj = self.makeTorchAdj(trnMat)
		
		# self.torchTpAdj = self.makeTorchAdj(trnMat.transpose())
		
		trnData = TrnData(trnMat)
		self.trnLoader = DataLoader(trnData, batch_size=args.batch, shuffle=True, num_workers=0)
		tstData = TstData(tstMat, trnMat)
		self.tstLoader = DataLoader(tstData, batch_size=args.test_batch, shuffle=False, num_workers=0) # TODO: whether to shuffle

class TrnData(Dataset): # usr: pos: neg = 1: 1: 1
    def __init__(self, coomat):
        self.rows = coomat.row
        self.cols = coomat.col
        self.coomat = coomat
        self.dokmat = coomat.todok()
        self.negs = np.zeros(len(self.rows)).astype(np.int32)

    def negSampling(self):
        for i in range(len(self.rows)):
            u = self.rows[i]
            while True:
                iNeg = np.random.randint(args.item)
                if (u, iNeg) not in self.dokmat:
                    break
            self.negs[i] = iNeg

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx): # usr: pos: neg = 1: 1: 1
        return self.rows[idx], self.cols[idx], self.negs[idx]

class TstData(Dataset): # usr, trnMask
    def __init__(self, coomat, trnMat):
        self.csrmat = (trnMat.tocsr() != 0) * 1.0

        tstLocs = [None] * coomat.shape[0]
        tstUsrs = set()
        for i in range(len(coomat.data)): # check each edge
            row = coomat.row[i] # usr id
            col = coomat.col[i] # itm id
            if tstLocs[row] is None:
                tstLocs[row] = list()
            tstLocs[row].append(col) # record all interacted item for each active user
            tstUsrs.add(row) # record all active user
        tstUsrs = np.array(list(tstUsrs))
        self.tstUsrs = tstUsrs
        self.tstLocs = tstLocs

    def __len__(self):
        return len(self.tstUsrs)

    def __getitem__(self, idx):
        return self.tstUsrs[idx], np.reshape(self.csrmat[self.tstUsrs[idx]].toarray(), [-1]) # return uid and all item for uid