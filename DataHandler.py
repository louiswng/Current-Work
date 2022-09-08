import pickle
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
from Params import args
import scipy.sparse as sp
from Utils.TimeLogger import log
import torch as t
import torch.utils.data as data
import torch.utils.data as dataloader

# 抽负样本
def negSamp(temLabel, sampSize, nodeNum):
	negset = [None] * sampSize
	cur = 0
	while cur < sampSize:
		rdmItm = np.random.choice(nodeNum)
		if temLabel[rdmItm] == 0:
			negset[cur] = rdmItm
			cur += 1
	return negset

def transpose(mat):
	coomat = sp.coo_matrix(mat)
	return csr_matrix(coomat.transpose())

class DataHandler:
	def __init__(self):
		if args.data == 'yelp':
			predir = 'Data/yelp/'
		elif args.data == 'tmall':
			predir = 'Data/tmall/'
		elif args.data == 'gowalla':
			predir = 'Data/gowalla/'
		self.predir = predir
		self.trnfile = predir + 'trnMat_new.pkl'
		self.tstfile = predir + 'tstMat_new.pkl'
		self.uufile = predir + 'uuMat_new.pkl'

	def LoadOneFile(self, filename):
		with open(filename, 'rb') as fs:
			ret = (pickle.load(fs) != 0).astype(np.float32)
		if type(ret) != coo_matrix:
			ret = sp.coo_matrix(ret)
		return ret

	def normalizeAdj(self, mat):
		degree = np.array(mat.sum(axis=-1))
		dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
		dInvSqrt[np.isinf(dInvSqrt)] = 0.0
		dInvSqrtMat = sp.diags(dInvSqrt)
		return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

	def makeTorchAdj(self, mat):
		# make ui adj
		a = sp.csr_matrix((args.user, args.user))
		b = sp.csr_matrix((args.item, args.item))
		# sp.hstack([a, mat]): [usr, usr+itm]
		# sp.hstack([mat.transpose(), b]): [itm, itm+usr]
		mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])]) # (usr+itm, itm+usr)
		mat = (mat != 0) * 1.0
		mat = (mat + sp.eye(mat.shape[0])) * 1.0
		mat = self.normalizeAdj(mat)
		
		# make cuda tensor
		idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
		vals = t.from_numpy(mat.data.astype(np.float32))
		shape = t.Size(mat.shape)
		return t.sparse.FloatTensor(idxs, vals, shape).cuda()

	# indices: mat 中 1 的横竖坐标
	# shape: mat 维度
	def transToLsts(self, mat, mask=False, norm=False):
		shape = [mat.shape[0], mat.shape[1]]
		coomat = sp.coo_matrix(mat)
		# zip: 将对象中对应的元素打包成一个个元组, 返回由这些元组组成的列表
		# map(function, []): 将函数 f 依次作用在 list 每个元素上，返回新 list。map 函数要经过 list 转换
		# map(list, []): 将中每个对象（这里为元组）换为 list。[(1, 2), (2, 3)]-->[[1, 2], [2, 3]]
		# list 转换为 np.array: array([[1, 2],
		#   						  [2, 3]])
		indices = np.array(list(map(list, zip(coomat.row, coomat.col))), dtype=np.int32)
		# astype: 改变 np.array 中所有数据元素的数据类型
		data = coomat.data.astype(np.float32)

		if norm:
			rowD = np.squeeze(np.array(1 / (np.sqrt(np.sum(mat, axis=1) + 1e-8) + 1e-8)))
			colD = np.squeeze(np.array(1 / (np.sqrt(np.sum(mat, axis=0) + 1e-8) + 1e-8)))
			for i in range(len(data)):
				row = indices[i, 0]
				col = indices[i, 1]
				data[i] = data[i] * rowD[row] * colD[col]

		# half mask
		if mask:
			spMask = (np.random.uniform(size=data.shape) > 0.5) * 1.0
			data = data * spMask

		if indices.shape[0] == 0:
			indices = np.array([[0, 0]], dtype=np.int32)
			data = np.array([0.0], np.float32)
		return indices, data, shape

	def LoadData(self):
		trnMat = self.LoadOneFile(self.trnfile)
		tstMat = self.LoadOneFile(self.tstfile)
		uuMat = self.LoadOneFile(self.uufile)
		args.user, args.item = trnMat.shape
		
		idx, data, shape = self.transToLsts(trnMat, norm=True)
		idx = t.t(t.from_numpy(idx.astype(np.int64)))
		data = t.from_numpy(data.astype(np.float32))
		shape = t.Size(shape)
		self.torchAdj = t.sparse.FloatTensor(idx, data, shape).cuda()
		
		idx, data, shape = self.transToLsts(uuMat, norm=True)
		idx = t.t(t.from_numpy(idx.astype(np.int64)))
		data = t.from_numpy(data.astype(np.float32))
		shape = t.Size(shape)
		self.torchuAdj = t.sparse.FloatTensor(idx, data, shape).cuda()

		self.trnMat = trnMat
		self.uuMat = uuMat
		args.edgeNum = len(trnMat.data)
		args.uuEdgeNum = len(uuMat.data)

		# trnData = TrnData(self.trnMat)
		# self.trnLoader = dataloader.DataLoader(trnData, batch_size=args.batch, shuffle=True, num_workers=0)
		tstData = TstData(tstMat, trnMat)
		self.tstLoader = dataloader.DataLoader(tstData, batch_size=args.batch, shuffle=False, num_workers=0)

class TrnData(data.Dataset):
	def __init__(self, coomat):
		self.rows = coomat.row
		self.cols = coomat.col
		self.coomat = coomat.toarray()
		self.dokmat = coomat.todok()   

	def negSampling(self, idx, sampSize, nodeNum): # sample negtive sample for user idx
		negset = [None] * sampSize # 采样 sampSize 个负例
		cur = 0
		while cur < sampSize:
			iNeg = np.random.choice(nodeNum) # choose an item randomly
			if (idx, iNeg) not in self.dokmat: # usr i have no edge with rdmItm
				negset[cur] = iNeg
				cur += 1
		return negset  

	def __len__(self):
		return len(self.rows)

	def __getitem__(self, idx): # batch 中 每个 idx 返回的东西
		user_idx = self.rows[idx]
		posset = np.reshape(np.argwhere(self.coomat[user_idx]!=0), [-1]) # items that interact with user idx
		sampNum = min(args.sampNum, len(posset))
		if sampNum == 0: # usr idx have no interacted item, choose one idx randomly
			poslocs = [np.random.choice(args.item)]
			neglocs = [poslocs[0]]
		else:
			poslocs = np.random.choice(posset, sampNum) # sample n items in posset
			neglocs = self.negSampling(idx, sampNum, args.item) # sample n negtive items
		poslocs = np.pad(poslocs, (0, args.sampNum-sampNum))
		neglocs = np.pad(neglocs, (0, args.sampNum-sampNum))
		
		return user_idx, poslocs, neglocs, sampNum

class TstData(data.Dataset):
	def __init__(self, coomat, trnMat):
		self.csrmat = (trnMat.tocsr() != 0) * 1.0

		tstLocs = [None] * coomat.shape[0]
		tstUsrs = set()
		for i in range(len(coomat.data)):
			row = coomat.row[i]
			col = coomat.col[i]
			if tstLocs[row] is None:
				tstLocs[row] = list()
			tstLocs[row].append(col)
			tstUsrs.add(row)
		tstUsrs = np.array(list(tstUsrs))
		self.tstUsrs = tstUsrs
		self.tstLocs = tstLocs

	def __len__(self):
		return len(self.tstUsrs)

	def __getitem__(self, idx):
		return self.tstUsrs[idx], np.reshape(self.csrmat[self.tstUsrs[idx]].toarray(), [-1])