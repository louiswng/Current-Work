from Params import args
import os
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu # put this line before any cuda using (eg: import torch as t)
from setproctitle import setproctitle
setproctitle("louis-our")
import torch as t
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
from Model import Our, SpAdjDropEdge
from DataHandler import DataHandler, negSamp
import numpy as np
import pickle
import nni
from nni.utils import merge_parameter
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter(log_dir='runs')

class Recommender:
    def __init__(self, handler):
        self.handler = handler
        print('USER', args.user, 'ITEM', args.item)
        print('NUM OF INTERACTIONS', len(self.handler.trnMat.data))
        print('NUM OF USER-USER EDGE', args.uuEdgeNum)
		
        self.metrics = dict()
        mets = ['Loss', 'preLoss', 'Recall', 'NDCG']
        for met in mets:
            self.metrics['Train' + met] = list()
            self.metrics['Test' + met] = list()

    def makePrint(self, name, ep, reses, save):
        ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
        for metric in reses:
            val = reses[metric]
            ret += '%s = %.4f, ' % (metric, val)
            tem = name + metric
            if save and tem in self.metrics:
                self.metrics[tem].append(val)
        ret = ret[:-2] + '  '
        return ret

    def run(self):
        self.prepareModel()
        log('Model Prepared')
        if args.load_model != None:
            self.loadModel()
            stloc = len(self.metrics['TrainLoss']) * args.tstEpoch - (args.tstEpoch - 1)
        else:
            stloc = 0
            log('Model Initialized')
        
        best_acc, best_acc2 = 0.0, 0.0
        for ep in range(stloc, args.epoch):
            tstFlag = (ep % args.tstEpoch == 0)
            reses = self.trainEpoch()
            # writer.add_scalar('Loss/train', reses['Loss'], ep)
            log(self.makePrint('Train', ep, reses, tstFlag))
            if tstFlag:
                reses = self.testEpoch()
                if reses['Recall'] > best_acc:
                    best_acc = reses['Recall']
                    best_acc2 = reses['NDCG']
                    es = 0
                    # print('best acc!')
                else:
                    es += 1
                    if es >= args.patience:
                        print("Early stopping with best Recall and NDCG are", best_acc, best_acc2)
                        break
                # writer.add_scalar('Recall/test', reses['Recall'], ep)
                # writer.add_scalar('Ndcg/test', reses['NDCG'], ep)
                nni.report_intermediate_result(reses['Recall'])
                log(self.makePrint('Test', ep, reses, tstFlag))
                self.saveHistory()
            self.sche.step()
            print()

        # reses = self.testEpoch()
        # log(self.makePrint('Test', args.epoch, reses, True))
        nni.report_final_result(best_acc)
        print("best Recall and NDCG are", best_acc, best_acc2)
        self.saveHistory()

    def prepareModel(self):
        self.model = Our().cuda()
        self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr)
        self.sche = t.optim.lr_scheduler.ExponentialLR(self.opt, gamma=args.decay)

    def sampleTrainBatch(self, batIds, labelMat, otherNum, edgeNum):
        labelMat = labelMat.tocsr()
        temLabel = labelMat[batIds].toarray()
        batch = len(batIds)
        temlen = batch * 2 * args.sampNum
        uLocs = [None] * temlen
        iLocs = [None] * temlen
        cur = 0
        for i in range(batch):
            posset = np.reshape(np.argwhere(temLabel[i]!=0), [-1])
            sampNum = min(args.sampNum, len(posset))
            if sampNum == 0:
                poslocs = [np.random.choice(otherNum)]
                neglocs = [poslocs[0]]
            else:
                poslocs = np.random.choice(posset, sampNum)
                neglocs = negSamp(temLabel[i], sampNum, otherNum)
            for j in range(sampNum):
                posloc = poslocs[j]
                negloc = neglocs[j]
                uLocs[cur] = uLocs[cur+temlen//2] = batIds[i]
                iLocs[cur] = posloc
                iLocs[cur+temlen//2] = negloc
                cur += 1
        uLocs = uLocs[:cur] + uLocs[temlen//2: temlen//2 + cur]
        iLocs = iLocs[:cur] + iLocs[temlen//2: temlen//2 + cur]
        
        edgeSampNum = int(args.edgeSampRate * edgeNum)
        if edgeSampNum % 2 == 1:
            edgeSampNum += 1
        edgeids = np.random.choice(edgeNum, edgeSampNum)
        return uLocs, iLocs, edgeids
        
    def trainEpoch(self):
        num = args.user
        sfIds = np.random.permutation(num)[:args.trnNum]
        epLoss, epPreLoss, epuuPreLoss, epsslLoss, epssuLoss = [0] * 5
        num = len(sfIds)
        steps = int(np.ceil(num / args.batch))
        adj = self.handler.torchAdj
        tpAdj = t.transpose(adj, 0 ,1)
        uAdj = self.handler.torchuAdj
        self.model.train()
        for i in range(steps):
            st = i * args.batch
            ed = min((i+1) * args.batch, num)
            batIds = sfIds[st: ed]
            uLocs, iLocs, _ = self.sampleTrainBatch(batIds, self.handler.trnMat, args.item, args.edgeNum)
            uu_Locs1, uu_Locs2, uu_edgeids = self.sampleTrainBatch(batIds, self.handler.uuMat, args.user, args.uuEdgeNum)

            edgeSampNum = int(args.edgeSampRate * args.edgeNum)
            if edgeSampNum % 2 == 1:
                edgeSampNum += 1
            edgeids1 = np.random.choice(args.edgeNum, edgeSampNum)
            edgeids2 = np.random.choice(args.edgeNum, edgeSampNum)

            uLocs = t.tensor(uLocs)
            iLocs = t.tensor(iLocs)
            edgeids1 = t.tensor(edgeids1)
            edgeids2 = t.tensor(edgeids2)

            preLoss, uuPreLoss, sslLoss, ssuLoss = self.model.calcLosses(adj, uAdj, uLocs, iLocs, edgeids1, edgeids2, self.handler.trnMat, uu_Locs1, uu_Locs2, uu_edgeids, self.handler.uuMat)
                      
            uuPreLoss *= args.lambda_u           
            ssuLoss *= args.ssu_reg

            regLoss = 0
            for W in self.model.parameters():
                regLoss += W.norm(2).square()   
            regLoss *= args.reg

            loss = preLoss + uuPreLoss + sslLoss + ssuLoss + regLoss         
            epLoss += loss.item()
            epPreLoss += preLoss.item()
            epuuPreLoss += uuPreLoss.item()
            epsslLoss += sslLoss.item()
            epssuLoss += ssuLoss.item()

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            log('Step %d/%d: loss = %.2f, preLoss = %.2f, uuPreLoss = %.2f, sslLoss = %.2f, ssuLoss = %.2f, regLoss = %.2f      ' % (i, steps, loss, preLoss, uuPreLoss, sslLoss, ssuLoss, regLoss), save=False, oneline=True)
        
        ret = dict()
        ret['Loss'] = epLoss / steps
        ret['preLoss'] = epPreLoss / steps
        ret['uuPreLoss'] = epuuPreLoss / steps
        ret['sslLoss'] = epsslLoss / steps
        ret['ssuLoss'] = epssuLoss / steps
        return ret

    def testEpoch(self):
        tstLoader = self.handler.tstLoader
        epRecall, epNdcg = [0] * 2
        i = 0
        num = tstLoader.dataset.__len__()
        steps = num // args.batch
        self.model.eval()
        with t.no_grad():
            for usr, trnMask in tstLoader:
                i += 1
                usr = usr.long().cuda()
                trnMask = trnMask.cuda()

                topLocs = self.model.test(usr, trnMask, self.handler.torchAdj, self.handler.torchuAdj)

                recall, ndcg = self.calcRes(topLocs.cpu().numpy(), self.handler.tstLoader.dataset.tstLocs, usr)
                epRecall += recall
                epNdcg += ndcg
                log('Steps %d/%d: recall = %.2f, ndcg = %.2f          ' % (i, steps, recall, ndcg), save=False, oneline=True)
        ret = dict()
        ret['Recall'] = epRecall / num
        ret['NDCG'] = epNdcg / num
        return ret

    def calcRes(self, topLocs, tstLocs, batIds):
        assert topLocs.shape[0] == len(batIds)
        allRecall = allNdcg = 0
        for i in range(len(batIds)):
            temTopLocs = list(topLocs[i])
            temTstLocs = tstLocs[batIds[i]]
            tstNum = len(temTstLocs)
            maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, args.topk))])
            recall = dcg = 0
            for val in temTstLocs:
                if val in temTopLocs:
                    recall += 1
                    dcg += np.reciprocal(np.log2(temTopLocs.index(val) + 2))
            recall = recall / tstNum
            ndcg = dcg / maxDcg
            allRecall += recall
            allNdcg += ndcg
        return allRecall, allNdcg

    def saveHistory(self):
        if args.epoch == 0:
            return
        with open('History/' + args.save_path + '.his', 'wb') as fs:
            pickle.dump(self.metrics, fs)

        content = {
			'Our': self.model
		}
        t.save(content, 'Models/' + args.save_path + '.mod')
        log('Model Saved: %s' % args.save_path)

    def loadModel(self):
        ckp = t.load('Models/' + args.load_model + '.mod')
        self.model = ckp['Our']
        
        with open('History/' + args.load_model + '.his', 'rb') as fs:
            self.metrics = pickle.load(fs)
        self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr)
        self.sche = t.optim.lr_scheduler.ExponentialLR(self.opt, gamma=args.decay)
        log('Model Loaded')	

if __name__ == '__main__':
    logger.saveDefault = True
    device = "cuda" if t.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # get parameters form tuner
    tuner_params = nni.get_next_parameter()
    params = vars(merge_parameter(args, tuner_params))
    print(params)
    
    log('Start')
    handler = DataHandler()
    handler.LoadData()
    log('Load Data')
    
    recom = Recommender(handler)
    recom.run()