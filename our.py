from Params import args
import os
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu # comment when nni
from setproctitle import setproctitle
setproctitle("EXP@lou")
import torch as t
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np
import pickle
import nni
from nni.utils import merge_parameter
from DataHandler import DataHandler
from Model import Our
import Utils.TimeLogger as logger
from Utils.TimeLogger import log

writer = SummaryWriter(log_dir='runs')

def setup_seed(seed=1024):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	t.manual_seed(seed)
	t.cuda.manual_seed(seed)
	t.cuda.manual_seed_all(seed) # if you are using multi-GPU.

setup_seed(args.seed)

device = "cuda" if t.cuda.is_available() else "cpu"
log(f"Using {device} device")

class Recommender():
    def __init__(self, handler):
        self.handler = handler
        print('User', args.user, 'Item', args.item)
        print('Num of interactions', args.edgeNum)
        print('Num of social interactions', args.uuEdgeNum)
        self.metrics = dict()
        mets = ['Loss', 'preLoss', 'HR', 'NDCG']
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
        self.preperaModel()
        log('Model Prepared')
        if args.load_model != None:
            self.loadModel()
            stloc = len(self.metrics['TrainLoss']) * args.tstEpoch - (args.tstEpoch - 1)
        else:
            stloc = 0
            log('Model Initialized')
        bstMtc = {'HR': 0.0, 'NDCG': 0.0}
        for ep in range(stloc, args.epoch):
            tstFlag = (ep % args.tstEpoch == 0)
            reses = self.trainEpoch()
            writer.add_scalar('Loss/train', reses['Loss'], ep)
            log(self.makePrint('Train', ep, reses, tstFlag))
            if tstFlag:
                reses = self.testEpoch()
                if reses['HR'] > bstMtc['HR']:
                    bstMtc = reses
                    es = 0
                else:
                    es += 1
                    if es >= args.patience:
                        log('Early stop')
                        break
                writer.add_scalar('HR/test', reses['HR'], ep)
                writer.add_scalar('NDCG/test', reses['NDCG'], ep)
                # nni.report_intermediate_result(reses['HR'])
                log(self.makePrint('Test', ep, reses, tstFlag))
                # self.saveHistory()
            self.sche.step()
            print()
        # nni.report_final_result(bstMtc['HR'])
        log('The best metric are %.4f, %.4f \n' % (bstMtc['HR'], bstMtc['NDCG']), save=True, oneline=True)
        self.saveHistory()

    def preperaModel(self):
        self.model = Our().to(device)
        self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr)
        self.sche = t.optim.lr_scheduler.ExponentialLR(self.opt, gamma=args.decay)

    def sampSocialGraph(self, uuMat): # 随机选出 batch 个 usr, usrP, usrN
        batIdx = t.randint(high=len(uuMat.data), size=(args.batch,))
        usr0 = t.from_numpy(uuMat.row[batIdx])
        usrP = t.from_numpy(uuMat.col[batIdx])
        usrN = t.from_numpy(self.handler.trnLoader.dataset.uuNegs[batIdx])
        usr1 = t.randint(high=args.user, size=(args.sBatch,))
        usr2 = t.randint(high=args.user, size=(args.sBatch,))
        return usr0, usrP, usrN, usr1, usr2

    def sampEdge(self, edgeSampRate, edgeNum):
        edgeSampNum = int(edgeSampRate * edgeNum)
        if edgeSampNum % 2 == 1:
            edgeSampNum += 1
        return np.random.choice(edgeNum, edgeSampNum)

    def trainEpoch(self):
        trnLoader = self.handler.trnLoader
        trnLoader.dataset.negSampling()
        epLoss, epPreLoss, epuuPreLoss, epsalLoss, epsslLoss = [0] * 5
        steps = len(trnLoader.dataset) // args.batch
        self.model.train()
        for i, (usr, itmP, itmN) in enumerate(trnLoader):
            usr, itmP, itmN = usr.long().to(device), itmP.long().to(device), itmN.long().to(device)
            usr0, usrP, usrN, usr1, usr2 = self.sampSocialGraph(self.handler.uuMat)
            usr0, usrP, usrN = usr0.long().to(device), usrP.long().to(device), usrN.long().to(device)
            usr1, usr2 = usr1.long().to(device), usr2.long().to(device)
            preLoss, uuPreLoss, salLoss, sslLoss = self.model.calcLosses(self.handler.torchAdj, usr, itmP, itmN, self.handler.torchuAdj, usr0, usrP, usrN, usr1, usr2)

            regLoss = 0
            for W in self.model.parameters():
                regLoss += W.norm(2).square()
            regLoss *= args.reg

            loss = preLoss + uuPreLoss + salLoss + sslLoss + regLoss
            epLoss += loss.item()
            epPreLoss += preLoss.item()
            epuuPreLoss += uuPreLoss.item()
            epsalLoss += salLoss.item()
            epsslLoss += sslLoss.item()

            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()
            # log('Step %d/%d: loss = %.2f, preLoss = %.2f, regLoss = %.2f         ' % (i, steps, loss, preLoss, regLoss), save=False, oneline=True)
        ret = dict()
        ret['Loss'] = epLoss / steps
        ret['preLoss'] = epPreLoss / steps
        ret['uuPreLoss'] = epuuPreLoss / steps
        ret['salLoss'] = epsalLoss / steps
        ret['sslLoss'] = epsslLoss / steps
        return ret

    def testEpoch(self):
        tstLoader = self.handler.tstLoader
        epHr, epNdcg = [0] * 2
        size = len(tstLoader.dataset)
        tstBat = args.test_batch * 100
        steps = size // tstBat
        self.model.eval()
        with t.no_grad():
            for i, (usr, itm) in enumerate(tstLoader):
                usr, itm = usr.long().to(device), itm.long().to(device)
                batch = usr.shape[0] / 100
                preds = self.model.predPairs(self.handler.torchAdj, usr, itm, self.handler.torchuAdj)
                hr, ndcg = self.calcRes(preds, itm)
                epHr += hr
                epNdcg += ndcg
                # log('Steps %d/%d: hr = %.2f, ndcg = %.2f          ' % (i, steps, hr/batch, ndcg/batch), save=False, oneline=True)
        ret = dict()
        ret['HR'] = epHr / (size/100)
        ret['NDCG'] = epNdcg / (size/100)
        return ret

    def calcMet(self, gtItm, recommends):
        hr, ndcg = [0] * 2
        if gtItm in recommends:
            hr = 1
            index = recommends.index(gtItm)
            ndcg = np.reciprocal(np.log2(index+2))
        return hr, ndcg

    def calcRes(self, preds, itm):
        batch = int(preds.shape[0] / 100)
        batHR, batNdcg = [0] * 2
        for i in range(batch):
            batch_scores = preds[i*100:(i+1)*100].view(-1)
            _, topLocs = t.topk(batch_scores, args.topk)
            tmpItm = itm[i*100: (i+1)*100] # pos:neg = 1:99
            recommends = t.take(tmpItm, topLocs).cpu().numpy().tolist()
            gtItm = tmpItm[0].item()
            hr, ndcg = self.calcMet(gtItm, recommends)
            batHR += hr
            batNdcg += ndcg
        return batHR, batNdcg

    def saveHistory(self):
        if args.epoch == 0:
            return
        with open('History/' + args.save_name + '.his', 'wb') as fs:
            pickle.dump(self.metrics, fs)

        content = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.opt.state_dict(),
        }
        t.save(content, 'Models/' + args.save_name + '.mod')
        log('Model Saved: %s' % args.save_name)

    def loadModel(self):
        ckp = t.load('Models/' + args.load_model + '.mod')
        self.model.load_state_dict(ckp['model_state_dict'])
        self.opt.load_state_dict(ckp['optimizer_state_dict'])

        with open('History' + args.load_model + '.his', 'rb') as fs:
            self.metrics = pickle.load(fs)
        log('Model Loaded')

if __name__=="__main__":
    logger.saveDefault = True

    # get parameters form tuner
    # tuner_params = nni.get_next_parameter()
    # params = vars(merge_parameter(args, tuner_params))
    # print(params)

    log('Start')
    handler = DataHandler()
    handler.LoadData()
    log('Data Loaded')

    recommender = Recommender(handler)
    recommender.run()