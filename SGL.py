from Params import args
import os
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
from setproctitle import setproctitle
setproctitle("louis-sgl")
import torch as t
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np
import pickle
from DataHandler import DataHandler
from Model import SGL
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
                log(self.makePrint('Test', ep, reses, tstFlag))
                # self.saveHistory()
            self.sche.step()
            print()
        log('The best metric are %.4f, %.4f \n' % (bstMtc['HR'], bstMtc['NDCG']), save=True, oneline=True)
        self.saveHistory()

    def preperaModel(self):
        self.model = SGL().to(device)
        self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr)
        self.sche = t.optim.lr_scheduler.ExponentialLR(self.opt, gamma=args.decay)

    def trainEpoch(self):
        trnLoader = self.handler.trnLoader
        trnLoader.dataset.negSampling()
        epLoss, epPreLoss, epsslLoss = [0] * 3
        steps = len(trnLoader.dataset) // args.batch
        self.model.train()
        for i, (usr, itmP, itmN) in enumerate(trnLoader):
            usr, itmP, itmN = usr.long().to(device), itmP.long().to(device), itmN.long().to(device)
            preLoss, sslLoss = self.model.calcLosses(self.handler.torchAdj, usr, itmP, itmN)

            regLoss = 0
            for W in self.model.parameters():
                regLoss += W.norm(2).square()
            regLoss *= args.reg

            loss = preLoss + regLoss + sslLoss
            epLoss += loss.item()
            epPreLoss += preLoss.item()
            epsslLoss += sslLoss.item()

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            # log('Step %d/%d: loss = %.2f, preLoss = %.2f, sslLoss = %.2f, regLoss = %.2f         ' % (i, steps, loss, preLoss, sslLoss, regLoss), save=False, oneline=True)
        ret = dict()
        ret['Loss'] = epLoss / steps
        ret['preLoss'] = epPreLoss / steps
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
                preds = self.model.predPairs(self.handler.torchAdj, usr, itm)
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

    log('Start')
    handler = DataHandler()
    handler.LoadData()
    log('Data Loaded')

    recommender = Recommender(handler)
    recommender.run()