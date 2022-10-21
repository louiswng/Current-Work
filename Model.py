import torch as t
# t.set_printoptions(profile="full")
from torch import nn
import torch.nn.functional as F
from Params import args

device = "cuda" if t.cuda.is_available() else "cpu"

xavierInit = nn.init.xavier_uniform_
zeroInit = lambda x: nn.init.constant_(x, 0.0)
normalInit = lambda x: nn.init.normal_(x, 0.0, 0.3)

class Our(nn.Module):
    def __init__(self):
        super(Our, self).__init__()
        self.uEmbeds0 = nn.Parameter(xavierInit(t.empty(args.user, args.latdim)))
        self.iEmbeds0 = nn.Parameter(xavierInit(t.empty(args.item, args.latdim)))
        self.LightGCN = LightGCN(self.uEmbeds0, self.iEmbeds0).to(device)
        self.LightGCN2 = LightGCN2(self.uEmbeds0).to(device)
        self.linear1 = nn.Linear(2*args.latdim, args.latdim)
        self.linear2 = nn.Linear(args.latdim, 1)
        self.dropout = nn.Dropout(args.dropRate)
        self.leakyrelu = nn.LeakyReLU(args.leaky)
        self.sigmoid = nn.Sigmoid()
        self.keepRate = args.keepRate

    def forward(self, adj, uAdj):
        ui_uEmbed0, ui_iEmbed0 = self.LightGCN(adj) # (usr, d)
        uu_Embed0 = self.LightGCN2(uAdj)
        return ui_uEmbed0, ui_iEmbed0, uu_Embed0

    def calcSSL(self, embeds1, embeds2, nodes):
        # pckEmbeds1 = F.normalize(embeds1[nodes], 2)
        # pckEmbeds2 = F.normalize(embeds2[nodes], 2)
        pckEmbeds1 = embeds1[nodes]
        pckEmbeds2 = embeds2[nodes]
        nume = t.exp(t.sum(pckEmbeds1 * pckEmbeds2, axis=-1) / args.temp) # same node
        deno = t.sum(t.exp(t.mm(pckEmbeds1, t.transpose(pckEmbeds2, 0, 1)) / args.temp), axis=-1) # select neg node (low solidity) 
        ssl = t.sum(- t.log(nume / deno))
        return ssl

    def calcUSSL(self, embeds1, embeds2, nodes, mask):
        # pckEmbeds1 = F.normalize(embeds1[nodes], 2)
        # pckEmbeds2 = F.normalize(embeds2[nodes], 2)
        pckEmbeds1 = embeds1[nodes]
        pckEmbeds2 = embeds2[nodes]
        nume = t.exp(t.sum(pckEmbeds1 * pckEmbeds2, axis=-1) / args.temp) # same node
        negs = t.mm(pckEmbeds1, t.transpose(pckEmbeds2, 0, 1))[mask] # select negtive nodes (low solidity) 
        deno = t.sum(t.exp(negs / args.temp), axis=-1)
        ssl = t.sum(- t.log(nume / deno))
        return ssl

    def randEdgeDrop(self, adj):
        vals = adj._values()
        idxs = adj._indices()
        edgeNum = vals.size()
        mask = ((t.rand(edgeNum) + self.keepRate).floor()).type(t.bool)
        newVals = vals[mask] / self.keepRate
        newIdxs = idxs[:, mask]
        return t.sparse.FloatTensor(newIdxs, newVals, adj.shape)

    def labeledEdgeDrop(self, adj, trnMat, edgeids, uEmbeds, iEmbeds):
        coo = trnMat.tocoo()
        usrs, itms = coo.row[edgeids], coo.col[edgeids]
        uEmbed = uEmbeds[usrs]
        iEmbed = iEmbeds[itms]
        scores = self.label(uEmbed, iEmbed)
        _, topLocs = t.topk(scores, int(len(edgeids) * self.keepRate))

        val = adj._values()
        idxs = adj._indices()
        newVals = val[topLocs]
        newIdxs = idxs[:, topLocs]
        return t.sparse.FloatTensor(newIdxs, newVals, adj.shape)  

    def label(self, lat1, lat2):
        lat = t.cat([lat1, lat2], dim=-1)
        lat = self.leakyrelu(self.dropout(self.linear1(lat))) + lat1 + lat2
        ret = t.reshape(self.sigmoid(self.dropout(self.linear2(lat))), [-1])
        return ret

    def selNegSamp(self, uEmbeds, usr):
        uEmbed = uEmbeds[usr]
        score = uEmbed @ t.t(uEmbed)
        zeros = t.zeros_like(score)
        mask = t.where(score<0.0, zeros, score).type(t.bool) # score <= 0 is False
        return mask

    def calcLosses(self, adj, usr, itmP, itmN, uAdj, usr0, usrP, usrN, usr1, usr2, trnMat, edgeids1, edgeids2):
        ui_uEmbed, ui_iEmbed, uu_Embed = self.forward(adj, uAdj)
        
        # preds on ui graph
        pckUlat = ui_uEmbed[usr]
        pckIlatP = ui_iEmbed[itmP]
        pckIlatN = ui_iEmbed[itmN]
        predsP = (pckUlat * pckIlatP).sum(-1)
        predsN = (pckUlat * pckIlatN).sum(-1)
        scoreDiff = predsP - predsN
        preLoss = -(scoreDiff).sigmoid().log().sum() / args.batch # bprloss

        # preds on uu graph
        pckUlat = uu_Embed[usr0]
        pckUlatP = uu_Embed[usrP]
        pckUlatN = uu_Embed[usrN]
        predsP = (pckUlat * pckUlatP).sum(-1)
        predsN = (pckUlat * pckUlatN).sum(-1)
        scoreDiff = predsP - predsN
        uuPreLoss = args.uuPre_reg * -(scoreDiff.sigmoid()+1e-8).log().sum() / args.batch # bprloss
        # uuPreLoss = t.tensor(0.0)

        # denoise
        scores = self.label(ui_uEmbed[usr1], ui_uEmbed[usr2])
        _preds = (uu_Embed[usr1] * uu_Embed[usr2]).sum(-1)
        salLoss = args.sal_reg * (t.maximum(t.tensor(0.0), 1.0-scores*_preds)).sum()
        # salLoss = t.tensor(0.0)
 
        # adj1 = self.randEdgeDrop(adj)
        # adj2 = self.randEdgeDrop(adj)
        adj1 = self.labeledEdgeDrop(adj, trnMat, edgeids1, ui_uEmbed, ui_iEmbed)
        adj2 = self.labeledEdgeDrop(adj, trnMat, edgeids2, ui_uEmbed, ui_iEmbed)
        uEmbeds1, iEmbeds1, _ = self.forward(adj1, uAdj)
        uEmbeds2, iEmbeds2, _ = self.forward(adj2, uAdj)
        usrSet = t.unique(usr)
        itmSet = t.unique(t.concat([itmP, itmN]))
        mask = self.selNegSamp(uu_Embed, usrSet)
        sslLoss = args.ssl_reg * (self.calcUSSL(uEmbeds1, uEmbeds2, usrSet, mask) + self.calcSSL(iEmbeds1, iEmbeds2, itmSet))
        # sslLoss = t.tensor(0.0)
        
        return preLoss, uuPreLoss, salLoss, sslLoss

    def predPairs(self, adj, usr, itm, uAdj):
        ret = self.forward(adj, uAdj)
        uEmbeds, iEmbeds = ret[:2]
        uEmbed = uEmbeds[usr]
        iEmbed = iEmbeds[itm]
        return (uEmbed * iEmbed).sum(-1)

class LightGCN(nn.Module):
    def __init__(self, uEmbeds=None, iEmbeds=None, node_dropout=False, msg_dropout=True, pool='sum'):
        super(LightGCN, self).__init__()
        self.uEmbeds = uEmbeds if uEmbeds is not None else nn.Parameter(xavierInit(t.empty(args.user, args.latdim)))
        self.iEmbeds = iEmbeds if iEmbeds is not None else nn.Parameter(xavierInit(t.empty(args.item, args.latdim)))
        self.gcnLayers = nn.Sequential(*[GCNLayer() for i in range(args.gnn_layer)])
        self.node_dropout = node_dropout
        self.msg_dropout = msg_dropout
        self.dropout = nn.Dropout(p=args.dropRate)
        self.pool = pool

    def pooling(self, embeds):
        if self.pool == 'mean':
            return embeds.mean(0)
        elif self.pool == 'sum':
            return embeds.sum(0)
        elif self.pool == 'concat':
            return embeds.view(embeds.shape[1], -1)
        else: # final
            return embeds[-1]

    def forward(self, adj):
        embeds = t.concat([self.uEmbeds, self.iEmbeds], axis=0)
        embedLst = [embeds]
        for gcn in self.gcnLayers:
            embeds = gcn(adj, embedLst[-1])
            if self.msg_dropout:
                embeds = self.dropout(embeds)
            embedLst.append(embeds)
        embeds = t.stack(embedLst, dim=0)
        embeds = self.pooling(embeds)
        return embeds[:args.user], embeds[args.user:]

class LightGCN2(nn.Module):
    def __init__(self, uEmbeds=None):
        super(LightGCN2, self).__init__()
        self.uEmbeds = uEmbeds if uEmbeds is not None else nn.Parameter(xavierInit(t.empty(args.user, args.latdim)))
        self.gnnLayers = nn.Sequential(*[GCNLayer() for i in range(args.uugnn_layer)])
    
    def forward(self, adj):
        ulats = [self.uEmbeds]
        for gcn in self.gnnLayers:
            temulat = gcn(adj, ulats[-1])
            ulats.append(temulat)
        return sum(ulats)

class GCNLayer(nn.Module):
    def __init__(self):
        super(GCNLayer, self).__init__()
    
    def forward(self, adj, embeds):
        return t.spmm(adj, embeds)
