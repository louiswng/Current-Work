import torch as t
from torch import nn
import torch.nn.functional as F
from Params import args

xavierInit = nn.init.xavier_uniform_
zeroInit = lambda x: nn.init.constant_(x, 0.0)

class Our(nn.Module):
    def __init__(self):
        super(Our, self).__init__()
        self.uEmbeds0 = nn.Parameter(xavierInit(t.empty(args.user, args.latdim)))
        self.iEmbeds0 = nn.Parameter(xavierInit(t.empty(args.item, args.latdim)))
        self.LightGCN = LightGCN(self.uEmbeds0, self.iEmbeds0).cuda()
        self.LightGCN2 = LightGCN2(self.uEmbeds0).cuda()
        self.prepareKey1 = prepareKey().cuda()
        self.prepareKey2 = prepareKey().cuda()
        self.prepareKey3 = prepareKey().cuda()
        self.HypergraphTransormer1 = HypergraphTransormer().cuda()
        self.HypergraphTransormer2 = HypergraphTransormer().cuda()
        self.HypergraphTransormer3 = HypergraphTransormer().cuda()
        self.label = LabelNetwork().cuda()
        self.label2 = LabelNetwork2().cuda()
        self.SpAdjDropEdge = SpAdjDropEdge(args.keepRate).cuda()
        
    def forward(self, adj, tpAdj, uAdj):
        ui_uEmbed_gcn, ui_iEmbed_gcn = self.LightGCN(adj, tpAdj) # (usr, d)
        uu_Embed_gcn = self.LightGCN2(uAdj)
        # Residual Connection, add positional information
        ui_uEmbed0 = self.uEmbeds0 + ui_uEmbed_gcn
        ui_iEmbed0 = self.iEmbeds0 + ui_iEmbed_gcn
        uu_Embed0 = self.uEmbeds0 + uu_Embed_gcn

        ui_uKey = self.prepareKey1(ui_uEmbed0)
        ui_iKey = self.prepareKey2(ui_iEmbed0)
        uu_Key = self.prepareKey3(uu_Embed0)
        ui_ulat, ui_uHyper = self.HypergraphTransormer1(ui_uEmbed0, ui_uKey)
        ui_ilat, ui_iHyper = self.HypergraphTransormer2(ui_iEmbed0, ui_iKey)
        uu_lat, uu_Hyper = self.HypergraphTransormer3(uu_Embed0, uu_Key)

        return ui_uEmbed0, ui_iEmbed0, ui_ulat, ui_ilat, ui_uKey, ui_iKey, ui_uHyper, ui_iHyper, uu_Embed0, uu_lat, uu_Key, uu_Hyper

    def calcSSL(self, embeds1, embeds2, nodes):
        pckEmbeds1 = F.normalize(embeds1[nodes], 2)
        pckEmbeds2 = F.normalize(embeds2[nodes], 2)
        nume = t.exp(t.sum(pckEmbeds1 * pckEmbeds2, axis=-1) / args.temp) # same node
        deno = t.sum(t.exp(t.mm(pckEmbeds1, t.transpose(embeds2, 0, 1)) / args.temp), axis=-1)
        ssl = t.sum(- t.log(nume / deno))
        return ssl

    def calcLosses(self, adj, tpAdj, uAdj, uids, iids, edgeids1, edgeids2, trnMat, uu_ids1, uu_ids2, uuedgeids, uuMat):
        uEmbeds, iEmbeds, ui_ulat, ui_ilat, ui_uKey, ui_iKey, ui_uHyper, ui_iHyper, uu_Embed0, uu_lat, uu_Key, uu_Hyper = self.forward(adj, tpAdj, uAdj)
        ui_pckUlat = ui_ulat[uids] # (batch, d)
        ui_pckIlat = ui_ilat[iids]
        ui_preds = (ui_pckUlat * ui_pckIlat).sum(-1) # (batch, batch, d)

        sampNum = len(uids) // 2
        posPred = ui_preds[:sampNum]
        negPred = ui_preds[sampNum:]
        preLoss = (t.maximum(t.tensor(0.0), 1.0 - (posPred - negPred))).sum() / args.batch

        uu_pcklat1 = uu_lat[uu_ids1] # (batch, d)
        uu_pcklat2 = uu_lat[uu_ids2]
        uu_preds = (uu_pcklat1 * uu_pcklat2).sum(-1) # (batch, batch, d)

        sampNum = len(uu_ids1) // 2
        posPred = uu_preds[:sampNum]
        negPred = uu_preds[sampNum:]
        uuPreLoss = t.sum(t.maximum(t.tensor(0.0), 1.0 - (posPred - negPred))) / args.batch
        
        adj1, tpAdj1 = self.SpAdjDropEdge(adj, tpAdj) # update per batch
        adj2, tpAdj2 = self.SpAdjDropEdge(adj, tpAdj)
        ret = self.forward(adj1, tpAdj1, uAdj)
        uEmbeds1, iEmbeds1 = ret[2:4] # global
        ret = self.forward(adj2, tpAdj2, uAdj)
        uEmbeds2, iEmbeds2 = ret[2:4]

        usrSet = t.unique(uids)
        itmSet = t.unique(iids)
        sslLoss = args.ssl_reg * (self.calcSSL(uEmbeds1, uEmbeds2, usrSet) + self.calcSSL(iEmbeds1, iEmbeds2, itmSet))

        coo = uuMat.tocoo()
        usrs1, usrs2 = coo.row[uuedgeids], coo.col[uuedgeids]
        uu_Key = t.reshape(t.permute(uu_Key, dims=[1, 0, 2]), [-1, args.latdim])
        usrKey1 = uu_Key[usrs1] # (batch, d)
        usrKey2 = uu_Key[usrs2]
        uu_Hyper = (uu_Hyper + ui_uHyper) / 2
        usrLat1 = ui_ulat[usrs1] # (batch, d)
        usrLat2 = ui_ulat[usrs2]
        uu_scores = self.label2(usrKey1, usrKey2, usrLat1, usrLat2, uu_Hyper) # (batch, k)
        _uu_preds = (uu_Embed0[usrs1]*uu_Embed0[usrs2]).sum(1)

        halfNum = uu_scores.shape[0] // 2
        fstScores = uu_scores[:halfNum]
        scdScores = uu_scores[halfNum:]
        fstPreds = _uu_preds[:halfNum]
        scdPreds = _uu_preds[halfNum:]
        ssuLoss = (t.maximum(t.tensor(0.0), 1.0 - (fstPreds - scdPreds) * args.mult * (fstScores-scdScores))).sum()

        return preLoss, uuPreLoss, sslLoss, ssuLoss

    def test(self, usr, trnMask, adj, tpAdj, uAdj):
        ret = self.forward(adj, tpAdj, uAdj)
        ui_ulat, ui_ilat = ret[2:4]
        pckUlat = ui_ulat[usr]
        allPreds = pckUlat @ t.t(ui_ilat)
        allPreds = allPreds * (1 - trnMask) - trnMask * 1e8
        _, topLocs = t.topk(allPreds, args.topk)
        return topLocs

class LightGCN(nn.Module):
    def __init__(self, uEmbeds=None, iEmbeds=None):
        super(LightGCN, self).__init__()

        self.uEmbeds = uEmbeds if uEmbeds is not None else nn.Parameter(xavierInit(t.empty(args.user, args.latdim)))
        self.iEmbeds = iEmbeds if iEmbeds is not None else nn.Parameter(xavierInit(t.empty(args.item, args.latdim)))
        self.gnnLayers = nn.Sequential(*[GCNLayer() for i in range(args.gcn_hops)])
    
    def forward(self, adj, tpAdj):
        ulats = [self.uEmbeds]
        ilats = [self.iEmbeds]
        for gcn in self.gnnLayers:
            temulat = gcn(adj, ilats[-1])
            temilat = gcn(tpAdj, ulats[-1])
            ulats.append(temulat)
            ilats.append(temilat)
        return sum(ulats[1:]), sum(ilats[1:])

class LightGCN2(nn.Module):
    def __init__(self, uEmbeds=None):
        super(LightGCN2, self).__init__()

        self.uEmbeds = uEmbeds if uEmbeds is not None else nn.Parameter(xavierInit(t.empty(args.user, args.latdim)))
        self.gnnLayers = nn.Sequential(*[GCNLayer() for i in range(args.gcn_hops)])
    
    def forward(self, adj):
        ulats = [self.uEmbeds]
        for gcn in self.gnnLayers:
            temulat = gcn(adj, ulats[-1])
            ulats.append(temulat)
        return sum(ulats[1:])

class GCNLayer(nn.Module):
    def __init__(self):
        super(GCNLayer, self).__init__()
    
    def forward(self, adj, embeds):
        return t.spmm(adj, embeds)

class prepareKey(nn.Module):
    def __init__(self):
        super(prepareKey, self).__init__()
        self.K = nn.Parameter(xavierInit(t.empty(args.latdim, args.latdim)))
    
    def forward(self, nodeEmbed):
        key = t.reshape(nodeEmbed @ self.K, [-1, args.att_head, args.latdim//args.att_head])
        key = t.permute(key, dims=[1, 0, 2])
        return key

class prepareValue(nn.Module):
    def __init__(self):
        super(prepareValue, self).__init__()
        self.V = nn.Parameter(xavierInit(t.empty(args.latdim, args.latdim)))
    
    def forward(self, nodeEmbed):
        value = t.reshape(nodeEmbed @ self.V, [-1, args.att_head, args.latdim//args.att_head])
        value = t.permute(value, dims=[1, 2, 0])
        return value

class HypergraphTransormer(nn.Module):
    def __init__(self):
        super(HypergraphTransormer, self).__init__()
        self.hypergraphLayers = nn.Sequential(*[HypergraphTransformerLayer() for i in range(args.hgnn_layer)])
        self.Hyper = nn.Parameter(xavierInit(t.empty(args.hyperNum, args.latdim)))
        self.V = nn.Parameter(xavierInit(t.empty(args.latdim, args.latdim)))
        self.prepareValue = prepareValue()

    def forward(self, Embed0, Key):
        lats = [Embed0]
        for hypergraph in self.hypergraphLayers:
            Value = self.prepareValue(lats[-1])
            lats = hypergraph(lats, Key, Value, self.Hyper, self.V)
        return sum(lats), self.Hyper

class HypergraphTransformerLayer(nn.Module):
    def __init__(self):
        super(HypergraphTransformerLayer, self).__init__()
        self.linear1 = nn.Linear(args.hyperNum, args.hyperNum, bias=False)
        self.linear2 = nn.Linear(args.hyperNum, args.hyperNum, bias=False)
        self.leakyrelu = nn.LeakyReLU(args.leaky)

    def forward(self, lats, key, value, hyper, V):
        temlat1 = value @ key
        # prepare query
        hyper = t.reshape(hyper, [-1, args.att_head, args.latdim//args.att_head])
        hyper = t.permute(hyper, dims=[1, 2, 0])
        temlat1 = t.reshape(temlat1 @ hyper, [args.latdim, -1])
        temlat2 = self.leakyrelu(self.linear1(temlat1)) + temlat1
        temlat3 = self.leakyrelu(self.linear2(temlat2)) + temlat2

        preNewLat = t.reshape(t.t(temlat3) @ V, [-1, args.att_head, args.latdim//args.att_head])
        preNewLat = t.permute(preNewLat, [1, 0, 2])
        preNewLat = hyper @ preNewLat
        newLat = key @ preNewLat
        newLat = t.reshape(t.permute(newLat, [1, 0, 2]), [-1, args.latdim])
        lats.append(newLat)
        return lats

class Meta(nn.Module):
    def __init__(self):
        super(Meta, self).__init__()
        self.linear1 = nn.Linear(args.latdim, args.latdim * args.latdim, bias=True)
        self.linear2 = nn.Linear(args.latdim, args.latdim, bias=True)
        self.leakyrelu = nn.LeakyReLU(args.leaky)
    def forward(self, hyper):
        hyper_mean = t.mean(hyper, dim=0, keepdim=True)
        hyper = hyper_mean
        W1 = t.reshape(self.linear1(hyper), [args.latdim, args.latdim])
        b1 = self.linear2(hyper)
        def mapping(key):
            ret = self.leakyrelu(key @ W1 + b1)
            return ret
        return mapping

class LabelNetwork(nn.Module):
    def __init__(self):
        super(LabelNetwork, self).__init__()
        self.meta = Meta()
        self.linear1 = nn.Linear(2*args.latdim, args.latdim, bias=True)
        self.linear2 = nn.Linear(args.latdim, 1, bias=True)
        self.leakyrelu = nn.LeakyReLU(args.leaky)
        self.sigmoid = nn.Sigmoid()
    def forward(self, usrKey, itmKey, uHyper, iHyper):
        uMapping = self.meta(uHyper)
        iMapping = self.meta(iHyper)
        ulat = uMapping(usrKey)
        ilat = iMapping(itmKey)
        lat = t.cat((ulat, ilat), dim=-1)
        lat = self.leakyrelu(self.linear1(lat)) + ulat + ilat
        ret = t.reshape(self.sigmoid(self.linear2(lat)), [-1])
        return ret

class Meta2(nn.Module):
    def __init__(self):
        super(Meta2, self).__init__()
        self.linear1 = nn.Linear(args.latdim, args.latdim * args.latdim, bias=True)
        self.linear2 = nn.Linear(args.latdim, args.latdim, bias=True)
        self.linear3 = nn.Linear(args.latdim, args.latdim * args.latdim, bias=True)
        self.linear4 = nn.Linear(args.latdim, args.latdim, bias=True)
        self.leakyrelu = nn.LeakyReLU(args.leaky)
    def forward(self, hyper):
        hyper_mean = t.mean(hyper, dim=0, keepdim=True)
        hyper = hyper_mean
        W1 = t.reshape(self.linear1(hyper), [args.latdim, args.latdim])
        b1 = self.linear2(hyper)
        W2 = t.reshape(self.linear3(hyper), [args.latdim, args.latdim])
        b2 = self.linear4(hyper)
        def mapping(key, lat):
            ret = self.leakyrelu(key @ W1 + b1)
            ret += self.leakyrelu(lat @ W2 + b2)
            return ret
        return mapping

class LabelNetwork2(nn.Module):
    def __init__(self):
        super(LabelNetwork2, self).__init__()
        self.meta = Meta2()
        self.linear1 = nn.Linear(2*args.latdim, args.latdim, bias=True)
        self.linear2 = nn.Linear(args.latdim, 1, bias=True)
        self.leakyrelu = nn.LeakyReLU(args.leaky)
        self.sigmoid = nn.Sigmoid()
    def forward(self, Key1, Key2, Lat1, Lat2, Hyper):
        Mapping = self.meta(Hyper)
        lat1 = Mapping(Key1, Lat1)
        lat2 = Mapping(Key2, Lat2)
        lat = t.cat((lat1, lat2), dim=-1)
        lat = self.leakyrelu(self.linear1(lat)) + lat1 + lat2
        ret = t.reshape(self.sigmoid(self.linear2(lat)), [-1])
        return ret

class SpAdjDropEdge(nn.Module):
    def __init__(self, keepRate):
        super(SpAdjDropEdge, self).__init__()
        self.keepRate = keepRate
    
    def forward(self, adj, tpAdj):
        vals = adj._values()
        idxs = adj._indices()
        edgeNum = vals.size()
        mask = ((t.rand(edgeNum) + self.keepRate).floor()).type(t.bool)
        newVals = vals[mask] / self.keepRate
        newIdxs = idxs[:, mask]
        adj = t.sparse.FloatTensor(newIdxs, newVals, adj.shape)

        vals = tpAdj._values()
        idxs = tpAdj._indices()
        edgeNum = vals.size()
        mask = t.t(mask)
        newVals = vals[mask] / self.keepRate
        newIdxs = idxs[:, mask]
        tpAdj = t.sparse.FloatTensor(newIdxs, newVals, tpAdj.shape)
        
        return adj, tpAdj

class SpAdjDropEdge2(nn.Module):
    def __init__(self, keepRate):
        super(SpAdjDropEdge, self).__init__()
    
    def forward(self, trnMat, adj, tpAdj, edgeids):
        coo = trnMat.tocoo()
        usrs, itms = coo.row[edgeids], coo.col[edgeids]
        ui_uKey = t.reshape(t.permute(ui_uKey, dims=[1, 0, 2]), [-1, args.latdim])
        ui_iKey = t.reshape(t.permute(ui_iKey, dims=[1, 0, 2]), [-1, args.latdim])
        usrKey = ui_uKey[usrs]
        itmKey = ui_iKey[itms]
        ui_scores = self.label(usrKey, itmKey, ui_uHyper, ui_iHyper)
        _ui_preds = (uEmbeds[usrs]*iEmbeds[itms]).sum(1)
        delta_scores = (ui_scores-_ui_preds).abs()

        indices = delta_scores.where(x<args.threshold) # select prefered edge

        
        val = adj._values()[edgeids]
        idxs = adj._indices()
        edgeNum = vals.size()


        return (ui_scores-_ui_preds).abs() # delta