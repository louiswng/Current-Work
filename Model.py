import torch as t
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
        self.prepareKey1 = prepareKey().to(device)
        self.prepareKey2 = prepareKey().to(device)
        self.prepareKey3 = prepareKey().to(device)
        self.HypergraphTransormer1 = HypergraphTransormer().to(device)
        self.HypergraphTransormer2 = HypergraphTransormer().to(device)
        self.HypergraphTransormer3 = HypergraphTransormer().to(device)
        # self.label = LabelNetwork().to(device)
        self.label2 = LabelNetwork2().to(device)
        self.SpAdjDropEdge = SpAdjDropEdge(args.keepRate).to(device)
        self.SpAdjDropEdge2 = SpAdjDropEdge2(args.keepRate).to(device)
        
    def forward(self, adj, uAdj):
        ui_uEmbed_gcn, ui_iEmbed_gcn = self.LightGCN(adj) # (usr, d)
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
        nume = ((pckEmbeds1 * pckEmbeds2).sum(-1) / args.temp).exp() # same node
        deno = (pckEmbeds1.mm(t.transpose(embeds2, 0, 1)) / args.temp).exp().sum(-1)
        ssl = (-(nume / deno).log()).sum()
        return ssl

    def calcLosses(self, adj, uAdj, usr, itmP, itmN, edgeids1, edgeids2, trnMat, usrP, usrN, uuedgeids, uuMat):
        uEmbeds, iEmbeds, ui_ulat, ui_ilat, ui_uKey, ui_iKey, ui_uHyper, ui_iHyper, uu_Embed0, uu_lat, uu_Key, uu_Hyper = self.forward(adj, uAdj)
        ui_pckUlat = ui_ulat[usr]
        ui_pckIlatP = ui_ilat[itmP]
        ui_pckIlatN = ui_ilat[itmN]
        predsP = (ui_pckUlat * ui_pckIlatP).sum(-1)
        predsN = (ui_pckUlat * ui_pckIlatN).sum(-1)
        scoreDiff = predsP - predsN
        preLoss = (t.maximum(t.tensor(0.0), 1.0 - scoreDiff)).sum() / args.batch

        uu_pcklat = uu_lat[usr]
        uu_pcklatP = uu_lat[usrP]
        uu_pcklatN = uu_lat[usrN]
        predsP = (uu_pcklat * uu_pcklatP).sum(-1)
        predsN = (uu_pcklat * uu_pcklatN).sum(-1)
        scoreDiff = predsP - predsN
        uuPreLoss = args.uuPre_reg * (t.maximum(t.tensor(0.0), 1.0 - scoreDiff)).sum() / args.batch
        
        # adj1 = self.SpAdjDropEdge(adj)
        # adj2 = self.SpAdjDropEdge(adj)
        adj1 = self.SpAdjDropEdge2(trnMat, adj, edgeids1, uEmbeds, iEmbeds, ui_uKey, ui_iKey, ui_uHyper, ui_iHyper) # update per batch
        adj2 = self.SpAdjDropEdge2(trnMat, adj, edgeids2, uEmbeds, iEmbeds, ui_uKey, ui_iKey, ui_uHyper, ui_iHyper)
        ret = self.forward(adj1, uAdj)
        uEmbeds1, iEmbeds1 = ret[2:4] # global
        ret = self.forward(adj2, uAdj)
        uEmbeds2, iEmbeds2 = ret[2:4]

        usrSet = t.unique(usr)
        itmSet = t.unique(t.concat([itmP, itmN]))
        sslLoss = args.ssl_reg * (self.calcSSL(uEmbeds1, uEmbeds2, usrSet) + self.calcSSL(iEmbeds1, iEmbeds2, itmSet))

        coo = uuMat.tocoo()
        usrs1, usrs2 = coo.row[uuedgeids], coo.col[uuedgeids]
        uu_Key = t.reshape(t.permute(uu_Key, dims=[1, 0, 2]), [-1, args.latdim])
        usrKey1 = uu_Key[usrs1]
        usrKey2 = uu_Key[usrs2]
        uu_Hyper = (uu_Hyper + ui_uHyper) / 2
        usrLat1 = ui_ulat[usrs1]
        usrLat2 = ui_ulat[usrs2]
        uu_scores = self.label2(usrKey1, usrKey2, usrLat1, usrLat2, uu_Hyper)
        _uu_preds = (uu_Embed0[usrs1]*uu_Embed0[usrs2]).sum(-1)

        halfNum = uu_scores.shape[0] // 2
        fstScores = uu_scores[:halfNum]
        scdScores = uu_scores[halfNum:]
        fstPreds = _uu_preds[:halfNum]
        scdPreds = _uu_preds[halfNum:]
        salLoss = args.sal_reg * (t.maximum(t.tensor(0.0), 1.0 - (fstPreds - scdPreds) * (fstScores-scdScores))).sum()

        return preLoss, uuPreLoss, sslLoss, salLoss

    def predPairs(self, adj, usr, itm):
        ret = self.forward(adj)
        uEmbeds, iEmbeds = ret[2:4] # global embeds
        uEmbed = uEmbeds[usr]
        iEmbed = iEmbeds[itm]
        return (uEmbed * iEmbed).sum(-1)

class LightGCN(nn.Module):
    def __init__(self, uEmbeds=None, iEmbeds=None):
        super(LightGCN, self).__init__()
        self.uEmbeds = uEmbeds if uEmbeds is not None else nn.Parameter(xavierInit(t.empty(args.user, args.latdim)))
        self.iEmbeds = iEmbeds if iEmbeds is not None else nn.Parameter(xavierInit(t.empty(args.item, args.latdim)))
        self.gcnLayers = nn.Sequential(*[GCNLayer() for i in range(args.gnn_layer)])

    def forward(self, adj):
        embeds = t.concat([self.uEmbeds, self.iEmbeds], axis=0)
        embedLst = [embeds]
        for gcn in self.gcnLayers:
            embeds = gcn(adj, embedLst[-1])
            embedLst.append(embeds)
        embeds = sum(embedLst)
        return embeds[:args.user], embeds[args.user:]

    def calcLosses(self, adj, usr, itmP, itmN):
        uEmbeds, iEmbeds = self.forward(adj)
        uEmbed = uEmbeds[usr]
        iEmbedP = iEmbeds[itmP]
        iEmbedN = iEmbeds[itmN]
        predsP = (uEmbed * iEmbedP).sum(-1)
        predsN = (uEmbed * iEmbedN).sum(-1)
        scoreDiff = predsP - predsN
        bprLoss = -(scoreDiff).sigmoid().log().sum() / args.batch
        return bprLoss

    def predPairs(self, adj, usr, itm):
        uEmbeds, iEmbeds = self.forward(adj)
        uEmbed = uEmbeds[usr]
        iEmbed = iEmbeds[itm]
        return (uEmbed * iEmbed).sum(-1)

    def predAll(self, adj, usr, itm=None):
        uEmbeds, iEmbeds = self.forward(adj)
        uEmbed = uEmbeds[usr]
        if itm is not None:
            iEmbeds = iEmbeds[itm]
        return t.mm(uEmbed, t.transpose(iEmbeds, 1, 0))

class SGL(nn.Module):
    def __init__(self):
        super(SGL, self).__init__()
        self.uEmbeds = nn.Parameter(xavierInit(t.empty(args.user, args.latdim)))
        self.iEmbeds = nn.Parameter(xavierInit(t.empty(args.item, args.latdim)))
        self.LightGCN = LightGCN(self.uEmbeds, self.iEmbeds).to(device)
        self.SpAdjDropEdge = SpAdjDropEdge(args.keepRate).to(device)

    def forward(self, adj):
        uEmbeds, iEmbeds = self.LightGCN(adj)
        return uEmbeds, iEmbeds

    def calcSSL(self, embeds1, embeds2, nodes):
        pckEmbeds1 = F.normalize(embeds1[nodes], 2)
        pckEmbeds2 = F.normalize(embeds2[nodes], 2)
        nume = t.exp(t.sum(pckEmbeds1 * pckEmbeds2, axis=-1) / args.temp) # same node
        deno = t.sum(t.exp(t.mm(pckEmbeds1, t.transpose(embeds2, 0, 1)) / args.temp), axis=-1)
        ssl = t.sum(- t.log(nume / deno))
        return ssl

    def calcLosses(self, adj, usr, itmP, itmN):
        uEmbeds, iEmbeds = self.forward(adj)
        uEmbed = uEmbeds[usr]
        iEmbedP = iEmbeds[itmP]
        iEmbedN = iEmbeds[itmN]
        predsP = (uEmbed * iEmbedP).sum(-1)
        predsN = (uEmbed * iEmbedN).sum(-1)
        scoreDiff = predsP - predsN
        bprLoss = -(scoreDiff).sigmoid().log().sum() / args.batch

        adj1 = self.SpAdjDropEdge(adj)
        adj2 = self.SpAdjDropEdge(adj)
        uEmbeds1, iEmbeds1 = self.forward(adj1)
        uEmbeds2, iEmbeds2 = self.forward(adj2)
        usrSet = t.unique(usr)
        itmSet = t.unique(t.concat([itmP, itmN]))
        sslLoss = args.ssl_reg * (self.calcSSL(uEmbeds1, uEmbeds2, usrSet) + self.calcSSL(iEmbeds1, iEmbeds2, itmSet))

        return bprLoss, sslLoss

    def predPairs(self, adj, usr, itm):
        uEmbeds, iEmbeds = self.forward(adj)
        uEmbed = uEmbeds[usr]
        iEmbed = iEmbeds[itm]
        return (uEmbed * iEmbed).sum(-1)

class LightGCN2(nn.Module):
    def __init__(self, uEmbeds=None):
        super(LightGCN2, self).__init__()

        self.uEmbeds = uEmbeds if uEmbeds is not None else nn.Parameter(xavierInit(t.empty(args.user, args.latdim)))
        self.gnnLayers = nn.Sequential(*[GCNLayer() for i in range(args.gnn_layer)])
    
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
    
    def forward(self, adj):
        vals = adj._values()
        idxs = adj._indices()
        edgeNum = vals.size()
        mask = ((t.rand(edgeNum) + self.keepRate).floor()).type(t.bool)
        newVals = vals[mask] / self.keepRate
        newIdxs = idxs[:, mask]
        return t.sparse.FloatTensor(newIdxs, newVals, adj.shape)
        
class SpAdjDropEdge2(nn.Module):
    def __init__(self, keepRate):
        super(SpAdjDropEdge2, self).__init__()
        self.label = LabelNetwork().to(device)
        self.keepRate = keepRate

    def forward(self, trnMat, adj, edgeids, uEmbeds, iEmbeds, ui_uKey, ui_iKey, ui_uHyper, ui_iHyper):
        coo = trnMat.tocoo()
        usrs, itms = coo.row[edgeids], coo.col[edgeids]
        ui_uKey = t.reshape(t.permute(ui_uKey, dims=[1, 0, 2]), [-1, args.latdim])
        ui_iKey = t.reshape(t.permute(ui_iKey, dims=[1, 0, 2]), [-1, args.latdim])
        usrKey = ui_uKey[usrs]
        itmKey = ui_iKey[itms]
        ui_scores = self.label(usrKey, itmKey, ui_uHyper, ui_iHyper)      
        _, topLocs = t.topk(ui_scores, int(len(edgeids) * self.keepRate))

        val = adj._values()
        idxs = adj._indices()
        newVals = val[topLocs]
        newIdxs = idxs[:, topLocs]
        return t.sparse.FloatTensor(newIdxs, newVals, adj.shape)  

class SHT(nn.Module):
    def __init__(self):
        super(SHT, self).__init__()
        self.uEmbeds = nn.Parameter(xavierInit(t.empty(args.user, args.latdim)))
        self.iEmbeds = nn.Parameter(xavierInit(t.empty(args.item, args.latdim)))
        self.LightGCN = LightGCN(self.uEmbeds, self.iEmbeds).to(device)
        self.prepareKey1 = prepareKey().to(device)
        self.prepareKey2 = prepareKey().to(device)
        self.HypergraphTransormer1 = HypergraphTransormer().to(device)
        self.HypergraphTransormer2 = HypergraphTransormer().to(device)
        self.label = LabelNetwork().to(device)
        
    def forward(self, adj):
        uEmbeds, iEmbeds = self.LightGCN(adj)
        uEmbeds0 = uEmbeds + self.uEmbeds # residual connection
        iEmbeds0 = iEmbeds + self.iEmbeds
        uKey = self.prepareKey1(uEmbeds0)
        iKey = self.prepareKey2(iEmbeds0)
        ulat, uHyper = self.HypergraphTransormer1(uEmbeds0, uKey)
        ilat, iHyper = self.HypergraphTransormer2(iEmbeds0, iKey)
        
        return uEmbeds0, iEmbeds0, ulat, ilat, uKey, iKey, uHyper, iHyper

    def calcLosses(self, adj, usr, itmP, itmN, edgeids, trnMat):
        uEmbeds0, iEmbeds0, ulat, ilat, uKey, iKey, uHyper, iHyper = self.forward(adj)

        pckUlat = ulat[usr]
        pckIlatP = ilat[itmP]
        pckIlatN = ilat[itmN]
        predsP = (pckUlat * pckIlatP).sum(-1)
        predsN = (pckUlat * pckIlatN).sum(-1)
        scoreDiff = predsP - predsN
        preLoss = (t.maximum(t.tensor(0.0), 1.0 - scoreDiff)).sum() / args.batch

        coo = trnMat.tocoo()
        usrs, itms = coo.row[edgeids], coo.col[edgeids]
        uKey = t.reshape(t.permute(uKey, dims=[1, 0, 2]), [-1, args.latdim])
        iKey = t.reshape(t.permute(iKey, dims=[1, 0, 2]), [-1, args.latdim])
        usrKey = uKey[usrs]
        itmKey = iKey[itms]
        scores = self.label(usrKey, itmKey, uHyper, iHyper)
        _preds = (uEmbeds0[usrs]*iEmbeds0[itms]).sum(1)

        halfNum = scores.shape[0] // 2
        fstScores = scores[:halfNum]
        scdScores = scores[halfNum:]
        fstPreds = _preds[:halfNum]
        scdPreds = _preds[halfNum:]
        sslLoss = args.ssl_reg * (t.maximum(t.tensor(0.0), 1.0 - (fstPreds - scdPreds) * (fstScores-scdScores))).sum()
        return preLoss, sslLoss

    def predPairs(self, adj, usr, itm):
        ret = self.forward(adj)
        uEmbeds, iEmbeds = ret[2:4] # global embeds
        uEmbed = uEmbeds[usr]
        iEmbed = iEmbeds[itm]
        return (uEmbed * iEmbed).sum(-1)