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
        self.label = LabelNetwork3().to(device)
        self.linear = nn.Linear(args.latdim, args.latdim)
        self.dropout = nn.Dropout(args.dropRate)
        self.leakyrelu = nn.LeakyReLU(args.leaky)
        self.SpAdjDropEdge = SpAdjDropEdge(args.keepRate).to(device)

    def forward(self, adj, uAdj):
        ui_uEmbed0, ui_iEmbed0 = self.LightGCN(adj) # (usr, d)
        uu_Embed0 = self.LightGCN2(uAdj)
        return ui_uEmbed0, ui_iEmbed0, uu_Embed0

    def calcSSL(self, embeds1, embeds2, nodes):
        pckEmbeds1 = F.normalize(embeds1[nodes], 2)
        pckEmbeds2 = F.normalize(embeds2[nodes], 2)
        nume = t.exp(t.sum(pckEmbeds1 * pckEmbeds2, axis=-1) / args.temp) # same node
        deno = t.sum(t.exp(t.mm(pckEmbeds1, t.transpose(embeds2, 0, 1)) / args.temp), axis=-1)
        ssl = t.sum(- t.log(nume / deno))
        return ssl

    def calcLosses(self, adj, usr, itmP, itmN, uAdj, usr0, usrP, usrN, usr1, usr2):
        ui_uEmbed0, ui_iEmbed0, uu_Embed0 = self.forward(adj, uAdj)

        # preds on ui graph
        pckUlat = ui_uEmbed0[usr]
        pckIlatP = ui_iEmbed0[itmP]
        pckIlatN = ui_iEmbed0[itmN]
        predsP = (pckUlat * pckIlatP).sum(-1)
        predsN = (pckUlat * pckIlatN).sum(-1)
        scoreDiff = predsP - predsN
        preLoss = -(scoreDiff).sigmoid().log().sum() / args.batch # bprloss

        # preds on uu graph
        pckUlat = uu_Embed0[usr0]
        pckUlatP = uu_Embed0[usrP]
        pckUlatN = uu_Embed0[usrN]
        predsP = (pckUlat * pckUlatP).sum(-1)
        predsN = (pckUlat * pckUlatN).sum(-1)
        scoreDiff = predsP - predsN
        uuPreLoss = args.uuPre_reg * -(scoreDiff).sigmoid().log().sum() / args.batch # bprloss

        # denoise
        scores = self.label(ui_uEmbed0[usr1], ui_uEmbed0[usr2])
        _preds = (uu_Embed0[usr1] * uu_Embed0[usr1]).sum(-1)
        salLoss = args.sal_reg * (t.maximum(t.tensor(0.0), 1.0-scores*_preds)).sum()

        # contrastive learning
        adj1 = self.SpAdjDropEdge(adj)
        adj2 = self.SpAdjDropEdge(adj)
        uEmbeds1, iEmbeds1, _ = self.forward(adj1, uAdj)
        uEmbeds2, iEmbeds2, _ = self.forward(adj2, uAdj)
        usrSet = t.unique(usr)
        itmSet = t.unique(t.concat([itmP, itmN]))
        sslLoss = args.ssl_reg * (self.calcSSL(uEmbeds1, uEmbeds2, usrSet) + self.calcSSL(iEmbeds1, iEmbeds2, itmSet))
        # sslLoss = t.tensor(0.0)

        return preLoss, uuPreLoss, salLoss, sslLoss

    def predPairs(self, adj, usr, itm, uAdj):
        ret = self.forward(adj, uAdj)
        uEmbeds, iEmbeds = ret[:2]
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

class prepareKey(nn.Module):
    def __init__(self):
        super(prepareKey, self).__init__()
        self.K = nn.Parameter(xavierInit(t.empty(args.latdim, args.latdim)))
    
    def forward(self, nodeEmbed):
        key = t.reshape(nodeEmbed @ self.K, [-1, args.att_head, args.latdim//args.att_head])
        key = t.permute(key, dims=[1, 0, 2])
        return key # (head, n, d')

class prepareValue(nn.Module):
    def __init__(self, V=None):
        super(prepareValue, self).__init__()
        self.V = V if V is not None else nn.Parameter(xavierInit(t.empty(args.latdim, args.latdim)))
    
    def forward(self, nodeEmbed):
        value = t.reshape(nodeEmbed @ self.V, [-1, args.att_head, args.latdim//args.att_head])
        value = t.permute(value, dims=[1, 2, 0])
        return value # (head, d', n)

class HypergraphTransormer(nn.Module):
    def __init__(self):
        super(HypergraphTransormer, self).__init__()
        self.hypergraphLayers = nn.Sequential(*[HypergraphTransformerLayer() for i in range(args.hgnn_layer)])
        self.Hyper = nn.Parameter(xavierInit(t.empty(args.hyperNum, args.latdim)))
        self.V = nn.Parameter(xavierInit(t.empty(args.latdim, args.latdim)))
        self.prepareValue = prepareValue(self.V)

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
        self.dropout = nn.Dropout(args.dropRate)
        self.leakyrelu = nn.LeakyReLU(args.leaky)

    def forward(self, lats, key, value, hyper, V):
        temlat1 = value @ key # (head, d', d')
        # prepare query
        hyper = t.reshape(hyper, [-1, args.att_head, args.latdim//args.att_head]) # (hyperNum, head, d')
        hyper = t.permute(hyper, dims=[1, 2, 0]) # (head, d', hyperNum)
        temlat1 = t.reshape(temlat1 @ hyper, [args.latdim, -1]) # (d, hyperNum)
        temlat2 = self.leakyrelu(self.dropout(self.linear1(temlat1))) + temlat1
        temlat3 = self.leakyrelu(self.dropout(self.linear2(temlat2))) + temlat2

        preNewLat = t.reshape(t.t(temlat3) @ V, [-1, args.att_head, args.latdim//args.att_head])
        preNewLat = t.permute(preNewLat, [1, 0, 2])
        preNewLat = hyper @ preNewLat
        newLat = key @ preNewLat
        newLat = t.reshape(t.permute(newLat, [1, 0, 2]), [-1, args.latdim])
        lats.append(newLat)
        return lats

class HypergraphTransormer1(nn.Module):
    def __init__(self):
        super(HypergraphTransormer1, self).__init__()
        self.hypergraphLayers = nn.Sequential(*[HypergraphTransformerLayer1() for i in range(args.hgnn_layer)])
        self.Hyper = nn.Parameter(xavierInit(t.empty(args.hyperNum, args.latdim)))
        self.K = nn.Parameter(xavierInit(t.empty(args.latdim, args.latdim)))
        self.V = nn.Parameter(xavierInit(t.empty(args.latdim, args.latdim)))

    def forward(self, Embed0):
        lats = [Embed0]
        for hypergraph in self.hypergraphLayers:
            lat = hypergraph(lats[-1], self.K, self.V, self.Hyper)
            lats.append(lat)
        return sum(lats), self.Hyper

class HypergraphTransformerLayer1(nn.Module):
    def __init__(self):
        super(HypergraphTransformerLayer1, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(args.latdim, num_heads=args.att_head, dropout=args.dropRate)
        self.linear1 = nn.Linear(args.hyperNum, args.hyperNum)
        self.linear2 = nn.Linear(args.hyperNum, args.hyperNum)
        self.dropout = nn.Dropout(args.dropRate)
        self.leakyrelu = nn.LeakyReLU(args.leaky)

    def forward(self, nodeEmbeds, K, V, hyper):
        query = hyper # (k, d)
        key = nodeEmbeds @ K # (n, d)
        value = nodeEmbeds @ V
        hyper, _ = self.multihead_attn(query, key, value) # (k, d)
        # hyper = 0.0001 * hyper
        temlat1 = t.t(hyper)
        temlat2 = self.leakyrelu(self.dropout(self.linear1(temlat1))) + temlat1
        temlat3 = self.leakyrelu(self.dropout(self.linear2(temlat2))) + temlat2
        hyper = t.t(temlat3)
        query = key # (n, d)
        key = hyper
        value = hyper @ V
        lats, _ = self.multihead_attn(query, key, value) # (n, d)
        return lats

class Meta(nn.Module):
    def __init__(self):
        super(Meta, self).__init__()
        self.linear1 = nn.Linear(args.latdim, args.latdim * args.latdim, bias=True)
        self.linear2 = nn.Linear(args.latdim, args.latdim, bias=True)
        self.dropout = nn.Dropout(args.dropRate)
        self.leakyrelu = nn.LeakyReLU(args.leaky)
    def forward(self, hyper):
        hyper_mean = t.mean(hyper, dim=0, keepdim=True)
        hyper = hyper_mean
        W1 = t.reshape(self.dropout(self.linear1(hyper)), [args.latdim, args.latdim])
        b1 = self.dropout(self.linear2(hyper))
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
        self.bn1 = nn.BatchNorm1d(args.latdim)
        self.bn2 = nn.BatchNorm1d(1)
        self.leakyrelu = nn.LeakyReLU(args.leaky)
        self.sigmoid = nn.Sigmoid()
    def forward(self, usrKey, itmKey, uHyper, iHyper):
        uMapping = self.meta(uHyper)
        iMapping = self.meta(iHyper)
        ulat = uMapping(usrKey)
        ilat = iMapping(itmKey)
        lat = t.cat((ulat, ilat), dim=-1)
        lat = self.leakyrelu(self.bn1(self.linear1(lat))) + ulat + ilat
        ret = t.reshape(self.sigmoid(self.bn2(self.linear2(lat))), [-1])
        return ret

class Meta2(nn.Module):
    def __init__(self):
        super(Meta2, self).__init__()
        self.linear1 = nn.Linear(args.latdim, args.latdim * args.latdim, bias=True)
        self.linear2 = nn.Linear(args.latdim, args.latdim, bias=True)
        self.linear3 = nn.Linear(args.latdim, args.latdim * args.latdim, bias=True)
        self.linear4 = nn.Linear(args.latdim, args.latdim, bias=True)
        self.dropout = nn.Dropout(args.dropRate)
        self.leakyrelu = nn.LeakyReLU(args.leaky)
    def forward(self, hyper):
        hyper_mean = t.mean(hyper, dim=0, keepdim=True)
        hyper = hyper_mean
        W1 = t.reshape(self.dropout(self.linear1(hyper)), [args.latdim, args.latdim])
        b1 = self.dropout(self.linear2(hyper))
        W2 = t.reshape(self.dropout(self.linear3(hyper)), [args.latdim, args.latdim])
        b2 = self.dropout(self.linear4(hyper))
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
        self.bn1 = nn.BatchNorm1d(args.latdim)
        self.bn2 = nn.BatchNorm1d(1)
        self.leakyrelu = nn.LeakyReLU(args.leaky)
        self.sigmoid = nn.Sigmoid()
    def forward(self, Key1, Key2, Lat1, Lat2, Hyper):
        Mapping = self.meta(Hyper)
        lat1 = Mapping(Key1, Lat1)
        lat2 = Mapping(Key2, Lat2)
        lat = t.cat((lat1, lat2), dim=-1)
        lat = self.leakyrelu(self.bn1(self.linear1(lat))) + lat1 + lat2
        ret = t.reshape(self.sigmoid(self.bn2(self.linear2(lat))), [-1])
        return ret

class Meta3(nn.Module):
    def __init__(self):
        super(Meta3, self).__init__()
        self.linear1 = nn.Linear(args.latdim, args.latdim * args.latdim, bias=True)
        self.linear2 = nn.Linear(args.latdim, args.latdim, bias=True)
        self.dropout = nn.Dropout(args.dropRate)
        self.leakyrelu = nn.LeakyReLU(args.leaky)
    def forward(self, graphEmbed):
        W1 = t.reshape(self.dropout(self.linear1(graphEmbed)), [args.latdim, args.latdim])
        b1 = self.dropout(self.linear2(graphEmbed))
        def mapping(lat):
            ret = self.leakyrelu(lat @ W1 + b1)
            return ret
        return mapping

class LabelNetwork3(nn.Module):
    def __init__(self):
        super(LabelNetwork3, self).__init__()
        self.linear1 = nn.Linear(2*args.latdim, args.latdim, bias=True)
        self.linear2 = nn.Linear(args.latdim, 1, bias=True)
        self.dropout = nn.Dropout(args.dropRate)
        self.leakyrelu = nn.LeakyReLU(args.leaky)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, lat1, lat2):
        # lat = t.cat((lat1, lat2), dim=-1)
        # lat = self.leakyrelu(self.dropout(self.linear1(lat))) + lat1 + lat2
        # ret = t.reshape(self.sigmoid(self.dropout(self.linear2(lat))), [-1])
        ret = (lat1 * lat2).sum(-1)
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

    def forward(self, trnMat, adj, edgeids, ui_uKey, ui_iKey, ui_uHyper, ui_iHyper):
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
        uKey = self.prepareKey1(uEmbeds)
        iKey = self.prepareKey2(iEmbeds)
        ulat, uHyper = self.HypergraphTransormer1(uEmbeds, uKey)
        ilat, iHyper = self.HypergraphTransormer2(iEmbeds, iKey)
        
        return uEmbeds, iEmbeds, ulat, ilat, uKey, iKey, uHyper, iHyper

    def calcLosses(self, adj, usr, itmP, itmN, edgeids, trnMat):
        uEmbeds, iEmbeds, ulat, ilat, uKey, iKey, uHyper, iHyper = self.forward(adj)

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
        _preds = (uEmbeds[usrs]*iEmbeds[itms]).sum(1)

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