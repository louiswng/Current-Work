import torch as t
from torch import nn
from Params import args

xavierInit = nn.init.xavier_uniform_
zeroInit = lambda x: nn.init.constant_(x, 0.0)

class Our(nn.Module):
    def __init__(self, device):
        super(Our, self).__init__()
        self.device = device
        self.uEmbeds0 = nn.Parameter(xavierInit(t.empty(args.user, args.latdim)))
        self.iEmbeds0 = nn.Parameter(xavierInit(t.empty(args.item, args.latdim)))
        self.LightGCN = LightGCN(self.uEmbeds0, self.iEmbeds0).to(self.device)
        self.LightGCN2 = LightGCN2(self.uEmbeds0).to(self.device)
        self.prepareKey1 = prepareKey().to(self.device)
        self.prepareKey2 = prepareKey().to(self.device)
        self.prepareKey3 = prepareKey().to(self.device)
        self.HypergraphTransormer1 = HypergraphTransormer().to(self.device)
        self.HypergraphTransormer2 = HypergraphTransormer().to(self.device)
        self.HypergraphTransormer3 = HypergraphTransormer().to(self.device)
        self.label = LabelNetwork().to(self.device)
        
    def forward(self, adj, tpAdj, uAdj, uids, iids, edgeids, trnMat, uu_ids1, uu_ids2, uuedgeids, uuMat):
        ui_uEmbed_gcn, ui_iEmbed_gcn = self.LightGCN(adj, tpAdj) # (usr, d)
        uu_Embed_gcn = self.LightGCN2(uAdj)
        # Residual Connection, add positional information
        ui_uEmbed0 = self.uEmbeds0 + ui_uEmbed_gcn
        ui_iEmbed0 = self.iEmbeds0 + ui_iEmbed_gcn
        uu_Embed0 = self.uEmbeds0 + uu_Embed_gcn

        ui_uKey = self.prepareKey1(ui_uEmbed0)
        ui_iKey = self.prepareKey2(ui_iEmbed0)
        uu_Key = self.prepareKey3(uu_Embed0)
        self.ui_ulat, ui_uHyper = self.HypergraphTransormer1(ui_uEmbed0, ui_uKey)
        self.ui_ilat, ui_iHyper = self.HypergraphTransormer2(ui_iEmbed0, ui_iKey)
        uu_lat, uu_Hyper = self.HypergraphTransormer3(uu_Embed0, uu_Key)

        ui_pckUlat = self.ui_ulat[uids] # (batch, d)
        ui_pckIlat = self.ui_ilat[iids]
        ui_preds = t.sum(ui_pckUlat * ui_pckIlat, dim=-1) # (batch, batch, d)

        uu_pcklat1 = uu_lat[uu_ids1] # (batch, d)
        uu_pcklat2 = uu_lat[uu_ids2]
        uu_preds = t.sum(uu_pcklat1 * uu_pcklat2, dim=-1) # (batch, batch, d)  
        
        coo = uuMat.tocoo()
        usrs1, usrs2 = coo.row[uuedgeids], coo.col[uuedgeids]
        uu_Key = t.reshape(t.permute(uu_Key, dims=[1, 0, 2]), [-1, args.latdim])
        usrKey1 = uu_Key[usrs1] # (batch, d)
        usrKey2 = uu_Key[usrs2]
        uu_Hyper = (uu_Hyper + ui_uHyper) / 2
        usrLat1 = self.ui_ulat[usrs1] # (batch, d)
        usrLat2 = self.ui_ulat[usrs2]
        uu_scores = self.label(usrKey1, usrKey2, usrLat1, usrLat2, uu_Hyper) # (batch, k)
        _uu_preds = t.sum(uu_lat[usrs1]*uu_lat[usrs2], dim=1)

        halfNum = uu_scores.shape[0] // 2
        fstScores = uu_scores[:halfNum]
        scdScores = uu_scores[halfNum:]
        fstPreds = _uu_preds[:halfNum]
        scdPreds = _uu_preds[halfNum:]
        ssuLoss = t.sum(t.maximum(t.tensor(0.0), 1.0 - (fstPreds - scdPreds) * args.mult * (fstScores-scdScores)))
        ssuLoss = t.tensor(0.0)
        
        return ui_preds, uu_preds, ssuLoss

    def test(self, usr, trnMask):
        pckUlat = self.ui_ulat[usr]
        allPreds = pckUlat @ t.t(self.ui_ilat)
        allPreds = allPreds * (1 - trnMask) - trnMask * 1e8
        _, topLocs = t.topk(allPreds, args.shoot)
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
        self.hypergraphLayers = nn.Sequential(*[HypergraphTransformerLayer() for i in range(args.gnn_layer)])
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

class LabelNetwork(nn.Module):
    def __init__(self):
        super(LabelNetwork, self).__init__()
        self.meta = Meta()
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

