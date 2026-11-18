"""Data windows, models and training for the Jaipur wind experiments (PyTorch).

Architectures follow the paper (Attention-TCN: causal dilated TCN blocks with
residual connections, additive attention over days, Dense -> Dropout -> 1).
"""
import math
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from common import FEATURES, TARGET, load

torch.set_num_threads(2)
DEV = 'cpu'

MAIN_SPLIT = {'train': (2000, 2014), 'val': (2015, 2017), 'test': (2018, 2020)}
SECOND_SPLIT = {'train': (2000, 2010), 'val': (2011, 2012), 'test': (2013, 2020)}
STATION_SPLIT = {'train': (2005, 2014), 'val': (2015, 2017), 'test': (2018, 2020)}


# ----------------------------------------------------------------- data
class Data:
    """Windows of L days ending on day t; target is wind speed on day t + h.
    Each sample is assigned to a split by its target date. Scaling uses the
    training split only."""

    def __init__(self, L=7, h=0, split=MAIN_SPLIT, df=None, target=TARGET, features=FEATURES):
        d = load() if df is None else df
        self.df = d
        X = d[features].values.astype(np.float64)
        y = d[target].values.astype(np.float64)
        years = d.date.dt.year.values
        n = len(d)
        tgt = np.arange(L - 1 + h, n)          # index of target day
        tgt = tgt[np.isfinite(y[tgt])]          # only days with a valid target
        self.L, self.h = L, h
        self.idx = {}
        for k, (a, b) in split.items():
            self.idx[k] = tgt[(years[tgt] >= a) & (years[tgt] <= b)]
        # scaler from training-period days (inputs and target)
        tr_days = np.where((years >= split['train'][0]) & (years <= split['train'][1]))[0]
        self.mx, self.sx = X[tr_days].mean(0), X[tr_days].std(0)
        yt = y[tr_days][np.isfinite(y[tr_days])]
        self.my, self.sy = yt.mean(), yt.std()
        self.Xz = (X - self.mx) / self.sx
        self.yz = (y - self.my) / self.sy
        self.y = y

    def get(self, k):
        t = self.idx[k]
        offs = np.arange(-(self.L - 1), 1) - self.h
        win = t[:, None] + offs[None, :]
        return self.Xz[win].astype(np.float32), self.yz[t].astype(np.float32), t

    def to_ms(self, z):
        return np.asarray(z) * self.sy + self.my


# ----------------------------------------------------------------- models
class AdditiveAttention(nn.Module):
    """e_t = w2 . tanh(W1 h_t + b1) + b2 ; alpha = softmax(e) ; c = sum alpha_t h_t"""

    def __init__(self, dim, att):
        super().__init__()
        self.w1 = nn.Linear(dim, att)
        self.w2 = nn.Linear(att, 1)
        self.last_weights = None

    def forward(self, H):
        e = self.w2(torch.tanh(self.w1(H))).squeeze(-1)
        a = torch.softmax(e, dim=1)
        self.last_weights = a.detach()
        return (a.unsqueeze(-1) * H).sum(1)


class CausalConv(nn.Module):
    def __init__(self, cin, cout, k, d):
        super().__init__()
        self.pad = (k - 1) * d
        self.conv = nn.Conv1d(cin, cout, k, dilation=d)

    def forward(self, x):  # x: (B, C, T)
        return self.conv(nn.functional.pad(x, (self.pad, 0)))


class TCNBlock(nn.Module):
    def __init__(self, cin, cout, k, d, drop):
        super().__init__()
        self.c1, self.c2 = CausalConv(cin, cout, k, d), CausalConv(cout, cout, k, d)
        self.drop = nn.Dropout(drop)
        self.proj = nn.Conv1d(cin, cout, 1) if cin != cout else None

    def forward(self, x):
        o = self.drop(torch.relu(self.c1(x)))
        o = self.drop(torch.relu(self.c2(o)))
        res = x if self.proj is None else self.proj(x)
        return torch.relu(o + res)


class Head(nn.Module):
    def __init__(self, dim, dense, drop):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dense), nn.ReLU(), nn.Dropout(drop), nn.Linear(dense, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class TCNNet(nn.Module):
    def __init__(self, nf, filters=32, kernel=5, blocks=2, dropout=0.2, att=8, dense=16, attention=True, **_):
        super().__init__()
        layers, cin = [], nf
        for b in range(blocks):
            layers.append(TCNBlock(cin, filters, kernel, 2 ** b, dropout))
            cin = filters
        self.tcn = nn.Sequential(*layers)
        self.att = AdditiveAttention(filters, att) if attention else None
        self.head = Head(filters, dense, dropout)

    def forward(self, x):  # x: (B, T, F)
        H = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        c = self.att(H) if self.att is not None else H[:, -1]
        return self.head(c)


class LSTMNet(nn.Module):
    def __init__(self, nf, units=32, layers=1, dropout=0.2, dense=16, bidirectional=False, attention=False, att=8, **_):
        super().__init__()
        self.rnn = nn.LSTM(nf, units, layers, batch_first=True, bidirectional=bidirectional,
                           dropout=dropout if layers > 1 else 0.0)
        dim = units * (2 if bidirectional else 1)
        self.att = AdditiveAttention(dim, att) if attention else None
        self.bi, self.units = bidirectional, units
        self.drop = nn.Dropout(dropout)
        self.head = Head(dim, dense, dropout)

    def forward(self, x):
        H, _ = self.rnn(x)
        if self.att is not None:
            c = self.att(H)
        elif self.bi:  # forward state at last step, backward state at first step
            c = torch.cat([H[:, -1, :self.units], H[:, 0, self.units:]], -1)
        else:
            c = H[:, -1]
        return self.head(self.drop(c))


class TransformerNet(nn.Module):
    def __init__(self, nf, d_model=32, heads=4, layers=1, ff=64, dropout=0.1, dense=16, **_):
        super().__init__()
        self.inp = nn.Linear(nf, d_model)
        pe = torch.zeros(64, d_model)
        pos = torch.arange(64).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer('pe', pe)
        layer = nn.TransformerEncoderLayer(d_model, heads, ff, dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = Head(d_model, dense, dropout)

    def forward(self, x):
        z = self.inp(x) + self.pe[: x.shape[1]]
        return self.head(self.enc(z)[:, -1])  # representation of the last day


def build(name, nf, hp):
    if name == 'Attention-TCN':
        return TCNNet(nf, attention=True, **hp)
    if name == 'TCN':
        return TCNNet(nf, attention=False, **hp)
    if name == 'LSTM':
        return LSTMNet(nf, **hp)
    if name == 'BiLSTM':
        return LSTMNet(nf, bidirectional=True, **hp)
    if name == 'Attention-BiLSTM':
        return LSTMNet(nf, bidirectional=True, attention=True, **hp)
    if name == 'Transformer':
        return TransformerNet(nf, **hp)
    raise ValueError(name)


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ----------------------------------------------------------------- training
def train_net(name, hp, data, seed, max_epochs=300, patience=15, weight_fn=None, return_curve=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xtr, ytr, _ = data.get('train')
    Xva, yva, _ = data.get('val')
    Xtr_t, ytr_t = torch.tensor(Xtr), torch.tensor(ytr)
    Xva_t = torch.tensor(Xva)
    w_t = torch.tensor(weight_fn(ytr), dtype=torch.float32) if weight_fn else torch.ones(len(ytr))
    model = build(name, Xtr.shape[2], hp).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=hp['lr'])
    l2 = hp.get('l2', 0.0)
    huber = nn.HuberLoss(reduction='none')
    bs = hp['batch']
    best, best_state, bad, curve = np.inf, None, 0, []
    g = torch.Generator().manual_seed(seed)
    t0 = time.time()
    for ep in range(max_epochs):
        model.train()
        perm = torch.randperm(len(ytr), generator=g)
        tot = 0.0
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            loss = (huber(model(Xtr_t[b]), ytr_t[b]) * w_t[b]).sum() / w_t[b].sum()
            if l2:
                loss = loss + l2 * sum((p ** 2).sum() for n_, p in model.named_parameters() if p.dim() > 1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            pv = model(Xva_t).numpy()
        vloss = float(np.mean(huber(torch.tensor(pv), torch.tensor(yva)).numpy()))
        vrmse = float(np.sqrt(np.mean((pv - yva) ** 2)))
        curve.append((ep + 1, tot / len(ytr), vloss))
        if vrmse < best - 1e-6:
            best, bad = vrmse, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    info = {'val_rmse_z': best, 'epochs': len(curve), 'train_s': time.time() - t0}
    if return_curve:
        info['curve'] = curve
    return model, info


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X)).numpy()


# ----------------------------------------------------------------- search spaces
SPACES = {
    'Attention-TCN': dict(filters=[16, 32, 64], kernel=[2, 3, 5], blocks=[1, 2, 3], dropout=[0.0, 0.1, 0.2, 0.3],
                          l2=[0.0, 1e-5, 1e-4, 1e-3], att=[8, 16, 32], dense=[16, 32, 64],
                          lr=[3e-4, 1e-3, 3e-3], batch=[32, 64, 128]),
    'TCN': dict(filters=[16, 32, 64], kernel=[2, 3, 5], blocks=[1, 2, 3], dropout=[0.0, 0.1, 0.2, 0.3],
                l2=[0.0, 1e-5, 1e-4, 1e-3], dense=[16, 32, 64], lr=[3e-4, 1e-3, 3e-3], batch=[32, 64, 128]),
    'LSTM': dict(units=[16, 32, 64], layers=[1, 2], dropout=[0.0, 0.1, 0.2, 0.3], l2=[0.0, 1e-5, 1e-4, 1e-3],
                 dense=[16, 32, 64], lr=[3e-4, 1e-3, 3e-3], batch=[32, 64, 128]),
    'Transformer': dict(d_model=[16, 32, 64], heads=[2, 4], layers=[1, 2], ff=[32, 64, 128],
                        dropout=[0.0, 0.1, 0.2, 0.3], l2=[0.0, 1e-5, 1e-4, 1e-3], dense=[16, 32, 64],
                        lr=[3e-4, 1e-3, 3e-3], batch=[32, 64, 128]),
}
SPACES['BiLSTM'] = SPACES['LSTM']
SPACES['Attention-BiLSTM'] = dict(SPACES['LSTM'], att=[8, 16, 32])
XGB_SPACE = dict(max_depth=[2, 3, 4, 6, 8], learning_rate=[0.01, 0.03, 0.1], subsample=[0.6, 0.8, 1.0],
                 colsample_bytree=[0.5, 0.8, 1.0], min_child_weight=[1, 5, 10], reg_lambda=[0.0, 1.0, 10.0])

# hand-chosen setting from the original paper (Table 1)
FIXED = dict(filters=32, kernel=5, blocks=2, dropout=0.2, l2=0.0, att=8, dense=16, lr=1e-3, batch=64)


def sample(space, rng):
    return {k: v[rng.integers(len(v))] for k, v in space.items()}
