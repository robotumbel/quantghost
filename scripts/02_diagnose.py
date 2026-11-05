"""Diagnostic: WHY did cross-model AUC collapse at X=8?

Separates three hypotheses by running a fast small SRNet under conditions:
  (A) within-model split  : same models in train & test, disjoint tiles
  (B) cross-model split   : held-out models at test
  x payload in {random, structured}

Reading the result matrix:
  within HIGH, cross LOW   -> generalization gap; need few-shot OSL (methodology)
  within LOW (~0.5)        -> stego signal not present in our GF rep for this payload
  random LOW, struct HIGH  -> detectability is payload-structure dependent
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn as nn
from wstego import embed, gf_image, detector, zoo

def auc(s, y):
    o = np.argsort(s); r = np.empty_like(o, float); r[o] = np.arange(1, len(s)+1)
    p = y == 1; npos, nneg = p.sum(), (~p).sum()
    return float("nan") if npos==0 or nneg==0 else float((r[p].sum()-npos*(npos+1)/2)/(npos*nneg))

def struct_payload(nbits, seed=0):
    """Low-entropy structured payload: repeated short motif (like a real binary's runs)."""
    rng = np.random.default_rng(seed)
    motif = rng.integers(0, 2, size=64, dtype=np.uint8)        # 64-bit motif
    motif[16:48] = 0                                            # long zero run -> structure
    reps = int(np.ceil(nbits/len(motif)))
    return np.tile(motif, reps)[:nbits]

def make_tiles(W, x, tile, per_model, payload_kind, seed):
    Xs, ys, mids = [], [], []
    for mid,(name,w) in enumerate(W.items()):
        cap = 1<<16
        pl = embed.random_payload_bits(cap, seed+mid) if payload_kind=="random" \
             else struct_payload(cap, seed+mid)
        wst = embed.embed_xlsb(w, pl, x=x, fill=True)
        tc = gf_image.weights_to_tiles(w,   tile=tile, max_tiles=per_model, shuffle_seed=seed+mid)
        ts = gf_image.weights_to_tiles(wst, tile=tile, max_tiles=per_model, shuffle_seed=seed+mid)
        Xs += [tc, ts]; ys += [np.zeros(len(tc)), np.ones(len(ts))]
        mids += [np.full(len(tc),mid), np.full(len(ts),mid)]
    return np.concatenate(Xs), np.concatenate(ys).astype(np.int64), np.concatenate(mids)

def train_eval(Xtr,ytr,Xte,yte,dev,epochs,bs,width):
    net = detector.make_detector(width=width, device=dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    lf = nn.CrossEntropyLoss(); best=0.0
    for ep in range(epochs):
        net.train(); idx=np.random.permutation(len(Xtr))
        for i in range(0,len(Xtr),bs):
            j=idx[i:i+bs]
            xb=torch.from_numpy(Xtr[j].astype(np.float32)[:,None]/255.).to(dev)
            yb=torch.from_numpy(ytr[j]).to(dev)
            opt.zero_grad(); lf(net(xb),yb).backward(); opt.step()
        net.eval(); sc=[]
        with torch.no_grad():
            for i in range(0,len(Xte),bs):
                xb=torch.from_numpy(Xte[i:i+bs].astype(np.float32)[:,None]/255.).to(dev)
                sc.append(torch.softmax(net(xb),1)[:,1].cpu().numpy())
        best=max(best, auc(np.concatenate(sc), yte))
    return best

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--x",type=int,default=8); ap.add_argument("--tile",type=int,default=64)
    ap.add_argument("--per_model",type=int,default=200); ap.add_argument("--epochs",type=int,default=6)
    ap.add_argument("--bs",type=int,default=64); ap.add_argument("--width",type=float,default=0.25)
    ap.add_argument("--n_models",type=int,default=6)
    a=ap.parse_args()
    np.random.seed(0); torch.manual_seed(0)
    dev="cuda" if torch.cuda.is_available() else "cpu"
    names=zoo.DEFAULT_ZOO[:a.n_models]
    print(f"loading {names}",flush=True); W=zoo.load_zoo(names,verbose=False)
    names=list(W.keys())
    for payload in ["random","structured"]:
        X,y,m=make_tiles(W,a.x,a.tile,a.per_model,payload,seed=0)
        # within-model: split tiles 70/30 regardless of model
        ix=np.random.permutation(len(X)); cut=int(0.7*len(X))
        tr,te=ix[:cut],ix[cut:]
        within=train_eval(X[tr],y[tr],X[te],y[te],dev,a.epochs,a.bs,a.width)
        # cross-model: last 2 models held out
        tmods=set(range(len(names)))- set(range(len(names)-2,len(names)))
        trm=np.isin(m,list(tmods)); tem=~trm
        cross=train_eval(X[trm],y[trm],X[tem],y[tem],dev,a.epochs,a.bs,a.width)
        print(f"payload={payload:10s}  within-model AUC={within:.3f}   cross-model AUC={cross:.3f}",flush=True)

if __name__=="__main__":
    main()
