"""Decisive test: does the Grayscale-Fourpart representation make X=8 detectable
where the interleaved representation did not? Fast small CNN, within & cross model.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from wstego import embed, gf_image
torch.backends.cudnn.benchmark = True

def auc(s, y):
    o=np.argsort(s); r=np.empty_like(o,float); r[o]=np.arange(1,len(s)+1)
    p=y==1; npos,nneg=p.sum(),(~p).sum()
    return float('nan') if npos==0 or nneg==0 else float((r[p].sum()-npos*(npos+1)/2)/(npos*nneg))

class TinyCNN(nn.Module):
    def __init__(s):
        super().__init__()
        s.f=nn.Sequential(
            nn.Conv2d(1,16,3,padding=1),nn.BatchNorm2d(16),nn.ReLU(True),nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(True),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(True),nn.AdaptiveAvgPool2d(1))
        s.h=nn.Linear(64,2)
    def forward(s,x): return s.h(s.f(x).flatten(1))

def build(W, x, rep, seed):
    Xs,ys,mids=[],[],[]
    for mid,w in enumerate(W):
        pl=embed.random_payload_bits(1<<16, seed+mid)
        wst=embed.embed_xlsb(w,pl,x=x,fill=True)
        if rep=="fourpart":
            tc=gf_image.weights_to_fourpart(w,chunk=2500)
            ts=gf_image.weights_to_fourpart(wst,chunk=2500)
        else:
            tc=gf_image.weights_to_tiles(w,tile=100,max_tiles=56,shuffle_seed=seed+mid)
            ts=gf_image.weights_to_tiles(wst,tile=100,max_tiles=56,shuffle_seed=seed+mid)
        Xs+=[tc,ts]; ys+=[np.zeros(len(tc)),np.ones(len(ts))]; mids+=[np.full(len(tc),mid),np.full(len(ts),mid)]
    return np.concatenate(Xs),np.concatenate(ys).astype(np.int64),np.concatenate(mids)

def run(Xtr,ytr,Xte,yte,dev,epochs=6,bs=64):
    net=TinyCNN().to(dev); opt=torch.optim.Adam(net.parameters(),lr=1e-3,weight_decay=1e-4)
    for ep in range(epochs):
        idx=np.random.permutation(len(Xtr))
        for i in range(0,len(Xtr),bs):
            j=idx[i:i+bs]
            xb=torch.from_numpy(Xtr[j].astype(np.float32)[:,None]/255.).to(dev); yb=torch.from_numpy(ytr[j]).to(dev)
            opt.zero_grad(); F.cross_entropy(net(xb),yb).backward(); opt.step()
    net.eval(); sc=[]
    with torch.no_grad():
        for i in range(0,len(Xte),bs):
            xb=torch.from_numpy(Xte[i:i+bs].astype(np.float32)[:,None]/255.).to(dev)
            sc.append(torch.softmax(net(xb),1)[:,1].cpu().numpy())
    return auc(np.concatenate(sc),yte)

def main():
    np.random.seed(0); torch.manual_seed(0)
    dev="cuda" if torch.cuda.is_available() else "cpu"
    Z=np.load("data/zoo_cifar.npz"); W=Z["weights"]; M=len(W)
    print(f"zoo {W.shape} acc={Z['acc'].mean():.3f}",flush=True)
    for rep in ["interleaved","fourpart"]:
        for x in [8]:
            X,y,m=build(W,x,rep,seed=0)
            # within: 70/30 tiles
            ix=np.random.permutation(len(X)); c=int(0.7*len(X))
            within=run(X[ix[:c]],y[ix[:c]],X[ix[c:]],y[ix[c:]],dev)
            # cross: last 8 models test
            tem=np.isin(m,np.arange(M-8,M)); trm=~tem
            cross=run(X[trm],y[trm],X[tem],y[tem],dev)
            print(f"rep={rep:12s} X={x}  within-AUC={within:.3f}  cross-AUC={cross:.3f}",flush=True)

if __name__=="__main__":
    main()
