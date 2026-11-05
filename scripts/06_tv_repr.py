"""Isolate the variable: fourpart vs interleaved on WELL-TRAINED torchvision
models (where signal is known to exist). Tells us if the failure is the
representation or the weak CIFAR zoo."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from wstego import embed, gf_image, zoo
torch.backends.cudnn.benchmark = True

def auc(s,y):
    o=np.argsort(s); r=np.empty_like(o,float); r[o]=np.arange(1,len(s)+1)
    p=y==1; a,b=p.sum(),(~p).sum()
    return float('nan') if a==0 or b==0 else float((r[p].sum()-a*(a+1)/2)/(a*b))

class TinyCNN(nn.Module):
    def __init__(s):
        super().__init__()
        s.f=nn.Sequential(
            nn.Conv2d(1,16,3,padding=1),nn.BatchNorm2d(16),nn.ReLU(True),nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(True),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(True),nn.AdaptiveAvgPool2d(1))
        s.h=nn.Linear(64,2)
    def forward(s,x): return s.h(s.f(x).flatten(1))

def build(W,x,rep,seed,per=56):
    Xs,ys=[],[]
    for mid,w in enumerate(W):
        pl=embed.random_payload_bits(1<<16,seed+mid); wst=embed.embed_xlsb(w,pl,x=x,fill=True)
        if rep=="fourpart":
            tc=gf_image.weights_to_fourpart(w,chunk=2500)[:per]; ts=gf_image.weights_to_fourpart(wst,chunk=2500)[:per]
        else:
            tc=gf_image.weights_to_tiles(w,tile=100,max_tiles=per,shuffle_seed=seed+mid)
            ts=gf_image.weights_to_tiles(wst,tile=100,max_tiles=per,shuffle_seed=seed+mid)
        Xs+=[tc,ts]; ys+=[np.zeros(len(tc)),np.ones(len(ts))]
    return np.concatenate(Xs),np.concatenate(ys).astype(np.int64)

def run(Xtr,ytr,Xte,yte,dev,epochs=6,bs=64):
    net=TinyCNN().to(dev); opt=torch.optim.Adam(net.parameters(),lr=1e-3,weight_decay=1e-4)
    for _ in range(epochs):
        idx=np.random.permutation(len(Xtr))
        for i in range(0,len(Xtr),bs):
            j=idx[i:i+bs]; xb=torch.from_numpy(Xtr[j].astype(np.float32)[:,None]/255.).to(dev); yb=torch.from_numpy(ytr[j]).to(dev)
            opt.zero_grad(); F.cross_entropy(net(xb),yb).backward(); opt.step()
    net.eval(); sc=[]
    with torch.no_grad():
        for i in range(0,len(Xte),bs):
            xb=torch.from_numpy(Xte[i:i+bs].astype(np.float32)[:,None]/255.).to(dev); sc.append(torch.softmax(net(xb),1)[:,1].cpu().numpy())
    return auc(np.concatenate(sc),yte)

def main():
    np.random.seed(0); torch.manual_seed(0)
    dev="cuda" if torch.cuda.is_available() else "cpu"
    W=list(zoo.load_zoo(zoo.DEFAULT_ZOO[:8],verbose=False).values())
    # truncate each to same length so fourpart chunking is comparable
    P=min(len(w) for w in W); W=[w[:P] for w in W]
    print(f"{len(W)} torchvision models, P={P:,}",flush=True)
    for rep in ["interleaved","fourpart"]:
        X,y=build(W,8,rep,seed=0)
        ix=np.random.permutation(len(X)); c=int(0.7*len(X))
        within=run(X[ix[:c]],y[ix[:c]],X[ix[c:]],y[ix[c:]],dev)
        print(f"rep={rep:12s} X=8  within-AUC={within:.3f}",flush=True)

if __name__=="__main__":
    main()
