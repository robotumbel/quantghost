"""Grounding: reproduce Model X-Ray's detection behavior in our torch pipeline,
now using a STRUCTURED (real binary) payload -- their actual protocol.

Expectation (matching their fsl_osl.csv shape): detection ~chance for X<8, then a
sharp jump at X=8 (one full payload byte per weight lands in the low-byte plane).
Then: re-embed the SAME payload ENCRYPTED (uniform) at X=8 and show detection
collapses to chance -> the evasion, grounded.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from wstego import embed, gf_image
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

def load_binary_payload_bits(nbits):
    """Use a real structured binary (the python executable, PE format) as payload."""
    with open(sys.executable, "rb") as f:
        data = f.read(max(64*1024, nbits//8 + 16))
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    reps = int(np.ceil(nbits/len(bits)))
    return np.tile(bits, reps)[:nbits].astype(np.uint8)

def build(W, x, payload_kind, seed, per=56):
    Xs,ys,mids=[],[],[]
    for mid,w in enumerate(W):
        cap=1<<16
        if payload_kind=="structured":
            pl=load_binary_payload_bits(cap)
        else:  # encrypted == uniform random
            pl=embed.random_payload_bits(cap, seed+mid)
        wst=embed.embed_xlsb(w,pl,x=x,fill=True)
        tc=gf_image.weights_to_fourpart(w,chunk=2500)[:per]
        ts=gf_image.weights_to_fourpart(wst,chunk=2500)[:per]
        Xs+=[tc,ts]; ys+=[np.zeros(len(tc)),np.ones(len(ts))]; mids+=[np.full(len(tc),mid),np.full(len(ts),mid)]
    return np.concatenate(Xs),np.concatenate(ys).astype(np.int64),np.concatenate(mids)

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
    Z=np.load("data/zoo_cifar.npz"); W=Z["weights"]; M=len(W)
    print(f"zoo {W.shape} acc={Z['acc'].mean():.3f}  | fourpart rep, cross-model held-out\n",flush=True)
    print("--- GROUNDING: structured (real binary) payload, detection vs X ---",flush=True)
    for x in [1,4,8,16]:
        X,y,m=build(W,x,"structured",seed=0)
        tem=np.isin(m,np.arange(M-8,M));
        a=run(X[~tem],y[~tem],X[tem],y[tem],dev)
        print(f"  X={x:>2}  detection AUC={a:.3f}",flush=True)
    print("\n--- EVASION: same capacity X=8, ENCRYPTED (uniform) payload ---",flush=True)
    X,y,m=build(W,8,"encrypted",seed=0)
    tem=np.isin(m,np.arange(M-8,M))
    a=run(X[~tem],y[~tem],X[tem],y[tem],dev)
    print(f"  X= 8  detection AUC={a:.3f}   (<- collapses to chance = evaded)",flush=True)

if __name__=="__main__":
    main()
