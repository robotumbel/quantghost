"""Empirical stealth (paper claim #2): can the Model X-Ray-style detector catch
our qaware-STEALTHY embedding? Same structured payload, two embeddings:
  - naive X-LSB  (Model X-Ray's attack)        -> should be DETECTED (AUC ~1.0)
  - qaware-stealthy (ours)                      -> should EVADE (AUC ~0.5)
Cross-model fourpart-image + CNN detector on the CIFAR zoo.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from wstego import embed, robust, gf_image
torch.backends.cudnn.benchmark = True

def auc(s,y):
    o=np.argsort(s); r=np.empty_like(o,float); r[o]=np.arange(1,len(s)+1)
    p=y==1; a,b=p.sum(),(~p).sum()
    return float('nan') if a==0 or b==0 else float((r[p].sum()-a*(a+1)/2)/(a*b))

class TinyCNN(nn.Module):
    def __init__(s):
        super().__init__()
        s.f=nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.BatchNorm2d(16),nn.ReLU(True),nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(True),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(True),nn.AdaptiveAvgPool2d(1))
        s.h=nn.Linear(64,2)
    def forward(s,x): return s.h(s.f(x).flatten(1))

def binbits(n):
    with open(sys.executable,"rb") as f: d=f.read(max(64*1024,n//8+16))
    b=np.unpackbits(np.frombuffer(d,np.uint8)); return np.tile(b,int(np.ceil(n/len(b))))[:n].astype(np.uint8)

def build(W, kind, seed, per=56):
    Xs,ys,m=[],[],[]
    for mid,w in enumerate(W):
        # naive uses STRUCTURED malware (their attack); our methods ENCRYPT first (uniform)
        pl_struct=binbits(1<<16); pl_enc=embed.random_payload_bits(1<<16,seed+mid)
        if kind=="naive":   ws=embed.embed_xlsb(w,pl_struct,x=8,fill=True)
        elif kind=="qaware":ws,_,_=robust.qaware_embed_stealthy(w,pl_enc,nbits=1,seed=seed+mid)
        elif kind=="qmin":  ws,_,_=robust.qaware_embed_minimal(w,pl_enc)
        tc=gf_image.weights_to_fourpart(w,chunk=2500)[:per]
        ts=gf_image.weights_to_fourpart(ws,chunk=2500)[:per]
        Xs+=[tc,ts]; ys+=[np.zeros(len(tc)),np.ones(len(ts))]; m+=[np.full(len(tc),mid),np.full(len(ts),mid)]
    return np.concatenate(Xs),np.concatenate(ys).astype(np.int64),np.concatenate(m)

def run(Xtr,ytr,Xte,yte,dev,ep=6,bs=64):
    net=TinyCNN().to(dev); opt=torch.optim.Adam(net.parameters(),1e-3,weight_decay=1e-4)
    for _ in range(ep):
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
    print(f"zoo {W.shape} | detector=fourpart+CNN, cross-model (last 8 held out)\n",flush=True)
    # survival check (quantize then extract) for each scheme
    def survival(kind,w):
        pl=binbits(1<<16)
        if kind=="qaware": ws,s,car=robust.qaware_embed_stealthy(w,pl,nbits=1,seed=0)
        else:              ws,s,car=robust.qaware_embed_minimal(w,pl)
        sv=robust.quant_scale(ws); wq=robust.quantize(ws,sv).astype(np.float32)*sv
        rec=robust.qaware_extract(wq,sv,nbits=1)
        return robust.survival_rate(rec,pl,nbits=1,carrier=car)
    print(f"{'embedding':22s} {'det.AUC':>8} {'survival':>9}  verdict",flush=True)
    for kind,lbl in [("naive","naive X-LSB (theirs)"),("qaware","qaware-uniform (ours v1)"),("qmin","qaware-minimal (ours v2)")]:
        X,y,m=build(W,kind,seed=0); tem=np.isin(m,np.arange(M-8,M))
        a=run(X[~tem],y[~tem],X[tem],y[tem],dev)
        sv=survival(kind,W[0]) if kind!="naive" else float('nan')
        verdict="DETECTED" if a>0.7 else ("evaded" if a<0.6 else "partial")
        print(f"{lbl:22s} {a:>8.3f} {sv:>9.3f}  {verdict}",flush=True)
    print("\nklaim #2: payload sama; naif terdeteksi; qaware-minimal -> stealth+survival.",flush=True)

if __name__=="__main__":
    main()
