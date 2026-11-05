"""Phase 3, hardest part: payload survival through FINE-TUNING and PRUNING.

Quantization doesn't move weights (sub-quantum placement survives). Fine-tuning
and pruning DO change weights -> need REDUNDANCY. Each message bit is embedded in
many weights (position g carries message bit g mod n_msg); extraction majority-
votes all copies. Smaller message = higher redundancy = more robust. We measure
message-recovery vs redundancy through: fine-tuning, then pruning, then quant.
"""
import sys, os, copy, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torchvision as tv
from wstego import robust
torch.backends.cudnn.benchmark = True

class SmallCNN(nn.Module):
    def __init__(s,nc=10):
        super().__init__()
        def blk(i,o): return nn.Sequential(nn.Conv2d(i,o,3,padding=1),nn.BatchNorm2d(o),nn.ReLU(True))
        s.f=nn.Sequential(blk(3,32),blk(32,32),nn.MaxPool2d(2),blk(32,64),blk(64,64),nn.MaxPool2d(2),blk(64,128),nn.AdaptiveAvgPool2d(1))
        s.h=nn.Linear(128,10)
    def forward(s,x): return s.h(s.f(x).flatten(1))

def cifar(dev):
    tr=tv.datasets.CIFAR10("data/cifar",train=True,download=True); te=tv.datasets.CIFAR10("data/cifar",train=False,download=True)
    f=lambda d:(torch.from_numpy(d.data).to(dev).permute(0,3,1,2).contiguous(),torch.tensor(d.targets,device=dev))
    return f(tr),f(te)

def acc(net,Xte,yte,dev):
    net.eval(); c=0
    with torch.no_grad():
        for i in range(0,len(Xte),1000): c+=(net(Xte[i:i+1000].float()/255.).argmax(1)==yte[i:i+1000]).sum().item()
    return c/len(Xte)

def train(net,Xtr,ytr,dev,epochs,lr,bs=512,sub=10000):
    sel=torch.randperm(len(Xtr),device=dev)[:sub]; Xtr,ytr=Xtr[sel],ytr[sel]
    opt=torch.optim.Adam(net.parameters(),lr,weight_decay=5e-4)
    for _ in range(epochs):
        p=torch.randperm(len(Xtr),device=dev)
        for i in range(0,len(Xtr),bs):
            idx=p[i:i+bs]; opt.zero_grad(); F.cross_entropy(net(Xtr[idx].float()/255.),ytr[idx]).backward(); opt.step()

def weight_tensors(net):
    return [p for p in net.parameters() if p.dim()>=2]

def embed_model(net, msg, seed=0):
    """Embed message bits redundantly: global weight position g carries msg[g % len(msg)]."""
    g0=0
    with torch.no_grad():
        for p in weight_tensors(net):
            w=p.detach().cpu().numpy().astype(np.float32).ravel(); n=w.size
            bits=msg[(np.arange(g0,g0+n))%len(msg)].astype(np.uint8)
            w2,s,carrier=robust.qaware_embed_stealthy(w,bits,nbits=1,seed=seed+g0)
            p.copy_(torch.from_numpy(w2.reshape(p.shape)).to(p.device)); g0+=n

def extract_model(net, n_msg):
    """Majority-vote recover message of length n_msg from all weights."""
    votes=np.zeros((n_msg,2),dtype=np.int64); g0=0
    with torch.no_grad():
        for p in weight_tensors(net):
            w=p.detach().cpu().numpy().astype(np.float32).ravel(); n=w.size
            s=robust.quant_scale(w); bits=robust.qaware_extract(w,s,nbits=1)
            pos=(np.arange(g0,g0+n))%n_msg
            np.add.at(votes,(pos,bits),1); g0+=n
    return (votes[:,1]>votes[:,0]).astype(np.uint8)

def prune(net,frac):
    """Unstructured magnitude pruning: zero the `frac` smallest-|w| per tensor."""
    with torch.no_grad():
        for p in weight_tensors(net):
            w=p.abs(); thr=torch.quantile(w.flatten(),frac); p[w<thr]=0.0

def prune_structured(net,frac):
    """Structured pruning: zero the `frac` fraction of OUTPUT channels (dim 0) with
    smallest L2 norm per tensor -> removes contiguous weight blocks (harder case)."""
    with torch.no_grad():
        for p in weight_tensors(net):
            oc=p.shape[0]; k=int(round(frac*oc))
            if k<=0: continue
            norms=p.detach().reshape(oc,-1).pow(2).sum(1)
            kill=torch.topk(norms,k,largest=False).indices
            p[kill]=0.0

def main():
    np.random.seed(0); torch.manual_seed(0)
    dev="cuda" if torch.cuda.is_available() else "cpu"
    (Xtr,ytr),(Xte,yte)=cifar(dev)
    base=SmallCNN().to(dev); train(base,Xtr,ytr,dev,epochs=6,lr=1e-3)
    P=sum(p.numel() for p in weight_tensors(base)); a0=acc(base,Xte,yte,dev)
    print(f"baseline acc={a0:.3f}, weights={P:,}\n",flush=True)
    rng=np.random.default_rng(1)
    print(f"{'transform':22s} {'n_msg':>7} {'redund':>7} {'msg-recover':>12} {'acc':>6}",flush=True)
    for n_msg in [1024, 64]:
        msg=rng.integers(0,2,n_msg,dtype=np.uint8); red=P//n_msg
        # baseline (no transform): sanity
        net=copy.deepcopy(base); embed_model(net,msg)
        rec=extract_model(net,n_msg); base_sv=(rec==msg).mean()
        # + quantize
        netq=copy.deepcopy(net)
        with torch.no_grad():
            for p in weight_tensors(netq):
                w=p.detach().cpu().numpy().astype(np.float32); s=robust.quant_scale(w.ravel())
                p.copy_(torch.from_numpy((robust.quantize(w,s).astype(np.float32)*s)).to(p.device))
        rq=(extract_model(netq,n_msg)==msg).mean()
        # + fine-tune (light) then extract
        netf=copy.deepcopy(net); train(netf,Xtr,ytr,dev,epochs=2,lr=2e-4); af=acc(netf,Xte,yte,dev)
        rf=(extract_model(netf,n_msg)==msg).mean()
        print(f"{'(none)':22s} {n_msg:>7} {red:>7} {base_sv:>12.3f} {a0:>6.3f}",flush=True)
        print(f"{'quantize INT8':22s} {n_msg:>7} {red:>7} {rq:>12.3f} {'-':>6}",flush=True)
        print(f"{'fine-tune 2ep':22s} {n_msg:>7} {red:>7} {rf:>12.3f} {af:>6.3f}",flush=True)
        # + pruning sweep: unstructured {30,50,70,90}% then structured {30,50,70,90}%
        for frac in (0.30,0.50,0.70,0.90):
            netp=copy.deepcopy(net); prune(netp,frac); ap=acc(netp,Xte,yte,dev)
            rp=(extract_model(netp,n_msg)==msg).mean()
            print(f"{f'prune unstruct {int(frac*100)}%':22s} {n_msg:>7} {red:>7} {rp:>12.3f} {ap:>6.3f}",flush=True)
        for frac in (0.30,0.50,0.70,0.90):
            netp=copy.deepcopy(net); prune_structured(netp,frac); ap=acc(netp,Xte,yte,dev)
            rp=(extract_model(netp,n_msg)==msg).mean()
            print(f"{f'prune struct {int(frac*100)}%':22s} {n_msg:>7} {red:>7} {rp:>12.3f} {ap:>6.3f}",flush=True)
        print(flush=True)

if __name__=="__main__":
    main()
