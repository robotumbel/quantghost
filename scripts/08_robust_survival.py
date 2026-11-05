"""Phase 3, milestone 1: does a quantization-AWARE embedding survive FP32->INT8
quantization where naive X-LSB does not -- and at what accuracy cost?

Trains one SmallCNN on CIFAR-10, embeds a payload per-tensor with each scheme,
simulates the victim's per-tensor INT8 quantization, and reports payload survival
+ FP32/INT8 test accuracy + capacity.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torchvision as tv
from wstego import embed as emb, robust
torch.backends.cudnn.benchmark = True

class SmallCNN(nn.Module):
    def __init__(s, nc=10):
        super().__init__()
        def blk(i,o): return nn.Sequential(nn.Conv2d(i,o,3,padding=1),nn.BatchNorm2d(o),nn.ReLU(True))
        s.f=nn.Sequential(blk(3,32),blk(32,32),nn.MaxPool2d(2),blk(32,64),blk(64,64),nn.MaxPool2d(2),blk(64,128),nn.AdaptiveAvgPool2d(1))
        s.h=nn.Linear(128,10)
    def forward(s,x): return s.h(s.f(x).flatten(1))

def cifar(dev):
    tr=tv.datasets.CIFAR10("data/cifar",train=True,download=True)
    te=tv.datasets.CIFAR10("data/cifar",train=False,download=True)
    f=lambda d:(torch.from_numpy(d.data).to(dev).permute(0,3,1,2).contiguous(),torch.tensor(d.targets,device=dev))
    return f(tr),f(te)

def acc(net,Xte,yte,dev):
    net.eval(); c=0
    with torch.no_grad():
        for i in range(0,len(Xte),1000):
            xb=Xte[i:i+1000].float()/255.; c+=(net(xb).argmax(1)==yte[i:i+1000]).sum().item()
    return c/len(Xte)

def train(net,Xtr,ytr,dev,epochs=6,bs=512,sub=10000):
    sel=torch.randperm(len(Xtr),device=dev)[:sub]; Xtr,ytr=Xtr[sel],ytr[sel]
    opt=torch.optim.Adam(net.parameters(),1e-3,weight_decay=5e-4)
    for _ in range(epochs):
        p=torch.randperm(len(Xtr),device=dev)
        for i in range(0,len(Xtr),bs):
            idx=p[i:i+bs]; xb=Xtr[idx].float()/255.; opt.zero_grad(); F.cross_entropy(net(xb),ytr[idx]).backward(); opt.step()

def apply_scheme(net, scheme, payload, nbits=1):
    """Modify each weight tensor in-place; return total survival after re-quant."""
    surv_num=surv_den=0
    with torch.no_grad():
        for name,p in net.named_parameters():
            if p.dim()<2: continue
            w=p.detach().cpu().numpy().astype(np.float32).ravel()
            if scheme=="naive":
                w2=emb.embed_xlsb(w,payload,x=8,fill=True)
                carrier=None
            else:
                w2,scale,carrier=robust.qaware_embed(w,payload,nbits=nbits)
            p.copy_(torch.from_numpy(w2.reshape(p.shape)).to(p.device))
            # victim re-quantizes per-tensor (recomputes its OWN scale) & we recover
            s=robust.quant_scale(w2)
            if scheme=="naive":
                wq=robust.quantize(w2,s).astype(np.float32)*s
                rec=emb.extract_xlsb(wq,x=8)
                surv=robust.survival_rate(rec,payload)
                cnt=w.size
            else:
                rec=robust.qaware_extract(w2,s,nbits=nbits)
                surv=robust.survival_rate(rec,payload,nbits=nbits,carrier=carrier)
                cnt=int(carrier.sum())
            surv_num+=surv*cnt; surv_den+=cnt
    return surv_num/surv_den

def quantize_model(net):
    """Apply per-tensor INT8 quant-dequant to all weight tensors (simulate deploy)."""
    with torch.no_grad():
        for p in net.parameters():
            if p.dim()<2: continue
            w=p.detach().cpu().numpy().astype(np.float32)
            s=robust.quant_scale(w.ravel()); wq=robust.quantize(w,s).astype(np.float32)*s
            p.copy_(torch.from_numpy(wq).to(p.device))

def main():
    np.random.seed(0); torch.manual_seed(0)
    dev="cuda" if torch.cuda.is_available() else "cpu"
    (Xtr,ytr),(Xte,yte)=cifar(dev)
    base=SmallCNN().to(dev); train(base,Xtr,ytr,dev)
    a0=acc(base,Xte,yte,dev)
    P=sum(p.numel() for p in base.parameters() if p.dim()>=2)
    print(f"baseline FP32 acc={a0:.3f} | embeddable weight params={P:,} "
          f"(=> {P/8/1e3:.0f} KB @1bit/w, {P/4/1e3:.0f} KB @2bit/w)\n",flush=True)
    payload=emb.random_payload_bits(1<<16,0)
    import copy
    print(f"{'scheme':16s} {'survival':>9} {'FP32 acc':>9} {'INT8 acc':>9}",flush=True)
    # reference: clean model under quantization
    netq=copy.deepcopy(base); quantize_model(netq)
    print(f"{'(clean->quant)':16s} {'-':>9} {a0:>9.3f} {acc(netq,Xte,yte,dev):>9.3f}",flush=True)
    for scheme,nb in [("naive",8),("qaware-1bit",1),("qaware-2bit",2)]:
        net=copy.deepcopy(base)
        surv=apply_scheme(net,"naive" if scheme=="naive" else "qaware",payload,nbits=nb)
        fp=acc(net,Xte,yte,dev)
        quantize_model(net); iq=acc(net,Xte,yte,dev)
        print(f"{scheme:16s} {surv:>9.3f} {fp:>9.3f} {iq:>9.3f}",flush=True)
    print("\nbaca: survival~1.0 = payload selamat lewat kuantisasi; naive~0.5=hancur",flush=True)

if __name__=="__main__":
    main()
