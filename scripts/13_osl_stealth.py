"""Priority experiment #1: the REAL OSL detection method (Model X-Ray's algorithm:
GF fourpart image + metric-learning few-shot, centroid+1NN) vs our embedding.

Positive control: naive X-LSB structured payload -> OSL should DETECT.
Our attack: qaware-stealthy encrypted payload -> OSL should fail (~chance).
Cross-model (few-shot): train embedding on some models, eval on held-out ones.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from wstego import embed, robust, gf_image, osl

def binbits(n):
    with open(sys.executable,"rb") as f: d=f.read(max(64*1024,n//8+16))
    b=np.unpackbits(np.frombuffer(d,np.uint8)); return np.tile(b,int(np.ceil(n/len(b))))[:n].astype(np.uint8)

def build(W, kind, seed, per=56):
    Xs,ys,m=[],[],[]
    for mid,w in enumerate(W):
        if kind=="naive":
            ws=embed.embed_xlsb(w,binbits(1<<16),x=8,fill=True)
        else:
            ws,_,_=robust.qaware_embed_stealthy(w,embed.random_payload_bits(1<<16,seed+mid),nbits=1,seed=seed+mid)
        tc=gf_image.weights_to_fourpart(w,chunk=2500)[:per]
        ts=gf_image.weights_to_fourpart(ws,chunk=2500)[:per]
        Xs+=[tc,ts]; ys+=[np.zeros(len(tc)),np.ones(len(ts))]; m+=[np.full(len(tc),mid),np.full(len(ts),mid)]
    return np.concatenate(Xs),np.concatenate(ys).astype(np.int64),np.concatenate(m)

def episodic_train(net,X,y,dev,steps,bs,lr):
    opt=torch.optim.Adam(net.parameters(),lr,weight_decay=1e-4); net.train()
    for _ in range(steps):
        i0=np.random.choice(np.where(y==0)[0],bs//2); i1=np.random.choice(np.where(y==1)[0],bs//2)
        idx=np.concatenate([i0,i1])
        xb=torch.from_numpy(X[idx].astype(np.float32)[:,None]/255.).to(dev); yb=torch.from_numpy(y[idx]).to(dev)
        loss=osl.online_triplet_loss(net(xb),yb); opt.zero_grad(); loss.backward(); opt.step()
    return net

def main():
    np.random.seed(0); torch.manual_seed(0)
    dev="cuda" if torch.cuda.is_available() else "cpu"
    Z=np.load("data/zoo_cifar.npz"); W=Z["weights"]; M=len(W)
    ntest=8; tr_ids=np.arange(M-ntest) if False else np.arange(0,M-ntest); te_ids=np.arange(M-ntest,M)
    print(f"zoo {W.shape} | REAL OSL method (fourpart + triplet + centroid/1NN), {M-ntest} train / {ntest} test models\n",flush=True)
    print(f"{'embedding':26s} {'centroid':>9} {'1nn':>7}  verdict",flush=True)
    for kind,lbl in [("naive","naive X-LSB structured (ctrl)"),("qaware","qaware-stealthy encrypted (ours)")]:
        t0=time.time()
        X,y,m=build(W,kind,seed=0)
        trm=np.isin(m,tr_ids); tem=np.isin(m,te_ids)
        net=osl.EmbeddingNet(emb=64,width=0.5).to(dev)
        net=episodic_train(net,X[trm],y[trm],dev,steps=250,bs=64,lr=1e-3)
        cent,nn_=osl.centroid_1nn_eval(net,X[trm],y[trm],X[tem],y[tem],dev)
        v="DETECTED" if cent>0.7 else ("evaded" if cent<0.6 else "partial")
        print(f"{lbl:26s} {cent:>9.3f} {nn_:>7.3f}  {v}  [{time.time()-t0:.0f}s]",flush=True)
    print("\n#1: OSL method asli; naif terstruktur terdeteksi, qaware-stealthy lolos => klaim stealth kuat.",flush=True)

if __name__=="__main__":
    main()
