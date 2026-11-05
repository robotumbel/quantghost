"""Channel-structured NeuPerm test (faithful to real neuron permutation).

Real NeuPerm permutes a conv layer's output channels (pi_out) and the matching
input channels (pi_in), but NEVER the spatial (kh,kw) positions. So a payload
indexed by FLAT position breaks, but one indexed by the permutation-INVARIANT
spatial position survives -- at the cost of capacity (only kh*kw distinguishable
slots per conv tensor, x redundancy across channels). We measure both on a real
conv weight tensor.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import torchvision.models as M
from wstego import robust

def chan_perm(W, rng):
    """Apply realistic neuron permutation: permute out_ch (dim0) and in_ch (dim1)."""
    o,i = W.shape[0], W.shape[1]
    po, pi = rng.permutation(o), rng.permutation(i)
    return W[po][:, pi]

def embed_tensor(W, msg, index_fn, seed=0):
    """index_fn(shape)->flat int array assigning each weight a message position."""
    flatW = W.reshape(-1).astype(np.float32)
    idx = index_fn(W.shape)
    bits = msg[idx % len(msg)].astype(np.uint8)
    w2,s,_ = robust.qaware_embed_stealthy(flatW, bits, nbits=1, seed=seed)
    return w2.reshape(W.shape)

def extract_tensor(W, n_msg, index_fn):
    flatW = W.reshape(-1).astype(np.float32)
    s = robust.quant_scale(flatW); bits = robust.qaware_extract(flatW, s, nbits=1)
    idx = index_fn(W.shape)
    votes = np.zeros((n_msg,2), np.int64); np.add.at(votes,(idx % n_msg, bits),1)
    return (votes[:,1]>votes[:,0]).astype(np.uint8)

def idx_flat(shape):
    return np.arange(int(np.prod(shape)))

def idx_spatial(shape):
    # invariant index = position within (kh,kw); same for every (out,in)
    o,i,kh,kw = shape
    sp = np.arange(kh*kw)
    return np.tile(sp, o*i)

def main():
    rng = np.random.default_rng(0)
    model = M.resnet18(weights="DEFAULT")
    # pick a large 3x3 conv weight tensor
    W = None
    for n,p in model.named_parameters():
        if p.dim()==4 and p.shape[2]==3 and p.shape[0]>=256:
            W = p.detach().numpy().astype(np.float32); name=n; break
    print(f"conv tensor {name} shape={W.shape} ({W.size:,} weights)\n",flush=True)
    perm = lambda X: chan_perm(X, np.random.default_rng(42))

    print(f"{'scheme':16s} {'n_msg':>6} {'redund':>8} {'no-perm':>9} {'after-CHANPERM':>15}",flush=True)
    # A: flat positional
    msgA = rng.integers(0,2,256,dtype=np.uint8)
    Wa = embed_tensor(W, msgA, idx_flat)
    a0 = (extract_tensor(Wa,256,idx_flat)==msgA).mean()
    a1 = (extract_tensor(perm(Wa),256,idx_flat)==msgA).mean()
    print(f"{'flat-positional':16s} {256:>6} {W.size//256:>8} {a0:>9.3f} {a1:>15.3f}",flush=True)
    # B: spatial-invariant (n_msg = kh*kw = 9)
    nB = W.shape[2]*W.shape[3]
    msgB = rng.integers(0,2,nB,dtype=np.uint8)
    Wb = embed_tensor(W, msgB, idx_spatial)
    b0 = (extract_tensor(Wb,nB,idx_spatial)==msgB).mean()
    b1 = (extract_tensor(perm(Wb),nB,idx_spatial)==msgB).mean()
    print(f"{'spatial-invar':16s} {nB:>6} {W.size//nB:>8} {b0:>9.3f} {b1:>15.3f}",flush=True)
    print(f"\nspatial-invar kapasitas/tensor = {nB} bit (x banyak tensor). "
          f"flat = {W.size:,} bit tapi hancur permutasi.",flush=True)

if __name__=="__main__":
    main()
