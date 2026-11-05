"""Defeat NeuPerm (neuron-permutation defense, 2025).

NeuPerm scrambles weight ORDER (functionally-equivalent neuron permutation),
which breaks POSITION-based payloads. We (1) confirm the vulnerability, then (2)
test a PERMUTATION-INVARIANT scheme: embed along the canonical order obtained by
SORTING weights by value (a permutation never changes the value multiset, so the
sorted sequence is invariant). Redundancy + majority vote absorb the small
reorderings caused by the sub-quantum embedding itself.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from wstego import robust

def embed_positional(w, msg, seed=0):
    n=w.size; bits=msg[np.arange(n)%len(msg)].astype(np.uint8)
    w2,s,_=robust.qaware_embed_stealthy(w,bits,nbits=1,seed=seed); return w2

def extract_positional(w, n_msg):
    s=robust.quant_scale(w); bits=robust.qaware_extract(w,s,nbits=1)
    votes=np.zeros((n_msg,2),np.int64); np.add.at(votes,(np.arange(w.size)%n_msg,bits),1)
    return (votes[:,1]>votes[:,0]).astype(np.uint8)

def embed_sorted(w, msg, seed=0):
    order=np.argsort(w, kind="stable")          # canonical order by value
    bits_canon=msg[np.arange(w.size)%len(msg)].astype(np.uint8)
    bits=np.empty(w.size,np.uint8); bits[order]=bits_canon   # place bit at each weight
    w2,s,_=robust.qaware_embed_stealthy(w,bits,nbits=1,seed=seed); return w2

def extract_sorted(w, n_msg):
    s=robust.quant_scale(w); bits=robust.qaware_extract(w,s,nbits=1)
    order=np.argsort(w, kind="stable")          # re-derive canonical order
    canon_pos=np.empty(w.size,np.int64); canon_pos[order]=np.arange(w.size)
    votes=np.zeros((n_msg,2),np.int64); np.add.at(votes,(canon_pos%n_msg,bits),1)
    return (votes[:,1]>votes[:,0]).astype(np.uint8)

def main():
    from wstego import zoo
    w=zoo.load_model_weights('resnet18')[:1_000_000].astype(np.float32)
    rng=np.random.default_rng(0); n_msg=256; msg=rng.integers(0,2,n_msg,dtype=np.uint8)
    perm=rng.permutation(w.size)                 # simulate NeuPerm (weight reorder)
    print(f"weights={w.size:,}  msg={n_msg} bits  redundancy~{w.size//n_msg}\n",flush=True)
    print(f"{'scheme':14s} {'no-perm':>9} {'after-PERM':>11}",flush=True)
    # positional
    wp=embed_positional(w,msg)
    s0=(extract_positional(wp,n_msg)==msg).mean()
    s1=(extract_positional(wp[perm],n_msg)==msg).mean()
    print(f"{'positional':14s} {s0:>9.3f} {s1:>11.3f}",flush=True)
    # sorted (permutation-invariant)
    ws=embed_sorted(w,msg)
    t0=(extract_sorted(ws,n_msg)==msg).mean()
    t1=(extract_sorted(ws[perm],n_msg)==msg).mean()
    print(f"{'sorted-canon':14s} {t0:>9.3f} {t1:>11.3f}",flush=True)
    print("\nharapan: positional hancur after-PERM (~0.5); sorted-canon bertahan (~1.0)",flush=True)

if __name__=="__main__":
    main()
