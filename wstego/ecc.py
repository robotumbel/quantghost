"""B8: capacity-efficient redundancy via an LT fountain code (pure numpy, GF(2)).

Naive repetition wastes capacity: to survive a fraction e of erased/destroyed
weights it needs R copies per bit (R chosen so e**R is negligible). A rateless
fountain (LT) code instead spreads K message bits over N = K*(1+overhead) coded
bits; the decoder peels the message back from ANY sufficiently large surviving
subset, so a small overhead tolerates the same erasure rate at far higher rate.

Bit-level GF(2): each coded bit is the XOR of a random subset of message bits, with
subset structure derived deterministically from `seed` so the decoder reconstructs
the graph without side information. Decoding is the standard peeling algorithm.
"""
from __future__ import annotations
import numpy as np


def _robust_soliton_degrees(K, N, seed, c=0.03, delta=0.5):
    """Sample N output-symbol degrees from the robust soliton distribution."""
    rng = np.random.default_rng(seed)
    rho = np.zeros(K + 1)
    rho[1] = 1.0 / K
    for d in range(2, K + 1):
        rho[d] = 1.0 / (d * (d - 1))
    R = c * np.log(K / delta) * np.sqrt(K)
    tau = np.zeros(K + 1)
    kr = max(int(round(K / R)), 1)
    for d in range(1, kr):
        tau[d] = R / (d * K)
    if kr <= K:
        tau[kr] = R * np.log(R / delta) / K
    mu = rho + tau
    mu /= mu.sum()
    degrees = rng.choice(np.arange(K + 1), size=N, p=mu)
    degrees[degrees < 1] = 1
    return degrees


def _graph(K, N, seed):
    """Deterministic bipartite graph: for each of N coded bits, the message indices
    it XORs. Same seed -> same graph at encoder and decoder."""
    degrees = _robust_soliton_degrees(K, N, seed)
    rng = np.random.default_rng(seed + 1)
    return [rng.choice(K, size=min(int(d), K), replace=False) for d in degrees]


def encode(msg_bits: np.ndarray, overhead: float = 0.25, seed: int = 0):
    """Encode K message bits into N = ceil(K*(1+overhead)) coded bits."""
    msg_bits = msg_bits.astype(np.uint8)
    K = msg_bits.size
    N = int(np.ceil(K * (1 + overhead)))
    nbrs = _graph(K, N, seed)
    coded = np.array([int(np.bitwise_xor.reduce(msg_bits[ix])) for ix in nbrs], dtype=np.uint8)
    return coded, N


def decode(coded_bits: np.ndarray, received_mask: np.ndarray, K: int, seed: int):
    """Peeling decode. coded_bits length N (erased entries ignored via mask).
    Returns (recovered_bits[K], ok: bool)."""
    N = coded_bits.size
    nbrs = _graph(K, N, seed)
    # active coded symbols = received ones; track unresolved neighbors per symbol
    val = coded_bits.copy().astype(np.uint8)
    rem = [set(int(j) for j in nbrs[i]) if received_mask[i] else None for i in range(N)]
    sol = np.full(K, 255, dtype=np.uint8)          # 255 = unknown
    # index: message bit -> coded symbols that touch it
    touch = [[] for _ in range(K)]
    for i in range(N):
        if rem[i] is not None:
            for j in rem[i]:
                touch[j].append(i)
    # ripple: queue of degree-1 symbols
    queue = [i for i in range(N) if rem[i] is not None and len(rem[i]) == 1]
    while queue:
        i = queue.pop()
        if rem[i] is None or len(rem[i]) != 1:
            continue
        j = next(iter(rem[i]))
        sol[j] = val[i]
        for k in touch[j]:                          # remove solved bit from neighbors
            if rem[k] is not None and j in rem[k]:
                rem[k].discard(j)
                val[k] ^= sol[j]
                if len(rem[k]) == 1:
                    queue.append(k)
    ok = bool((sol != 255).all())
    sol[sol == 255] = 0
    return sol, ok


def repetition_copies_needed(erase_rate: float, n_bits: int, target_fail: float = 1.0):
    """Min R so expected repetition failures over n_bits < target_fail."""
    R = 1
    while n_bits * (erase_rate ** R) >= target_fail and R < 64:
        R += 1
    return R
