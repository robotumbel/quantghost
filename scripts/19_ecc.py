"""B8: fountain ECC vs naive repetition for capacity-efficient robustness.

Channel = erasure: a fraction `e` of coded bits are destroyed (a stand-in for
weights wiped by pruning/sanitization, whose zeroed positions the recipient can
detect). We show the LT fountain code recovers K message bits byte-exact from
N = K*(1+overhead) coded bits, while repetition needs R copies (R*K bits) for the
same erasure rate -> fountain delivers far more payload per weight at equal robustness.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from wstego import ecc


def main():
    rng = np.random.default_rng(0)
    K = 4000                                      # message bits
    print(f"message K={K} bits\n", flush=True)
    print(f"{'erase rate':>10} {'fountain ovh':>13} {'fountain ok':>12} {'rep copies':>11} "
          f"{'rep rate vs fountain':>21}", flush=True)
    for e in (0.05, 0.10, 0.20, 0.30):
        msg = rng.integers(0, 2, K, dtype=np.uint8)
        # fountain: pick smallest overhead in a small grid that decodes
        decoded_ok = False
        for ovh in (0.10, 0.20, 0.35, 0.50, 0.75):
            coded, N = ecc.encode(msg, overhead=ovh, seed=123)
            mask = rng.random(N) >= e               # surviving coded bits
            rec, ok = ecc.decode(np.where(mask, coded, 0).astype(np.uint8), mask, K, seed=123)
            if ok and (rec == msg).all():
                decoded_ok = True; used_ovh = ovh; used_N = N; break
        R = ecc.repetition_copies_needed(e, K)
        # repetition coded bits = R*K; fountain coded bits = used_N
        ratio = (R * K) / used_N if decoded_ok else float('nan')
        print(f"{e:>10.2f} {used_ovh*100:>11.0f}% {str(decoded_ok):>12} {R:>11} "
              f"{ratio:>20.2f}x", flush=True)
    print("\nbaca: fountain pulih byte-exact dgn overhead kecil; repetition butuh R salinan "
          "(R*K bit). rasio = berapa kali lebih boros repetition pada robustness setara.", flush=True)


if __name__ == "__main__":
    main()
