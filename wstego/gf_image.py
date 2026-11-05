"""GF (grayscale-float) byte-image representation of model weights.

Model X-Ray converts weights to an image the steganalysis CNN can read:
each float32 weight -> its 4 IEEE-754 bytes -> 4 uint8 pixel values, laid out
into a square grayscale image. The low-order mantissa bytes (where the payload
lives) carry the steganographic signal; the CNN learns the spatial structure
that LSB-fill destroys.

We tile a model's full byte stream into many fixed-size HxH images so even a
handful of models yields a large, cross-model dataset.
"""
from __future__ import annotations
import numpy as np


def weights_to_bytes(w: np.ndarray) -> np.ndarray:
    """Flatten float32 weights to a little-endian uint8 byte stream.

    Little-endian => byte 0 is the LOW mantissa byte (the payload-bearing one),
    which we keep adjacent so the CNN sees the stego plane coherently.
    """
    w = np.ascontiguousarray(w, dtype=np.float32)
    return w.view(np.uint8)  # length 4 * len(w)


def byte_stream_to_tiles(b: np.ndarray, tile: int = 256,
                         max_tiles: int | None = None,
                         shuffle_seed: int | None = None) -> np.ndarray:
    """Cut a byte stream into HxH uint8 tiles. Returns (N, tile, tile) uint8."""
    per = tile * tile
    n = b.shape[0] // per
    if n == 0:
        raise ValueError(f"byte stream too short for one {tile}x{tile} tile")
    tiles = b[: n * per].reshape(n, tile, tile)
    if shuffle_seed is not None:
        rng = np.random.default_rng(shuffle_seed)
        idx = rng.permutation(n)
        tiles = tiles[idx]
    if max_tiles is not None:
        tiles = tiles[:max_tiles]
    return np.ascontiguousarray(tiles)


def weights_to_tiles(w: np.ndarray, tile: int = 256,
                     max_tiles: int | None = None,
                     shuffle_seed: int | None = None) -> np.ndarray:
    """Convenience: weights -> byte stream -> HxH tiles."""
    return byte_stream_to_tiles(weights_to_bytes(w), tile=tile,
                                max_tiles=max_tiles, shuffle_seed=shuffle_seed)


def weights_to_fourpart(w: np.ndarray, chunk: int = 2500) -> np.ndarray:
    """Grayscale-Fourpart representation (Model X-Ray's GF), one image per chunk.

    Each weight contributes 4 bytes (b0..b3, little-endian). Instead of
    interleaving, each BYTE POSITION becomes its own coherent d x d plane, and
    the 4 planes are tiled into 2x2 quadrants -> a (2d x 2d) image. Spatial
    adjacency = weight adjacency, so the low-mantissa-byte plane (where the
    payload lives) is spatially coherent and the CNN can read its structure and
    its correlation with the magnitude planes.

    chunk=2500 -> d=50 -> a 100x100 image directly (no resize), matching their
    imsize=100. Returns (N, 2d, 2d) uint8 over contiguous weight chunks.
    """
    b = np.ascontiguousarray(w, dtype=np.float32).view(np.uint8).reshape(-1, 4)
    P = b.shape[0]
    n = P // chunk
    if n == 0:
        raise ValueError("weights too few for one chunk")
    d = int(np.ceil(np.sqrt(chunk)))
    imgs = np.empty((n, 2 * d, 2 * d), dtype=np.uint8)
    for i in range(n):
        cb = b[i * chunk:(i + 1) * chunk]
        if cb.shape[0] < d * d:
            cb = np.pad(cb, ((0, d * d - cb.shape[0]), (0, 0)))
        planes = cb.T.reshape(4, d, d)                       # b0,b1,b2,b3 planes
        top = np.concatenate([planes[0], planes[1]], axis=1)
        bot = np.concatenate([planes[2], planes[3]], axis=1)
        imgs[i] = np.concatenate([top, bot], axis=0)
    return imgs


def low_byte_plane(w: np.ndarray, tile: int = 256,
                   max_tiles: int | None = None) -> np.ndarray:
    """Extract ONLY the lowest mantissa byte of each weight, tiled.

    Useful as a diagnostic: this plane is where X-LSB (X<=8) writes. A marginal
    histogram of it looks ~uniform for both benign and stego (why naive
    statistics fail); a CNN over its spatial layout is what separates them.
    """
    b = weights_to_bytes(w)[0::4]  # little-endian low byte
    return byte_stream_to_tiles(b, tile=tile, max_tiles=max_tiles)
